// Matrix products for the GPU lane of grouped expert prefill: one expert's matrix times a group of activation columns,
// for the Kimi VRAM tier's batched engine, which loads this library at run time. A product takes ggml's kernels for the
// same weight type (ggml_cuda_mul_mat_select): MMQ at every column count where ggml_cuda_should_use_mmq says so, so a
// type's chain runs one kernel family and its time grows with its columns (ggml's MUL_MAT takes MMVQ up to 8 columns;
// Muse's chain on an RTX 5060 Ti was faster with it only at 1-2 routes, and 1.85x slower at 8); otherwise the vector
// kernels (MMVF, MMVQ) up to 8 columns and a cuBLAS GEMM on half-precision copies above (the weights dequantized into a
// scratch of one matrix, as ggml_cuda_op_mul_mat_cublas does, with FP32 accumulation). So every type
// ggml_cuda_moe_stream_mmvq_supports serves in decode has a grouped kernel. MMQ loads a row's blocks a whole iteration
// (MMQ_ITER_K values) at a time, so it reads past the row's end when the row is not a whole number of iterations. ggml
// pads every quantized tensor for that (ggml_backend_cuda_buffer_type_get_alloc_size); an expert's roles sit in tier
// slots, arenas and ring buffers without that padding, so such rows do not take MMQ.
//
// The activation operand is made apart from the product, so that up and gate share one quantization: quantize gathers
// the columns' rows (by device row ids, or in order) into the representation the product of that column count reads.
// Its Q8_1 block sums are those of the quantized values (quantize_*_quantized_sums_cuda): a weight type's minimum or
// offset then multiplies the same activations as its scale, and the product is as close to the exact one as the CPU's.
// A context owns one ggml_backend_cuda_context: MMQ's stream-k fixup comes from its pool and the GEMM uses its cuBLAS
// handle, as in ggml's own MUL_MAT. reserve sizes the pool, the dequantize scratch and the cuBLAS workspace up front,
// so a product made after it allocates no device memory; release gives all of it back. reserve also loads every kernel
// the products can take: CUDA loads a kernel, and takes device memory for its module, on the kernel's first launch
// (lazy loading) and keeps it for the process, so that happens in reserve and not in the middle of a pass.
#include "../common.cuh"
#include "../convert.cuh"
#include "../mmq.cuh"
#include "../mmvf.cuh"
#include "../mmvq.cuh"
#include "../quantize.cuh"
#include "backend-functions.h"

#include <algorithm>
#include <memory>
#include <new>
#include <vector>

extern "C" bool ggml_cuda_moe_stream_mmvf_dev(
    int, const void *, int64_t, int64_t, size_t, const float *, float *, int64_t, cudaStream_t);
extern "C" bool ggml_cuda_moe_stream_mmvq_supports(int);
extern "C" size_t ggml_cuda_moe_stream_mm_operand_bytes(void *, int, int64_t, int64_t);
extern "C" bool ggml_cuda_moe_stream_mm_quantize(void *, int, const float *, const int32_t *, int64_t, int64_t, int64_t, void *,
    cudaStream_t);
extern "C" bool ggml_cuda_moe_stream_mm(void *, int, const void *, int64_t, int64_t, const void *, int64_t, float *, cudaStream_t);

namespace {
// The kernel a product runs (ggml_cuda_moe_stream_mm_kernel).
enum moe_stream_mm_kernel {
    MOE_STREAM_MM_MMVF     = 0,
    MOE_STREAM_MM_MMVQ     = 1,
    MOE_STREAM_MM_MMQ      = 2,
    MOE_STREAM_MM_MMQ_FP4  = 3, // MMQ with FP4 activations (MXFP4 on Blackwell)
    MOE_STREAM_MM_GEMM     = 4,
};

constexpr size_t moe_stream_mm_workspace_bytes = 4u << 20; // cuBLAS's recommended workspace below Hopper

struct moe_stream_mm {
    int device;
    std::unique_ptr<ggml_backend_cuda_context> ctx; // its pool holds the fixup, its cuBLAS handle serves the GEMM
    int cc;
    void * dequant = nullptr;  // one matrix as half (bf16 weights need none), for types without a quantized kernel
    size_t dequant_bytes = 0;
    void * workspace = nullptr; // the cuBLAS workspace, set after every cublasSetStream
    size_t fixup_bytes = 0;     // MMQ's stream-k fixup, reserved in the pool
    // The types and shapes whose kernels are loaded, up to `columns` columns. A loaded kernel stays loaded, so this
    // outlives release.
    struct warm_shape { ggml_type type; int64_t nrows, ncols, columns; };
    std::vector<warm_shape> warm;

    explicit moe_stream_mm(int device_) :
        device(device_), ctx(new ggml_backend_cuda_context(device_)), cc(ggml_cuda_info().devices[device_].cc) {}
    ~moe_stream_mm() {
        ggml_cuda_set_device(device);
        if (dequant)   { (void) cudaFree(dequant); }
        if (workspace) { (void) cudaFree(workspace); }
    }

    void release() {
        ggml_cuda_set_device(device);
        if (dequant)   { (void) cudaFree(dequant);   dequant = nullptr; }
        if (workspace) { (void) cudaFree(workspace); workspace = nullptr; }
        dequant_bytes = fixup_bytes = 0;
        ctx.reset(new ggml_backend_cuda_context(device)); // a new context holds no pool and no handle yet
    }
};

bool is_float(ggml_type type) {
    return type == GGML_TYPE_F16 || type == GGML_TYPE_BF16;
}

// The types the decode engine serves: each has a vector kernel and a GEMM conversion, and the ones MMQ takes, MMQ.
bool supported(int type) {
    return ggml_cuda_moe_stream_mmvq_supports(type);
}

// MMQ's loads stay inside a row of ncols values: the row is a whole number of its iterations (get_iter_k in mmq.cuh).
bool mmq_rows_whole(const moe_stream_mm & mm, ggml_type type, int64_t ncols) {
    const int64_t iter_k = blackwell_mma_available(mm.cc) && type == GGML_TYPE_MXFP4 ? MMQ_ITER_K_MXFP4_FP4 : MMQ_ITER_K;
    return ncols % iter_k == 0;
}

moe_stream_mm_kernel kernel_for(const moe_stream_mm & mm, ggml_type type, int64_t ncols, int64_t n) {
    if (!is_float(type) && mmq_rows_whole(mm, type, ncols) && ggml_cuda_should_use_mmq(type, mm.cc, n, /*n_experts=*/0)) {
        return blackwell_mma_available(mm.cc) && type == GGML_TYPE_MXFP4 ? MOE_STREAM_MM_MMQ_FP4 : MOE_STREAM_MM_MMQ;
    }
    if (n <= (is_float(type) ? MMVF_MAX_BATCH_SIZE : MMVQ_MAX_BATCH_SIZE)) {
        return is_float(type) ? MOE_STREAM_MM_MMVF : MOE_STREAM_MM_MMVQ;
    }
    return MOE_STREAM_MM_GEMM;
}

size_t aligned_bytes(size_t bytes) {
    return (bytes + 255) & ~size_t(255);
}

// The operand of n columns for a kernel: what quantize writes and the product reads.
size_t operand_bytes_for(const moe_stream_mm & mm, ggml_type type, moe_stream_mm_kernel kernel, int64_t ncols, int64_t n) {
    const int64_t padded = GGML_PAD(ncols, MATRIX_ROW_PADDING);
    switch (kernel) {
        case MOE_STREAM_MM_MMVF:
            return aligned_bytes(size_t(n) * ncols * sizeof(float));
        case MOE_STREAM_MM_MMVQ: // the Q8_1 rows, then the gathered f32 rows they are quantized from
            return aligned_bytes(size_t(n) * padded / QK8_1 * sizeof(block_q8_1)) + aligned_bytes(size_t(n) * ncols * sizeof(float));
        case MOE_STREAM_MM_MMQ:
        case MOE_STREAM_MM_MMQ_FP4: // as ggml_cuda_mul_mat_q sizes src1_q8_1 (an upper bound for the FP4 layout)
            return aligned_bytes(size_t(n) * padded * sizeof(block_q8_1) / QK8_1 + get_mmq_x_max_host(mm.cc) * sizeof(block_q8_1_mmq));
        case MOE_STREAM_MM_GEMM:
            return aligned_bytes(size_t(n) * ncols * (type == GGML_TYPE_BF16 ? sizeof(nv_bfloat16) : sizeof(half)));
    }
    return 0;
}

template <typename T>
__global__ void moe_stream_mm_gather(const float * __restrict__ x, const int32_t * __restrict__ ids, const int64_t stride,
        const int64_t ncols, T * __restrict__ dst) {
    const int64_t column = blockIdx.y;
    const float * src = x + (ids ? int64_t(ids[column]) : column) * stride;
    T * out = dst + column * ncols;
    for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < ncols; i += int64_t(gridDim.x) * blockDim.x) {
        out[i] = ggml_cuda_cast<T>(src[i]);
    }
}

template <typename T>
void gather(const float * x, const int32_t * ids, int64_t stride, int64_t ncols, int64_t n, T * dst, cudaStream_t stream) {
    const int64_t blocks = std::min<int64_t>((ncols + 255) / 256, 32);
    moe_stream_mm_gather<T><<<dim3(blocks, n, 1), 256, 0, stream>>>(x, ids, stride, ncols, dst);
}

// Loads every kernel a product of this type and shape can take up to max_columns columns (the gather, the quantizers,
// the product's own kernel, the stream-k fixup, cuBLAS's GEMM), by a product of every column count on a zeroed matrix in
// a scratch freed afterwards. Once per type and shape: a kernel stays loaded for the process.
bool warm_kernels(moe_stream_mm & mm, ggml_type type, int64_t nrows, int64_t ncols, int64_t max_columns) {
    for (const auto & w : mm.warm) {
        if (w.type == type && w.nrows == nrows && w.ncols == ncols && w.columns >= max_columns) {
            return true;
        }
    }
    const size_t weight_bytes  = aligned_bytes(ggml_row_size(type, ncols) * nrows);
    const size_t input_bytes   = aligned_bytes(size_t(ncols) * sizeof(float));
    const size_t ids_bytes     = aligned_bytes(size_t(max_columns) * sizeof(int32_t));
    const size_t operand_bytes = aligned_bytes(ggml_cuda_moe_stream_mm_operand_bytes(&mm, type, ncols, max_columns));
    const size_t dst_bytes     = aligned_bytes(size_t(max_columns) * nrows * sizeof(float));
    char * scratch = nullptr;
    if (ggml_cuda_malloc_reclaiming(reinterpret_cast<void **>(&scratch),
            weight_bytes + input_bytes + ids_bytes + operand_bytes + dst_bytes, mm.device) != cudaSuccess) {
        (void) cudaGetLastError();
        return false;
    }
    const float * input = reinterpret_cast<const float *>(scratch + weight_bytes);
    const int32_t * ids = reinterpret_cast<const int32_t *>(scratch + weight_bytes + input_bytes);
    void * operand = scratch + weight_bytes + input_bytes + ids_bytes;
    float * dst = reinterpret_cast<float *>(scratch + weight_bytes + input_bytes + ids_bytes + operand_bytes);
    cudaStream_t stream = nullptr;
    bool ok = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess &&
        cudaMemsetAsync(scratch, 0, weight_bytes + input_bytes + ids_bytes, stream) == cudaSuccess;
    // every column gathers the one zeroed input row, so the gathering kernels load too
    for (int64_t n = 1; ok && n <= max_columns; ++n) {
        ok = ggml_cuda_moe_stream_mm_quantize(&mm, type, input, ids, ncols, ncols, n, operand, stream) &&
            ggml_cuda_moe_stream_mm(&mm, type, scratch, nrows, ncols, operand, n, dst, stream);
    }
    ok = stream && cudaStreamSynchronize(stream) == cudaSuccess && ok;
    if (stream) {
        (void) cudaStreamDestroy(stream);
    }
    (void) cudaFree(scratch);
    if (!ok) {
        (void) cudaGetLastError();
        return false;
    }
    mm.warm.push_back({ type, nrows, ncols, max_columns });
    return true;
}
} // namespace

extern "C" bool ggml_cuda_moe_stream_mm_supports(int type) {
    return supported(type);
}

extern "C" void * ggml_cuda_moe_stream_mm_create(int device) {
    if (device < 0 || device >= ggml_cuda_info().device_count) {
        return nullptr;
    }
    ggml_cuda_set_device(device);
    return new (std::nothrow) moe_stream_mm(device);
}

extern "C" void ggml_cuda_moe_stream_mm_destroy(void * context) {
    delete static_cast<moe_stream_mm *>(context);
}

extern "C" int ggml_cuda_moe_stream_mm_kernel(void * context, int type, int64_t ncols, int64_t n) {
    if (!context || !supported(type) || ncols < 1 || n < 1) {
        return -1;
    }
    return kernel_for(*static_cast<moe_stream_mm *>(context), (ggml_type) type, ncols, n);
}

// Readies the context for products of this weight type and matrix shape with up to max_columns columns: the stream-k
// fixup in the pool, for the GEMM a dequantize scratch of one matrix and the cuBLAS handle and workspace, and the
// kernels loaded (warm_kernels; the first time for a type and shape, a scratch of one matrix and its operands is taken
// and given back). Calls add up (a layer's up/gate and down types and shapes each call once).
extern "C" bool ggml_cuda_moe_stream_mm_reserve(void * context, int type, int64_t nrows, int64_t ncols, int64_t max_columns) {
    auto * mm = static_cast<moe_stream_mm *>(context);
    if (!mm || !supported(type) || nrows < 1 || ncols < 1 || max_columns < 1 || ncols % ggml_blck_size((ggml_type) type)) {
        return false;
    }
    ggml_cuda_set_device(mm->device);
    bool mmq = false, gemm = false;
    for (int64_t n = 1; n <= max_columns; ++n) {
        const auto kernel = kernel_for(*mm, (ggml_type) type, ncols, n);
        mmq  = mmq  || kernel == MOE_STREAM_MM_MMQ || kernel == MOE_STREAM_MM_MMQ_FP4;
        gemm = gemm || kernel == MOE_STREAM_MM_GEMM;
    }
    if (mmq) {
        mm->fixup_bytes = std::max(mm->fixup_bytes, ggml_cuda_mul_mat_q_fixup_pool_size());
        ggml_cuda_pool & pool = mm->ctx->pool();
        if (!pool.reserve(mm->fixup_bytes)) {
            return false;
        }
        // a pool that allocates on demand keeps the largest fixup from here on
        ggml_cuda_pool_alloc<char> warm(pool, mm->fixup_bytes);
    }
    if (gemm) {
        const size_t bytes = ggml_is_quantized((ggml_type) type) ? size_t(nrows) * ncols * sizeof(half) : 0;
        if (bytes > mm->dequant_bytes) {
            if (mm->dequant) {
                CUDA_CHECK(cudaFree(mm->dequant));
                mm->dequant = nullptr;
                mm->dequant_bytes = 0;
            }
            if (ggml_cuda_malloc_reclaiming(&mm->dequant, bytes, mm->device) != cudaSuccess) {
                (void) cudaGetLastError();
                mm->dequant = nullptr;
                return false;
            }
            mm->dequant_bytes = bytes;
        }
        if (!mm->workspace) {
            if (ggml_cuda_malloc_reclaiming(&mm->workspace, moe_stream_mm_workspace_bytes, mm->device) != cudaSuccess) {
                (void) cudaGetLastError();
                mm->workspace = nullptr;
                return false;
            }
        }
        (void) mm->ctx->cublas_handle();
    }
    return warm_kernels(*mm, (ggml_type) type, nrows, ncols, max_columns);
}

// Gives back the device memory the reservations hold (the pool, the cuBLAS handle and workspace, the dequantize
// scratch). Call it with no product in flight; reserve again before the next.
extern "C" void ggml_cuda_moe_stream_mm_release(void * context) {
    if (context) {
        static_cast<moe_stream_mm *>(context)->release();
    }
}

// The bytes of an operand that holds up to max_columns columns of ncols values for this weight type (enough for any
// column count up to it, whichever kernel that count takes).
extern "C" size_t ggml_cuda_moe_stream_mm_operand_bytes(void * context, int type, int64_t ncols, int64_t max_columns) {
    const auto * mm = static_cast<const moe_stream_mm *>(context);
    if (!mm || !supported(type) || ncols < 1 || max_columns < 1) {
        return 0;
    }
    size_t bytes = 0;
    for (int64_t n = 1; n <= max_columns; ++n) {
        bytes = std::max(bytes, operand_bytes_for(*mm, (ggml_type) type, kernel_for(*mm, (ggml_type) type, ncols, n), ncols, n));
    }
    return bytes;
}

// The operand of n columns for products with weights of `type`: column i is row ids[i] of x (ids in device memory), or
// row i without ids; rows are x_stride floats apart and hold ncols values.
extern "C" bool ggml_cuda_moe_stream_mm_quantize(void * context, int type, const float * x, const int32_t * ids,
        int64_t x_stride, int64_t ncols, int64_t n, void * operand, cudaStream_t stream) {
    auto * mm = static_cast<moe_stream_mm *>(context);
    if (!mm || !supported(type) || !x || !operand || ncols < 1 || x_stride < ncols || n < 0) {
        return false;
    }
    if (n == 0) {
        return true;
    }
    ggml_cuda_set_device(mm->device);
    const ggml_type t = (ggml_type) type;
    const int64_t padded = GGML_PAD(ncols, MATRIX_ROW_PADDING);
    switch (kernel_for(*mm, t, ncols, n)) {
        case MOE_STREAM_MM_MMVF:
            gather(x, ids, x_stride, ncols, n, static_cast<float *>(operand), stream);
            break;
        case MOE_STREAM_MM_MMVQ: {
            const float * rows = x;
            int64_t stride = x_stride;
            if (ids) {
                float * staging = reinterpret_cast<float *>(static_cast<char *>(operand) +
                    aligned_bytes(size_t(n) * padded / QK8_1 * sizeof(block_q8_1)));
                gather(x, ids, x_stride, ncols, n, staging, stream);
                rows = staging;
                stride = ncols;
            }
            quantize_row_q8_1_quantized_sums_cuda(rows, nullptr, operand, t, ncols, stride, n * stride, n * stride, padded, n,
                1, 1, stream);
            break;
        }
        case MOE_STREAM_MM_MMQ:
            quantize_mmq_q8_1_quantized_sums_cuda(x, ids, operand, t, ncols, x_stride, 0, 0, padded, n, 1, 1, stream);
            break;
        case MOE_STREAM_MM_MMQ_FP4:
            quantize_mmq_mxfp4_cuda(x, ids, operand, t, ncols, x_stride, 0, 0, padded, n, 1, 1, stream);
            break;
        case MOE_STREAM_MM_GEMM:
            if (t == GGML_TYPE_BF16) {
                gather(x, ids, x_stride, ncols, n, static_cast<nv_bfloat16 *>(operand), stream);
            } else {
                gather(x, ids, x_stride, ncols, n, static_cast<half *>(operand), stream);
            }
            break;
    }
    return cudaGetLastError() == cudaSuccess;
}

// dst (n columns of nrows floats) = w (nrows contiguous rows of ncols values of `type`) times the operand of n columns
// that quantize made for this type.
extern "C" bool ggml_cuda_moe_stream_mm(void * context, int type, const void * w, int64_t nrows, int64_t ncols,
        const void * operand, int64_t n, float * dst, cudaStream_t stream) {
    auto * mm = static_cast<moe_stream_mm *>(context);
    const ggml_type t = (ggml_type) type;
    if (!mm || !supported(type) || !w || !operand || !dst || nrows < 1 || ncols < 1 || n < 0 || ncols % ggml_blck_size(t)) {
        return false;
    }
    if (n == 0) {
        return true;
    }
    ggml_cuda_set_device(mm->device);
    const size_t row_bytes = ggml_row_size(t, ncols);
    switch (kernel_for(*mm, t, ncols, n)) {
        case MOE_STREAM_MM_MMVF:
            return ggml_cuda_moe_stream_mmvf_dev(type, w, nrows, ncols, row_bytes, static_cast<const float *>(operand), dst, n, stream);
        case MOE_STREAM_MM_MMVQ:
            ggml_cuda_moe_stream_mmvq(t, w, ncols, nrows, ncols / ggml_blck_size(t), operand, n, dst, stream);
            break;
        case MOE_STREAM_MM_MMQ:
        case MOE_STREAM_MM_MMQ_FP4:
            if (!mm->fixup_bytes) {
                return false; // not reserved
            }
            ggml_cuda_moe_stream_mmq(*mm->ctx, t, static_cast<const char *>(w), ncols, nrows, ncols / ggml_blck_size(t),
                static_cast<const int *>(operand), n, dst, stream);
            break;
        case MOE_STREAM_MM_GEMM: {
            if (!mm->workspace || (ggml_is_quantized(t) && mm->dequant_bytes < size_t(nrows) * ncols * sizeof(half))) {
                return false; // not reserved for this shape
            }
            const void * a = w;
            cudaDataType_t ab_type = t == GGML_TYPE_BF16 ? CUDA_R_16BF : CUDA_R_16F;
            if (ggml_is_quantized(t)) {
                const to_fp16_cuda_t to_fp16 = ggml_get_to_fp16_cuda(t);
                if (!to_fp16) {
                    return false;
                }
                to_fp16(w, static_cast<half *>(mm->dequant), nrows * ncols, stream);
                a = mm->dequant;
            }
            cublasHandle_t handle = mm->ctx->cublas_handle();
            const float alpha = 1.0f, beta = 0.0f;
            if (cublasSetStream(handle, stream) != CUBLAS_STATUS_SUCCESS ||
                    cublasSetWorkspace(handle, mm->workspace, moe_stream_mm_workspace_bytes) != CUBLAS_STATUS_SUCCESS ||
                    cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, int(nrows), int(n), int(ncols),
                        &alpha, a, ab_type, int(ncols), operand, ab_type, int(ncols),
                        &beta, dst, CUDA_R_32F, int(nrows), CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP) != CUBLAS_STATUS_SUCCESS) {
                return false;
            }
            break;
        }
    }
    return cudaGetLastError() == cudaSuccess;
}

// The device memory the context holds: the pool (the fixup), the dequantize scratch and the cuBLAS workspace.
extern "C" size_t ggml_cuda_moe_stream_mm_device_bytes(void * context) {
    auto * mm = static_cast<moe_stream_mm *>(context);
    if (!mm) {
        return 0;
    }
    size_t pool = mm->fixup_bytes;
    if (mm->fixup_bytes) {
        const size_t capacity = mm->ctx->pool().capacity();
        if (capacity != SIZE_MAX) {
            pool = capacity;
        }
    }
    return pool + mm->dequant_bytes + (mm->workspace ? moe_stream_mm_workspace_bytes : 0);
}
