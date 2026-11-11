// The CUDA backend's grouped-expert matrix products (ggml_cuda_moe_stream_mm_*, ggml/src/ggml-cuda/ssd/group-matmul.cu)
// against ggml's CPU MUL_MAT on the same data. Every weight type the decode engine serves
// (ggml_cuda_moe_stream_mmvq_supports, which must be the products' set) at 1, 7, 26, 129, 150 and 300 activation
// columns, gathered from scattered rows of a wider matrix by device row ids, and once without ids: NMSE <= 5e-4, or
// 2e-2 where MXFP4 takes FP4 activations (Blackwell), as test-backend-ops allows for MUL_MAT, on test-backend-ops' data
// (weights and activations uniform in [-1, 1], an importance of ones where the type needs one). Each line also gives
// both backends' NMSE against the exact product of the dequantized weights (doubles), for the smaller shapes, and there
// the GPU's must be at most 1.5x the CPU's (+1e-6): its activation sums are those of the quantized values. Where MXFP4
// takes FP4 activations, the product must also match ggml's own CUDA MUL_MAT on the same data (NMSE <= 1e-8; ggml takes
// MMQ above 8 columns, so fewer columns are compared within a product of 9). Every type also at a row length that is
// not a multiple of 512 (the first multiple of its block size from 1040): a row that is not a whole number of MMQ's
// iterations must not be read past its end. Every matrix ends its allocation and 0xFF bytes follow it, which decode to
// NaN scales, so a read past the matrix shows as a non-finite product. Q4_K and Q5_K also at Muse's expert shape (4096
// x 4096). One product's time per type at 8, 26, 150 and 300 columns is printed. After reserve, no product changes the
// device memory the context holds, nor the device's free memory (reserve loads the kernels). Needs an owned GPU, so it
// is not registered with ctest.
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-cuda.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

extern "C" {
bool   ggml_cuda_moe_stream_mmvq_supports(int type);
bool   ggml_cuda_moe_stream_mm_supports(int type);
void * ggml_cuda_moe_stream_mm_create(int device);
void   ggml_cuda_moe_stream_mm_destroy(void * mm);
int    ggml_cuda_moe_stream_mm_kernel(void * mm, int type, int64_t ncols, int64_t n);
bool   ggml_cuda_moe_stream_mm_reserve(void * mm, int type, int64_t nrows, int64_t ncols, int64_t max_columns);
size_t ggml_cuda_moe_stream_mm_operand_bytes(void * mm, int type, int64_t ncols, int64_t max_columns);
bool   ggml_cuda_moe_stream_mm_quantize(void * mm, int type, const float * x, const int32_t * ids, int64_t x_stride,
                                        int64_t ncols, int64_t n, void * operand, cudaStream_t stream);
bool   ggml_cuda_moe_stream_mm(void * mm, int type, const void * w, int64_t nrows, int64_t ncols, const void * operand,
                               int64_t n, float * dst, cudaStream_t stream);
size_t ggml_cuda_moe_stream_mm_device_bytes(void * mm);
}

namespace {
void require(bool ok, const std::string & what) {
    if (!ok) { throw std::runtime_error(what); }
}
void cuda(cudaError_t rc, const char * what) {
    if (rc != cudaSuccess) { throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(rc)); }
}

const char * kernel_name(int kernel) {
    static const char * names[] = { "mmvf", "mmvq", "mmq", "mmq_fp4", "gemm" };
    return kernel >= 0 && kernel < 5 ? names[kernel] : "none";
}

struct device_buffer {
    void * p = nullptr;
    explicit device_buffer(size_t bytes) { cuda(cudaMalloc(&p, std::max<size_t>(bytes, 1)), "cudaMalloc"); }
    ~device_buffer() { if (p) { (void) cudaFree(p); } }
    template <class T> T * as() const { return static_cast<T *>(p); }
};

// dst[n][nrows] = w x columns, on ggml's CPU backend (its own activation quantization and dot products)
std::vector<float> cpu_mul_mat(ggml_type type, const std::vector<uint8_t> & w, int64_t nrows, int64_t ncols,
        const std::vector<float> & columns, int64_t n) {
    const size_t bytes = w.size() + columns.size() * sizeof(float) + size_t(nrows) * n * sizeof(float);
    ggml_init_params params = { bytes + size_t(n) * ncols * 8 + (64u << 20), nullptr, false };
    ggml_context * ctx = ggml_init(params);
    require(ctx, "CPU reference context");
    ggml_tensor * a = ggml_new_tensor_2d(ctx, type, ncols, nrows);
    ggml_tensor * b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ncols, n);
    std::memcpy(a->data, w.data(), w.size());
    std::memcpy(b->data, columns.data(), columns.size() * sizeof(float));
    ggml_tensor * c = ggml_mul_mat(ctx, a, b);
    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, c);
    require(ggml_graph_compute_with_ctx(ctx, graph, int(std::max(1u, std::thread::hardware_concurrency()))) == GGML_STATUS_SUCCESS,
        "CPU reference compute");
    std::vector<float> out(size_t(nrows) * n);
    std::memcpy(out.data(), c->data, out.size() * sizeof(float));
    ggml_free(ctx);
    return out;
}

// dst[n][nrows] = w x columns on ggml's CUDA MUL_MAT. With fewer than 9 columns ggml takes MMVQ, so the product is of
// the columns repeated to 9 (each output column depends on its own input column only) and the first n are kept.
std::vector<float> cuda_mul_mat(ggml_backend_t backend, ggml_type type, const std::vector<uint8_t> & w, int64_t nrows,
        int64_t ncols, const std::vector<float> & columns, int64_t n) {
    const int64_t m = std::max<int64_t>(n, 9);
    std::vector<float> wide(size_t(m) * ncols);
    for (int64_t c = 0; c < m; ++c) {
        std::memcpy(wide.data() + c * ncols, columns.data() + std::min(c, n - 1) * ncols, size_t(ncols) * sizeof(float));
    }
    ggml_init_params params = { 4 * ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(params);
    require(ctx, "CUDA reference context");
    ggml_tensor * a = ggml_new_tensor_2d(ctx, type, ncols, nrows);
    ggml_tensor * b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ncols, m);
    ggml_tensor * c = ggml_mul_mat(ctx, a, b);
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    require(buffer, "CUDA reference buffer");
    ggml_backend_tensor_set(a, w.data(), 0, w.size());
    ggml_backend_tensor_set(b, wide.data(), 0, wide.size() * sizeof(float));
    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, c);
    require(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS, "CUDA reference compute");
    std::vector<float> out(size_t(nrows) * m);
    ggml_backend_tensor_get(c, out.data(), 0, out.size() * sizeof(float));
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    out.resize(size_t(nrows) * n);
    return out;
}

// dst[n][nrows] = the dequantized w times the columns, in doubles
std::vector<float> exact_mul_mat(ggml_type type, const std::vector<uint8_t> & w, int64_t nrows, int64_t ncols,
        const std::vector<float> & columns, int64_t n) {
    std::vector<float> weights(size_t(nrows) * ncols), out(size_t(nrows) * n);
    const auto * traits = ggml_get_type_traits(type);
    const size_t row_bytes = ggml_row_size(type, ncols);
    for (int64_t r = 0; r < nrows; ++r) {
        traits->to_float(w.data() + size_t(r) * row_bytes, weights.data() + size_t(r) * ncols, ncols);
    }
    for (int64_t c = 0; c < n; ++c) {
        for (int64_t r = 0; r < nrows; ++r) {
            double sum = 0.0;
            for (int64_t k = 0; k < ncols; ++k) { sum += double(weights[size_t(r) * ncols + k]) * double(columns[size_t(c) * ncols + k]); }
            out[size_t(c) * nrows + r] = float(sum);
        }
    }
    return out;
}

size_t device_free() {
    size_t free = 0, total = 0;
    cuda(cudaMemGetInfo(&free, &total), "cudaMemGetInfo");
    return free;
}

double nmse(const std::vector<float> & got, const std::vector<float> & expected) {
    double error = 0.0, norm = 0.0;
    for (size_t i = 0; i < got.size(); ++i) {
        const double d = double(got[i]) - double(expected[i]);
        error += d * d;
        norm += double(expected[i]) * double(expected[i]);
    }
    return norm > 0.0 ? error / norm : error;
}

struct totals { size_t checks = 0; double worst = 0.0; };

// One product's time at 8, 26, 150 and 300 columns, after a warm-up, and the operand's quantization apart.
void time_products(void * mm, ggml_type type, int64_t nrows, int64_t ncols, const void * w, const float * x, const int32_t * ids,
        int64_t stride, void * operand, float * dst, cudaStream_t stream) {
    const std::vector<int64_t> timed = { 8, 26, 150, 300 };
    cudaEvent_t start, end;
    cuda(cudaEventCreate(&start), "event");
    cuda(cudaEventCreate(&end), "event");
    std::string times;
    for (const int64_t n : timed) {
        constexpr int reps = 20;
        float quantize_ms = 0, mm_ms = 0;
        for (int pass = 0; pass < 2; ++pass) {
            cuda(cudaEventRecord(start, stream), "record");
            for (int r = 0; r < reps; ++r) {
                require(ggml_cuda_moe_stream_mm_quantize(mm, type, x, ids, stride, ncols, n, operand, stream), "quantize");
            }
            cuda(cudaEventRecord(end, stream), "record");
            cuda(cudaEventSynchronize(end), "sync");
            cuda(cudaEventElapsedTime(&quantize_ms, start, end), "elapsed");
            cuda(cudaEventRecord(start, stream), "record");
            for (int r = 0; r < reps; ++r) {
                require(ggml_cuda_moe_stream_mm(mm, type, w, nrows, ncols, operand, n, dst, stream), "mm");
            }
            cuda(cudaEventRecord(end, stream), "record");
            cuda(cudaEventSynchronize(end), "sync");
            cuda(cudaEventElapsedTime(&mm_ms, start, end), "elapsed");
        }
        const double us = 1e3 * mm_ms / reps;
        char item[128];
        std::snprintf(item, sizeof(item), " n%lld=%.1fus(%.1f TFLOPS, quantize %.1fus)", (long long) n, us,
            2.0 * double(nrows) * double(ncols) * double(n) / us * 1e-6, 1e3 * quantize_ms / reps);
        times += item;
    }
    (void) cudaEventDestroy(start);
    (void) cudaEventDestroy(end);
    std::printf("[kimi_vram_mm_time] type=%s rows=%lld cols=%lld%s\n", ggml_type_name(type), (long long) nrows, (long long) ncols,
        times.c_str());
}

constexpr size_t poison_bytes = 4096; // after every matrix: a read past it meets NaN scales

void run_type(void * mm, ggml_backend_t backend, ggml_type type, int64_t nrows, int64_t ncols, cudaStream_t stream, totals & t,
        bool timing) {
    const std::vector<int64_t> counts = { 1, 7, 26, 129, 150, 300 };
    const int64_t max_columns = 300, rows_total = 512, stride = ncols + 64; // x is wider than a column and scattered
    std::mt19937 rng(uint32_t(20261021 + int(type) * 131 + ncols));
    std::uniform_real_distribution<float> sample(-1.0f, 1.0f);
    const bool exact = nrows * ncols <= 1 << 20;

    std::vector<float> values(size_t(nrows) * ncols);
    for (auto & v : values) { v = sample(rng); }
    std::vector<uint8_t> w(ggml_row_size(type, ncols) * nrows);
    const std::vector<float> importance(ggml_quantize_requires_imatrix(type) ? size_t(ncols) : 0, 1.0f);
    require(ggml_quantize_chunk(type, values.data(), w.data(), 0, nrows, ncols, importance.empty() ? nullptr : importance.data()) == w.size(),
        std::string("quantize ") + ggml_type_name(type));
    std::vector<float> x(size_t(rows_total) * stride);
    for (auto & v : x) { v = sample(rng); }

    require(ggml_cuda_moe_stream_mm_reserve(mm, type, nrows, ncols, max_columns), std::string("reserve ") + ggml_type_name(type));
    const size_t operand_bytes = ggml_cuda_moe_stream_mm_operand_bytes(mm, type, ncols, max_columns);
    require(operand_bytes > 0, "operand bytes");
    const size_t reserved = ggml_cuda_moe_stream_mm_device_bytes(mm);

    device_buffer d_w(w.size() + poison_bytes), d_x(x.size() * sizeof(float)), d_ids(size_t(max_columns) * sizeof(int32_t)),
        d_operand(operand_bytes), d_dst(size_t(max_columns) * nrows * sizeof(float));
    cuda(cudaMemcpy(d_w.p, w.data(), w.size(), cudaMemcpyHostToDevice), "weights H2D");
    cuda(cudaMemset(d_w.as<char>() + w.size(), 0xff, poison_bytes), "weights poison");
    cuda(cudaMemcpy(d_x.p, x.data(), x.size() * sizeof(float), cudaMemcpyHostToDevice), "x H2D");

    std::string line;
    double loosest = 0.0;
    for (const int64_t n : counts) {
        const bool fp4 = ggml_cuda_moe_stream_mm_kernel(mm, type, ncols, n) == 3; // MMQ with FP4 activations
        const double tolerance = fp4 ? 2e-2 : 5e-4;
        loosest = std::max(loosest, tolerance);
        for (const bool with_ids : { true, false }) {
            if (!with_ids && n != 26) { continue; }
            std::vector<int32_t> ids(static_cast<size_t>(rows_total));
            for (int32_t i = 0; i < int32_t(rows_total); ++i) { ids[size_t(i)] = i; }
            std::shuffle(ids.begin(), ids.end(), rng);
            ids.resize(size_t(n));
            if (!with_ids) { for (int32_t i = 0; i < int32_t(n); ++i) { ids[size_t(i)] = i; } }
            std::vector<float> columns(size_t(n) * ncols);
            for (int64_t c = 0; c < n; ++c) {
                std::memcpy(columns.data() + c * ncols, x.data() + size_t(ids[size_t(c)]) * stride, size_t(ncols) * sizeof(float));
            }
            cuda(cudaMemcpy(d_ids.p, ids.data(), ids.size() * sizeof(int32_t), cudaMemcpyHostToDevice), "ids H2D");
            cuda(cudaMemset(d_dst.p, 0xff, size_t(n) * nrows * sizeof(float)), "dst poison");
            const size_t free_before = device_free();
            require(ggml_cuda_moe_stream_mm_quantize(mm, type, d_x.as<float>(), with_ids ? d_ids.as<int32_t>() : nullptr,
                stride, ncols, n, d_operand.p, stream), "quantize");
            require(ggml_cuda_moe_stream_mm(mm, type, d_w.p, nrows, ncols, d_operand.p, n, d_dst.as<float>(), stream), "mm");
            cuda(cudaStreamSynchronize(stream), "product");
            require(device_free() == free_before, std::string("a product took device memory after reserve: ") +
                ggml_type_name(type) + " n=" + std::to_string(n));
            std::vector<float> got(size_t(n) * nrows);
            cuda(cudaMemcpy(got.data(), d_dst.p, got.size() * sizeof(float), cudaMemcpyDeviceToHost), "dst D2H");
            const auto expected = cpu_mul_mat(type, w, nrows, ncols, columns, n);
            for (float v : got) { require(std::isfinite(v), std::string("non-finite product ") + ggml_type_name(type)); }
            const double e = nmse(got, expected);
            t.worst = std::max(t.worst, tolerance == 5e-4 ? e : 0.0);
            require(e <= tolerance, std::string("NMSE ") + std::to_string(e) + " above " + std::to_string(tolerance) + " for " +
                ggml_type_name(type) + " n=" + std::to_string(n) + (with_ids ? "" : " without ids"));
            const std::string what = std::string(ggml_type_name(type)) + " rows=" + std::to_string(nrows) + " cols=" +
                std::to_string(ncols) + " n=" + std::to_string(n) + (with_ids ? "" : " without ids");
            char item[192];
            int used = std::snprintf(item, sizeof(item), " n%lld%s=%s:%.2e", (long long) n, with_ids ? "" : "/seq",
                kernel_name(ggml_cuda_moe_stream_mm_kernel(mm, type, ncols, n)), e);
            if (fp4) {
                // the FP4 path's own reference: ggml's MUL_MAT takes the same quantizer and kernel
                const double g = nmse(got, cuda_mul_mat(backend, type, w, nrows, ncols, columns, n));
                used += std::snprintf(item + used, sizeof(item) - used, "(ggml_cuda=%.1e)", g);
                require(g <= 1e-8, "the FP4 product differs from ggml's CUDA MUL_MAT for " + what + ": NMSE " + std::to_string(g));
                ++t.checks;
            } else if (exact) {
                const auto truth = exact_mul_mat(type, w, nrows, ncols, columns, n);
                const double gpu = nmse(got, truth), cpu = nmse(expected, truth);
                used += std::snprintf(item + used, sizeof(item) - used, "(exact:gpu=%.1e,cpu=%.1e)", gpu, cpu);
                require(gpu <= 1.5 * cpu + 1e-6, "the GPU product is farther from the exact one than the CPU's for " + what +
                    ": " + std::to_string(gpu) + " against " + std::to_string(cpu));
                ++t.checks;
            }
            line += item;
            ++t.checks;
        }
    }
    std::printf("[kimi_vram_mm] type=%s rows=%lld cols=%lld tolerance=%.0e%s\n", ggml_type_name(type), (long long) nrows,
        (long long) ncols, loosest, line.c_str());

    if (timing) {
        const size_t free_before = device_free();
        time_products(mm, type, nrows, ncols, d_w.p, d_x.as<float>(), d_ids.as<int32_t>(), stride, d_operand.p, d_dst.as<float>(), stream);
        require(device_free() == free_before, std::string("a timed product took device memory: ") + ggml_type_name(type));
    }

    const size_t after = ggml_cuda_moe_stream_mm_device_bytes(mm);
    require(after == reserved, std::string("the context grew after reserve: ") + std::to_string(reserved) + " -> " +
        std::to_string(after) + " bytes for " + ggml_type_name(type));
    ++t.checks;
}
} // namespace

int main() {
    try {
        ggml_cpu_init();
        int device = 0;
        cuda(cudaSetDevice(device), "cudaSetDevice");
        cudaDeviceProp prop = {};
        cuda(cudaGetDeviceProperties(&prop, device), "properties");
        cudaStream_t stream = nullptr;
        cuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "stream");
        // the grouped products serve exactly the types the decode engine serves
        std::vector<ggml_type> types;
        for (int type = 0; type < GGML_TYPE_COUNT; ++type) {
            if (ggml_cuda_moe_stream_mm_supports(type) != ggml_cuda_moe_stream_mmvq_supports(type)) {
                throw std::runtime_error("the grouped and decode type sets differ at type " + std::to_string(type));
            }
            if (ggml_cuda_moe_stream_mm_supports(type)) {
                types.push_back(ggml_type(type));
            }
        }
        const size_t covered = types.size();
        ggml_backend_t backend = ggml_backend_cuda_init(device);
        require(backend, "ggml CUDA backend");
        totals t;
        // a fresh context per type, so each one's reservation is checked alone
        for (const ggml_type type : types) {
            void * mm = ggml_cuda_moe_stream_mm_create(device);
            require(mm, "create");
            run_type(mm, backend, type, 320, 1024, stream, t, true);
            run_type(mm, backend, type, 320, GGML_PAD(1040, ggml_blck_size(type)), stream, t, false);
            ggml_cuda_moe_stream_mm_destroy(mm);
        }
        // Muse Spark's routed experts: up/gate Q4_K and down Q5_K, 4096 x 4096 each
        void * mm = ggml_cuda_moe_stream_mm_create(device);
        require(mm, "create");
        run_type(mm, backend, GGML_TYPE_Q4_K, 4096, 4096, stream, t, true);
        run_type(mm, backend, GGML_TYPE_Q5_K, 4096, 4096, stream, t, true);
        ggml_cuda_moe_stream_mm_destroy(mm);
        ggml_backend_free(backend);
        cuda(cudaStreamDestroy(stream), "stream destroy");
        std::printf("KIMI_VRAM_MM_PASS device=%s cc=%d.%d types=%zu checks=%zu worst_nmse=%.3g\n", prop.name, prop.major,
            prop.minor, covered, t.checks, t.worst);
        return 0;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "KIMI_VRAM_MM_FAIL: %s\n", e.what());
        return 1;
    }
}
