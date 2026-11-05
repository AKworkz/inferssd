// The grouped prefill engine for any expert format (vram_group_compute_api, vram-compute-api.h). The matrix products
// are the CUDA backend's (ggml_cuda_moe_stream_mm_*, its MUL_MAT kernels), normally linked into the backend like the decode
// engine's matrix-vector products (q2-compute-adapter.cu). Per expert and chunk of at most `chunk` tokens:
//   1. the chunk's input rows, gathered by row and quantized for the up/gate type (one operand for both);
//   2. up and gate;
//   3. the activation (SwiGLU, clamped or not, or SiTU);
//   4. the hidden rows quantized for the down type;
//   5. down;
//   6. per token: the post norm (multiply by its weights, RMS-normalize) when the layer has one, the routing weight,
//      and the add into the accumulator row. A group's rows ascend strictly and groups run in order on one stream, so
//      every row is written by one block at a time; nothing is atomic and the sum is deterministic.
#include "vram-compute-api.h"
#include "vram-activation.cuh"
#include "ggml.h"

#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include "backend-functions.h"
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {
using mm_supports_fn = bool (*)(int);
using mm_create_fn = void * (*)(int);
using mm_destroy_fn = void (*)(void *);
using mm_release_fn = void (*)(void *);
using mm_reserve_fn = bool (*)(void *, int, int64_t, int64_t, int64_t);
using mm_operand_bytes_fn = size_t (*)(void *, int, int64_t, int64_t);
using mm_quantize_fn = bool (*)(void *, int, const float *, const int32_t *, int64_t, int64_t, int64_t, void *, cudaStream_t);
using mm_fn = bool (*)(void *, int, const void *, int64_t, int64_t, const void *, int64_t, float *, cudaStream_t);
using mm_device_bytes_fn = size_t (*)(void *);

void cuda_check(cudaError_t rc, const char * operation) {
    if (rc != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(rc));
    }
}

size_t aligned(size_t bytes) { return (bytes + 255) & ~size_t(255); }

__global__ void group_activate(const float * up, const float * gate, float * hidden, int count,
        int kind, float situ_beta, float situ_linear_beta, float swiglu_limit) {
    const int i = int(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= count) {
        return;
    }
    hidden[i] = kimi_vram_activate(up[i], gate[i], kind, situ_beta, situ_linear_beta, swiglu_limit);
}

constexpr int scatter_threads = 256;

// One block per column (token) of the chunk: the post norm when there is one, the routing weight, the add into the
// token's accumulator row.
__global__ void group_scatter(const float * down, int latent, const int32_t * rows, const float * weights,
        const float * norm, float eps, float * accumulator) {
    const int column = blockIdx.x;
    const float * values = down + size_t(column) * latent;
    float scale = 1.0f;
    if (norm) {
        __shared__ float partial[scatter_threads / 32];
        float sum = 0.0f;
        for (int i = threadIdx.x; i < latent; i += blockDim.x) {
            const float v = values[i] * norm[i];
            sum += v * v;
        }
        for (int offset = 16; offset > 0; offset >>= 1) {
            sum += __shfl_xor_sync(0xffffffff, sum, offset);
        }
        if ((threadIdx.x & 31) == 0) {
            partial[threadIdx.x >> 5] = sum;
        }
        __syncthreads();
        sum = 0.0f;
        for (int w = 0; w < scatter_threads / 32; ++w) {
            sum += partial[w];
        }
        scale = 1.0f / sqrtf(sum / float(latent) + eps);
    }
    const float weight = weights[column];
    float * out = accumulator + size_t(rows[column]) * latent;
    for (int i = threadIdx.x; i < latent; i += blockDim.x) {
        const float v = norm ? values[i] * norm[i] * scale : values[i];
        out[i] += v * weight;
    }
}

struct group_engine {
    static constexpr int chunk = 256; // tokens of one expert per chain pass
    int device = -1, latent = 0, hidden = 0;
    // every format the engine serves (up/gate type, down type), and the types of the layer begin_layer started
    std::vector<std::pair<int, int>> formats;
    int up_type = 0, down_type = 0;
    mm_destroy_fn mm_destroy = nullptr;
    mm_release_fn mm_release = nullptr;
    mm_reserve_fn mm_reserve = nullptr;
    mm_operand_bytes_fn mm_operand_bytes = nullptr;
    mm_quantize_fn mm_quantize = nullptr;
    mm_fn mm = nullptr;
    mm_device_bytes_fn mm_device_bytes = nullptr;
    void * mm_context = nullptr;
    cudaStream_t stream = nullptr;
    // reserved with the first pass, released with the last: the chunk scratch and the post norm, and the products'
    // own reservations
    char * scratch = nullptr;
    size_t scratch_bytes = 0;
    float * up = nullptr, * gate = nullptr, * act = nullptr, * down = nullptr, * norm = nullptr;
    void * up_operand = nullptr, * down_operand = nullptr;
    float * host_norm = nullptr;
    // per pass
    int rows_capacity = 0, routes_capacity = 0;
    float * input = nullptr, * accumulator = nullptr;
    int32_t * route_rows = nullptr;
    float * route_weights = nullptr;
    float * host_input = nullptr, * host_output = nullptr;
    int32_t * host_rows = nullptr;
    float * host_weights = nullptr;
    // the layer
    int rows = 0, groups = 0;
    bool has_norm = false;
    vram_group_activation activation = {};
    std::vector<int32_t> first;
    std::vector<cudaEvent_t> events;      // two an expert, reused across layers
    std::vector<int> computed;            // groups computed this layer, in order
    std::vector<uint8_t> done_groups;
    cudaEvent_t begin_start = nullptr, begin_end = nullptr, finish_start = nullptr, finish_end = nullptr;
    uint64_t layer_h2d = 0;

    group_engine(int device_, int latent_, int hidden_, int count, const int32_t * weight_types, const int32_t * down_types) :
        device(device_), latent(latent_), hidden(hidden_) {
        if (latent <= 0 || hidden <= 0 || latent > 16384 || hidden > 16384 || latent % 4 || hidden % 4) {
            throw std::runtime_error("unsupported grouped VRAM expert geometry");
        }
        if (count < 1 || !weight_types || !down_types) {
            throw std::runtime_error("grouped VRAM engine formats");
        }
        for (int f = 0; f < count; ++f) {
            if (weight_types[f] < 0 || weight_types[f] >= GGML_TYPE_COUNT || down_types[f] < 0 || down_types[f] >= GGML_TYPE_COUNT) {
                throw std::runtime_error("grouped VRAM engine formats");
            }
            formats.push_back({ weight_types[f], down_types[f] });
        }
        up_type = formats[0].first;
        down_type = formats[0].second;
        const auto supports = ggml_cuda_moe_stream_mm_supports;
        const auto create = ggml_cuda_moe_stream_mm_create;
        mm_destroy = ggml_cuda_moe_stream_mm_destroy;
        mm_release = ggml_cuda_moe_stream_mm_release;
        mm_reserve = ggml_cuda_moe_stream_mm_reserve;
        mm_operand_bytes = ggml_cuda_moe_stream_mm_operand_bytes;
        mm_quantize = ggml_cuda_moe_stream_mm_quantize;
        mm = ggml_cuda_moe_stream_mm;
        mm_device_bytes = ggml_cuda_moe_stream_mm_device_bytes;
        try {
            if (!supports || !create || !mm_destroy || !mm_release || !mm_reserve || !mm_operand_bytes || !mm_quantize || !mm ||
                    !mm_device_bytes) {
                throw std::runtime_error("CUDA grouped expert products are missing from the SSD executor");
            }
            for (const auto & f : formats) {
                if (!supports(f.first) || !supports(f.second)) {
                    throw std::runtime_error(std::string("grouped VRAM engine: no CUDA matrix product for ") +
                        ggml_type_name(ggml_type(f.first)) + "/" + ggml_type_name(ggml_type(f.second)) + " experts");
                }
                if (latent % ggml_blck_size(ggml_type(f.first)) || hidden % ggml_blck_size(ggml_type(f.second))) {
                    throw std::runtime_error("grouped VRAM engine: expert rows are not whole blocks");
                }
            }
            cuda_check(cudaSetDevice(device), "cudaSetDevice");
            mm_context = create(device);
            if (!mm_context) {
                throw std::runtime_error("CUDA grouped expert product context");
            }
            cuda_check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "cudaStreamCreate");
            cuda_check(cudaMallocHost(reinterpret_cast<void **>(&host_norm), size_t(latent) * sizeof(float)), "cudaMallocHost grouped norm");
            for (cudaEvent_t * e : { &begin_start, &begin_end, &finish_start, &finish_end }) {
                cuda_check(cudaEventCreate(e), "cudaEventCreate");
            }
        } catch (...) {
            close();
            throw;
        }
    }

    // The part of a reservation that does not grow with the pass: the products' own (the stream-k fixup, a GEMM's
    // scratch, their kernels loaded), the chunk scratch and the post norm, and the engine's own kernels loaded (CUDA
    // loads a kernel, and takes device memory for it, on its first use).
    void reserve_fixed() {
        if (scratch) {
            return;
        }
        cudaFuncAttributes attributes = {};
        cuda_check(cudaFuncGetAttributes(&attributes, group_activate), "grouped activation kernel");
        cuda_check(cudaFuncGetAttributes(&attributes, group_scatter), "grouped scatter kernel");
        // up and gate: hidden rows of latent values; down: latent rows of hidden values. Every format's products (the
        // reservations add up), and operands as large as the largest format's.
        size_t up_operand_bytes = 0, down_operand_bytes = 0;
        for (const auto & f : formats) {
            if (!mm_reserve(mm_context, f.first, hidden, latent, chunk) || !mm_reserve(mm_context, f.second, latent, hidden, chunk)) {
                mm_release(mm_context);
                throw std::runtime_error("CUDA grouped expert product reservation");
            }
            up_operand_bytes = std::max(up_operand_bytes, mm_operand_bytes(mm_context, f.first, latent, chunk));
            down_operand_bytes = std::max(down_operand_bytes, mm_operand_bytes(mm_context, f.second, hidden, chunk));
        }
        const size_t hidden_bytes = aligned(size_t(chunk) * hidden * sizeof(float));
        const size_t latent_bytes = aligned(size_t(chunk) * latent * sizeof(float));
        scratch_bytes = 3 * hidden_bytes + latent_bytes + aligned(size_t(latent) * sizeof(float)) +
            aligned(up_operand_bytes) + aligned(down_operand_bytes);
        const cudaError_t rc = cudaMalloc(reinterpret_cast<void **>(&scratch), scratch_bytes);
        if (rc != cudaSuccess) {
            (void) cudaGetLastError();
            scratch = nullptr;
            scratch_bytes = 0;
            mm_release(mm_context);
            throw std::runtime_error(std::string("cudaMalloc grouped scratch: ") + cudaGetErrorString(rc));
        }
        char * p = scratch;
        auto take = [&](size_t bytes) { char * q = p; p += bytes; return q; };
        up = reinterpret_cast<float *>(take(hidden_bytes));
        gate = reinterpret_cast<float *>(take(hidden_bytes));
        act = reinterpret_cast<float *>(take(hidden_bytes));
        down = reinterpret_cast<float *>(take(latent_bytes));
        norm = reinterpret_cast<float *>(take(aligned(size_t(latent) * sizeof(float))));
        up_operand = take(aligned(up_operand_bytes));
        down_operand = take(aligned(down_operand_bytes));
        cuda_check(cudaMemsetAsync(scratch, 0, scratch_bytes, stream), "grouped scratch prefault");
        cuda_check(cudaStreamSynchronize(stream), "grouped scratch prefault drain");
    }

    void free_fixed() noexcept {
        if (scratch) { (void) cudaFree(scratch); scratch = nullptr; }
        scratch_bytes = 0;
        if (mm_context) { mm_release(mm_context); }
    }

    void free_pass() noexcept {
        if (input)         { (void) cudaFree(input);           input = nullptr; }
        if (accumulator)   { (void) cudaFree(accumulator);     accumulator = nullptr; }
        if (route_rows)    { (void) cudaFree(route_rows);      route_rows = nullptr; }
        if (route_weights) { (void) cudaFree(route_weights);   route_weights = nullptr; }
        if (host_input)    { (void) cudaFreeHost(host_input);  host_input = nullptr; }
        if (host_output)   { (void) cudaFreeHost(host_output); host_output = nullptr; }
        if (host_rows)     { (void) cudaFreeHost(host_rows);   host_rows = nullptr; }
        if (host_weights)  { (void) cudaFreeHost(host_weights); host_weights = nullptr; }
        rows_capacity = routes_capacity = 0;
    }

    void close() noexcept {
        if (device >= 0) { (void) cudaSetDevice(device); }
        if (stream) { (void) cudaStreamSynchronize(stream); }
        free_pass();
        for (cudaEvent_t e : events) { (void) cudaEventDestroy(e); }
        events.clear();
        for (cudaEvent_t * e : { &begin_start, &begin_end, &finish_start, &finish_end }) {
            if (*e) { (void) cudaEventDestroy(*e); *e = nullptr; }
        }
        free_fixed();
        if (host_norm) { (void) cudaFreeHost(host_norm); host_norm = nullptr; }
        if (mm_context) { mm_destroy(mm_context); mm_context = nullptr; }
        if (stream) { (void) cudaStreamDestroy(stream); stream = nullptr; }
    }

    ~group_engine() { close(); }

    size_t fixed_bytes() const { return scratch ? scratch_bytes + mm_device_bytes(mm_context) : 0; }

    void reserve(int rows_wanted, int routes_wanted) {
        if (rows_wanted < 1 || routes_wanted < 1) {
            throw std::runtime_error("grouped VRAM engine reserve");
        }
        cuda_check(cudaSetDevice(device), "cudaSetDevice");
        const bool first = !scratch;
        reserve_fixed();
        if (first) {
            std::string names;
            for (const auto & f : formats) {
                names += std::string(names.empty() ? "" : ",") + ggml_type_name(ggml_type(f.first)) + "/" + ggml_type_name(ggml_type(f.second));
            }
            std::fprintf(stderr, "[kimi_vram_group_engine] status=reserved format=%s latent=%d hidden=%d chunk=%d fixed_bytes=%zu\n",
                names.c_str(), latent, hidden, chunk, fixed_bytes());
        }
        if (rows_wanted <= rows_capacity && routes_wanted <= routes_capacity) {
            return;
        }
        cuda_check(cudaStreamSynchronize(stream), "grouped engine drain");
        const int r = std::max(rows_wanted, rows_capacity), n = std::max(routes_wanted, routes_capacity);
        free_pass();
        const size_t row_bytes = size_t(r) * latent * sizeof(float);
        try {
            cuda_check(cudaMalloc(reinterpret_cast<void **>(&input), row_bytes), "cudaMalloc grouped input");
            cuda_check(cudaMalloc(reinterpret_cast<void **>(&accumulator), row_bytes), "cudaMalloc grouped accumulator");
            cuda_check(cudaMalloc(reinterpret_cast<void **>(&route_rows), size_t(n) * sizeof(int32_t)), "cudaMalloc grouped routes");
            cuda_check(cudaMalloc(reinterpret_cast<void **>(&route_weights), size_t(n) * sizeof(float)), "cudaMalloc grouped weights");
            cuda_check(cudaMallocHost(reinterpret_cast<void **>(&host_input), row_bytes), "cudaMallocHost grouped input");
            cuda_check(cudaMallocHost(reinterpret_cast<void **>(&host_output), row_bytes), "cudaMallocHost grouped output");
            cuda_check(cudaMallocHost(reinterpret_cast<void **>(&host_rows), size_t(n) * sizeof(int32_t)), "cudaMallocHost grouped routes");
            cuda_check(cudaMallocHost(reinterpret_cast<void **>(&host_weights), size_t(n) * sizeof(float)), "cudaMallocHost grouped weights");
            cuda_check(cudaMemsetAsync(input, 0, row_bytes, stream), "grouped input prefault");
            cuda_check(cudaMemsetAsync(accumulator, 0, row_bytes, stream), "grouped accumulator prefault");
            cuda_check(cudaMemsetAsync(route_rows, 0, size_t(n) * sizeof(int32_t), stream), "grouped routes prefault");
            cuda_check(cudaMemsetAsync(route_weights, 0, size_t(n) * sizeof(float), stream), "grouped weights prefault");
            std::memset(host_input, 0, row_bytes); std::memset(host_output, 0, row_bytes);
            std::memset(host_rows, 0, size_t(n) * sizeof(int32_t)); std::memset(host_weights, 0, size_t(n) * sizeof(float));
            cuda_check(cudaStreamSynchronize(stream), "grouped pass prefault drain");
        } catch (...) {
            free_pass();
            throw;
        }
        rows_capacity = r;
        routes_capacity = n;
    }

    void release() {
        cuda_check(cudaSetDevice(device), "cudaSetDevice");
        cuda_check(cudaStreamSynchronize(stream), "grouped engine drain");
        free_pass();
        free_fixed();
        rows = groups = 0;
    }

    void begin_layer(int format, int rows_, const vram_group_route * routes, const int32_t * first_, int groups_,
            const vram_group_activation * activation_) {
        if (!activation_ || !scratch || rows_ < 1 || rows_ > rows_capacity || groups_ < 0 || (groups_ && (!routes || !first_))) {
            throw std::runtime_error("grouped VRAM engine layer shape");
        }
        if (format < 0 || format >= int(formats.size())) {
            throw std::runtime_error("grouped VRAM engine layer format");
        }
        const auto & a = activation_->base;
        if (a.kind == KIMI_VRAM_ACTIVATION_SWIGLU_CLAMP) {
            if (std::isnan(a.swiglu_limit) || a.swiglu_limit <= 1e-6f) {
                throw std::runtime_error("grouped VRAM engine invalid SwiGLU clamp");
            }
        } else if (a.kind == KIMI_VRAM_ACTIVATION_SITU) {
            if (!std::isfinite(a.situ_beta) || a.situ_beta <= 0.0f || !std::isfinite(a.situ_linear_beta)) {
                throw std::runtime_error("grouped VRAM engine invalid SiTU parameters");
            }
        } else if (a.kind != KIMI_VRAM_ACTIVATION_SWIGLU) {
            throw std::runtime_error("grouped VRAM engine activation kind");
        }
        if (activation_->post_norm && (!std::isfinite(activation_->post_norm_eps) || activation_->post_norm_eps <= 0.0f)) {
            throw std::runtime_error("grouped VRAM engine post-norm epsilon");
        }
        const int count = groups_ ? first_[groups_] : 0;
        if (groups_ && (first_[0] != 0 || count > routes_capacity)) {
            throw std::runtime_error("grouped VRAM engine route list");
        }
        for (int g = 0; g < groups_; ++g) {
            if (first_[g + 1] < first_[g]) {
                throw std::runtime_error("grouped VRAM engine route groups");
            }
            for (int i = first_[g]; i < first_[g + 1]; ++i) {
                if (routes[i].row < 0 || routes[i].row >= rows_ || (i > first_[g] && routes[i].row <= routes[i - 1].row) ||
                        !std::isfinite(routes[i].weight) || routes[i].weight < 0.0f) {
                    throw std::runtime_error("grouped VRAM engine route mapping");
                }
            }
        }
        cuda_check(cudaSetDevice(device), "cudaSetDevice");
        // the staging below may still feed a layer that was not finished
        cuda_check(cudaStreamSynchronize(stream), "grouped engine drain");
        up_type = formats[size_t(format)].first;
        down_type = formats[size_t(format)].second;
        rows = rows_;
        groups = groups_;
        activation = *activation_;
        has_norm = activation_->post_norm != nullptr;
        first.assign(first_, first_ + (groups_ ? groups_ + 1 : 0));
        computed.clear();
        done_groups.assign(size_t(groups_), 0);
        while (events.size() < 2 * size_t(groups_)) {
            cudaEvent_t e = nullptr;
            cuda_check(cudaEventCreate(&e), "cudaEventCreate");
            events.push_back(e);
        }
        for (int i = 0; i < count; ++i) {
            host_rows[i] = routes[i].row;
            host_weights[i] = routes[i].weight;
        }
        const size_t input_bytes = size_t(rows) * latent * sizeof(float);
        cuda_check(cudaEventRecord(begin_start, stream), "event");
        cuda_check(cudaMemcpyAsync(input, host_input, input_bytes, cudaMemcpyHostToDevice, stream), "grouped input H2D");
        layer_h2d = input_bytes;
        if (count) {
            cuda_check(cudaMemcpyAsync(route_rows, host_rows, size_t(count) * sizeof(int32_t), cudaMemcpyHostToDevice, stream),
                "grouped routes H2D");
            cuda_check(cudaMemcpyAsync(route_weights, host_weights, size_t(count) * sizeof(float), cudaMemcpyHostToDevice, stream),
                "grouped weights H2D");
            layer_h2d += size_t(count) * (sizeof(int32_t) + sizeof(float));
        }
        if (has_norm) {
            std::memcpy(host_norm, activation_->post_norm, size_t(latent) * sizeof(float));
            cuda_check(cudaMemcpyAsync(norm, host_norm, size_t(latent) * sizeof(float), cudaMemcpyHostToDevice, stream),
                "grouped post norm H2D");
            layer_h2d += size_t(latent) * sizeof(float);
        }
        cuda_check(cudaMemsetAsync(accumulator, 0, input_bytes, stream), "grouped accumulator zero");
        cuda_check(cudaEventRecord(begin_end, stream), "event");
    }

    void compute(int group, const vram_compute_expert & w, cudaEvent_t ready, cudaEvent_t * done) {
        if (group < 0 || group >= groups || done_groups[size_t(group)]) {
            throw std::runtime_error("grouped VRAM engine group");
        }
        if (!w.up || !w.gate || !w.down) {
            throw std::runtime_error("grouped VRAM engine null weight");
        }
        cuda_check(cudaSetDevice(device), "cudaSetDevice");
        if (ready) {
            cuda_check(cudaStreamWaitEvent(stream, ready, 0), "grouped engine wait");
        }
        const size_t slot = computed.size();
        cudaEvent_t start = events[2 * slot], end = events[2 * slot + 1];
        cuda_check(cudaEventRecord(start, stream), "event");
        const auto & a = activation.base;
        for (int c = first[size_t(group)]; c < first[size_t(group) + 1]; c += chunk) {
            const int n = std::min(chunk, first[size_t(group) + 1] - c);
            if (!mm_quantize(mm_context, up_type, input, route_rows + c, latent, latent, n, up_operand, stream) ||
                    !mm(mm_context, up_type, w.up, hidden, latent, up_operand, n, up, stream) ||
                    !mm(mm_context, up_type, w.gate, hidden, latent, up_operand, n, gate, stream)) {
                throw std::runtime_error("grouped VRAM engine up/gate launch failed");
            }
            const int elements = n * hidden;
            group_activate<<<(elements + 255) / 256, 256, 0, stream>>>(up, gate, act, elements,
                a.kind, a.situ_beta, a.situ_linear_beta, a.swiglu_limit);
            cuda_check(cudaGetLastError(), "grouped activation launch");
            if (!mm_quantize(mm_context, down_type, act, nullptr, hidden, hidden, n, down_operand, stream) ||
                    !mm(mm_context, down_type, w.down, latent, hidden, down_operand, n, down, stream)) {
                throw std::runtime_error("grouped VRAM engine down launch failed");
            }
            group_scatter<<<n, scatter_threads, 0, stream>>>(down, latent, route_rows + c, route_weights + c,
                has_norm ? norm : nullptr, activation.post_norm_eps, accumulator);
            cuda_check(cudaGetLastError(), "grouped scatter launch");
        }
        cuda_check(cudaEventRecord(end, stream), "event");
        computed.push_back(group);
        done_groups[size_t(group)] = 1;
        if (done) {
            *done = end;
        }
    }

    void finish_layer(const float ** output, vram_group_layer_stats * stats, double * group_us) {
        if (!output) {
            throw std::runtime_error("grouped VRAM engine null output");
        }
        if (computed.size() != size_t(groups)) {
            // a skipped expert's contributions would be missing from its tokens' rows
            throw std::runtime_error("grouped VRAM engine finish before every group was computed");
        }
        cuda_check(cudaSetDevice(device), "cudaSetDevice");
        const size_t bytes = size_t(rows) * latent * sizeof(float);
        cuda_check(cudaEventRecord(finish_start, stream), "event");
        cuda_check(cudaMemcpyAsync(host_output, accumulator, bytes, cudaMemcpyDeviceToHost, stream), "grouped output D2H");
        cuda_check(cudaEventRecord(finish_end, stream), "event");
        cuda_check(cudaEventSynchronize(finish_end), "grouped engine finish");
        *output = host_output;
        float ms = 0.0f;
        if (stats || group_us) {
            vram_group_layer_stats local = {};
            vram_group_layer_stats & st = stats ? *stats : local;
            st = {};
            for (size_t i = 0; i < computed.size(); ++i) {
                const int g = computed[i];
                const double routes = double(first[size_t(g) + 1] - first[size_t(g)]);
                cuda_check(cudaEventElapsedTime(&ms, events[2 * i], events[2 * i + 1]), "event time");
                const double us = 1e3 * double(ms);
                if (group_us) { group_us[g] = us; }
                st.experts += 1;
                st.routes += routes;
                st.routes_sq += routes * routes;
                st.us += us;
                st.route_us += routes * us;
            }
            if (!computed.empty()) {
                cuda_check(cudaEventElapsedTime(&ms, events[0], events[2 * computed.size() - 1]), "event time");
                st.span_us = 1e3 * double(ms);
            }
        }
        if (stats) {
            cuda_check(cudaEventElapsedTime(&ms, begin_start, begin_end), "event time");
            stats->begin_us = 1e3 * double(ms);
            cuda_check(cudaEventElapsedTime(&ms, finish_start, finish_end), "event time");
            stats->finish_us = 1e3 * double(ms);
            stats->h2d_bytes = layer_h2d;
            stats->d2h_bytes = bytes;
        }
    }

    void geometry(vram_group_geometry & g) const {
        g.route_bytes = sizeof(int32_t) + sizeof(float);
        g.token_bytes = size_t(latent) * sizeof(float);
        g.output_bytes = size_t(latent) * sizeof(float);
        g.fixed_bytes = fixed_bytes();
        g.bytes_per_token = 2 * size_t(latent) * sizeof(float);
    }

    size_t device_bytes() const {
        return fixed_bytes() + size_t(rows_capacity) * 2 * latent * sizeof(float) + size_t(routes_capacity) * (sizeof(int32_t) + sizeof(float));
    }

    size_t host_bytes() const {
        return size_t(rows_capacity) * 2 * latent * sizeof(float) + size_t(routes_capacity) * (sizeof(int32_t) + sizeof(float)) +
            size_t(latent) * sizeof(float);
    }
};

int fail(const std::exception & e, char * error, size_t error_size) {
    if (error && error_size) { std::snprintf(error, error_size, "%s", e.what()); }
    return 1;
}

void * create(int device, int latent, int hidden, int formats, const int32_t * weight_types, const int32_t * down_types,
        char * error, size_t error_size) {
    try {
        return new group_engine(device, latent, hidden, formats, weight_types, down_types);
    } catch (const std::exception & e) {
        fail(e, error, error_size);
        return nullptr;
    }
}

void destroy(void * context) { delete static_cast<group_engine *>(context); }

int reserve(void * context, int rows, int routes, char * error, size_t error_size) {
    try {
        if (!context) { throw std::runtime_error("null grouped VRAM engine"); }
        static_cast<group_engine *>(context)->reserve(rows, routes);
        return 0;
    } catch (const std::exception & e) { return fail(e, error, error_size); }
}

int release(void * context, char * error, size_t error_size) {
    try {
        if (!context) { throw std::runtime_error("null grouped VRAM engine"); }
        static_cast<group_engine *>(context)->release();
        return 0;
    } catch (const std::exception & e) { return fail(e, error, error_size); }
}

float * input_rows(void * context) { return context ? static_cast<group_engine *>(context)->host_input : nullptr; }

int begin_layer(void * context, int format, int rows, const vram_group_route * routes, const int32_t * first, int groups,
        const vram_group_activation * activation, char * error, size_t error_size) {
    try {
        if (!context) { throw std::runtime_error("null grouped VRAM engine"); }
        static_cast<group_engine *>(context)->begin_layer(format, rows, routes, first, groups, activation);
        return 0;
    } catch (const std::exception & e) { return fail(e, error, error_size); }
}

int compute(void * context, int group, const vram_compute_expert * weights, CUevent_st * ready, CUevent_st ** done,
        char * error, size_t error_size) {
    try {
        if (!context || !weights) { throw std::runtime_error("null grouped VRAM engine argument"); }
        cudaEvent_t end = nullptr;
        static_cast<group_engine *>(context)->compute(group, *weights, ready, done ? &end : nullptr);
        if (done) { *done = end; }
        return 0;
    } catch (const std::exception & e) { return fail(e, error, error_size); }
}

int finish_layer(void * context, const float ** output, vram_group_layer_stats * stats, double * group_us,
        char * error, size_t error_size) {
    try {
        if (!context) { throw std::runtime_error("null grouped VRAM engine"); }
        static_cast<group_engine *>(context)->finish_layer(output, stats, group_us);
        return 0;
    } catch (const std::exception & e) { return fail(e, error, error_size); }
}

void geometry(void * context, vram_group_geometry * g) {
    if (!g) { return; }
    *g = {};
    if (context) { static_cast<group_engine *>(context)->geometry(*g); }
}

size_t device_bytes(void * context) { return context ? static_cast<group_engine *>(context)->device_bytes() : 0; }
size_t host_bytes(void * context) { return context ? static_cast<group_engine *>(context)->host_bytes() : 0; }
} // namespace

extern "C" __attribute__((visibility("default")))
const vram_group_compute_api * kimi_vram_group_compute_get_api() {
    static const vram_group_compute_api api = {
        3, sizeof(vram_group_compute_api), create, destroy, reserve, release, input_rows, begin_layer, compute,
        finish_layer, geometry, device_bytes, host_bytes
    };
    return &api;
}
