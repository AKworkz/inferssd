#include "vram-compute-api.h"
#include "vram-activation.cuh"
#include "ggml.h"
#define GGML_COMMON_DECL_CUDA
#include "../../ggml-common.h"

#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include "backend-functions.h"
#include <stdexcept>
#include <string>

namespace {
using mmvq_fn = bool (*)(int, const void *, int64_t, int64_t, size_t,
                         const float *, void *, float *, cudaStream_t);
using clock_type = std::chrono::steady_clock;

void cuda_check(cudaError_t rc, const char * operation) {
    if (rc != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(rc));
    }
}

size_t q8_1_bytes(int columns) {
    constexpr int row_padding = 512;
    const size_t padded = (size_t(columns) + row_padding - 1) / row_padding * row_padding;
    return padded / QK8_1 * sizeof(block_q8_1);
}

__global__ void generic_activate(const float * up_gate, float * hidden, int hits, int rows,
        int kind, float situ_beta, float situ_linear_beta, float swiglu_limit) {
    const int index = int(blockIdx.x * blockDim.x + threadIdx.x);
    if (index >= hits * rows) {
        return;
    }
    const int hit = index / rows;
    const int row = index - hit * rows;
    hidden[index] = kimi_vram_activate(up_gate[(2 * hit) * rows + row], up_gate[(2 * hit + 1) * rows + row], kind,
        situ_beta, situ_linear_beta, swiglu_limit);
}

struct generic_engine {
    int device = -1;
    int latent = 0;
    int hidden = 0;
    int max_hits = 0;
    int up_type = GGML_TYPE_COUNT;
    int down_type_ = GGML_TYPE_COUNT;
    size_t up_stride = 0;
    size_t down_stride = 0;
    size_t q8_bytes = 0;
    cudaStream_t stream = nullptr;
    // A call's input row is staged at its first row: calls in flight at once do not share it.
    float * device_input = nullptr;
    float * device_up_gate = nullptr;
    float * device_hidden = nullptr;
    float * device_down = nullptr;
    void * device_q8 = nullptr;
    float * host_input = nullptr;
    float * host_down = nullptr;
    mmvq_fn mmvq = nullptr;
    uint64_t calls = 0;
    uint64_t hits = 0;
    double total_us = 0.0;
    bool poisoned = false;
    bool closed = false;

    generic_engine(int device_, int latent_, int hidden_, int max_hits_, int weight_type, int down_type) :
        device(device_), latent(latent_), hidden(hidden_), max_hits(max_hits_) {
        if (weight_type == GGML_TYPE_IQ2_XXS && down_type == GGML_TYPE_Q2_K) {
            throw std::runtime_error("IQ2_XXS/Q2_K expert format is retired");
        }
        auto is_float = [](int type) { return type == GGML_TYPE_BF16 || type == GGML_TYPE_F16; };
        const bool floating = is_float(weight_type) && is_float(down_type);
        if (latent <= 0 || hidden <= 0 || latent > 16384 || hidden > 16384 ||
                latent % 256 || hidden % 256 || max_hits < 1 || max_hits > 16) {
            throw std::runtime_error("unsupported generic VRAM expert geometry");
        }
        mmvq = ggml_cuda_moe_stream_mmvq_dev;
        const auto supports = ggml_cuda_moe_stream_mmvq_supports;
        if (!supports(weight_type) || !supports(down_type)) {
            throw std::runtime_error(std::string("GPU expert layer: no CUDA matrix-vector kernel for ") +
                ggml_type_name(ggml_type(weight_type)) + "/" + ggml_type_name(ggml_type(down_type)) + " experts");
        }
        up_type = weight_type; down_type_ = down_type;
        up_stride = ggml_row_size(ggml_type(up_type), latent);
        down_stride = ggml_row_size(ggml_type(down_type_), hidden);
        q8_bytes = floating ? 0 : std::max(q8_1_bytes(latent), q8_1_bytes(hidden));
        try {
            cuda_check(cudaSetDevice(device), "cudaSetDevice");
            cuda_check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "cudaStreamCreate");
            cuda_check(cudaMalloc(reinterpret_cast<void **>(&device_input), size_t(max_hits) * latent * sizeof(float)), "cudaMalloc input");
            cuda_check(cudaMalloc(reinterpret_cast<void **>(&device_up_gate), size_t(2 * max_hits) * hidden * sizeof(float)), "cudaMalloc up/gate");
            cuda_check(cudaMalloc(reinterpret_cast<void **>(&device_hidden), size_t(max_hits) * hidden * sizeof(float)), "cudaMalloc hidden");
            cuda_check(cudaMalloc(reinterpret_cast<void **>(&device_down), size_t(max_hits) * latent * sizeof(float)), "cudaMalloc down");
            if (q8_bytes) { cuda_check(cudaMalloc(&device_q8, q8_bytes), "cudaMalloc Q8_1"); }
            cuda_check(cudaMallocHost(reinterpret_cast<void **>(&host_input), size_t(max_hits) * latent * sizeof(float)), "cudaMallocHost input");
            cuda_check(cudaMallocHost(reinterpret_cast<void **>(&host_down), size_t(max_hits) * latent * sizeof(float)), "cudaMallocHost down");
        } catch (...) {
            close();
            throw;
        }
        std::fprintf(stderr,
            "[kimi_vram_bridge] format=%s/%s latent=%d hidden=%d input=f32 hidden_quantizer=%s bitwise=0\n",
            ggml_type_name(ggml_type(up_type)), ggml_type_name(ggml_type(down_type_)), latent, hidden, floating ? "none_f32" : "cuda_q8_1");
    }

    void close() noexcept {
        if (closed) { return; }
        closed = true;
        const auto checked = [](cudaError_t rc, const char * operation) noexcept {
            if (rc != cudaSuccess) {
                std::fprintf(stderr, "fatal: generic VRAM engine cannot release ownership: %s: %s\n",
                    operation, cudaGetErrorString(rc));
                std::abort();
            }
        };
        if (stream || device_input || device_up_gate || device_hidden || device_down || device_q8 || host_input || host_down) {
            checked(cudaSetDevice(device), "cudaSetDevice during close");
        }
        if (stream) { checked(cudaStreamSynchronize(stream), "compute drain during close"); }
        if (host_down)  { checked(cudaFreeHost(host_down), "free host down"); host_down = nullptr; }
        if (host_input) { checked(cudaFreeHost(host_input), "free host input"); host_input = nullptr; }
        if (device_q8)      { checked(cudaFree(device_q8), "free Q8 scratch"); device_q8 = nullptr; }
        if (device_down)    { checked(cudaFree(device_down), "free device down"); device_down = nullptr; }
        if (device_hidden)  { checked(cudaFree(device_hidden), "free device hidden"); device_hidden = nullptr; }
        if (device_up_gate) { checked(cudaFree(device_up_gate), "free device up/gate"); device_up_gate = nullptr; }
        if (device_input)   { checked(cudaFree(device_input), "free device input"); device_input = nullptr; }
        if (stream) { checked(cudaStreamDestroy(stream), "destroy compute stream"); stream = nullptr; }

    }

    ~generic_engine() {
        std::fprintf(stderr, "[kimi_vram_generic_timing] calls=%llu hits=%llu total_us=%.0f\n",
            (unsigned long long) calls, (unsigned long long) hits, total_us);
        close();
    }

    size_t device_bytes() const {
        return (device_input ? size_t(max_hits) * latent * sizeof(float) : 0) +
            (device_up_gate ? size_t(2 * max_hits) * hidden * sizeof(float) : 0) +
            (device_hidden ? size_t(max_hits) * hidden * sizeof(float) : 0) +
            (device_down ? size_t(max_hits) * latent * sizeof(float) : 0) + (device_q8 ? q8_bytes : 0);
    }

    size_t host_bytes() const {
        return (host_input ? size_t(max_hits) * latent * sizeof(float) : 0) +
            (host_down ? size_t(max_hits) * latent * sizeof(float) : 0);
    }

    void check(const vram_compute_expert * experts, int count, int first, const void * input, size_t input_bytes,
            const kimi_vram_activation & activation) const {
        if (closed || poisoned) { throw std::runtime_error("generic VRAM engine is closed"); }
        if (!experts || !input || count < 1 || first < 0 || count > max_hits - first ||
                input_bytes != size_t(latent) * sizeof(float)) {
            throw std::runtime_error("generic VRAM engine input shape");
        }
        if (activation.kind == KIMI_VRAM_ACTIVATION_SWIGLU_CLAMP) {
            if (std::isnan(activation.swiglu_limit) || activation.swiglu_limit <= 1e-6f) {
                throw std::runtime_error("generic VRAM engine invalid SwiGLU clamp");
            }
        } else if (activation.kind == KIMI_VRAM_ACTIVATION_SITU) {
            if (!std::isfinite(activation.situ_beta) || activation.situ_beta <= 0.0f ||
                    !std::isfinite(activation.situ_linear_beta)) {
                throw std::runtime_error("generic VRAM engine invalid SiTU parameters");
            }
        } else if (activation.kind != KIMI_VRAM_ACTIVATION_SWIGLU) {
            throw std::runtime_error("generic VRAM engine activation kind");
        }
        for (int i = 0; i < count; ++i) {
            if (!experts[i].up || !experts[i].gate || !experts[i].down) {
                throw std::runtime_error("generic VRAM engine null resident weight");
            }
        }
    }

    // Queues a call on the stream over rows [first, first + count): its input, the chains, and its down rows' copy
    // into host_down.
    void enqueue(const vram_compute_expert * experts, int count, int first, const void * input, size_t input_bytes,
            const kimi_vram_activation & activation) {
        float * staged = host_input + size_t(first) * latent;
        float * row_input = device_input + size_t(first) * latent;
        std::memcpy(staged, input, input_bytes);
        cuda_check(cudaSetDevice(device), "cudaSetDevice");
        cuda_check(cudaMemcpyAsync(row_input, staged, input_bytes, cudaMemcpyHostToDevice, stream), "generic input H2D");
        for (int i = first; i < first + count; ++i) {
            const auto & e = experts[i - first];
            if (!mmvq(up_type, e.up, hidden, latent, up_stride,
                    row_input, device_q8, device_up_gate + size_t(2 * i) * hidden, stream) ||
                !mmvq(up_type, e.gate, hidden, latent, up_stride,
                    row_input, device_q8, device_up_gate + size_t(2 * i + 1) * hidden, stream)) {
                throw std::runtime_error("VRAM expert up/gate MMVQ launch failed");
            }
        }
        const int rows = count * hidden;
        generic_activate<<<(rows + 255) / 256, 256, 0, stream>>>(device_up_gate + size_t(2 * first) * hidden,
            device_hidden + size_t(first) * hidden, count, hidden, activation.kind, activation.situ_beta,
            activation.situ_linear_beta, activation.swiglu_limit);
        cuda_check(cudaGetLastError(), "generic activation launch");
        for (int i = first; i < first + count; ++i) {
            if (!mmvq(down_type_, experts[i - first].down, latent, hidden, down_stride,
                    device_hidden + size_t(i) * hidden, device_q8,
                    device_down + size_t(i) * latent, stream)) {
                throw std::runtime_error("VRAM expert down MMVQ launch failed");
            }
        }
        cuda_check(cudaMemcpyAsync(host_down + size_t(first) * latent, device_down + size_t(first) * latent,
            size_t(count) * latent * sizeof(float), cudaMemcpyDeviceToHost, stream), "generic output D2H");
        ++calls;
        hits += uint64_t(count);
    }

    void run(const vram_compute_expert * experts, int count, const void * input, size_t input_bytes,
            const kimi_vram_activation & activation, float * output) {
        check(experts, count, 0, input, input_bytes, activation);
        if (!output) { throw std::runtime_error("generic VRAM engine input shape"); }
        const auto start = clock_type::now();
        enqueue(experts, count, 0, input, input_bytes, activation);
        cuda_check(cudaStreamSynchronize(stream), "generic compute synchronize");
        std::memcpy(output, host_down, size_t(count) * latent * sizeof(float));
        total_us += std::chrono::duration<double, std::micro>(clock_type::now() - start).count();
    }

    void launch(const vram_compute_expert * experts, int count, int first, const void * input, size_t input_bytes,
            const kimi_vram_activation & activation, cudaEvent_t ready, cudaEvent_t start, cudaEvent_t end) {
        check(experts, count, first, input, input_bytes, activation);
        if (!start || !end) { throw std::runtime_error("generic VRAM engine launch without its events"); }
        cuda_check(cudaSetDevice(device), "cudaSetDevice");
        if (ready) { cuda_check(cudaStreamWaitEvent(stream, ready, 0), "generic launch wait"); }
        cuda_check(cudaEventRecord(start, stream), "generic launch start");
        enqueue(experts, count, first, input, input_bytes, activation);
        cuda_check(cudaEventRecord(end, stream), "generic launch end");
    }
};

void * create(int device, int latent, int hidden, int max_hits, int weight_type, int down_type,
        char * error, size_t error_size) {
    try {
        return new generic_engine(device, latent, hidden, max_hits, weight_type, down_type);
    } catch (const std::exception & e) {
        if (error && error_size) { std::snprintf(error, error_size, "%s", e.what()); }
        return nullptr;
    } catch (...) {
        if (error && error_size) { std::snprintf(error, error_size, "unexpected generic VRAM create failure"); }
        return nullptr;
    }
}

void destroy(void * context) {
    delete static_cast<generic_engine *>(context);
}

// A failed call leaves its stream and scratch unknown: the engine closes and refuses every later call.
template <class F> int guarded(void * context, char * error, size_t error_size, F && call) {
    try {
        if (!context) { throw std::runtime_error("null generic VRAM engine"); }
        call(*static_cast<generic_engine *>(context));
        return 0;
    } catch (const std::exception & e) {
        if (context) {
            auto * engine = static_cast<generic_engine *>(context);
            engine->poisoned = true;
            engine->close();
        }
        if (error && error_size) { std::snprintf(error, error_size, "%s", e.what()); }
        return 1;
    } catch (...) {
        if (context) {
            auto * engine = static_cast<generic_engine *>(context);
            engine->poisoned = true;
            engine->close();
        }
        if (error && error_size) { std::snprintf(error, error_size, "unexpected generic VRAM compute failure"); }
        return 1;
    }
}

int compute(void * context, const vram_compute_expert * experts, int hits,
        const void * input, size_t input_bytes, const kimi_vram_activation * activation,
        float * output, char * error, size_t error_size) {
    return guarded(context, error, error_size, [&](generic_engine & engine) {
        if (!activation) { throw std::runtime_error("null generic VRAM compute argument"); }
        engine.run(experts, hits, input, input_bytes, *activation, output);
    });
}

int launch(void * context, const vram_compute_expert * experts, int hits, int first,
        const void * input, size_t input_bytes, const kimi_vram_activation * activation,
        CUevent_st * ready, CUevent_st * start, CUevent_st * end, char * error, size_t error_size) {
    return guarded(context, error, error_size, [&](generic_engine & engine) {
        if (!activation) { throw std::runtime_error("null generic VRAM launch argument"); }
        engine.launch(experts, hits, first, input, input_bytes, *activation, ready, start, end);
    });
}

const float * host_rows(void * context) {
    return context ? static_cast<generic_engine *>(context)->host_down : nullptr;
}

size_t device_bytes(void * context) {
    return context ? static_cast<generic_engine *>(context)->device_bytes() : 0;
}

size_t host_bytes(void * context) {
    return context ? static_cast<generic_engine *>(context)->host_bytes() : 0;
}
} // namespace

extern "C" const vram_format_compute_api * kimi_vram_generic_format_compute_get_api() {
    static const vram_format_compute_api api = {
        4, sizeof(vram_format_compute_api), create, destroy, compute, device_bytes, host_bytes, launch, host_rows
    };
    return &api;
}
