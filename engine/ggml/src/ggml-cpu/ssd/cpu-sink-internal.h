#pragma once

#include "cpu-sink.h"
#include <cmath>
#include "ggml-ssd/telemetry/telemetry.h"

// Matches mul(weight) followed by ggml_rms_norm, which squares in float and sums in double, then route scaling.
inline void kimi_cpu_sink_normalize_expert(float * values, int rows, const float * norm, float eps) {
    double sum = 0.0;
    for (int row = 0; row < rows; ++row) {
        values[row] *= norm[row];
        sum += double(values[row] * values[row]);
    }
    const float scale = 1.0f / std::sqrt(float(sum / rows) + eps);
    for (int row = 0; row < rows; ++row) { values[row] *= scale; }
}


// Private executor hooks. The stream serializes all accesses to the shared
// decode ring and compute phases. Call parallel only from the stream consumer,
// and parallel_last only from the caller of a stream that returned, as the
// open call's last phase; callbacks must not recursively enter either executor
// or allocate/throw.
// Streaming is a prefill operation: it disables decode cache (normally releases
// it; optional cross-request retention checks the same memory guard) and
// always uses ring targets. Cache admission requires a subsequent decode begin.
using kimi_cpu_sink_parallel_fn = void (*)(void * userdata, int ith, int nth);
using kimi_cpu_sink_consumer_fn = void (*)(void * userdata, size_t expert,
        const void * up, const void * gate, const void * down);

struct kimi_cpu_sink_prefill_batch {
    const float * f32_input;
    size_t f32_stride;
    const int32_t * route_request_index;
    const float * weights;
    int nslots;
    int ntokens;
    struct kimi_cpu_sink_activation activation;
    float * output;
    uint64_t gpu_experts;
    uint64_t gpu_routes;
    uint64_t gpu_transient_experts;
    uint64_t gpu_transient_routes;
    uint64_t gpu_transient_upload_bytes;
    double gpu_transient_upload_us;
    uint64_t gpu_ssd_transient_candidates;
    uint64_t gpu_ssd_transient_experts;
    uint64_t gpu_ssd_transient_routes;
    uint64_t gpu_ssd_transient_upload_bytes;
    double gpu_ssd_transient_upload_us;
    double gpu_worker_us;
    double gpu_expert_us;    // the GPU's time in its expert chains
    double gpu_read_wait_us; // the GPU worker's wait for streamed SSD reads
    uint64_t gpu_cold_admit_candidates;
    uint64_t gpu_cold_admit_experts;
    uint64_t gpu_cold_admit_routes;
    uint64_t gpu_cold_admit_upload_bytes;
    double gpu_cold_admit_upload_us;
    uint64_t ram_experts;
    uint64_t ram_routes;
    uint64_t ram_bytes;
    uint64_t ssd_experts;
    uint64_t ssd_routes;
    uint64_t ssd_bytes;
};

void kimi_cpu_sink_parallel(struct kimi_cpu_sink * sink,
        kimi_cpu_sink_parallel_fn function, void * userdata);
void kimi_cpu_sink_parallel_last(struct kimi_cpu_sink * sink,
        kimi_cpu_sink_parallel_fn function, void * userdata);
bool kimi_cpu_sink_stream_hybrid(struct kimi_cpu_sink * sink,
        const struct ggml_tensor * up, const struct ggml_tensor * gate,
        const struct ggml_tensor * down, const int32_t * ids, int nexperts,
        struct kimi_cpu_sink_prefill_batch * batch,
        kimi_cpu_sink_consumer_fn consumer, void * userdata,
        struct kimi_cpu_sink_stats * stats, char * error, size_t error_size);

// Borrowed immutable selection; the prefill owner never outlives its sink.
const ssd::telemetry::debug_channels & kimi_cpu_sink_debug(const kimi_cpu_sink * sink) noexcept;
