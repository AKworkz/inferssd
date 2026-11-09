#pragma once

#include "cpu-sink.h"

#ifdef __cplusplus
extern "C" {
#endif

struct kimi_cpu_prefill;

struct kimi_cpu_prefill_stats {
    struct kimi_cpu_sink_stats stream; // experts/read bytes count the distinct expert union
    uint64_t routed_slots;
    uint64_t expert_token_pairs;       // duplicate slots for one token are computed only once
    uint64_t max_group_tokens;
    uint64_t workspace_bytes;          // excludes the shared sink's ring
    uint64_t gpu_experts;              // distinct retained VRAM hits
    uint64_t gpu_routes;               // token/slot routes computed on GPU
    uint64_t gpu_transient_experts;    // leased RAM Q2 experts uploaded to scratch
    uint64_t gpu_transient_routes;
    uint64_t gpu_transient_upload_bytes;
    double gpu_transient_upload_us;
    uint64_t gpu_ssd_transient_candidates; // direct-read Q2 experts passing the SSD scheduler gate
    uint64_t gpu_ssd_transient_experts;    // staged SSD experts computed through scratch
    uint64_t gpu_ssd_transient_routes;
    uint64_t gpu_ssd_transient_upload_bytes;
    double gpu_ssd_transient_upload_us;
    double gpu_worker_us;              // resident + transient GPU worker wall
    uint64_t gpu_cold_admit_candidates; // SSD candidates meeting the cold Q2 gate
    uint64_t gpu_cold_admit_experts;    // successfully seeded empty decode slots
    uint64_t gpu_cold_admit_routes;
    uint64_t gpu_cold_admit_upload_bytes;
    double gpu_cold_admit_upload_us;
    uint64_t ram_experts;              // distinct read-only retained RAM hits
    uint64_t ram_routes;               // token/slot routes sourced from retained RAM
    uint64_t ram_bytes;
    uint64_t ssd_experts;              // distinct direct-read misses
    uint64_t ssd_routes;               // token/slot routes sourced from direct reads
    uint64_t ssd_bytes;
};

// Borrow the decode sink's I/O ring and calls (kimi_cpu_sink_call_open): a run
// computes on the participants of the sink's open call. The sink must outlive
// this context. Do not concurrently destroy either context while a run is active.
GGML_BACKEND_API struct kimi_cpu_prefill * kimi_cpu_prefill_create(
        struct kimi_cpu_sink * sink, char * error, size_t error_size);
GGML_BACKEND_API void kimi_cpu_prefill_destroy(struct kimi_cpu_prefill * prefill);

// Token-major contiguous input/output [ntokens][latent] and routes
// [ntokens][nslots]. Token count follows the configured ubatch; 0..16 route slots. Tensor and
// activation requirements match kimi_cpu_sink_run_activation. Every distinct
// live expert is read once. Each token's routes are added in slot order from its
// first live slot's product, as the graph does, for every batch size; the route
// workspace (about ntokens * nslots * latent floats) is priced to the sink's
// host-memory ledger as it grows, and a run whose envelope cannot hold it fails
// before any read. Negative IDs are skipped. On failure discard output; no
// hidden mmap/GPU fallback is attempted and in-flight reads are drained.
GGML_BACKEND_API bool kimi_cpu_prefill_run_activation(
        struct kimi_cpu_prefill * prefill,
        const struct ggml_tensor * up, const struct ggml_tensor * gate,
        const struct ggml_tensor * down, const float * input,
        const int32_t * ids, const float * weights, int nslots, int ntokens,
        const struct kimi_cpu_sink_activation * activation, float * output,
        struct kimi_cpu_prefill_stats * stats, char * error, size_t error_size);

// Compatibility wrapper for Kimi-K3 SiTU callers.
GGML_BACKEND_API bool kimi_cpu_prefill_run(
        struct kimi_cpu_prefill * prefill,
        const struct ggml_tensor * up, const struct ggml_tensor * gate,
        const struct ggml_tensor * down, const float * input,
        const int32_t * ids, const float * weights, int nslots, int ntokens,
        float situ_beta, float situ_linear_beta, float * output,
        struct kimi_cpu_prefill_stats * stats, char * error, size_t error_size);

#ifdef __cplusplus
}
#endif
