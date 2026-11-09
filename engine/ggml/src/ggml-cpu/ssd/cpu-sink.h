#pragma once

#include "ggml-cpu.h"
#include "ggml-ssd/gpu/source.h"
#include "ggml-ssd/host/cache-stats.h"
#include "ggml-ssd/host/cache-stats.h"
#include "ggml-ssd/host/cache-policy.h"

#ifdef __cplusplus
extern "C" {
#endif

struct kimi_cpu_sink;

// Borrowed descriptors must remain open until run returns. No tensor payload is
// accessed: resolve the packed expert by its tensor name and expert index.


// The model-side native-range resolver checks this before using the callback ABI.
GGML_BACKEND_API size_t kimi_cpu_sink_source_size(void);

typedef bool (*kimi_cpu_sink_resolver)(
        void * userdata, const char * tensor_name, int32_t expert,
        struct kimi_cpu_sink_source * source);





// Sources are direct-read descriptors (platform::open_direct), read through the
// OS read ring (io/read-ring.h). The sink owns an I/O thread (and, with
// prefetch, its readers); it computes on the threads of its calls (below).
// A context serializes run calls. This unconfigured factory is for explicit
// CPU-only callers. Production uses ssd::cpu::create_configured with an owned
// typed snapshot and borrowed context services. Test policies are explicit
// constructor inputs; no environment configuration is read here.
GGML_BACKEND_API struct kimi_cpu_sink * kimi_cpu_sink_create(
        int ring_experts, kimi_cpu_sink_resolver resolver,
        void * userdata, char * error, size_t error_size);
// No call may be open.
GGML_BACKEND_API void kimi_cpu_sink_destroy(struct kimi_cpu_sink * sink);

// A call: nth compute participants for the runs in between, as a ggml graph's
// CPU threads enter a node. The coordinator (ith 0) opens it with nth, runs,
// and closes it; every other thread joins with its (ith, nth). Each compute
// phase of a run splits its rows over the participants, a decode phase over
// the first ones the decode thread choice takes. Without an open call the
// coordinator computes alone. Joining blocks until the call closes; closing
// waits until all nth - 1 have joined, so close on every path, errors
// included. A call may span several runs; one is open at a time. Open fails
// only for nth outside 1..512 or a call already open.
GGML_BACKEND_API bool kimi_cpu_sink_call_open(struct kimi_cpu_sink * sink, int nth, char * error, size_t error_size);
GGML_BACKEND_API void kimi_cpu_sink_call_close(struct kimi_cpu_sink * sink);
GGML_BACKEND_API void kimi_cpu_sink_call_join(struct kimi_cpu_sink * sink, int ith, int nth);
// The threads of a decode pool of `pool` that a decode graph runs on: all of them, less the CPUs the machine's other
// work took over the decode tokens so far, so that no thread of the graph waits behind that work at a barrier;
// `pool` where cpu.decode_participants fixes the count or the CPUs' busy time cannot be read. Call before the graph,
// with no run in progress.
GGML_BACKEND_API int kimi_cpu_sink_decode_threads(struct kimi_cpu_sink * sink, int pool);
// Next-layer expert prefetch: the experts that `layer` is predicted to select,
// best first, given before the decode run of the layer before it. Ignored when prefetch is off.
GGML_BACKEND_API void kimi_cpu_sink_prefetch_hint(struct kimi_cpu_sink * sink, int32_t layer, const int32_t * experts, int n);

// Configure while no run is in progress. Layer quotas are assigned in priority
// order, up to max_experts_per_layer each; a final layer may receive a partial
// quota. Layer IDs refer to canonical blk.N.ffn_{up,gate,down}_exps.weight names.
// Payload is an ordinary anonymous arena (platform/memory.h), never
// pinned/registered. Configuration does not touch payload pages. budget=0
// disables the cache. Reconfiguration releases the old arena first.
// max_experts_per_layer must be 1..896.
// Admission requires a readable host-ledger envelope (Linux: finite cgroup v2
// limits) and the context's host budget client. Only newly resident slots and
// conservative metadata/page-table costs are charged; the virtual payload
// budget is not prefaulted wholesale.
// Production fixes count_decay2/uniform retention with partial reclaim and
// reclaimable-file accounting. Explicit test policies exercise other supported
// algorithms through the testing factory. Unreadable resource guards reject
// admission; known shortages may discard low-score pages while retaining the
// mapping and policy heat. On Linux the caller must use swap0 and keep cgroup
// membership fixed until workers stop and resident mappings are released.
GGML_BACKEND_API bool kimi_cpu_sink_cache_configure(
        struct kimi_cpu_sink * sink, uint64_t budget_bytes,
        const int32_t * layer_priority, int n_layers, int max_experts_per_layer,
        char * error, size_t error_size);

// Call once before each logical token, and with decode=false before prefill or
// an unsupported batch. False disables cache use and by default unmaps pages;
// optional retention keeps weights only while headroom permits. A changed epoch
// (model/context identity), or nonconsecutive token without an intervening false,
// clears the mapping. Callers
// must serialize begin/configure/run as one sequence. Packs must be immutable
// during an epoch; replacing weights requires a new epoch even at the same fd.
GGML_BACKEND_API bool kimi_cpu_sink_cache_begin(
        struct kimi_cpu_sink * sink, uint64_t epoch, uint64_t token, bool decode,
        char * error, size_t error_size);
// A request's prefill boundary: cache_begin with decode=false, which also opens a
// new prefill phase when one is open. Repeated cache_begin(false) calls stay in an
// open phase, and a prompt evaluated outside any request leaves its phase open.
GGML_BACKEND_API bool kimi_cpu_sink_request_begin(
        struct kimi_cpu_sink * sink, uint64_t epoch, char * error, size_t error_size);
// Completes the phase transition after the scheduler reserve. decode=false
// only disarms the prefill OOM reclaimer; decode=true also grows the optional
// global VRAM tier. No-op when the elastic extension is unavailable or off.
GGML_BACKEND_API bool kimi_cpu_sink_vram_phase_complete(
        struct kimi_cpu_sink * sink, bool decode, char * error, size_t error_size);
// Reserve the GPU batch accumulator for this actual ubatch before streamed I/O begins.
GGML_BACKEND_API bool kimi_cpu_sink_vram_reserve_accumulator(
        struct kimi_cpu_sink * sink, int tokens, char * error, size_t error_size);
// At load, after the context reserved its worst-case graph, with the routed layers' expert tensors (`experts`: up,
// gate and down of each of n_layers layers): with a GPU tier, the sink keeps each layer's expert types, which it
// declares to the tier with every layer it gives it, once it has them for every configured layer; the layers must
// share latent, hidden and the expert count. Then on unified memory the GPU tier takes its storage for these routed
// experts (slots of the largest) and grows into what the host-memory ledger leaves, after the RAM cache's demand;
// elsewhere (with a GPU tier or without) a RAM cache kept across requests that can give room back
// (CACHE_PARTIAL_RECLAIM or CACHE_ELASTIC) takes its geometry from the largest layer's strides, its arena and the
// pages of the slots the ledger admits now. A cache that cannot give room back takes nothing here: it is sized at
// the first decode token, after the first prefill's ledger requests, which an arena made now would refuse.
// experts_used: the model's routed experts a token.
GGML_BACKEND_API bool kimi_cpu_sink_vram_reserve(struct kimi_cpu_sink * sink, const struct ggml_tensor * const * experts,
        int n_layers, int experts_used, char * error, size_t error_size);
GGML_BACKEND_API void kimi_cpu_sink_cache_get_stats(
        struct kimi_cpu_sink * sink, struct kimi_cpu_sink_cache_stats * stats);
// Room in host memory for other memory about to be allocated, decided by the host-memory ledger
// (kimi-host-ledger.h): `host_bytes` of host memory (host compute buffers, checkpoints, prompt-cache blobs) and
// `device_bytes` of GPU memory (device compute buffers), which count only where the GPU allocates out of the same
// DRAM. Both ledger terms, the envelope and the available memory less the floor, must still hold. On a unified-memory
// GPU the GPU tier gives blocks back first, then RAM cache slots go. Returns false when either would still be
// exceeded. With a null sink (before the first graph creates one) the ledger alone decides.
GGML_BACKEND_API bool kimi_cpu_sink_cache_make_room(struct kimi_cpu_sink * sink, uint64_t host_bytes, uint64_t device_bytes);
// Room for a prefill before it allocates anything: the ledger holds `pages` for the file pages of the weight rows its
// graph gathers by index (they come into the page cache while it runs) from now until the first decode token, beside
// the reserve and the floor, and makes room for its compute buffers (`host_bytes`, `device_bytes`, as
// kimi_cpu_sink_cache_make_room) beside them: the RAM cache gives back what both need at once, and the first decode
// token takes it back as far as the room allows then. A later call replaces the pages. Returns false when the room
// would still be short. With a null sink the ledger alone decides; the load step (kimi_cpu_sink_vram_reserve) leaves
// the pages held then uncommitted.
GGML_BACKEND_API bool kimi_cpu_sink_cache_prefill_room(struct kimi_cpu_sink * sink, uint64_t pages, uint64_t host_bytes,
        uint64_t device_bytes);
// Explicit reset/errors always release, even when cross-request retention is on.
GGML_BACKEND_API void kimi_cpu_sink_cache_reset(struct kimi_cpu_sink * sink);

// Single token. up/gate=[latent,hidden,experts], down=[hidden,latent,experts].
// Tensors must be contiguous. Kernels come from each tensor's CPU type traits:
// up and gate share one type (e.g. Q3_K with a Q8_K input, MXFP4 with a Q8_0
// input); down may use another. Column counts must be multiples of the weight
// and input block sizes. ids<0 are skipped; each live weight must be finite and
// nonnegative. Caller supplies the final normalized weights. Each expert's down
// output is multiplied by its weight, then added in slot order, starting from
// the first live slot's product as the graph's sum does. On failure output
// must be discarded. This is kimi_cpu_sink_run_activation with SITU.
GGML_BACKEND_API bool kimi_cpu_sink_run(
        struct kimi_cpu_sink * sink,
        const struct ggml_tensor * up, const struct ggml_tensor * gate,
        const struct ggml_tensor * down, const float * input,
        const int32_t * ids, const float * weights, int nslots,
        float situ_beta, float situ_linear_beta, float * output,
        struct kimi_cpu_sink_stats * stats, char * error, size_t error_size);

// Routed-expert activation between the up/gate and down projections.
// SITU (Kimi-K3): beta*tanh(g/beta)*sigmoid(g) * lb*tanh(u/lb), where lb <= 0
// keeps u linear; beta must be finite and positive and lb finite.
// SWIGLU_CLAMP (DeepSeek-V4.1): silu(min(g, L)) * clamp(u, -L, L), computed as
// the scalar GGML_GLU_OP_SWIGLU_CLAMP CPU kernel; L must be > 1e-6. An infinite
// L is SwiGLU in value but not in bits: the graph's SwiGLU kernel is vectorized.
// SWIGLU: silu(g) * u as GGML_GLU_OP_SWIGLU (ggml_swiglu_split) computes it, the
// vectorized CPU kernel over each whole hidden row; no parameter.
enum kimi_cpu_sink_activation_kind {
    KIMI_CPU_SINK_ACTIVATION_SITU         = 0,
    KIMI_CPU_SINK_ACTIVATION_SWIGLU_CLAMP = 1,
    KIMI_CPU_SINK_ACTIVATION_SWIGLU       = 2,
};

struct kimi_cpu_sink_activation {
    int32_t kind; // enum kimi_cpu_sink_activation_kind
    float   situ_beta;
    float   situ_linear_beta;
    float   swiglu_limit; // SWIGLU_CLAMP only
    const float * post_exp_norm; // optional latent-length weight, borrowed for run
    float post_exp_norm_eps;
};

// kimi_cpu_sink_run with an explicit activation. A configured GPU tier serves
// Q3_K SiTU experts through its original ABI and other expert formats only when
// it exports kimi_vram_format_get_api; otherwise the call fails closed.
GGML_BACKEND_API bool kimi_cpu_sink_run_activation(
        struct kimi_cpu_sink * sink,
        const struct ggml_tensor * up, const struct ggml_tensor * gate,
        const struct ggml_tensor * down, const float * input,
        const int32_t * ids, const float * weights, int nslots,
        const struct kimi_cpu_sink_activation * activation, float * output,
        struct kimi_cpu_sink_stats * stats, char * error, size_t error_size);

// Optional shared-location provider. nullptr detaches at a serialized boundary.
// Default absent; the placement owner installs its adapter once, not per token.
struct kimi_cache_location_api;
GGML_BACKEND_API bool kimi_cpu_sink_set_cache_locations(
        struct kimi_cpu_sink *, const struct kimi_cache_location_api *, void *, char *, size_t);

// Separate ABI: the original stats and run entry remain unchanged.
struct kimi_cache_compute_stats {
    struct kimi_cache_compute_plan plan;
    uint32_t computed_mask;
    uint32_t released_mask;
    uint64_t route_visible_us;
    uint64_t prepared_us;
    uint64_t finished_us;
    uint64_t baseline_miss_bytes;
    uint64_t submitted_bytes;
    uint64_t unused_read_bytes;
    // Native slot: queue, submit begin/end, up/gate/down CQE, CPU begin/end.
    // For registered VRAM jobs, the final pair is a batch submit-to-finish
    // host envelope, not per-expert CUDA kernel timing (see trace vram_mask).
    uint64_t events[16][8];
    uint64_t completed_bytes[16];
    float final_weights[16];
};
GGML_BACKEND_API bool kimi_cpu_sink_run_cache_compute(
        struct kimi_cpu_sink * sink,
        const struct ggml_tensor * up, const struct ggml_tensor * gate,
        const struct ggml_tensor * down, const float * input,
        const int32_t * ids, const float * weights, int nslots,
        const int32_t * native_ids, const float * native_weights, int native_count,
        float routed_scale, bool guarded, const struct kimi_cache_compute_config * config,
        float situ_beta, float situ_linear_beta, float * output,
        struct kimi_cpu_sink_stats * stats, struct kimi_cache_compute_stats * decision,
        char * error, size_t error_size);

#ifdef __cplusplus
}
#endif
