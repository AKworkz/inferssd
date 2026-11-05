#pragma once
#include <stddef.h>
#include <stdint.h>
#include "ggml-ssd/gpu/kimi-vram-tier-api.h"
struct vram_compute_expert { const void * up; const void * gate; const void * down; };
// Decode engine of the tier (generic-compute-adapter.cu): an f32 input row, up and gate, the activation and down over
// up to max_hits experts of one format on the CUDA backend's matrix-vector kernels; returns unweighted down rows.
struct vram_format_compute_api {
    uint32_t version, size; // 3
    void * (*create)(int device, int latent, int hidden, int max_hits, int weight_type, int down_type, char *, size_t);
    void (*destroy)(void *);
    int (*compute)(void *, const vram_compute_expert *, int hits, const void * input, size_t input_bytes,
                   const kimi_vram_activation * activation, float * host_down, char * error, size_t error_size);
    size_t (*device_scratch_bytes)(void *);
    size_t (*host_scratch_bytes)(void *);
};
typedef const vram_format_compute_api * (*vram_format_compute_get_api_fn)();
// Engine of grouped prefill for any expert format the CUDA backend multiplies (group-compute-adapter.cu). Per layer, the
// input rows of the tokens with GPU work go to the device once; each expert's chain (up, gate, the activation, down, the
// optional per-expert post norm and the routing weight) runs over its tokens in chunks on the engine's stream and adds
// into a device accumulator, which comes back once. Weights are read where they are: a tier slot, a transient device
// slot, or host memory on a GPU that reads it in place. Nothing synchronizes per expert: each expert's chain is timed by
// a pair of CUDA events. reserve allocates every buffer a pass needs, the products' own reservations included, and
// loads every kernel a pass runs, so no device memory is taken between it and release, which gives the buffers back
// (before decode grows the tier). The kernels stay loaded for the process: the first reserve of a type and shape takes
// their device memory for good, and meanwhile a scratch of one matrix and its operands, which it frees.
struct CUevent_st;
struct vram_group_activation {
    struct kimi_vram_activation base;
    const float * post_norm;    // latent weights of the per-expert post norm, or null; copied by begin_layer
    float post_norm_eps;
};
// One route of a layer: the input row of its token and its routing weight. A group's routes are one expert's; their
// rows ascend strictly (duplicate routes of a token are merged, their weights summed).
struct vram_group_route { int32_t row; float weight; };
struct vram_group_geometry {
    uint64_t route_bytes;     // per route: the route list, host to device and in device memory
    uint64_t token_bytes;     // per input row: host to device
    uint64_t output_bytes;    // per input row: device to host (the accumulated rows)
    uint64_t fixed_bytes;     // device memory, as reserved, that does not grow with the pass: the chunk scratch and the
                              // products' own reservations (the stream-k fixup; a GEMM's dequantize scratch)
    uint64_t bytes_per_token; // device memory per reserved row: its input and its accumulator
};
// One layer, from CUDA events. experts..route_us are the expert chains' sums (routes = the expert's tokens: count,
// sum of routes, of routes squared, of the times, and of routes times the time), for a least-squares line.
struct vram_group_layer_stats {
    double experts, routes, routes_sq, us, route_us;
    double begin_us;            // the layer's uploads (input rows, route list, post norm) and the accumulator's zeroing
    double finish_us;           // the accumulated rows' download
    uint64_t h2d_bytes, d2h_bytes;
    double span_us;             // version 2: from the first chain's start to the last one's end (the chains and
                                // whatever the stream waited for between them: the copies they were ready after)
};
struct vram_group_compute_api {
    uint32_t version, size;   // 3
    // An engine for `formats` expert formats of one geometry: format f has up and gate of weight_types[f] and down of
    // down_types[f]. A reserve covers every one of them; each layer names its format at begin_layer.
    void * (*create)(int device, int latent, int hidden, int formats, const int32_t * weight_types,
                     const int32_t * down_types, char * error, size_t error_size);
    void (*destroy)(void *);
    // Buffers for passes of up to `rows` input rows and `routes` routes a layer (device and page-locked host memory,
    // the products' own reservations). Grows only; release frees all of it. A pass reserves once, before its first
    // layer, for its tokens and tokens x slots routes (no layer has more rows with GPU work); a later reserve that grows
    // drains the engine and reallocates, which moves input_rows().
    int (*reserve)(void *, int rows, int routes, char * error, size_t error_size);
    int (*release)(void *, char * error, size_t error_size);
    // Page-locked rows [reserved rows][latent] that the caller fills with the layer's input rows before begin_layer.
    // Valid until a reserve that grows, or release.
    float * (*input_rows)(void *);
    // A layer of format `format` (an index of create's), `rows` input rows and `groups` experts: group g's routes are
    // routes[first[g] .. first[g + 1]). Queues the uploads and the zeroing; copies every argument before it returns.
    int (*begin_layer)(void *, int format, int rows, const struct vram_group_route * routes, const int32_t * first,
                       int groups, const struct vram_group_activation * activation, char * error, size_t error_size);
    // Queues group g's chain with these weights. The chain waits for `ready` (a weights upload, or null) on the device;
    // *done (optional) receives an event that completes when the chain no longer reads the weights, valid until the
    // next begin_layer.
    int (*compute)(void *, int group, const struct vram_compute_expert * weights, struct CUevent_st * ready,
                   struct CUevent_st ** done, char * error, size_t error_size);
    // Waits for the layer and returns its accumulated rows [rows][latent] (page-locked, valid until the next
    // begin_layer) and the layer's stats; group_us (optional, one per group) each chain's time. Every group of the
    // layer must have been computed.
    int (*finish_layer)(void *, const float ** output, struct vram_group_layer_stats * stats, double * group_us,
                        char * error, size_t error_size);
    void (*geometry)(void *, struct vram_group_geometry * geometry);
    size_t (*device_bytes)(void *);
    size_t (*host_bytes)(void *);
};
typedef const vram_group_compute_api * (*vram_group_compute_get_api_fn)();

extern "C" {
const vram_format_compute_api * kimi_vram_generic_format_compute_get_api();
const vram_group_compute_api * kimi_vram_group_compute_get_api();
}
