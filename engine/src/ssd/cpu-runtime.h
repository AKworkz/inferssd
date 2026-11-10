#pragma once

#include "ggml-cpu/ssd/cpu-sink.h"
#include "cpu-prefill.h"
#include "ggml-ssd/settings/settings.h"
#include "ggml-ssd/telemetry/telemetry.h"
#include "ggml-ssd/store/source.h"

#include <cerrno>
#include <algorithm>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <map>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>


// Only metadata is retained here. Expert tensors are deliberately absent from
// the custom op's graph sources, so the scheduler cannot copy their payloads.
struct llama_kimi_cpu_sink_runtime;

struct llama_kimi_cpu_sink_node {
    ggml_tensor up;
    ggml_tensor gate;
    ggml_tensor down;
    ggml_backend_reg_t cpu_reg;
    llama_kimi_cpu_sink_runtime * runtime;
    float situ_beta;
    float situ_linear_beta;
    // SITU nodes keep the K3 entries (run, grouped prefill).
    // Other activations are decode-only through kimi_cpu_sink_run_activation.
    kimi_cpu_sink_activation activation = { KIMI_CPU_SINK_ACTIVATION_SITU, 0.0f, 0.0f, 0.0f };
    int layer;
    bool post_exp_norm = false;
    float post_exp_norm_eps = 0.0f;
    bool prefill = false;
};

struct llama_kimi_cpu_sink_runtime {
    // Resolve the logical native range; the callback below aligns its physical I/O.
    // resolve_raw_direct_shape(name, expert, logical bytes, n_experts, fd, logical offset, logical bytes)
    const ssd::settings::snapshot & settings;
    const ssd::runtime_services & services;
    const ssd::store::source_view source;
    std::mutex mutex;
    kimi_cpu_sink * sink = nullptr;
    llama_kimi_cpu_prefill_runtime prefill;
    decltype(&kimi_cpu_sink_run) run = nullptr;
    decltype(&kimi_cpu_sink_run_activation) run_activation = nullptr;
    decltype(&kimi_cpu_sink_destroy) destroy = nullptr;
    decltype(&kimi_cpu_sink_cache_configure) cache_configure = nullptr;
    decltype(&kimi_cpu_sink_cache_begin) cache_begin = nullptr;
    decltype(&kimi_cpu_sink_request_begin) request_begin = nullptr;
    decltype(&kimi_cpu_sink_vram_phase_complete) vram_phase_complete = nullptr;
    decltype(&kimi_cpu_sink_cache_get_stats) cache_get_stats = nullptr;
    decltype(&ssd::cpu::cache_make_room_configured) cache_make_room = nullptr; // optional
    decltype(&ssd::cpu::cache_prefill_room_configured) cache_prefill_room = nullptr; // optional
    ggml_backend_reg_t cpu_reg = nullptr;
    const ssd::cpu::api_v2 * cpu_api = nullptr;
    const llama_kimi_cpu_sink_node * active_node = nullptr;
    uint64_t calls = 0;
    uint64_t read_requests = 0;
    uint64_t read_bytes = 0;
    uint64_t read_errors = 0;
    uint64_t fallback_count = 0;
    double total_us = 0.0;
    const uint64_t epoch;
    uint64_t cache_budget = 0;
    uint64_t vram_budget = 0;
    int cache_max_per_layer = 8;
    std::vector<int32_t> layer_priority;
    // Expert tensor shapes of every layer, so that the sink's next-layer prefetch can resolve the next layer's
    // experts while the current layer's node is active: name -> (bytes per expert, experts).
    std::map<std::string, std::pair<size_t, int64_t>> expert_shapes;
    decltype(&kimi_cpu_sink_prefetch_hint) prefetch_hint = nullptr; // optional
    bool sequence_known = false;
    llama_seq_id sequence = 0;
    bool cache_decode_active = false;
    // A prefill phase is open from a prefill begin() until phase_complete(true) or a decode begin(); the graph
    // reserves inside it repeat the request-boundary transition and are skipped. begin_request() opens a new one.
    bool prefill_open = false;
    bool cache_keep_across_requests = false;
    // The file pages the weight rows of the last prefill graph reserved gather (prefill_room): the load step leaves
    // them uncommitted for the first prefill.
    uint64_t prefill_pages = 0;

    llama_kimi_cpu_sink_runtime(int n_layers, uint64_t context_epoch,
            int64_t n_expert, const std::vector<int32_t> & routed_layers,
            const ssd::settings::snapshot & configured, const ssd::runtime_services & runtime_services,
            ssd::store::source_view source_view)
        : settings(configured), services(runtime_services), source(source_view), epoch(context_epoch) {
        if (n_expert <= 0) { throw std::runtime_error("CPU sink requires a positive routed expert count"); }
        // Supported v0.1 profile: resource-sized RAM cache, model-order priority,
        // fixed eight-expert read ring.
        cache_budget = std::numeric_limits<uint64_t>::max();
        cache_keep_across_requests = true;
        vram_budget = settings.vram_budget()->tier_bytes(services.gpu != nullptr);
        cache_max_per_layer = int(std::min<int64_t>(n_expert, 896));
        for (int32_t layer : routed_layers) {
            if (layer >= 0 && layer < n_layers &&
                    std::find(layer_priority.begin(), layer_priority.end(), layer) == layer_priority.end()) {
                layer_priority.push_back(layer);
            }
        }
        if (layer_priority.empty()) { throw std::runtime_error("CPU sink requires routed model layers"); }
    }

    ~llama_kimi_cpu_sink_runtime() {
        prefill.reset();
        if (sink) {
            std::fprintf(stderr,
                "[kimi_cpu_sink_total] calls=%" PRIu64 " read_requests=%" PRIu64
                " read_bytes=%" PRIu64 " total_us=%.0f read_errors=%" PRIu64
                " fallback_count=%" PRIu64 "\n",
                calls, read_requests, read_bytes, total_us, read_errors, fallback_count);
            print_cache_stats("before_destroy", cache_decode_active, 0);
            destroy(sink);
        }
    }

    static bool resolve_source(void * userdata, const char * tensor_name, int32_t expert,
            kimi_cpu_sink_source * source) {
        auto & runtime = *static_cast<llama_kimi_cpu_sink_runtime *>(userdata);
        if (!runtime.active_node || !runtime.source.resolve || !source) {
            return false;
        }
        const auto & node = *runtime.active_node;
        auto range = [&](size_t logical_nbytes, int64_t experts) {
            *source = {};
            if (!runtime.source.resolve(runtime.source.userdata, tensor_name, expert, logical_nbytes, experts,
                    &source->fd_direct, &source->offset, &source->nbytes)) { return false; }
            source->data_offset = source->offset % 4096;
            source->payload_nbytes = logical_nbytes;
            if (logical_nbytes > SIZE_MAX - source->data_offset - 4095) { return false; }
            source->offset -= source->data_offset;
            source->nbytes = (source->data_offset + logical_nbytes + 4095) / 4096 * 4096;
            return true;
        };
        for (const ggml_tensor * tensor : { &node.up, &node.gate, &node.down }) {
            if (std::strcmp(tensor_name, tensor->name) == 0) {
                return range(tensor->nb[2], tensor->ne[2]);
            }
        }
        const auto shape = runtime.expert_shapes.find(tensor_name);
        if (shape != runtime.expert_shapes.end()) {
            return range(shape->second.first, shape->second.second);
        }
        return false;
    }

    void bind_provider(ggml_backend_reg_t registry) {
        if (!registry) { throw std::runtime_error("CPU sink requires a loaded CPU backend"); }
        if (cpu_api) {
            if (registry != cpu_reg) { throw std::runtime_error("CPU sink backend changed while borrowed"); }
            return;
        }
        const auto get = reinterpret_cast<ssd::cpu::get_api_v2>(
            ggml_backend_reg_get_proc_address(registry, "ggml_backend_cpu_ssd_api_v2"));
        const auto * api = get ? get() : nullptr;
        if (!ssd::cpu::valid(api) || api->source_size() != sizeof(kimi_cpu_sink_source)) {
            throw std::runtime_error("CPU sink executor ABI mismatch; rebuild the CPU backend and model together");
        }
        cpu_api = api;
        cpu_reg = registry;
        cache_make_room = api->cache_make_room;
        cache_prefill_room = api->cache_prefill_room;
    }

    void bind_provider_before_create() {
        if (cpu_api) { return; }
        const auto registry = ggml_backend_reg_by_name("CPU");
        if (!registry) {
            throw std::runtime_error("CPU sink requires a loaded CPU backend for host memory admission");
        }
        bind_provider(registry);
    }

    void initialize(ggml_backend_reg_t registry) {
        if (sink) {
            GGML_ASSERT(registry == cpu_reg);
            return;
        }
        bind_provider(registry);
        const auto create = cpu_api->create;
        run = cpu_api->run;
        run_activation = cpu_api->run_activation;
        destroy = cpu_api->destroy;
        cache_configure = cpu_api->cache_configure;
        cache_begin = cpu_api->cache_begin;
        request_begin = cpu_api->request_begin;
        vram_phase_complete = cpu_api->vram_phase_complete;
        cache_get_stats = cpu_api->cache_get_stats;
        prefetch_hint = cpu_api->prefetch_hint;
        constexpr int ring_experts = 8;
        char error[512] = {};
        sink = create(&settings, &services, resolve_source, this, error, sizeof(error));
        if (!sink) {
            throw std::runtime_error(std::string("CPU sink initialization failed: ") + error + "; no fallback");
        }
        if (!cache_configure(sink, cache_budget, layer_priority.data(),
                static_cast<int>(layer_priority.size()), cache_max_per_layer, error, sizeof(error))) {
            throw std::runtime_error(std::string("CPU sink cache configuration failed: ") + error + "; no fallback");
        }
        std::fprintf(stderr,
            "[kimi_cpu_sink] enabled ring_experts=%d n_tokens=1 expert_h2d_bytes=0"
            " cache_budget=%" PRIu64 " cache_max_experts_per_layer=%d epoch=%" PRIu64
            " cache_admission=direct_read fixed_scope=ring_only\n",
            ring_experts, cache_budget, cache_max_per_layer, epoch);
    }

    // Before a graph with sink nodes runs: its CPU threads join this executor's calls
    // (llama_kimi_cpu_sink_compute), so it exists before the first of them.
    void prepare_graph(ggml_backend_reg_t registry) {
        std::lock_guard<std::mutex> lock(mutex);
        initialize(registry);
    }

    // The threads of the decode pool (`pool`) a decode graph with sink nodes runs on, after prepare_graph: the pool
    // less the CPUs the machine's other work takes (kimi_cpu_sink_decode_threads).
    int decode_threads(int pool) {
        std::lock_guard<std::mutex> lock(mutex);
        return cpu_api->decode_threads(sink, pool);
    }

    void print_cache_stats(const char * reason, bool decode, uint64_t token) {
        if (!sink || !cache_budget) { return; }
        kimi_cpu_sink_cache_stats stats = {};
        cache_get_stats(sink, &stats);
        std::fprintf(stderr,
            "[kimi_cpu_sink_cache] reason=%s epoch=%" PRIu64 " token=%" PRIu64 " decode=%d"
            " budget_bytes=%" PRIu64 " capacity_bytes=%" PRIu64 " mapped_bytes=%" PRIu64
            " committed_bytes=%" PRIu64 " enabled_bytes=%" PRIu64 " valid_bytes=%" PRIu64 " peak_bytes=%" PRIu64
            " hits=%" PRIu64 " misses=%" PRIu64 " admitted_bytes=%" PRIu64 " evictions=%" PRIu64
            " admission_skips=%" PRIu64 " clears=%" PRIu64 " pressure_stops=%" PRIu64 " copy_us=%.0f"
            " page_table_estimate_bytes=%" PRIu64 " last_guard_extra_bytes=%" PRIu64
            " peak_guard_extra_bytes=%" PRIu64
            " shrink_ops=%" PRIu64 " phase_grows=%" PRIu64
            " released_bytes=%" PRIu64 " refaulted_bytes=%" PRIu64
            " guard_reclaims=%" PRIu64 " shrink_us=%" PRIu64
            " prefill_hits=%" PRIu64 " prefill_misses=%" PRIu64
            " prefill_hit_bytes=%" PRIu64 " prefill_leases=%" PRIu64 "\n",
            reason, epoch, token, decode ? 1 : 0, stats.budget_bytes, stats.capacity_bytes,
            stats.mapped_bytes, stats.committed_bytes, stats.enabled_bytes, stats.valid_bytes, stats.peak_bytes,
            stats.hits, stats.misses, stats.admitted_bytes, stats.evictions,
            stats.admission_skips, stats.clears, stats.pressure_stops, stats.copy_us,
            stats.page_table_bytes, stats.last_guard_extra_bytes, stats.peak_guard_extra_bytes,
            stats.shrink_ops, stats.phase_grows, stats.released_bytes,
            stats.refaulted_bytes, stats.guard_reclaims, stats.shrink_us,
            stats.prefill_hits, stats.prefill_misses, stats.prefill_hit_bytes, stats.prefill_leases);
    }

    void begin(ggml_backend_reg_t registry, uint64_t token, bool decode, llama_seq_id seq = 0) {
        std::lock_guard<std::mutex> lock(mutex);
        transition(registry, token, decode, seq, false);
    }
    // A request's prefill boundary (llama_ssd_prefill_phase): a new prefill phase, also when one is open. A prompt
    // evaluated outside any request leaves its phase open, since no decode token follows it: the server evaluates two
    // tokens at load to probe sequence removal, and its first request would otherwise run inside that phase.
    void begin_request() {
        std::lock_guard<std::mutex> lock(mutex);
        transition(nullptr, 0, false, 0, true);
    }
    // Requires mutex.
    void transition(ggml_backend_reg_t registry, uint64_t token, bool decode, llama_seq_id seq, bool request) {
        const bool observe = services.debug && (services.debug->trace || services.debug->profile);
        if (!cache_budget && !vram_budget && !observe) { return; }
        if (decode) { initialize(registry); }
        if (!sink || (!decode && prefill_open && !request)) { return; }
        char error[512] = {};
        if (decode && sequence_known && sequence != seq) {
            if (!cache_begin(sink, epoch, token, false, error, sizeof(error))) {
                GGML_ABORT("CPU sink cache sequence reset failed: %s; no fallback", error);
            }
        }
        const bool begun = request ? request_begin(sink, epoch, error, sizeof(error)) :
            cache_begin(sink, epoch, token, decode, error, sizeof(error));
        if (!begun) {
            GGML_ABORT("CPU sink cache phase/token transition failed: %s; no fallback", error);
        }
        sequence_known = decode;
        sequence = seq;
        cache_decode_active = decode;
        prefill_open = !decode;
        if (!decode) {
            kimi_cpu_sink_cache_stats stats = {};
            cache_get_stats(sink, &stats);
            if (!cache_keep_across_requests && (stats.mapped_bytes || stats.committed_bytes || stats.valid_bytes)) {
                GGML_ABORT("CPU sink cache retained payload at prefill/unsupported batch boundary");
            }
            print_cache_stats(stats.mapped_bytes ? "retain" : "release", false, token);
        } else {
            if (services.debug && services.debug->trace) { print_cache_stats("token_begin", true, token); }
        }
    }

    // At load, after the context reserved its worst-case graph: the sink, and on unified memory the GPU tier's
    // storage for these routed experts, so that no request waits for its page population (the tier decides; a
    // discrete GPU keeps its arena after the first prefill). Off unified memory the RAM cache takes its pages here
    // instead, with a GPU tier or without one, when it can give room back to the first prefill's ledger requests
    // (CACHE_PARTIAL_RECLAIM or CACHE_ELASTIC; the sink decides). The sink exists from here on, so the first request's
    // prefill goes through cache_begin and make_room as every later one does; a cache that cannot give room back is
    // still sized at the first decode token. A failure leaves both to the first decode. experts: up, gate and down of
    // every routed layer, whose expert types the sink keeps for the GPU tier.
    void reserve_tier(ggml_backend_reg_t registry, const std::vector<const ggml_tensor *> & experts, int experts_used) {
        std::lock_guard<std::mutex> lock(mutex);
        if ((!vram_budget && !cache_budget) || !registry || experts.size() < 3) { return; }
        initialize(registry);
        const auto reserve = cpu_api->vram_reserve;
        // the load-time graph's pages, held from here to the first decode token: the load step leaves their room
        if (prefill_pages && cache_prefill_room) { cache_prefill_room(sink, &settings, &services, prefill_pages, 0, 0); }
        char error[512] = {};
        if (reserve && !reserve(sink, experts.data(), int(experts.size() / 3), experts_used, error, sizeof(error))) {
            std::fprintf(stderr, "[kimi_vram_elastic] event=reserve status=fail error=%s\n", error);
        }
    }

    // Room for `host_bytes` of host memory and `device_bytes` of GPU memory the caller is about to allocate, from
    // the CPU backend's host-memory ledger, which counts device memory only where the GPU shares the DRAM: the
    // expert caches give back what it needs (on unified memory the GPU tier first). Before the first graph creates
    // the sink the ledger alone decides.
    bool make_room(uint64_t host_bytes, uint64_t device_bytes = 0) {
        std::lock_guard<std::mutex> lock(mutex);
        bind_provider_before_create();
        return cache_make_room(sink, &settings, &services, host_bytes, device_bytes);
    }

    ssd::result<ssd::host::allocation_ticket> take_room(uint64_t host_bytes,
            uint64_t device_bytes = 0, uint64_t pages = UINT64_MAX) {
        std::lock_guard<std::mutex> lock(mutex);
        bind_provider_before_create();
        if (pages != UINT64_MAX) { prefill_pages = pages; }
        return cpu_api->take_room(sink, &settings, &services, host_bytes, device_bytes, pages);
    }

    // Room for a prefill graph about to be allocated (kimi_cpu_sink_cache_prefill_room): its compute buffers and the
    // file pages of the weight rows it gathers, which the ledger holds until the first decode token.
    bool prefill_room(uint64_t pages, uint64_t host_bytes, uint64_t device_bytes) {
        std::lock_guard<std::mutex> lock(mutex);
        prefill_pages = pages;
        bind_provider_before_create();
        return cache_prefill_room(sink, &settings, &services, pages, host_bytes, device_bytes);
    }

    bool phase_complete(bool decode) {
        std::lock_guard<std::mutex> lock(mutex);
        if (decode) { prefill_open = false; }
        if (!sink) { return true; }
        char error[512] = {};
        if (!vram_phase_complete(sink, decode, error, sizeof(error))) {
            std::fprintf(stderr, "[kimi_vram_elastic] phase completion failed: %s\n", error);
            return false;
        }
        return true;
    }
};

// One call of the executor per node: the CPU backend's ggml threads are its
// compute participants. ith 0 opens it and runs the layer; the call closes on
// every exit, after all nth - 1 others joined.
struct llama_kimi_cpu_sink_call {
    const llama_kimi_cpu_sink_runtime & runtime;
    llama_kimi_cpu_sink_call(const llama_kimi_cpu_sink_runtime & owner, int nth) : runtime(owner) {
        char error[256] = {};
        if (!runtime.cpu_api->call_open(runtime.sink, nth, error, sizeof(error))) {
            GGML_ABORT("CPU sink call failed to open: %s", error);
        }
    }
    ~llama_kimi_cpu_sink_call() { runtime.cpu_api->call_close(runtime.sink); }
    llama_kimi_cpu_sink_call(const llama_kimi_cpu_sink_call &) = delete;
    llama_kimi_cpu_sink_call & operator=(const llama_kimi_cpu_sink_call &) = delete;
};

inline void llama_kimi_cpu_sink_compute(ggml_tensor * dst, int ith, int nth, void * userdata) {
    // Every CPU-backend ggml thread enters (n_tasks = GGML_N_TASKS_MAX). No C++
    // exception may escape this callback.
    GGML_ASSERT(ith >= 0 && ith < nth);
    auto & node = *static_cast<llama_kimi_cpu_sink_node *>(userdata);
    GGML_ASSERT(node.runtime);
    auto & runtime = *node.runtime;
    // The graph prepared the executor before it started (prepare_graph).
    GGML_ASSERT(runtime.sink && runtime.cpu_api);
    if (ith != 0) {
        runtime.cpu_api->call_join(runtime.sink, ith, nth);
        return;
    }
    const llama_kimi_cpu_sink_call call(runtime, nth);
    std::lock_guard<std::mutex> lock(runtime.mutex);
    runtime.active_node = &node;

    const ggml_tensor * input = dst->src[0];
    const ggml_tensor * ids = dst->src[1];
    const ggml_tensor * weights = dst->src[2];
    GGML_ASSERT(input->type == GGML_TYPE_F32 && ggml_is_contiguous(input));
    GGML_ASSERT(ids->type == GGML_TYPE_I32 && ggml_is_contiguous(ids));
    GGML_ASSERT(weights->type == GGML_TYPE_F32 && ggml_is_contiguous(weights));
    GGML_ASSERT(ggml_nelements(ids) == ggml_nelements(weights));
    const int ntokens = static_cast<int>(dst->ne[1]);
    GGML_ASSERT(ntokens >= 1);
    GGML_ASSERT(ggml_nelements(input) == node.up.ne[0] * ntokens);
    GGML_ASSERT(ggml_nelements(ids) % ntokens == 0);
    const int nslots = static_cast<int>(ggml_nelements(ids) / ntokens);

    kimi_cpu_sink_stats stats = {};
    kimi_cpu_prefill_stats grouped_stats = {};
    char error[512] = {};
    bool ok;
    kimi_cpu_sink_activation activation = node.activation;
    if (node.post_exp_norm) {
        const ggml_tensor * norm = dst->src[3];
        GGML_ASSERT(norm && norm->type == GGML_TYPE_F32 && ggml_is_contiguous(norm));
        GGML_ASSERT(ggml_nelements(norm) == node.down.ne[1]);
        activation.post_exp_norm = static_cast<const float *>(norm->data);
        activation.post_exp_norm_eps = node.post_exp_norm_eps;
    }
    const bool situ = node.activation.kind == KIMI_CPU_SINK_ACTIVATION_SITU;
    if (node.prefill) {
        runtime.prefill.initialize(node.cpu_reg, runtime.sink);
        ok = runtime.prefill.run_activation(runtime.prefill.context, &node.up, &node.gate, &node.down,
            static_cast<const float *>(input->data), static_cast<const int32_t *>(ids->data),
            static_cast<const float *>(weights->data), nslots, ntokens,
            &activation, static_cast<float *>(dst->data),
            &grouped_stats, error, sizeof(error));
        stats = grouped_stats.stream;
    } else if (situ) {
        GGML_ASSERT(ntokens == 1);
        ok = runtime.run(runtime.sink, &node.up, &node.gate, &node.down,
            static_cast<const float *>(input->data), static_cast<const int32_t *>(ids->data),
            static_cast<const float *>(weights->data), nslots,
            node.situ_beta, node.situ_linear_beta, static_cast<float *>(dst->data),
            &stats, error, sizeof(error));
    } else {
        GGML_ASSERT(ntokens == 1);
        // the next layer's predicted experts, for the sink's prefetch
        if (const ggml_tensor * hint = dst->src[4]; hint && runtime.prefetch_hint) {
            GGML_ASSERT(hint->type == GGML_TYPE_I32 && ggml_is_contiguous(hint));
            runtime.prefetch_hint(runtime.sink, node.layer + 1, static_cast<const int32_t *>(hint->data),
                static_cast<int>(ggml_nelements(hint)));
        }
        ok = runtime.run_activation(runtime.sink, &node.up, &node.gate, &node.down,
            static_cast<const float *>(input->data), static_cast<const int32_t *>(ids->data),
            static_cast<const float *>(weights->data), nslots, &activation,
            static_cast<float *>(dst->data), &stats, error, sizeof(error));
    }
    runtime.active_node = nullptr;
    if (!ok) {
        GGML_ABORT("CPU sink layer=%d failed: %s read_errors=%" PRIu64
            " fallback_count=%" PRIu64 "; output discarded", node.layer, error,
            stats.read_errors, stats.fallback_count);
    }
    runtime.calls++;
    runtime.read_requests += stats.read_requests;
    runtime.read_bytes += stats.read_bytes;
    runtime.read_errors += stats.read_errors;
    runtime.fallback_count += stats.fallback_count;
    runtime.total_us += stats.total_us;
    if (node.prefill && runtime.services.events) {
        const auto published = runtime.services.events->event("ssd.prefill.layer",
            "{\"layer\":" + std::to_string(node.layer) + ",\"n_tokens\":" + std::to_string(ntokens) +
            ",\"read_bytes\":" + std::to_string(grouped_stats.ssd_bytes) + "}");
        if (!published) { GGML_ABORT("prefill telemetry failed after layer drain: %s", published.error.c_str()); }
    }
    if (runtime.calls == 1 || (runtime.services.debug && runtime.services.debug->trace)) {
        if (node.prefill) {
            std::fprintf(stderr,
                "[kimi_cpu_prefill_layer] layer=%d n_tokens=%d slots=%d unique_experts=%" PRIu64
                " routed_slots=%" PRIu64 " expert_token_pairs=%" PRIu64 " max_group_tokens=%" PRIu64
                " workspace_bytes=%" PRIu64 " gpu_experts=%" PRIu64 " gpu_routes=%" PRIu64
                " gpu_transient_experts=%" PRIu64 " gpu_transient_routes=%" PRIu64
                " gpu_transient_upload_bytes=%" PRIu64 " gpu_transient_upload_us=%.3f gpu_worker_us=%.3f"
                " gpu_ssd_transient_candidates=%" PRIu64 " gpu_ssd_transient_experts=%" PRIu64
                " gpu_ssd_transient_routes=%" PRIu64 " gpu_ssd_transient_upload_bytes=%" PRIu64
                " gpu_ssd_transient_upload_us=%.3f"
                " ram_experts=%" PRIu64 " ram_routes=%" PRIu64 " ram_bytes=%" PRIu64
                " ssd_experts=%" PRIu64 " ssd_routes=%" PRIu64 " ssd_bytes=%" PRIu64 "\n",
                node.layer, ntokens, nslots, stats.experts, grouped_stats.routed_slots,
                grouped_stats.expert_token_pairs, grouped_stats.max_group_tokens, grouped_stats.workspace_bytes,
                grouped_stats.gpu_experts, grouped_stats.gpu_routes,
                grouped_stats.gpu_transient_experts, grouped_stats.gpu_transient_routes,
                grouped_stats.gpu_transient_upload_bytes, grouped_stats.gpu_transient_upload_us,
                grouped_stats.gpu_worker_us,
                grouped_stats.gpu_ssd_transient_candidates, grouped_stats.gpu_ssd_transient_experts,
                grouped_stats.gpu_ssd_transient_routes, grouped_stats.gpu_ssd_transient_upload_bytes,
                grouped_stats.gpu_ssd_transient_upload_us,
                grouped_stats.ram_experts, grouped_stats.ram_routes, grouped_stats.ram_bytes,
                grouped_stats.ssd_experts, grouped_stats.ssd_routes, grouped_stats.ssd_bytes);
        }
        std::fprintf(stderr,
            "[kimi_cpu_sink_layer] layer=%d calls=%" PRIu64 " experts=%" PRIu64
            " reads=%" PRIu64 " bytes=%" PRIu64 " ring_bytes=%" PRIu64
            " total_us=%.0f io_us=%.0f compute_us=%.0f wait_io_us=%.0f"
            " input_quantize_us=%.0f hidden_quantize_us=%.0f up_gate_us=%.0f down_reduce_us=%.0f"
            " demand_bytes=%" PRIu64 " cache_hits=%" PRIu64 " cache_misses=%" PRIu64
            " cache_admitted=%" PRIu64 " cache_admitted_bytes=%" PRIu64 " cache_copy_us=%.0f"
            " cache_fill_read_requests=%" PRIu64 " cache_fill_read_bytes=%" PRIu64
            " ring_read_requests=%" PRIu64 " ring_read_bytes=%" PRIu64
            " fixed_read_requests=%" PRIu64 " read_completions=%" PRIu64 " pending_reads=%" PRIu64
            " cache_new_slot_bytes=%" PRIu64
            " read_errors=%" PRIu64 " fallback_count=%" PRIu64 "\n",
            node.layer, runtime.calls, stats.experts, stats.read_requests,
            stats.read_bytes, stats.ring_bytes, stats.total_us, stats.io_us,
            stats.compute_us, stats.wait_io_us, stats.input_quantize_us, stats.hidden_quantize_us,
            stats.up_gate_us, stats.down_reduce_us, stats.demand_bytes, stats.cache_hits, stats.cache_misses,
            stats.cache_admitted, stats.cache_admitted_bytes, stats.cache_copy_us,
            stats.cache_fill_read_requests, stats.cache_fill_read_bytes, stats.ring_read_requests, stats.ring_read_bytes,
            stats.fixed_read_requests, stats.read_completions, stats.pending_reads, stats.cache_new_slot_bytes,
            stats.read_errors, stats.fallback_count);
    }
}

inline ggml_tensor llama_kimi_cpu_sink_metadata(const ggml_tensor * tensor) {
    ggml_tensor result = *tensor;
    result.data = nullptr;
    result.buffer = nullptr;
    result.extra = nullptr;
    result.view_src = nullptr;
    std::memset(result.src, 0, sizeof(result.src));
    return result;
}
