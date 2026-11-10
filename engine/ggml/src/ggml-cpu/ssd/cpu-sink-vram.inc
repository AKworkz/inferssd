// Included after sink_cache. Private, default-off GPU tier adapter.
static_assert(int(KIMI_VRAM_ACTIVATION_SITU) == int(KIMI_CPU_SINK_ACTIVATION_SITU) &&
    int(KIMI_VRAM_ACTIVATION_SWIGLU_CLAMP) == int(KIMI_CPU_SINK_ACTIVATION_SWIGLU_CLAMP) &&
    int(KIMI_VRAM_ACTIVATION_SWIGLU) == int(KIMI_CPU_SINK_ACTIVATION_SWIGLU),
    "VRAM tier and CPU sink activation kinds must match");
struct sink_vram {
    const kimi_vram_tier_api * api = nullptr;
    const kimi_vram_residency_api * residency = nullptr;
    const kimi_vram_format_api * format = nullptr; // optional explicit-format extension
    const kimi_vram_prefill_api * prefill_api = nullptr;
    const kimi_vram_elastic_api * elastic = nullptr;
    bool format_path = false;
    void * context = nullptr;
    std::vector<kimi_vram_request> requests;
    std::vector<int> slots;
    // the routes of each request of the next prefill prepare's union, set by the stream (prepare_prefill sends them)
    std::vector<int32_t> prefill_seed_routes;
    int hits = 0;
    // The format's grouped-prefill lane (the tier's, prefill API v9): whether its passes run on the batched engine, and
    // this layer's use of it, and whether this pass's streamed transients are read in place.
    kimi_vram_prefill_lane prefill_lane = {};
    bool prefill_batched = false, prefill_in_place = false;
    bool exclusive = false, decode = false, prefill_open = false, async_upload = false, prefill_enabled = false;
    // A tier with no residency table holds nothing across layers: it serves grouped prefill alone, decode never calls it.
    bool prefill_only = false;
    bool prefill_job_active = false;
    // The batched lanes' transient candidates and the per-layer SSD transient cap: the makespan decides by default
    // (1 route, no cap: a streamed transient holds only a ring slot and a device slot, whatever the count);
    // The supported profile fixes these thresholds; the learned cost table chooses each lane.
    int prefill_ram_gpu_min_routes = 1;
    int prefill_ssd_gpu_min_routes = 1;
    int prefill_ssd_gpu_max_experts_per_layer = -1; // -1: no cap
    bool prefill_legacy_schedule = false;
    // The start values of the cost table's lines the sink fits (kimi-prefill-cost.h, kimi_place_table) and the fitted
    // lines. Typed place.costs overlays named priors; each calibrated line reports the same names.
    using prefill_costs = ssd::place::cost_lines;
    prefill_costs cost, prior;
    ssd::place::prefill_learning learning;
    // This layer's plan, set by select_prefill_transient for the makespan rule: its predicted lanes.
    struct prefill_plan_state {
        bool valid;
        int ntokens;
        double cpu_us, gpu_us, link_us, drive_us, fill_us, gpu_overhead_us, layer_us;
    } prefill_plan = {};
    // Whether a RAM cache slot is page-locked (the sink's decode registrar; unset: none is), and whether the read
    // ring is (B3: registered when a GPU lane copies from it).
    std::function<bool(int)> ram_slot_locked;
    bool ring_locked = false;
    int prefill_latent = 0;
    std::vector<kimi_vram_prefill_transient> prefill_transient;
    std::vector<ssd::place::prefill_expert> prefill_policy_experts;
    kimi_vram_prefill_stats prefill_stats = {};
    uint64_t epoch = 0, ram_drops = 0, ram_drop_blocked = 0, ready_confirmations = 0;
    std::vector<int> ram_leases;
    // Opt-in deferred uploads: the RAM source of an upload that outlives its layer stays leased until collected.
    const kimi_vram_deferred_api * deferred_api = nullptr;
    bool deferred_upload = false, have_token = false;
    uint64_t token = 0, deferred_queued = 0, deferred_collected = 0, deferred_stale = 0, deferred_aborted = 0;
    struct deferred_source { kimi_vram_request key; sink_request source; int lease; uint64_t cache_generation; };
    std::vector<deferred_source> deferred;
    std::vector<uint8_t> upload_queued; // per request of the current layer
    // Optional: the GPU computes experts straight from host memory (kimi_vram_host_api).
    const kimi_vram_host_api * host_api = nullptr;
    uint64_t host_experts = 0;
    const kimi_vram_policy_api * policy_api = nullptr; // optional: miss cost, host demand and floor, host reclaim
    // The CPUs the tier's threads were given (policy set_thread_cpus): the process's other than those of the run's
    // coordinator, read at the first layer on the tier after each begin, once ggml has placed the graph's threads. The
    // process's are read when the sink starts, before ggml places any thread (Linux reads the calling thread's mask).
    ssd::platform::cpu_list thread_cpus, coordinator_cpus, process_cpus;
    bool placement_due = true;
    // The routed layers as the tier is given them (kimi_vram_expert_format): their IDs in the cache's priority order
    // and, once every one has the expert types recorded at load (record_expert_types), each one's types; without them
    // a call's types stand for every layer.
    struct layer_list {
        std::vector<int32_t> ids, up_types, down_types;
        explicit layer_list(const sink_cache & cache) {
            bool typed = true;
            ids.reserve(cache.layers.size());
            for (const auto & l : cache.layers) { ids.push_back(l.id); typed = typed && l.up_type >= 0 && l.down_type >= 0; }
            for (const auto & l : cache.layers) {
                if (!typed) { break; }
                up_types.push_back(l.up_type); down_types.push_back(l.down_type);
            }
        }
        // the declaration with the call's layer's tensors
        kimi_vram_expert_format format(const ggml_tensor * up, const ggml_tensor * down, int latent, int hidden) const {
            return { latent, hidden, int32_t(up->type), int32_t(down->type), int32_t(up->ne[2]), int32_t(ids.size()),
                ids.data(), up_types.empty() ? nullptr : up_types.data(), down_types.empty() ? nullptr : down_types.data() };
        }
    };

    void initialize(bool ready_first, const sink_configuration & configuration) {
        if (!ssd::platform::process_cpus(process_cpus)) { process_cpus.clear(); }
        uint64_t bytes = configuration.policy.vram_budget;
        if (!configuration.production) {
            async_upload = configuration.policy.vram_async;
            exclusive = configuration.policy.vram_exclusive;
            deferred_upload = configuration.policy.vram_deferred;
            prefill_enabled = configuration.policy.vram_prefill;
        }
        if (configuration.production) {
            const auto * budget = configuration.settings.vram_budget();
            if (!budget) { throw std::runtime_error("missing typed vram.budget"); }
            bytes = budget->tier_bytes(configuration.has_gpu);
            async_upload = exclusive = deferred_upload = prefill_enabled = bytes != 0;
        }
        {
            const auto * costs = configuration.production ? configuration.settings.costs() : &configuration.policy.place_costs;
            if (!costs) { throw std::runtime_error("missing typed place.costs"); }
            static const struct { const char * key; double prefill_costs::* field; } terms[] = {
                { "cpu_fixed_us", &prefill_costs::cpu_fixed_us }, { "cpu_route_us", &prefill_costs::cpu_route_us },
                { "gpu_fixed_us", &prefill_costs::gpu_fixed_us }, { "gpu_route_us", &prefill_costs::gpu_route_us },
                { "gpu_host_fixed_us", &prefill_costs::gpu_host_fixed_us },
                { "gpu_host_route_us", &prefill_costs::gpu_host_route_us },
                { "link_gbps", &prefill_costs::link_gbps }, { "pageable_gbps", &prefill_costs::pageable_gbps },
                { "drive_gbps", &prefill_costs::drive_gbps }, { "drive_fixed_us", &prefill_costs::drive_fixed_us },
                { "gpu_token_us", &prefill_costs::gpu_token_us },
            };
            for (const auto & term : terms) {
                const auto found = costs->find(term.key);
                if (found != costs->end()) { cost.*(term.field) = found->second; }
            }
        }
        prior = cost;
        std::fprintf(stderr,
            "[kimi_cpu_prefill_cost] rule=%s cpu_fixed_us=%.3f cpu_route_us=%.3f gpu_fixed_us=%.3f "
            "gpu_route_us=%.3f gpu_host_fixed_us=%.3f gpu_host_route_us=%.3f link_gbps=%.3f pageable_gbps=%.3f "
            "drive_gbps=%.3f drive_fixed_us=%.3f gpu_token_us=%.3f ram_min_routes=%d ssd_min_routes=%d ssd_limit=%d\n",
            prefill_legacy_schedule ? "legacy" : "makespan",
            cost.cpu_fixed_us, cost.cpu_route_us, cost.gpu_fixed_us, cost.gpu_route_us, cost.gpu_host_fixed_us,
            cost.gpu_host_route_us, cost.link_gbps, cost.pageable_gbps, cost.drive_gbps, cost.drive_fixed_us,
            cost.gpu_token_us, prefill_ram_gpu_min_routes, prefill_ssd_gpu_min_routes,
            prefill_ssd_gpu_max_experts_per_layer);
        if (!bytes) { return; }
        if (!ready_first) { throw std::runtime_error("GPU tier requires READY_FIRST=1 for ordered reduction"); }
        if (!configuration.has_gpu) { throw std::runtime_error("GPU provider was not injected; no fallback"); }
        prefill_only = !configuration.gpu.residency;
        if (prefill_only) { async_upload = exclusive = deferred_upload = false; }
        api = configuration.gpu.api;
        if (!api || api->version != 2 || api->size != sizeof(*api) || !api->create || !api->destroy ||
                !api->begin || !api->prepare || !api->launch || !api->admit || !api->finish || !api->cancel) {
            throw std::runtime_error("GPU tier ABI mismatch");
        }
        if (configuration.gpu.format) {
            format = configuration.gpu.format;
            if (!format || format->version != 2 || format->size != sizeof(*format) || !format->prepare || !format->launch) {
                throw std::runtime_error("GPU tier explicit-format ABI mismatch");
            }
        }
        if (prefill_enabled) {
            prefill_api = configuration.gpu.prefill;
            if (!prefill_api || prefill_api->version != 10 || prefill_api->size != sizeof(*prefill_api) ||
                    !prefill_api->prepare || !prefill_api->transient_available || !prefill_api->launch ||
                    !prefill_api->finish || !prefill_api->cancel || !prefill_api->cold_admission_active ||
                    !prefill_api->admit_cold || !prefill_api->lane || !prefill_api->seed_routes ||
                    !prefill_api->seed_host || !prefill_api->seed_wait) {
                throw std::runtime_error("GPU tier grouped-prefill ABI missing or incompatible; no fallback");
            }
            if (!format && !prefill_only) { throw std::runtime_error("GPU tier grouped prefill requires the explicit-format ABI"); }
        }
        if (configuration.gpu.elastic) {
            elastic = configuration.gpu.elastic;
            if (!elastic || elastic->version != 4 || elastic->size != sizeof(*elastic) ||
                    !elastic->phase_complete || !elastic->reserve_accumulate || !elastic->reserve) {
                throw std::runtime_error("GPU tier elastic lifecycle ABI mismatch");
            }
        }
        if (!prefill_only) {
            // what the tier holds (the next-layer prefetch asks), and with VRAM_EXCLUSIVE its ready leases
            residency = configuration.gpu.residency;
            if (!residency || residency->version != 2 || residency->size != sizeof(*residency) || !residency->held ||
                    (exclusive && (!residency->acquire || !residency->validate || !residency->release))) {
                throw std::runtime_error("GPU residency ABI missing or incompatible; no fallback");
            }
        }
        if (deferred_upload) {
            if (!async_upload) { throw std::runtime_error("VRAM_DEFERRED_UPLOAD requires ASYNC_UPLOAD=1"); }
            deferred_api = configuration.gpu.deferred;
            if (!deferred_api || deferred_api->version != 3 || deferred_api->size != sizeof(*deferred_api) ||
                    !deferred_api->admit || !deferred_api->collect || !deferred_api->wait) {
                throw std::runtime_error("GPU tier deferred-upload ABI missing or incompatible; no fallback");
            }
            deferred.reserve(64);
        }
        if (exclusive || async_upload) { ram_leases.reserve(16); }
        char error[512] = {};
        if (configuration.production) {
            const auto * settings_api = configuration.gpu.settings;
            if (!settings_api || settings_api->version != 2 || settings_api->size != sizeof(*settings_api) ||
                    settings_api->runtime_services_size != sizeof(ssd::runtime_services) || !settings_api->create) {
                throw std::runtime_error("GPU tier typed-settings ABI missing or incompatible");
            }
            const ssd::runtime_services services{ configuration.events, &configuration.debug, configuration.host,
                &configuration.gpu, configuration.gpu_device, configuration.source, configuration.gpu_transport };
            context = settings_api->create(&configuration.settings, &services, error, sizeof(error));
        } else {
            const auto * test_api = configuration.gpu.testing;
            if (!test_api || test_api->version != 3 || test_api->size != sizeof(*test_api) || !test_api->create) {
                throw std::runtime_error("explicit CUDA fixture factory missing");
            }
            context = test_api->create(bytes, &configuration.policy.tier, error, sizeof(error));
        }
        if (!context) { throw std::runtime_error(std::string("GPU tier create: ") + error); }
        if (configuration.gpu.policy) {
            policy_api = configuration.gpu.policy;
            if (!policy_api || policy_api->version != 4 || policy_api->size != sizeof(*policy_api) ||
                    !policy_api->set_miss_cost || !policy_api->set_host_demand || !policy_api->set_host_floor ||
                    !policy_api->reclaim_host || !policy_api->host_bytes || !policy_api->set_thread_cpus) {
                throw std::runtime_error("GPU tier admission-policy ABI mismatch");
            }
        }
        if (configuration.gpu.host) {
            host_api = configuration.gpu.host;
            if (!host_api || host_api->version != 3 || host_api->size != sizeof(*host_api) ||
                    !host_api->available || !host_api->launch || !host_api->costs || !host_api->register_host ||
                    !host_api->unregister_host || !host_api->costs2) {
                throw std::runtime_error("GPU tier host-memory ABI mismatch");
            }
        }
    }
    ~sink_vram() {
        if (host_experts) { std::fprintf(stderr, "[kimi_vram_host_sink] gpu_host_experts=%llu\n", (unsigned long long) host_experts); }
        if (context) { api->destroy(context); }
        if (exclusive) { std::fprintf(stderr, "[kimi_vram_exclusive] ram_drops=%llu ram_drop_blocked=%llu ready_confirmations=%llu\n",
            (unsigned long long) ram_drops, (unsigned long long) ram_drop_blocked, (unsigned long long) ready_confirmations); }
        if (deferred_upload) { std::fprintf(stderr, "[kimi_vram_deferred_sink] queued=%llu collected=%llu aborted=%llu stale_leases=%llu\n",
            (unsigned long long) deferred_queued, (unsigned long long) deferred_collected, (unsigned long long) deferred_aborted,
            (unsigned long long) deferred_stale); }
    }
    void check(int result, const char * error) {
        if (result) { throw std::runtime_error(std::string("GPU tier: ") + error); }
    }
    // The tier's threads copy and wait beside this thread, the coordinator, which launches a decode layer's engine calls
    // itself and polls through the layer: they take the process's other CPUs, where it does not wait behind them (all
    // of them when none is left, its own when the process's are unknown). Set again when the coordinator's change.
    void place_threads() {
        if (!placement_due || !policy_api) { return; }
        std::string why;
        if (!ssd::platform::current_thread_cpus(coordinator_cpus, &why)) {
            throw std::runtime_error("GPU tier thread placement: " + why);
        }
        ssd::platform::cpu_list cpus;
        for (int cpu : process_cpus) {
            if (std::find(coordinator_cpus.begin(), coordinator_cpus.end(), cpu) == coordinator_cpus.end()) {
                cpus.push_back(cpu);
            }
        }
        if (cpus.empty()) { cpus = process_cpus.empty() ? coordinator_cpus : process_cpus; }
        if (cpus != thread_cpus) {
            char error[512] = {};
            check(policy_api->set_thread_cpus(context, cpus.data(), int(cpus.size()), error, sizeof(error)), error);
            thread_cpus = cpus;
        }
        placement_due = false;
    }
    void begin(uint64_t epoch, uint64_t token, bool decode, sink_cache & cache, bool request = false) {
        if (!context) { return; }
        placement_due = true;
        // Graph-layer prefill calls repeat the context boundary transition.
        // The first transition owns shrink/reclaimer arming; repeating it after
        // scheduler reserve would incorrectly re-arm OOM reclamation mid-graph.
        // A prefill phase stays open until phase_complete(true) or a decode
        // begin, not until the next decode graph: a request that samples no
        // decode step must still let the next request reopen prefill. A
        // request's boundary opens a new phase even then: a prompt evaluated
        // outside any request leaves its phase open.
        if (prefill_open && !decode && !request) { return; }
        if (deferred_upload) {
            // Anything but the next decode token of this epoch may release the arena or the RAM cache.
            const bool next = decode && this->decode && have_token && epoch == this->epoch && token == this->token + 1;
            collect_deferred(cache, next ? KIMI_VRAM_COLLECT_NONE : KIMI_VRAM_COLLECT_ALL);
        }
        char error[512] = {};
        const int result = api->begin(context, epoch, token, decode, error, sizeof(error));
        check(result, error);
        this->epoch = epoch;
        this->decode = decode;
        prefill_open = !decode;
        this->token = token;
        have_token = decode;
    }
    // A pass's engine buffers, announced before its reads (none without the GPU's grouped prefill).
    void reserve_accumulate(int tokens) {
        if (!context || !elastic || !prefill_enabled) { return; }
        char error[512] = {};
        check(elastic->reserve_accumulate(context, tokens, error, sizeof(error)), error);
    }
    // At load (unified memory): the tier's storage before any request, for experts of this geometry, given with one
    // layer's tensors (the request's layer). The role bytes are the tensors' per-expert strides, as the decode
    // requests carry them; the key is not needed.
    void reserve(const ggml_tensor * up, const ggml_tensor * gate, const ggml_tensor * down, const layer_list & layers) {
        if (!context || !elastic || (!format && !prefill_only) || layers.ids.empty()) { return; }
        const int layer = ssd::host::routed_layer(up);
        if (layer < 0) { throw std::runtime_error("GPU tier reserve requires a canonical routed layer"); }
        kimi_vram_request r = {};
        r.layer = layer; r.weight = 1.0f;
        const ggml_tensor * tensors[] = { up, gate, down };
        for (int role = 0; role < 3; ++role) { r.role[role].bytes = tensors[role]->nb[2]; }
        const kimi_vram_expert_format f = layers.format(up, down, int32_t(up->ne[0]), int32_t(up->ne[1]));
        char error[512] = {};
        check(elastic->reserve(context, &r, &f, error, sizeof(error)), error);
    }
    void phase_complete(bool decode) {
        if (decode) { prefill_open = false; } // the tier leaves prefill: scratch released, arena grown
        if (!context || !elastic) { return; }
        char error[512] = {};
        check(elastic->phase_complete(context, decode ? 1 : 0, error, sizeof(error)), error);
    }
    // Expert geometry is supplied through the explicit-format extension.
    void prepare(std::vector<sink_request> & source, const ggml_tensor * up, const ggml_tensor * down,
                 int latent, int hidden, const layer_list & layers, sink_cache & cache, bool unweighted = false) {
        hits = 0;
        if (!context || prefill_only) { return; }
        place_threads();
        const int layer = ssd::host::routed_layer(up);
        if (layer < 0 || layer >= KIMI_VRAM_MAX_LAYERS || source.size() > 16) {
            throw std::runtime_error("GPU tier requires a canonical routed layer and <=16 slots");
        }
        if (!format) {
            throw std::runtime_error("GPU tier lacks the explicit expert format ABI for these experts; no fallback");
        }
        if (deferred_upload && decode) {
            // The RAM cache refuses a layer's prepare while that layer's entries are leased.
            collect_deferred(cache, cache.allocation == sink_cache::global ? KIMI_VRAM_COLLECT_ALL : layer);
        }
        upload_queued.assign(source.size(), 0);
        requests.resize(source.size()); slots.assign(source.size(), -1);
        for (size_t i = 0; i < source.size(); ++i) {
            auto & r = requests[i]; r.layer = layer; r.expert = source[i].id; r.weight = unweighted ? 1.0f : source[i].weight;
            for (int role = 0; role < 3; ++role) {
                const auto k = sink_cache::key(source[i].role[role]);
                r.role[role] = { k.device, k.inode, k.size, k.offset, k.bytes,
                                k.mtime_s, k.mtime_ns, k.ctime_s, k.ctime_ns, k.inode_hi };
            }
        }
        char error[512] = {};
        int result;
        {
            const kimi_vram_expert_format f = layers.format(up, down, latent, hidden);
            result = format->prepare(context, requests.data(), int(requests.size()), &f, slots.data(), error, sizeof(error));
        }
        format_path = true;
        check(result, error);
        for (size_t i = 0; i < source.size(); ++i) { source[i].vram_slot = slots[i]; hits += slots[i] >= 0; }
    }
    // The tier's engines take the f32 input row.
    void launch(const float * input, size_t bytes, const kimi_cpu_sink_activation & activation) {
        if (!context || prefill_only) { return; }
        char error[512] = {};
        int result;
        if (format_path) {
            const kimi_vram_activation a = { activation.kind, activation.situ_beta, activation.situ_linear_beta, activation.swiglu_limit };
            result = format->launch(context, input, bytes, &a, error, sizeof(error));
        } else {
            throw std::runtime_error("GPU launch requires an explicit expert format");
        }
        check(result, error);
    }
    void pin_ram_batch(sink_cache & cache, const std::vector<sink_request> & source) {
        if ((!exclusive && !async_upload) || !decode) { return; }
        if (!ram_leases.empty()) { throw std::runtime_error("GPU tier still holds a previous RAM batch"); }
        // One lease per physical slot covers CPU, its read CQEs and a possible
        // borrowed upload. Repeated router IDs cannot release it twice.
        for (const auto & r : source) {
            const int index = r.cache_hit >= 0 ? r.cache_hit : r.cache_admit;
            if (index < 0 || std::find(ram_leases.begin(), ram_leases.end(), index) != ram_leases.end()) { continue; }
            cache.acquire_lease(index, r); ram_leases.push_back(index);
        }
    }
    void admit(size_t index, const void * const * data, bool borrow_until_finish) {
        if (!context || prefill_only) { return; }
        char error[512] = {};
        if (deferred_upload && borrow_until_finish) {
            int queued = 0;
            check(deferred_api->admit(context, &requests.at(index), data[0], data[1], data[2], &queued, error, sizeof(error)), error);
            if (queued) { upload_queued.at(index) = 1; ++deferred_queued; }
            return;
        }
        const int result = api->admit(context, &requests.at(index), data[0], data[1], data[2],
                                      borrow_until_finish, error, sizeof(error));
        check(result, error);
    }
    // Which of `count` experts of `layer` the tier holds now (out: 1 or 0); none without a tier.
    void held(int32_t layer, const int32_t * experts, int count, uint8_t * out) const {
        std::fill(out, out + count, uint8_t(0));
        if (context && residency) { residency->held(context, layer, experts, count, out); }
    }
    void set_host_demand(uint64_t bytes) {
        if (context && policy_api) { policy_api->set_host_demand(context, bytes); }
    }
    // The host-memory ledger's floor, which the tier keeps free on unified memory.
    void set_host_floor(uint64_t bytes) {
        if (context && policy_api) { policy_api->set_host_floor(context, bytes); }
    }
    // Unified memory: tier blocks go so that `bytes` of host memory can be allocated (false: a layer is live, or not
    // enough could go). Deferred uploads are collected first by the caller.
    bool reclaim_host(uint64_t bytes) {
        return context && policy_api && !policy_api->reclaim_host(context, bytes);
    }
    // DRAM the tier holds on unified memory (0 on a discrete GPU).
    uint64_t host_bytes() {
        return context && policy_api ? policy_api->host_bytes(context) : 0;
    }
    void set_miss_cost(double us) {
        if (context && policy_api && decode && us > 0) { policy_api->set_miss_cost(context, us); }
    }
    // The GPU can compute a decode expert from host memory in this layer (in place, or after a copy: costs2 says).
    bool host_ready() {
        return context && host_api && decode && format_path && host_api->available(context);
    }
    void host_costs(double & resident_us, double & host_us) {
        resident_us = host_us = 0;
        if (context && host_api) { host_api->costs(context, &resident_us, &host_us); }
    }
    // Page-locks host memory for the device's copies (no-op where it reads host memory in place). false: refused.
    bool register_host(const void * pointer, size_t bytes, std::string * why = nullptr) {
        if (!context || !host_api) { return false; }
        char error[256] = {};
        const bool ok = host_api->register_host(context, pointer, bytes, error, sizeof(error)) == 0;
        if (!ok && why) { *why = error; }
        return ok;
    }
    bool unregister_host(const void * pointer) {
        if (!context || !host_api) { return false; }
        char error[256] = {};
        return host_api->unregister_host(context, pointer, error, sizeof(error)) == 0;
    }
    // Queues experts behind the layer's resident batch, as one engine batch where possible; false if the tier
    // has no launched layer to follow.
    bool launch_host(const std::vector<kimi_vram_host_job> & jobs) {
        char error[512] = {};
        int accepted = 0;
        check(host_api->launch(context, jobs.data(), int(jobs.size()), &accepted, error, sizeof(error)), error);
        if (accepted) { host_experts += jobs.size(); }
        return accepted != 0;
    }
    void finish(float * contributions, size_t stride, sink_cache & cache, const std::vector<sink_request> & source) {
        if (!context || prefill_only) { return; }
        char error[512] = {};
        const int result = api->finish(context, contributions, stride, error, sizeof(error));
        check(result, error);
        if (!decode) { return; }
        if (deferred_upload) { retain_deferred(cache, source); }
        release_ram_batch(cache);
        drop_ready(cache,source);
        if (deferred_upload) { collect_deferred(cache, KIMI_VRAM_COLLECT_NONE); }
    }
    // The layer's grouped prefill on the tier: every format the tier has a lane for (it initializes the format on a cold
    // prefill, its decode arena deferred to the first decode token).
    int prepare_prefill(std::vector<sink_request> & source, const ggml_tensor * up, const ggml_tensor * down,
            int latent, int hidden, const layer_list & layers, bool post_norm) {
        hits = 0;
        prefill_plan.valid = false;
        learning.cpu_fit = {};
        prefill_stats = {};
        prefill_batched = prefill_in_place = false;
        if (!context || !prefill_enabled || source.empty()) { return 0; }
        place_threads();
        const int layer = ssd::host::routed_layer(up);
        if (layer < 0 || layer >= KIMI_VRAM_MAX_LAYERS || source.size() > static_cast<size_t>(up->ne[2])) {
            throw std::runtime_error("GPU grouped prefill requires a canonical routed layer and a distinct expert union");
        }
        prefill_latent = latent;
        if (!learning.window) { learning.window = std::max<size_t>(1, layers.ids.size()); }
        requests.resize(source.size()); slots.assign(source.size(), -1);
        for (size_t i = 0; i < source.size(); ++i) {
            auto & r = requests[i]; r.layer = layer; r.expert = source[i].id; r.weight = 1.0f;
            for (int role = 0; role < 3; ++role) {
                const auto k = sink_cache::key(source[i].role[role]);
                r.role[role] = { k.device, k.inode, k.size, k.offset, k.bytes,
                                k.mtime_s, k.mtime_ns, k.ctime_s, k.ctime_ns, k.inode_hi };
            }
        }
        const kimi_vram_expert_format f = layers.format(up, down, latent, hidden);
        char error[512] = {};
        // the pass's routes of each request, so that the prepare seeds the layer's empty slots with the experts the
        // pass routes most (kimi_vram_prefill_api v10)
        if (prefill_seed_routes.size() == requests.size()) {
            check(prefill_api->seed_routes(context, prefill_seed_routes.data(), int(prefill_seed_routes.size()), error,
                sizeof(error)), error);
        }
        prefill_seed_routes.clear();
        check(prefill_api->prepare(context, requests.data(), int(requests.size()), &f,
                slots.data(), error, sizeof(error)), error);
        prefill_lane = {};
        prefill_api->lane(context, &prefill_lane);
        if (prefill_lane.engine == KIMI_VRAM_PREFILL_ENGINE_NONE || (post_norm && !prefill_lane.post_norm)) {
            // the tier pinned the layer's residents only where it has a lane: give them back
            if (std::any_of(slots.begin(), slots.end(), [](int slot) { return slot >= 0; })) {
                check(prefill_api->cancel(context, error, sizeof(error)), error);
            }
            std::fill(slots.begin(), slots.end(), -1);
            return 0;
        }
        prefill_batched = prefill_lane.engine == KIMI_VRAM_PREFILL_ENGINE_BATCHED;
        for (size_t i = 0; i < source.size(); ++i) {
            source[i].vram_slot = slots[i];
            hits += slots[i] >= 0;
        }
        return hits;
    }
    // The layer's experts as the cost table's planner takes them (kimi_place_item), for decode and grouped prefill
    // alike: routes(i) its tokens in the layer (0: no work), movable(i) whether the caller can hand it to the GPU,
    // needs_slot(i) whether a move takes one of the tier's free device slots. Its host memory is page-locked where its
    // RAM cache slot is (the decode registrar), or, for a read into the ring, where the ring is.
    template <class Routes, class Movable, class NeedsSlot>
    std::vector<kimi_place_item> place_items(const std::vector<sink_request> & source, Routes routes, Movable movable,
            NeedsSlot needs_slot) const {
        std::vector<kimi_place_item> items(source.size());
        auto locked = [&](int entry) { return entry >= 0 && ram_slot_locked && ram_slot_locked(entry); };
        for (size_t i = 0; i < source.size(); ++i) {
            const auto & r = source[i];
            auto & x = items[i];
            x.routes = routes(i);
            x.mb = double(r.role[0].nbytes + r.role[1].nbytes + r.role[2].nbytes) / 1e6;
            x.mb_early = double(r.role[0].nbytes + r.role[1].nbytes) / 1e6;
            x.resident = r.vram_slot >= 0;
            if (x.resident || x.routes <= 0) { continue; }
            x.read = r.cache_hit < 0;
            x.locked = r.cache_hit >= 0 ? locked(r.cache_hit) : r.cache_admit >= 0 ? locked(r.cache_admit) : ring_locked;
            x.movable = movable(i);
            x.needs_slot = needs_slot(i);
        }
        return items;
    }
    void select_prefill_transient(std::vector<sink_request> & source,
            kimi_cpu_sink_prefill_batch & batch, sink_cache & cache, size_t ssd_staging_limit, size_t ring_slots,
            const kimi_place_table & table) {
        prefill_transient.clear();
        for (auto & request : source) {
            request.vram_transient = false;
            request.vram_ssd_transient = false;
            request.vram_ssd_streaming = false;
            request.vram_ssd_staging = -1;
            request.lane_start_us = 0.0;
            request.prefill_routes = 0;
        }
        const bool batched = prefill_batched;
        if (!context || !prefill_enabled || !batched || source.empty()) { return; }
        const int ram_threshold = prefill_ram_gpu_min_routes;
        const int ssd_threshold = prefill_ssd_gpu_min_routes;
        const int ssd_cap = prefill_ssd_gpu_max_experts_per_layer;
        const bool transient_ready = (ram_threshold || ssd_threshold) &&
            prefill_api->transient_available(context);
        if (!transient_ready) { return; }

        std::vector<int> token_routes(source.size(), 0);
        std::vector<int> last_token(source.size(), -1);
        for (int token_index = 0; token_index < batch.ntokens; ++token_index) {
            for (int slot = 0; slot < batch.nslots; ++slot) {
                const size_t route = size_t(token_index) * size_t(batch.nslots) + size_t(slot);
                const int32_t request = batch.route_request_index[route];
                if (request >= 0 && size_t(request) < source.size() &&
                        last_token[size_t(request)] != token_index) {
                    ++token_routes[size_t(request)];
                    last_token[size_t(request)] = token_index;
                }
            }
        }
        for (size_t i = 0; i < source.size(); ++i) { source[i].prefill_routes = token_routes[i]; }

        if (transient_ready) {
            // The makespan rule (every batched lane, unless the legacy switch): the layer is placed from the cost
            // table (kimi-prefill-cost.h, kimi_place_layer), as a decode layer is: the CPU and the GPU at their
            // measured lines for this pass's size, the copies at the measured rate of their transport (from the
            // page-locked ring or RAM slot: DMA on the link while the GPU computes; pageable: staged by the GPU's
            // worker), the drive's line, and the form the reads take (the list schedule when they fit the ring,
            // the coupled streamed form otherwise). The candidates move in order of routes, and the planner keeps the
            // fewest within its form's resolution of the lowest time. The reads are then issued in the order the
            // lanes need them (lane_start_us), so that the lanes overlap. The legacy selector keeps its
            // fixed 2026-09 fits and serial greedy rule.
            const bool makespan = batched && !prefill_legacy_schedule;
            auto expert_bytes = [&source](size_t i) {
                return source[i].role[0].nbytes + source[i].role[1].nbytes + source[i].role[2].nbytes;
            };
            auto ram_movable = [&](size_t i) {
                return ram_threshold && source[i].cache_hit >= 0 && source[i].cache_lease && token_routes[i] >= ram_threshold;
            };
            auto ssd_movable = [&](size_t i) { return ssd_threshold && source[i].cache_hit < 0 && token_routes[i] >= ssd_threshold; };
            prefill_policy_experts.resize(source.size());
            for (size_t i = 0; i < source.size(); ++i) {
                prefill_policy_experts[i] = { source[i].id, source[i].vram_slot >= 0,
                    source[i].cache_hit >= 0, source[i].cache_lease, expert_bytes(i) };
            }
            std::vector<kimi_place_item> items;
            const ssd::place::prefill_input input = {prefill_policy_experts, token_routes, items, table,
                batch.ntokens, ram_threshold, ssd_threshold, ssd_cap, ssd_staging_limit, ring_slots, makespan};
            const auto ranked = ssd::place::rank_prefill(input, batch.gpu_ssd_transient_candidates);
            if (makespan) {
                items = place_items(source, [&](size_t i) { return token_routes[i]; },
                    [&](size_t i) { return ram_movable(i) || ssd_movable(i); }, [](size_t) { return false; });
            }
            const auto selected_plan = ssd::place::plan_prefill(input, ranked, prefill_in_place);
            const auto & placed = selected_plan.placed;
            size_t ram_selected = 0, ssd_selected = 0;
            uint64_t ram_selected_routes = 0, ssd_selected_routes = 0;
            uint64_t ram_selected_bytes = 0, ssd_selected_bytes = 0;
            for (const auto & item : selected_plan.chosen) {
                const size_t index = item.index;
                const int routes = token_routes[index];
                const uint64_t bytes = expert_bytes(index);
                kimi_vram_prefill_transient selected = {};
                selected.request_index = int32_t(index);
                selected.source = item.source;
                if (item.source == KIMI_VRAM_PREFILL_TRANSIENT_RAM) {
                    for (int role = 0; role < 3; ++role) {
                        selected.role[role] = static_cast<const char *>(cache.data(source[index].cache_hit, role)) + source[index].role[role].data_offset;
                    }
                    ++ram_selected;
                    ram_selected_routes += uint64_t(routes);
                    ram_selected_bytes += bytes;
                } else {
                    source[index].vram_ssd_transient = true;
                    source[index].vram_ssd_staging = int(ssd_selected++);
                    ssd_selected_routes += uint64_t(routes);
                    ssd_selected_bytes += bytes;
                }
                prefill_transient.push_back(selected);
                source[index].vram_transient = true;
            }
            if (makespan) {
                prefill_plan = { true, batch.ntokens, placed.cpu, placed.gpu_lane, placed.link, placed.drive,
                    placed.fill, placed.overhead, placed.time };
                // Each lane's predicted start of its experts (the GPU's in selection order, a page-locked copy at its
                // place on the link; the CPU's in request order), for the reads' issue order.
                for (size_t i = 0; i < source.size(); ++i) { source[i].lane_start_us = placed.start[i]; }
            }
            std::fprintf(stderr,
                "[kimi_cpu_prefill_schedule] format=%s rule=%s ram_threshold=%d ssd_threshold=%d ssd_limit=%zu "
                "ram_candidates=%zu ram_selected=%zu ram_selected_routes=%llu ram_upload_bytes=%llu "
                "ssd_candidates=%zu ssd_selected=%zu ssd_selected_routes=%llu ssd_upload_bytes=%llu "
                "cpu_backlog_before_us=%.1f cpu_backlog_after_us=%.1f gpu_backlog_before_us=%.1f "
                "gpu_backlog_after_us=%.1f link_backlog_us=%.1f "
                "drive_us=%.1f fill_us=%.1f gpu_overhead_us=%.1f ring=%zu form=%s ring_locked=%d in_place=%d "
                "makespan_before_us=%.1f makespan_after_us=%.1f scratch_available=1 displacement_slots=0\n",
                "batched", makespan ? "makespan" : "legacy",
                ram_threshold,ssd_threshold,selected_plan.ssd_limit,
                selected_plan.ram_candidates,ram_selected,(unsigned long long)ram_selected_routes,
                (unsigned long long)ram_selected_bytes,selected_plan.ssd_candidates,ssd_selected,
                (unsigned long long)ssd_selected_routes,(unsigned long long)ssd_selected_bytes,
                selected_plan.cpu_before,selected_plan.cpu_after,selected_plan.gpu_before,selected_plan.gpu_after,selected_plan.link_after,
                placed.drive,placed.fill,placed.overhead,ring_slots,selected_plan.in_ring?"ring":"streamed",int(ring_locked),
                int(prefill_in_place),selected_plan.time_before,selected_plan.time_after);
        }

    }
    void bind_prefill_ssd_transient(size_t request_index, const void * const data[3],
            int (*stage)(void *, int32_t, int) = nullptr, void * context = nullptr) {
        for (auto & item : prefill_transient) {
            if (item.source != KIMI_VRAM_PREFILL_TRANSIENT_SSD ||
                    item.request_index != int32_t(request_index)) { continue; }
            for (int role = 0; role < 3; ++role) {
                if (!data[role] || item.role[role]) {
                    throw std::runtime_error("invalid SSD prefill staging binding");
                }
                item.role[role] = data[role];
            }
            item.stage = stage;
            item.stage_context = context;
            return;
        }
        throw std::runtime_error("SSD prefill staging request was not scheduled");
    }
    void launch_prefill(const float * input, size_t stride, const int32_t * route_requests, const float * weights,
            int nslots, int ntokens, const kimi_cpu_sink_activation & activation) {
        if (!context || !prefill_enabled || (!hits && prefill_transient.empty())) { return; }
        const kimi_vram_prefill_launch options = {
            { activation.kind, activation.situ_beta, activation.situ_linear_beta, activation.swiglu_limit },
            activation.post_exp_norm, activation.post_exp_norm_eps, int32_t(prefill_in_place),
        };
        char error[512] = {};
        check(prefill_api->launch(context, input, stride, route_requests, weights, nslots, ntokens,
                prefill_transient.data(), int(prefill_transient.size()), &options, error, sizeof(error)), error);
        prefill_job_active = true;
    }
    void finish_prefill(float * output, size_t stride, kimi_cpu_sink_prefill_batch & batch) {
        if (!context || !prefill_enabled || !prefill_job_active) { return; }
        char error[512] = {};
        prefill_stats = {};
        check(prefill_api->finish(context, output, stride, &prefill_stats, error, sizeof(error)), error);
        batch.gpu_transient_experts = prefill_stats.ram_transient_experts;
        batch.gpu_transient_routes = prefill_stats.ram_transient_routes;
        batch.gpu_transient_upload_bytes = prefill_stats.ram_transient_upload_bytes;
        batch.gpu_transient_upload_us = prefill_stats.ram_transient_upload_us;
        batch.gpu_ssd_transient_experts = prefill_stats.ssd_transient_experts;
        batch.gpu_ssd_transient_routes = prefill_stats.ssd_transient_routes;
        batch.gpu_ssd_transient_upload_bytes = prefill_stats.ssd_transient_upload_bytes;
        batch.gpu_ssd_transient_upload_us = prefill_stats.ssd_transient_upload_us;
        batch.gpu_worker_us = prefill_stats.total_worker_us;
        batch.gpu_expert_us = prefill_stats.gpu_fit_us;
        batch.gpu_read_wait_us = prefill_stats.read_wait_us;
        hits = 0; prefill_job_active = false; prefill_transient.clear();
    }
    // A request the CPU computes in a sink prefill: the tier copies its payload into the seed slot it reserved for it,
    // if any (kimi_vram_prefill_api v10). seed_wait returns once the copy is done, before the payload's ring slot or
    // RAM cache lease goes back.
    void seed_host(size_t request, const void * const * data) {
        if (!context || !prefill_api) { return; }
        char error[512] = {};
        check(prefill_api->seed_host(context, int32_t(request), data[0], data[1], data[2], error, sizeof(error)), error);
    }
    void seed_wait(size_t request) {
        if (!context || !prefill_api) { return; }
        char error[512] = {};
        check(prefill_api->seed_wait(context, int32_t(request), error, sizeof(error)), error);
    }
    // The CPU consumer's time for one expert of the current grouped-prefill layer.
    void observe_prefill_cpu(int routes, double us) {
        if (prefill_plan.valid && routes > 0) { learning.cpu_fit.add(double(routes), us); }
    }
    ssd::place::lane_geometry lane_geometry() const {
        return { prefill_lane.token_bytes, prefill_lane.output_bytes,
            prefill_lane.engine == KIMI_VRAM_PREFILL_ENGINE_BATCHED, prefill_lane.in_place != 0 };
    }
    double overhead_row_prior(double locked_us_per_mb) const {
        return ssd::place::overhead_row_prior(prior, lane_geometry(), locked_us_per_mb);
    }
    // After each grouped-prefill layer: read_us is the I/O time with reads in flight, layer_us the layer so far. The
    // batched lanes' lines are refit over the last pass's layers; table() then builds the cost table from them (with
    // decode's lines and the tier's copy rates), and the calibrated line prints it by the names the start values take,
    // with this layer's prediction (made before it ran, on the terms of then) against what it measured.
    template <class Table>
    void calibrate_prefill(double cpu_us, const kimi_cpu_sink_prefill_batch & batch, uint64_t read_bytes,
            double read_us, double layer_us, Table table) {
        if (!prefill_batched || prefill_legacy_schedule || !prefill_plan.valid) { return; }
        const prefill_plan_state plan = prefill_plan;
        prefill_plan.valid = false;
        // the batched lanes' lines: a one-token pass ran the vector engine and is priced as decode
        if (plan.ntokens <= prefill_lane.vector_tokens) { return; }
        ssd::place::prefill_feedback measured = {};
        measured.gpu = { prefill_stats.gpu_fit_experts, prefill_stats.gpu_fit_routes,
            prefill_stats.gpu_fit_routes_sq, prefill_stats.gpu_fit_us, prefill_stats.gpu_fit_route_us };
        measured.host = { prefill_stats.host_fit_experts, prefill_stats.host_fit_routes,
            prefill_stats.host_fit_routes_sq, prefill_stats.host_fit_us, prefill_stats.host_fit_route_us };
        measured.upload_us = batch.gpu_transient_upload_us;
        measured.ssd_upload_us = batch.gpu_ssd_transient_upload_us;
        measured.engine_overhead_us = prefill_stats.engine_overhead_us;
        measured.transient_wait_us = prefill_stats.transient_wait_us;
        measured.output_add_us = prefill_stats.output_add_us;
        measured.total_worker_us = prefill_stats.total_worker_us;
        measured.read_wait_us = prefill_stats.read_wait_us;
        measured.read_bytes = read_bytes; measured.read_us = read_us; measured.ntokens = plan.ntokens;
        const auto fitted = ssd::place::observe_prefill_lines(learning, cost, prior, measured);
        const auto & overhead = fitted.overhead;
        ssd::place::observe_prefill_overhead(learning, cost, overhead, overhead_row_prior(table().locked_us_per_mb));
        const kimi_place_table t = table();
        std::fprintf(stderr,
            "[kimi_cpu_prefill_cost] calibrated=%llu tokens=%d samples=%zu cpu_fixed_us=%.3f cpu_route_us=%.3f "
            "gpu_fixed_us=%.3f gpu_route_us=%.3f gpu_host_fixed_us=%.3f gpu_host_route_us=%.3f link_gbps=%.3f "
            "pageable_gbps=%.3f drive_gbps=%.3f drive_fixed_us=%.3f gpu_overhead_fixed_us=%.3f gpu_token_us=%.3f "
            "gpu_overhead_samples=%.0f gpu_overhead_min_tokens=%d gpu_overhead_max_tokens=%d cpu_one_us=%.1f "
            "predicted_us=%.1f measured_us=%.1f cpu_predicted_us=%.1f cpu_us=%.1f gpu_predicted_us=%.1f gpu_us=%.1f "
            "drive_predicted_us=%.1f drive_us=%.1f gpu_overhead_predicted_us=%.1f gpu_overhead_us=%.1f\n",
            (unsigned long long) learning.calibrations, plan.ntokens, learning.samples.size(),
            cost.cpu_fixed_us, cost.cpu_route_us, cost.gpu_fixed_us, cost.gpu_route_us, cost.gpu_host_fixed_us,
            cost.gpu_host_route_us, t.locked_us_per_mb > 0 ? 1e3 / t.locked_us_per_mb : 0.0,
            t.pageable_us_per_mb > 0 ? 1e3 / t.pageable_us_per_mb : 0.0, t.drive_us_per_mb > 0 ? 1e3 / t.drive_us_per_mb : 0.0,
            t.drive_fixed_us, cost.gpu_overhead_fixed_us, cost.gpu_token_us,
            overhead.samples.n, overhead.min_tokens, overhead.max_tokens, t.cpu_one_us,
            plan.layer_us, layer_us, plan.cpu_us, cpu_us, plan.gpu_us, fitted.gpu_lane_us,
            plan.drive_us, read_us, plan.gpu_overhead_us, fitted.gpu_overhead_us);
    }
    // The prefill window's drive samples (MB, us), for the table's drive line (fitted with decode's layers).
    void cancel_prefill() {
        if (!context || !prefill_enabled ||
                !prefill_job_active) { return; }
        char error[512] = {};
        if (prefill_api->cancel(context, error, sizeof(error))) {
            std::fprintf(stderr, "[kimi_vram_prefill] failed to drain GPU work: %s; aborting\n", error);
            std::abort();
        }
        hits = 0; prefill_job_active = false; prefill_transient.clear();
    }
    // Moves the RAM leases of this layer's queued uploads out of the batch; collect_deferred releases them.
    void retain_deferred(sink_cache & cache, const std::vector<sink_request> & source) {
        for (size_t i = 0; i < source.size() && i < upload_queued.size(); ++i) {
            if (!upload_queued[i]) { continue; }
            const int index = source[i].cache_hit >= 0 ? source[i].cache_hit : source[i].cache_admit;
            const auto lease = std::find(ram_leases.begin(), ram_leases.end(), index);
            if (index < 0 || lease == ram_leases.end()) { throw std::runtime_error("deferred VRAM upload without a RAM lease"); }
            ram_leases.erase(lease);
            deferred.push_back({requests[i], source[i], index, cache.generation});
        }
    }
    static bool same_key(const kimi_vram_request & a, const kimi_vram_request & b) {
        if (a.layer != b.layer || a.expert != b.expert) { return false; }
        for (int r = 0; r < 3; ++r) {
            const auto & x = a.role[r];
            const auto & y = b.role[r];
            if (x.device != y.device || x.inode != y.inode || x.inode_hi != y.inode_hi || x.size != y.size ||
                    x.offset != y.offset || x.bytes != y.bytes ||
                    x.mtime_s != y.mtime_s || x.mtime_ns != y.mtime_ns || x.ctime_s != y.ctime_s || x.ctime_ns != y.ctime_ns) {
                return false;
            }
        }
        return true;
    }
    // Publishes completed deferred uploads, releases their RAM leases and, when exclusive, drops the RAM copy.
    void collect_deferred(sink_cache & cache, int32_t wait_layer) {
        if (!deferred_upload || !context) { return; }
        try {
            kimi_vram_request done[16] = {};
            int32_t aborted[16] = {};
            std::vector<kimi_vram_request> keys;
            std::vector<sink_request> sources;
            int count = 0;
            do {
                char error[512] = {};
                check(deferred_api->collect(context, wait_layer, done, aborted, 16, &count, error, sizeof(error)), error);
                wait_layer = -2;
                keys.clear(); sources.clear();
                for (int k = 0; k < count; ++k) {
                    const auto it = std::find_if(deferred.begin(), deferred.end(),
                        [&](const deferred_source & d) { return same_key(d.key, done[k]); });
                    if (it == deferred.end()) { throw std::runtime_error("GPU tier collected an upload without a CPU sink lease"); }
                    ++deferred_collected;
                    // A RAM cache released since the lease (pressure pause, geometry change) no longer holds it.
                    if (it->cache_generation != cache.generation) { ++deferred_stale; deferred.erase(it); continue; }
                    cache.release_lease(it->lease);
                    // An aborted upload's slot went with a reclaimed block: the RAM copy stays the only one.
                    if (aborted[k]) { ++deferred_aborted; }
                    else if (exclusive && decode && have_token) { keys.push_back(it->key); sources.push_back(it->source); }
                    deferred.erase(it);
                }
                if (!keys.empty()) { drop_collected(cache, keys, sources); }
            } while (count == 16);
        } catch (...) {
            abandon_deferred(cache);
            throw;
        }
    }
    void drop_collected(sink_cache & cache, const std::vector<kimi_vram_request> & keys, const std::vector<sink_request> & sources) {
        char error[512] = {};
        int ready[16] = {}; kimi_vram_ready_lease lease = {};
        int rc = residency->acquire(context, epoch, keys.data(), int(keys.size()), &lease, ready, error, sizeof(error));
        check(rc, error);
        rc = residency->validate(context, &lease, error, sizeof(error)); check(rc, error);
        for (size_t i = 0; i < keys.size(); ++i) {
            if (!ready[i]) { continue; }
            ++ready_confirmations;
            const int dropped = cache.drop_duplicate(keys[i].layer, sources[i]);
            ram_drops += dropped > 0; ram_drop_blocked += dropped < 0;
        }
        rc = residency->release(context, &lease, error, sizeof(error)); check(rc, error);
    }
    // After a failed collect: let every copy finish (a failed tier has already drained), then return the leases.
    void abandon_deferred(sink_cache & cache) {
        char error[512] = {};
        if (context) { deferred_api->wait(context, error, sizeof(error)); }
        for (const auto & d : deferred) {
            if (d.cache_generation != cache.generation) { continue; }
            const auto & e = cache.entries.at(size_t(d.lease));
            if (e.leases && !e.reserved) { cache.release_lease(d.lease); }
        }
        deferred.clear();
    }
    // Called by the RAM cache before it unmaps its arena.
    void wait_deferred() {
        if (!deferred_upload || !context || deferred.empty()) { return; }
        char error[512] = {};
        check(deferred_api->wait(context, error, sizeof(error)), error);
    }
    void release_ram_batch(sink_cache & cache) {
        for (int index : ram_leases) { cache.release_lease(index); }
        ram_leases.clear();
    }
    void drop_ready(sink_cache & cache,const std::vector<sink_request> & source) {
        if (!exclusive || source.empty()) { return; }
        char error[512]={};
        int ready[16] = {}; kimi_vram_ready_lease lease = {};
        int rc = residency->acquire(context, epoch, requests.data(), int(requests.size()), &lease, ready, error, sizeof(error));
        check(rc, error);
        rc = residency->validate(context, &lease, error, sizeof(error)); check(rc, error);
        for (size_t i = 0; i < source.size(); ++i) {
            if (!ready[i]) { continue; }
            ++ready_confirmations;
            const int dropped = cache.drop_duplicate(requests[i].layer, source[i]);
            ram_drops += dropped > 0; ram_drop_blocked += dropped < 0;
        }
        rc = residency->release(context, &lease, error, sizeof(error)); check(rc, error);
    }
    void cancel() {
        if (!context) { return; }
        char error[512] = {};
        if (api->cancel(context, error, sizeof(error))) {
            std::fprintf(stderr, "[kimi_vram_tier] failed to drain GPU work: %s; aborting\n", error);
            std::abort();
        }
        ram_leases.clear(); // caller releases RAM only after this drain
        deferred.clear(); upload_queued.clear();
        prefill_open = false;
    }
};
