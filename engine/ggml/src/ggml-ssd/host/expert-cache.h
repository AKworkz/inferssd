// SPDX-License-Identifier: Apache-2.0
#pragma once
// RAM expert-cache owner. Payload arena, budget admission, first touch,
// retention, reclaim and leases live here; executors borrow ranges for compute.
#include "arbiter.h"
#include "cache-stats.h"
#include "expert-request.h"
#include "../gpu/kimi-vram-tier-api.h"
#include "../platform/file.h"
#include "../platform/memory.h"
#include "ggml.h"
#include <algorithm>
#include <atomic>
#include <climits>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <map>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace ssd { namespace host {
// The layer ID of a routed layer's up experts ("blk.N.ffn_up_exps.weight"), or -1 for any other tensor.
inline int routed_layer(const ggml_tensor * up) {
    int layer = -1, consumed = 0;
    if (!up || std::sscanf(up->name, "blk.%d.ffn_up_exps.weight%n", &layer, &consumed) != 1 || !consumed ||
            up->name[consumed] || layer < 0) {
        return -1;
    }
    return layer;
}
struct expert_cache_options {
    context_client * host = nullptr;
    client_config budget;
    bool require_shared = true;
    // A GPU tier keeps experts in this DRAM (a unified-memory GPU with a nonzero vram.budget): an automatic RAM
    // cache then keeps only the slots the next-layer prefetch lands in.
    bool gpu_tier = false;
    std::string cache_policy = "window", cache_allocation = "priority";
    bool cache_keep = false, cache_prefill_read = false, cache_elastic = false, cache_partial_reclaim = false;
    size_t cache_prefill_slots = 0;
    std::map<int32_t, size_t> cache_layer_quotas;
};
namespace detail {
using cache_clock = std::chrono::steady_clock;
inline double cache_elapsed_us(cache_clock::time_point start) {
    return std::chrono::duration<double, std::micro>(cache_clock::now() - start).count();
}
inline std::runtime_error cache_sys_error(const char * operation) {
    return std::runtime_error(std::string(operation) + ": " + std::strerror(errno));
}
template<class T> inline T cache_host_value(ssd::result<T> result) {
    if (!result) { throw std::runtime_error(result.error); }
    return std::move(result.value);
}
inline void cache_host_value(ssd::result<void> result) {
    if (!result) { throw std::runtime_error(result.error); }
}
} // namespace detail
inline const char * binding_term(const ssd::host::room & r) {
    return r.cgroup_shortage && r.system_shortage ? "both" : r.cgroup_shortage ? "cgroup" : r.system_shortage ? "system" : "none";
}
inline void touch_pages(void * address, size_t bytes, size_t page) {
    if (!bytes) { return; }
    auto * data = static_cast<volatile unsigned char *>(address);
    for (size_t at = 0; at < bytes;) {
        data[at] = 0;
        if (bytes - at <= page) { break; }
        at += page;
    }
    data[bytes - 1] = 0;
}
struct cache_client_view {
    std::shared_ptr<ssd::host::process_arbiter> test_owner;
    std::unique_ptr<ssd::host::context_client> test_client;
    ssd::host::context_client * client = nullptr;
    explicit cache_client_view(context_client * borrowed, const client_config & config, bool require_shared) : client(borrowed) {
        if (!client) {
            if (require_shared) { throw std::runtime_error("missing process host budget client"); }
            test_owner = std::make_shared<ssd::host::process_arbiter>();
            test_client = detail::cache_host_value(test_owner->make_client()); client = test_client.get();
        }
        detail::cache_host_value(client->configure(config));
    }
    ssd::host::snapshot state() const { return detail::cache_host_value(client->state()); }
    ssd::host::room measure(uint64_t extra, bool full = false) const { return detail::cache_host_value(client->measure(extra, full)); }
    ssd::host::room project_decode_capacity(uint64_t extra, bool full = false) const {
        return detail::cache_host_value(client->project_decode_capacity(extra, full));
    }
    uint64_t floor() const { return state().floor; }
    uint64_t dram_bytes(uint64_t host, uint64_t device) const { return detail::cache_host_value(client->dram_bytes(host, device)); }
    void set_prefill_pages(uint64_t pages) { detail::cache_host_value(client->set_prefill_pages(pages)); }
    bool update_file_reserve() { return detail::cache_host_value(client->update_file_reserve()); }
    void begin_interval(uint64_t held) { detail::cache_host_value(client->begin_interval(held)); }
    bool end_interval(bool consecutive, uint64_t held, uint64_t & unknown) {
        return detail::cache_host_value(client->end_interval(consecutive, held, unknown));
    }
    ssd::host::room make_room(uint64_t extra) { return detail::cache_host_value(client->make_room(extra)); }
    bool memory_available_at_full_boost(uint64_t & available, uint64_t & headroom) const {
        const auto r = measure(0, true); available = r.available; headroom = r.boost_headroom; return r.system;
    }
};
struct expert_cache {
    struct source_key {
        uint64_t device, inode, size, offset, bytes;
        int64_t mtime_s, mtime_ns, ctime_s, ctime_ns;
        uint64_t inode_hi;
        bool operator==(const source_key & o) const {
            return device == o.device && inode == o.inode && inode_hi == o.inode_hi && size == o.size &&
                offset == o.offset && bytes == o.bytes && mtime_s == o.mtime_s && mtime_ns == o.mtime_ns &&
                ctime_s == o.ctime_s && ctime_ns == o.ctime_ns;
        }
    };
    struct entry {
        source_key source[3] = {};
        int32_t id = -1;
        size_t layer_index = 0;
        uint64_t last_touch = 0;
        bool valid = false, reserved = false, resident = false, enabled = false, was_released = false, protected_now = false;
        bool prefetched = false; // read ahead by the next-layer prefetch and not requested yet
        bool inflight = false;   // a prefetch read writes into the slot; nothing else may take it
        bool lent = false;       // a prefill read-ahead borrowed it for a later layer (prefetch_borrow)
        uint32_t leases = 0;
        uint64_t generation = 0; // cache generation at a prefetch reservation
    };
    struct layer {
        int32_t id;
        // its experts' ggml types (up and gate, down), recorded at load from every routed layer's tensors; -1 before
        int32_t up_type = -1, down_type = -1;
        size_t begin = 0, count = 0;
        uint64_t token = 0;
        bool seen = false;
        std::map<int32_t, uint64_t> frequency;
        std::map<int32_t, double> count_frequency;
        uint64_t count_step = 0;
    };
    std::vector<layer> layers;
    std::vector<entry> entries;
    kimi_cpu_sink_cache_stats stats = {};
    char * arena = nullptr;
    size_t stride = 0, role_bytes[3] = {};
    int quota = 8;
    uint64_t epoch = 0, token = 0;
    bool active = false, has_token = false, paused = false, elastic_prefill = false;
    enum policy_type { window, lru, lfu, decay, count_decay2 } policy = window;
    // Room for the cache comes from the host-memory ledger, which every other host allocation asks too. Its cgroup
    // reserve is CACHE_RESERVE_BYTES; opt-in CACHE_RECLAIMABLE_FILE counts file-backed pages as cgroup headroom.
    cache_client_view ledger;
    // The GPU tier keeps the cached experts in this DRAM (expert_cache_options::gpu_tier on unified memory).
    const bool tier_caches;
    // The last answer check_room got and the rejections per short term, for the receipts: on a discrete GPU the
    // cgroup term is meant to bind, so a system-term rejection there points at memory the cgroup does not see.
    mutable ssd::host::room last_room;
    uint64_t cgroup_rejections = 0, system_rejections = 0;
    void count_rejection() {
        cgroup_rejections += last_room.cgroup_shortage != 0;
        system_rejections += last_room.system_shortage != 0;
    }
    using room_state = ssd::host::room_state;
    static constexpr room_state room_ok = ssd::host::room_ok, room_pressure = ssd::host::room_pressure,
        room_unreadable = ssd::host::room_unreadable;
    bool decode_projection = false;
    bool keep_across_requests = false;
    bool adopt = false; // the arena was made at load (reserve_at_load): the first decode keeps it, whatever its epoch
    // An automatic budget's prices off unified memory (geometry): a slot, and the metadata room kept beside the slots.
    uint64_t slot_cost_bytes = 0, metadata_bytes = 0;
    bool prefill_read = false;
    bool elastic = false;
    bool partial_reclaim = false;
    size_t prefill_slots = 0;
    enum allocation_type { priority_order, uniform, global } allocation = priority_order;
    uint64_t clock = 0, age_steps = 0;
    uint64_t prefetch_hits = 0, prefetch_unused = 0;
    // CACHE_BYTES unset: the cache is sized from this machine when the first decode token learns the slot size.
    static constexpr uint64_t auto_budget = UINT64_MAX;
    int staging_per_layer = 1; // slots per layer the next-layer prefetch lands in
    // The slots a prefill read-ahead borrowed (prefetch_borrow), lent out by their own layers until return_borrowed.
    std::vector<size_t> lent;
    // Changes whenever release() drops every entry. A hook lets deferred GPU uploads that borrow entries finish first.
    uint64_t generation = 0;
    void (*before_release)(void *) = nullptr;
    void * before_release_arg = nullptr;
    void (*before_unmap)(void *) = nullptr; // waits for prefetch reads into the arena
    void * before_unmap_arg = nullptr;
    // Before slots' pages are discarded or the arena released: entries [first, first + count). The decode
    // registrar's page-locks must go first, or the device would keep the old pages.
    void (*before_discard)(void *, size_t first, size_t count) = nullptr;
    void * before_discard_arg = nullptr;
    // Optional fixed geometry learned from earlier requests. This only assigns
    // physical RAM slots; source identity, admission and leases stay unchanged.
    std::map<int32_t, size_t> layer_quotas;
    const uint64_t page_size = uint64_t(ssd::platform::page_size());

    static uint64_t add(uint64_t a, uint64_t b) {
        if (b > UINT64_MAX - a) { throw std::runtime_error("CPU sink cache size addition overflow"); }
        return a + b;
    }
    static uint64_t multiply(uint64_t a, uint64_t b) {
        if (a && b > UINT64_MAX / a) { throw std::runtime_error("CPU sink cache size multiplication overflow"); }
        return a * b;
    }
    static uint64_t ceil_div(uint64_t a, uint64_t b) { return a / b + (a % b != 0); }
    uint64_t page_tables(uint64_t bytes) const {
        if (!bytes) { return 0; }
        // Five levels, with an extra boundary table at each level. Summing
        // this independently for each new slot overcounts shared tables.
        uint64_t pages = ceil_div(bytes, page_size), tables = 1;
        for (int level = 0; level < 4; ++level) {
            pages = add(ceil_div(pages, page_size / sizeof(uint64_t)), 1);
            tables = add(tables, pages);
        }
        return multiply(tables, page_size);
    }
    uint64_t allocation_cost(uint64_t bytes) const {
        if (!bytes) { return 0; }
        // std::vector/map storage belongs to libc. Allow allocator headers,
        // unaligned ends and the heap's largest page (Linux THP) at both boundaries. Payload
        // itself has no huge pages (arena_no_hugepage) and uses the much smaller ordinary-page bound.
        const uint64_t huge = uint64_t(ssd::platform::heap_page_size());
        const uint64_t resident = add(multiply(ceil_div(add(bytes, 64), huge), huge), huge);
        return add(add(resident, page_tables(resident)), 64 * 1024);
    }
    bool guard(uint64_t extra) {
        stats.last_guard_extra_bytes = extra;
        stats.peak_guard_extra_bytes = std::max(stats.peak_guard_extra_bytes, extra);
        return check_room(extra, nullptr) == room_ok;
    }
    // The cache's own growth: the context's reclaimer gives back colder slots for it (requester::ram_cache).
    ssd::result<ssd::host::allocation_ticket> take_guard(uint64_t extra) {
        stats.last_guard_extra_bytes = extra;
        stats.peak_guard_extra_bytes = std::max(stats.peak_guard_extra_bytes, extra);
        auto ticket = ledger.client->take(extra, 0, ssd::host::requester::ram_cache, true, ssd::host::allocation_class::tracked);
        last_room = ticket.value.observation();
        return ticket;
    }
    // term: the ledger term that was short when the pause is a rejection for room ("none" otherwise).
    void pause(std::vector<expert_request> & requests, kimi_cpu_sink_stats & current, const char * reason,
               const char * term = "none") {
        // No I/O job or read_target has been published yet. In particular,
        // hits planned before the pressure check must become ring misses.
        for (auto & r : requests) { r.cache_hit = r.cache_admit = -1; }
        current.cache_hits = current.cache_new_slot_bytes = 0;
        release();
        std::vector<entry>().swap(entries);
        stride = 0; std::fill(role_bytes, role_bytes + 3, 0);
        stats.capacity_bytes = stats.enabled_bytes = 0;
        paused = true; ++stats.pressure_stops;
        std::fprintf(stderr, "[kimi_cpu_sink_cache] paused=1 reason=%s term=%s reserve_bytes=%llu extra_bytes=%llu\n",
                reason, term, (unsigned long long) ledger.state().reserve,
                (unsigned long long) stats.last_guard_extra_bytes);
    }

    const char * policy_name() const {
        switch (policy) {
            case window: return "window";
            case lru:    return "lru";
            case lfu:    return "lfu";
            case decay:  return "decay";
            case count_decay2: return "count_decay2";
        }
        return "unknown";
    }
    const char * allocation_name() const {
        switch (allocation) {
            case priority_order: return "priority";
            case uniform:        return "uniform";
            case global:         return "global";
        }
        return "unknown";
    }

    explicit expert_cache(const expert_cache_options & configuration) : ledger(configuration.host, configuration.budget, configuration.require_shared), tier_caches(configuration.budget.unified && configuration.gpu_tier) {
        if (page_size < 4096 || page_size > 65536 || (page_size & (page_size - 1))) {
            throw std::runtime_error("CPU sink cache requires a supported system page size");
        }
        const auto & p = configuration;
        if (p.cache_policy == "window") { policy = window; }
        else if (p.cache_policy == "lru") { policy = lru; }
        else if (p.cache_policy == "lfu") { policy = lfu; }
        else if (p.cache_policy == "decay") { policy = decay; }
        else if (p.cache_policy == "count_decay2") { policy = count_decay2; }
        else { throw std::runtime_error("invalid explicit test cache policy"); }
        if (p.cache_allocation == "priority") { allocation = priority_order; }
        else if (p.cache_allocation == "uniform") { allocation = uniform; }
        else if (p.cache_allocation == "global") { allocation = global; }
        else { throw std::runtime_error("invalid explicit test cache allocation"); }
        keep_across_requests = p.cache_keep;
        prefill_read = p.cache_prefill_read;
        elastic = p.cache_elastic;
        partial_reclaim = p.cache_partial_reclaim;
        prefill_slots = p.cache_prefill_slots;
        layer_quotas = p.cache_layer_quotas;
        if (keep_across_requests && policy == window) {
            throw std::runtime_error("cache retention requires a persistent policy");
        }
        if (prefill_read && !keep_across_requests) {
            throw std::runtime_error("read-only prefill cache requires retention");
        }
        if (policy == window && allocation == global) {
            throw std::runtime_error("window cache cannot use global allocation");
        }
        if ((elastic || partial_reclaim) && (!keep_across_requests || policy == window || allocation != uniform)) {
            throw std::runtime_error("cache reclaim requires retention, a persistent policy, and uniform allocation");
        }
        if ((elastic || partial_reclaim) && !layer_quotas.empty()) {
            throw std::runtime_error("cache reclaim cannot use explicit layer quotas");
        }
        std::fprintf(stderr, "[kimi_host_ledger] unified=%d reserve=%llu margin=%llu explicit_reserve=%llu reclaimable_file=%d cgroup=%s\n",
                int(ledger.state().unified), (unsigned long long) ledger.state().reserve, (unsigned long long) ledger.state().margin,
                (unsigned long long) ledger.state().explicit_reserve, int(ledger.state().reclaimable_file),
                ledger.state().cgroup.empty() ? "-" : ledger.state().cgroup.c_str());
    }
    ~expert_cache() { release(); }
    void release() {
        if (arena && before_release) { before_release(before_release_arg); }
        if (arena && before_unmap) { before_unmap(before_unmap_arg); }
        if (arena && before_discard) { before_discard(before_discard_arg, 0, entries.size()); }
        ++generation;
        if (arena) {
            ssd::platform::arena_release(arena, stats.mapped_bytes);
            arena = nullptr;
            ++stats.clears;
        }
        stats.mapped_bytes = stats.committed_bytes = stats.enabled_bytes = stats.valid_bytes = stats.page_table_bytes = 0;
        elastic_prefill = adopt = false;
        // A room-sized cache comes back at the size it last had (capacity_bytes); every other cache at all its slots.
        const bool sized = room_sized();
        for (auto & e : entries) { e = {}; e.enabled = !sized; }
        lent.clear();
        for (size_t layer_index = 0; layer_index < layers.size(); ++layer_index) {
            auto & l = layers[layer_index];
            l.seen = false; l.frequency.clear(); l.count_frequency.clear(); l.count_step = 0;
            if (allocation != global) {
                for (size_t index = l.begin; index < l.begin + l.count && index < entries.size(); ++index) {
                    entries[index].layer_index = layer_index;
                }
            }
        }
        if (!entries.empty() && !sized) { stats.enabled_bytes = stats.capacity_bytes; }
        if (!entries.empty() && sized && stride) { enable_to(size_t(stats.capacity_bytes / stride)); }
        clock = age_steps = 0;
    }
    void configure(uint64_t budget, const int32_t * priority, int n, int per_layer) {
        if (budget > std::numeric_limits<size_t>::max() ||
            n < 0 || n > 256 || (n && !priority) ||
            (budget && !n) || per_layer < 1 || per_layer > 896) {
            throw std::runtime_error("invalid CPU sink cache budget, layers or quota");
        }
        std::vector<layer> next;
        next.reserve(size_t(n)); // at most 256 small configuration records
        for (int i = 0; i < n; ++i) {
            if (priority[i] < 0 || priority[i] > 1023 || std::any_of(next.begin(), next.end(),
                    [&](const layer & l) { return l.id == priority[i]; })) {
                throw std::runtime_error("CPU sink cache layers must be distinct integers in [0,1023]");
            }
            layer item; item.id = priority[i]; next.push_back(item);
        }
        if (!layer_quotas.empty()) {
            if (layer_quotas.size() != next.size()) {
                throw std::runtime_error("CACHE_LAYER_QUOTAS must name every configured layer exactly once");
            }
            for (const auto & l : next) {
                const auto item = layer_quotas.find(l.id);
                if (item == layer_quotas.end() || item->second > size_t(per_layer)) {
                    throw std::runtime_error("CACHE_LAYER_QUOTAS contains an unknown layer or exceeds CACHE_MAX_PER_LAYER");
                }
            }
        }
        release();
        layers = std::move(next);
        std::vector<entry>().swap(entries);
        stats = {}; stats.budget_bytes = budget;
        stride = 0; std::fill(role_bytes, role_bytes + 3, 0);
        quota = per_layer; active = has_token = paused = elastic_prefill = false;
        if (ledger.state().reclaimable_file) {
            std::fprintf(stderr, "[kimi_cpu_sink_cache] reclaimable_file=1 active_file_counts_as_headroom=1 inactive_file_counts_as_headroom=1\n");
        }
        if (!layer_quotas.empty() || ledger.state().reserve != 2ull * 1024 * 1024 * 1024) {
            std::fprintf(stderr, "[kimi_cpu_sink_cache] fixed_layer_quotas=%zu reserve_bytes=%llu\n",
                    layer_quotas.size(), (unsigned long long) ledger.state().reserve);
        }
        if (elastic) {
            std::fprintf(stderr,
                    "[kimi_cpu_sink_cache] elastic=1 prefill_slots=%zu reserve_bytes=%llu mapping=retained\n",
                    prefill_slots, (unsigned long long) ledger.state().reserve);
        }
        if (partial_reclaim) {
            std::fprintf(stderr,
                    "[kimi_cpu_sink_cache] partial_reclaim=1 reserve_bytes=%llu mapping=retained_on_known_pressure\n",
                    (unsigned long long) ledger.state().reserve);
        }
    }
    void begin(uint64_t next_epoch, uint64_t next_token, bool decode) {
        if (!decode) {
            // No run can be in flight under run_mutex. The optional read-only
            // prefill path may lease retained valid entries, but never trains,
            // admits or evicts here. Recheck headroom before each grouped
            // workspace/stream; an error or explicit configure still unmaps.
            if (elastic && arena) {
                shrink_to(prefill_slots, false, "prefill_phase", true);
                elastic_prefill = true;
                const bool room = partial_reclaim
                        ? reclaim_for_guard(0, "prefill_headroom") : guard(0);
                if (room) {
                    std::fprintf(stderr,
                            "[kimi_cpu_sink_cache] retained_prefill=1 elastic=1 enabled_bytes=%llu resident_bytes=%llu valid_bytes=%llu\n",
                            (unsigned long long) stats.enabled_bytes,
                            (unsigned long long) stats.committed_bytes,
                            (unsigned long long) stats.valid_bytes);
                    for (auto & l : layers) { l.seen = false; }
                } else {
                    ++stats.pressure_stops; count_rejection();
                    std::fprintf(stderr, "[kimi_cpu_sink_cache] paused=1 reason=prefill_headroom term=%s reserve_bytes=%llu\n",
                            binding_term(last_room), (unsigned long long) ledger.state().reserve);
                    release();
                }
            } else if (keep_across_requests && arena &&
                    (partial_reclaim ? reclaim_for_guard(0, "prefill_headroom") : guard(0))) {
                if (active) {
                    std::fprintf(stderr, "[kimi_cpu_sink_cache] retained_prefill=1 valid_bytes=%llu\n",
                            (unsigned long long) stats.valid_bytes);
                }
                for (auto & l : layers) { l.seen = false; }
            } else {
                if (keep_across_requests && arena) {
                    ++stats.pressure_stops; count_rejection();
                    std::fprintf(stderr, "[kimi_cpu_sink_cache] paused=1 reason=prefill_headroom term=%s reserve_bytes=%llu\n",
                            binding_term(last_room), (unsigned long long) ledger.state().reserve);
                }
                release();
            }
            active = has_token = paused = false;
            return;
        }
        return_borrowed(-1); // decode's layers keep their own slots
        // A retained mapping is bound to the model/context epoch even after
        // prefill cleared has_token. Per-entry file identity keys also reject fd reuse.
        if (keep_across_requests && arena && next_epoch != epoch && !adopt) { release(); }
        // The first decode token after a prefill: the ledger no longer holds the prefill's pages, and the slots the
        // prefill gave back come back as far as the room allows now.
        if (!has_token) {
            ledger.set_prefill_pages(0);
            regrow(adopt ? "load" : "prefill");
        }
        adopt = false;
        if (has_token && (next_epoch != epoch || token == UINT64_MAX || next_token != token + 1)) { release(); }
        if (elastic) { grow_decode(); }
        epoch = next_epoch; token = next_token; active = has_token = true;
        if (policy == decay || policy == count_decay2) { ++age_steps; }
        if (policy == decay && age_steps % 32 == 0) {
            for (auto & l : layers) {
                for (auto & item : l.frequency) { item.second >>= 1; }
            }
        }
    }
    // Room for the cache's own admission: both terms (room.h), and only under a readable finite envelope. full_boost:
    // the system term at the kernel's full watermark boost, as the arbiter's take and make_room measure it.
    room_state check_room(uint64_t extra, uint64_t * shortage, uint64_t * surplus = nullptr, bool full_boost = false) const {
        if (shortage) { *shortage = 0; }
        if (surplus) { *surplus = 0; }
        const auto r = decode_projection ? ledger.project_decode_capacity(extra, full_boost) : ledger.measure(extra, full_boost);
        last_room = r;
        if (r.cgroup == room_unreadable) { return room_unreadable; }
        if (shortage) { *shortage = r.shortage(); }
        if (surplus) { *surplus = r.surplus(); }
        return r.fits() ? room_ok : room_pressure;
    }
    std::pair<uint64_t, double> score(size_t layer_id, int32_t expert) const {
        if (policy == count_decay2) {
            const auto & item = layers.at(layer_id);
            const auto found = item.count_frequency.find(expert);
            const double value = found == item.count_frequency.end() ? 0.0 : found->second;
            return { 0, value * std::exp2(-double(age_steps - item.count_step) / 2.0) };
        }
        const auto & counts = layers.at(layer_id).frequency;
        const auto found = counts.find(expert);
        return { found == counts.end() ? uint64_t(0) : found->second, 0.0 };
    }
    bool colder(size_t left, size_t right) const {
        const entry & a = entries[left];
        const entry & b = entries[right];
        if (a.valid != b.valid) { return !a.valid; }
        const auto a_score = score(a.layer_index, a.id);
        const auto b_score = score(b.layer_index, b.id);
        if (policy != lru && a_score != b_score) { return a_score < b_score; }
        if (a.last_touch != b.last_touch) { return a.last_touch < b.last_touch; }
        return left < right;
    }
    size_t enabled_slots() const {
        return size_t(std::count_if(entries.begin(), entries.end(), [](const entry & e) { return e.enabled; }));
    }
    size_t shrink_to(size_t target, bool guard_reclaim, const char * reason, bool strict) {
        if ((!elastic && !partial_reclaim) || target > entries.size() || !stride || stride % page_size) {
            throw std::runtime_error("invalid CPU sink cache shrink");
        }
        const size_t before = enabled_slots();
        if (target >= before) { return before; }
        const uint64_t started = ggml_time_us();
        const uint64_t released_before = stats.released_bytes;
        for (size_t layer_index = 0; layer_index < layers.size(); ++layer_index) {
            const auto & item = layers[layer_index];
            const size_t wanted = target / layers.size() + (layer_index < target % layers.size());
            if (wanted > item.count) { throw std::runtime_error("cache shrink target exceeds a layer's maximum quota"); }
            size_t have = 0;
            std::vector<size_t> candidates;
            candidates.reserve(item.count);
            for (size_t index = item.begin; index < item.begin + item.count; ++index) {
                entry & e = entries[index];
                if (!e.enabled) { continue; }
                ++have;
                if (!e.leases && !e.reserved && !e.protected_now && !e.inflight) { candidates.push_back(index); }
            }
            if (have <= wanted) { continue; }
            std::sort(candidates.begin(), candidates.end(), [&](size_t a, size_t b) { return colder(a, b); });
            const size_t count = std::min(have - wanted, candidates.size());
            for (size_t offset = 0; offset < count; ++offset) {
                entry & e = entries[candidates[offset]];
                if (e.resident) {
                    if (arena && before_discard) { before_discard(before_discard_arg, candidates[offset], 1); }
                    if (!arena || !ssd::platform::arena_discard(arena + candidates[offset] * stride, stride)) {
                        const auto error = detail::cache_sys_error("CPU sink cache slot discard");
                        release();
                        throw error;
                    }
                    e.resident = false;
                    e.was_released = true;
                    stats.committed_bytes -= stride;
                    stats.released_bytes = add(stats.released_bytes, stride);
                }
                if (e.valid) {
                    e.valid = false;
                    stats.valid_bytes -= stride;
                    ++stats.evictions;
                }
                e.enabled = false;
                stats.enabled_bytes -= stride;
            }
        }
        const size_t after = enabled_slots();
        if (strict && after != target) {
            throw std::runtime_error("elastic CPU sink cache shrink found a busy slot at a phase boundary");
        }
        if (after < before) {
            ++stats.shrink_ops;
            if (guard_reclaim) { ++stats.guard_reclaims; }
            stats.shrink_us = add(stats.shrink_us, ggml_time_us() - started);
            std::fprintf(stderr,
                    "[kimi_cpu_sink_cache] cache_shrink=1 reason=%s target_slots=%zu before_slots=%zu after_slots=%zu released_bytes=%llu valid_bytes=%llu\n",
                    reason, target, before, after,
                    (unsigned long long) (stats.released_bytes - released_before),
                    (unsigned long long) stats.valid_bytes);
        }
        return after;
    }
    void grow_decode() {
        if (!elastic || !elastic_prefill || entries.empty()) { elastic_prefill = false; return; }
        const size_t before = enabled_slots();
        for (auto & e : entries) { e.enabled = true; }
        stats.enabled_bytes = multiply(entries.size(), stride);
        elastic_prefill = false;
        if (before != entries.size()) {
            ++stats.phase_grows;
            std::fprintf(stderr,
                    "[kimi_cpu_sink_cache] elastic_grow=1 before_slots=%zu after_slots=%zu resident_bytes=%llu\n",
                    before, entries.size(), (unsigned long long) stats.committed_bytes);
        }
    }
    // Slots given back until the room for `extra` fits; full_boost for the host arbiter's admissions (the context's
    // reclaimer), which measure at the full watermark boost: stopping at MemAvailable as it is would leave them short by
    // the boost's headroom.
    bool reclaim_for_guard(uint64_t extra, const char * reason, bool full_boost = false) {
        stats.last_guard_extra_bytes = extra;
        stats.peak_guard_extra_bytes = std::max(stats.peak_guard_extra_bytes, extra);
        uint64_t shortage = 0;
        room_state state = check_room(extra, &shortage, nullptr, full_boost);
        if (state == room_ok) { return true; }
        if (state == room_unreadable || !partial_reclaim || !arena || !stride) { return false; }
        size_t before = enabled_slots();
        while (state == room_pressure && before) {
            const size_t slots = size_t(std::min<uint64_t>(before, ceil_div(shortage, stride)));
            const size_t after = shrink_to(before - std::max<size_t>(slots, 1), true, reason, false);
            if (after >= before) { break; }
            before = after;
            state = check_room(extra, &shortage, nullptr, full_boost);
        }
        return state == room_ok;
    }
    void reject_admission(std::vector<expert_request> & requests, kimi_cpu_sink_stats & current, const char * reason) {
        count_rejection();
        if (!partial_reclaim) { pause(requests, current, reason, binding_term(last_room)); return; }
        for (auto & r : requests) {
            if (r.cache_admit >= 0) { entries[size_t(r.cache_admit)].reserved = false; }
            r.cache_hit = r.cache_admit = -1;
        }
        for (auto & e : entries) { e.protected_now = false; }
        current.cache_hits = current.cache_new_slot_bytes = 0;
        ++stats.pressure_stops;
        std::fprintf(stderr,
                "[kimi_cpu_sink_cache] admission_rejected=1 reason=%s term=%s mapping_retained=1 reserve_bytes=%llu extra_bytes=%llu enabled_slots=%zu\n",
                reason, binding_term(last_room), (unsigned long long) ledger.state().reserve,
                (unsigned long long) stats.last_guard_extra_bytes, enabled_slots());
    }
    static source_key key(const kimi_cpu_sink_source & s) {
        const auto v = kimi_vram_source_identity(s.file, s.offset + s.data_offset, s.payload_nbytes ? s.payload_nbytes : s.nbytes);
        return { v.device, v.inode, v.size, v.offset, v.bytes, v.mtime_s, v.mtime_ns, v.ctime_s, v.ctime_ns, v.inode_hi };
    }
    static bool matches(const entry & e, const expert_request & r) {
        for (int role = 0; role < 3; ++role) { if (!(e.source[role] == key(r.role[role]))) { return false; } }
        return true;
    }
    bool geometry(const size_t * requested, const ggml_tensor * tensor, size_t max_requests) {
        // Native GGUF roles can differ only in page-prefix padding across
        // layers. Keep the largest buffers instead of clearing the cache as
        // those physical read lengths alternate.
        const size_t bytes[3] = { std::max(requested[0], role_bytes[0]),
            std::max(requested[1], role_bytes[1]), std::max(requested[2], role_bytes[2]) };
        if (std::equal(bytes, bytes + 3, role_bytes)) { return true; }
        uint64_t sum = add(add(bytes[0], bytes[1]), bytes[2]);
        // A slot that gives its own pages back starts on a page. Roles pad to 4 KiB, which is less than the 16 KiB
        // pages of Apple silicon.
        if (gives_room_back() && sum) { sum = multiply(ceil_div(sum, page_size), page_size); }
        if (!sum || sum > std::numeric_limits<size_t>::max()) { throw std::runtime_error("invalid CPU sink cache stride"); }
        const size_t next_stride = size_t(sum);
        uint64_t budget = stats.budget_bytes;
        if (budget == auto_budget) {
            // With a GPU tier on a unified-memory GPU the tier keeps the same experts in the same DRAM and
            // computes them faster, so the RAM cache keeps only the slots the next-layer prefetch lands in.
            // Otherwise, a unified GPU without a tier included, it takes what the cgroup guard would admit now.
            // The receipt names both terms: on a discrete GPU the cgroup term should be the smaller one.
            uint64_t surplus = 0;
            const room_state state = check_room(0, nullptr, &surplus);
            budget = tier_caches
                    ? multiply(multiply(layers.size(), uint64_t(std::max(1, staging_per_layer))), next_stride)
                    : (state == room_ok ? surplus : 0);
            const auto & r = last_room;
            // binding: the term that sizes the cache, or the short one when the room is already taken
            std::fprintf(stderr, "[kimi_cpu_sink_cache] auto_budget=%llu unified=%d tier=%d slot_bytes=%zu cgroup_surplus=%llu system_surplus=%llu available=%llu floor=%llu binding=%s\n",
                    (unsigned long long) budget, int(ledger.state().unified), int(tier_caches), next_stride,
                    (unsigned long long) (r.cgroup == room_unreadable ? 0 : r.cgroup_surplus), (unsigned long long) r.system_surplus,
                    (unsigned long long) r.available, (unsigned long long) r.floor,
                    state == room_unreadable ? "unreadable" : state == room_pressure ? binding_term(r) :
                    !r.system || r.cgroup_surplus <= r.system_surplus ? "cgroup" : "system");
        }
        // The slots the cache may ever hold: every expert of every layer up to the per-layer quota. A cache sized
        // from the room (room_sized) keeps that many slots' metadata and address space, holds `fits` of them now and
        // grows into the rest as the room allows at later requests (regrow); every other cache holds all of its slots.
        const size_t limit = static_cast<size_t>(multiply(layers.size(), uint64_t(quota)));
        const bool sized = room_sized();
        size_t count = static_cast<size_t>(std::min<uint64_t>(budget / next_stride, limit)), fits = count;
        if (stats.budget_bytes == auto_budget && !tier_caches) {
            // Price the same page rounding/tables and allocation guards used
            // by admission. Payload-only division can spend their headroom,
            // making an automatically chosen arena pause before it fills.
            const uint64_t pages = add(ceil_div(next_stride, page_size), next_stride % page_size != 0);
            const uint64_t payload = multiply(pages, page_size);
            const uint64_t slot_cost = add(payload, page_tables(payload));
            const bool frequencies = policy == count_decay2 || policy == lfu || policy == decay;
            const uint64_t node = sizeof(std::pair<const int32_t, uint64_t>) + 4 * sizeof(void *);
            const uint64_t node_storage = frequencies
                    ? allocation_cost(multiply(multiply(layers.size(), uint64_t(tensor->ne[2])), node)) : 0;
            const uint64_t node_guard = frequencies ? multiply(max_requests, allocation_cost(node)) : 0;
            // a room-sized cache's metadata covers every slot it may hold, whatever it holds now
            const auto cost = [&](size_t slots) {
                if (!slots) { return uint64_t(0); }
                return add(add(multiply(slots, slot_cost), allocation_cost(multiply(sized ? limit : slots, sizeof(entry)))),
                           add(add(node_storage, node_guard), 64 * 1024));
            };
            size_t low = 0, high = count;
            while (low < high) {
                const size_t middle = low + (high - low + 1) / 2;
                if (cost(middle) <= budget) { low = middle; }
                else { high = middle - 1; }
            }
            fits = count = low;
            if (sized) { count = limit; }
            slot_cost_bytes = slot_cost;
            metadata_bytes = add(add(node_storage, node_guard), 64 * 1024);
            std::fprintf(stderr,
                    "[kimi_cpu_sink_cache] auto_capacity_bytes=%llu capacity_cost_bytes=%llu surplus_bytes=%llu slot_cost_bytes=%llu metadata_node_storage_bytes=%llu metadata_guard_bytes=%llu limit_slots=%zu\n",
                    (unsigned long long) multiply(fits, next_stride), (unsigned long long) cost(fits),
                    (unsigned long long) budget, (unsigned long long) slot_cost,
                    (unsigned long long) node_storage, (unsigned long long) node_guard, sized ? limit : fits);
        }
        const uint64_t capacity = multiply(fits, next_stride);
        if (elastic && prefill_slots > count) {
            throw std::runtime_error("CACHE_PREFILL_SLOTS exceeds the decode cache slot count");
        }
        if (!layer_quotas.empty()) {
            uint64_t total = 0;
            for (const auto & item : layer_quotas) { total = add(total, item.second); }
            if (total != count) {
                throw std::runtime_error("CACHE_LAYER_QUOTAS sum must equal the cache budget's physical slot count");
            }
        }
        if (count > size_t(INT_MAX) || count > entries.max_size() ||
                multiply(count, next_stride) > uint64_t(std::numeric_limits<ptrdiff_t>::max())) {
            throw std::runtime_error("CPU sink cache geometry exceeds addressable storage");
        }
        // Check before vector value-initialization can touch metadata pages.
        // Existing allocations are already included in memory.current.
        auto metadata_ticket = take_guard(allocation_cost(multiply(count, sizeof(entry))));
        if (!metadata_ticket && sized) {
            // The room holds not even the metadata now: no cache until a later request finds room (regrow).
            release();
            std::vector<entry>().swap(entries);
            for (auto & l : layers) { l.begin = l.count = 0; }
            std::copy(bytes, bytes + 3, role_bytes);
            stride = next_stride;
            stats.capacity_bytes = stats.enabled_bytes = 0;
            return true;
        }
        if (!metadata_ticket) { return false; }
        std::vector<entry> next(count);
        metadata_ticket.value.commit();
        release();
        // Geometry is first learned during the first decode token's prepare.
        age_steps = active ? 1 : 0;
        std::copy(bytes, bytes + 3, role_bytes);
        stride = next_stride;
        entries = std::move(next);
        size_t assigned = 0;
        for (size_t i = 0; i < layers.size(); ++i) {
            auto & l = layers[i];
            l.begin = assigned;
            l.count = allocation == uniform ? count / layers.size() + (i < count % layers.size()) :
                std::min(size_t(quota), count - assigned);
            if (!layer_quotas.empty()) { l.count = layer_quotas.at(l.id); }
            if (allocation == global) { l.begin = 0; l.count = count; }
            if (allocation == global) { continue; }
            for (size_t index = l.begin; index < l.begin + l.count; ++index) {
                entries[index].layer_index = i;
            }
            assigned += l.count;
        }
        stats.capacity_bytes = capacity;
        if (sized) {
            for (auto & e : entries) { e.enabled = false; }
            stats.enabled_bytes = 0;
            enable_to(fits);
        } else {
            for (auto & e : entries) { e.enabled = true; }
            stats.enabled_bytes = capacity;
        }
        return true;
    }
    // A cache sized from the room (an automatic budget for the RAM cache itself: no unified GPU tier, uniform
    // allocation, partial reclaim or none): its arena may hold every slot, and how many it holds follows the room.
    bool room_sized() const {
        return stats.budget_bytes == auto_budget && !tier_caches && allocation == uniform && layer_quotas.empty() &&
            !elastic;
    }
    // Enables slots evenly across the layers until `target` are enabled; disables none.
    void enable_to(size_t target) {
        if (layers.empty()) { return; }
        for (size_t layer_index = 0; layer_index < layers.size(); ++layer_index) {
            const auto & item = layers[layer_index];
            const size_t wanted = std::min(item.count, target / layers.size() + (layer_index < target % layers.size()));
            size_t have = 0;
            for (size_t i = item.begin; i < item.begin + item.count; ++i) { have += entries[i].enabled; }
            for (size_t i = item.begin; i < item.begin + item.count && have < wanted; ++i) {
                if (entries[i].enabled) { continue; }
                entries[i].enabled = true; ++have;
                stats.enabled_bytes = add(stats.enabled_bytes, stride);
            }
        }
    }
    void prepare(std::vector<expert_request> & requests, const ggml_tensor * const * tensors,
                 const size_t * bytes, kimi_cpu_sink_stats & current) {
        if (!active || paused || !stats.budget_bytes) { return; }
        const int id = routed_layer(tensors[0]);
        if (id < 0) { return; }
        char name[GGML_MAX_NAME];
        std::snprintf(name, sizeof(name), "blk.%d.ffn_gate_exps.weight", id);
        if (std::strcmp(name, tensors[1]->name)) { return; }
        std::snprintf(name, sizeof(name), "blk.%d.ffn_down_exps.weight", id);
        if (std::strcmp(name, tensors[2]->name)) { return; }
        auto found = std::find_if(layers.begin(), layers.end(), [&](const layer & l) { return l.id == id; });
        if (found == layers.end()) { return; }
        try {
            if (!requests.empty() && !geometry(bytes, tensors[0], requests.size())) { reject_admission(requests, current, "cgroup_metadata_headroom"); return; }
        } catch (const std::bad_alloc &) { pause(requests, current, "metadata_allocation_failed"); return; }
        layer & l = *found;
        const size_t layer_index = size_t(found - layers.begin());
        if (!l.count || (l.seen && l.token == token)) { return; }
        // Entries from a skipped token are not a previous-token window.
        const bool previous = l.seen && l.token != UINT64_MAX && l.token + 1 == token;
        for (size_t i = l.begin; i < l.begin + l.count; ++i) {
            auto & e = entries[i];
            if (e.leases) { throw std::runtime_error("CPU sink prepare with an outstanding cache lease"); }
            e.protected_now = false;
            if (!e.enabled) { e.reserved = false; continue; }
            if (!e.valid) { e.reserved = false; continue; }
            if (e.layer_index != layer_index) { continue; }
            auto r = std::find_if(requests.begin(), requests.end(), [&](const expert_request & request) { return request.id == e.id; });
            if ((policy == window && (!previous || r == requests.end())) ||
                    (r != requests.end() && !matches(e, *r))) {
                e.valid = e.reserved = false; ++stats.evictions; stats.valid_bytes -= stride;
            }
        }
        l.seen = true; l.token = token;
        if (policy == count_decay2) {
            size_t nodes = 0;
            for (const auto & r : requests) { nodes += l.count_frequency.find(r.id) == l.count_frequency.end(); }
            auto frequency_ticket = take_guard(multiply(nodes, allocation_cost(sizeof(std::pair<const int32_t, double>) + 4 * sizeof(void *))));
            if (!frequency_ticket) {
                reject_admission(requests, current, "cgroup_count_frequency_headroom"); return;
            }
            const double factor = std::exp2(-double(age_steps - l.count_step) / 2.0);
            for (auto & item : l.count_frequency) { item.second *= factor; }
            l.count_step = age_steps;
            try {
                // Selection frequency, independent of routing weight. Count a
                // duplicated ID once per layer/token, as the offline set did.
                for (size_t j = 0; j < requests.size(); ++j) {
                    const auto duplicate = std::find_if(requests.begin(), requests.begin() + j,
                        [&](const expert_request & r) { return r.id == requests[j].id; });
                    if (duplicate == requests.begin() + j) { l.count_frequency[requests[j].id] += 1.0; }
                }
            } catch (const std::bad_alloc &) { pause(requests, current, "count_frequency_allocation_failed"); return; }
            frequency_ticket.value.commit();
        }
        if (policy == lfu || policy == decay) {
            size_t nodes = 0;
            for (const auto & r : requests) { nodes += l.frequency.find(r.id) == l.frequency.end(); }
            // Each map node may fault a different allocator page/THP. Duplicate
            // IDs deliberately overcount; no allocation occurs before this guard.
            auto frequency_ticket = take_guard(multiply(nodes, allocation_cost(sizeof(std::pair<const int32_t, uint64_t>) + 4 * sizeof(void *))));
            if (!frequency_ticket) {
                reject_admission(requests, current, "cgroup_frequency_headroom"); return;
            }
            try {
                for (const auto & r : requests) {
                    auto & frequency = l.frequency[r.id];
                    if (frequency != UINT64_MAX) { ++frequency; }
                }
            } catch (const std::bad_alloc &) { pause(requests, current, "frequency_allocation_failed"); return; }
            frequency_ticket.value.commit();
        }
        // Protect every resident hit before reserving any direct-read target.
        // A miss at the front of the batch must never overwrite a later hit.
        for (auto & r : requests) {
            for (size_t i = l.begin; i < l.begin + l.count; ++i) {
                if (entries[i].enabled && entries[i].valid && entries[i].layer_index == layer_index &&
                        entries[i].id == r.id && matches(entries[i], r)) {
                    r.cache_hit = int(i);
                    entries[i].protected_now = true;
                    if (entries[i].prefetched) { entries[i].prefetched = false; ++prefetch_hits; }
                    entries[i].last_touch = ++clock;
                    break;
                }
            }
            if (r.cache_hit >= 0) {
                ++current.cache_hits;
                current.cache_hit_bytes = add(current.cache_hit_bytes, stride);
            }
        }
        for (auto & r : requests) {
            if (r.cache_hit >= 0) { continue; }
            const bool already_reserved = std::any_of(entries.begin() + l.begin, entries.begin() + l.begin + l.count,
                [&](const entry & e) { return e.enabled && e.reserved && e.layer_index == layer_index && e.id == r.id; });
            if (already_reserved) { ++stats.admission_skips; continue; }
            auto free = std::find_if(entries.begin() + l.begin, entries.begin() + l.begin + l.count,
                [](const entry & e) { return e.enabled && !e.valid && !e.reserved && !e.inflight; });
            if (free == entries.begin() + l.begin + l.count && policy != window) {
                auto frequency = [&](const entry & e) { return score(e.layer_index, e.id); };
                for (auto candidate = entries.begin() + l.begin; candidate != entries.begin() + l.begin + l.count; ++candidate) {
                    if (!candidate->enabled || candidate->protected_now || candidate->reserved || candidate->inflight) { continue; }
                    if (free == entries.begin() + l.begin + l.count ||
                            (policy != lru && frequency(*candidate) < frequency(*free)) ||
                            ((policy == lru || frequency(*candidate) == frequency(*free)) && candidate->last_touch < free->last_touch)) {
                        free = candidate;
                    }
                }
                if (free != entries.begin() + l.begin + l.count && policy != lru && score(layer_index, r.id) < frequency(*free)) {
                    free = entries.begin() + l.begin + l.count;
                }
                if (free != entries.begin() + l.begin + l.count) {
                    free->valid = false;
                    ++stats.evictions; stats.valid_bytes -= stride;
                    if (free->prefetched) { free->prefetched = false; ++prefetch_unused; }
                }
            }
            if (free == entries.begin() + l.begin + l.count) { ++stats.admission_skips; continue; }
            free->id = r.id; free->layer_index = layer_index; free->reserved = true; free->last_touch = ++clock;
            for (int role = 0; role < 3; ++role) { free->source[role] = key(r.role[role]); }
            r.cache_admit = int(free - entries.begin());
        }
        uint64_t new_slots = 0;
        for (const auto & r : requests) {
            if (r.cache_admit >= 0 && !entries[r.cache_admit].resident) { ++new_slots; }
        }
        // The arena is page aligned; a non-page-aligned stride can straddle one
        // additional page. No unused slot or whole-layer quota is charged here.
        const uint64_t slot_pages = add(ceil_div(stride, page_size), stride % page_size != 0);
        const uint64_t payload_pages = multiply(slot_pages, page_size);
        const uint64_t new_tables = multiply(new_slots, page_tables(payload_pages));
        const uint64_t extra = add(add(multiply(new_slots, payload_pages), new_tables), arena ? 0 : 64 * 1024);
        auto slot_ticket = take_guard(extra);
        if (!slot_ticket) {
            reject_admission(requests, current, "cgroup_headroom"); return;
        }
        if (!arena && stats.capacity_bytes && !requests.empty()) {
            if (const char * failed = map_arena()) { pause(requests, current, failed); return; }
        }
        // A discard may retain upper-level page tables. Keep a lifetime
        // upper bound, while capping refaults so repeated phase churn cannot
        // make the diagnostic grow beyond one per mapped slot.
        const uint64_t maximum_tables = multiply(entries.size(), page_tables(payload_pages));
        stats.page_table_bytes = std::min(stats.page_table_bytes, maximum_tables);
        stats.page_table_bytes += std::min(new_tables, maximum_tables - stats.page_table_bytes);
        for (const auto & r : requests) {
            if (r.cache_admit < 0 || entries[r.cache_admit].resident) { continue; }
            if (entries[r.cache_admit].was_released) {
                stats.refaulted_bytes = add(stats.refaulted_bytes, stride);
                entries[r.cache_admit].was_released = false;
            }
            // Finish the admitted first touch before publishing resident slots
            // or asynchronous read targets. Only this new slot is written.
            if (!commit_slots(size_t(r.cache_admit), 1)) { pause(requests, current, "commit_failed"); return; }
            touch_pages(arena + size_t(r.cache_admit) * stride, stride, size_t(page_size));
            entries[r.cache_admit].resident = true;
            stats.committed_bytes = add(stats.committed_bytes, stride);
            current.cache_new_slot_bytes = add(current.cache_new_slot_bytes, stride);
        }
        stats.peak_bytes = std::max(stats.peak_bytes, stats.committed_bytes);
        slot_ticket.value.commit();
    }
    // The arena, address space only: the guard admits each resident slot against the cgroup, so the kernel's
    // overcommit heuristic must not refuse a mapping larger than RAM plus swap. Ordinary pages: partial reclaim gives
    // single slots back (arena_discard). Every slot's address space: a room-sized cache grows into the slots it does
    // not hold yet. nullptr, or why it failed.
    const char * map_arena() {
        const uint64_t bytes = multiply(entries.size(), stride);
        void * memory = ssd::platform::arena_reserve(size_t(bytes));
        if (!memory) { return "mmap_failed"; }
        arena = static_cast<char *>(memory); stats.mapped_bytes = bytes;
        return ssd::platform::arena_no_hugepage(arena, stats.mapped_bytes) ? nullptr : "nohugepage_failed";
    }
    // Commits the pages of slots [first, first + count) before their first touch, and again after a discard: Windows
    // charges commit here and macOS takes the pages back; Linux charges at the touch. Pages a neighbouring slot shares
    // stay as they are.
    bool commit_slots(size_t first, size_t count) const {
        const uint64_t begin = multiply(first, stride) / page_size * page_size;
        const uint64_t end = ceil_div(multiply(add(first, count), stride), page_size) * page_size;
        return ssd::platform::arena_commit(arena + begin, size_t(end - begin));
    }
    // A cache that can give room back to the ledger: partial reclaim, or elastic (its prefill shrinks to its prefill
    // slots).
    bool gives_room_back() const { return partial_reclaim || elastic; }
    // At load (the sink's load hook, off unified memory): the geometry from the expert tensors' strides (bytes: each
    // role as a decode request reserves it; max_requests: the model's experts a token, as the first decode layer's
    // requests price the frequency nodes its admissions add), the arena, and the slots the guard admits now (all of
    // them when the cache was sized from this room) committed and paged in from `threads` threads, so that no decode
    // token pays their first touch (13.8 ms a token over decode tokens 1-128 on an RTX 5090, 1.77 s a cold start).
    // Only a cache kept across requests keeps them through the first prefill, which may give slots back through the
    // ledger as every later request's prefill does; the first decode adopts the arena whatever its epoch, and takes
    // back what the room allows then (regrow). So only a cache that can give room back takes its storage here
    // (partial reclaim, or elastic): an arena that cannot would refuse the first prefill's ledger requests, where a
    // cache sized at the first decode token comes after them. The slots enabled are the ones the first decode token
    // would size, when no prefill holds room; the slots committed leave the room the ledger holds for the first prefill's
    // pages (prefill_pages: the load-time graph's, the largest prefill's), which the first decode token takes back.
    // Returns the slots committed.
    size_t reserve_at_load(const size_t * bytes, const ggml_tensor * tensor, size_t max_requests, int threads, double & us) {
        us = 0;
        if (!keep_across_requests || !gives_room_back() || paused || !stats.budget_bytes || arena || layers.empty()) {
            return 0;
        }
        std::vector<std::thread> pool;
        pool.reserve(size_t(std::max(1, threads))); // before anything changes: a failure here leaves the cache as it was
        struct decode_view {
            bool & projection;
            explicit decode_view(bool & p) : projection(p) { projection = true; }
            ~decode_view() { projection = false; }
        };
        {
            decode_view view(decode_projection);
            if (!geometry(bytes, tensor, std::max<size_t>(1, max_requests)) || !stats.capacity_bytes) { return 0; }
        }
        const uint64_t slot_pages = add(ceil_div(stride, page_size), stride % page_size != 0);
        const uint64_t payload_pages = multiply(slot_pages, page_size), tables = page_tables(payload_pages);
        auto cost = [&](size_t n) { return add(multiply(n, add(payload_pages, tables)), 64 * 1024); };
        uint64_t surplus = 0;
        if (check_room(0, nullptr, &surplus) == room_unreadable) { return 0; }
        size_t n = std::min<size_t>(entries.size(), surplus > 64 * 1024 ? (surplus - 64 * 1024) / add(payload_pages, tables) : 0);
        // The slots committed, as runs [first, first + count): the first n, or for a room-sized cache the first of
        // each layer's enabled slots, n in all.
        std::vector<std::pair<size_t, size_t>> runs;
        if (room_sized()) {
            size_t left = n;
            for (const auto & l : layers) {
                size_t k = 0;
                while (k < l.count && entries[l.begin + k].enabled) { ++k; }
                k = std::min(k, left);
                left -= k;
                if (k) { runs.emplace_back(l.begin, k); }
            }
            n -= left;
        } else if (n) {
            runs.emplace_back(0, n);
        }
        if (!n) { return 0; }
        auto load_ticket = take_guard(cost(n));
        if (!load_ticket) { return 0; }
        if (map_arena()) { release(); return 0; }
        for (const auto & run : runs) { if (!commit_slots(run.first, run.second)) { release(); return 0; } }
        const auto start = detail::cache_clock::now();
        const size_t workers = size_t(std::max(1, std::min(threads, int(n))));
        std::atomic<bool> touched{false};
        // each worker pages in its share of every run
        auto populate = [&](size_t w) {
            for (const auto & run : runs) {
                char * begin = arena + (run.first + run.second * w / workers) * stride;
                char * end = arena + (run.first + run.second * (w + 1) / workers) * stride;
                if (end > begin && !ssd::platform::arena_populate(begin, size_t(end - begin))) { touched = true; }
            }
        };
        // this thread takes part 0 and whatever no started thread took (the pool's storage is reserved, so only a
        // thread's creation can fail, and then nothing was added)
        size_t started = 1;
        try {
            for (; started < workers; ++started) { pool.emplace_back(populate, started); }
        } catch (...) {}
        for (size_t w = started; w < workers; ++w) { populate(w); }
        populate(0);
        for (auto & t : pool) { t.join(); }
        // Page-aligned production strides are fully populated by the workers;
        // also touch the tail for a valid nonaligned legacy/test geometry.
        for (const auto & run : runs) { touch_pages(arena + (run.first + run.second) * stride - 1, 1, size_t(page_size)); }
        load_ticket.value.commit();
        us = detail::cache_elapsed_us(start);
        for (const auto & run : runs) {
            for (size_t i = run.first; i < run.first + run.second; ++i) { entries[i].resident = true; }
        }
        stats.committed_bytes = multiply(n, stride);
        stats.page_table_bytes = multiply(n, tables);
        stats.peak_bytes = std::max(stats.peak_bytes, stats.committed_bytes);
        adopt = true;
        std::fprintf(stderr, "[kimi_cpu_sink_cache] reason=load slots=%zu of=%zu slot_bytes=%zu committed_bytes=%llu us=%.0f threads=%zu pages=%s\n",
                n, entries.size(), stride, (unsigned long long) stats.committed_bytes, us, workers,
                touched ? "touched" : "populated");
        return n;
    }
    // The first decode token after a prefill (partial reclaim, not elastic): the slots the prefill gave back (the room
    // it held for its pages and its buffers, given back at its start) come back as far as the room allows now (an
    // automatic budget: geometry's price of a slot past the metadata room; a fixed one: all of them), so the cache
    // holds what one sized at this token would, within the arena's slots. They are paged in at their admission, as
    // every slot was without the load step. So are the enabled slots the load step did not commit (the room it saw
    // was smaller than the one geometry sized from, or it left the first prefill's pages): with an automatic budget
    // they take their room first (owed). `after`: "load" for the first request of an arena made at load, "prefill".
    // A room-sized cache grows here past the size it had, up to every slot of its arena, so a cache that started
    // short of memory takes the room that frees up later; one that had no room for its metadata learns its geometry
    // again at this request.
    void regrow(const char * after) {
        if (!partial_reclaim || elastic || !stride || paused) { return; }
        const bool sized = room_sized();
        if (sized && entries.empty()) { stride = 0; std::fill(role_bytes, role_bytes + 3, 0); return; }
        if (!arena && !sized) { return; }
        const size_t before = enabled_slots();
        uint64_t more = entries.size() - before, surplus = 0, owed = 0;
        if (!more) { return; }
        if (stats.budget_bytes == auto_budget) {
            if (check_room(0, nullptr, &surplus) == room_unreadable || !slot_cost_bytes) { return; }
            for (const auto & e : entries) { owed += e.enabled && !e.resident; }
            const uint64_t room = surplus > metadata_bytes ? (surplus - metadata_bytes) / slot_cost_bytes : 0;
            more = std::min<uint64_t>(more, room > owed ? room - owed : 0);
        }
        enable_to(before + size_t(more));
        if (sized) { stats.capacity_bytes = std::max(stats.capacity_bytes, multiply(enabled_slots(), stride)); }
        std::fprintf(stderr, "[kimi_cpu_sink_cache] %s_regrow=1 before_slots=%zu after_slots=%zu of=%zu owed_slots=%llu surplus_bytes=%llu committed_bytes=%llu\n",
                after, before, enabled_slots(), entries.size(), (unsigned long long) owed, (unsigned long long) surplus,
                (unsigned long long) stats.committed_bytes);
    }
    // Host memory the cache may still commit before a reset: its enabled but not yet resident slots (an elastic
    // cache enables all of them again at the next decode phase), or before the first decode token maps the arena,
    // what geometry() will size it to for this slot size. Without a unified GPU tier the automatic size depends on
    // the cgroup at that time, and only the GPU tier of a unified device asks.
    uint64_t pending_bytes(size_t slot_bytes) const {
        if (!stats.budget_bytes || paused) { return 0; }
        if (arena) {
            const uint64_t usable = elastic ? stats.capacity_bytes : stats.enabled_bytes;
            return usable > stats.committed_bytes ? usable - stats.committed_bytes : 0;
        }
        if (!slot_bytes || layers.empty()) { return 0; }
        uint64_t budget = stats.budget_bytes;
        if (budget == auto_budget) {
            if (!tier_caches) { return 0; }
            budget = multiply(multiply(layers.size(), uint64_t(std::max(1, staging_per_layer))), slot_bytes);
        }
        return multiply(std::min(budget / slot_bytes, multiply(layers.size(), uint64_t(quota))), slot_bytes);
    }
    char * data(int index, int role) const {
        size_t offset = size_t(index) * stride;
        for (int r = 0; r < role; ++r) { offset += role_bytes[r]; }
        return arena + offset;
    }
    // Whether the layer's slots hold the expert or a read of it is on its way. Changes nothing: the next-layer
    // prefetch's statistics ask it for every rank of a hint, read or not.
    bool holds(int32_t layer_id, int32_t expert) const {
        if (!active || !arena) { return false; }
        const auto found = std::find_if(layers.begin(), layers.end(), [&](const layer & l) { return l.id == layer_id; });
        if (found == layers.end()) { return false; }
        const size_t layer_index = size_t(found - layers.begin());
        for (size_t i = found->begin; i < found->begin + found->count; ++i) {
            const auto & e = entries[i];
            if (e.enabled && (e.valid || e.reserved || e.inflight) && e.layer_index == layer_index && e.id == expert) { return true; }
        }
        return false;
    }
    // Whether a grouped prefill's read of the layer's retained entries (prepare_prefill_read) could serve the expert, its
    // sources aside. holds() answers for decode: a prefill phase leaves the cache inactive, its retained entries readable.
    bool prefill_holds(int32_t layer_id, int32_t expert) const {
        if (!prefill_read || !arena || !stride || !stats.valid_bytes) { return false; }
        const auto found = std::find_if(layers.begin(), layers.end(), [&](const layer & l) { return l.id == layer_id; });
        if (found == layers.end()) { return false; }
        const size_t layer_index = size_t(found - layers.begin());
        for (const size_t i : lent) {
            const auto & e = entries[i];
            if (e.valid && e.resident && e.layer_index == layer_index && e.id == expert) { return true; }
        }
        for (size_t i = found->begin; i < found->begin + found->count; ++i) {
            const auto & e = entries[i];
            if (e.enabled && e.valid && e.resident && e.layer_index == layer_index && e.id == expert) { return true; }
        }
        return false;
    }
    // Next-layer prefetch, of a decode run or of a grouped prefill (its read-ahead, where the prefill reads the retained
    // entries): take a resident, idle slot of the layer (a free one first, else the coldest) for an expert that is not
    // cached. The slot is inflight, invisible to lookups and allocation, until prefetch_done publishes it (or
    // prefetch_activate publishes it early as a hit that its run waits for).
    // Returns -2 when the expert is already cached or on its way, -1 when there is no slot.
    int prefetch_reserve(int32_t layer_id, int32_t expert, const kimi_cpu_sink_source * roles) {
        if (!(active || prefill_read) || paused || !stats.budget_bytes || !arena || !stride) { return -1; }
        const auto found = std::find_if(layers.begin(), layers.end(), [&](const layer & l) { return l.id == layer_id; });
        if (found == layers.end() || !found->count) { return -1; }
        const size_t layer_index = size_t(found - layers.begin());
        for (int role = 0; role < 3; ++role) { if (roles[role].nbytes > role_bytes[role]) { return -1; } }
        source_key keys[3];
        for (int role = 0; role < 3; ++role) { keys[role] = key(roles[role]); }
        int victim = -1;
        for (size_t i = found->begin; i < found->begin + found->count; ++i) {
            const auto & e = entries[i];
            if (!e.enabled) { continue; }
            if ((e.valid || e.reserved || e.inflight) && e.layer_index == layer_index && e.id == expert &&
                    e.source[0] == keys[0] && e.source[1] == keys[1] && e.source[2] == keys[2]) { return -2; }
            if (!e.resident || e.reserved || e.inflight || e.leases || e.protected_now || e.lent) { continue; }
            if (victim < 0 || (entries[size_t(victim)].valid && (!e.valid || colder(i, size_t(victim))))) { victim = int(i); }
        }
        for (const size_t i : lent) {
            const auto & e = entries[i];
            if ((e.valid || e.inflight) && e.layer_index == layer_index && e.id == expert &&
                    e.source[0] == keys[0] && e.source[1] == keys[1] && e.source[2] == keys[2]) { return -2; }
        }
        if (victim < 0) { return -1; }
        auto & e = entries[size_t(victim)];
        if (e.valid) {
            e.valid = false; ++stats.evictions; stats.valid_bytes -= stride;
            if (e.prefetched) { ++prefetch_unused; }
        }
        e.id = expert; e.layer_index = layer_index; e.inflight = true; e.prefetched = false;
        e.generation = generation; e.last_touch = ++clock;
        for (int role = 0; role < 3; ++role) { e.source[role] = keys[role]; }
        return victim;
    }
    // A prefill read-ahead for `layer_id` that found no slot of its own (prefetch_reserve): a slot of a layer whose
    // call this pass has finished (an id below `finished_below`), an empty one first, else the coldest of them, as
    // prefetch_reserve takes one. Such a layer takes none of its slots again in this pass, and decode is what its
    // entries are kept for: the slot serves the borrower's prefill read once, then comes back empty
    // (return_borrowed), so that a pass's read-ahead cycles a few slots rather than empty the earlier layers. -1: none.
    int prefetch_borrow(int32_t layer_id, int32_t expert, const kimi_cpu_sink_source * roles, int32_t finished_below) {
        if (!prefill_read || paused || !stats.budget_bytes || !arena || !stride) { return -1; }
        const auto found = std::find_if(layers.begin(), layers.end(), [&](const layer & l) { return l.id == layer_id; });
        if (found == layers.end()) { return -1; }
        for (int role = 0; role < 3; ++role) { if (roles[role].nbytes > role_bytes[role]) { return -1; } }
        int victim = -1;
        for (const auto & l : layers) {
            if (l.id >= finished_below || l.id == layer_id) { continue; }
            for (size_t i = l.begin; i < l.begin + l.count; ++i) {
                const auto & e = entries[i];
                if (!e.enabled || !e.resident || e.reserved || e.inflight || e.leases || e.protected_now || e.lent) { continue; }
                if (victim < 0 || (entries[size_t(victim)].valid && (!e.valid || colder(i, size_t(victim))))) { victim = int(i); }
            }
        }
        if (victim < 0) { return -1; }
        auto & e = entries[size_t(victim)];
        if (e.valid) {
            e.valid = false; ++stats.evictions; stats.valid_bytes -= stride;
            if (e.prefetched) { ++prefetch_unused; }
        }
        e.id = expert; e.layer_index = size_t(found - layers.begin()); e.inflight = true; e.prefetched = false;
        e.lent = true; e.generation = generation; e.last_touch = ++clock;
        for (int role = 0; role < 3; ++role) { e.source[role] = key(roles[role]); }
        lent.push_back(size_t(victim));
        return victim;
    }
    // The borrowed slots back to their own layers, empty: all of them but those lent to `keep_layer`, whose call has
    // not read them yet (-1: all). A slot still read into or leased stays lent until a later call.
    void return_borrowed(int32_t keep_layer) {
        for (size_t k = 0; k < lent.size();) {
            const size_t index = lent[k];
            auto & e = entries[index];
            if ((keep_layer >= 0 && layers[e.layer_index].id == keep_layer) || e.inflight || e.leases) { ++k; continue; }
            if (e.valid) { e.valid = false; stats.valid_bytes -= stride; }
            e.id = -1; e.prefetched = false; e.lent = false;
            for (size_t l = 0; l < layers.size(); ++l) {
                if (index >= layers[l].begin && index < layers[l].begin + layers[l].count) { e.layer_index = l; break; }
            }
            lent.erase(lent.begin() + std::ptrdiff_t(k));
        }
    }
    // prefetched: a decode prefetch's entry, which its statistics follow until a run takes it (a grouped prefill's
    // read-ahead publishes an ordinary entry).
    void prefetch_done(int index, uint64_t reserved_generation, bool ok, bool prefetched = true) {
        if (index < 0 || size_t(index) >= entries.size() || reserved_generation != generation) { return; }
        auto & e = entries[size_t(index)];
        if (!e.inflight || e.generation != reserved_generation) { return; }
        e.inflight = false;
        if (e.valid) { // published early as a hit
            if (!ok) { e.valid = false; e.prefetched = false; e.id = -1; stats.valid_bytes -= stride; }
            return;
        }
        if (ok && e.enabled && e.resident) {
            e.valid = true; e.prefetched = prefetched; e.last_touch = ++clock;
            stats.valid_bytes += stride;
        } else {
            e.id = -1;
        }
    }
    bool prefetch_activate(int index, uint64_t reserved_generation) {
        if (index < 0 || size_t(index) >= entries.size() || reserved_generation != generation) { return false; }
        auto & e = entries[size_t(index)];
        if (!e.inflight || e.valid || e.generation != reserved_generation || !e.enabled || !e.resident) { return false; }
        e.valid = true; e.prefetched = true; e.last_touch = ++clock;
        stats.valid_bytes += stride;
        return true;
    }
    void prepare_prefill_read(std::vector<expert_request> & requests,
            const ggml_tensor * up, const ggml_tensor * gate, const ggml_tensor * down,
            kimi_cpu_sink_stats & current) {
        if (!prefill_read || !arena || !stride || !stats.valid_bytes || requests.empty()) { return; }
        const int layer_id = routed_layer(up);
        if (layer_id < 0) { return; }
        char name[GGML_MAX_NAME];
        std::snprintf(name, sizeof(name), "blk.%d.ffn_gate_exps.weight", layer_id);
        if (std::strcmp(name, gate->name)) { return; }
        std::snprintf(name, sizeof(name), "blk.%d.ffn_down_exps.weight", layer_id);
        if (std::strcmp(name, down->name)) { return; }
        const auto found = std::find_if(layers.begin(), layers.end(),
            [&](const layer & l) { return l.id == layer_id; });
        if (found == layers.end() || !found->count) { return; }
        const size_t layer_index = size_t(found - layers.begin());
        size_t bytes[3] = {};
        for (int role = 0; role < 3; ++role) {
            for (const auto & request : requests) {
                bytes[role] = std::max(bytes[role], request.role[role].nbytes);
            }
        }
        if (bytes[0] > role_bytes[0] || bytes[1] > role_bytes[1] || bytes[2] > role_bytes[2]) { return; }

        // the layer's own slots, then the ones a read-ahead borrowed for it
        std::vector<size_t> slots(found->count);
        std::iota(slots.begin(), slots.end(), found->begin);
        slots.insert(slots.end(), lent.begin(), lent.end());
        for (auto & request : requests) {
            if (request.vram_slot >= 0 || request.cache_hit >= 0) { continue; }
            for (const size_t index : slots) {
                auto & e = entries[index];
                if (!e.enabled || !e.valid || !e.resident || e.layer_index != layer_index ||
                        e.id != request.id || !matches(e, request)) { continue; }
                acquire_lease(int(index), request);
                request.cache_hit = int(index);
                request.cache_lease = true;
                e.protected_now = true;
                ++current.cache_hits;
                current.cache_hit_bytes = add(current.cache_hit_bytes, stride);
                ++stats.prefill_hits;
                stats.prefill_hit_bytes = add(stats.prefill_hit_bytes, stride);
                ++stats.prefill_leases;
                break;
            }
        }
    }
    void release_prefill_leases(std::vector<expert_request> & requests) {
        for (auto & request : requests) {
            if (!request.cache_lease) { continue; }
            release_lease(request.cache_hit);
            request.cache_lease = false;
        }
    }
    void admit(const expert_request & request, kimi_cpu_sink_stats & current) {
        if (request.cache_admit < 0) { return; }
        auto & e = entries[request.cache_admit];
        if (!e.enabled || !e.resident) { throw std::runtime_error("CPU sink cache admitted a disabled slot"); }
        // All three reads completed and no compute worker retains this slot.
        // Publication changes metadata only; the expert payload never moves.
        e.valid = true; e.reserved = false;
        stats.valid_bytes += stride;
        stats.admitted_bytes += stride;
        ++current.cache_admitted; current.cache_admitted_bytes += stride;
    }
    void acquire_lease(int index, const expert_request & request) {
        auto & e = entries.at(size_t(index));
        if (!e.enabled || (!e.valid && !e.reserved) || e.id != request.id ||
                !matches(e, request) || e.leases == UINT32_MAX) {
            throw std::runtime_error("CPU sink cache lease identity mismatch");
        }
        ++e.leases;
    }
    void release_lease(int index) {
        auto & e = entries.at(size_t(index));
        if (!e.leases || e.reserved) { throw std::runtime_error("CPU sink cache lease released before completion"); }
        if (!--e.leases) { e.protected_now = false; }
    }
    // Called only while the other tier holds a versioned ready lease. Return
    // -1 if an executor still owns this RAM copy, 0 if absent, 1 if dropped.
    int drop_duplicate(int layer_id, const expert_request & request) {
        if (entries.empty()) { return 0; } // RAM admission may have paused.
        const auto found = std::find_if(layers.begin(), layers.end(), [&](const layer & l) { return l.id == layer_id; });
        if (found == layers.end()) { return 0; }
        const size_t layer_index = size_t(found - layers.begin());
        for (size_t i = found->begin; i < found->begin + found->count; ++i) {
            auto & e = entries[i];
            if (!e.valid || e.id != request.id || e.layer_index != layer_index || !matches(e, request)) { continue; }
            return drop_entry(i);
        }
        return 0;
    }
    // Entry `index`'s copy, which the other tier holds ready too: 1 dropped, -1 an executor still owns it, 0 not valid.
    int drop_entry(size_t index) {
        auto & e = entries.at(index);
        if (!e.valid) { return 0; }
        if (e.leases || e.protected_now || e.reserved || e.inflight) { return -1; }
        e.valid = false;
        stats.valid_bytes -= stride;
        // Keep resident and committed/page-table accounting: the physical
        // slot is reused in place, without a second payload allocation.
        return 1;
    }
};

}} // namespace ssd::host
