// Inspect private metadata without creating workers, a GPU context or a payload
// arena. Real file identities exercise version rejection and RAM lease vetoes.
#include "ggml-ssd/host/expert-cache.h"
#include "test-resources.h"
#include <stdexcept>
#include <vector>
using sink_cache = ssd::host::expert_cache;
using sink_request = ssd::host::expert_request;
int main() {
    try {
    int checks = 0;
    auto require = [&](bool ok) { ++checks; if (!ok) throw std::runtime_error("cache lease assertion"); };
    const std::string path = test_resources::temp_file("kimi-cache-lease");
    const int fd = ssd::platform::open_buffered(path.c_str()); require(fd >= 0);
    ssd::host::expert_cache_options configuration;
    configuration.require_shared = false; configuration.cache_policy = "lru";
    sink_cache cache(configuration); sink_cache::layer layer; layer.id = 1; layer.count = 1;
    cache.layers.push_back(layer); cache.entries.resize(1); cache.stride = 12288;
    sink_request r; r.id = 5;
    auto & e = cache.entries[0]; e.id = 5; e.valid = e.resident = e.enabled = true;
    for (int role = 0; role < 3; ++role) {
        r.role[role] = {fd,uint64_t(role)*4096,4096}; e.source[role] = sink_cache::key(r.role[role]);
    }
    cache.stats.valid_bytes = cache.stats.committed_bytes = cache.stride;
    cache.acquire_lease(0,r); require(e.leases == 1);
    e.protected_now = true; cache.release_lease(0); require(!e.protected_now && !e.leases && e.valid);
    {
        // A read-ahead slot comes from an entry its pass is not likely to use (keep), the older one first otherwise;
        // a load-time preload takes only an empty slot, and none while decode runs.
        sink_cache c(configuration); sink_cache::layer l; l.id = 1; l.count = 2;
        c.layers.push_back(l); c.entries.resize(2); c.stride = 12288; c.stats.budget_bytes = 1; c.prefill_read = true;
        for (int role = 0; role < 3; ++role) { c.role_bytes[role] = 4096; }
        std::vector<char> arena(2 * c.stride); c.arena = arena.data();
        for (int i = 0; i < 2; ++i) {
            auto & x = c.entries[size_t(i)]; x.id = 5 + i; x.valid = x.resident = x.enabled = true; x.last_touch = uint64_t(i + 1);
        }
        c.stats.valid_bytes = 2 * c.stride;
        sink_request n; n.id = 9;
        for (int role = 0; role < 3; ++role) { n.role[role] = {fd,uint64_t(8 + role)*4096,4096}; }
        std::vector<uint8_t> keep(16, 0); keep[5] = 1;
        require(c.prefetch_reserve(1, 9, n.role) == 0);
        c.prefetch_done(0, c.generation, false);
        c.entries[0].id = 5; c.entries[0].valid = true; c.stats.valid_bytes += c.stride;
        require(c.prefetch_reserve(1, 9, n.role, &keep) == 1 && c.entries[0].valid && c.entries[1].inflight);
        c.prefetch_done(1, c.generation, true, false);
        require(c.entries[1].valid && c.entries[1].id == 9 && !c.entries[1].prefetched);
        require(c.preload_reserve(1, 11, n.role) == -1);
        c.entries[0].valid = false; c.stats.valid_bytes -= c.stride;
        c.active = true; require(c.preload_reserve(1, 11, n.role) == -1); c.active = false;
        require(c.preload_reserve(1, 11, n.role) == 0 && c.entries[0].inflight);
        c.prefetch_done(0, c.generation, false);
        // A copy of an expert the GPU tier holds (tier_held) is a backup: a read-ahead takes its slot before any other
        // entry's, even one older, and even where the pass is likely to use the expert (the tier serves it).
        c.entries[0].id = 5; c.entries[0].valid = true; c.entries[0].last_touch = 1; c.stats.valid_bytes += c.stride;
        c.entries[1].last_touch = 2;
        c.tier_held = [](void *, int32_t layer, const int32_t * experts, int count, uint8_t * held) {
            for (int i = 0; i < count; ++i) { held[i] = layer == 1 && experts[i] == 9; } };
        keep[9] = 1;
        require(c.prefetch_reserve(1, 12, n.role, &keep) == 1 && c.entries[0].valid && c.entries[1].inflight);
        c.prefetch_done(1, c.generation, false);
        c.arena = nullptr; // the test's own buffer
    }
    {
        // A guard's shrink for room: a layer whose slots past its share are busy leaves the rest to the others' coldest
        // idle slots, and every busy slot stays.
        sink_cache c(configuration); c.partial_reclaim = true; c.stride = 3 * c.page_size;
        for (int id = 1; id <= 2; ++id) {
            sink_cache::layer l; l.id = id; l.begin = size_t(4 * (id - 1)); l.count = 4; c.layers.push_back(l);
        }
        c.entries.resize(8);
        for (size_t i = 0; i < 8; ++i) {
            auto & x = c.entries[i]; x.enabled = true; x.layer_index = i / 4; x.id = int32_t(i % 4); x.last_touch = i + 1;
        }
        c.stats.enabled_bytes = 8 * c.stride;
        for (size_t i = 1; i < 4; ++i) { c.entries[i].leases = 1; } // layer 1: one idle slot of four
        require(c.shrink_to(4, true, "test", false) == 4);
        require(c.entries[1].enabled && c.entries[2].enabled && c.entries[3].enabled && !c.entries[0].enabled);
        size_t second = 0;
        for (size_t i = 4; i < 8; ++i) { second += c.entries[i].enabled; }
        require(second == 1 && c.entries[7].enabled); // the coldest of layer 2 went, its warmest stayed
        // a strict shrink (a phase boundary) keeps each layer to its share and refuses a busy one
        c.enable_to(8);
        c.entries[0].leases = 1;
        bool refused = false;
        try { c.shrink_to(4, false, "test", true); } catch (const std::runtime_error &) { refused = true; }
        require(refused);
    }
    ssd::platform::close_fd(fd); test_resources::remove_file(path); std::printf("CACHE_LEASE_PASS assertions=%d payload_allocated=0\n",checks); return 0;
} catch (const std::exception & e) { std::fprintf(stderr,"CACHE_LEASE_FAIL: %s\n",e.what()); return 1; } }
