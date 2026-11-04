#include "explicit-fixture-inputs.h"
// CUDA lifecycle test of the elastic GPU tier's unified-memory behaviour, on any CUDA device:
//  - GGML_KIMI_CPU_SINK_VRAM_BYTES=auto starts from one block; a discrete GPU grows at once, a unified one at the
//    token boundary after its first decode step;
//  - an OOM reclaim while a layer is live is refused without poisoning the tier; with a deferred upload still
//    copying it waits for the copy, gives the block back, and collect reports that upload as aborted;
//  - reclaim_host gives blocks back for a host allocation until free memory covers it, keeping one block, and
//    refuses while a layer is live;
//  - auto is refused without an elastic tier and gives back on demand only;
//  - with ggml's memory query replaced by a model of a unified DRAM (so this runs on any CUDA device): the
//    host-memory ledger's floor and the RAM cache's demand, published before the first prepare, bound the base and
//    the growth and are ggml's reserve; growth at the second decode boundary waits for a deferred upload still
//    copying instead of skipping; a demand rise at a token boundary gives blocks back while an upload into them is
//    in flight, and collect reports that upload as aborted;
//  - with the host-memory ledger's envelope (a cgroup memory.max) tighter than the modeled DRAM: the ledger counts
//    the tier's storage, which the cgroup does not charge, so auto grows to the envelope less the RAM cache's demand;
//    a tier no phase armed gives a block back to an allocation that would fail.
// Shared lifecycle assertions use the retained explicit Q4_K format engine.
#include "ggml-backend.h"
#include "ggml-cuda.h"
#include "gpu-kernels.h"
#include "vram-tier.cpp"
#include "test-kimi-vram-format-fixture.h"
#include <future>

namespace {
std::mutex gate_mutex;
std::condition_variable gate_cv;
bool held = false, entered = false;
bool upload_gate(void *) {
    std::unique_lock<std::mutex> lock(gate_mutex);
    entered = true; gate_cv.notify_all();
    gate_cv.wait(lock, [] { return !held; });
    return true;
}
void hold() { std::lock_guard<std::mutex> lock(gate_mutex); held = true; entered = false; }
void open_gate() { std::lock_guard<std::mutex> lock(gate_mutex); held = false; gate_cv.notify_all(); }
bool copy_started() {
    std::unique_lock<std::mutex> lock(gate_mutex);
    return gate_cv.wait_for(lock, std::chrono::seconds(5), [] { return entered; });
}
template <class T> bool done_within(std::future<T> & f, int ms) {
    return f.wait_for(std::chrono::milliseconds(ms)) == std::future_status::ready;
}
// A unified DRAM: `pool` bytes are available before the modeled tier's blocks, which come out of it.
size_t pool = 0, published = 0;
const tier * modeled = nullptr;
void model_get(int, size_t * free_bytes, size_t * total) {
    const size_t used = modeled ? size_t(modeled->capacity_bytes) : 0;
    *free_bytes = pool > used ? pool - used : 0; *total = pool + (size_t(64) << 30);
}
bool model_unified(int) { return true; }
void model_reserve(int, size_t bytes) { published = bytes; }
size_t model_free() { size_t free_bytes = 0, total = 0; model_get(0, &free_bytes, &total); return free_bytes; }
// A discrete card under a reserve: `card_room` bytes are free when the tier first takes memory, less what this process
// has taken since and what other processes have.
ssd::gpu::reserve_budget card_budget;
size_t card_room = 0, card_others = 0, card_reserve = 0, card_held_at_anchor = 0;
size_t card_free() {
    const size_t used = card_budget.held() - card_held_at_anchor + card_others;
    return card_room > used ? card_room - used : 0;
}
void card_get(int, size_t * free_bytes, size_t * total) {
    const size_t now = card_free();
    card_budget.observe(now);
    *free_bytes = card_budget.free_bytes(now, card_reserve); *total = size_t(64) << 30;
}
bool card_unified(int) { return false; }
void card_hold(int, size_t bytes) { card_budget.hold(bytes); }
void card_release(int, size_t bytes) { card_budget.release(bytes); }
void card_anchor(int) { card_held_at_anchor = card_budget.held(); card_budget.anchor(card_free()); }
void card_state(int, size_t * held_bytes, size_t * reachable) { *held_bytes = card_budget.held(); *reachable = card_budget.reachable(); }
// The envelope the host-memory ledger reads: a cgroup memory.max, its usage, and a MemAvailable that never binds.
struct envelope_files {
    uint64_t limit = 0, current = uint64_t(1) << 30;
    bool read(const std::string & path, std::string & text) const {
        if (path == "/proc/self/cgroup") { text = "0::/fixture\n"; }
        else if (path == "/sys/fs/cgroup/fixture/memory.max") { text = std::to_string(limit); }
        else if (path == "/sys/fs/cgroup/fixture/memory.current") { text = std::to_string(current); }
        else if (path == "/sys/fs/cgroup/fixture/memory.stat") { text = "active_file 0\ninactive_file 0\nfile 0\nworkingset_refault_file 0\n"; }
        else if (path == "/proc/meminfo") { text = "MemAvailable: " + std::to_string((uint64_t(256) << 30) / 1024) + " kB\n"; }
        else { return false; }
        return true;
    }
};
int assertions = 0;
void require(bool ok, const char * what) { ++assertions; if (!ok) throw std::runtime_error(what); }
const size_t role = lifecycle_role_bytes(), stride = 3 * role;
kimi_vram_request key(int layer, int expert) {
    kimi_vram_request r = {}; r.layer = layer; r.expert = expert; r.weight = .5f;
    for (int i = 0; i < 3; ++i) { r.role[i] = {1, 2, 100000000, uint64_t(layer * 1000 + expert * 3 + i) * role, role, 3, 4, 5, 6}; }
    return r;
}
}

int main() {
    fixture_inputs inputs; inputs.set_runtime_services(test_gpu_services()); try {
    inputs.set("vram_async", "1");
    inputs.set("vram_deferred", "1");
    inputs.set("tier.policy", "lru");
    inputs.set("tier.min_count", "0");
    inputs.set("tier.global", "1");
    inputs.set("tier.keep_across_requests", "1");
    inputs.set("tier.elastic", "1");
    inputs.set("tier.demand_reclaim", "1");
    inputs.set("tier.margin_mib", "0");
    inputs.set("tier.watermark", "0");
    inputs.unset("tier.safety_blocks");
    inputs.unset("tier.fail_malloc_at");
    inputs.unset("tier.fail_free_at");
    std::vector<unsigned char> payload(stride, 7);
    const void * data[] = {payload.data(), payload.data() + role, payload.data() + 2 * role};
    std::vector<float> out(256);
    struct cleanup { ~cleanup() { open_gate(); } } cleanup_on_error;
    // ggml's own answer; the tier finds the same through the CUDA backend it looks up at run time.
    const bool unified = ggml_backend_cuda_device_is_unified(0);

    // auto: one block first; growth at once on a discrete GPU, after the first decode step on unified memory.
    inputs.set("tier.max_slots", "40");
    {
        tier t(tier::automatic_budget, inputs.tier_policy(), nullptr, test_cpu_kernels());
        const auto k = key(1, 0);
        int slot = -1;
        t.begin(51, 1, true);
        lifecycle_prepare(t,&k, 1,&slot);
        require(t.unified == unified, "the tier and ggml disagree about unified memory");
        require(t.elastic_active && t.base_slots == tier::elastic_block_slots && t.max_slots == 40,
                "auto did not start from one block under the slot cap");
        if (unified) {
            require(t.slot_locations.size() == tier::elastic_block_slots && t.grow_events == 0,
                    "unified memory grew before the first decode step ended");
        } else {
            require(t.slot_locations.size() == 40 && t.grow_events == 1, "a discrete GPU did not grow at once");
        }
        t.finish(out.data(), 256);
        t.begin(51, 2, true);
        require(t.slot_locations.size() == 40 && t.grow_events == 1, "the tier did not grow after the first decode step");
    }

    // OOM reclaim: refused inside a live layer, then waits for a deferred copy and aborts its reservation.
    inputs.set("tier.max_slots", "2");
    {
        tier t(2 * stride, inputs.tier_policy(), nullptr, test_cpu_kernels()); // one block of two slots
        t.deferred_hook = upload_gate;
        const auto g = key(2, 1);
        int slot = -1, queued = 0;
        hold();
        t.begin(52, 1, true);
        lifecycle_prepare(t,&g, 1,&slot);
        require(t.elastic_active && t.blocks.size() == 1 && t.table->admissions[0] >= 0, "reclaim fixture has no admission");
        t.reclaimer_armed = true; // a prefill arms it on a discrete GPU; unified memory reclaims in any phase
        size_t freed = 0;
        require(t.reclaim_one(4096, &freed) == 0 && !freed && !t.poisoned && t.blocks.size() == 1 && t.live_reclaim_refusals == 1,
                "a reclaim inside a live layer was not refused cleanly");
        t.admit_deferred(g, data, &queued);
        require(queued == 1 && copy_started(), "the deferred upload did not start");
        t.finish(out.data(), 256);
        // the residency the next-layer prefetch asks: an upload in flight is held (the next prepare collects it)
        const auto * residency = kimi_vram_residency_get_api();
        uint8_t held = 0;
        residency->held(&t, g.layer, &g.expert, 1, &held);
        require(residency->version == 2 && held == 1, "a deferred upload in flight was not held");
        auto reclaim = std::async(std::launch::async, [&] { return t.reclaim_one(4096, &freed); });
        require(!done_within(reclaim, 150), "the reclaim did not wait for the deferred copy");
        open_gate();
        require(done_within(reclaim, 5000), "the reclaim did not return after the copy");
        require(reclaim.get() == 1 && freed == 2 * stride && !t.poisoned && t.blocks.empty(),
                "the reclaim did not give the block back");
        residency->held(&t, g.layer, &g.expert, 1, &held);
        require(held == 0, "an upload whose block was given back is still held");
        kimi_vram_request collected[4] = {};
        int32_t aborted[4] = {};
        int count = -1;
        t.collect(-1, collected, aborted, 4, &count);
        require(count == 1 && aborted[0] == 1 && vram_slot_table::same(collected[0], g) &&
                t.deferred_published == 0 && t.deferred_aborted == 1, "collect did not report the aborted upload");
        t.begin(52, 2, true);
        lifecycle_prepare(t,&g, 1,&slot);
        require(slot < 0 && !t.poisoned, "an aborted upload served a hit");
        t.finish(out.data(), 256);
    }

    // reclaim_host: blocks go until free memory covers the allocation, one stays, a live layer refuses.
    inputs.set("tier.max_slots", "40");
    {
        tier t(40 * stride, inputs.tier_policy(), nullptr, test_cpu_kernels()); // a base of three blocks: 16, 16 and 8 slots
        const auto k = key(3, 2);
        int slot = -1;
        t.begin(53, 1, true); lifecycle_prepare(t,&k, 1,&slot); t.finish(out.data(), 256);
        require(t.slot_locations.size() == 40 && t.blocks.size() == 3, "reclaim_host fixture has no three blocks");
        // The arithmetic is the same on any device; on a discrete GPU this only lets the test exercise it.
        t.unified = true;
        size_t free_bytes = 0, total = 0;
        t.mem_info(&free_bytes, &total);
        const size_t low = t.watermark_low_mib << 20;
        require(free_bytes > low, "no room above the low watermark");
        const size_t before = t.slot_locations.size();
        require(t.reclaim_host(free_bytes - low + 10 * stride, "test") && t.slot_locations.size() < before &&
                !t.blocks.empty() && t.host_reclaims == 1, "reclaim_host did not give blocks back for an allocation");
        require(!t.reclaim_host(SIZE_MAX / 2, "test") && t.blocks.size() == 1 && !t.poisoned,
                "reclaim_host did not keep one block");
        t.begin(53, 2, true);
        lifecycle_prepare(t,&k, 1,&slot);
        const size_t slots = t.slot_locations.size();
        require(!t.reclaim_host(SIZE_MAX / 2, "test") && t.slot_locations.size() == slots && t.host_reclaim_refusals == 1,
                "reclaim_host did not refuse inside a live layer");
        t.finish(out.data(), 256);
    }
    // At or under the floor free memory reads 0: a request of nothing, the tier's own at a token boundary, still gives back.
    {
        tier t(40 * stride, inputs.tier_policy(), nullptr, test_cpu_kernels());
        const auto k = key(3, 2);
        int slot = -1;
        t.begin(53, 1, true); lifecycle_prepare(t,&k, 1,&slot); t.finish(out.data(), 256);
        require(t.blocks.size() == 3, "reclaim_host floor fixture has no three blocks");
        t.unified = true;
        t.host_floor = SIZE_MAX / 2;
        t.watermark_low_mib = 0;
        t.reclaim_host(0, "test");
        require(t.blocks.size() == 1 && t.host_reclaims == 1 && !t.poisoned, "reclaim_host(0) under the floor gave nothing back");
        t.host_floor = 0;
    }
    // auto sizes an elastic tier only, and has no base to shrink to at a prefill: it gives back on demand.
    {
        inputs.set("tier.elastic", "0");
        bool refused = false;
        try { tier t(tier::automatic_budget, inputs.tier_policy(), nullptr, test_cpu_kernels()); } catch (const std::exception &) { refused = true; }
        require(refused, "auto was accepted without an elastic tier");
        inputs.set("tier.elastic", "1");
        inputs.set("tier.demand_reclaim", "0");
        { tier t(tier::automatic_budget, inputs.tier_policy(), nullptr, test_cpu_kernels()); require(t.elastic_demand_reclaim, "auto did not reclaim on demand"); }
        { tier t(2 * stride, inputs.tier_policy(), nullptr, test_cpu_kernels()); require(!t.elastic_demand_reclaim, "a byte budget changed its prefill policy"); }
        inputs.set("tier.demand_reclaim", "1");
    }

    // The floor on a modeled unified DRAM. Low watermark 1 MiB, growth leaves 2 MiB, no fixed safety reserve.
    auto & query = ggml_memory_query::instance();
    query.get = model_get; query.unified = model_unified; query.set_reserve = model_reserve;
    inputs.set("tier.safety_mib", "0");
    inputs.set("tier.margin_mib", "2");
    inputs.set("tier.low_mib", "1");
    inputs.set("tier.high_mib", "2");
    inputs.set("tier.max_slots", "40");
    const size_t low = size_t(1) << 20, safety = size_t(2) << 20, floor = size_t(3) << 30, demand = size_t(1) << 30;
    // Published before the first prepare (the sink does so at its first decode layer), floor and demand bound the base.
    {
        tier t(tier::automatic_budget, inputs.tier_policy(), nullptr, test_cpu_kernels());
        modeled = &t; pool = floor + demand + low + 5 * stride + stride / 2; published = 0;
        policy_set_host_floor(&t, floor); policy_set_host_demand(&t, demand);
        const auto k = key(4, 0);
        int slot = -1;
        t.begin(61, 1, true); lifecycle_prepare(t,&k, 1,&slot);
        require(t.unified && published == floor + demand, "ggml's reserve is not the floor plus the demand");
        require(t.slot_locations.size() == 5 && t.grow_events == 0 && model_free() >= floor + demand + low,
                "the base did not keep the floor and the demand free");
        t.finish(out.data(), 256);
        t.begin(61, 2, true);
        require(t.grow_events == 1 && t.slot_locations.size() == 5 && model_free() >= floor + demand + low,
                "growth ate into the floor or the demand");
        modeled = nullptr;
    }
    // Growth at the second decode boundary waits for a copy still in flight; a demand rise gives blocks back.
    {
        tier t(tier::automatic_budget, inputs.tier_policy(), nullptr, test_cpu_kernels());
        modeled = &t; pool = floor + demand + safety + 20 * stride + stride / 2; published = 0;
        t.deferred_hook = upload_gate;
        policy_set_host_floor(&t, floor); policy_set_host_demand(&t, demand);
        kimi_vram_request keys[17];
        for (int i = 0; i < 17; ++i) { keys[i] = key(4, i); }
        int slots[16] = {}, queued = 0;
        t.begin(62, 1, true); lifecycle_prepare(t,keys, 16,slots);
        require(t.slot_locations.size() == 16 && t.grow_events == 0, "the auto base is not one block");
        for (int i = 0; i < 15; ++i) { t.admit_deferred(keys[i], data, &queued); require(queued == 1, "an upload was not deferred"); }
        t.drain_deferred();
        hold();
        t.admit_deferred(keys[15], data, &queued);
        require(queued == 1 && copy_started(), "the held upload did not start");
        t.finish(out.data(), 256);
        auto grow = std::async(std::launch::async, [&] { t.begin(62, 2, true); });
        require(!done_within(grow, 150), "growth did not wait for the copy in flight");
        open_gate();
        require(done_within(grow, 5000), "growth did not finish after the copy");
        grow.get();
        require(t.grow_events == 1 && t.slot_locations.size() == 20 && model_free() >= floor + demand + low,
                "the tier did not grow at the second decode boundary while an upload was in flight");
        kimi_vram_request collected[20] = {};
        int32_t aborted[20] = {};
        int count = -1;
        t.collect(-1, collected, aborted, 20, &count);
        require(count == 16 && std::none_of(aborted, aborted + count, [](int32_t a) { return a != 0; }),
                "growth aborted an upload");
        int slot = -1;
        lifecycle_prepare(t,&keys[16], 1,&slot);
        require(t.table->admissions[0] == 16, "the fixture's upload does not target the grown block");
        hold();
        t.admit_deferred(keys[16], data, &queued);
        require(queued == 1 && copy_started(), "the second held upload did not start");
        t.finish(out.data(), 256);
        const size_t raised = demand + 13 * stride;
        policy_set_host_demand(&t, raised);
        require(published == floor + raised, "a demand rise was not published to ggml");
        auto shrink = std::async(std::launch::async, [&] { t.begin(62, 3, true); });
        require(!done_within(shrink, 150), "the host reclaim did not wait for the copy in flight");
        open_gate();
        require(done_within(shrink, 5000), "the host reclaim did not finish after the copy");
        shrink.get();
        require(t.slot_locations.size() == 16 && t.blocks.size() == 1 && t.host_reclaims == 1 && !t.poisoned &&
                model_free() >= floor + raised + low, "the demand rise did not give the grown block back");
        t.collect(-1, collected, aborted, 20, &count);
        require(count == 1 && aborted[0] == 1 && vram_slot_table::same(collected[0], keys[16]) && t.deferred_aborted == 1,
                "collect did not report the upload into the freed block as aborted");
        lifecycle_prepare(t,&keys[16], 1,&slot);
        require(slot < 0, "an aborted upload served a hit");
        t.finish(out.data(), 256);
        modeled = nullptr;
    }
    // The envelope binds before the modeled DRAM: the ledger counts the tier's storage in it, so auto grows to the
    // envelope less the RAM cache's demand (30 slots) although MemAvailable has room for the slot cap (40); a tier no
    // phase armed (a load-time reserve, a disarmed prefill) still gives a block back to an allocation that would fail.
    {
        envelope_files files;
        const uint64_t reserve = uint64_t(512) << 20;
        files.limit = files.current + reserve + demand + 30 * stride + safety + stride / 2;
        auto owner = std::make_shared<ssd::host::process_arbiter>(
            [&files](const std::string & path, std::string & text) { return files.read(path, text); });
        auto made = owner->make_client();
        require(bool(made), "the host-memory client was not created");
        auto client = std::move(made.value);
        ssd::host::client_config config; config.unified = true;
        require(bool(client->configure(config)), "the host-memory client was not configured");
        tier t(tier::automatic_budget, inputs.tier_policy(), client.get(), test_cpu_kernels());
        modeled = &t; pool = floor + demand + safety + 60 * stride; published = 0;
        policy_set_host_floor(&t, floor); policy_set_host_demand(&t, demand);
        const auto k = key(5, 0);
        int slot = -1;
        t.begin(63, 1, true); lifecycle_prepare(t, &k, 1, &slot); t.finish(out.data(), 256);
        require(t.slot_locations.size() == tier::elastic_block_slots && client->state().value.device_held == 16 * stride,
                "the auto base did not publish its storage to the envelope");
        t.begin(63, 2, true);
        const auto room = client->measure(0);
        require(t.grow_events == 1 && t.slot_locations.size() == 30 && client->state().value.device_held == 30 * stride &&
                bool(room) && room.value.cgroup_surplus >= demand && model_free() > floor + demand + 20 * stride,
                "auto did not grow to the envelope less the RAM cache's demand");
        t.reclaimer_armed = false;
        size_t freed = 0;
        require(t.reclaim_one(4096, &freed) == 1 && freed == 14 * stride && t.slot_locations.size() == 16 &&
                client->state().value.device_held == 16 * stride && !t.poisoned,
                "an unarmed tier did not give a block back on unified memory");
        modeled = nullptr;
    }
    // A reserve on a discrete card is kept against what this process holds: other processes cannot take the tier's
    // memory, and what they release beyond the reserve is grown into.
    {
        query.get = card_get; query.unified = card_unified; query.hold = card_hold; query.release = card_release;
        query.anchor = card_anchor; query.state = card_state;
        inputs.set("tier.watermark", "1");
        inputs.set("tier.max_slots", "200");
        inputs.set("tier.safety_mib", "2");
        inputs.set("tier.margin_mib", "2");
        inputs.set("tier.low_mib", "1");
        inputs.set("tier.high_mib", "2");
        const size_t line = size_t(2) << 20;
        // A full block of 16 slots takes whole device pages, so three of them are 48 slots.
        const size_t block = ssd::gpu::device_pages(16 * stride);
        card_reserve = size_t(1) << 20; card_others = 0; card_room = line + 3 * block + stride / 2;
        tier t(tier::automatic_budget, inputs.tier_policy(), nullptr, test_cpu_kernels());
        const auto k = key(6, 0);
        int slot = -1;
        const auto token = [&](uint64_t n) { t.finish(out.data(), 256); t.drain(); t.drain_deferred(); t.begin(64, n, true); };
        t.begin(64, 1, true); lifecycle_prepare(t, &k, 1, &slot);
        require(!t.unified && t.slot_locations.size() == 48 && card_free() == line + stride / 2,
                "the tier did not grow to the reserve and its margin");

        // Another process takes the reserve, then the margin too: no block goes.
        card_others = card_reserve;
        token(2);
        require(t.watermark_shrinks == 0 && t.slot_locations.size() == 48, "the tier gave way to another process's reserve");
        card_others = line + stride / 2;
        token(3);
        require(t.watermark_shrinks == 0 && t.slot_locations.size() == 48, "the tier gave way to another process past the reserve");

        // It releases all of that: the reserve is back, and the tier does not grow into it.
        card_others = 0;
        token(4);
        require(t.watermark_grows == 0 && t.slot_locations.size() == 48, "the tier grew into the reserve");

        // A process that held memory at the start exits: the tier grows into what is above the reserve and the margin.
        card_room += block;
        token(5);
        require(t.watermark_grows == 1 && t.slot_locations.size() == 64 && card_free() == line + stride / 2,
                "the tier did not grow into memory another process released");

        // This process itself goes past the margin: the next boundary gives a block back, as it does not for others.
        card_budget.hold(line);
        token(6);
        require(t.watermark_shrinks == 1 && t.slot_locations.size() < 64, "the tier did not give back what this process overran");
        card_budget.release(line);
        token(7);
        require(t.slot_locations.size() == 64, "the tier did not grow back");

        // Mid-decode the tier cannot give back at once: it says it will, so the allocation goes through.
        size_t freed = 0;
        require(t.reclaim_one(4096, &freed) == GGML_CUDA_RECLAIM_LATER && freed == 0 && t.slot_locations.size() == 64,
                "a discrete tier did not defer a decode allocation to its next boundary");
        t.finish(out.data(), 256);
    }
    std::printf("REAL_CUDA_VRAM_RECLAIM_PASS assertions=%d unified=%d\n", assertions, int(unified));
    return 0;
} catch (const std::exception & e) { open_gate(); std::fprintf(stderr, "VRAM_RECLAIM_FAIL: %s\n", e.what()); return 1; } }
