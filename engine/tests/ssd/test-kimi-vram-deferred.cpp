#include "explicit-fixture-inputs.h"
// CUDA lifecycle test of deferred VRAM uploads (GGML_KIMI_CPU_SINK_VRAM_DEFERRED_UPLOAD=1). A borrowed upload
// outlives finish, keeps its slot out of every eviction, is published only by collect, and release waits for it.
// A hook on the upload thread holds a copy in flight or makes it fail, independently of GPU/CPU speed.
#include "gpu-kernels.h"
#include "vram-tier.cpp"
#include "test-kimi-vram-format-fixture.h"
#include <future>

namespace {
std::mutex gate_mutex;
std::condition_variable gate_cv;
bool held = false, entered = false, fail_next = false;
bool upload_gate(void *) {
    std::unique_lock<std::mutex> lock(gate_mutex);
    entered = true; gate_cv.notify_all();
    gate_cv.wait(lock, [] { return !held; });
    if (fail_next) { fail_next = false; return false; }
    return true;
}
void hold(bool fail = false) { std::lock_guard<std::mutex> lock(gate_mutex); held = true; entered = false; fail_next = fail; }
void open_gate() { std::lock_guard<std::mutex> lock(gate_mutex); held = false; gate_cv.notify_all(); }
bool copy_started() {
    std::unique_lock<std::mutex> lock(gate_mutex);
    return gate_cv.wait_for(lock, std::chrono::seconds(5), [] { return entered; });
}
bool done_within(std::future<void> & f, int ms) { return f.wait_for(std::chrono::milliseconds(ms)) == std::future_status::ready; }
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
    std::vector<unsigned char> payload(stride);
    for (size_t i = 0; i < payload.size(); ++i) { payload[i] = uint8_t(i * 131 + 7); }
    const void * data[] = {payload.data(), payload.data() + role, payload.data() + 2 * role};
    std::vector<float> out(256);
    int slots[2] = {}, queued = 0, count = -1;
    kimi_vram_request collected[4] = {};
    int32_t aborted[4] = {};
    kimi_vram_ready_lease lease = {};
    int ready[2] = {};
    struct cleanup { ~cleanup() { open_gate(); } } cleanup_on_error;
    tier t(2 * stride, inputs.tier_policy(), nullptr, test_cpu_kernels()); // two slots, one global pool for every layer
    t.deferred_hook = upload_gate;

    // The upload outlives finish; its slot is reserved, not ready, and collect does not publish it.
    const auto a = key(1, 0);
    hold();
    t.begin(31, 1, true);
    lifecycle_prepare(t,&a, 1,slots);
    require(slots[0] < 0 && t.table->admissions[0] >= 0, "first request was not an admission");
    const int slot_a = t.table->admissions[0];
    t.admit_deferred(a, data, &queued);
    require(queued == 1 && copy_started(), "borrowed upload was not deferred to the upload thread");
    t.finish(out.data(), 256);
    require(t.table->slots[slot_a].uploading && !t.table->slots[slot_a].valid, "finish published or dropped an unfinished upload");
    t.acquire_ready(31, &a, 1, &lease, ready);
    require(!ready[0], "an unfinished upload counted as ready");
    t.release_ready(lease);
    t.collect(-2, collected, aborted, 4, &count);
    require(count == 0, "collect published an upload still in flight");

    // Another layer cannot evict the uploading slot: of two candidates only one is admitted, into the free slot.
    const kimi_vram_request bc[2] = {key(2, 1), key(2, 2)};
    lifecycle_prepare(t,bc, 2,slots);
    require(t.table->admissions[0] != slot_a && t.table->admissions[1] != slot_a, "an uploading slot was chosen as a victim");
    require((t.table->admissions[0] >= 0) != (t.table->admissions[1] >= 0), "expected exactly one admission into the free slot");
    t.finish(out.data(), 256);

    // collect(layer) waits for that layer's copy, then publishes the borrowed bytes.
    auto waiter = std::async(std::launch::async, [&] { t.collect(1, collected, aborted, 4, &count); });
    require(!done_within(waiter, 150), "collect(layer) returned while its upload was held");
    open_gate();
    require(done_within(waiter, 5000), "collect(layer) did not return after the upload completed");
    waiter.get();
    require(count == 1 && !aborted[0] && vram_slot_table::same(collected[0], a), "collect did not return the completed upload");
    require(t.table->slots[slot_a].valid && !t.table->slots[slot_a].uploading, "collect did not publish the slot");
    std::vector<unsigned char> copied(stride);
    cuda_check(cudaMemcpy(copied.data(), t.slot_base(size_t(slot_a)), stride, cudaMemcpyDeviceToHost));
    require(!std::memcmp(copied.data(), payload.data(), stride), "deferred upload corrupted the borrowed payload");
    t.acquire_ready(31, &a, 1, &lease, ready);
    require(ready[0], "published upload is not ready");
    t.release_ready(lease);
    t.begin(31, 2, true);
    lifecycle_prepare(t,&a, 1,slots);
    require(slots[0] == slot_a, "published upload did not serve a hit");
    {
        char error[256] = {}; int c = 1;
        require(deferred_collect(&t, KIMI_VRAM_COLLECT_IF_IDLE, collected, aborted, 4, &c, error, sizeof(error)) == 0 && c == 0 && t.table,
                "a reclaimer's collect inside a layer did not leave the layer alone");
    }
    {
        char error[256] = {}; int c = 0;
        require(deferred_collect(&t, KIMI_VRAM_COLLECT_NONE, collected, aborted, 4, &c, error, sizeof(error)) != 0, "collect accepted inside a layer");
        require(!t.table && !t.arena, "refused collect did not release the tier");
    }

    // release waits for an in-flight deferred upload before freeing the arena.
    const auto d = key(3, 5);
    hold();
    t.begin(31, 3, true);
    lifecycle_prepare(t,&d, 1,slots);
    t.admit_deferred(d, data, &queued);
    require(queued == 1 && copy_started(), "deferred upload before release did not start");
    t.finish(out.data(), 256);
    auto releaser = std::async(std::launch::async, [&] { t.release(); });
    require(!done_within(releaser, 150), "release returned while a deferred upload was in flight");
    open_gate();
    require(done_within(releaser, 5000), "release did not return after the upload completed");
    releaser.get();
    require(!t.arena && t.blocks.empty() && !t.table && t.deferred.empty(), "release left deferred work or the arena");

    // A failing copy surfaces at the next collect, which releases the tier.
    hold(true);
    t.begin(31, 4, true);
    lifecycle_prepare(t,&d, 1,slots);
    t.admit_deferred(d, data, &queued);
    require(queued == 1 && copy_started(), "failing upload did not start");
    t.finish(out.data(), 256);
    open_gate();
    {
        char error[256] = {}; int c = 0;
        require(deferred_collect(&t, KIMI_VRAM_COLLECT_ALL, collected, aborted, 4, &c, error, sizeof(error)) != 0 &&
                std::strstr(error, "injected deferred VRAM upload error"), "deferred upload failure was hidden");
        require(!t.arena && t.blocks.empty() && !t.table && t.deferred.empty(), "upload failure left deferred work or the arena");
    }

    // Capacity: with one in-flight place, the second admission waits for the first copy.
    t.deferred_capacity = 1;
    const kimi_vram_request ef[2] = {key(4, 6), key(4, 7)};
    hold();
    t.begin(31, 5, true);
    lifecycle_prepare(t,ef, 2,slots);
    t.admit_deferred(ef[0], data, &queued);
    require(queued == 1 && copy_started(), "first capacity upload did not start");
    int queued_second = 0;
    auto second = std::async(std::launch::async, [&] { t.admit_deferred(ef[1], data, &queued_second); });
    require(!done_within(second, 150), "admission exceeded the deferred upload capacity");
    open_gate();
    require(done_within(second, 5000), "capacity wait did not end");
    second.get();
    require(queued_second == 1 && t.deferred_capacity_waits == 1, "second upload was not deferred after the capacity wait");
    t.finish(out.data(), 256);
    t.collect(-1, collected, aborted, 4, &count);
    require(count == 2, "collect(-1) did not publish both uploads");

    // Normal server shutdown can race the final collect: the copy is allowed
    // to finish while its table entry is still reserved as uploading. The
    // destructor must wait, abort that unpublished reservation, and free the
    // elastic blocks without treating the live tail as corruption.
    {
        auto * pending = new tier(2 * stride, inputs.tier_policy(), nullptr, test_cpu_kernels());
        pending->deferred_hook = upload_gate;
        hold();
        pending->begin(32, 1, true);
        lifecycle_prepare(*pending, &a, 1, slots);
        pending->admit_deferred(a, data, &queued);
        require(queued == 1 && copy_started(), "shutdown upload did not start");
        pending->finish(out.data(), 256);
        auto destroyer = std::async(std::launch::async, [&] { delete pending; });
        require(!done_within(destroyer, 150), "destructor returned while its deferred upload was in flight");
        open_gate();
        require(done_within(destroyer, 5000), "destructor did not drain an uncollected deferred upload");
        destroyer.get();
    }
    std::printf("REAL_CUDA_DEFERRED_UPLOAD_PASS assertions=%d uploads=%llu published=%llu\n", assertions,
        (unsigned long long) t.deferred_uploads, (unsigned long long) t.deferred_published);
    return 0;
} catch (const std::exception & e) { open_gate(); std::fprintf(stderr, "DEFERRED_UPLOAD_FAIL: %s\n", e.what()); return 1; } }
