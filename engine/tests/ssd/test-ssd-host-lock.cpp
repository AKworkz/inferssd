// The CPU sink's registrar alone (ggml/src/ggml-ssd/host/page-lock.h), with fake page-locking that records its calls:
// nothing is page-locked outside an offer in the drive's shadow; before a range of slots loses its pages every locked
// slot of it is unlocked, a hand-off not yet taken is withdrawn and a registration in flight is waited for; a refused
// registration leaves the slot pageable; the rent-or-buy rule.
#include "../../ggml/src/ggml-ssd/host/page-lock.h"

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <string>
#include <vector>

static int failures = 0;
static void check(bool ok, const char * what) {
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", what); ++failures; }
}

namespace {
struct fake_pages {
    std::mutex mutex;
    std::condition_variable cv;
    std::vector<std::string> log;     // "lock <slot>", "locked <slot>", "unlock <slot>"
    char * arena = nullptr;
    size_t stride = 4096;
    bool hold = false;                // a registration waits while set
    size_t refuse = SIZE_MAX;         // this slot's registration fails
    void attach(kimi_host_lock & h) {
        h.lock_pages = [this](const void * p, size_t bytes, std::string * why) {
            const size_t slot = size_t(static_cast<const char *>(p) - arena) / stride;
            std::unique_lock<std::mutex> lock(mutex);
            log.push_back("lock " + std::to_string(slot));
            cv.notify_all();
            cv.wait(lock, [&] { return !hold; });
            if (slot == refuse || bytes != stride) { if (why) { *why = "refused"; } return false; }
            log.push_back("locked " + std::to_string(slot));
            return true;
        };
        h.unlock_pages = [this](const void * p) {
            std::lock_guard<std::mutex> lock(mutex);
            log.push_back("unlock " + std::to_string(size_t(static_cast<const char *>(p) - arena) / stride));
            return true;
        };
    }
    std::vector<std::string> calls() { std::lock_guard<std::mutex> lock(mutex); return log; }
    void release() { { std::lock_guard<std::mutex> lock(mutex); hold = false; } cv.notify_all(); }
};
// Waits (at most 5 s) until pred holds under the registrar's mutex.
template <class Pred> static bool eventually(kimi_host_lock & h, Pred pred) {
    for (int i = 0; i < 5000; ++i) {
        { std::lock_guard<std::mutex> lock(h.mutex); if (pred()) { return true; } }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    return false;
}
const auto any_slot = [](size_t) { return true; };
}

int main() {
    std::vector<char> arena(16 * 4096);
    const size_t slots = 16, stride = 4096;
    const double prior = 1000.0; // a registration's price before the first is measured (us)

    // Wanted slots wait for a shadow: nothing is page-locked without an offer, and an offer shorter than two
    // registrations hands nothing over.
    {
        fake_pages f; f.arena = arena.data(); f.stride = stride;
        kimi_host_lock h; f.attach(h); h.start();
        check(h.missed({ 3 }, { 3 }, 600.0, prior, slots, true) == 0, "below one registration: not wanted");
        check(h.missed({ 3 }, { 3 }, 600.0, prior, slots, true) == 1, "rent reaches one registration: wanted");
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        check(f.calls().empty(), "nothing page-locked without an offer");
        check(!h.offer(1999.0, prior, arena.data(), stride, slots, any_slot), "a short shadow hands nothing over");
        check(h.offer(2000.0, prior, arena.data(), stride, slots, any_slot), "a long shadow hands the wanted slot over");
        check(eventually(h, [&] { return h.registered == 1; }) && h.is_locked(3), "the slot is locked");
        check(!h.offer(1e9, prior, arena.data(), stride, slots, any_slot), "nothing more is wanted");
        h.stop();
        h.unlock(0, slots, arena.data(), stride);
        check(f.calls() == std::vector<std::string>({ "lock 3", "locked 3", "unlock 3" }), "lock, then unlock at the end");
    }

    // Unlock a range: every locked slot of it, none outside it; a hand-off not yet taken is withdrawn.
    {
        fake_pages f; f.arena = arena.data(); f.stride = stride;
        kimi_host_lock h; f.attach(h); h.start();
        for (size_t s : { 2, 5, 9 }) {
            h.missed({ s }, { s }, prior, prior, slots, true);
            check(h.offer(1e9, prior, arena.data(), stride, slots, any_slot), "offer");
            check(eventually(h, [&] { return h.busy_slot == SIZE_MAX && !h.pending && h.locked[s]; }), "locked");
        }
        h.unlock(4, 6, arena.data(), stride); // slots 4..9
        check(h.is_locked(2) && !h.is_locked(5) && !h.is_locked(9), "the range unlocked, slot 2 kept");
        auto c = f.calls();
        check(c.size() == 8 && c[6] == "unlock 5" && c[7] == "unlock 9", "unlock 5 and 9 only");
        h.stop();
        // A hand-off that the thread has not taken (no thread here) is withdrawn by an unlock of its range.
        kimi_host_lock g; f.attach(g); g.enabled = true;
        g.missed({ 7 }, { 7 }, prior, prior, slots, true);
        check(g.offer(1e9, prior, arena.data(), stride, slots, any_slot) && g.pending, "handed over");
        g.unlock(6, 2, arena.data(), stride);
        check(!g.pending, "the hand-off was withdrawn");
        h.unlock(0, slots, arena.data(), stride);
    }

    // A registration in flight is waited for: the unlock returns only after it, and then undoes it.
    {
        fake_pages f; f.arena = arena.data(); f.stride = stride; f.hold = true;
        kimi_host_lock h; f.attach(h); h.start();
        h.missed({ 4 }, { 4 }, prior, prior, slots, true);
        check(h.offer(1e9, prior, arena.data(), stride, slots, any_slot), "offer");
        check(eventually(h, [&] { return h.busy_slot == 4; }), "registration in flight");
        std::atomic<bool> returned{ false };
        std::thread t([&] { h.unlock(0, 8, arena.data(), stride); returned = true; });
        std::this_thread::sleep_for(std::chrono::milliseconds(30));
        check(!returned, "the unlock waits for the registration in flight");
        f.release();
        t.join();
        check(f.calls() == std::vector<std::string>({ "lock 4", "locked 4", "unlock 4" }) && !h.is_locked(4),
            "registered, then unregistered before the pages go");
    }

    // A refused registration leaves the slot pageable and is counted.
    {
        fake_pages f; f.arena = arena.data(); f.stride = stride; f.refuse = 6;
        kimi_host_lock h; f.attach(h); h.start();
        h.missed({ 6 }, { 6 }, prior, prior, slots, true);
        check(h.offer(1e9, prior, arena.data(), stride, slots, any_slot), "offer");
        check(eventually(h, [&] { return h.failures == 1; }) && !h.is_locked(6) && h.last_error == "refused",
            "refused: not locked, counted");
        h.unlock(0, slots, arena.data(), stride);
        check(f.calls() == std::vector<std::string>({ "lock 6" }), "nothing to unlock");
    }

    // A wanted slot that lost its pages before the shadow came is skipped.
    {
        fake_pages f; f.arena = arena.data(); f.stride = stride;
        kimi_host_lock h; f.attach(h); h.start();
        h.missed({ 1 }, { 1 }, prior, prior, slots, true);
        h.missed({ 8 }, { 8 }, prior, prior, slots, true);
        check(h.offer(1e9, prior, arena.data(), stride, slots, [](size_t s) { return s != 1; }), "offer");
        check(eventually(h, [&] { return h.registered == 1; }) && h.is_locked(8) && !h.is_locked(1), "slot 1 skipped");
        h.unlock(0, slots, arena.data(), stride);
    }

    // Before a page-locked copy is measured: no rent, one slot locked to measure it, nothing more until then.
    {
        fake_pages f; f.arena = arena.data(); f.stride = stride;
        kimi_host_lock h; f.attach(h); h.start();
        check(h.missed({ 5, 6 }, { 5, 6 }, 10.0, prior, slots, false) == 1, "unmeasured: the first slot is wanted at any gain");
        check(h.missed({ 7 }, { 7 }, 1e6, prior, slots, false) == 0, "unmeasured: one slot at a time");
        check(h.offer(1e9, prior, arena.data(), stride, slots, any_slot), "offer");
        check(eventually(h, [&] { return h.registered == 1; }) && h.is_locked(5), "the measuring slot is locked");
        check(h.missed({ 7 }, { 7 }, 1e6, prior, slots, false) == 0 && !h.offer(1e9, prior, arena.data(), stride, slots, any_slot),
            "unmeasured after a registration: nothing more");
        check(h.rent.empty() || h.rent[7] == 0.0, "unmeasured: no rent accrued");
        { std::lock_guard<std::mutex> lock(h.mutex); h.us_mean = prior; } // a registration as measured
        check(h.missed({ 7 }, { 7 }, 600.0, prior, slots, true) == 0 && h.missed({ 7 }, { 7 }, 600.0, prior, slots, true) == 1,
            "measured: rent or buy");
        h.unlock(0, slots, arena.data(), stride);
    }

    // The rent is the occupant's: a slot that holds another expert starts again from zero.
    {
        fake_pages f; f.arena = arena.data(); f.stride = stride;
        kimi_host_lock h; f.attach(h);
        check(h.missed({ 2 }, { 100 }, 600.0, prior, slots, true) == 0, "occupant 100: 600 of 1000");
        check(h.missed({ 2 }, { 101 }, 600.0, prior, slots, true) == 0, "occupant 101: its own 600, not 1200");
        check(h.missed({ 2 }, { 101 }, 600.0, prior, slots, true) == 1, "occupant 101: 1200, wanted");
    }

    // The read ring (kimi_ring_lock): every buffer registered once per allocation; a reallocation unregisters them all
    // first and registers again; a refused registration undoes the ones before it and is not retried until the next
    // allocation; at the end every registration has its unregistration.
    {
        std::vector<char> ring(24 * stride);
        std::vector<std::pair<const void *, size_t>> buffers;
        for (size_t i = 0; i < 24; ++i) { buffers.push_back({ ring.data() + i * stride, stride }); }
        std::vector<const void *> live;
        size_t calls = 0, refuse_at = SIZE_MAX;
        kimi_ring_lock r;
        r.lock_pages = [&](const void * p, size_t, std::string * why) {
            if (calls++ == refuse_at) { if (why) { *why = "refused"; } return false; }
            live.push_back(p); return true;
        };
        r.unlock_pages = [&](const void * p) {
            const auto it = std::find(live.begin(), live.end(), p);
            if (it == live.end()) { return false; }
            live.erase(it); return true;
        };
        check(r.lock(buffers) && r.locked() && live.size() == 24 && r.registered == 24, "ring: every buffer page-locked");
        check(r.lock(buffers) && r.registered == 24, "ring: once per allocation");
        r.unlock();                                                   // the ring grows
        check(live.empty() && !r.locked() && r.unregistered == 24, "ring: unlocked before a reallocation");
        refuse_at = calls + 5;
        std::string why;
        check(!r.lock(buffers, &why) && live.empty() && why == "refused" && r.refused == 1, "ring: a refusal undoes the rest");
        check(!r.lock(buffers) && calls == refuse_at + 1, "ring: not tried again before the next allocation");
        r.unlock();                                                   // the next allocation
        check(r.lock(buffers) && live.size() == 24, "ring: tried again after a reallocation");
        r.unlock();                                                   // the sink goes
        check(live.empty() && r.registered == r.unregistered && r.registered == 24 + 5 + 24, "ring: registrations balance");
    }

    if (failures) { std::fprintf(stderr, "%d failures\n", failures); return 1; }
    std::printf("KIMI_HOST_LOCK_OK\n");
    return 0;
}
