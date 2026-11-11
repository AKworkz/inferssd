#pragma once
// Page-locking of a RAM cache's slots for the GPU's copies (kimi_vram_host_api v2 register_host), on a discrete GPU:
// the CPU sink's registrar (ggml-cpu/ssd/cpu-sink.cpp). Kept apart from the sink so that its rules are tested alone
// (tests/ssd/test-ssd-host-lock.cpp, with fake page-locking).
//   - Which slots: a registration is bought once and the moves it allows pay per use (rent or buy). The decode plan
//     reports, per layer, the time the moves it could not make from unlocked slots would have saved (missed); each
//     such slot accrues its share, and is wanted once that reaches one registration. The rent is the occupant's: a
//     slot that holds another expert starts again from zero (its next occupant may never be wanted). A gain is only
//     as good as the copy price it was computed at: before a page-locked copy has been measured nothing accrues, and
//     the first slot the plan would move from at all is wanted alone; its registration measures the copy (the GPU
//     tier times one copy of the first range registered), and no other slot is locked until then.
//   - When: the registrar thread takes one wanted slot at a time, handed over only while a decode layer waits for reads
//     expected to last longer than twice a registration (offer, "the drive's shadow"). A registration holds the CUDA
//     driver for about a millisecond per DeepSeek slot (RTX 5090: 1.1-1.3 ms for 19 MB, the same with huge pages) and
//     delays other threads' CUDA calls meanwhile; in that shadow none is on the critical path.
//   - Until when: a locked slot stays locked until its pages are discarded or unmapped. The owner of the pages calls
//     unlock(first, count) before MADV_DONTNEED or munmap (sink_cache::before_discard): a hand-off not yet taken is
//     withdrawn, a registration in flight is waited for, and every locked slot of the range is unregistered. Else the
//     device would keep the old pages and copy stale bytes.
#include <algorithm>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <chrono>
#include <deque>
#include <functional>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

struct kimi_host_lock {
    // Page-lock [pointer, pointer + bytes) for the device's copies (false: refused, why says why), and undo it.
    std::function<bool(const void * pointer, size_t bytes, std::string * why)> lock_pages;
    std::function<bool(const void * pointer)> unlock_pages;

    std::mutex mutex;
    std::condition_variable cv;
    std::thread thread;
    bool stop_requested = false, enabled = false, decided = false;
    std::vector<uint8_t> locked;              // per slot
    std::deque<size_t> wanted;                // slots whose missed moves are worth a registration
    std::vector<double> rent;                 // per slot: the time the plan's moves from its occupant would have saved
    std::vector<uint64_t> owner;              // per slot: that occupant
    const char * pending = nullptr;           // handed to the thread: [pending, pending + pending_bytes) of pending_slot
    size_t pending_slot = SIZE_MAX, pending_bytes = 0;
    size_t busy_slot = SIZE_MAX;              // being registered now
    double us_mean = 0, us_total = 0, max_us = 0;
    uint64_t registered = 0, failures = 0, unregistered = 0, signals = 0, wanted_total = 0;
    std::string last_error;
    static constexpr size_t wanted_cap = 256;

    ~kimi_host_lock() { stop(); }

    void start() {
        enabled = true;
        thread = std::thread([this] { run(); });
    }
    // Ends the thread; a hand-off not yet taken is dropped. Locked slots stay locked (unlock them first or after).
    void stop() {
        {
            std::lock_guard<std::mutex> lock(mutex);
            stop_requested = true;
            pending = nullptr;
        }
        cv.notify_all();
        if (thread.joinable()) { thread.join(); }
    }
    // The registration's price: the measured mean, else the prior.
    double price(double prior_us) const { return us_mean > 0 ? us_mean : prior_us; }

    // The decode plan's layer: the moves it would have made from these unlocked slots (held by these occupants) saved
    // gain_us in all, at a measured copy price or (measured false) at none. Measured: each slot accrues its share to
    // its occupant, and a slot whose rent reaches one registration (prior_us before the first) is wanted. Not
    // measured: the first slot is wanted when no slot has been registered, tried or handed over yet. Returns the
    // slots it made wanted.
    size_t missed(const std::vector<size_t> & slots, const std::vector<uint64_t> & occupants, double gain_us,
            double prior_us, size_t slot_count, bool measured) {
        if (slots.empty() || slots.size() != occupants.size() || !(gain_us > 0)) { return 0; }
        std::lock_guard<std::mutex> lock(mutex);
        if (!measured) {
            if (registered || failures || pending || busy_slot != SIZE_MAX || !wanted.empty()) { return 0; }
            wanted.push_back(slots.front());
            ++wanted_total;
            return 1;
        }
        if (rent.size() != slot_count) { rent.assign(slot_count, 0.0); owner.assign(slot_count, UINT64_MAX); }
        const double p = price(prior_us), share = gain_us / double(slots.size());
        size_t made = 0;
        for (size_t i = 0; i < slots.size(); ++i) {
            const size_t s = slots[i];
            if (s >= rent.size()) { continue; }
            if (owner[s] != occupants[i]) { owner[s] = occupants[i]; rent[s] = 0.0; }
            double & r = rent[s];
            if (r < p && r + share >= p && wanted.size() < wanted_cap) { wanted.push_back(s); ++wanted_total; ++made; }
            r += share;
        }
        return made;
    }

    // In a decode layer's wait for reads: hands the thread one wanted slot when the reads left (left_us) are expected
    // to take longer than twice a registration (prior_us before the first) and the thread is idle. lockable(slot): the
    // slot holds pages now. True when a slot was handed over.
    bool offer(double left_us, double prior_us, char * arena, size_t stride, size_t slot_count,
            const std::function<bool(size_t)> & lockable) {
        std::lock_guard<std::mutex> lock(mutex);
        if (!enabled || pending || busy_slot != SIZE_MAX || left_us < 2.0 * price(prior_us)) { return false; }
        if (locked.size() != slot_count) { locked.assign(slot_count, 0); }
        size_t slot = SIZE_MAX;
        while (!wanted.empty() && slot == SIZE_MAX) {
            const size_t s = wanted.front();
            wanted.pop_front();
            if (s < locked.size() && !locked[s] && lockable(s)) { slot = s; }
        }
        if (slot == SIZE_MAX) { return false; }
        pending = arena + slot * stride;
        pending_slot = slot;
        pending_bytes = stride;
        ++signals;
        cv.notify_all();
        return true;
    }

    // Before slots [first, first + count) lose their pages (arena: their base, stride: a slot's bytes).
    void unlock(size_t first, size_t count, char * arena, size_t stride) {
        std::unique_lock<std::mutex> lock(mutex);
        const auto inside = [&](size_t s) { return s >= first && s - first < count; };
        if (pending && inside(pending_slot)) { pending = nullptr; }
        cv.wait(lock, [&] { return !inside(busy_slot); });
        for (size_t s = first; s < locked.size() && s - first < count; ++s) {
            if (!locked[s]) { continue; }
            if (!unlock_pages(arena + s * stride)) { ++failures; }
            locked[s] = 0;
            ++unregistered;
        }
    }

    bool is_locked(size_t slot) {
        std::lock_guard<std::mutex> lock(mutex);
        return slot < locked.size() && locked[slot];
    }
    std::vector<uint8_t> snapshot() {
        std::lock_guard<std::mutex> lock(mutex);
        return locked;
    }
    size_t locked_count() {
        std::lock_guard<std::mutex> lock(mutex);
        return size_t(std::count(locked.begin(), locked.end(), uint8_t(1)));
    }

    // [kimi_host_register] receipt (mutex held by the caller or no thread running).
    void report(FILE * out, const char * reason, size_t stride) const {
        const size_t n = size_t(std::count(locked.begin(), locked.end(), uint8_t(1)));
        std::fprintf(out, "[kimi_host_register] reason=%s enabled=%d locked=%zu bytes=%llu registered=%llu failures=%llu "
            "unregistered=%llu signals=%llu wanted=%llu us_mean=%.0f us_max=%.0f us_total=%.0f error=%s\n",
            reason, int(enabled), n, (unsigned long long) (uint64_t(n) * stride), (unsigned long long) registered,
            (unsigned long long) failures, (unsigned long long) unregistered, (unsigned long long) signals,
            (unsigned long long) wanted_total, us_mean, max_us, us_total, last_error.empty() ? "-" : last_error.c_str());
    }

private:
    void run() {
        std::unique_lock<std::mutex> lock(mutex);
        for (;;) {
            cv.wait(lock, [&] { return stop_requested || pending; });
            if (stop_requested) { return; }
            const char * pointer = pending;
            const size_t slot = pending_slot, bytes = pending_bytes;
            pending = nullptr;
            busy_slot = slot;
            lock.unlock();
            std::string why;
            const auto start = std::chrono::steady_clock::now();
            const bool ok = lock_pages(pointer, bytes, &why);
            const double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - start).count();
            lock.lock();
            busy_slot = SIZE_MAX;
            if (ok && slot < locked.size()) {
                locked[slot] = 1;
                ++registered;
                us_total += us;
                max_us = std::max(max_us, us);
                us_mean = us_mean > 0 ? 0.9 * us_mean + 0.1 * us : us;
                if (registered % 256 == 0) { report(stderr, "periodic", bytes); }
            } else if (!ok) {
                ++failures;
                last_error = why;
            }
            cv.notify_all();
        }
    }
};

// The read ring's buffers page-locked for the GPU tier's copies (B3): every buffer of the ring one range, registered
// together once per allocation of the ring, at a quiescent point the owner picks, and unregistered before the owner
// reallocates or frees them (unlock). A refused registration undoes the ones made before it: the ring stays pageable,
// and is not tried again before the next allocation (an unlock).
struct kimi_ring_lock {
    std::function<bool(const void * pointer, size_t bytes, std::string * why)> lock_pages;
    std::function<bool(const void * pointer)> unlock_pages;
    std::vector<const void *> ranges;   // the registered buffers
    bool tried = false;
    uint64_t registered = 0, unregistered = 0, refused = 0;
    bool locked() const { return !ranges.empty(); }
    // buffers: every (pointer, bytes) of the ring. True when all of them are page-locked.
    bool lock(const std::vector<std::pair<const void *, size_t>> & buffers, std::string * why = nullptr) {
        if (tried) { return locked(); }
        tried = true;
        for (const auto & b : buffers) {
            if (!b.first || !b.second) { continue; }
            if (!lock_pages(b.first, b.second, why)) {
                ++refused;
                unlock();
                tried = true;
                return false;
            }
            ranges.push_back(b.first);
            ++registered;
        }
        return locked();
    }
    // Before the buffers are reallocated or freed; the next lock tries again.
    void unlock() {
        for (const void * p : ranges) { unlock_pages(p); ++unregistered; }
        ranges.clear();
        tried = false;
    }
};
