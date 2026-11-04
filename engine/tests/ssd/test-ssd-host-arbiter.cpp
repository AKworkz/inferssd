// SPDX-License-Identifier: Apache-2.0
#include "../../ggml/src/ggml-ssd/host/arbiter.h"
#include "../../ggml/src/ggml-ssd/host/ledger.h"
#include <atomic>
#include <condition_variable>
#include <future>
#include <iostream>
#include <map>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <vector>

using namespace ssd::host;
constexpr uint64_t MiB = 1ull << 20, GiB = 1ull << 30;
static void require(bool value, const char * why) { if (!value) { throw std::runtime_error(why); } }
template<class F> static void must_throw(F f, const char * why) {
    bool threw = false; try { f(); } catch (const std::exception &) { threw = true; }
    require(threw, why);
}
template<class T> static T checked(ssd::result<T> value) {
    if (!value) { throw std::runtime_error(value.error); }
    return std::move(value.value);
}
static void checked(ssd::result<void> value) { if (!value) { throw std::runtime_error(value.error); } }
// Tests adapt result/status explicitly; product code never receives these
// throwing helpers.
struct checked_client {
    std::unique_ptr<context_client> raw;
    explicit checked_client(std::unique_ptr<context_client> value) : raw(std::move(value)) {}
    void configure(const client_config & c) { checked(raw->configure(c)); }
    snapshot state() const { return checked(raw->state()); }
    void set_prefill_pages(uint64_t p) { checked(raw->set_prefill_pages(p)); }
    void clear_prefill_pages() { checked(raw->clear_prefill_pages()); }
    room measure(uint64_t x, bool full = false) const { return checked(raw->measure(x, full)); }
    room project_decode_capacity(uint64_t x = 0, bool full = false) const { return checked(raw->project_decode_capacity(x, full)); }
    void set_reclaimer(const actions & a) { checked(raw->set_reclaimer(a)); }
    void clear_reclaimer() { raw->clear_reclaimer(); }
    allocation_ticket take(uint64_t h, uint64_t d = 0, requester who = requester::outside, bool finite = false,
            allocation_class kind = allocation_class::external) {
        auto out = raw->take(h, d, who, finite, kind);
        if (!out && out.code == ssd::status::invalid) { throw std::runtime_error(out.error); }
        return std::move(out.value);
    }
    bool update_file_reserve() { return checked(raw->update_file_reserve()); }
    void begin_interval(uint64_t h) { checked(raw->begin_interval(h)); }
    bool end_interval(bool consecutive, uint64_t h, uint64_t & unknown) { return checked(raw->end_interval(consecutive, h, unknown)); }
    void close() { raw->close(); }
};
struct test_owner {
    std::shared_ptr<process_arbiter> raw;
    explicit test_owner(process_arbiter::reader reader) : raw(std::make_shared<process_arbiter>(std::move(reader))) {}
    std::unique_ptr<checked_client> make_client() {
        return std::make_unique<checked_client>(checked(raw->make_client()));
    }
};
struct files {
    std::atomic<uint64_t> limit{8*GiB}, current{GiB}, available{8*GiB}, file{0}, refault{0};
    std::atomic<bool> readable{true};
    std::atomic<bool> fail_usage{false};
    bool read(const std::string & path, std::string & text) const {
        if (!readable) { return false; }
        if (path == "/proc/self/cgroup") { text = "0::/fixture\n"; }
        else if (path == "/sys/fs/cgroup/fixture/memory.max") { text = std::to_string(limit.load()); }
        else if (path == "/sys/fs/cgroup/fixture/memory.current") {
            if (fail_usage) { return false; }
            text = std::to_string(current.load());
        }
        else if (path == "/sys/fs/cgroup/fixture/memory.stat") {
            text = "active_file 0\ninactive_file 0\nfile " + std::to_string(file.load()) +
                "\nworkingset_refault_file " + std::to_string(refault.load()) + "\n";
        } else if (path == "/proc/meminfo") { text = "MemAvailable: " + std::to_string(available.load()/1024) + " kB\n"; }
        else { return false; }
        return true;
    }
    process_arbiter::reader reader() { return [this](const std::string & path, std::string & out) { return read(path, out); }; }
};
struct barrier {
    std::mutex mutex; std::condition_variable changed;
    unsigned count, generation = 0;
    explicit barrier(unsigned n) : count(n), original(n) {}
    void wait() {
        std::unique_lock<std::mutex> lock(mutex); const auto old = generation;
        if (!--count) { ++generation; count = original; changed.notify_all(); }
        else { changed.wait(lock, [&] { return generation != old; }); }
    }
  private: const unsigned original;
};
static client_config config(bool unified = false) { client_config c; c.unified = unified; return c; }
static void equal_room(const room & a, const room & b) {
    require(int(a.cgroup) == int(b.cgroup) && a.cgroup_shortage == b.cgroup_shortage &&
        a.cgroup_surplus == b.cgroup_surplus && a.system == b.system && a.available == b.available &&
        a.floor == b.floor && a.system_shortage == b.system_shortage && a.system_surplus == b.system_surplus &&
        a.boost_headroom == b.boost_headroom && a.fits() == b.fits() && a.surplus() == b.surplus(), "one-client equation changed");
}
// One client measures what the ledger measures over the same files, with the open prefill's pages, the file
// reserve and the margin the client set.
static void serial_equivalence() {
    files f; auto owner = std::make_shared<test_owner>(f.reader()); auto a = owner->make_client();
    must_throw([&] { auto t = a->take(1); }, "unconfigured client admitted");
    a->configure(config(true)); a->configure(config(true));
    kimi_host_ledger old; old.read = f.reader(); old.cgroup = "/sys/fs/cgroup/fixture";
    old.unified = true; old.reserve = old.margin = 512*MiB; old.reclaimable_file = true;
    old.page_size = uint64_t(sysconf(_SC_PAGESIZE));
    for (const uint64_t pages : {uint64_t(0), GiB, 3*GiB}) {
        a->set_prefill_pages(pages); old.prefill_pages = pages;
        for (const uint64_t extra : {uint64_t(0), GiB, 7*GiB}) {
            equal_room(a->measure(extra), old.measure(extra));
            equal_room(a->measure(extra, true), old.measure(extra, true));
        }
    }
    a->clear_prefill_pages(); old.prefill_pages = 0;
    f.file = 256*MiB; f.refault = 7;
    require(a->update_file_reserve() == old.update_file_reserve(), "first file reserve changed");
    f.refault = 10;
    require(a->update_file_reserve() == old.update_file_reserve(), "refault growth changed");
    require(a->state().file_reserve == old.file_reserve, "refault reserve differs");
    a->begin_interval(GiB);
    f.available -= 2*GiB; uint64_t x = 0;
    require(a->end_interval(true, GiB, x) && x == 2*GiB && a->state().margin == 2*GiB, "serial interval changed");
    old.margin = 2*GiB;
    f.readable = false; equal_room(a->measure(1), old.measure(1));
    require(!a->take(1), "unreadable budget admitted real allocation");
}
static void holds_and_pending() {
    files f; auto owner = std::make_shared<test_owner>(f.reader());
    auto a = owner->make_client(), b = owner->make_client(); a->configure(config()); b->configure(config());
    a->set_prefill_pages(GiB); b->set_prefill_pages(2*GiB);
    require(a->state().all_prefill_pages == 3*GiB, "holds were not summed");
    const auto before = b->measure(0);
    require(a->project_decode_capacity().cgroup_surplus == before.cgroup_surplus + GiB, "projection omitted wrong hold");
    require(b->measure(0).cgroup_surplus == before.cgroup_surplus, "projection mutated published hold");
    a->clear_prefill_pages(); require(b->state().all_prefill_pages == 2*GiB, "A cleared B hold");
    b->set_prefill_pages(GiB); b->set_prefill_pages(GiB);
    require(b->state().all_prefill_pages == GiB, "repeated hold added");
    b->set_prefill_pages(9*GiB); require(!b->take(1) && b->state().prefill_pages == 9*GiB, "refusal lost prefill hold");
    require(bool(b->take(0)), "a shortage refused a take of nothing");
    b->clear_prefill_pages(); f.limit = 2*GiB; f.current = 0; f.available = 2*GiB;
    auto t = a->take(GiB); require(bool(t), "first take failed");
    require(!b->take(GiB), "unmaterialized allocation granted twice");
    require(a->state().pending_bytes == GiB && a->state().granted == GiB, "pending/grant not published");
    allocation_ticket moved = std::move(t); require(!t && bool(moved), "ticket move failed");
    moved.cancel(); moved.commit(); require(a->state().pending_bytes == 0 && a->state().granted == 0, "cancel did not roll back");
    auto committed = b->take(GiB); f.current = GiB; f.available = GiB; committed.commit();
    require(!a->take(GiB) && b->state().pending_bytes == 0 && b->state().granted == GiB, "commit lost OS accounting");
    a->set_prefill_pages(UINT64_MAX); require(!b->take(1), "saturated holds admitted");
    a->clear_prefill_pages(); a->close(); require(b->state().all_prefill_pages == 0, "close left peer hold");
    must_throw([&] { a->measure(0); }, "closed client remained live");
}
static void races_and_reclaim() {
    files f; f.limit = f.available = 2*GiB; f.current = 0;
    auto owner = std::make_shared<test_owner>(f.reader()); auto a = owner->make_client(), b = owner->make_client();
    a->configure(config(true)); b->configure(config(true));
    barrier ready(2), held(2); std::atomic<unsigned> granted{0};
    auto run = [&](checked_client & client) { ready.wait(); auto t = client.take(GiB); granted += bool(t); held.wait(); };
    std::thread ta(run, std::ref(*a)), tb(run, std::ref(*b)); ta.join(); tb.join();
    require(granted == 1 && a->state().pending_bytes == 0, "barrier race over-admitted");
    require(!a->take(UINT64_MAX, 1), "DRAM overflow admitted");
    std::vector<int> order;
    actions reclaim;
    reclaim.collect_deferred = [&] { (void)b->state(); order.push_back(1); };
    reclaim.reclaim_tier = [&](uint64_t) { (void)a->measure(0); order.push_back(2); f.available = GiB; return true; };
    reclaim.reclaim_cache = [&](uint64_t) { (void)b->state(); order.push_back(3); f.current = 0; return true; };
    a->set_reclaimer(reclaim);
    must_throw([&] { a->set_reclaimer(reclaim); }, "a second reclaimer replaced the context's");
    // A take that fits measures once and asks nothing.
    auto fits = a->take(256*MiB); require(bool(fits) && order.empty(), "a fitting take ran the reclaimer");
    fits.cancel();
    // Every take of the context runs its reclaimer, in order, without naming it; a peer's take runs none.
    f.current = 2*GiB; f.available = 0;
    require(!b->take(256*MiB) && order.empty(), "a peer's take ran this context's reclaimer");
    auto ticket = a->take(256*MiB);
    require(bool(ticket) && order == std::vector<int>({1,2,3}), "reclaim order or unlocked reentry changed");
    ticket.cancel();
    // The RAM cache's own growth asks only the RAM cache (the system term stays short); the GPU tier's growth
    // asks nothing.
    order.clear(); f.current = 2*GiB; f.available = 0;
    require(!a->take(256*MiB, 0, requester::ram_cache, false, allocation_class::tracked) &&
        order == std::vector<int>({3}), "the RAM cache's growth asked the GPU tier");
    order.clear(); f.current = 2*GiB; f.available = 0;
    require(!a->take(0, 256*MiB, requester::gpu_tier, false, allocation_class::tracked) && order.empty(),
        "the GPU tier's growth ran the reclaimer");
    a->clear_reclaimer();
    require(!a->take(256*MiB) && order.empty(), "a cleared reclaimer still ran");
    f.current = 0; f.available = 2*GiB;
    auto changed = config(true); changed.margin_bytes = GiB;
    must_throw([&] { a->configure(changed); }, "client changed immutable config");
    auto c = owner->make_client(); must_throw([&] { c->configure(config(false)); }, "incompatible physical domain admitted");
}
static void process_reserves_and_rollback() {
    files f; auto owner = std::make_shared<test_owner>(f.reader()); auto a = owner->make_client(), b = owner->make_client();
    auto ca = config(true), cb = config(true);
    ca.margin_bytes = GiB; ca.explicit_reserve_bytes = 256*MiB;
    cb.reserve_bytes = GiB; cb.margin_bytes = 2*GiB; cb.explicit_reserve_bytes = 512*MiB;
    a->configure(ca); b->configure(cb);
    require(a->state().reserve == GiB && a->state().margin == 2*GiB && a->state().explicit_reserve == 512*MiB,
        "common reserves were added instead of applied once");
    try { auto t = a->take(MiB); require(bool(t), "exception setup take failed"); throw std::bad_alloc(); }
    catch (const std::bad_alloc &) {}
    require(a->state().pending_bytes == 0 && a->state().granted == 0, "exception leaked a pending grant");
    a->close(); b->close(); auto next = owner->make_client();
    must_throw([&] { next->configure(config(false)); }, "physical budget domain changed after close");
    next->configure(config(true));
    require(next->state().reserve == 512*MiB && next->state().margin == 2*GiB && next->state().explicit_reserve == 0,
        "reserve release or monotonic margin changed");
}
static void interval_isolation_and_close() {
    files f; auto owner = std::make_shared<test_owner>(f.reader()); auto a = owner->make_client(), b = owner->make_client();
    a->configure(config(true)); b->configure(config(true));
    a->begin_interval(0); b->begin_interval(0); f.available -= 2*GiB; uint64_t untracked = 99;
    require(!a->end_interval(true, 0, untracked) && !untracked, "A learned B overlap");
    require(!b->end_interval(true, 0, untracked) && a->state().overlap_skips == 1 && b->state().overlap_skips == 1, "intervals were mixed");
    a->begin_interval(0); b->set_prefill_pages(1);
    require(!a->end_interval(true, 0, untracked) && a->state().overlap_skips == 2, "peer hold did not invalidate sample");
    b->clear_prefill_pages(); a->begin_interval(0);
    auto t = a->take(256*MiB); f.available -= 256*MiB; t.commit();
    require(!a->end_interval(true, 0, untracked) && untracked == 0, "own grant was learned as driver growth");
    auto pending = b->take(1); require(bool(pending), "close setup failed");
    std::promise<void> started; std::atomic<bool> closed{false};
    std::thread closer([&] { started.set_value(); b->close(); closed = true; }); started.get_future().wait();
    // Wait for the actual closing flag, rather than asserting elapsed sleeps.
    for (;;) {
        try { (void)b->state(); std::this_thread::yield(); }
        catch (const std::runtime_error &) { break; }
    }
    require(!closed, "close returned with pending allocation");
    pending.cancel(); closer.join(); require(closed, "close did not finish after ticket cancellation");
    require(a->state().pending_bytes == 0, "closed client left process pending");
}
static void close_waits_for_reclaimer() {
    files f; f.current = f.limit.load(); f.available = 0;
    auto owner = std::make_shared<test_owner>(f.reader()); auto a = owner->make_client(), b = owner->make_client();
    a->configure(config(true)); b->configure(config(true));
    std::promise<void> entered, release; auto proceed = release.get_future();
    actions reclaim;
    reclaim.collect_deferred = [&] { entered.set_value(); proceed.wait(); };
    reclaim.reclaim_tier = [](uint64_t) { return false; };
    std::atomic<bool> rejected{false}, closed{false};
    a->set_reclaimer(reclaim);
    std::thread worker([&] {
        try { auto t = a->take(1); }
        catch (const std::runtime_error &) { rejected = true; }
    });
    entered.get_future().wait();
    (void)b->measure(0); // a blocked user callback cannot own the arbiter mutex
    std::thread closer([&] { a->close(); closed = true; });
    for (;;) {
        try { (void)a->state(); std::this_thread::yield(); }
        catch (const std::runtime_error &) { break; }
    }
    const bool waited = !closed.load(); release.set_value(); worker.join(); closer.join();
    require(waited && rejected && closed, "close did not wait for an active reclaimer");
    // Clearing the reclaimer drops it at once, so later takes run none, and waits for a reclaim still running on
    // another thread.
    std::promise<void> slow_entered, resume; auto go = resume.get_future(); std::atomic<int> calls{0};
    actions slow;
    slow.reclaim_cache = [&](uint64_t) { if (!calls++) { slow_entered.set_value(); go.wait(); } return false; };
    b->set_reclaimer(slow);
    std::atomic<bool> cleared{false};
    std::thread taker([&] { auto t = b->take(1); });
    slow_entered.get_future().wait();
    std::thread clearer([&] { b->clear_reclaimer(); cleared = true; });
    for (int before = calls;; before = calls) {
        (void)b->take(1);
        if (calls == before) { break; } // this take found no reclaimer
        std::this_thread::yield();
    }
    const bool waited_for_run = !cleared.load(); resume.set_value(); taker.join(); clearer.join();
    require(waited_for_run && cleared, "clearing did not wait for a running reclaim");
}
static void result_errors_and_tracked_allocations() {
    files f;
    auto owner = std::make_shared<process_arbiter>(f.reader());
    auto a = checked(owner->make_client());
    require(a->take(1).code == ssd::status::invalid, "unconfigured core admission did not return invalid");
    checked(a->configure(config(true)));
    f.fail_usage = true;
    require(a->take(1).code == ssd::status::io, "missing bounded cgroup usage fell through to system-only admission");
    f.fail_usage = false;
    checked(a->begin_interval(0));
    auto tracked = a->take(GiB, 0, requester::outside, false, allocation_class::tracked);
    require(bool(tracked), "tracked ticket setup failed");
    f.available = 6*GiB; f.current = 2*GiB; tracked.value.commit();
    uint64_t unknown = 0;
    require(checked(a->end_interval(true, GiB, unknown)) && unknown == GiB && checked(a->state()).granted == 0,
        "cache held plus ticket grants hid real driver growth");
    auto crossing = a->take(256*MiB); require(bool(crossing), "crossing ticket setup failed");
    checked(a->begin_interval(0)); f.available -= 256*MiB; crossing.value.commit();
    require(!checked(a->end_interval(true, 0, unknown)) && unknown == 0 && checked(a->state()).overlap_skips == 1,
        "unmaterialized grant at interval start became driver growth");
    checked(a->begin_interval(0));
    // An external free can hide unknown driver growth if only the cumulative
    // grant counter is consulted; notify must invalidate the actor, not just peers.
    f.available += 128*MiB; checked(a->note_known_memory_change());
    require(!checked(a->end_interval(true, 0, unknown)) && unknown == 0 && checked(a->state()).overlap_skips == 2,
        "external free did not invalidate actor learning");
    checked(a->begin_interval(0)); f.available -= 2*GiB;
    require(checked(a->end_interval(true, 0, unknown)) && unknown == 2*GiB,
        "historical external grants masked later driver growth");
    auto zero = a->take(0); require(bool(zero) && bool(zero.value), "zero-byte successful result lost ticket status");
    zero.value.cancel();
    f.current = f.limit.load(); f.available = 0;
    actions reclaim; reclaim.reclaim_cache = [](uint64_t) -> bool { throw std::runtime_error("fixture callback"); };
    checked(a->set_reclaimer(reclaim));
    auto refused = a->take(1);
    require(!refused && refused.code == ssd::status::io && checked(a->state()).pending_bytes == 0,
        "callback exception escaped core or leaked pending");
    require(refused.error == "host callback: fixture callback", "a callback's error lost what it threw");
    auto broken = std::make_shared<process_arbiter>([](const std::string &, std::string &) -> bool { throw std::runtime_error("fixture reader"); });
    const auto unread = broken->make_client();
    require(unread.code == ssd::status::io && unread.error == "host owner initialization: fixture reader",
        "reader exception escaped owner construction or lost what it threw");
    auto exhausted = std::make_shared<process_arbiter>([](const std::string &, std::string &) -> bool { throw std::bad_alloc(); });
    require(exhausted->make_client().code == ssd::status::resource, "allocation failure escaped owner construction");
}
// On unified memory the cgroup does not charge the GPU tier's device memory. The envelope counts what the caches
// publish as held, so the RAM cache and the tier share it, and the system term (MemAvailable) does not count it
// twice. Off unified memory device memory is not this DRAM.
static void unified_device_held() {
    files f; f.limit = 8*GiB; f.current = GiB; f.available = 64*GiB;
    auto owner = std::make_shared<test_owner>(f.reader()); auto a = owner->make_client(), b = owner->make_client();
    a->configure(config(true)); b->configure(config(true));
    const auto before = b->measure(0);
    require(before.cgroup_surplus == 8*GiB - GiB - 512*MiB, "fixture envelope changed");
    checked(a->raw->set_device_held(5*GiB));
    require(a->state().device_held == 5*GiB && b->state().device_held == 0 && b->state().all_device_held == 5*GiB,
        "device memory was not published for the process");
    require(b->measure(0).cgroup_surplus == before.cgroup_surplus - 5*GiB &&
        b->measure(0).system_surplus == before.system_surplus,
        "the envelope did not count the tier's device memory, or MemAvailable counted it twice");
    // Neither the RAM cache nor the tier grows past the envelope the tier holds.
    require(!b->take(2*GiB, 0, requester::ram_cache, false, allocation_class::tracked),
        "the RAM cache grew into the envelope the GPU tier holds");
    require(!a->take(0, 2*GiB, requester::gpu_tier, false, allocation_class::tracked), "the GPU tier grew past the envelope");
    auto grows = a->take(0, GiB, requester::gpu_tier, false, allocation_class::tracked);
    require(bool(grows), "the GPU tier could not grow within the envelope");
    grows.cancel();
    // An allocation outside the caches that only the envelope refuses has the GPU tier give room first.
    std::vector<int> order;
    actions reclaim;
    reclaim.reclaim_tier = [&](uint64_t) { order.push_back(2); checked(a->raw->set_device_held(GiB)); return true; };
    reclaim.reclaim_cache = [&](uint64_t) { order.push_back(3); return false; };
    a->set_reclaimer(reclaim);
    auto outside = a->take(3*GiB);
    require(bool(outside) && order == std::vector<int>({2}), "the GPU tier did not give room in the envelope");
    outside.cancel(); a->clear_reclaimer();
    a->close();
    require(b->state().all_device_held == 0 && b->measure(0).cgroup_surplus == before.cgroup_surplus,
        "a closed context's device memory stayed in the envelope");
    files g; auto discrete = std::make_shared<test_owner>(g.reader()); auto d = discrete->make_client();
    d->configure(config(false));
    const auto plain = d->measure(0);
    checked(d->raw->set_device_held(5*GiB));
    require(d->state().device_held == 0 && d->measure(0).cgroup_surplus == plain.cgroup_surplus,
        "a discrete GPU's memory was counted in the host envelope");
}
// A machine of one cgroup scope under an unbounded root, whose files the scenarios below change between steps.
struct machine {
    std::string scope = "/sys/fs/cgroup/user.slice/run-1.scope";
    uint64_t limit = 8*GiB, current = 0, available = 0, file_pages = 0, refaults = 0;
    std::string zoneinfo; // empty: no /proc/zoneinfo, so no watermark boost
    process_arbiter::reader reader() {
        return [this](const std::string & path, std::string & text) {
            std::map<std::string, std::string> f;
            f["/proc/self/cgroup"] = "0::/user.slice/run-1.scope\n";
            f["/sys/fs/cgroup/user.slice/memory.max"] = "max\n";
            f[scope + "/memory.max"] = limit ? std::to_string(limit) + "\n" : "max\n";
            f[scope + "/memory.current"] = std::to_string(current) + "\n";
            f[scope + "/memory.stat"] = "anon 4096\nfile " + std::to_string(file_pages) + "\nactive_file 0\ninactive_file " +
                std::to_string(file_pages) + "\nworkingset_refault_file " + std::to_string(refaults) + "\n";
            f["/proc/meminfo"] = "MemTotal:       127000000 kB\nMemFree:          100000 kB\nMemAvailable:   " +
                std::to_string(available / 1024) + " kB\nActive(file):    1048576 kB\nInactive(file):  3145728 kB\n"
                "KReclaimable:     512000 kB\nHugePages_Total:       0\nHugepagesize:       2048 kB\n";
            if (!zoneinfo.empty()) { f["/proc/zoneinfo"] = zoneinfo; f["/proc/sys/vm/watermark_boost_factor"] = "15000\n"; }
            const auto it = f.find(path);
            if (it == f.end()) { return false; }
            text = it->second;
            return true;
        };
    }
};
// The zones of a DGX Spark (GB10, 4 KiB pages): their low watermarks reach the largest boost at DMA 613 and
// Normal 111957 pages, and MemAvailable keeps back (112570 + 20799) pages more at the full boost than with none.
static std::string spark_zones(uint64_t dma_boost, uint64_t normal_boost) {
    auto zone = [&](const char * name, uint64_t boost, uint64_t low, uint64_t high, uint64_t managed) {
        return std::string("Node 0, zone ") + name + "\n  pages free     1000\n        boost    " + std::to_string(boost) +
            "\n        min      61\n        low      " + std::to_string(low + boost) + "\n        high     " +
            std::to_string(high + boost) + "\n        spanned  524288\n        managed  " + std::to_string(managed) +
            "\n  pagesets\n    cpu: 0\n              count: 12\n              high:  999999\n              batch: 63\n";
    };
    return zone("DMA", dma_boost, 235, 409, 174798) + zone("DMA32", 0, 0, 0, 0) +
        zone("Normal", normal_boost, 42934, 74638, 31726918) + zone("Movable", 0, 32, 32, 0);
}
// What a take outside the caches asks of the context's reclaimer, in order: on unified memory the deferred uploads,
// then the GPU tier for the allocation and the open prefill's pages, then the RAM cache; on a discrete GPU the RAM
// cache alone. A take at the full watermark boost makes room for the boost too.
static void reclaim_order() {
    machine m;
    std::vector<std::string> calls;
    uint64_t tier_bytes = 0;
    bool tier_live = false;
    actions reclaim;
    reclaim.collect_deferred = [&] { calls.push_back("collect"); };
    reclaim.reclaim_tier = [&](uint64_t bytes) {
        calls.push_back("tier"); tier_bytes = bytes;
        if (tier_live) { return false; }
        m.available += 512*MiB; // one block of the tier, outside the cgroup
        return true;
    };
    reclaim.reclaim_cache = [&](uint64_t) {
        calls.push_back("cache");
        if (!m.current) { return false; }
        m.current -= 512*MiB; m.available += 512*MiB; // RAM cache slots are cgroup memory
        return true;
    };
    auto client = [&](bool unified) {
        auto owner = std::make_shared<test_owner>(m.reader());
        auto c = owner->make_client();
        auto conf = config(unified); conf.reclaimable_file = false;
        c->configure(conf); c->set_reclaimer(reclaim);
        return std::make_pair(owner, std::move(c));
    };
    // A discrete GPU: the cgroup term binds; the tier is never asked, the RAM cache gives back.
    m.available = 100*GiB; m.current = 7*GiB + 768*MiB;
    {
        auto [owner, d] = client(false);
        const auto r = d->measure(256*MiB);
        require(r.cgroup == room_pressure && r.cgroup_shortage == 512*MiB && r.system && !r.system_shortage,
            "discrete: the cgroup term did not bind");
        auto t = d->take(256*MiB);
        require(bool(t) && calls == std::vector<std::string>{ "cache" }, "discrete: a take did not reclaim only the RAM cache");
        require(checked(d->raw->dram_bytes(1*MiB, 2*MiB)) == 1*MiB, "discrete: device memory counted against the DRAM");
    }
    // Unified memory: the GPU tier is invisible to the cgroup, so the system term binds; the take collects the
    // deferred uploads, then asks the tier for room for the allocation, and the RAM cache keeps its slots.
    calls.clear();
    m.limit = 100*GiB; m.current = 1*GiB; m.file_pages = 300*MiB; m.refaults = 10; m.available = 2*GiB + 256*MiB;
    const uint64_t checkpoint = 1930*MiB;
    {
        auto [owner, u] = client(true);
        require(u->update_file_reserve() && u->state().floor == 812*MiB, "unified: the file reserve was not measured");
        require(checked(u->raw->dram_bytes(1*MiB, 2*MiB)) == 3*MiB, "unified: device memory not counted against the DRAM");
        const auto r = u->measure(checkpoint);
        require(r.cgroup == room_ok && r.system_shortage == 812*MiB + checkpoint - m.available, "unified: the system term did not bind");
        auto t = u->take(checkpoint);
        require(bool(t) && calls == std::vector<std::string>({ "collect", "tier" }) && tier_bytes == checkpoint &&
            u->state().granted == checkpoint, "unified: a take did not drain deferred uploads and then reclaim the tier");
        t.cancel();
        // A live layer: the tier refuses without blocking, the RAM cache cannot cover the rest, the answer is no.
        calls.clear(); tier_live = true; m.available = 2*GiB; m.current = 0;
        require(!u->take(checkpoint) && calls == std::vector<std::string>({ "collect", "tier", "cache" }),
            "unified: a refused tier reclaim was not reported");
        tier_live = false;
    }
    // An open prefill's pages: the tier keeps them free beside the allocation.
    calls.clear();
    m.limit = 100*GiB; m.current = 1*GiB; m.file_pages = 0; m.available = 1*GiB;
    {
        auto [owner, u] = client(true);
        u->set_prefill_pages(1*GiB);
        auto t = u->take(256*MiB);
        require(calls.size() >= 2 && calls[1] == "tier" && tier_bytes == 256*MiB + 1*GiB,
            "prefill pages: the tier was not asked to keep them free beside the allocation");
    }
    // At the full watermark boost: room for what the boost can still withdraw from MemAvailable.
    calls.clear();
    m.zoneinfo = spark_zones(0, 0); m.limit = 100*GiB; m.current = 1*GiB; m.file_pages = 300*MiB; m.refaults = 10;
    {
        auto [owner, u] = client(true);
        require(u->update_file_reserve(), "boost: the file reserve was not measured");
        const uint64_t page = uint64_t(sysconf(_SC_PAGESIZE)), full = (112570 + 20799) * page;
        m.available = 812*MiB + checkpoint + 64*MiB; // fits MemAvailable as it is, not at the full boost
        const auto plain = u->measure(checkpoint), boosted = u->measure(checkpoint, true);
        require(plain.fits() && plain.boost_headroom == 0 && !boosted.fits() && boosted.boost_headroom == full &&
            boosted.system_shortage == full - 64*MiB, "boost: the two readings of MemAvailable");
        auto t = u->take(checkpoint);
        require(bool(t) && calls == std::vector<std::string>({ "collect", "tier" }) && m.available >= 812*MiB + checkpoint + full,
            "boost: a take did not make room for the boost");
    }
}
// The margin learns untracked growth within a decode token on unified memory and never shrinks. An interval begins
// after a boundary resized the caches and ends at the next boundary before it resizes anything.
static void margin_learning() {
    machine m; m.limit = 100*GiB; m.available = 50*GiB;
    auto owner = std::make_shared<test_owner>(m.reader());
    auto u = owner->make_client(); u->configure(config(true));
    uint64_t untracked = 0;
    require(!u->end_interval(true, 10*GiB, untracked) && untracked == 0, "an interval ended that never began");
    u->begin_interval(10*GiB);
    m.available -= 1*GiB; // the tier grew a block: tracked
    require(!u->end_interval(true, 11*GiB, untracked) && untracked == 0, "tracked growth counted as untracked");
    u->begin_interval(11*GiB);
    m.available -= 300*MiB; // untracked, below the margin
    require(!u->end_interval(true, 11*GiB, untracked) && untracked == 300*MiB && u->state().margin == 512*MiB,
        "growth below the margin raised it");
    u->begin_interval(11*GiB);
    m.available -= 900*MiB; // untracked, beyond the margin: the margin becomes it
    require(u->end_interval(true, 11*GiB, untracked) && untracked == 900*MiB && u->state().margin == 900*MiB &&
        u->state().floor == 900*MiB && u->state().reserve == 512*MiB, "untracked growth did not raise the margin alone");
    // The boundary gives 3 GiB of tier blocks back and MemAvailable lags the free: outside every interval.
    u->begin_interval(8*GiB);
    m.available += 3*GiB;
    require(!u->end_interval(true, 8*GiB, untracked) && untracked == 0, "a boundary shrink counted as untracked");
    u->begin_interval(8*GiB);
    m.available -= 600*MiB;
    require(!u->end_interval(true, 8*GiB, untracked) && u->state().margin == 900*MiB, "the margin shrank or moved");
    u->begin_interval(8*GiB);
    auto checkpoint = u->take(2*GiB); m.available -= 2*GiB; checkpoint.commit(); // an allocation the arbiter admitted
    require(!u->end_interval(true, 8*GiB, untracked) && untracked == 0, "a grant counted as untracked");
    u->begin_interval(8*GiB);
    m.available -= 5*GiB; // a prefill came between: nothing is learned
    require(!u->end_interval(false, 8*GiB, untracked) && u->state().margin == 900*MiB, "a non-consecutive boundary learned");
    // The boost the kernel raises within a decode token is not untracked growth: MemAvailable falls by what the
    // headroom shrinks.
    const uint64_t page = uint64_t(sysconf(_SC_PAGESIZE));
    m.zoneinfo = spark_zones(0, 0); m.available = 50*GiB;
    u->begin_interval(10*GiB);
    m.zoneinfo = spark_zones(0, 111957); m.available -= (111957 + 20799) * page; // the reclaimable half's share as well
    require(!u->end_interval(true, 10*GiB, untracked) && untracked == 0, "boost: a raised boost counted as growth");
    u->begin_interval(10*GiB);
    m.zoneinfo = spark_zones(0, 0); m.available += (111957 + 20799) * page; m.available -= 1*GiB; // kswapd dropped it
    require(u->end_interval(true, 10*GiB, untracked) && untracked == 1*GiB && u->state().margin == 1*GiB,
        "boost: growth behind a dropped boost");
    // A discrete GPU learns nothing.
    machine dm; dm.limit = 100*GiB; dm.available = 50*GiB;
    auto discrete_owner = std::make_shared<test_owner>(dm.reader());
    auto d = discrete_owner->make_client(); d->configure(config(false));
    d->begin_interval(0);
    dm.available -= 5*GiB;
    require(!d->end_interval(true, 0, untracked) && d->state().margin == 512*MiB, "a discrete GPU learned a margin");
}
int main() {
    try {
        serial_equivalence(); holds_and_pending(); races_and_reclaim(); process_reserves_and_rollback();
        interval_isolation_and_close(); close_waits_for_reclaimer();
        result_errors_and_tracked_allocations(); unified_device_held(); reclaim_order(); margin_learning();
        std::cout << "host arbiter: serial equations, holds, atomic tickets, reclaim order, margin learning, interval, close and unified device gates passed\n";
    } catch (const std::exception & error) { std::cerr << error.what() << '\n'; return 1; }
}
