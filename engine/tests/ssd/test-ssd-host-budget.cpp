// SPDX-License-Identifier: Apache-2.0
// The native host budget sources (macOS, Windows) on the running machine: the ledger reads a finite envelope and
// a system term of sane magnitude; the arbiter admits a small take, holds the RAM cache to the envelope, and lets an
// allocation the run needs go up to the limit itself.
#include "ggml-ssd/host/arbiter.h"
#include "ggml-ssd/host/ledger-native.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#endif

using namespace ssd::host;
constexpr uint64_t MiB = 1ull << 20, GiB = 1ull << 30;

static void require(bool ok, const char * what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        std::exit(1);
    }
}

static unsigned long long mib(uint64_t bytes) { return (unsigned long long) (bytes / MiB); }

static void print(const char * label, const room & r) {
    std::printf("%s: envelope %s surplus %llu MiB shortage %llu MiB; system %s available %llu MiB floor %llu MiB\n",
                label, r.cgroup == room_ok ? "ok" : r.cgroup == room_pressure ? "pressure" : "unreadable",
                mib(r.cgroup_surplus), mib(r.cgroup_shortage), r.system ? "read" : "unreadable", mib(r.available),
                mib(r.floor));
}

static void print(const char * label, const envelopes & e) {
    for (size_t i = 0; i < e.count; ++i) {
        std::printf("%s: %s limit %llu MiB used %llu MiB\n", label, e.item[i].source, mib(e.item[i].limit),
                    mib(e.item[i].used));
    }
}

static uint64_t process_use() {
    uint64_t used = 0;
    require(native::process_use(used) && used > 0, "process use readable");
    return used;
}

static std::unique_ptr<context_client> client(const std::shared_ptr<process_arbiter> & owner,
                                              const client_config & config) {
    auto made = owner->make_client();
    require(bool(made), "make_client");
    require(bool(made.value->configure(config)), "configure");
    return std::move(made.value);
}

// The arbiter on the real sources: a 64 MiB take fits. The RAM cache's growth `past` (less than the reserve) beyond
// the envelope's surplus is refused by the envelope. An allocation the run needs of the same size goes ahead, as the
// reserve is the caches', and one `past` beyond the limit itself is refused.
static void admission(const char * label, context_client & c, uint64_t past) {
    const uint64_t reserve = client_config{}.reserve_bytes;
    require(past < reserve, "admission: past must be less than the reserve");
    const auto measured = c.measure(0);
    require(bool(measured), "measure");
    print(label, measured.value);
    require(measured.value.cgroup == room_ok && measured.value.limits == room_ok, "envelope readable with room");
    {
        auto small = c.take(64 * MiB, 0, requester::outside, true);
        require(bool(small) && bool(small.value), "small take admitted");
        small.value.cancel();
    }
    const uint64_t surplus = measured.value.cgroup_surplus;
    auto cache = c.take(surplus + past, 0, requester::ram_cache, true, allocation_class::tracked);
    require(!cache && cache.code == ssd::status::resource && !cache.value, "the RAM cache grew past the envelope");
    require(cache.value.observation().cgroup_shortage > 0, "the envelope refused it");
    {
        auto needed = c.take(surplus + past);
        require(bool(needed) && bool(needed.value), "an allocation the run needs was held to the caches' reserve");
        needed.value.cancel();
    }
    auto large = c.take(surplus + reserve + past);
    require(!large && large.code == ssd::status::resource && !large.value && large.error == "host memory limit refused",
            "an allocation past the limit itself was admitted");
    require(large.value.observation().limits_shortage > 0, "the limit refused it");
}

// Use is the process's own memory: touching 128 MiB of heap raises it by about that much.
static void use_tracks_allocation() {
    const uint64_t before = process_use();
    const size_t bytes = size_t(128 * MiB);
    // Volatile, so the compiler cannot drop an allocation nobody reads.
    char * volatile block = static_cast<char *>(std::malloc(bytes));
    require(block != nullptr, "allocation");
    std::memset(block, 1, bytes);
    const uint64_t after = process_use();
    std::printf("use: %llu MiB before, %llu MiB with 128 MiB touched\n", mib(before), mib(after));
    require(after >= before + 120 * MiB && after <= before + 256 * MiB, "use follows the process's own memory");
    std::free(block);
}

// host.budget_mib 0 and no job limit: the limit is the machine's physical memory against the process's own use.
static void machine() {
    uint64_t total = 0;
    require(native::physical(total) && total > 0, "physical memory readable");
    native_ledger ledger;
    require(ledger.machine == total, "the ledger read the machine's memory");
    envelopes found;
    require(ledger.limits(found) && found.count >= 1, "envelope readable");
    print("machine", found);
    const envelope & e = found.item[found.count - 1];
    if (std::strcmp(e.source, "machine") == 0) {
        require(e.limit == total && e.used < total, "the machine envelope is its physical memory");
    } else {
        std::printf("machine: this process runs under a job memory limit (%s)\n", e.source);
    }
    const room r = ledger.measure(0);
    print("machine", r);
    require(r.cgroup == room_ok && r.cgroup_surplus > 0 && r.cgroup_surplus < total, "envelope room sane");
    require(r.limits == room_ok && !r.limits_shortage, "the limits hold nothing more");
    require(r.system && r.available <= total, "system term within physical memory");
    require(r.floor == 512 * MiB, "floor is the margin");
    const room full = ledger.measure(0, true);
    require(full.system && full.boost_headroom == 0, "no watermark boost");
    const room over = ledger.measure(r.cgroup_surplus + GiB);
    require(over.cgroup == room_pressure && over.cgroup_shortage > GiB / 2, "envelope binds past its surplus");
    require(over.limits == room_pressure && over.limits_shortage > 256 * MiB && over.limits_shortage <= GiB,
            "the limits bind past the limit itself");
    const room within = ledger.measure(r.cgroup_surplus + 256 * MiB);
    require(within.cgroup == room_pressure && within.limits == room_ok, "the reserve is the envelope's, not the limits'");
    require(!ledger.update_file_reserve() && ledger.floor() == 512 * MiB, "file reserve stays inert");
    ledger.prefill_pages = 256 * MiB;
    const room held = ledger.measure(0);
    const uint64_t need = held.floor + 256 * MiB; // whatever the machine has available now
    require(held.system && held.system_surplus == (held.available > need ? held.available - need : 0) &&
            held.system_shortage == (need > held.available ? need - held.available : 0),
            "the system term holds prefill pages");
    require(held.cgroup_surplus + 16 * MiB >= r.cgroup_surplus, "the envelope does not hold prefill pages");
    require(held.limits == room_ok, "the limits do not hold prefill pages");

    auto owner = std::make_shared<process_arbiter>();
    auto c = client(owner, client_config{});
    admission("machine arbiter", *c, 64 * MiB);
}

// host.budget_mib against the process's own memory; the tightest client budget binds.
static void explicit_budget() {
    const uint64_t budget = process_use() + 512 * MiB + 256 * MiB;
    native_ledger ledger;
    ledger.budget = budget;
    envelopes found;
    require(ledger.limits(found) && found.count >= 1, "budget envelope readable");
    print("budget", found);
    const envelope & set = found.item[found.count - 1];
    require(std::strcmp(set.source, "budget") == 0 && set.limit == budget, "budget envelope in force");
    const room r = ledger.measure(0);
    print("budget", r);
    require(r.cgroup == room_ok && r.cgroup_surplus <= 256 * MiB && r.cgroup_surplus > 128 * MiB,
            "budget room is the budget less use and reserve");

    auto owner = std::make_shared<process_arbiter>();
    client_config config;
    config.budget_bytes = budget;
    auto tight = client(owner, config);
    admission("budget arbiter", *tight, 64 * MiB);
    config.budget_bytes = budget + 4 * GiB;
    auto loose = client(owner, config);
    // The tighter budget binds every client: the loose one's RAM cache within the envelope, and what its run needs
    // within the limit.
    auto refused = loose->take(512 * MiB, 0, requester::ram_cache, true, allocation_class::tracked);
    require(!refused && refused.code == ssd::status::resource, "the tighter budget did not bind the RAM cache");
    auto past = loose->take(GiB);
    require(!past && past.code == ssd::status::resource, "the tighter budget did not limit what the run needs");
    require(!client(std::make_shared<process_arbiter>(), config)->configure(client_config{}), "budget change refused");
}

#if defined(_WIN32)
// A job memory limit on commit (the job's and the process's), set the way a deployment would.
static void job_limit() {
    // Commit made before joining the job must count as the job's use too.
    const size_t bytes = size_t(128 * MiB);
    char * volatile block = static_cast<char *>(std::malloc(bytes));
    require(block != nullptr, "allocation");
    std::memset(block, 1, bytes);
    const uint64_t used = process_use();
    HANDLE job = CreateJobObjectW(nullptr, nullptr);
    require(job != nullptr, "CreateJobObject");
    JOBOBJECT_EXTENDED_LIMIT_INFORMATION limits{};
    limits.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_JOB_MEMORY | JOB_OBJECT_LIMIT_PROCESS_MEMORY;
    limits.JobMemoryLimit = SIZE_T(used + 512 * MiB + 512 * MiB);
    limits.ProcessMemoryLimit = SIZE_T(used + 512 * MiB + 256 * MiB);
    require(SetInformationJobObject(job, JobObjectExtendedLimitInformation, &limits, sizeof(limits)) != 0,
            "SetInformationJobObject");
    // The process may already be in a job without limits (OpenSSH); this one nests under it.
    if (!AssignProcessToJobObject(job, GetCurrentProcess())) {
        std::fprintf(stderr, "AssignProcessToJobObject: error %lu\n", GetLastError());
        require(false, "AssignProcessToJobObject");
    }
    native_ledger ledger;
    envelopes found;
    require(ledger.limits(found) && found.count == 2, "both job limits read");
    print("job", found);
    require(std::strcmp(found.item[0].source, "job process") == 0 &&
                found.item[0].limit == uint64_t(limits.ProcessMemoryLimit), "process memory limit");
    require(std::strcmp(found.item[1].source, "job") == 0 && found.item[1].limit == uint64_t(limits.JobMemoryLimit),
            "job memory limit");
    const uint64_t job_used = found.item[1].used, own = found.item[0].used;
    require(own >= 120 * MiB && job_used + 16 * MiB >= own && job_used <= own + 64 * MiB,
            "job commit is the process's commit, from before it joined");
    const room r = ledger.measure(0);
    print("job", r);
    require(r.cgroup == room_ok && r.cgroup_surplus <= 256 * MiB && r.cgroup_surplus > 128 * MiB,
            "the tighter (process) limit binds");
    auto owner = std::make_shared<process_arbiter>();
    auto c = client(owner, client_config{});
    admission("job arbiter", *c, 64 * MiB);
    std::free(block);
    envelopes after;
    require(ledger.limits(after) && after.count == 2 && after.item[1].used + 100 * MiB <= job_used,
            "job commit follows a free");
    print("job after free", after);
    CloseHandle(job);
}
#endif

int main() {
    use_tracks_allocation();
    machine();
    explicit_budget();
#if defined(_WIN32)
    job_limit(); // last: the process stays in the job
#endif
    std::printf("ok\n");
    return 0;
}
