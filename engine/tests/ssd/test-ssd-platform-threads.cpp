// SPDX-License-Identifier: Apache-2.0
// Platform thread layer: process CPUs, pinning with read-back, scoped restore,
// thread roles, per-thread CPU time and the CPUs' busy time.
#if defined(_WIN32) && !defined(_WIN32_WINNT)
#define _WIN32_WINNT 0x0A00 // GetThreadSelectedCpuSets
#endif
#include "ggml-ssd/platform/threads.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#elif defined(__APPLE__)
#include <pthread.h>
#elif defined(__linux__)
#include <pthread.h>
#include <sched.h>
#endif

using namespace ssd::platform;

static void require(bool value, const std::string & why) {
    if (!value) { throw std::runtime_error(why); }
}

static std::string show(const cpu_list & cpus) {
    std::string text;
    for (int cpu : cpus) { text += (text.empty() ? "" : ",") + std::to_string(cpu); }
    return "[" + text + "]";
}

static bool contains(const cpu_list & cpus, int cpu) { return std::find(cpus.begin(), cpus.end(), cpu) != cpus.end(); }

// The thread's placement as the OS reports it, to check scoped restores.
static std::vector<unsigned long long> placement() {
    std::vector<unsigned long long> out;
#if defined(__linux__)
    cpu_set_t set;
    require(pthread_getaffinity_np(pthread_self(), sizeof(set), &set) == 0, "pthread_getaffinity_np");
    for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
        if (CPU_ISSET(cpu, &set)) { out.push_back(cpu); }
    }
#elif defined(_WIN32)
    ULONG ids[256] = {}, count = 0;
    require(GetThreadSelectedCpuSets(GetCurrentThread(), ids, 256, &count) != 0, "GetThreadSelectedCpuSets");
    out.push_back(count);
    for (ULONG i = 0; i < count; ++i) { out.push_back(ids[i]); }
    GROUP_AFFINITY group = {};
    require(GetThreadGroupAffinity(GetCurrentThread(), &group) != 0, "GetThreadGroupAffinity");
    out.push_back(group.Group);
    out.push_back(group.Mask);
#endif
    return out;
}

// The calling thread's CPUs as the platform layer reads them.
static cpu_list thread_cpus_now() {
    cpu_list cpus;
    std::string why;
    require(current_thread_cpus(cpus, &why), "current_thread_cpus: " + why);
    return cpus;
}

// The scheduler moves a pinned thread at its next dispatch.
static bool runs_within(const cpu_list & cpus) {
    if (!affinity_supported()) { return true; }
    for (int i = 0; i < 500; ++i) {
        if (contains(cpus, current_cpu())) { return true; }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    return false;
}

static cpu_list test_process_cpus() {
    cpu_list cpus;
    std::string why;
    require(process_cpus(cpus, &why), "process_cpus: " + why);
    require(!cpus.empty() && std::is_sorted(cpus.begin(), cpus.end()) &&
            std::adjacent_find(cpus.begin(), cpus.end()) == cpus.end(), "process CPUs are ascending and distinct");
    std::printf("process_cpus=%s affinity=%d\n", show(cpus).c_str(), int(affinity_supported()));
    return cpus;
}

static void test_pin(const cpu_list & cpus) {
    const auto before = placement();
    require(thread_cpus_now() == cpus, "an unpinned thread runs on the process CPUs");
    require(pin_current_thread({}), "an empty pin is a no-op");
    require(placement() == before, "an empty pin leaves the thread as it was");
    if (affinity_supported()) {
        require(!pin_current_thread({ -1 }) && !pin_current_thread({ 100000 }), "invalid CPUs are refused");
        require(placement() == before, "a refused pin leaves the thread as it was");
    }
    for (int cpu : { cpus.front(), cpus.back() }) {
        require(pin_current_thread({ cpu }), "pin the calling thread to CPU " + std::to_string(cpu));
#if defined(_WIN32)
        GROUP_AFFINITY hard = {};
        require(GetThreadGroupAffinity(GetCurrentThread(), &hard) != 0 && hard.Group == 0 &&
                hard.Mask == (KAFFINITY(1) << cpu), "a pin sets hard affinity even for a parked CPU");
#endif
        require(runs_within({ cpu }), "the calling thread runs on CPU " + std::to_string(cpu));
        require(thread_cpus_now() == (affinity_supported() ? cpu_list{ cpu } : cpus),
                "the calling thread reads its pin back as CPU " + std::to_string(cpu));
    }
#if defined(_WIN32)
    if (cpus.size() > 1) {
        // A hard affinity outranks CPU Sets: a pin outside it must fail or really move the thread.
        GROUP_AFFINITY saved = {}, hard = {};
        require(GetThreadGroupAffinity(GetCurrentThread(), &saved) != 0, "GetThreadGroupAffinity");
        hard.Mask = KAFFINITY(1) << cpus.front();
        require(SetThreadGroupAffinity(GetCurrentThread(), &hard, nullptr) != 0, "SetThreadGroupAffinity");
        require(thread_cpus_now() == cpu_list{ cpus.front() }, "selected CPU Sets outside the hard affinity are ignored");
        const bool pinned = pin_current_thread({ cpus.back() });
        const bool moved = pinned && runs_within({ cpus.back() });
        require(SetThreadGroupAffinity(GetCurrentThread(), &saved, nullptr) != 0, "restore the group affinity");
        require(!pinned || moved, "a pin reports success only where the thread runs");
    }
#endif
    require(pin_current_thread(cpus), "pin the calling thread to every process CPU");
    require(thread_cpus_now() == cpus, "the calling thread reads every process CPU back");

    const int target = cpus.front();
    std::atomic<int> go{0}, seen{-1};
    std::thread worker([&] {
        while (!go.load()) { std::this_thread::yield(); }
        seen = runs_within({ target }) ? 1 : 0;
    });
    const bool pinned = pin_thread(worker.native_handle(), { target });
    go = 1;
    worker.join();
    require(pinned && seen.load() == 1, "pin another thread and see it run there");
}

static void test_roles() {
    for (thread_role role : { thread_role::io, thread_role::background }) {
        bool ok = false;
        std::thread([&] {
            ok = set_thread_role(role);
#if defined(__APPLE__)
            const qos_class_t want = role == thread_role::io ? QOS_CLASS_USER_INITIATED : QOS_CLASS_UTILITY;
            ok = ok && qos_class_self() == want;
#endif
        }).join();
        require(ok, "set_thread_role " + std::to_string(int(role)));
    }
}

static void test_clock() {
    const uint64_t first = thread_cpu_time_ns();
    uint64_t last = first;
    bool monotonic = true;
    volatile uint64_t work = 0;
    const auto start = std::chrono::steady_clock::now();
    while (std::chrono::steady_clock::now() - start < std::chrono::milliseconds(150)) {
        for (int i = 0; i < 20000; ++i) { work = work + uint64_t(i); }
        cpu_relax();
        const uint64_t now = thread_cpu_time_ns();
        monotonic = monotonic && now >= last;
        last = now;
    }
    std::printf("thread_cpu_time advanced %.3f ms over 150 ms of spinning\n", double(last - first) * 1e-6);
    require(monotonic && last > first, "thread CPU time is monotonic and advances");
    const int cpu = current_cpu();
    require(affinity_supported() ? cpu >= 0 : cpu == -1, "current_cpu");
}

// The CPUs' busy time covers this thread's spin and no more than the CPUs' wall time, within a tick on each CPU.
static void test_busy(const cpu_list & cpus) {
    uint64_t before = 0, after = 0, tick = 0;
    std::string why;
    if (!cpus_busy_ns(cpus, before, tick, &why)) {
#if defined(_WIN32)
        std::printf("cpus_busy_ns: %s\n", why.c_str());
        return;
#else
        require(false, "cpus_busy_ns: " + why);
#endif
    }
    const uint64_t cpu = thread_cpu_time_ns();
    const auto start = std::chrono::steady_clock::now();
    volatile uint64_t work = 0;
    while (std::chrono::steady_clock::now() - start < std::chrono::milliseconds(300)) { work = work + 1; }
    const uint64_t spun = thread_cpu_time_ns() - cpu;
    const double wall = std::chrono::duration<double, std::nano>(std::chrono::steady_clock::now() - start).count();
    require(cpus_busy_ns(cpus, after, tick, &why), "cpus_busy_ns again: " + why);
    const double busy = double(after) - double(before), slack = double(tick) * double(cpus.size());
    std::printf("cpus_busy advanced %.1f ms over %.1f ms; this thread spun %.1f ms; tick %.1f ms\n", busy * 1e-6,
                wall * 1e-6, double(spun) * 1e-6, double(tick) * 1e-6);
    require(after >= before && tick > 0, "busy time is monotonic and counted in ticks");
    require(busy + slack >= double(spun) && busy <= wall * double(cpus.size()) + slack,
            "busy time covers this thread's spin and no more than the CPUs' wall time");
}

int main() {
    try {
        const cpu_list cpus = test_process_cpus();
        test_busy(cpus);
        test_pin(cpus);
        test_roles();
        test_clock();
    } catch (const std::exception & e) {
        std::fprintf(stderr, "FAIL: %s\n", e.what());
        return 1;
    }
    std::printf("ssd platform threads: OK\n");
    return 0;
}
