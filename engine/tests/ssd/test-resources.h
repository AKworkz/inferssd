// SPDX-License-Identifier: Apache-2.0
#pragma once
// Test prerequisites only: never add a runtime fallback or hide numerical/I/O defects.
// A missing resource throws `unavailable` (exit 77, a skip); anything else is a failure.
// Every OS reaches files, CPUs and memory through the platform layer, as the engine does.
#include "../../ggml/src/ggml-ssd/platform/file.h"
#include "../../ggml/src/ggml-ssd/platform/memory.h"
#include "../../ggml/src/ggml-ssd/platform/threads.h"
#include <atomic>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <new>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <system_error>

#if defined(__linux__)
#include "../../ggml/src/ggml-ssd/host/ledger.h"
#include <linux/io_uring.h>
#include <sys/syscall.h>
#else
#include "../../ggml/src/ggml-ssd/host/ledger-native.h"
#endif
#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <psapi.h>
#include <io.h>
#else
#include <sys/resource.h>
#include <unistd.h>
#endif

namespace test_resources {
struct unavailable : std::runtime_error { using std::runtime_error::runtime_error; };

inline void need(bool available, const std::string & reason) {
    if (!available) { throw unavailable(reason); }
}

inline void check_counts(int compute, int io) {
    if (compute < 1 || io < 0 || compute > 255 || io > 255) {
        throw std::runtime_error("invalid test compute/IO thread count");
    }
}

// The CPUs this process may run on (Linux: the calling thread's affinity).
inline ssd::platform::cpu_list process_cpus() {
    ssd::platform::cpu_list cpus;
    std::string why;
    if (!ssd::platform::process_cpus(cpus, &why)) { throw std::runtime_error("test process CPUs: " + why); }
    return cpus;
}

// Compute participants and I/O threads may share CPUs. Placement changes timing only, never
// results, so a small machine still runs the correctness cases.
inline void cpu_participants(int compute, int io) { check_counts(compute, io); }

// The read ring of this OS. io_uring can be compiled out (ENOSYS) or forbidden (EPERM, EACCES);
// the Windows IoRing (with its overlapped backend) and the worker-pool ring need no probe.
inline void io_ring() {
#if defined(__linux__)
    io_uring_params parameters = {};
    const int fd = int(syscall(__NR_io_uring_setup, 2, &parameters));
    if (fd < 0) {
        const int error = errno;
        const std::string message = std::string("io_uring_setup (no fallback): ") + std::strerror(error);
        if (error == ENOSYS || error == EPERM || error == EACCES) { throw unavailable(message); }
        throw std::runtime_error(message);
    }
    close(fd);
#endif
}

inline bool direct_unsupported(int error) { return error == EINVAL || error == EOPNOTSUPP || error == ENOSYS; }

// Probe the fixture's actual file, not another mount, with one aligned direct read of its first
// granule. ENOENT/EACCES/EIO/short reads are errors. The platform reports both a filesystem that
// refuses direct I/O and a path that is not a regular file as EINVAL; only the first is a skip.
inline int open_direct(const std::string & path) {
    std::string why;
    const int fd = ssd::platform::open_direct(path.c_str(), &why);
    if (fd < 0) {
        const int error = errno;
        std::error_code ec;
        if (direct_unsupported(error) && std::filesystem::is_regular_file(path, ec)) {
            throw unavailable("direct open unsupported on fixture filesystem: " + why);
        }
        throw std::runtime_error("fixture open: " + (why.empty() ? path + ": " + std::strerror(error) : why));
    }
    const size_t granule = ssd::platform::direct_alignment(fd) > 4096 ? ssd::platform::direct_alignment(fd) : 4096;
    void * buffer = ssd::platform::aligned_alloc_bytes(granule, granule);
    if (!buffer) { ssd::platform::close_fd(fd); throw std::bad_alloc(); }
    const int64_t count = ssd::platform::pread_some(fd, buffer, granule, 0);
    ssd::platform::aligned_free(buffer);
    if (count != int64_t(granule)) {
        ssd::platform::close_fd(fd);
        if (count < 0 && direct_unsupported(int(-count))) {
            throw unavailable("aligned direct read unsupported on fixture filesystem: " + path);
        }
        throw std::runtime_error("fixture direct read: " + path + ": " +
                (count < 0 ? std::strerror(int(-count)) : "short header page"));
    }
    return fd;
}

#if defined(__linux__)
inline void finite_cgroup(kimi_host_ledger & ledger) {
    ssd::host::room room;
    ledger.cgroup_term(0, room);
    need(room.cgroup != ssd::host::room_unreadable,
            "cache fixture needs readable finite cgroup v2 memory.max/current ancestors");
    // Do not infer a payload budget here: individual fixtures change reserves
    // and intentionally exercise pressure. Their size/ownership checks remain authoritative.
}
#endif
// The RAM cache admits only under a finite, readable memory envelope. Linux: a cgroup v2
// memory.max on this process or an ancestor, which the environment may not provide. macOS and
// Windows: the native ledger always has one (a job limit, host.budget_mib or the one derived at
// creation), so an unreadable one is an OS query failure, not a missing resource.
inline void finite_envelope() {
#if defined(__linux__)
    kimi_host_ledger ledger;
    ledger.locate();
    finite_cgroup(ledger);
#else
    ssd::host::native_ledger ledger;
    ssd::host::room room;
    ledger.envelope_term(0, room);
    if (room.cgroup == ssd::host::room_unreadable) {
        throw std::runtime_error("the native memory envelope is unreadable; the RAM cache would admit nothing");
    }
#endif
}

// These messages originate from the named production syscalls. Check their exact suffix;
// EINVAL/EFAULT from registration, unrelated ENOMEM, and other exceptions remain failures.
inline bool operation_error(const std::string & message, const char * operation, int error) {
    const std::string suffix = std::string(operation) + ": " + std::strerror(error);
    return message == suffix || (message.size() > suffix.size() &&
            message.compare(message.size() - suffix.size(), suffix.size(), suffix) == 0 &&
            message[message.size() - suffix.size() - 1] == ' ');
}
// Only io_uring reports these; the IoRing and worker-pool rings take their registration
// limits from the ring backend, which fails rather than skips.
inline bool syscall_unavailable(const std::string & message) {
    for (int error : { ENOSYS, EPERM, EACCES }) {
        if (operation_error(message, "io_uring_setup (no fallback)", error)) { return true; }
    }
    // The actual registration is authoritative: it includes the fixture's real
    // geometry and other live rings. A tiny probe cannot establish its pin budget.
    for (int error : { EPERM, ENOMEM }) {
        if (operation_error(message, "IORING_REGISTER_BUFFERS (check RLIMIT_MEMLOCK; no fallback)", error)) { return true; }
    }
    return false;
}
inline int report(const char * prefix, const std::exception & error) {
    const bool skip = dynamic_cast<const unavailable *>(&error) || syscall_unavailable(error.what());
    std::fprintf(stderr, "%s_%s: %s\n", prefix, skip ? "SKIP" : "FAILED", error.what());
#if defined(__linux__)
    if (skip && std::strstr(error.what(), "IORING_REGISTER_BUFFERS")) {
        rlimit limit = {};
        if (!getrlimit(RLIMIT_MEMLOCK, &limit)) {
            std::fprintf(stderr, "actual fixture buffer registration refused; RLIMIT_MEMLOCK=%llu bytes\n",
                    static_cast<unsigned long long>(limit.rlim_cur));
        }
    }
#endif
    return skip ? 77 : 1;
}

// Fixture files. A new empty file in the OS temporary directory (TMPDIR first on POSIX).
inline std::string temp_file(const std::string & stem) {
    static std::atomic<uint64_t> serial{0};
    std::random_device entropy;
    const auto directory = std::filesystem::temp_directory_path();
    for (int attempt = 0; attempt < 64; ++attempt) {
        std::ostringstream name;
        name << stem << '-' << std::hex << ((uint64_t(entropy()) << 32) ^ entropy()) << '-' << serial++;
        const auto path = directory / name.str();
        std::error_code ec;
        if (std::filesystem::exists(path, ec) || ec) { continue; }
        std::ofstream file(path, std::ios::binary);
        if (file) { return path.string(); }
    }
    throw std::runtime_error("cannot create a fixture file in " + directory.string());
}
inline void remove_file(const std::string & path) noexcept {
    std::error_code ec;
    std::filesystem::remove(path, ec);
}

// Descriptor plumbing for fixtures: the POSIX calls, or the Windows CRT's.
inline int fd_dup(int fd) {
#if defined(_WIN32)
    return _dup(fd);
#else
    return dup(fd);
#endif
}
inline int fd_dup2(int from, int to) {
#if defined(_WIN32)
    return _dup2(from, to) == 0 ? to : -1;
#else
    return dup2(from, to);
#endif
}
inline void fd_close(int fd) {
#if defined(_WIN32)
    _close(fd);
#else
    close(fd);
#endif
}
inline int fd_of(std::FILE * stream) {
#if defined(_WIN32)
    return _fileno(stream);
#else
    return fileno(stream);
#endif
}

// What the OS charges this process for its own memory: Linux Anonymous (smaps_rollup), macOS
// phys_footprint, Windows private commit (the native ledger's use). The RAM cache's commit and
// discard move it by exactly the slots they touch.
inline uint64_t anonymous_bytes() {
#if defined(__linux__)
    std::ifstream file("/proc/self/smaps_rollup");
    if (!file) { throw std::runtime_error("cannot open smaps_rollup for physical release check"); }
    std::string line;
    while (std::getline(file, line)) {
        std::istringstream in(line); std::string label, unit; uint64_t kib = 0;
        if (in >> label >> kib >> unit && label == "Anonymous:") {
            if (unit != "kB") { throw std::runtime_error("unexpected smaps_rollup units"); }
            return kib * 1024;
        }
    }
    throw std::runtime_error("smaps_rollup has no Anonymous counter");
#else
    uint64_t bytes = 0;
    if (!ssd::host::native::process_use(bytes)) { throw std::runtime_error("cannot read this process's memory use"); }
    return bytes;
#endif
}
// What else moves anonymous_bytes() while a fixture releases its cache: allocator and counter noise, and on
// Windows the heap the call itself commits, which a free does not decommit.
#if defined(_WIN32)
constexpr uint64_t anonymous_noise_bytes = 2ull << 20;
#else
constexpr uint64_t anonymous_noise_bytes = 256ull << 10;
#endif
// Thread CPU clocks advance in scheduler ticks on Windows (platform/threads.h): a run shorter than a tick can read
// zero there. Elsewhere a thread that did work reads a positive time.
#if defined(_WIN32)
constexpr bool thread_clock_ticks = true;
#else
constexpr bool thread_clock_ticks = false;
#endif
// Resident memory of this process: Linux statm (file pages included); elsewhere the native use.
inline uint64_t resident_bytes() {
#if defined(__linux__)
    std::ifstream statm("/proc/self/statm");
    uint64_t size = 0, pages = 0;
    if (!(statm >> size >> pages)) { throw std::runtime_error("cannot read statm"); }
    return pages * uint64_t(ssd::platform::page_size());
#else
    return anonymous_bytes();
#endif
}

// Process CPU times and counters for measurement output; never part of a pass/fail decision.
struct process_usage {
    double user_us = 0, system_us = 0;
    long voluntary_switches = 0, involuntary_switches = 0, minor_faults = 0, major_faults = 0, max_rss_kib = 0;
};
inline process_usage usage() {
    process_usage out;
#if defined(_WIN32)
    FILETIME created, exited, kernel, user;
    if (!GetProcessTimes(GetCurrentProcess(), &created, &exited, &kernel, &user)) {
        throw std::runtime_error("GetProcessTimes failed");
    }
    auto us = [](const FILETIME & t) { return double((uint64_t(t.dwHighDateTime) << 32) | t.dwLowDateTime) / 10.0; };
    out.user_us = us(user);
    out.system_us = us(kernel);
    PROCESS_MEMORY_COUNTERS counters = {};
    counters.cb = sizeof(counters);
    if (GetProcessMemoryInfo(GetCurrentProcess(), &counters, sizeof(counters))) {
        out.minor_faults = long(counters.PageFaultCount);
        out.max_rss_kib = long(counters.PeakWorkingSetSize / 1024);
    }
#else
    rusage value = {};
    if (getrusage(RUSAGE_SELF, &value) != 0) { throw std::runtime_error("getrusage failed"); }
    auto us = [](const timeval & t) { return t.tv_sec * 1e6 + t.tv_usec; };
    out.user_us = us(value.ru_utime);
    out.system_us = us(value.ru_stime);
    out.voluntary_switches = value.ru_nvcsw;
    out.involuntary_switches = value.ru_nivcsw;
    out.minor_faults = value.ru_minflt;
    out.major_faults = value.ru_majflt;
#if defined(__APPLE__)
    out.max_rss_kib = long(value.ru_maxrss / 1024); // bytes on macOS
#else
    out.max_rss_kib = value.ru_maxrss;
#endif
#endif
    return out;
}
} // namespace test_resources
