// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <atomic>
#include <cstdint>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86)
#include <immintrin.h>
#endif

namespace ssd { namespace platform {

// Logical CPU numbers, the bit positions of affinity masks and of ggml
// cpumasks: kernel CPU ids on Linux, group-0 logical processor indices on
// Windows. macOS cannot pin, so there the numbers only carry counts.
using cpu_list = std::vector<int>;

// The CPUs this code may run on, ascending. Linux: the calling thread's
// sched_getaffinity mask (inherited from the process). Windows: the process
// affinity mask; machines with more than one processor group are refused.
// macOS: 0 .. hw.logicalcpu - 1.
bool process_cpus(cpu_list & cpus, std::string * why = nullptr);

// The CPUs the calling thread runs on now, ascending: its affinity mask
// (Linux, as process_cpus reads it); its group-0 affinity narrowed by the CPU
// Sets it selected where they meet it (Windows); process_cpus (macOS).
bool current_thread_cpus(cpu_list & cpus, std::string * why = nullptr);

// The process CPUs split into the fastest cluster and the rest. A homogeneous
// or unreadable machine has every CPU in `fast`. Only a candidate for thread
// placement; callers still measure.
struct cpu_topology {
    cpu_list fast;
    cpu_list slow;
};
bool topology(cpu_topology & out, std::string * why = nullptr);

namespace detail {
// The Linux cluster rule on (cpu, capacity or max frequency) pairs, all
// positive: split at the largest relative drop between sorted values; a drop
// under 10% means one cluster. Exposed so tests can feed synthetic machines.
bool split_fastest(const std::vector<std::pair<int, double>> & values, cpu_topology & out);
} // namespace detail

// False on macOS, where pinning is a successful no-op.
bool affinity_supported();

// Restrict a thread to exactly `cpus` and read the result back; false if the
// OS refused or the read-back differs. An empty list is a no-op. Windows uses
// group affinity and matching CPU Sets so parked cores cannot ignore the pin.
bool pin_current_thread(const cpu_list & cpus);
bool pin_thread(std::thread::native_handle_type thread, const cpu_list & cpus);

// Scheduler hints for the calling thread. io opts out of Windows EcoQoS
// throttling and takes the macOS user-initiated QoS; background returns to the
// system-managed default (macOS: utility QoS). Linux: no-op. Compute threads
// are ggml's, placed by its threadpool.
enum class thread_role { io, background };
bool set_thread_role(thread_role role);

// CPU time consumed by the calling thread; 0 if unreadable. Windows advances
// in scheduler ticks, so short intervals can read as zero.
uint64_t thread_cpu_time_ns();

// The CPU the calling thread runs on, or -1 (macOS, or not in group 0).
int current_cpu();

// A 32-bit word that threads park on until another thread changes it and
// wakes them, with no lock (Linux futex, macOS os_sync_wait_on_address,
// Windows WaitOnAddress). wait_on returns once the word differs from
// `expected`, after a wake_all, or spuriously, so callers recheck their
// condition; wake_all wakes every thread parked on the word.
void wait_on(const std::atomic<uint32_t> & word, uint32_t expected) noexcept;
void wake_all(const std::atomic<uint32_t> & word) noexcept;

// Spin-wait hint. _mm_pause works with every x86 compiler, including MSVC,
// which defines only _M_X64 and has no GCC builtins.
inline void cpu_relax() noexcept {
#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86)
    _mm_pause();
#elif defined(__aarch64__) || defined(_M_ARM64)
    __asm__ __volatile__("yield" ::: "memory");
#endif
}

} } // namespace ssd::platform
