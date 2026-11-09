// SPDX-License-Identifier: Apache-2.0
#if defined(_WIN32) && !defined(_WIN32_WINNT)
#define _WIN32_WINNT 0x0A00 // CPU Sets need Windows 10; set before any header pulls in the SDK
#endif
#include "threads.h"

#include <algorithm>
#include <cerrno>
#include <cstddef>
#include <cstdio>
#include <cstring>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#elif defined(__APPLE__)
#include <mach/mach.h>
#include <os/os_sync_wait_on_address.h>
#include <pthread.h>
#include <sched.h>
#include <sys/sysctl.h>
#include <time.h>
#include <unistd.h>
#elif defined(__linux__)
#include <climits>
#include <fstream>
#include <linux/futex.h>
#include <pthread.h>
#include <sched.h>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>
#else
#error "InferSSD platform layer: unsupported OS"
#endif

namespace ssd { namespace platform {

static_assert(sizeof(std::atomic<uint32_t>) == sizeof(uint32_t) && std::atomic<uint32_t>::is_always_lock_free,
              "a parking word is a plain 32-bit word");

static void set_why(std::string * why, std::string text) {
    if (why) { *why = std::move(text); }
}

bool pin_current_thread(const cpu_list & cpus) {
#if defined(_WIN32)
    return pin_thread(GetCurrentThread(), cpus);
#else
    return pin_thread(pthread_self(), cpus);
#endif
}

#if defined(__linux__)

bool process_cpus(cpu_list & cpus, std::string * why) {
    cpus.clear();
    cpu_set_t allowed;
    if (sched_getaffinity(0, sizeof(allowed), &allowed) != 0) {
        const int error = errno;
        set_why(why, std::string("sched_getaffinity: ") + std::strerror(error));
        return false;
    }
    for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
        if (CPU_ISSET(cpu, &allowed)) { cpus.push_back(cpu); }
    }
    return true;
}

// sched_getaffinity(0) reads the calling thread's mask.
bool current_thread_cpus(cpu_list & cpus, std::string * why) { return process_cpus(cpus, why); }

bool affinity_supported() { return true; }

bool pin_thread(std::thread::native_handle_type thread, const cpu_list & cpus) {
    if (cpus.empty()) { return true; }
    cpu_set_t wanted, actual;
    CPU_ZERO(&wanted);
    for (int cpu : cpus) {
        if (cpu < 0 || cpu >= CPU_SETSIZE) { return false; }
        CPU_SET(cpu, &wanted);
    }
    return pthread_setaffinity_np(thread, sizeof(wanted), &wanted) == 0 &&
        pthread_getaffinity_np(thread, sizeof(actual), &actual) == 0 && CPU_EQUAL(&wanted, &actual);
}

// The kernel scheduler needs no hints; Linux behaviour stays as it is.
bool set_thread_role(thread_role) { return true; }

int current_cpu() { return sched_getcpu(); }

// /proc/stat per CPU: user nice system idle iowait irq softirq steal guest guest_nice, in USER_HZ ticks; guest time
// is already in user.
bool cpus_busy_ns(const cpu_list & cpus, uint64_t & busy_ns, uint64_t & tick_ns, std::string * why) {
    const long hz = sysconf(_SC_CLK_TCK);
    std::ifstream in("/proc/stat");
    if (hz <= 0 || !in) {
        set_why(why, "/proc/stat is not readable");
        return false;
    }
    tick_ns = uint64_t(1000000000 / hz);
    uint64_t ticks = 0;
    size_t found = 0;
    std::string line;
    while (std::getline(in, line)) {
        int cpu = -1;
        unsigned long long user, nice, system, idle, iowait, irq, softirq, steal = 0;
        if (std::sscanf(line.c_str(), "cpu%d %llu %llu %llu %llu %llu %llu %llu %llu", &cpu, &user, &nice, &system, &idle,
                        &iowait, &irq, &softirq, &steal) < 8) {
            continue;
        }
        if (!std::binary_search(cpus.begin(), cpus.end(), cpu)) { continue; }
        ticks += user + nice + system + irq + softirq + steal;
        ++found;
    }
    if (found != cpus.size()) {
        set_why(why, "/proc/stat lists " + std::to_string(found) + " of " + std::to_string(cpus.size()) + " CPUs");
        return false;
    }
    busy_ns = ticks * tick_ns;
    return true;
}

void wait_on(const std::atomic<uint32_t> & word, uint32_t expected) noexcept {
    syscall(SYS_futex, &word, FUTEX_WAIT_PRIVATE, expected, nullptr, nullptr, 0);
}

void wake_all(const std::atomic<uint32_t> & word) noexcept {
    syscall(SYS_futex, &word, FUTEX_WAKE_PRIVATE, INT_MAX, nullptr, nullptr, 0);
}

#elif defined(__APPLE__)

static int sysctl_int(const char * name) {
    int value = 0;
    size_t size = sizeof(value);
    return sysctlbyname(name, &value, &size, nullptr, 0) == 0 && size == sizeof(value) ? value : 0;
}

bool process_cpus(cpu_list & cpus, std::string * why) {
    cpus.clear();
    const int count = sysctl_int("hw.logicalcpu");
    if (count <= 0) {
        set_why(why, "sysctl hw.logicalcpu failed");
        return false;
    }
    for (int cpu = 0; cpu < count; ++cpu) { cpus.push_back(cpu); }
    return true;
}

// No thread affinity: every thread may run on every CPU.
bool current_thread_cpus(cpu_list & cpus, std::string * why) { return process_cpus(cpus, why); }

// macOS has no thread affinity; QoS roles steer cluster placement instead.
bool affinity_supported() { return false; }

bool pin_thread(std::thread::native_handle_type, const cpu_list &) { return true; }

bool set_thread_role(thread_role role) {
    const qos_class_t qos = role == thread_role::io ? QOS_CLASS_USER_INITIATED : QOS_CLASS_UTILITY;
    return pthread_set_qos_class_self_np(qos, 0) == 0;
}

int current_cpu() { return -1; }

// Each CPU's user, system and nice ticks (host_processor_info), counted at the clock tick rate.
bool cpus_busy_ns(const cpu_list & cpus, uint64_t & busy_ns, uint64_t & tick_ns, std::string * why) {
    natural_t count = 0;
    processor_info_array_t info = nullptr;
    mach_msg_type_number_t words = 0;
    if (host_processor_info(mach_host_self(), PROCESSOR_CPU_LOAD_INFO, &count, &info, &words) != KERN_SUCCESS) {
        set_why(why, "host_processor_info failed");
        return false;
    }
    const auto * load = reinterpret_cast<const processor_cpu_load_info_data_t *>(info);
    uint64_t ticks = 0;
    bool ok = true;
    for (int cpu : cpus) {
        if (cpu < 0 || natural_t(cpu) >= count) { ok = false; break; }
        ticks += uint64_t(load[cpu].cpu_ticks[CPU_STATE_USER]) + load[cpu].cpu_ticks[CPU_STATE_SYSTEM] +
                 load[cpu].cpu_ticks[CPU_STATE_NICE];
    }
    vm_deallocate(mach_task_self(), vm_address_t(info), vm_size_t(words) * sizeof(integer_t));
    const long hz = sysconf(_SC_CLK_TCK);
    if (!ok || hz <= 0) {
        set_why(why, ok ? "no clock tick rate" : "a CPU beyond the processors the host reports");
        return false;
    }
    tick_ns = uint64_t(1000000000 / hz);
    busy_ns = ticks * tick_ns;
    return true;
}

// Before macOS 14.4 there is no public wait on an address; the waiter yields and rechecks.
void wait_on(const std::atomic<uint32_t> & word, uint32_t expected) noexcept {
    if (__builtin_available(macOS 14.4, *)) {
        os_sync_wait_on_address(const_cast<std::atomic<uint32_t> *>(&word), expected, sizeof(uint32_t),
                                OS_SYNC_WAIT_ON_ADDRESS_NONE);
    } else {
        sched_yield();
    }
}

void wake_all(const std::atomic<uint32_t> & word) noexcept {
    if (__builtin_available(macOS 14.4, *)) {
        os_sync_wake_by_address_all(const_cast<std::atomic<uint32_t> *>(&word), sizeof(uint32_t),
                                    OS_SYNC_WAKE_BY_ADDRESS_NONE);
    }
}

#elif defined(_WIN32)

// CPU Sets of processor group 0, by logical processor index. Ids and
// efficiency classes are fixed while the system runs.
struct cpu_set_table {
    bool available = false;       // every active group-0 processor has a CPU Set
    std::vector<ULONG> id;
    std::vector<int> efficiency;  // -1: no CPU Set
    ULONG total = 0;              // CPU Sets in all groups
};

static cpu_set_table read_cpu_sets() {
    cpu_set_table table;
    const DWORD active = GetActiveProcessorCount(0);
    if (active == 0 || active > sizeof(DWORD_PTR) * 8) { return table; }
    table.id.assign(active, 0);
    table.efficiency.assign(active, -1);
    ULONG bytes = 0;
    GetSystemCpuSetInformation(nullptr, 0, &bytes, GetCurrentProcess(), 0);
    if (bytes == 0) { return table; }
    std::vector<uint64_t> storage((bytes + sizeof(uint64_t) - 1) / sizeof(uint64_t)); // entries hold DWORD64
    auto * base = reinterpret_cast<unsigned char *>(storage.data());
    if (!GetSystemCpuSetInformation(reinterpret_cast<PSYSTEM_CPU_SET_INFORMATION>(base), bytes, &bytes,
            GetCurrentProcess(), 0)) {
        return table;
    }
    for (ULONG at = 0; at + sizeof(DWORD) * 2 <= bytes;) {
        const auto * entry = reinterpret_cast<const SYSTEM_CPU_SET_INFORMATION *>(base + at);
        if (entry->Size == 0 || at + entry->Size > bytes) { break; }
        if (entry->Type == CpuSetInformation) {
            ++table.total;
            const DWORD index = entry->CpuSet.LogicalProcessorIndex;
            if (entry->CpuSet.Group == 0 && index < active) {
                table.id[index] = entry->CpuSet.Id;
                table.efficiency[index] = entry->CpuSet.EfficiencyClass;
            }
        }
        at += entry->Size;
    }
    table.available = std::all_of(table.efficiency.begin(), table.efficiency.end(), [](int e) { return e >= 0; });
    return table;
}

static const cpu_set_table & cpu_sets() {
    static const cpu_set_table table = read_cpu_sets();
    return table;
}

// Group-0 affinity masks are one DWORD_PTR (64 bits on x64, LLP64).
constexpr int mask_bits = int(sizeof(DWORD_PTR) * 8);

static bool single_group() { return GetActiveProcessorGroupCount() == 1; }

static bool process_mask(DWORD_PTR & mask) {
    DWORD_PTR system = 0;
    mask = 0;
    return single_group() && GetProcessAffinityMask(GetCurrentProcess(), &mask, &system) && mask != 0;
}

bool process_cpus(cpu_list & cpus, std::string * why) {
    cpus.clear();
    if (!single_group()) {
        set_why(why, "more than one processor group (over 64 logical CPUs) is not supported");
        return false;
    }
    DWORD_PTR mask = 0;
    if (!process_mask(mask)) {
        set_why(why, "GetProcessAffinityMask failed: error " + std::to_string(GetLastError()));
        return false;
    }
    for (int cpu = 0; cpu < mask_bits; ++cpu) {
        if ((mask >> cpu) & 1) { cpus.push_back(cpu); }
    }
    return true;
}

bool affinity_supported() { return true; }

static bool cpu_mask(const cpu_list & cpus, DWORD_PTR & mask) {
    mask = 0;
    for (int cpu : cpus) {
        if (cpu < 0 || cpu >= mask_bits) { return false; }
        mask |= DWORD_PTR(1) << cpu;
    }
    return true;
}

// A thread's selected CPU Sets as a group-0 mask (0: none selected).
static bool selected_mask(HANDLE thread, const cpu_set_table & sets, DWORD_PTR & mask) {
    std::vector<ULONG> ids(std::max<ULONG>(1, sets.total));
    ULONG count = 0;
    if (!GetThreadSelectedCpuSets(thread, ids.data(), ULONG(ids.size()), &count)) { return false; }
    mask = 0;
    for (ULONG i = 0; i < count; ++i) {
        const auto at = std::find(sets.id.begin(), sets.id.end(), ids[i]);
        if (at == sets.id.end()) { return false; }
        mask |= DWORD_PTR(1) << (at - sets.id.begin());
    }
    return true;
}

static bool select_mask(HANDLE thread, const cpu_set_table & sets, DWORD_PTR mask) {
    std::vector<ULONG> ids;
    for (size_t cpu = 0; cpu < sets.id.size(); ++cpu) {
        if ((mask >> cpu) & 1) { ids.push_back(sets.id[cpu]); }
    }
    return SetThreadSelectedCpuSets(thread, ids.empty() ? nullptr : ids.data(), ULONG(ids.size())) != 0;
}

// The hard affinity, narrowed by the selected CPU Sets when they meet it; the
// scheduler ignores sets outside it.
bool current_thread_cpus(cpu_list & cpus, std::string * why) {
    cpus.clear();
    if (!single_group()) {
        set_why(why, "more than one processor group (over 64 logical CPUs) is not supported");
        return false;
    }
    const HANDLE thread = GetCurrentThread();
    GROUP_AFFINITY hard = {};
    if (!GetThreadGroupAffinity(thread, &hard) || hard.Group != 0) {
        set_why(why, "GetThreadGroupAffinity failed: error " + std::to_string(GetLastError()));
        return false;
    }
    DWORD_PTR mask = DWORD_PTR(hard.Mask), selected = 0;
    const auto & sets = cpu_sets();
    if (sets.available) {
        if (!selected_mask(thread, sets, selected)) {
            set_why(why, "GetThreadSelectedCpuSets failed: error " + std::to_string(GetLastError()));
            return false;
        }
        if (mask & selected) { mask &= selected; }
    }
    for (int cpu = 0; cpu < mask_bits; ++cpu) {
        if ((mask >> cpu) & 1) { cpus.push_back(cpu); }
    }
    return true;
}

bool pin_thread(std::thread::native_handle_type handle, const cpu_list & cpus) {
    if (cpus.empty()) { return true; }
    const HANDLE thread = static_cast<HANDLE>(handle);
    DWORD_PTR wanted = 0, allowed = 0;
    // Like a Linux mask outside the allowed set, CPUs the process may not use are refused.
    if (!cpu_mask(cpus, wanted) || !process_mask(allowed) || (wanted & ~allowed) != 0) { return false; }
    const auto & sets = cpu_sets();
    GROUP_AFFINITY saved = {}, affinity = {}, actual = {};
    DWORD_PTR saved_selected = 0, selected = 0;
    if (!GetThreadGroupAffinity(thread, &saved) || saved.Group != 0 ||
            (sets.available && !selected_mask(thread, sets, saved_selected))) { return false; }
    affinity.Mask = KAFFINITY(wanted);
    affinity.Group = 0;
    // CPU Sets are a scheduler preference: a parked core may be ignored even
    // when the selected IDs read back correctly. Hard affinity makes the pin exact.
    if (!SetThreadGroupAffinity(thread, &affinity, nullptr)) { return false; }
    if (GetThreadGroupAffinity(thread, &actual) && actual.Group == 0 && actual.Mask == affinity.Mask &&
            (!sets.available || (select_mask(thread, sets, wanted) &&
                selected_mask(thread, sets, selected) && selected == wanted))) { return true; }
    SetThreadGroupAffinity(thread, &saved, nullptr);
    if (sets.available) { select_mask(thread, sets, saved_selected); }
    return false;
}

// Windows 11 parks cores and moves throttled (EcoQoS) threads onto efficiency
// cores; io threads opt out, as ggml's own workers do.
bool set_thread_role(thread_role role) {
    THREAD_POWER_THROTTLING_STATE state = {};
    state.Version = THREAD_POWER_THROTTLING_CURRENT_VERSION;
    state.ControlMask = role == thread_role::background ? 0 : THREAD_POWER_THROTTLING_EXECUTION_SPEED;
    state.StateMask = 0;
    return SetThreadInformation(GetCurrentThread(), ThreadPowerThrottling, &state, sizeof(state)) != 0;
}

uint64_t thread_cpu_time_ns() {
    FILETIME creation, exit, kernel, user;
    if (!GetThreadTimes(GetCurrentThread(), &creation, &exit, &kernel, &user)) { return 0; }
    const auto ticks = [](const FILETIME & t) { return (uint64_t(t.dwHighDateTime) << 32) | t.dwLowDateTime; };
    return (ticks(kernel) + ticks(user)) * 100; // 100 ns units
}

// GetSystemTimes sums every processor of the group, its kernel time including idle time, as the clock interrupt
// advances it.
bool cpus_busy_ns(const cpu_list & cpus, uint64_t & busy_ns, uint64_t & tick_ns, std::string * why) {
    const DWORD count = GetActiveProcessorCount(0);
    if (!single_group() || cpus.size() != count || (count && cpus.back() != int(count) - 1)) {
        set_why(why, "Windows reads the busy time of every processor of the group only");
        return false;
    }
    FILETIME idle, kernel, user;
    DWORD adjustment = 0, increment = 0;
    BOOL disabled = FALSE;
    if (!GetSystemTimes(&idle, &kernel, &user) || !GetSystemTimeAdjustment(&adjustment, &increment, &disabled) ||
        !increment) {
        set_why(why, "GetSystemTimes failed");
        return false;
    }
    const auto ticks = [](const FILETIME & t) { return (uint64_t(t.dwHighDateTime) << 32) | t.dwLowDateTime; };
    tick_ns = uint64_t(increment) * 100;
    busy_ns = (ticks(kernel) + ticks(user) - ticks(idle)) * 100; // 100 ns units
    return true;
}

int current_cpu() {
    PROCESSOR_NUMBER number = {};
    GetCurrentProcessorNumberEx(&number);
    return number.Group == 0 ? int(number.Number) : -1;
}

void wait_on(const std::atomic<uint32_t> & word, uint32_t expected) noexcept {
    WaitOnAddress(const_cast<std::atomic<uint32_t> *>(&word), &expected, sizeof(uint32_t), INFINITE);
}

void wake_all(const std::atomic<uint32_t> & word) noexcept {
    WakeByAddressAll(const_cast<std::atomic<uint32_t> *>(&word));
}

#endif

#if !defined(_WIN32)
uint64_t thread_cpu_time_ns() {
    timespec value;
    if (clock_gettime(CLOCK_THREAD_CPUTIME_ID, &value) != 0) { return 0; }
    return uint64_t(value.tv_sec) * 1000000000ull + uint64_t(value.tv_nsec);
}
#endif

} } // namespace ssd::platform
