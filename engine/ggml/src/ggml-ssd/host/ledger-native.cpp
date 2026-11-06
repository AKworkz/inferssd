// SPDX-License-Identifier: Apache-2.0
// The native budget sources of ledger-native.h, one implementation per OS.
#include "ledger-native.h"

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <psapi.h>
#elif defined(__APPLE__)
#include <mach/mach.h>
#include <mach/vm_page_size.h>
#include <sys/sysctl.h>
#else
#error "ledger-native is the macOS and Windows budget source; Linux uses ledger.h"
#endif

namespace ssd { namespace host { namespace native {

#if defined(_WIN32)

bool process_use(uint64_t & bytes) {
    PROCESS_MEMORY_COUNTERS_EX counters{};
    counters.cb = sizeof(counters);
    if (!GetProcessMemoryInfo(GetCurrentProcess(), reinterpret_cast<PROCESS_MEMORY_COUNTERS *>(&counters),
                              sizeof(counters))) {
        return false;
    }
    bytes = uint64_t(counters.PrivateUsage);
    return true;
}

bool available(uint64_t & bytes) {
    MEMORYSTATUSEX status{};
    status.dwLength = sizeof(status);
    if (!GlobalMemoryStatusEx(&status)) { return false; }
    // Physical memory alone overstates room when the commit limit binds first.
    bytes = std::min<uint64_t>(status.ullAvailPhys, status.ullAvailPageFile);
    return true;
}

bool physical(uint64_t & bytes) {
    MEMORYSTATUSEX status{};
    status.dwLength = sizeof(status);
    if (!GlobalMemoryStatusEx(&status)) { return false; }
    bytes = uint64_t(status.ullTotalPhys);
    return true;
}

bool job_limits(envelopes & out) {
    BOOL in_job = FALSE;
    if (!IsProcessInJob(GetCurrentProcess(), nullptr, &in_job)) { return false; }
    if (!in_job) { return true; }
    // A null handle names the process's immediate job.
    JOBOBJECT_EXTENDED_LIMIT_INFORMATION limits{};
    if (!QueryInformationJobObject(nullptr, JobObjectExtendedLimitInformation, &limits, sizeof(limits), nullptr)) {
        return false;
    }
    const DWORD flags = limits.BasicLimitInformation.LimitFlags;
    if (flags & JOB_OBJECT_LIMIT_PROCESS_MEMORY) {
        uint64_t used = 0;
        if (!process_use(used)) { return false; }
        out.push({"job process", uint64_t(limits.ProcessMemoryLimit), used});
    }
    if (flags & JOB_OBJECT_LIMIT_JOB_MEMORY) {
        // JobMemory is the commit of every process in the job now; the extended limits carry only its peak.
        JOBOBJECT_LIMIT_VIOLATION_INFORMATION job{};
        if (!QueryInformationJobObject(nullptr, JobObjectLimitViolationInformation, &job, sizeof(job), nullptr)) {
            return false;
        }
        out.push({"job", uint64_t(limits.JobMemoryLimit), uint64_t(job.JobMemory)});
    }
    return true;
}

#else // __APPLE__

bool process_use(uint64_t & bytes) {
    task_vm_info_data_t info{};
    mach_msg_type_number_t count = TASK_VM_INFO_COUNT;
    if (task_info(mach_task_self(), TASK_VM_INFO, reinterpret_cast<task_info_t>(&info), &count) != KERN_SUCCESS ||
            count < TASK_VM_INFO_REV1_COUNT) {
        return false;
    }
    bytes = info.phys_footprint;
    return true;
}

// One of the kernel's page counters, read live. host_statistics64 holds the same counters, but the kernel rate
// limits it for unentitled processes: past a few calls a second it repeats its last answer, so room that a
// give-back has just made would not show.
struct page_counter {
    int mib[CTL_MAXNAME] = {};
    size_t length = CTL_MAXNAME;
    bool named = false;
    explicit page_counter(const char * name) : named(sysctlnametomib(name, mib, &length) == 0) {}
    bool read(uint64_t & pages) const {
        uint64_t value = 0; // the counters are 32-bit, read into the low half on these little-endian targets
        size_t size = sizeof(value);
        if (!named || sysctl(const_cast<int *>(mib), u_int(length), &value, &size, nullptr, 0) != 0 ||
                (size != sizeof(uint32_t) && size != sizeof(uint64_t))) {
            return false;
        }
        pages = size == sizeof(uint32_t) ? uint64_t(uint32_t(value)) : value;
        return true;
    }
};

bool available(uint64_t & bytes) {
    // What a new allocation takes before the kernel compresses anyone's memory, by the kernel's own live
    // thresholds: the free pages above its free target (below it the pageout scan runs); the file-backed pages
    // (the speculative ones among them) above its file cache minimum, which the scan takes before it compresses
    // anything and keeps; and the purgeable and reusable pages (MADV_FREE_REUSABLE: the RAM cache's discarded
    // slots and the allocator's free blocks), which it frees without compressing.
    static const page_counter free_pages("vm.page_free_count"), file_pages("vm.page_pageable_external_count"),
            purgeable_pages("vm.page_purgeable_count"), reusable_pages("vm.page_reusable_count"),
            free_target("vm.vm_page_free_target"), file_minimum("vm.vm_page_filecache_min");
    uint64_t free = 0, file = 0, purgeable = 0, reusable = 0, target = 0, minimum = 0;
    if (!free_pages.read(free) || !file_pages.read(file) || !purgeable_pages.read(purgeable) ||
            !reusable_pages.read(reusable) || !free_target.read(target) || !file_minimum.read(minimum)) {
        return false;
    }
    bytes = ((free > target ? free - target : 0) + (file > minimum ? file - minimum : 0) + purgeable + reusable) *
            uint64_t(vm_kernel_page_size);
    return true;
}

bool physical(uint64_t & bytes) {
    uint64_t value = 0;
    size_t size = sizeof(value);
    if (sysctlbyname("hw.memsize", &value, &size, nullptr, 0) != 0 || size != sizeof(value) || !value) { return false; }
    bytes = value;
    return true;
}

bool job_limits(envelopes &) { return true; }

#endif

}}} // namespace ssd::host::native
