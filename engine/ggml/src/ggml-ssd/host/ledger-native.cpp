// SPDX-License-Identifier: Apache-2.0
// The native budget sources of ledger-native.h, one implementation per OS.
#include "ledger-native.h"

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <psapi.h>
#include <winternl.h>

#include <algorithm>
#include <string>
#include <vector>
#elif defined(__APPLE__)
#include <mach/mach.h>
#include <mach/vm_page_size.h>
#include <sys/sysctl.h>
#else
#error "ledger-native is the macOS and Windows budget source; Linux uses ledger.h"
#endif

namespace ssd { namespace host { namespace native {

#if defined(_WIN32)

namespace {

template<class F> bool resolve(HMODULE module, const char * name, F & function) {
    // Through void (*)(), the generic function pointer type, as GetProcAddress results are meant to be cast.
    function = reinterpret_cast<F>(reinterpret_cast<void (*)()>(GetProcAddress(module, name)));
    return function != nullptr;
}

// D3DKMT as gdi32 exports it and the WDK's d3dkmthk.h declares it; DXGI's QueryVideoMemoryInfo calls it.
using kmt_handle = UINT;
struct kmt_adapter_info { kmt_handle adapter; LUID luid; ULONG sources; BOOL precise_present_regions; };
struct kmt_enum_adapters2 { ULONG count; kmt_adapter_info * adapters; };
struct kmt_close_adapter { kmt_handle adapter; };
enum kmt_segment_group : int { kmt_local = 0, kmt_non_local = 1 };
struct kmt_video_memory_info {
    HANDLE process; // null: this process
    kmt_handle adapter;
    kmt_segment_group group;
    UINT64 budget, current_usage, current_reservation, available_for_reservation;
    UINT physical_adapter;
};

// The adapters whose local segments are not host memory: those that also have a non-local segment group, the
// system memory a discrete GPU maps. A unified (integrated) adapter has the local group only, and its segments
// are host memory. Opened once and kept open; an adapter added later is not seen.
struct discrete_adapters {
    LONG (APIENTRY * query)(kmt_video_memory_info *) = nullptr;
    std::vector<kmt_handle> handles;
    discrete_adapters() {
        LONG (APIENTRY * enumerate)(kmt_enum_adapters2 *) = nullptr;
        LONG (APIENTRY * close)(const kmt_close_adapter *) = nullptr;
        const HMODULE gdi = LoadLibraryW(L"gdi32.dll");
        if (!gdi || !resolve(gdi, "D3DKMTEnumAdapters2", enumerate) || !resolve(gdi, "D3DKMTCloseAdapter", close) ||
                !resolve(gdi, "D3DKMTQueryVideoMemoryInfo", query)) {
            query = nullptr;
            return;
        }
        kmt_enum_adapters2 list{};
        if (enumerate(&list) != 0) { return; }
        std::vector<kmt_adapter_info> found(list.count);
        list.adapters = found.data();
        if (enumerate(&list) != 0) { return; }
        found.resize(list.count);
        for (const kmt_adapter_info & adapter : found) {
            kmt_video_memory_info non_local{};
            non_local.adapter = adapter.adapter;
            non_local.group = kmt_non_local;
            if (query(&non_local) == 0 && non_local.budget > 0) {
                handles.push_back(adapter.adapter);
                continue;
            }
            const kmt_close_adapter closing{adapter.adapter};
            (void) close(&closing);
        }
    }
};
const discrete_adapters & discrete() {
    static const discrete_adapters adapters;
    return adapters;
}

// What the paging files may still grow by, each up to its maximum size and its volume's free space: a
// system-managed paging file starts small, and the commit limit grows with it on demand.
uint64_t paging_growth() {
    using query_fn = LONG (NTAPI *)(ULONG, PVOID, ULONG, PULONG);
    static const query_fn query = [] {
        query_fn function = nullptr;
        const HMODULE ntdll = GetModuleHandleW(L"ntdll.dll");
        return ntdll && resolve(ntdll, "NtQuerySystemInformation", function) ? function : nullptr;
    }();
    if (!query) { return 0; }
    // SYSTEM_PAGEFILE_INFORMATION_EX (class 144): SYSTEM_PAGEFILE_INFORMATION, then the minimum and maximum
    // sizes. Sizes are in pages.
    struct entry { ULONG next, total, in_use, peak; UNICODE_STRING name; ULONG minimum, maximum; };
    alignas(entry) unsigned char buffer[4096];
    ULONG length = 0;
    if (query(144, buffer, ULONG(sizeof(buffer)), &length) != 0 || length < sizeof(entry)) { return 0; }
    SYSTEM_INFO system{};
    GetSystemInfo(&system);
    uint64_t growth = 0;
    for (size_t at = 0; at + sizeof(entry) <= length;) {
        const entry & e = *reinterpret_cast<const entry *>(buffer + at);
        uint64_t more = e.maximum > e.total ? uint64_t(e.maximum - e.total) * system.dwPageSize : 0;
        // The name is an NT path such as \??\C:\pagefile.sys; that volume's free space bounds the growth.
        const std::wstring name(e.name.Buffer ? e.name.Buffer : L"", e.name.Length / sizeof(wchar_t));
        if (more && name.size() > 6 && name.compare(0, 4, L"\\??\\") == 0 && name[5] == L':') {
            const std::wstring root = name.substr(4, 2) + L"\\";
            ULARGE_INTEGER space{};
            if (GetDiskFreeSpaceExW(root.c_str(), &space, nullptr, nullptr)) {
                more = std::min<uint64_t>(more, uint64_t(space.QuadPart));
            }
        }
        growth += more;
        if (!e.next) { break; }
        at += e.next;
    }
    return growth;
}

} // namespace

bool commit_use(uint64_t & bytes) {
    PROCESS_MEMORY_COUNTERS_EX counters{};
    counters.cb = sizeof(counters);
    if (!GetProcessMemoryInfo(GetCurrentProcess(), reinterpret_cast<PROCESS_MEMORY_COUNTERS *>(&counters),
                              sizeof(counters))) {
        return false;
    }
    bytes = uint64_t(counters.PrivateUsage);
    return true;
}

bool device_commit(uint64_t & bytes) {
    const discrete_adapters & adapters = discrete();
    bytes = 0;
    for (const kmt_handle adapter : adapters.handles) {
        kmt_video_memory_info local{};
        local.adapter = adapter;
        local.group = kmt_local;
        if (adapters.query(&local) != 0) { return false; }
        bytes += local.current_usage;
    }
    return true;
}

bool process_use(uint64_t & bytes) {
    uint64_t commit = 0, device = 0;
    if (!commit_use(commit) || !device_commit(device)) { return false; }
    bytes = commit > device ? commit - device : 0;
    return true;
}

bool available(uint64_t & bytes) {
    MEMORYSTATUSEX status{};
    status.dwLength = sizeof(status);
    if (!GlobalMemoryStatusEx(&status)) { return false; }
    // Commit is a hard limit, so room is the smaller of the physical memory available and the commit a new
    // allocation can still get: what is free under the commit limit now plus what the paging files may grow by.
    bytes = std::min<uint64_t>(status.ullAvailPhys, status.ullAvailPageFile + paging_growth());
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
    // The kernel holds both limits against commit, the device memory WDDM charges included.
    if (flags & JOB_OBJECT_LIMIT_PROCESS_MEMORY) {
        uint64_t used = 0;
        if (!commit_use(used)) { return false; }
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
