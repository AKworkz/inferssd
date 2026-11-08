// SPDX-License-Identifier: Apache-2.0
#pragma once
// Host-memory ledger on macOS and Windows: the two terms and the limits of room.h from native budget sources, as
// ledger.h measures them on Linux, so the arbiter and everything it admits run the same logic on every OS.
//
// Room for `extra` more bytes is the smaller of two terms.
//  - envelope, reported in the arbiter's cgroup fields: every finite limit on this process's memory keeps the
//    reserve free after `extra`. The model's file mapping is use on neither OS.
//      Windows: the limits the process's job sets in its LimitFlags (a job alone is no limit: OpenSSH and
//      terminals run processes in jobs without one): JOB_OBJECT_LIMIT_JOB_MEMORY against the job's commit,
//      JOB_OBJECT_LIMIT_PROCESS_MEMORY against the process's PrivateUsage. The kernel holds both against commit,
//      which on a discrete GPU includes the device memory the process allocates: WDDM charges a device-local
//      allocation to commit without taking host memory. Only the immediate job is visible; a limit set on the
//      parent of a nested job is not.
//      Both OSes: host.budget_mib against the process's host use: on Windows PrivateUsage less the video memory
//      the process holds in the local segments of discrete adapters (D3DKMTQueryVideoMemoryInfo), on macOS
//      phys_footprint, which on unified memory rightly counts the GPU's allocations. It applies beside a job
//      limit, the tighter binding.
//    With host.budget_mib 0 (the default) and no job limit, the limit is the machine's physical memory, the
//    budget the framework passes by default, against the same use.
//    A failed OS query leaves the term unreadable, and the RAM cache then admits nothing. The same limits, holding
//    `extra` with no reserve beside it, are the limits of room.h.
//  - system: memory available to a new allocation, less the floor and the open prefill's pages.
//      Windows: min(ullAvailPhys, ullAvailPageFile plus what the paging files may still grow by), as commit
//      is a hard limit there and a system-managed paging file grows on demand.
//      macOS: what a new allocation takes before the kernel compresses or swaps anyone's anonymous memory, by its
//      own live thresholds (sysctl): the free pages above its free target, the file-backed pages above its file
//      cache minimum (vm_page_filecache_min, which it keeps), and the purgeable and reusable pages. The counterpart
//      of MemAvailable, which keeps the zones' watermarks back the same way.
// floor = margin + explicit reserves, as on Linux.
//
// How the Linux-specific parts carry over:
//  - file_reserve keeps a unified-memory CUDA tier off the server's page cache and grows with the kernel's
//    refault counter; neither OS has that tier or counter, so update_file_reserve() never grows it.
//  - The open prefill's file pages (weight rows read by index) are held in the system term, as on Linux, since it
//    counts cached file pages as available: on macOS those above the file cache minimum, on Windows the standby
//    list. A row still in the working set is not on that list, so Windows holds it twice until the first decode
//    token, which errs safe. The envelope does not hold them: neither commit nor phys_footprint is charged for a
//    read-only mapping's file pages, so they never take its room.
//  - There is no watermark boost: memory_available_at_full_boost() keeps nothing back.
#include "room.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string>

namespace ssd { namespace host {

// One finite limit on this process's memory and the use the OS counts against it.
struct envelope {
    const char * source; // "job", "job process", "budget" or "machine"
    uint64_t limit, used;
};
struct envelopes {
    envelope item[3];
    size_t count = 0;
    void push(const envelope & e) { if (count < 3) { item[count++] = e; } }
};

namespace native {
// This process's host memory: Windows PrivateUsage less device_commit, macOS phys_footprint.
bool process_use(uint64_t & bytes);
#if defined(_WIN32)
// The process's commit (PrivateUsage), which the kernel holds against a job's memory limits.
bool commit_use(uint64_t & bytes);
// The video memory the process holds in the local segments of discrete adapters, which WDDM charges to its
// commit. An adapter is discrete when it also has a non-local segment group; 0 without one.
bool device_commit(uint64_t & bytes);
#endif
// Memory available to a new allocation (the system term before the floor).
bool available(uint64_t & bytes);
// The machine's physical memory.
bool physical(uint64_t & bytes);
// Windows: appends the memory limits of the process's job; true with none outside a job. Elsewhere true, none.
bool job_limits(envelopes & out);
} // namespace native

// reclaimable_file and the file counters of ledger_terms stay inert here.
struct native_ledger : ledger_terms {
    std::string cgroup; // always empty: there is no cgroup to walk
    uint64_t budget = 0;  // host.budget_mib in bytes; 0 leaves the machine's memory as the limit
    uint64_t machine = 0; // the machine's physical memory, read at creation; 0 when unreadable then

    native_ledger() {
        uint64_t bytes = 0;
        if (native::physical(bytes)) { machine = bytes; }
    }

    // There is always a limit (a job's, host.budget_mib or the machine's memory), so an unreadable envelope is a
    // failed query, never the absence of one.
    bool unbounded() const { return false; }

    // The finite envelopes in force; false when one that applies cannot be read.
    bool limits(envelopes & out) const {
        out.count = 0;
        if (!native::job_limits(out)) { return false; }
        if (!budget && out.count) { return true; }
        uint64_t used = 0;
        if (!native::process_use(used)) { return false; }
        if (budget) { out.push({"budget", budget, used}); }
        else if (machine) { out.push({"machine", machine, used}); }
        else { return false; }
        return true;
    }
    bool memory_available(uint64_t & bytes) const { return native::available(bytes); }
    bool memory_available_at_full_boost(uint64_t & bytes, uint64_t & headroom) const {
        headroom = 0;
        return native::available(bytes);
    }
    bool update_file_reserve() { return false; }

    // Every envelope must keep the reserve free after `extra`; the tightest one binds, as the cgroup ancestors do.
    // The limits must hold `extra` alone.
    void envelope_term(uint64_t extra, room & r) const {
        envelopes found;
        if (!limits(found)) { return; }
        uint64_t missing = 0, spare = UINT64_MAX, lacking = 0;
        for (size_t i = 0; i < found.count; ++i) {
            const envelope & e = found.item[i];
            bound(e.limit, e.used, add(reserve, extra), missing, spare);
            lacking = std::max(lacking, lack(e.limit, e.used, extra));
        }
        set_envelope(missing, spare, r);
        set_limits(lacking, r);
    }
    room measure(uint64_t extra, bool full_boost = false) const {
        room r;
        envelope_term(extra, r);
        uint64_t available = 0, headroom = 0;
        if (full_boost ? memory_available_at_full_boost(available, headroom) : memory_available(available)) {
            system_term(available, headroom, extra, r);
        }
        return r;
    }
};

}} // namespace ssd::host
