// SPDX-License-Identifier: Apache-2.0
#pragma once
// Host-memory ledger on macOS and Windows: the two terms of room.h from native budget sources, as ledger.h measures
// them on Linux, so the arbiter and everything it admits run the same logic on every OS.
//
// Room for `extra` more bytes is the smaller of two terms.
//  - envelope, reported in the arbiter's cgroup fields: every finite limit on this process's memory keeps the
//    reserve free after `extra`. Use is what the OS charges this process for its own memory (Windows commit,
//    macOS phys_footprint), so the model's file mapping is not use on either OS.
//      Windows: the limits the process's job sets in its LimitFlags (a job alone is no limit: OpenSSH and
//      terminals run processes in jobs without one): JOB_OBJECT_LIMIT_JOB_MEMORY against the job's commit,
//      JOB_OBJECT_LIMIT_PROCESS_MEMORY against the process's PrivateUsage. Only the immediate job is visible; a
//      limit set on the parent of a nested job is not.
//      Both OSes: host.budget_mib against the process's use (Windows PrivateUsage, macOS phys_footprint). It
//      applies beside a job limit, the tighter binding.
//    With host.budget_mib 0 (the default) and no job limit, the envelope is derived once, when the ledger is
//    created: the process's use then plus the memory available then. Its room starts at what was available less
//    the reserve and shrinks only by this process's own growth; the system term follows everyone else.
//    A failed OS query leaves the term unreadable, and the RAM cache then admits nothing.
//  - system: memory available to a new allocation, less the floor and the open prefill's pages.
//      Windows: min(ullAvailPhys, ullAvailPageFile), as commit is a hard limit there.
//      macOS: free, file-backed, purgeable and reusable pages (the kernel's live counters, through sysctl): what
//      the kernel gives without compressing or swapping anyone's anonymous memory, the counterpart of MemAvailable.
// floor = margin + explicit reserves, as on Linux.
//
// How the Linux-specific parts carry over:
//  - file_reserve keeps a unified-memory CUDA tier off the server's page cache and grows with the kernel's
//    refault counter; neither OS has that tier or counter, so update_file_reserve() never grows it.
//  - The open prefill's file pages (weight rows read by index) are held in the system term, as on Linux, since it
//    counts cached file pages as available: on macOS every file-backed page, on Windows the standby list. A row
//    still in the working set is not on that list, so Windows holds it twice until the first decode token,
//    which errs safe. The envelope does not hold them: neither commit nor phys_footprint is charged for a
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
    const char * source; // "job", "job process", "budget" or "derived"
    uint64_t limit, used;
};
struct envelopes {
    envelope item[3];
    size_t count = 0;
    void push(const envelope & e) { if (count < 3) { item[count++] = e; } }
};

namespace native {
// What the OS charges this process for its own memory: Windows PrivateUsage, macOS phys_footprint.
bool process_use(uint64_t & bytes);
// Memory available to a new allocation (the system term before the floor).
bool available(uint64_t & bytes);
// Windows: appends the memory limits of the process's job; true with none outside a job. Elsewhere true, none.
bool job_limits(envelopes & out);
} // namespace native

// reclaimable_file and the file counters of ledger_terms stay inert here.
struct native_ledger : ledger_terms {
    std::string cgroup; // always empty: there is no cgroup to walk
    uint64_t budget = 0;        // host.budget_mib in bytes; 0 derives the envelope
    uint64_t derived_limit = 0; // the derived envelope, fixed at creation; 0 when unreadable then

    native_ledger() {
        uint64_t used = 0, now = 0;
        if (native::process_use(used) && native::available(now)) { derived_limit = add(used, now); }
    }

    // There is always a limit (a job's, host.budget_mib or the derived one), so an unreadable envelope is a failed
    // query, never the absence of one.
    bool unbounded() const { return false; }

    // The finite envelopes in force; false when one that applies cannot be read.
    bool limits(envelopes & out) const {
        out.count = 0;
        if (!native::job_limits(out)) { return false; }
        if (!budget && out.count) { return true; }
        uint64_t used = 0;
        if (!native::process_use(used)) { return false; }
        if (budget) { out.push({"budget", budget, used}); }
        else if (derived_limit) { out.push({"derived", derived_limit, used}); }
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
    void envelope_term(uint64_t extra, room & r) const {
        envelopes found;
        if (!limits(found)) { return; }
        uint64_t missing = 0, spare = UINT64_MAX;
        for (size_t i = 0; i < found.count; ++i) { bound(found.item[i].limit, found.item[i].used, add(reserve, extra), missing, spare); }
        set_envelope(missing, spare, r);
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
