// SPDX-License-Identifier: Apache-2.0
#pragma once
// Room in this machine's DRAM for `extra` more bytes, the same arithmetic on every OS. Two terms:
//  - envelope: every finite limit on this process's memory (Linux: memory.max of its cgroup and every ancestor;
//    macOS and Windows: a job limit, host.budget_mib or the envelope derived at startup) keeps the reserve and the
//    open prefill's pages free after `extra`. The tightest limit binds.
//  - system: the memory a new allocation may take (Linux: MemAvailable) keeps the floor and the open prefill's pages
//    free after `extra`.
// The ledgers read the limits and the available memory (ledger.h on Linux, ledger-native.h elsewhere); the arbiter
// (arbiter.h) admits allocations and runs the caches' reclaimers with what they measure.
#include <algorithm>
#include <cstdint>

namespace ssd { namespace host {

inline uint64_t add(uint64_t a, uint64_t b) noexcept { return b > UINT64_MAX - a ? UINT64_MAX : a + b; }

enum room_state { room_ok, room_pressure, room_unreadable };

struct room {
    // The envelope term (named for Linux, where it is the cgroup). A term that cannot be read does not bind; the
    // arbiter and the RAM cache refuse to admit under an unreadable envelope themselves.
    room_state cgroup = room_unreadable;
    uint64_t cgroup_shortage = 0, cgroup_surplus = 0;
    bool system = false; // the available memory was read
    // available: what a new allocation may take, less boost_headroom when measured at the full watermark boost
    uint64_t available = 0, floor = 0, system_shortage = 0, system_surplus = 0, boost_headroom = 0;
    [[nodiscard]] bool fits() const noexcept { return !cgroup_shortage && !system_shortage; }
    [[nodiscard]] uint64_t shortage() const noexcept { return std::max(cgroup_shortage, system_shortage); }
    // What both readable terms leave; UINT64_MAX when neither binds.
    [[nodiscard]] uint64_t surplus() const noexcept {
        if (!fits()) { return 0; }
        uint64_t value = UINT64_MAX;
        if (cgroup != room_unreadable) { value = std::min(value, cgroup_surplus); }
        if (system) { value = std::min(value, system_surplus); }
        return value;
    }
};

// What both ledgers hold and compute; each adds only where its numbers come from.
struct ledger_terms {
    bool unified = false;            // the GPU allocates out of this DRAM
    bool reclaimable_file = false;   // the envelope counts the file LRU as headroom (Linux)
    uint64_t reserve = 512ull << 20; // what every envelope keeps free
    uint64_t margin = 512ull << 20;  // the floor's headroom for growth nobody accounts; the arbiter learns it
    uint64_t explicit_reserve = 0;   // vram.reserve_mib where the GPU shares the DRAM
    // The open prefill's file pages (the weight rows its graph gathers), held in both terms until the first decode.
    uint64_t prefill_pages = 0;
    // Unified memory on Linux (ledger.h): the server's file pages the floor keeps, grown by refaults.
    uint64_t file_reserve = 0, refaults = 0, file_updates = 0;

    [[nodiscard]] uint64_t floor() const noexcept { return add(add(file_reserve, margin), explicit_reserve); }

    // One finite limit of the envelope: what `used` leaves for `required` more bytes, folded into the tightest so far.
    static void bound(uint64_t limit, uint64_t used, uint64_t required, uint64_t & missing, uint64_t & spare) noexcept {
        uint64_t deficit = 0;
        if (used >= limit) { deficit = add(used - limit, required); }
        else if (required > limit - used) { deficit = required - (limit - used); }
        missing = std::max(missing, deficit);
        spare = std::min(spare, deficit ? uint64_t(0) : limit - used - required);
    }
    // The envelope term from its limits' folded shortage and spare.
    static void set_envelope(uint64_t missing, uint64_t spare, room & r) noexcept {
        r.cgroup = missing ? room_pressure : room_ok;
        r.cgroup_shortage = missing;
        r.cgroup_surplus = missing ? 0 : spare;
    }
    // The system term from the memory available to a new allocation (less `headroom` at the full boost).
    void system_term(uint64_t available, uint64_t headroom, uint64_t extra, room & r) const noexcept {
        r.system = true;
        r.available = available;
        r.boost_headroom = headroom;
        r.floor = floor();
        const uint64_t need = add(add(r.floor, prefill_pages), extra);
        r.system_shortage = need > available ? need - available : 0;
        r.system_surplus = available > need ? available - need : 0;
    }
};

}} // namespace ssd::host
