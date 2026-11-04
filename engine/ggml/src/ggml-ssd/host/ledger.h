#pragma once
// Host-memory ledger on Linux: the two terms of room.h from cgroup v2 and /proc. It measures; the arbiter
// (arbiter.h) admits allocations, runs the caches' reclaimers and learns the margin.
//  - envelope: every finite memory.max among this process's cgroup and its ancestors. On a discrete GPU it is the
//    term that binds. On a unified-memory GPU (DGX Spark, Jetson) the cgroup does not charge device allocations, so
//    the device memory the expert caches hold (device_held: the GPU tier's blocks and buffers) counts as used beside
//    memory.current: the GPU tier and the RAM cache come out of one envelope.
//  - system: MemAvailable less the floor. On a unified-memory GPU a device allocation comes out of the same DRAM
//    without being charged to the cgroup, so this term sees the GPU tier, the KV cache, the dense weights and pinned
//    buffers.
// floor = file reserve + margin + explicit reserve:
//  - file reserve (unified memory only): MemAvailable counts the server's own page cache as available, but the
//    kernel evicts it before it can touch the tier's pinned blocks. Keep the file pages the server holds when the
//    ledger first measures them, plus one page for every page the kernel read back after evicting it (a refault
//    means the file working set did not fit). It only grows.
//  - margin: headroom for growth nobody accounts, at least the reserve and host.headroom_mib; the arbiter learns it.
//    The envelope keeps its own reserve: a memory.max already leaves the deployment's headroom out, and growth
//    outside the cgroup (CUDA graphs and pools on unified memory) does not count against it.
//  - explicit reserve: vram.reserve_mib on unified memory, which reserves the same DRAM.
// The GPU tier of a unified device receives the floor and keeps it, with the RAM cache's uncommitted slots, free.
//
// On unified memory MemAvailable also moves with nothing allocated: it keeps back the zones' low watermarks, and the
// kernel boosts them after fragmenting allocations by up to 1.5 times the high watermark (budget-linux.h).
// Whoever takes memory up to the floor (the tier's growth, an allocation outside the caches) counts MemAvailable as
// it reads at the full boost, so that the boost coming back cannot take MemAvailable under the floor.
//
// A prefill reads rows of weights by index straight from the model's file mapping (token embeddings, DeepSeek-V4.1's
// Engram tables): their pages come into the page cache while it runs, charged to the cgroup, and they must stay there
// until the layer that reads them has run. File pages count as room (reclaimable_file, MemAvailable), so nothing
// kept room for them: in a full cgroup the kernel took the rows read ahead back before they were used, and the prefill
// read them again one synchronous fault at a time. The prefill's graph prices them (prefill_pages) before it runs; both
// terms hold that room from then until the first decode token, so the RAM cache gives it back at once at the prefill's
// start and takes it back at the first decode token.
#include "budget-linux.h"
#include "room.h"
#include <cstdint>
#include <cstring>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <fcntl.h>
#include <unistd.h>

struct kimi_host_ledger : ssd::host::ledger_terms {
    using room = ssd::host::room;
    // The files the ledger reads stay open: a cgroup or /proc file is generated again by every read from offset 0,
    // and opening one costs several times reading it (the RAM cache asks once or twice per layer and token). Copies
    // of a ledger share them.
    struct open_files {
        std::mutex mutex;
        std::map<std::string, int> fds;
        ~open_files() { for (const auto & f : fds) { close(f.second); } }
        bool read(const std::string & path, std::string & text) {
            std::lock_guard<std::mutex> lock(mutex);
            auto it = fds.find(path);
            if (it == fds.end()) {
                const int fd = open(path.c_str(), O_RDONLY | O_CLOEXEC);
                if (fd < 0) { return false; }
                it = fds.emplace(path, fd).first;
            }
            text.clear();
            char buffer[4096];
            for (off_t at = 0;;) {
                const ssize_t n = pread(it->second, buffer, sizeof(buffer), at);
                if (n < 0) { close(it->second); fds.erase(it); return false; } // a removed cgroup: open again next time
                if (!n || text.size() > (1u << 20)) { break; }
                text.append(buffer, size_t(n));
                at += n;
            }
            return true;
        }
    };
    std::shared_ptr<open_files> files = std::make_shared<open_files>();
    // Tests inject their own files.
    using reader = std::function<bool(const std::string & path, std::string & text)>;
    reader read;
    bool get(const std::string & path, std::string & text) const { return read ? read(path, text) : files->read(path, text); }
    std::string cgroup;             // "/sys/fs/cgroup/<this process's cgroup>", empty when unknown
    // Unified memory: device memory the expert caches hold in this DRAM, which the cgroup does not charge. The
    // envelope counts it as used; MemAvailable already leaves it out.
    uint64_t device_held = 0;
    uint64_t page_size = 4096;
    bool file_measured = false;

    static bool decimal(const std::string & text, uint64_t & value) {
        if (text.empty()) { return false; }
        value = 0;
        for (const char c : text) {
            if (c < '0' || c > '9' || value > (UINT64_MAX - uint64_t(c - '0')) / 10) { return false; }
            value = value * 10 + uint64_t(c - '0');
        }
        return true;
    }
    static std::string first_token(const std::string & text) {
        const size_t begin = text.find_first_not_of(" \t\n");
        if (begin == std::string::npos) { return ""; }
        return text.substr(begin, text.find_first_of(" \t\n", begin) - begin);
    }
    // The value of `key` in "key value" lines (memory.stat, /proc/vmstat) or "key: value kB" lines (/proc/meminfo).
    static bool field(const std::string & text, const char * key, uint64_t & value) {
        const size_t length = std::strlen(key);
        for (size_t at = 0; at < text.size();) {
            const size_t end = std::min(text.find('\n', at), text.size());
            if (end - at > length && !text.compare(at, length, key) &&
                    (text[at + length] == ' ' || text[at + length] == ':')) {
                size_t begin = at + length + 1;
                while (begin < end && (text[begin] == ' ' || text[begin] == '\t')) { ++begin; }
                size_t stop = begin;
                while (stop < end && text[stop] >= '0' && text[stop] <= '9') { ++stop; }
                if (!decimal(text.substr(begin, stop - begin), value)) { return false; }
                if (text.compare(stop, 3, " kB") == 0) {
                    if (value > UINT64_MAX / 1024) { return false; }
                    value *= 1024;
                }
                return true;
            }
            at = end + 1;
        }
        return false;
    }

    // This process's cgroup v2 directory (from /proc/self/cgroup) and the page size. The arbiter calls it once.
    void locate() {
        const long page = sysconf(_SC_PAGESIZE);
        page_size = page > 0 ? uint64_t(page) : 4096;
        std::string groups;
        cgroup.clear();
        if (!get("/proc/self/cgroup", groups)) { return; }
        for (size_t at = 0; at < groups.size();) {
            const size_t end = std::min(groups.find('\n', at), groups.size());
            if (!groups.compare(at, 3, "0::")) { cgroup = "/sys/fs/cgroup" + groups.substr(at + 3, end - at - 3); return; }
            at = end + 1;
        }
    }
    // Whether no ancestor of this process's cgroup has a finite memory.max: an unreadable envelope is then no limit,
    // not a missing one. False when the cgroup is unknown or a memory.max below the root cannot be read.
    bool unbounded() const {
        if (cgroup.empty()) { return false; }
        const std::string root = "/sys/fs/cgroup";
        for (std::string path = cgroup, text; path.size() >= root.size();) {
            if (!get(path + "/memory.max", text)) { return path == root || path == root + "/"; }
            if (first_token(text) != "max") { return false; }
            const auto slash = path.find_last_of('/');
            if (slash == std::string::npos) { break; }
            path.resize(slash);
        }
        return true;
    }

    // MemAvailable from /proc/meminfo.
    bool memory_available(uint64_t & bytes) const {
        std::string text;
        return get("/proc/meminfo", text) && field(text, "MemAvailable", bytes);
    }
    // Unified memory: MemAvailable as it reads at every zone's largest watermark boost, from the same /proc/meminfo,
    // and what that keeps back (headroom); elsewhere MemAvailable. An unreadable /proc/zoneinfo or boost factor keeps
    // nothing back.
    bool memory_available_at_full_boost(uint64_t & bytes, uint64_t & headroom) const {
        std::string text, zones, factor;
        headroom = 0;
        if (!get("/proc/meminfo", text) || !field(text, "MemAvailable", bytes)) { return false; }
        if (unified && get("/proc/zoneinfo", zones) && get("/proc/sys/vm/watermark_boost_factor", factor)) {
            headroom = std::min(bytes, kimi_memory_boost::headroom(text, zones, factor, page_size));
        }
        bytes -= headroom;
        return true;
    }
    // The envelope: every finite memory.max ancestor must keep the reserve and the open prefill's pages free after
    // `extra`, the device memory the caches hold on unified memory counted as used beside memory.current. Unreadable
    // controller files below the root disable the term (and with it RAM cache admission) conservatively.
    void cgroup_term(uint64_t extra, room & r) const {
        if (cgroup.empty()) { return; }
        const size_t root = std::strlen("/sys/fs/cgroup");
        std::string path = cgroup, text;
        bool bounded = false;
        uint64_t missing = 0, spare = UINT64_MAX;
        while (path.size() >= root) {
            const std::string maximum = get(path + "/memory.max", text) ? first_token(text) : "";
            if (maximum.empty()) {
                // The cgroup-v2 root has no memory.max.
                if (path != "/sys/fs/cgroup" && path != "/sys/fs/cgroup/") { return; }
            } else if (maximum != "max") {
                bounded = true;
                uint64_t limit = 0, used = 0;
                if (!decimal(maximum, limit) || !get(path + "/memory.current", text) || !decimal(first_token(text), used)) { return; }
                if (reclaimable_file || prefill_pages) {
                    // File-backed page cache is reclaimable before anonymous memory (dirty pages may first need
                    // writeback). With reclaimable_file both LRU lists count as headroom, and the open prefill's
                    // pages are required beside the reserve whether they have come in yet or not. Without it file
                    // pages are use, except up to the prefill's pages: those are the ones it holds room for (or
                    // pages the kernel takes back to make that room), not more use. Unreadable stats fail closed.
                    uint64_t active = 0, inactive = 0;
                    if (!get(path + "/memory.stat", text) || !field(text, "active_file", active) ||
                            !field(text, "inactive_file", inactive)) { return; }
                    const uint64_t file = ssd::host::add(active, inactive);
                    used -= std::min(used, reclaimable_file ? file : std::min(file, prefill_pages));
                }
                bound(limit, ssd::host::add(used, device_held), ssd::host::add(ssd::host::add(reserve, prefill_pages), extra),
                    missing, spare);
            }
            const size_t slash = path.find_last_of('/');
            if (slash == std::string::npos) { break; }
            path.resize(slash);
        }
        if (bounded) { set_envelope(missing, spare, r); }
    }
    // full_boost: the system term counts MemAvailable at the full watermark boost, for memory about to be taken up
    // to the floor (the arbiter's take and make_room). The RAM cache's per-layer guard reads MemAvailable as it is:
    // it commits only the slots whose room the tier keeps free as the cache's demand, with the boost headroom beside
    // it. MemAvailable counts page cache as available, so the open prefill's pages are required beside the floor, as
    // in the envelope.
    room measure(uint64_t extra, bool full_boost = false) const {
        room r;
        cgroup_term(extra, r);
        uint64_t available = 0, headroom = 0;
        if (full_boost ? memory_available_at_full_boost(available, headroom) : memory_available(available)) {
            system_term(available, headroom, extra, r);
        }
        return r;
    }

    // The server's file pages and refault count: its cgroup's memory.stat, else the system's counters.
    bool file_working_set(uint64_t & file, uint64_t & refault) const {
        std::string text;
        if (!cgroup.empty() && get(cgroup + "/memory.stat", text) && field(text, "file", file) &&
                field(text, "workingset_refault_file", refault)) { return true; }
        uint64_t pages = 0;
        if (get("/proc/vmstat", text) && field(text, "nr_file_pages", pages) && field(text, "workingset_refault_file", refault)) {
            file = pages > UINT64_MAX / page_size ? UINT64_MAX : pages * page_size;
            return true;
        }
        return false;
    }
    // Unified memory: measures the file reserve the first time, then adds the refaults since. True when it grew.
    bool update_file_reserve() {
        if (!unified) { return false; }
        uint64_t file = 0, refault = 0;
        if (!file_working_set(file, refault)) { return false; }
        uint64_t next = file_reserve;
        if (!file_measured) { next = std::max(next, file); file_measured = true; }
        else if (refault > refaults) {
            next = ssd::host::add(next, (refault - refaults) > UINT64_MAX / page_size ? UINT64_MAX : (refault - refaults) * page_size);
        }
        refaults = refault;
        if (next == file_reserve) { return false; }
        file_reserve = next;
        ++file_updates;
        return true;
    }
};
