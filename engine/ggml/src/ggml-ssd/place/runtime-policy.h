#pragma once
// Numeric policy boundary. Observations are copied scalars or borrowed expert
// arrays; plans contain request/rank indices, never sink, cache, worker or device
// handles. The executor gathers observations and applies each action at its
// existing synchronization point. All learned state belongs to one executor.
#include "layer-plan.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <deque>
#include <utility>

namespace ssd { namespace place {

struct decode_learning {
    double cpu_expert_us = 0, read_bytes_per_us = 0;
    std::deque<std::pair<double, double>> reads;
    double early = 0, fixed_us = 0, cpu_alone_us = 0, cpu_shared_us = 0;
    double cpu_moved() const {
        return cpu_alone_us > 0 && cpu_shared_us > 0 ? std::max(1.0, cpu_shared_us / cpu_alone_us) : 1.0;
    }
};
struct expert_feedback { double expert_us, early_us; bool no_reads, moved; };
inline void observe_expert(decode_learning & s, const expert_feedback & f) {
    s.cpu_expert_us = s.cpu_expert_us > 0 ? 0.9 * s.cpu_expert_us + 0.1 * f.expert_us : f.expert_us;
    if (f.expert_us > 0) {
        const double early = f.early_us / f.expert_us;
        s.early = s.early > 0 ? 0.9 * s.early + 0.1 * early : early;
        if (f.no_reads) {
            double & line = f.moved ? s.cpu_shared_us : s.cpu_alone_us;
            line = line > 0 ? 0.9 * line + 0.1 * f.expert_us : f.expert_us;
        }
    }
}
inline void observe_read_rate(decode_learning & s, uint64_t read_bytes, double io_us) {
    if (read_bytes && io_us > 0) {
        const double rate = double(read_bytes) / io_us;
        s.read_bytes_per_us = s.read_bytes_per_us > 0 ? 0.9 * s.read_bytes_per_us + 0.1 * rate : rate;
    }
}
inline double observe_layer(decode_learning & s, uint64_t read_bytes, double io_us, double lanes_us, double total_us) {
    if (read_bytes && io_us > 0) {
        s.reads.emplace_back(double(read_bytes) / 1e6, io_us);
        if (s.reads.size() > 64) { s.reads.pop_front(); }
    }
    const double fixed = std::max(0.0, total_us - lanes_us);
    s.fixed_us = s.fixed_us > 0 ? 0.9 * s.fixed_us + 0.1 * fixed : fixed;
    return fixed;
}
// The CPUs a decode pool leaves to the machine's other work while decode runs (the GPU backend's threads, the reads,
// other processes): the work's measured CPUs, other_ns of busy time over wall_ns, rounded up. The counters put the
// sums within error_ns; while that band holds a whole number of CPUs the current choice stays, since the measurement
// cannot tell the two counts apart.
inline int decode_reserve(int current, double other_ns, double wall_ns, double error_ns) {
    if (!(wall_ns > 0)) { return current; }
    const int low = std::max(0, int(std::ceil((other_ns - error_ns) / wall_ns)));
    const int high = std::max(0, int(std::ceil((other_ns + error_ns) / wall_ns)));
    return low == high ? low : current;
}
// Decode's threads: its pool, within the CPUs the reserve leaves of `cpus`, at least one.
inline int decode_participants(int pool, int cpus, int reserve) { return std::max(1, std::min(pool, cpus - reserve)); }

struct cost_lines {
    double cpu_fixed_us = 204.0, cpu_route_us = 87.0;
    double gpu_fixed_us = 25.0, gpu_route_us = 8.2;
    double gpu_host_fixed_us = 0.0, gpu_host_route_us = 0.0;
    double link_gbps = 6.74, pageable_gbps = 0.0;
    double drive_gbps = 0.0, drive_fixed_us = 0.0;
    double gpu_overhead_fixed_us = 0.0, gpu_token_us = -1.0;
};
struct lane_measurement {
    double resident_call_us, resident_expert_us, host_call_us, host_expert_us;
    int vector_tokens;
    double copy_us_per_mb, pageable_us_per_mb, link_busy_us;
    bool in_place, in_place_only;
};
struct lane_geometry {
    uint64_t token_bytes, output_bytes;
    bool batched, in_place;
};
struct prefill_sample {
    kimi_prefill_fit cpu, gpu, gpu_host;
    double read_bytes, read_us, gpu_overhead_us, gpu_tokens;
};
struct prefill_learning {
    kimi_prefill_fit cpu_fit;
    size_t window = 0;
    std::vector<prefill_sample> samples;
    size_t next = 0;
    uint64_t calibrations = 0;
};
inline double overhead_row_prior(const cost_lines & prior, const lane_geometry & lane, double locked_us_per_mb) {
    if (prior.gpu_token_us >= 0.0) { return prior.gpu_token_us; }
    const double us_per_mb = locked_us_per_mb > 0.0 ? locked_us_per_mb : 1e3 / prior.link_gbps;
    return (double(lane.token_bytes) + double(lane.output_bytes)) / 1e6 * us_per_mb;
}
inline void prefill_reads(const prefill_learning & s, kimi_prefill_fit & drive) {
    for (const auto & x : s.samples) {
        if (x.read_bytes > 0.0 && x.read_us > 0.0) { drive.add(x.read_bytes / 1e6, x.read_us); }
    }
}
inline kimi_place_table make_table(const cost_lines & s, const cost_lines & prior,
        const decode_learning & decode, const prefill_learning & prefill,
        const lane_measurement & c, const lane_geometry & lane, int ntokens, double route_bytes) {
    kimi_place_table t;
    t.cpu_fixed_us = s.cpu_fixed_us; t.cpu_route_us = s.cpu_route_us;
    t.cpu_one_us = decode.cpu_expert_us > 0 ? decode.cpu_expert_us : s.cpu_fixed_us + s.cpu_route_us;
    t.cpu_early = std::clamp(decode.early, 0.0, 1.0);
    t.cpu_moved = decode.cpu_moved();
    t.vector.device = { c.resident_call_us, c.resident_expert_us, 0.0 };
    t.vector.moved = { c.host_call_us > 0 ? c.host_call_us : c.resident_call_us,
        c.host_expert_us > 0 ? c.host_expert_us : c.resident_expert_us, 0.0 };
    t.vector_tokens = c.vector_tokens > 0 ? c.vector_tokens : 1;
    t.batched.device = { 0.0, s.gpu_fixed_us, s.gpu_route_us };
    if (s.gpu_host_fixed_us > 0 || s.gpu_host_route_us > 0) { t.batched.moved = { 0.0, s.gpu_host_fixed_us, s.gpu_host_route_us }; }
    t.locked_us_per_mb = c.copy_us_per_mb;
    const double rows_us_per_mb = c.copy_us_per_mb > 0 ? c.copy_us_per_mb : 1e3 / s.link_gbps;
    t.pageable_us_per_mb = c.pageable_us_per_mb > 0 ? c.pageable_us_per_mb :
        1e3 / (s.pageable_gbps > 0 ? s.pageable_gbps : s.link_gbps);
    t.link_busy_us = c.link_busy_us;
    t.in_place = c.in_place && (t.one(ntokens) || (lane.batched && lane.in_place));
    t.in_place_only = c.in_place_only;
    t.route_us = t.one(ntokens) ? 0.0 : route_bytes / 1e6 * rows_us_per_mb;
    t.overhead_fixed_us = s.gpu_overhead_fixed_us;
    t.token_us = s.gpu_token_us >= 0 ? s.gpu_token_us : overhead_row_prior(prior, lane, c.copy_us_per_mb);
    kimi_prefill_fit drive;
    for (const auto & x : decode.reads) { drive.add(x.first, x.second); }
    prefill_reads(prefill, drive);
    double fixed = s.drive_fixed_us, slope = s.drive_gbps > 0 ? 1e3 / s.drive_gbps :
        decode.read_bytes_per_us > 0 ? 1e6 / decode.read_bytes_per_us : 0.0;
    if (drive.n > 0) { kimi_prefill_fit_line(drive, s.drive_fixed_us, slope > 0 ? slope : 1.0, fixed, slope); }
    t.drive_fixed_us = fixed; t.drive_us_per_mb = slope;
    return t;
}

struct decode_plan {
    bool valid = false;
    std::vector<uint8_t> gpu;
    double predicted_us = 0, cpu_only_us = 0;
    size_t residents = 0, ram = 0, misses = 0, candidates = 0, moves = 0, locked = 0;
    size_t pending = 0, pending_moves = 0;
};
inline void describe_decode(decode_plan & plan, std::vector<kimi_place_item> & items, const std::vector<double> & pending) {
    plan.residents = plan.ram = plan.misses = plan.locked = plan.pending = plan.pending_moves = 0;
    for (size_t i = 0; i < items.size(); ++i) {
        auto & x = items[i];
        if (x.resident) { ++plan.residents; continue; }
        if (x.routes <= 0) { continue; }
        if (!x.read) {
            ++plan.ram; plan.locked += x.movable && x.locked;
            x.pending_mb = pending[i];
            plan.pending += pending[i] > 0.0;
        } else { ++plan.misses; }
    }
}
inline kimi_place_result plan_decode(decode_plan & plan, const kimi_place_table & table,
        size_t ring_slots, const std::vector<kimi_place_item> & items, const kimi_place_limits & limits) {
    auto placed = kimi_place_layer(table, 1, ring_slots, items.data(), items.size(), limits);
    plan.gpu = placed.gpu;
    for (size_t i : placed.moved) { plan.pending_moves += items[i].pending_mb > 0.0; }
    plan.cpu_only_us = placed.time_none; plan.predicted_us = placed.time;
    plan.candidates = placed.candidates; plan.moves = placed.moved.size(); plan.valid = true;
    return placed;
}
struct lock_plan { std::vector<size_t> requests; double saved_us = 0; bool considered = false; };
inline lock_plan plan_locks(const kimi_place_table & table, size_t ring_slots,
        const std::vector<kimi_place_item> & items, const kimi_place_limits & limits, double current_us) {
    std::vector<size_t> unlocked;
    auto would = items;
    for (size_t i = 0; i < items.size(); ++i) {
        auto & x = would[i];
        if (x.resident || x.routes <= 0) { continue; }
        if (x.read) { x.movable = false; continue; }
        if (x.movable && !x.locked) { x.locked = true; unlocked.push_back(i); }
    }
    lock_plan result;
    if (unlocked.empty()) { return result; }
    result.considered = true;
    kimi_place_table free_copy = table;
    if (!(free_copy.locked_us_per_mb > 0)) { free_copy.locked_us_per_mb = 1e-12; }
    auto alt = kimi_place_layer(free_copy, 1, ring_slots, would.data(), would.size(), limits);
    result.saved_us = current_us - alt.time;
    result.requests = std::move(alt.moved);
    result.requests.erase(std::remove_if(result.requests.begin(), result.requests.end(), [&](size_t i) {
        return std::find(unlocked.begin(), unlocked.end(), i) == unlocked.end();
    }), result.requests.end());
    return result;
}

struct prefetch_plan { int layer, ranks; bool hinted, open_gate; };
inline prefetch_plan plan_prefetch(int layer, int hint_layer, int hint_n, int topk, bool demand_reads) {
    const bool hinted = hint_n && hint_layer == layer + 1;
    return {layer + 1, hinted ? std::min(topk, hint_n) : 0, hinted, !demand_reads};
}
struct prefetch_rank { bool novel, skipped, request; };
inline prefetch_rank plan_prefetch_rank(int32_t expert, bool held, bool cached, int rank, int topk) {
    return {expert >= 0 && !held && !cached, expert >= 0 && held && rank < topk, expert >= 0 && !held && rank < topk};
}

// Grouped-prefill feedback is measured by the executor, then applied before
// the next layer builds its table. The two stages deliberately leave the
// backend-cost observation between fitting expert lines and fitting overhead.
struct prefill_feedback {
    kimi_prefill_fit gpu, host;
    double upload_us, ssd_upload_us, engine_overhead_us, transient_wait_us;
    double output_add_us, total_worker_us, read_wait_us;
    uint64_t read_bytes;
    double read_us;
    int ntokens;
};
struct prefill_fit_result {
    double gpu_lane_us, gpu_overhead_us;
    kimi_prefill_overhead_fit overhead;
};
inline prefill_fit_result observe_prefill_lines(prefill_learning & learning, cost_lines & cost,
        const cost_lines & prior, const prefill_feedback & f) {
    const double upload_us = f.upload_us + f.ssd_upload_us;
    const bool gpu_work = f.gpu.n + f.host.n > 0.0;
    const bool engine = f.engine_overhead_us > 0.0;
    const double gpu_lane_us = f.gpu.t + f.host.t + (engine ? f.transient_wait_us : upload_us);
    const double gpu_overhead_us = !gpu_work ? 0.0 : f.output_add_us + (engine ? f.engine_overhead_us :
        std::max(0.0, f.total_worker_us - f.read_wait_us - gpu_lane_us));
    const prefill_sample sample = { learning.cpu_fit, f.gpu, f.host, double(f.read_bytes), f.read_us,
        gpu_overhead_us, gpu_work ? double(f.ntokens) : 0.0 };
    if (learning.samples.size() < learning.window) { learning.samples.push_back(sample); }
    else { learning.samples[learning.next] = sample; }
    learning.next = (learning.next + 1) % learning.window;
    kimi_prefill_fit cpu, gpu, gpu_host;
    kimi_prefill_overhead_fit overhead;
    for (const auto & s : learning.samples) {
        cpu.add(s.cpu); gpu.add(s.gpu); gpu_host.add(s.gpu_host);
        if (s.gpu_tokens > 0.0) { overhead.add(int(s.gpu_tokens), s.gpu_overhead_us); }
    }
    kimi_prefill_fit_line(cpu, prior.cpu_fixed_us, prior.cpu_route_us, cost.cpu_fixed_us, cost.cpu_route_us);
    kimi_prefill_fit_line(gpu, prior.gpu_fixed_us, prior.gpu_route_us, cost.gpu_fixed_us, cost.gpu_route_us);
    kimi_prefill_fit_line(gpu_host, prior.gpu_host_fixed_us > 0 || prior.gpu_host_route_us > 0 ? prior.gpu_host_fixed_us :
        prior.gpu_fixed_us, prior.gpu_host_fixed_us > 0 || prior.gpu_host_route_us > 0 ? prior.gpu_host_route_us :
        prior.gpu_route_us, cost.gpu_host_fixed_us, cost.gpu_host_route_us);
    return { gpu_lane_us, gpu_overhead_us, overhead };
}
inline void observe_prefill_overhead(prefill_learning & learning, cost_lines & cost,
        const kimi_prefill_overhead_fit & overhead, double row_prior) {
    overhead.fit(row_prior, cost.gpu_overhead_fixed_us, cost.gpu_token_us);
    ++learning.calibrations;
}

// Only metadata crosses the prefill policy boundary. Payload addresses, cache
// leases and their lifetime stay with the executor; this scratch is reused.
struct prefill_expert { int32_t id; bool resident, ram, leased; uint64_t bytes; };
struct prefill_choice { size_t index; int source; }; // 0: RAM, 1: SSD
struct prefill_selection {
    std::vector<prefill_choice> chosen;
    size_t ram_candidates = 0, ssd_candidates = 0, ssd_limit = 0;
    double time_before = 0, time_after = 0, cpu_before = 0, gpu_before = 0;
    double cpu_after = 0, gpu_after = 0, link_after = 0;
    bool in_ring = false;
    kimi_place_result placed;
};
struct prefill_input {
    const std::vector<prefill_expert> & experts;
    const std::vector<int> & routes;
    const std::vector<kimi_place_item> & items;
    const kimi_place_table & table;
    int ntokens;
    size_t ssd_staging_limit, ring_slots;
};
struct prefill_candidates {
    size_t ram_candidates = 0, ssd_candidates = 0, ssd_limit = 0;
};
// The experts a layer could move to the GPU: from a leased RAM entry, or from the drive.
inline prefill_candidates count_prefill(const prefill_input & in, uint64_t & candidate_count) {
    const auto & source = in.experts;
    prefill_candidates out;
    for (size_t i = 0; i < source.size(); ++i) {
        if (source[i].resident || in.routes[i] <= 0) { continue; }
        if (source[i].ram && source[i].leased) { ++out.ram_candidates; }
        else if (!source[i].ram) { ++out.ssd_candidates; }
    }
    candidate_count = out.ssd_candidates;
    out.ssd_limit = in.ssd_staging_limit;
    return out;
}
inline prefill_selection plan_prefill(const prefill_input & in, const prefill_candidates & counted, bool & in_place) {
    const auto & source = in.experts;
    const auto & table = in.table;
    prefill_selection out;
    out.ram_candidates = counted.ram_candidates; out.ssd_candidates = counted.ssd_candidates;
    out.ssd_limit = counted.ssd_limit;
    kimi_place_limits limits;
    limits.reads = out.ssd_limit;
    out.placed = kimi_place_layer(table, in.ntokens, in.ring_slots, in.items.data(), in.items.size(), limits);
    in_place = !table.one(in.ntokens) && out.placed.in_place;
    for (size_t i : out.placed.moved) { out.chosen.push_back({ i, source[i].ram ? 0 : 1 }); }
    out.time_before = out.placed.time_none; out.time_after = out.placed.time; out.in_ring = out.placed.ring;
    for (size_t i = 0; i < source.size(); ++i) {
        if (!source[i].resident) { out.cpu_before += table.cpu(in.ntokens, in.routes[i]); }
    }
    out.cpu_after = out.placed.cpu; out.gpu_after = out.placed.gpu_lane; out.link_after = out.placed.link;
    return out;
}

inline double miss_cost(size_t ram, size_t ssd, double cpu_expert_us, double io_us,
        bool in_place, double host_expert_us) {
    double compute = cpu_expert_us;
    if (in_place && host_expert_us > 0) { compute = std::min(compute, host_expert_us); }
    const double read = ssd ? io_us / double(ssd) : 0.0;
    return (double(ram) * compute + double(ssd) * std::max(compute, read)) / double(ram + ssd);
}

}} // namespace ssd::place
