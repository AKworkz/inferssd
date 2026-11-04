#pragma once
// The placement of a layer's routed experts, for any number of tokens: the per-expert fits, the layer's predicted time
// and the choice of the experts that move to the GPU. One table of measured lines and one choice (kimi_place_layer,
// at the end) for grouped prefill (kimi-cpu-sink-vram.inc, select_prefill_transient) and decode (kimi-cpu-sink.cpp,
// the decode plan). Kept apart from the sink so that it is tested alone (tests/test-kimi-prefill-cost.cpp).
//
// A layer's time has two forms, chosen by the physics of its reads:
//   - its reads stream through the ring (more reads than ring slots): the drive and the lanes hand the reads over
//     through the ring, kimi_prefill_time (the coupled form);
//   - every read is in flight at once (reads <= ring slots; decode always): each expert's bytes land at a known time,
//     and the lanes are a list schedule over those times, kimi_place_ring_time.
// The choice is the same for both: of the moves taken in their order, the fewest whose predicted time is within the
// form's resolution of the lowest, kimi_place_best_prefix. The coupled form does not resolve below one read (the
// drive hands the lanes their work one expert at a time); the list schedule places every read's landing, so it keeps
// no margin. On DGX Spark decode traces one read kept 0.1 of 33 profitable moves a token: a layer's gain is often
// below one read, and the token is the sum of 40 layers.
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

// Sums for a least-squares line y = fixed + slope * x: samples, x, x^2, y, x*y.
struct kimi_prefill_fit {
    double n = 0, x = 0, xx = 0, t = 0, xt = 0;
    void add(double xv, double yv) { n += 1; x += xv; xx += xv * xv; t += yv; xt += xv * yv; }
    void add(const kimi_prefill_fit & o) { n += o.n; x += o.x; xx += o.xx; t += o.t; xt += o.xt; }
};

// The least-squares line through the samples, pulled by one pseudo-sample at their mean x toward the prior line scaled
// to the samples' mean: a window of one x value stays solvable and keeps the prior's shape on this machine's level, and
// against the hundreds of experts of a layer the pseudo-sample does not move the fit. Both terms stay >= 0. Without
// samples the terms are left as they are.
inline void kimi_prefill_fit_line(const kimi_prefill_fit & s, double prior_fixed, double prior_slope,
        double & fixed, double & slope) {
    if (s.n < 1.0) { return; }
    const double mean = s.x / s.n;
    const double prior_mean = prior_fixed + prior_slope * mean;
    if (prior_mean > 0.0 && s.t > 0.0) {
        const double scale = s.t / s.n / prior_mean;
        prior_fixed *= scale; prior_slope *= scale;
    }
    const double a = s.n + 1.0, b = s.x, d = s.xx + mean * mean;
    const double y0 = s.t + prior_fixed, y1 = s.xt + mean * mean * prior_slope;
    const double det = a * d - b * b;
    double f = det > 0 ? (y0 * d - b * y1) / det : -1.0, r = det > 0 ? (a * y1 - b * y0) / det : -1.0;
    if (f < 0) { f = 0; r = y1 / d; }
    else if (r < 0) { r = 0; f = y0 / a; }
    fixed = f; slope = r;
}

// GPU begin/finish and output addition have both a fixed cost and row-dependent work. A single row count cannot
// identify their slope: keep the transfer-geometry prior and assign the measured remainder to the fixed part.
// With two distinct row counts, fit both terms by nonnegative least squares. No observations leave the last fit
// alone; sample expiry/exploration is a separate placement policy.
struct kimi_prefill_overhead_fit {
    kimi_prefill_fit samples;
    int min_tokens = 0, max_tokens = 0;
    void add(int tokens, double us) {
        if (tokens <= 0 || !std::isfinite(us) || us < 0.0) { return; }
        if (samples.n == 0.0) { min_tokens = max_tokens = tokens; }
        min_tokens = std::min(min_tokens, tokens); max_tokens = std::max(max_tokens, tokens);
        samples.add(double(tokens), us);
    }
    void fit(double row_prior_us, double & fixed_us, double & row_us) const {
        const auto & s = samples;
        if (s.n == 0.0) { return; }
        const double mean_tokens = s.x / s.n, mean_us = s.t / s.n;
        const double det = s.n * s.xx - s.x * s.x;
        if (min_tokens == max_tokens || det <= 0.0) {
            // If even the prior exceeds the whole observation, its largest feasible nonnegative slope is the
            // measured ratio. Otherwise the prior is retained exactly, rather than scaling a tiny shape's ratio.
            row_us = std::min(std::max(0.0, row_prior_us), mean_us / mean_tokens);
            fixed_us = std::max(0.0, mean_us - row_us * mean_tokens);
            return;
        }
        row_us = (s.n * s.xt - s.x * s.t) / det;
        fixed_us = (s.t - row_us * s.x) / s.n;
        if (row_us < 0.0) { row_us = 0.0; fixed_us = mean_us; }
        else if (fixed_us < 0.0) { fixed_us = 0.0; row_us = s.xt / s.xx; }
    }
};

// The time a producer busy for a and a consumer busy for b take together when they hand the work over through a ring
// of n buffers: the throughput of a two-station tandem queue with n places between the stations (exponential
// service), as a time. It is max(a, b) when one is far shorter, rises smoothly as they approach, and is
// max * (n + 1) / n when they are equal: each station then often waits for the other, a full ring stopping the
// producer and an empty one the consumer.
inline double kimi_prefill_coupled(double a, double b, double n) {
    const double high = std::max(a, b), low = std::max(0.0, std::min(a, b));
    if (high <= 0.0) { return 0.0; }
    n = std::max(1.0, n);
    const double r = low / high;
    if (r > 1.0 - 1e-9) { return high * (n + 1.0) / n; }
    return high * (1.0 - std::pow(r, n + 1.0)) / (1.0 - std::pow(r, n));
}

// One grouped-prefill layer before any expert moves to the GPU (us).
struct kimi_prefill_layer {
    double cpu = 0, gpu = 0, link = 0; // the lanes' busy times: the CPU, the GPU worker, the link
    double drive = 0, fill = 0;        // the drive's time for the layer's reads, and for the first of them
    double ring = 1;                   // the reads the ring holds between the drive and the lanes
    double gpu_overhead = 0;           // the GPU work's time outside its experts, paid once the layer has GPU work
    bool gpu_work = false;             // resident experts give the layer GPU work before any move
};
// What moving one expert to the GPU takes off the CPU lane and adds to the GPU lane and the link (us). copy: its copy
// into one of the tier's two device slots, which runs while the GPU computes another expert: DMA on the upload stream
// from page-locked host memory, or staged by the GPU's worker from pageable memory (the worker blocks, the chains it
// launched before run on); 0: none (in place). The GPU lane is then a two-stage pipeline (kimi_place_pipeline): the
// residents' computation, then each move's copy and computation.
struct kimi_prefill_move { double cpu, gpu, link, copy = 0.0; };

// The predicted layer time: the first read, the drive coupled with the slower lane, and the GPU overhead.
inline double kimi_prefill_time(const kimi_prefill_layer & l, double cpu, double gpu, double link, bool gpu_work) {
    return l.fill + kimi_prefill_coupled(l.drive, std::max(std::max(cpu, gpu), link), l.ring) +
        (gpu_work ? l.gpu_overhead : 0.0);
}

// The GPU lane as a two-stage pipeline over `slots` device slots: copy i (d[i], the link) waits for the copy before it
// and for the computation that last used its slot (i - slots); computation i (c[i]) waits for the one before it and
// for its copy. The end of the last computation. With one stage longer than the other for every pair, it is the
// longer stage's sum plus the shorter stage of the first (c < d: d_1 + ... + d_n + c_n; c > d: d_1 + c_1 + ... + c_n).
inline double kimi_place_pipeline(const double * c, const double * d, size_t n, size_t slots = 2) {
    std::vector<double> copied(n), computed(n);
    for (size_t i = 0; i < n; ++i) {
        double start = i ? copied[i - 1] : 0.0;
        if (i >= slots) { start = std::max(start, computed[i - slots]); }
        copied[i] = start + d[i];
        computed[i] = std::max(i ? computed[i - 1] : 0.0, copied[i]) + c[i];
    }
    return n ? computed[n - 1] : 0.0;
}

// How many of the moves, taken in their order, to make: the fewest whose predicted time is within `fill` of the lowest
// (the time model's resolution). time(k) is the predicted layer time with the first k moves made, for k = 0..count.
// *chosen receives the chosen time.
// Each time(k) is evaluated once.
template <class Time>
inline size_t kimi_place_best_prefix(size_t count, double fill, Time && time, double * chosen = nullptr) {
    std::vector<double> t(count + 1);
    for (size_t k = 0; k <= count; ++k) { t[k] = time(k); }
    const double lowest = *std::min_element(t.begin(), t.end());
    size_t k = 0;
    while (t[k] > lowest + fill && k < count) { ++k; }
    if (chosen) { *chosen = t[k]; }
    return k;
}

// The streamed form's lanes for every prefix of the moves: the CPU and the link accumulate in their order; the GPU
// lane is the pipeline of the residents' computation (l.gpu) and the moves' copies and computations over two device
// slots, which is the running sum of the computations when no move has a DMA copy (M8's lane).
inline void kimi_prefill_prefix_lanes(const kimi_prefill_layer & l, const kimi_prefill_move * moves, size_t count,
        std::vector<double> & cpu, std::vector<double> & gpu, std::vector<double> & link) {
    cpu.assign(count + 1, 0.0); gpu.assign(count + 1, 0.0); link.assign(count + 1, 0.0);
    cpu[0] = l.cpu; gpu[0] = l.gpu; link[0] = l.link;
    double copied = 0.0;
    for (size_t i = 0; i < count; ++i) {
        cpu[i + 1] = cpu[i] - moves[i].cpu; link[i + 1] = link[i] + moves[i].link;
        if (moves[i].copy > 0.0) {
            // the copy waits for the one before it and for the computation that last used its slot (two slots)
            copied = std::max(copied, i >= 2 ? gpu[i - 1] : 0.0) + moves[i].copy;
            gpu[i + 1] = std::max(gpu[i], copied) + moves[i].gpu;
        } else {
            gpu[i + 1] = gpu[i] + moves[i].gpu;
        }
    }
}

// The streamed form's choice. Its resolution is one read (l.fill): a gain shorter than one expert's read does not
// shorten the layer; it would only move work to the GPU for nothing.
inline size_t kimi_prefill_best_prefix(const kimi_prefill_layer & l, const kimi_prefill_move * moves, size_t count,
        double * time = nullptr) {
    std::vector<double> cpu, gpu, link;
    kimi_prefill_prefix_lanes(l, moves, count, cpu, gpu, link);
    return kimi_place_best_prefix(count, l.fill, [&](size_t k) {
        return kimi_prefill_time(l, cpu[k], gpu[k], link[k], k ? true : l.gpu_work);
    }, time);
}

// One expert of a layer whose reads are all in flight at once (us from the layer's start).
struct kimi_place_expert {
    double cpu = 0;          // its CPU time
    double early = 0;        // the part of cpu the CPU may start at early_ready (decode's up/gate; 0: none)
    double early_ready = 0;  // when the bytes of that part are in host memory
    double ready = 0;        // when all its bytes are in host memory (0: host-ready)
    double gpu = 0;          // its share of a GPU call
    double link = 0;         // its copy into device memory before that call (0: none, or the GPU reads it in place)
    bool resident = false;   // in the tier: in the residents' call; never moves
};
// The rest of such a layer (us).
struct kimi_place_lanes {
    double resident_call = 0; // a GPU call's own time, for the residents' call
    double host_call = 0;     // the same for a call of moved experts
    double gpu_overhead = 0;  // once when the layer has GPU work (grouped prefill's zeroing and output addition)
    double reads_end = 0;     // the last read's landing (0: no reads)
    double link_free = 0;     // when the link can take this layer's first copy
    double cpu_moved = 1;     // the CPU's time per expert, relative, in a layer with moved experts: their copies (or
                              // the GPU's reads in place) share the DRAM with the CPU (decode measures it; 1: not)
};

// The GPU lane's end of such a layer (and the link's, which feeds it), see kimi_place_ring_time.
inline double kimi_place_gpu_end(const kimi_place_lanes & l, const kimi_place_expert * e, size_t n, const uint8_t * on_gpu) {
    double gpu = 0.0, link = l.link_free;
    bool gpu_work = false;
    double resident = 0.0;
    size_t residents = 0;
    for (size_t i = 0; i < n; ++i) { if (e[i].resident) { resident += e[i].gpu; ++residents; } }
    if (residents) { gpu = l.resident_call + resident; gpu_work = true; }
    std::vector<size_t> moved;
    for (size_t i = 0; i < n; ++i) { if (!e[i].resident && on_gpu && on_gpu[i]) { moved.push_back(i); } }
    std::stable_sort(moved.begin(), moved.end(), [&](size_t a, size_t b) { return e[a].ready < e[b].ready; });
    std::vector<double> copied(moved.size());
    for (size_t j = 0; j < moved.size(); ++j) {
        const auto & x = e[moved[j]];
        link = x.link > 0.0 ? std::max(link, x.ready) + x.link : link;
        copied[j] = x.link > 0.0 ? link : x.ready;
    }
    for (size_t j = 0; j < moved.size();) {
        const double start = std::max(gpu, e[moved[j]].ready);
        double call = l.host_call, done = 0.0;
        size_t k = j;
        while (k < moved.size() && e[moved[k]].ready <= start) { call += e[moved[k]].gpu; done = std::max(done, copied[k]); ++k; }
        gpu = std::max(start, done) + call;
        gpu_work = true;
        j = k;
    }
    if (gpu_work) { gpu += l.gpu_overhead; }
    return gpu;
}

// The layer's time when its reads are all in flight: a list schedule of the lanes over the experts' landing times.
//   CPU: one processor; it takes a ready early part first, else a ready rest (an expert's rest after its early part),
//        lowest index first, as the decode dispatcher does (up/gate work wins).
//   link: the moved experts' copies, one at a time, each once its bytes are in host memory and the link is free.
//   GPU: the residents' call from the start, then the moved experts in calls: a call takes the moved experts whose
//        bytes are in host memory when the GPU is free (the tier's worker takes every launched job), and starts once
//        their copies are done.
//   T = max(CPU end, GPU end, last landing). With moved experts the CPU's times are scaled by cpu_moved.
// on_gpu[i] nonzero moves expert i (ignored for residents). cpu_end and gpu_end, when given, receive the lanes' ends.
inline double kimi_place_ring_time(const kimi_place_lanes & l, const kimi_place_expert * e, size_t n,
        const uint8_t * on_gpu, double * cpu_end = nullptr, double * gpu_end = nullptr) {
    const double gpu = kimi_place_gpu_end(l, e, n, on_gpu);
    bool moved = false;
    for (size_t i = 0; i < n && !moved; ++i) { moved = !e[i].resident && on_gpu && on_gpu[i]; }
    const double scale = moved ? l.cpu_moved : 1.0;
    // CPU: the list schedule's picks in its order, O(m log m) for m CPU experts. An early part is released at its
    // early_ready, a whole expert without one at its ready, a rest once its early part is done and its ready has come;
    // at each pick the CPU takes the released early part of the lowest index, else the released rest of the lowest
    // index, else waits for the next release.
    struct release { double at; size_t j; };
    const auto later = [](const release & a, const release & b) { return a.at > b.at || (a.at == b.at && a.j > b.j); };
    std::vector<release> early_wait, rest_wait;         // min-heaps by release time
    std::vector<size_t> early_ready, rest_ready;        // min-heaps by index
    const auto higher = [](size_t a, size_t b) { return a > b; };
    size_t left = 0;
    for (size_t i = 0; i < n; ++i) {
        const auto & x = e[i];
        if (x.resident || (on_gpu && on_gpu[i]) || !(x.cpu > 0.0)) { continue; }
        ++left;
        if (x.early > 0.0) { early_wait.push_back({ x.early_ready, i }); } else { rest_wait.push_back({ x.ready, i }); }
    }
    std::make_heap(early_wait.begin(), early_wait.end(), later);
    std::make_heap(rest_wait.begin(), rest_wait.end(), later);
    double cpu = 0.0;
    auto take = [&](std::vector<release> & wait, std::vector<size_t> & ready) {
        while (!wait.empty() && wait.front().at <= cpu) {
            ready.push_back(wait.front().j); std::push_heap(ready.begin(), ready.end(), higher);
            std::pop_heap(wait.begin(), wait.end(), later); wait.pop_back();
        }
    };
    while (left) {
        take(early_wait, early_ready);
        take(rest_wait, rest_ready);
        if (!early_ready.empty()) {
            std::pop_heap(early_ready.begin(), early_ready.end(), higher);
            const size_t j = early_ready.back(); early_ready.pop_back();
            cpu += std::min(e[j].early, e[j].cpu) * scale;
            rest_wait.push_back({ e[j].ready, j }); std::push_heap(rest_wait.begin(), rest_wait.end(), later);
            continue;
        }
        if (!rest_ready.empty()) {
            std::pop_heap(rest_ready.begin(), rest_ready.end(), higher);
            const size_t j = rest_ready.back(); rest_ready.pop_back();
            cpu += (e[j].early > 0.0 ? std::max(0.0, e[j].cpu - e[j].early) : e[j].cpu) * scale;
            --left;
            continue;
        }
        double next = INFINITY;
        if (!early_wait.empty()) { next = std::min(next, early_wait.front().at); }
        if (!rest_wait.empty()) { next = std::min(next, rest_wait.front().at); }
        cpu = next;
    }
    if (cpu_end) { *cpu_end = cpu; }
    if (gpu_end) { *gpu_end = gpu; }
    return std::max(std::max(cpu, gpu), l.reads_end);
}

// The in-ring choice: the candidates move in their order (indices into e), the first k of them for time(k); the fewest
// with the lowest predicted time (no margin: the schedule places every landing). A layer without early parts (grouped
// prefill) has the CPU's end of any order that never idles while work is released: its busy periods follow the
// work's arrivals. So its CPU lane is a fold over the experts by landing, sorted once for every prefix: O(n) a prefix
// instead of the list schedule's O(n log n), for layers of up to 256 experts.
inline size_t kimi_place_ring_best_prefix(const kimi_place_lanes & l, const kimi_place_expert * e, size_t n,
        const size_t * candidates, size_t count, double * time = nullptr) {
    std::vector<uint8_t> on_gpu(n, 0);
    size_t set = 0;
    bool early = false;
    std::vector<size_t> order;
    for (size_t i = 0; i < n; ++i) {
        if (e[i].resident || !(e[i].cpu > 0.0)) { continue; }
        early = early || e[i].early > 0.0;
        order.push_back(i);
    }
    if (!early) { std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) { return e[a].ready < e[b].ready; }); }
    return kimi_place_best_prefix(count, 0.0, [&](size_t k) {
        if (k < set) { std::fill(on_gpu.begin(), on_gpu.end(), 0); set = 0; }
        while (set < k) { on_gpu[candidates[set++]] = 1; }
        if (early) { return kimi_place_ring_time(l, e, n, on_gpu.data()); }
        double cpu = 0.0;
        const double scale = k ? l.cpu_moved : 1.0;
        for (size_t i : order) { if (!on_gpu[i]) { cpu = std::max(cpu, e[i].ready) + e[i].cpu * scale; } }
        return std::max(std::max(cpu, kimi_place_gpu_end(l, e, n, on_gpu.data())), l.reads_end);
    }, time);
}

// ---- One cost table and one choice for a layer of any number of tokens ----
//
// A layer of n tokens, decode's (n = 1) or a grouped-prefill pass's, is placed from one table of measured lines
// (kimi_place_table) by one function (kimi_place_layer): it prices each expert with the lines of the code that runs
// at n, builds the lanes of the form its reads take (the list schedule when they fit the ring, the coupled streamed
// form when they stream through it) and keeps the fewest moves within that form's resolution of the lowest time.
// Decode and prefill fill the table from the same measurements and describe their experts the same way
// (kimi_place_item), so a one-token prefill pass and a decode token of the same layer get the same answer.
//   GPU    the tier runs a pass of at most vector_tokens tokens on its vector engine (one call for a batch of
//          experts) and a longer one on its batched engine (one chain an expert over its tokens). Each engine has the
//          line of experts in device memory and the line of moved ones (computed in place from host memory where
//          the GPU reads it so, else from device memory after their copies; unmeasured: the device line).
//   CPU    one token: the measured time of a one-token expert; longer: the grouped path's fixed + per route. At one
//          token the CPU may start an expert's up and gate once those two roles are in (cpu_early of its time).
//   link   a copy from page-locked host memory is DMA on its own stream at the measured rate; one from pageable memory
//          is staged by the GPU's worker at the measured pageable rate (the worker waits, the chains it launched run
//          on). Either is the pipeline's first stage: the GPU computes another expert meanwhile. At one token the
//          tier's vector path copies only page-locked memory, and a one-route expert's staging would cost more than its
//          compute, so pageable sources move only in longer passes. A pass's activation rows: a token's input row once
//          per layer in the batched engine's time a token (token_us), and per route what the engine sends for a route
//          (route_us: the route list, or the hand engine's row a route).
//   in place  where the GPU reads host memory in place (in_place): one token always (decode's rule); a batched pass
//          when its moved line at the layer's mean routes off the tier is no longer than the device line plus the copy
//          (the choice of the layer, kimi_place_result::in_place). No copy, no device slot, the moved line.
//   drive  a layer's reads land in their issue order on one line, fixed + per MB, over decode's and prefill's
//          layers; at one token roles land role-major (every up and gate, then the downs), as decode issues them.
//          A RAM hit an earlier read still fills (decode's prefetch, its rest read at once when the layer starts) lands
//          on the same line ahead of them, at its bytes yet to come; it is a candidate like any expert, priced with that
//          landing as a read is with its own, and the executor hands it to its lane once it has landed.
struct kimi_place_line {
    double call_us = 0, expert_us = 0, route_us = 0; // a call's own time; per expert of the call: fixed + per route
    double expert(double routes) const { return expert_us + route_us * routes; }
    bool known() const { return call_us > 0 || expert_us > 0 || route_us > 0; }
};
struct kimi_place_engine {
    kimi_place_line device, moved;  // moved: the vector engine's host jobs; the batched engine's chains in place
    const kimi_place_line & of_moved() const { return moved.known() ? moved : device; }
};
struct kimi_place_table {
    double cpu_one_us = 0;                     // CPU, an expert of a one-token layer
    double cpu_fixed_us = 0, cpu_route_us = 0; // CPU, an expert of a longer pass: fixed + per route
    double cpu_early = 0;                      // one token: the share of an expert's CPU time before its down role
    double cpu_moved = 1;                      // the CPU's time in a layer with moved experts, relative (>= 1)
    kimi_place_engine vector, batched;
    int vector_tokens = 1;                     // passes of at most this many tokens run the vector engine
    double overhead_fixed_us = 0;             // batched: GPU work's fixed begin/finish cost, paid once per layer
    double token_us = 0;                       // batched: row-dependent work, full layer rows as a conservative bound
    double route_us = 0;                       // batched: per route of a GPU expert, what the engine sends for it
    double locked_us_per_mb = 0;               // a copy from page-locked memory, measured (0: none is, none moves)
    double pageable_us_per_mb = 0;             // a copy from pageable memory, staged by the GPU's worker (0: none)
    double link_busy_us = 0;                   // copies already queued on the link
    double drive_fixed_us = 0, drive_us_per_mb = 0; // a layer's reads: fixed + per MB (0: not known, no read time)
    bool in_place = false;                     // this pass's engine can read host memory in place (one token: does)
    bool in_place_only = false;                // and it has no device memory to copy to: nothing to compare with
    bool one(int ntokens) const { return ntokens <= vector_tokens; }
    const kimi_place_engine & engine(int ntokens) const { return one(ntokens) ? vector : batched; }
    double cpu(int ntokens, double routes) const { return one(ntokens) ? cpu_one_us : cpu_fixed_us + cpu_route_us * routes; }
};
// One expert of the layer as its caller has it.
struct kimi_place_item {
    int routes = 0;                // its tokens in the layer
    double mb = 0, mb_early = 0;   // its bytes, and of them its up and gate roles (MB)
    bool resident = false;         // in the tier: the residents' call computes it; never moves
    bool read = false;             // read from the drive in this layer (reads are issued in item order)
    bool locked = false;           // its host memory (where it is, or where its read lands) is page-locked
    bool movable = false;          // the caller can hand it to the GPU (a payload that stays until the layer ends)
    bool needs_slot = false;       // a move takes one of the tier's free device slots (kimi_place_limits::slots)
    double pending_mb = 0;         // not read here, still landing from an earlier read (a prefetch): its bytes yet to
                                   // come, ahead of the layer's reads on the drive's line (0: in host memory)
};
struct kimi_place_limits {
    size_t slots = SIZE_MAX;       // free device slots the moves that need one can take
    size_t reads = SIZE_MAX;       // moved experts that are read in the layer
};
struct kimi_place_result {
    std::vector<uint8_t> gpu;      // per item: moved to the GPU
    std::vector<size_t> moved;     // the moved items in their order (the GPU's order)
    std::vector<double> start;     // per item: its predicted start on its lane (the reads' issue order follows it)
    size_t candidates = 0;
    double time = 0, time_none = 0; // the predicted layer time with the moves, and with none
    bool ring = false;             // its reads fit the ring (the list schedule), else they stream through it
    double cpu = 0, gpu_lane = 0, link = 0, drive = 0, overhead = 0; // the chosen lanes
    double fill = 0;               // of the GPU lane, its waits for copies (the pipeline's fill and stalls)
    bool in_place = false;         // the moves are computed in place from host memory (else copied first)
};

// Places a layer of `ntokens` tokens: its items in their read order, `ring` the reads the ring holds.
inline kimi_place_result kimi_place_layer(const kimi_place_table & t, int ntokens, size_t ring,
        const kimi_place_item * items, size_t n, const kimi_place_limits & limits = kimi_place_limits()) {
    kimi_place_result out;
    out.gpu.assign(n, 0);
    out.start.assign(n, 0.0);
    const bool one = t.one(ntokens);
    const kimi_place_engine & engine = t.engine(ntokens);
    // a copy's rate by its source: page-locked at the measured rate (0: none measured, no price), else pageable
    auto copy_rate = [&](const kimi_place_item & x) {
        return x.locked && t.locked_us_per_mb > 0.0 ? t.locked_us_per_mb : x.locked ? 0.0 : t.pageable_us_per_mb;
    };
    // in place: one token wherever the GPU can (decode's rule); a batched pass where its moved line at the layer's mean
    // routes off the tier is no longer than the device line plus the mean copy
    bool in_place = t.in_place;
    if (in_place && !one && !t.in_place_only) {
        double routes = 0.0, copy = 0.0, count = 0.0;
        for (size_t i = 0; i < n; ++i) {
            const auto & x = items[i];
            if (x.resident || x.routes <= 0) { continue; }
            routes += x.routes; count += 1.0;
            copy += x.mb * (x.locked && t.locked_us_per_mb > 0.0 ? t.locked_us_per_mb : t.pageable_us_per_mb);
        }
        if (count > 0.0) {
            in_place = engine.of_moved().expert(routes / count) <= engine.device.expert(routes / count) + copy / count;
        }
    }
    out.in_place = in_place;
    // a moved expert's line: in place (or the vector engine's host jobs) the moved line, else the device line after
    // its copy
    const kimi_place_line & moved_line = one || in_place ? engine.of_moved() : engine.device;
    const double r = t.drive_us_per_mb;
    const double d0 = r > 0.0 ? t.drive_fixed_us : 0.0;
    // landings: the earlier reads still landing first (whole: their rest is read at once), then the layer's reads in
    // item order (one token: role-major)
    auto pending = [&](const kimi_place_item & x) { return !x.resident && !x.read && x.pending_mb > 0.0; };
    std::vector<double> early_ready(n, 0.0), ready(n, 0.0);
    double read_mb = 0.0, pending_mb = 0.0, first_mb = 0.0, reads_end = 0.0;
    size_t reads = 0;
    for (size_t i = 0; i < n; ++i) {
        if (pending(items[i])) { pending_mb += items[i].pending_mb; }
        if (items[i].resident || !items[i].read) { continue; }
        ++reads; read_mb += items[i].mb;
        if (first_mb == 0.0) { first_mb = items[i].mb; }
    }
    if (r > 0.0 && (reads || pending_mb > 0.0)) {
        double mb = 0.0;
        for (size_t i = 0; i < n; ++i) {
            if (pending(items[i])) { mb += items[i].pending_mb; early_ready[i] = ready[i] = d0 + mb * r; }
        }
        if (one) {
            for (size_t i = 0; i < n; ++i) {
                if (!items[i].resident && items[i].read) { mb += items[i].mb_early; early_ready[i] = d0 + mb * r; }
            }
        }
        for (size_t i = 0; i < n; ++i) {
            if (items[i].resident || !items[i].read) { continue; }
            mb += one ? items[i].mb - items[i].mb_early : items[i].mb;
            ready[i] = d0 + mb * r;
            if (!one) { early_ready[i] = ready[i]; }
        }
        reads_end = d0 + mb * r;
    }
    // a move's terms: its computation on the GPU lane, its copy on the link (the pipeline's first stage)
    auto compute = [&](const kimi_place_item & x) { return moved_line.expert(x.routes); };
    auto copied = [&](const kimi_place_item & x) { return in_place ? 0.0 : x.mb * copy_rate(x); };
    // the candidates: movable, priced (a page-locked copy at a measured rate; a pageable one in a longer pass at a
    // known rate; none in place), by routes (the most CPU time first), then by landing (host memory first), then in
    // order; a move that needs a device slot only while one is free, a read only while the limit allows
    std::vector<size_t> order;
    for (size_t i = 0; i < n; ++i) {
        const auto & x = items[i];
        if (x.resident || !x.movable || x.routes <= 0) { continue; }
        const bool priced = in_place || (x.locked ? t.locked_us_per_mb > 0.0 : !one && t.pageable_us_per_mb > 0.0);
        if (priced) { order.push_back(i); }
    }
    std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) {
        if (items[a].routes != items[b].routes) { return items[a].routes > items[b].routes; }
        return ready[a] < ready[b];
    });
    {
        size_t slots = limits.slots, moved_reads = 0;
        std::vector<size_t> kept;
        for (size_t i : order) {
            const auto & x = items[i];
            if (x.needs_slot && !in_place) { if (!slots) { continue; } --slots; }
            if (x.read) { if (moved_reads >= limits.reads) { continue; } ++moved_reads; }
            kept.push_back(i);
        }
        order.swap(kept);
    }
    out.candidates = order.size();
    bool residents = false;
    for (size_t i = 0; i < n; ++i) { residents = residents || (items[i].resident && items[i].routes > 0); }
    const double overhead = one ? 0.0 : t.overhead_fixed_us + double(ntokens) * t.token_us;
    // a route's rows cross a link only to a GPU that does not read host memory (one token: in decode's lines)
    const double route_us = one || t.in_place ? 0.0 : t.route_us;
    size_t k = 0;
    if (reads <= ring) {
        // every read in flight at once: the list schedule over the landings
        out.ring = true;
        std::vector<kimi_place_expert> e(n);
        for (size_t i = 0; i < n; ++i) {
            const auto & x = items[i];
            auto & y = e[i];
            if (x.routes <= 0) { continue; }   // no work in this layer (decode's cache-only requests)
            if (x.resident) { y.resident = true; y.gpu = engine.device.expert(x.routes); continue; }
            y.cpu = t.cpu(ntokens, x.routes);
            y.early = one ? t.cpu_early * y.cpu : 0.0;
            y.early_ready = early_ready[i]; y.ready = ready[i];
            y.gpu = compute(x);
            y.link = copied(x);
        }
        kimi_place_lanes lanes = {};
        lanes.resident_call = engine.device.call_us;
        lanes.host_call = moved_line.call_us;
        lanes.gpu_overhead = overhead;
        lanes.reads_end = reads_end;
        lanes.link_free = t.link_busy_us;
        // the CPU's DRAM shared with the moves' copies: decode's measured ratio at one token; a longer pass's CPU line
        // is calibrated in its own passes, at their moves
        lanes.cpu_moved = one ? std::max(1.0, t.cpu_moved) : 1.0;
        std::vector<uint8_t> none(n, 0);
        double cpu_end = 0.0, gpu_end = 0.0;
        out.time_none = kimi_place_ring_time(lanes, e.data(), n, none.data());
        k = kimi_place_ring_best_prefix(lanes, e.data(), n, order.data(), order.size(), &out.time);
        for (size_t j = 0; j < k; ++j) { out.gpu[order[j]] = 1; }
        kimi_place_ring_time(lanes, e.data(), n, out.gpu.data(), &cpu_end, &gpu_end);
        out.cpu = cpu_end; out.gpu_lane = gpu_end; out.drive = reads_end;
        for (size_t j = 0; j < k; ++j) { out.link += e[order[j]].link; }
        out.overhead = k || residents ? overhead : 0.0;
    } else {
        // the reads stream through the ring: the coupled form, a move's copy on the link before its computation
        kimi_prefill_layer l = {};
        for (size_t i = 0; i < n; ++i) {
            const auto & x = items[i];
            if (x.routes <= 0) { continue; }
            if (x.resident) { l.gpu += engine.device.expert(x.routes); l.link += x.routes * route_us; continue; }
            l.cpu += t.cpu(ntokens, x.routes);
        }
        if (residents) { l.gpu += engine.device.call_us; l.gpu_work = true; }
        if (r > 0.0 && read_mb > 0.0) { l.drive = d0 + (read_mb + pending_mb) * r; l.fill = first_mb * r; }
        l.ring = double(std::max<size_t>(1, ring));
        l.gpu_overhead = overhead + (one ? moved_line.call_us : 0.0);
        std::vector<kimi_prefill_move> moves;
        for (size_t i : order) {
            const auto & x = items[i];
            const double d = copied(x);
            moves.push_back({ t.cpu(ntokens, x.routes), compute(x), d + x.routes * route_us, d });
        }
        out.time_none = kimi_prefill_time(l, l.cpu, l.gpu, l.link, l.gpu_work);
        k = kimi_prefill_best_prefix(l, moves.data(), moves.size(), &out.time);
        std::vector<double> cpu, gpu, link;
        kimi_prefill_prefix_lanes(l, moves.data(), moves.size(), cpu, gpu, link);
        for (size_t j = 0; j < k; ++j) { out.gpu[order[j]] = 1; }
        out.cpu = cpu[k]; out.gpu_lane = gpu[k]; out.link = link[k];
        double sum = l.gpu;
        for (size_t j = 0; j < k; ++j) { sum += moves[j].gpu; }
        out.fill = gpu[k] - sum;   // the pipeline's waits for copies
        out.drive = l.drive;
        out.overhead = k || l.gpu_work ? l.gpu_overhead : 0.0;
    }
    out.moved.assign(order.begin(), order.begin() + k);
    // each lane's predicted start of its experts: the GPU's moves in their order, a copy on the link before its
    // computation (the pipeline), the CPU's experts in item order
    double link = t.link_busy_us, gpu = 0.0, cpu = 0.0;
    for (size_t i : out.moved) {
        const auto & x = items[i];
        const double d = copied(x);
        out.start[i] = d > 0.0 ? link : gpu;
        if (d > 0.0) { link += d; gpu = std::max(gpu, link); }
        gpu += compute(x);
    }
    for (size_t i = 0; i < n; ++i) {
        if (items[i].resident || out.gpu[i] || items[i].routes <= 0) { continue; }
        out.start[i] = cpu;
        cpu += t.cpu(ntokens, items[i].routes);
    }
    return out;
}
