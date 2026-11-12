// Numeric policy contract: no backend, model, file, cache or worker is needed.
#include "../../ggml/src/ggml-ssd/place/runtime-policy.h"
#include <cstdio>
#include <cstdlib>
#include <cmath>

using namespace ssd::place;
static void check(bool ok, const char * message) {
    if (!ok) { std::fprintf(stderr, "runtime policy: %s\n", message); std::exit(1); }
}
static bool near(double a, double b) { return std::fabs(a - b) <= 1e-12 * std::max(1.0, std::max(std::fabs(a), std::fabs(b))); }

static void learning_contract() {
    decode_learning s;
    observe_expert(s, {100, 40, true, false});
    check(s.cpu_expert_us == 100 && s.early == .4 && s.cpu_alone_us == 100, "first expert supplies unscaled priors");
    observe_expert(s, {200, 100, true, true});
    check(s.cpu_expert_us == 110 && near(s.early, .41) && s.cpu_shared_us == 200 && s.cpu_moved() == 2,
        "per-expert updates preserve EMA and shared-DRAM slowdown");
    observe_read_rate(s, 1000, 10); observe_read_rate(s, 3000, 10);
    check(s.read_bytes_per_us == 120, "I/O observation updates rate after completion");
    observe_read_rate(s, 1, 0);
    check(s.read_bytes_per_us == 120, "zero-duration reads cannot train rate");
    check(observe_layer(s, 1000000, 10, 100, 80) == 0, "fixed lane overhead is nonnegative");
    for (int i = 1; i < 70; ++i) { observe_layer(s, uint64_t(i + 1) * 1000000, i + 10, 100, 110); }
    check(s.reads.size() == 64 && s.reads.front().first == 7 && s.reads.back().first == 70, "drive fit retains the last64 reading layers");
    const size_t before = s.reads.size(); observe_layer(s, 0, 99, 100, 110);
    check(s.reads.size() == before, "cache-only layers cannot evict read observations");

    // 0.7 CPUs of other work over 10 s, the counters within 0.5 CPU-s: one CPU stays with it.
    check(decode_reserve(0, 7e9, 10e9, 0.5e9) == 1, "measured other work rounds up to whole CPUs");
    check(decode_reserve(0, 0.2e9, 10e9, 0.1e9) == 1 && decode_reserve(3, 0, 10e9, 0) == 0, "any work keeps a CPU, none keeps none");
    // A band across a whole CPU (0.95..1.05) keeps the current choice; a narrower one decides.
    check(decode_reserve(1, 10e9, 10e9, 0.5e9) == 1 && decode_reserve(2, 10e9, 10e9, 0.5e9) == 2, "a straddling band keeps the choice");
    check(decode_reserve(1, 15e9, 10e9, 0.5e9) == 2, "a band past a whole CPU moves the choice");
    check(decode_reserve(2, 1e9, 0, 0) == 2, "no wall time measured keeps the choice");
    check(decode_participants(18, 18, 1) == 17 && decode_participants(16, 61, 2) == 16 && decode_participants(4, 4, 9) == 1,
          "decode keeps its pool within the CPUs the reserve leaves, and at least one thread");
}

static void prefetch_contract() {
    auto p = plan_prefetch(4, 5, 5, 2, false);
    check(p.hinted && p.layer == 5 && p.ranks == 2 && p.open_gate, "prefetch preserves prefix rank order and idle gate");
    check(plan_prefetch(4, 5, 5, 0, true).ranks == 0, "depth0 records hints but schedules no reads");
    check(!plan_prefetch(4, 6, 5, 2, true).hinted, "mismatched hints do not schedule a later layer");
    auto rank = plan_prefetch_rank(7, false, true, 0, 2);
    check(!rank.novel && rank.request && !rank.skipped, "cached candidate still reaches executor reservation and skip feedback");
    rank = plan_prefetch_rank(7, true, false, 0, 2);
    check(!rank.novel && !rank.request && rank.skipped, "GPU-held rank skips source resolution");
    rank = plan_prefetch_rank(7, false, false, 3, 2);
    check(rank.novel && !rank.request && !rank.skipped, "ranks past depth remain counted without requests");
    check(!plan_prefetch_rank(-1, false, false, 0, 2).request, "invalid expert never reaches resolver");
    // A read pays its use probability where the window outlasts it, p - r where the window ends with a share r of it
    // left, nothing where it ends first; no window seen: p. The rule of succession gives one half without evidence.
    const std::vector<std::pair<double, double>> long_windows = {{5000, 1}}, short_windows = {{500, 3}, {5000, 1}};
    check(near(prefetch_read_value(.6, 0, 1000, long_windows), .6), "a read the window covers is worth its use");
    check(near(prefetch_read_value(.6, 0, 1000, {{500, 1}}), .1), "a read the window ends halfway through is worth p - 1/2");
    check(near(prefetch_read_value(.6, 1000, 1000, {{500, 1}}), 0), "a read the window ends before is dropped at no cost");
    check(near(prefetch_read_value(.4, 0, 1000, short_windows), (3 * -.1 + .4) / 4), "the windows are weighed by how often they occur");
    check(near(prefetch_read_value(.7, 0, 1000, {}), .7), "no window seen: the read is worth its use");
    check(near(prefetch_use_probability(0, 0), .5) && near(prefetch_use_probability(9, 10), 10.0 / 12),
          "use probability by the rule of succession");
}

static void placement_contract() {
    cost_lines prior, cost;
    decode_learning decode; prefill_learning prefill;
    lane_measurement measured = {2, 3, 4, 5, 1, 100, 200, 7, true};
    lane_geometry lane = {4096, 2048, false, false};
    auto t = make_table(cost, prior, decode, prefill, measured, lane, 1, 4096);
    check(t.cpu_one_us == 291 && t.vector.device.call_us == 2 && t.vector.moved.expert_us == 5,
        "unknown CPU expert uses configured prior with measured vector lanes");
    check(t.in_place && t.route_us == 0 && near(t.token_us, .6144), "decode has no route transfers and retains row geometry prior");
    t = make_table(cost, prior, decode, prefill, measured, lane, 32, 4096);
    check(!t.in_place && near(t.route_us, .4096), "non-batched capability cannot promise grouped in-place reads");
    lane.batched = lane.in_place = true;
    check(make_table(cost, prior, decode, prefill, measured, lane, 32, 0).in_place, "batched capability enables grouped in-place evaluation");

    t.cpu_one_us = 100; t.vector.device = {0,1,0}; t.vector.moved = {0,1,0};
    t.link_busy_us = 0; t.in_place = false; t.locked_us_per_mb = 1;
    std::vector<kimi_place_item> items(3);
    for (auto & x : items) { x.routes = 1; x.mb = .01; x.locked = x.movable = x.needs_slot = true; }
    std::vector<double> pending(3, 0); pending[1] = .001;
    decode_plan plan; describe_decode(plan, items, pending);
    check(plan.ram == 3 && plan.pending == 1 && plan.locked == 3, "pending prefetch is explicit workload input");
    kimi_place_limits limits; limits.slots = 1;
    auto placed = plan_decode(plan, t, 4, items, limits);
    check(plan.valid && plan.moves == 1 && placed.moved.size() == 1 && plan.gpu[placed.moved[0]], "device-slot limit constrains actual GPU plan");
    items[0].locked = items[1].locked = items[2].locked = false;
    const auto locks = plan_locks(t, 4, items, limits, 300);
    check(locks.considered && !locks.requests.empty() && locks.saved_us > 0, "page-lock plan contains request indices and saved time only");
    const auto use = miss_costs(1, 100, 600, true, 80), hidden = miss_costs(4, 100, 200, false, 80);
    check(near(use.ram_us, 80) && near(use.drive_us, 600), "a use from RAM costs the in-place compute, one from the drive its share of the reads");
    check(near(hidden.ram_us, 100) && near(hidden.drive_us, 100), "a use from the drive costs no less than its compute");

    std::vector<prefill_expert> experts = {{9,false,true,true,1024},{2,false,true,true,1024},{1,false,false,false,1024}};
    std::vector<int> routes = {4,4,4}; std::vector<kimi_place_item> empty;
    prefill_input input{experts,routes,empty,t,32,1,2}; uint64_t candidates = 0;
    const auto counted = count_prefill(input,candidates);
    check(candidates == 1 && counted.ram_candidates == 2 && counted.ssd_candidates == 1 && counted.ssd_limit == 1,
        "prefill candidates: leased RAM entries and drive reads, the reads limited by the staging");
    bool in_place = false; const auto selected = plan_prefill(input,counted,in_place);
    check(selected.ssd_limit == 1 && selected.chosen.empty(), "a plan without placeable items moves nothing");
}

static void calibration_contract() {
    prefill_learning s; s.window = 2; cost_lines prior, cost;
    s.cpu_fit.add(1,100); s.cpu_fit.add(2,200);
    prefill_feedback f = {}; f.gpu.add(1,20); f.gpu.add(2,30); f.ntokens = 32;
    f.read_bytes = 1000000; f.read_us = 100; f.engine_overhead_us = 10; f.output_add_us = 3; f.transient_wait_us = 7;
    auto fit = observe_prefill_lines(s,cost,prior,f);
    check(fit.gpu_lane_us == 57 && fit.gpu_overhead_us == 13 && s.samples.size() == 1 && s.calibrations == 0,
        "event overhead excludes transient waits and defers publication count until overhead fit");
    observe_prefill_overhead(s,cost,fit.overhead,.1);
    check(s.calibrations == 1 && near(cost.gpu_overhead_fixed_us,9.8) && cost.gpu_token_us == .1, "one row count keeps transfer prior and fits fixed part");
    f.ntokens = 64; f.engine_overhead_us = 20; fit = observe_prefill_lines(s,cost,prior,f); observe_prefill_overhead(s,cost,fit.overhead,.1);
    f.ntokens = 128; f.engine_overhead_us = 40; fit = observe_prefill_lines(s,cost,prior,f); observe_prefill_overhead(s,cost,fit.overhead,.1);
    check(s.samples.size() == 2 && s.next == 1 && fit.overhead.min_tokens == 64 && fit.overhead.max_tokens == 128,
        "calibration expiry is the existing per-pass circular window");
    f.gpu = {}; f.engine_overhead_us = 0; f.upload_us = 4; f.ssd_upload_us = 6; f.total_worker_us = 50; f.read_wait_us = 10;
    fit = observe_prefill_lines(s,cost,prior,f);
    check(fit.gpu_lane_us == 10 && fit.gpu_overhead_us == 0, "CPU-only sample cannot train GPU overhead from worker bookkeeping");

    // The drive's line follows the newest layer with reads: two layers at 1 GB/s, then one at 10 GB/s and one without
    // reads, price the next layer's reads at 10 GB/s.
    prefill_learning d; d.window = 4; cost_lines none;
    prefill_feedback r = {}; r.ntokens = 32;
    for (const double us : {1e6, 1e6, 1e5, 0.0}) {
        r.read_bytes = us > 0 ? 1000000000 : 0; r.read_us = us; (void) observe_prefill_lines(d,none,prior,r);
    }
    decode_learning idle;
    lane_measurement measured = {2, 3, 4, 5, 1, 100, 200, 7, true};
    lane_geometry lane = {4096, 2048, true, true};
    const auto t = make_table(none, prior, idle, d, measured, lane, 32, 0);
    check(d.samples.size() == 4 && near(t.drive_us_per_mb, 100) && t.drive_fixed_us == 0,
        "the drive's line follows the newest grouped-prefill layer with reads");
}
int main() {
    learning_contract(); prefetch_contract(); placement_contract(); calibration_contract();
    std::puts("RUNTIME_POLICY_PASS learning placement prefetch calibration");
}
