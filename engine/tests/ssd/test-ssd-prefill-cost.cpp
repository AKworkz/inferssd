// The MXFP4 grouped-prefill cost model alone (ggml/src/ggml-ssd/place/layer-plan.h): the per-expert fit, the coupling
// of the drive with the lanes, and the choice of the experts that move to the GPU.
#include "../../ggml/src/ggml-ssd/place/layer-plan.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

static int failures = 0;
static void check(bool ok, const char * what) {
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", what); ++failures; }
}
static bool near(double a, double b, double rel) { return std::fabs(a - b) <= rel * std::max(std::fabs(a), std::fabs(b)); }

static void test_fit() {
    // us = 300 + 250 * routes with +-100 us deterministic noise, routes 1..400: the fit recovers both terms.
    kimi_prefill_fit s;
    for (int routes = 1; routes <= 400; ++routes) { s.add(routes, 300.0 + 250.0 * routes + 100.0 * std::sin(routes * 12.9898)); }
    double fixed = -1, slope = -1;
    kimi_prefill_fit_line(s, 204.0, 87.0, fixed, slope);
    check(near(slope, 250.0, 0.002) && std::fabs(fixed - 300.0) < 30.0, "fit recovers fixed and per-route terms");
    std::printf("fit: fixed=%.1f slope=%.2f (300, 250)\n", fixed, slope);

    // One route count only (a window of 1-token passes): solvable, the line passes through the data there, and keeps
    // the prior's shape: both terms scaled by the data over the prior there.
    kimi_prefill_fit one;
    for (int i = 0; i < 240; ++i) { one.add(1.0, 1000.0 + (i % 7)); }
    kimi_prefill_fit_line(one, 204.0, 87.0, fixed, slope);
    const double scale = (one.t / one.n) / (204.0 + 87.0);
    check(std::isfinite(fixed) && std::isfinite(slope) && fixed >= 0 && slope >= 0, "one route count stays solvable");
    check(near(fixed + slope, one.t / one.n, 1e-3), "one route count: the line passes through the data");
    check(near(slope, 87.0 * scale, 1e-3) && near(fixed, 204.0 * scale, 1e-3), "one route count: the prior's shape");
    std::printf("fit one route count: fixed=%.1f slope=%.2f\n", fixed, slope);

    // The drive's line per layer, read time against MB: 40 warm-up layers of ~190 MB and big layers of ~4.6 GB, a 12 ms
    // queue fill and drain per layer at 7.2 GB/s. The ratio of the sums would give 6.7 GB/s; the line keeps the rate.
    kimi_prefill_fit drive;
    for (int i = 0; i < 40; ++i) { const double mb = 150.0 + 2.0 * i; drive.add(mb, 12000.0 + mb * 1e3 / 7.2); }
    for (int i = 0; i < 10; ++i) { const double mb = 4400.0 + 40.0 * i; drive.add(mb, 12000.0 + mb * 1e3 / 7.2); }
    kimi_prefill_fit_line(drive, 0.0, 1.0, fixed, slope);
    check(near(1e3 / slope, 7.2, 5e-3) && near(fixed, 12000.0, 0.05), "the drive's line separates its fixed part");
    std::printf("fit drive: fixed=%.0f us rate=%.3f GB/s (12000, 7.2)\n", fixed, 1e3 / slope);

    // A negative intercept is clamped: the line through the origin.
    kimi_prefill_fit neg;
    for (int routes = 100; routes <= 200; ++routes) { neg.add(routes, 10.0 * routes - 500.0); }
    kimi_prefill_fit_line(neg, 0.0, 10.0, fixed, slope);
    check(fixed == 0.0 && slope > 0.0 && slope < 10.0, "negative intercept clamps to a line through the origin");

    // No samples: the terms are left as they are.
    fixed = 7; slope = 9;
    kimi_prefill_fit_line(kimi_prefill_fit(), 204.0, 87.0, fixed, slope);
    check(fixed == 7 && slope == 9, "no samples keep the terms");
}

static void test_overhead_fit() {
    // Recorded RTX 5090 calibration: 2 tokens, 279.656 us outside the GPU experts; MXFP4's output row is 5120 floats,
    // and its measured link rate was 33.617 GB/s. Inputs already have the lane's per-route charge. One token shape
    // does not identify the fixed and row terms: retain this geometry slope, not 279.656 / 2 us per future token.
    const double row_prior = (5120.0 * sizeof(float)) / 1e6 * (1e3 / 33.617);
    kimi_prefill_overhead_fit one;
    one.add(2, 279.656);
    double fixed = 0, row = -1;
    one.fit(row_prior, fixed, row);
    check(near(row, row_prior, 1e-12), "one overhead shape retains the transfer geometry slope");
    check(near(fixed + 2 * row, 279.656, 1e-12), "same-shape overhead matches the observation");
    const double full = fixed + 512 * row;
    // The recorded 512 layers' no-GPU predictions exceed their drive bound by at least 1828.5 us. The old
    // 71591.936 us charge ruled out every prefix even with free GPU computation; the identified fixed component
    // is paid once. This replay removes that arithmetic veto; it does not predict the unmeasured GPU's speed.
    check(full < 1828.5 && near((279.656 / 2) * 512, 71591.936, 1e-12), "2-to-512 replay does not multiply fixed overhead");
    const double initial_fixed = fixed, initial_row = row;
    kimi_prefill_overhead_fit().fit(row_prior, fixed, row);
    check(fixed == initial_fixed && row == initial_row, "empty overhead window retains its last fit");
    one.add(2, 281.656);
    one.fit(row_prior, fixed, row);
    check(near(fixed + 2 * row, 280.656, 1e-12) && row == row_prior, "same-shape repeats estimate the fixed residual");

    // A synthetic second shape makes the two coefficients identifiable. No prior scaling should distort the
    // exact affine data once the window contains that information.
    kimi_prefill_overhead_fit two;
    two.add(2, 280.0); two.add(512, 1300.0);
    two.fit(row_prior, fixed, row);
    check(near(fixed, 276.0, 1e-12) && near(row, 2.0, 1e-12), "second overhead shape identifies fixed and row costs");
    check(near(fixed + 128 * row, 532.0, 1e-12), "identified overhead interpolates a third shape");
    kimi_prefill_overhead_fit negative_fixed;
    negative_fixed.add(2, 1.0); negative_fixed.add(512, 1300.0);
    negative_fixed.fit(row_prior, fixed, row);
    check(fixed == 0.0 && row > 0.0 && std::isfinite(row), "negative overhead intercept uses the nonnegative boundary");
    kimi_prefill_overhead_fit negative_row;
    negative_row.add(2, 1000.0); negative_row.add(512, 100.0);
    negative_row.fit(row_prior, fixed, row);
    check(near(fixed, 550.0, 1e-12) && row == 0.0, "negative overhead slope uses the nonnegative boundary");
    kimi_prefill_overhead_fit below_prior;
    below_prior.add(2, 0.5);
    below_prior.fit(row_prior, fixed, row);
    check(fixed == 0.0 && row == 0.25, "one shape bounds an infeasible row prior without a negative intercept");

    kimi_place_table t;
    t.overhead_fixed_us = initial_fixed; t.token_us = initial_row;
    t.batched.device.expert_us = 10; t.vector.device.expert_us = 10;
    t.cpu_one_us = t.cpu_fixed_us = 1000;
    std::vector<kimi_place_item> items(2);
    for (auto & x : items) { x.routes = 512; x.resident = true; }
    const auto batched = kimi_place_layer(t, 512, 8, items.data(), items.size());
    check(near(batched.overhead, full, 1e-12) && near(batched.time, 20 + full, 1e-12), "fixed GPU overhead is once per layer, not once per expert");
    for (auto & x : items) { x.routes = 1; }
    const auto decode = kimi_place_layer(t, 1, 8, items.data(), items.size());
    check(decode.overhead == 0.0 && decode.time == 20, "the batched overhead fit does not change decode pricing");
    for (auto & x : items) { x.routes = 512; x.resident = false; }
    check(kimi_place_layer(t, 512, 8, items.data(), items.size()).overhead == 0.0, "no GPU work pays no GPU overhead");
    std::printf("overhead replay: 2 tokens 279.656 us, fixed %.3f us, row %.6f us, 512 tokens %.3f us (old 71591.936)\n",
                initial_fixed, initial_row, full);
}

static void test_coupled() {
    check(kimi_prefill_coupled(1000.0, 0.0, 8) == 1000.0, "coupled with an idle side is the busy side");
    check(near(kimi_prefill_coupled(1000.0, 1000.0, 8), 1125.0, 1e-12), "equal sides: max * (n + 1) / n");
    check(near(kimi_prefill_coupled(1000.0, 1000.0 * (1 - 1e-10), 8), 1125.0, 1e-6), "continuous at equal sides");
    check(kimi_prefill_coupled(1000.0, 300.0, 8) == kimi_prefill_coupled(300.0, 1000.0, 8), "symmetric");
    check(near(kimi_prefill_coupled(1000.0, 100.0, 8), 1000.0, 1e-7), "a far shorter side costs nothing");
    double previous = 0;
    bool monotone = true;
    for (int i = 0; i <= 300; ++i) {
        const double t = kimi_prefill_coupled(1000.0, 10.0 * i, 8);
        monotone &= t >= previous && t >= std::max(1000.0, 10.0 * i);
        previous = t;
    }
    check(monotone, "coupled rises with the shorter side and stays above the max");
    // The measured share on an RTX 5060 Ti (8-read ring): 0.136 with the lanes at the drive time, 0.002 with them
    // at half of it.
    std::printf("coupled n=8: equal %.4f, half %.4f, 0.95 %.4f\n", kimi_prefill_coupled(1, 1, 8) - 1,
        kimi_prefill_coupled(1, 0.5, 8) - 1, kimi_prefill_coupled(1, 0.95, 8) - 1);
}

// A layer of 330 cold experts like DeepSeek's at 4104 tokens on an RTX 5060 Ti: routes 1..~500, CPU 520 + 297 us
// per route, GPU 60 + 25 us per route plus an 18.8 MB upload at 5.4 GB/s, an 8-read ring.
static std::vector<kimi_prefill_move> moves_for(kimi_prefill_layer & layer, int experts) {
    std::vector<int> routes;
    for (int i = 0; i < experts; ++i) { routes.push_back(1 + int(500.0 * std::pow(double(experts - i) / experts, 3.0))); }
    std::vector<kimi_prefill_move> moves;
    layer = {};
    for (int r : routes) {
        const double cpu = 520.0 + 297.0 * r;
        layer.cpu += cpu;
        moves.push_back({ cpu, 60.0 + 25.0 * r + 18.8e6 / 5400.0, (18.8e6 + 4400.0 * r) / 5400.0 });
    }
    layer.ring = 8;
    layer.gpu_overhead = 4104 * 7.0;
    layer.fill = 18.8e6 / 6800.0;
    return moves;
}

static void test_select() {
    kimi_prefill_layer layer;
    auto moves = moves_for(layer, 330);
    const double cpu_all = layer.cpu;

    // CPU-bound (a fast drive): the lanes end balanced, within one move of each other.
    layer.drive = 0.2 * cpu_all;
    double t = 0;
    size_t k = kimi_prefill_best_prefix(layer, moves.data(), moves.size(), &t);
    double cpu = layer.cpu, gpu = 0;
    for (size_t i = 0; i < k; ++i) { cpu -= moves[i].cpu; gpu += moves[i].gpu; }
    check(k > 0 && k < moves.size() && std::fabs(cpu - gpu) < 2 * moves[k - 1].gpu + moves[k - 1].cpu + 2 * layer.fill,
        "a CPU-bound layer balances the lanes");
    std::printf("select cpu-bound: %zu of %zu, cpu %.0f gpu %.0f us\n", k, moves.size(), cpu, gpu);

    // A drive far longer than the CPU lane (a short batch): nothing moves, also when resident experts have already
    // paid the GPU overhead, where the lanes still gain a little: less than one read. A two-read ring couples more
    // (a share of r^2 instead of r^8), and nothing moves there once the drive is a thousand times the lanes.
    layer.drive = 20.0 * cpu_all;
    k = kimi_prefill_best_prefix(layer, moves.data(), moves.size(), &t);
    check(k == 0, "a drive-bound layer keeps its experts on the CPU");
    kimi_prefill_layer resident = layer;
    resident.gpu_work = true;
    check(kimi_prefill_best_prefix(resident, moves.data(), moves.size()) == 0,
        "a drive-bound layer with resident GPU work keeps its experts on the CPU");
    resident.ring = 2; resident.drive = 1000.0 * cpu_all; resident.fill = resident.drive / moves.size();
    check(kimi_prefill_best_prefix(resident, moves.data(), moves.size()) == 0,
        "a drive-bound layer with a two-read ring keeps its experts on the CPU");

    // No discontinuity: as the drive grows through the CPU lane's time, the number moved and the predicted time
    // change gradually.
    size_t previous_k = moves.size();
    double previous_t = 0;
    bool gradual = true;
    for (int i = 0; i <= 400; ++i) {
        layer.drive = cpu_all * (0.2 + 1.6 * i / 400.0);
        k = kimi_prefill_best_prefix(layer, moves.data(), moves.size(), &t);
        if (i) {
            gradual &= t >= previous_t - 1e-6 && t - previous_t <= 0.01 * cpu_all;
            gradual &= previous_k >= k ? previous_k - k <= moves.size() / 20 : k - previous_k <= moves.size() / 20;
        }
        previous_k = k; previous_t = t;
    }
    check(gradual, "the selection changes gradually with the drive time");
    layer.drive = 1.0001 * cpu_all;
    const size_t above = kimi_prefill_best_prefix(layer, moves.data(), moves.size());
    layer.drive = 0.9999 * cpu_all;
    const size_t below = kimi_prefill_best_prefix(layer, moves.data(), moves.size());
    check(above > 0 && below > 0 && (above > below ? above - below : below - above) <= 2,
        "the same choice just above and just below the CPU lane's time");
    std::printf("select at the drive: %zu (drive just above the CPU lane) %zu (just below)\n", above, below);

    // The GPU overhead decides a layer with little to gain.
    kimi_prefill_layer small = {};
    small.cpu = 6 * 1000.0; small.drive = 6 * 2800.0; small.fill = 2800.0; small.ring = 8; small.gpu_overhead = 7.0;
    std::vector<kimi_prefill_move> few(6, kimi_prefill_move{ 1000.0, 3600.0, 3500.0 });
    check(kimi_prefill_best_prefix(small, few.data(), few.size()) == 0, "a 1-token layer stays on the CPU");
}

// The streamed choice before it went through kimi_place_best_prefix (dev 8251825), as the reference of test_same_choice.
static size_t reference_best_prefix(const kimi_prefill_layer & l, const kimi_prefill_move * moves, size_t count,
        double * time) {
    double cpu = l.cpu, gpu = l.gpu, link = l.link;
    double lowest = kimi_prefill_time(l, cpu, gpu, link, l.gpu_work);
    for (size_t i = 0; i < count; ++i) {
        cpu -= moves[i].cpu; gpu += moves[i].gpu; link += moves[i].link;
        lowest = std::min(lowest, kimi_prefill_time(l, cpu, gpu, link, true));
    }
    cpu = l.cpu; gpu = l.gpu; link = l.link;
    double t = kimi_prefill_time(l, cpu, gpu, link, l.gpu_work);
    size_t chosen = 0;
    while (t > lowest + l.fill && chosen < count) {
        cpu -= moves[chosen].cpu; gpu += moves[chosen].gpu; link += moves[chosen].link;
        t = kimi_prefill_time(l, cpu, gpu, link, true);
        ++chosen;
    }
    *time = t;
    return chosen;
}

// A streamed layer chooses exactly what M8 chose: the same count and the same time, bit for bit, on random layers.
static void test_same_choice() {
    uint64_t state = 12345;
    auto unit = [&] { state = state * 6364136223846793005ull + 1442695040888963407ull; return double(state >> 11) / double(1ull << 53); };
    size_t same = 0, total = 0, moved = 0;
    for (int trial = 0; trial < 2000; ++trial) {
        kimi_prefill_layer layer;
        auto moves = moves_for(layer, 1 + int(unit() * 400));
        layer.drive = layer.cpu * (0.05 + 2.0 * unit());
        layer.gpu = unit() < 0.5 ? 0.0 : layer.cpu * 0.3 * unit();
        layer.gpu_work = layer.gpu > 0.0;
        layer.ring = 1 + int(unit() * 16);
        layer.fill = unit() < 0.2 ? 0.0 : layer.fill * unit() * 2.0;
        double t_old = 0, t_new = 0;
        const size_t k_old = reference_best_prefix(layer, moves.data(), moves.size(), &t_old);
        const size_t k_new = kimi_prefill_best_prefix(layer, moves.data(), moves.size(), &t_new);
        same += k_old == k_new && t_old == t_new;
        moved += k_new > 0;
        ++total;
    }
    check(same == total, "the streamed choice is M8's, bit for bit");
    std::printf("same choice: %zu of %zu random streamed layers (%zu move experts)\n", same, total, moved);
}

// The in-ring list schedule and its choice, each value derived from kimi_place_ring_time's rules (us).
static void test_ring() {
    // RTX 5090 decode terms: a call 150 + 20 an expert from device memory, a page-locked copy 355, the CPU 470 an
    // expert (320 of it up/gate). Three residents, two RAM hits, no read.
    //   no move:  GPU 150 + 3 * 20 = 210; CPU 470 + 470 = 940;              T = 940
    //   one:      copy 0..355; host call max(210, 355) + 150 + 20 = 525; CPU 470; T = 525
    //   two:      copies 0..355..710; call max(210, 710) + 150 + 40 = 900; CPU 0;  T = 900
    std::vector<kimi_place_expert> e(5);
    for (int i = 0; i < 3; ++i) { e[i].resident = true; e[i].gpu = 20; }
    for (int i = 3; i < 5; ++i) { e[i].cpu = 470; e[i].early = 320; e[i].gpu = 20; e[i].link = 355; }
    kimi_place_lanes lanes = {};
    lanes.resident_call = 150; lanes.host_call = 150;
    uint8_t none[5] = {}, one[5] = { 0, 0, 0, 1, 0 }, two[5] = { 0, 0, 0, 1, 1 };
    check(kimi_place_ring_time(lanes, e.data(), 5, none) == 940, "125: no move 940");
    check(kimi_place_ring_time(lanes, e.data(), 5, one) == 525, "125: one move 525");
    check(kimi_place_ring_time(lanes, e.data(), 5, two) == 900, "125: two moves 900");
    const size_t candidates[2] = { 3, 4 };
    double t = 0;
    check(kimi_place_ring_best_prefix(lanes, e.data(), 5, candidates, 2, &t) == 1 && t == 525, "125: one moves");
    // The link busy with an admission until 200: the copy starts there, 200 + 355 = 555, the call ends at 725.
    lanes.link_free = 200;
    check(kimi_place_ring_time(lanes, e.data(), 5, one) == 725, "125: a busy link delays the copy");
    lanes.link_free = 0;
    // The CPU 1.5x slower while copies share the DRAM: one move, CPU 705 > the GPU's 525: T = 705, still below 940.
    // At 2.5x one move leaves the CPU 1175, above 940; two leave it nothing (900): both move. Without a move the CPU
    // keeps its own time.
    lanes.cpu_moved = 1.5;
    check(kimi_place_ring_time(lanes, e.data(), 5, one) == 705 && kimi_place_ring_time(lanes, e.data(), 5, none) == 940,
        "125: the CPU slower beside a copy");
    lanes.cpu_moved = 2.5;
    check(kimi_place_ring_best_prefix(lanes, e.data(), 5, candidates, 2, &t) == 2 && t == 900,
        "125: a CPU much slower beside copies: both move");
    lanes.cpu_moved = 1;

    // RTX 5060 Ti terms: the copy 2700 (6.9 GB/s), the CPU 900. One move: call max(210, 2700) + 170 = 2870 > 1800.
    for (int i = 3; i < 5; ++i) { e[i].cpu = 900; e[i].early = 600; e[i].link = 2700; }
    check(kimi_place_ring_time(lanes, e.data(), 5, none) == 1800 && kimi_place_ring_time(lanes, e.data(), 5, one) == 2870,
        "233: 1800 and 2870");
    check(kimi_place_ring_best_prefix(lanes, e.data(), 5, candidates, 2, &t) == 0 && t == 1800, "233: nothing moves");

    // Unified memory: in place (no copy), 370 an expert in the host call; five residents at 190; the CPU 1200.
    //   no move: GPU 150 + 950 = 1100; CPU 2400 -> 2400. one: GPU 1100 + 150 + 370 = 1620; CPU 1200 -> 1620.
    //   two: GPU 1100 + 150 + 740 = 1990; CPU 0 -> 1990.
    std::vector<kimi_place_expert> u(7);
    for (int i = 0; i < 5; ++i) { u[i].resident = true; u[i].gpu = 190; }
    for (int i = 5; i < 7; ++i) { u[i].cpu = 1200; u[i].early = 800; u[i].gpu = 370; }
    const size_t uc[2] = { 5, 6 };
    uint8_t u1[7] = { 0, 0, 0, 0, 0, 1, 0 }, u2[7] = { 0, 0, 0, 0, 0, 1, 1 };
    check(kimi_place_ring_time(lanes, u.data(), 7, u1) == 1620 && kimi_place_ring_time(lanes, u.data(), 7, u2) == 1990,
        "unified: 1620 and 1990");
    check(kimi_place_ring_best_prefix(lanes, u.data(), 7, uc, 2, &t) == 1 && t == 1620, "unified: one moves");

    // Misses with the early up/gate: one RAM hit and two misses, gates landing at 1000 and 1500, downs at 2000 and
    // 2500 (role-major), residents as above (210). CPU: hit 0..320..470; miss 1 up/gate 1000..1320; miss 2 up/gate
    // 1500..1820; miss 1 down 2000..2150; miss 2 down 2500..2650. T = 2650. Moving the hit leaves the misses' times:
    // T = 2650: no gain, and a tie keeps the fewest moves.
    std::vector<kimi_place_expert> m(6);
    for (int i = 0; i < 3; ++i) { m[i].resident = true; m[i].gpu = 20; }
    m[3].cpu = 470; m[3].early = 320; m[3].gpu = 20; m[3].link = 355;
    m[4].cpu = 470; m[4].early = 320; m[4].early_ready = 1000; m[4].ready = 2000; m[4].gpu = 20; m[4].link = 355;
    m[5].cpu = 470; m[5].early = 320; m[5].early_ready = 1500; m[5].ready = 2500; m[5].gpu = 20; m[5].link = 355;
    lanes.reads_end = 2500;
    uint8_t m0[6] = {}, m1[6] = { 0, 0, 0, 1, 0, 0 };
    double cpu_end = 0, gpu_end = 0;
    check(kimi_place_ring_time(lanes, m.data(), 6, m0, &cpu_end, &gpu_end) == 2650 && cpu_end == 2650 && gpu_end == 210,
        "misses: 2650, the CPU's end");
    check(kimi_place_ring_time(lanes, m.data(), 6, m1) == 2650, "misses: moving the hit gains nothing");
    const size_t mc[3] = { 3, 4, 5 };
    check(kimi_place_ring_best_prefix(lanes, m.data(), 6, mc, 3, &t) == 0 && t == 2650, "misses: nothing moves");
    // A miss moved: its copy starts once it has landed (2000), 2355; the call ends at 2525. CPU: hit 470, miss 2
    // 1500..1820, 2500..2650. Still 2650.
    uint8_t m2[6] = { 0, 0, 0, 0, 1, 0 };
    check(kimi_place_ring_time(lanes, m.data(), 6, m2, &cpu_end, &gpu_end) == 2650 && gpu_end == 2525, "misses: a moved miss");

    // Grouped prefill, 2 tokens: six misses of 18.8 MB at 12.3 GB/s (1528 us each, 9170 in all), the CPU 380 an expert,
    // a transient 2800 (its upload in the worker) + 33. Every read lands before the CPU is busy: T = 9170 + 380 with no
    // move; a move adds the GPU's 2833 after its landing. Nothing moves.
    std::vector<kimi_place_expert> p(6);
    for (int i = 0; i < 6; ++i) { p[i].cpu = 380; p[i].ready = 1528.0 * (i + 1); p[i].gpu = 2833; }
    kimi_place_lanes pl = {};
    pl.reads_end = 9170;
    const size_t pc[6] = { 0, 1, 2, 3, 4, 5 };
    check(kimi_place_ring_best_prefix(pl, p.data(), 6, pc, 6, &t) == 0 && t == 1528.0 * 6 + 380, "2-token prefill: nothing moves");
    std::printf("ring: 125 one move 525 of 940; 233 none; unified one; misses none; 2-token prefill none\n");
}

// kimi_place_ring_time's CPU lane as first written (18bc4ec: a scan of every CPU expert at each pick), as the
// reference of test_ring_same: the heap schedule makes the same picks in the same order, so its end is the same, bit
// for bit.
static double reference_cpu_end(const kimi_place_expert * e, size_t n, const uint8_t * on_gpu) {
    std::vector<size_t> cpu_experts;
    for (size_t i = 0; i < n; ++i) { if (!e[i].resident && !(on_gpu && on_gpu[i]) && e[i].cpu > 0.0) { cpu_experts.push_back(i); } }
    std::vector<uint8_t> stage(cpu_experts.size(), 0);
    double cpu = 0.0;
    for (size_t left = cpu_experts.size(); left;) {
        size_t pick = cpu_experts.size();
        for (size_t j = 0; j < cpu_experts.size() && pick == cpu_experts.size(); ++j) {
            const auto & x = e[cpu_experts[j]];
            if (stage[j] == 0 && x.early > 0.0 && x.early_ready <= cpu) { pick = j; }
        }
        for (size_t j = 0; j < cpu_experts.size() && pick == cpu_experts.size(); ++j) {
            const auto & x = e[cpu_experts[j]];
            if ((stage[j] == 1 || (stage[j] == 0 && !(x.early > 0.0))) && x.ready <= cpu) { pick = j; }
        }
        if (pick == cpu_experts.size()) {
            double next = INFINITY;
            for (size_t j = 0; j < cpu_experts.size(); ++j) {
                const auto & x = e[cpu_experts[j]];
                if (stage[j] == 0 && x.early > 0.0) { next = std::min(next, x.early_ready); }
                else if (stage[j] < 2) { next = std::min(next, x.ready); }
            }
            cpu = next;
            continue;
        }
        const auto & x = e[cpu_experts[pick]];
        if (stage[pick] == 0 && x.early > 0.0) { cpu += std::min(x.early, x.cpu); stage[pick] = 1; continue; }
        cpu += stage[pick] == 1 ? std::max(0.0, x.cpu - x.early) : x.cpu;
        stage[pick] = 2;
        --left;
    }
    return cpu;
}

// Random layers, decode-like (early parts, misses landing in issue order) and prefill-like (no early part, up to 256
// experts): the CPU's end is the reference's, bit for bit, with and without moves.
static void test_ring_same() {
    uint64_t state = 777;
    auto unit = [&] { state = state * 6364136223846793005ull + 1442695040888963407ull; return double(state >> 11) / double(1ull << 53); };
    size_t same = 0, total = 0;
    for (int trial = 0; trial < 3000; ++trial) {
        const bool decode = trial % 2 == 0;
        const size_t n = decode ? 1 + size_t(unit() * 10) : 1 + size_t(unit() * 256);
        std::vector<kimi_place_expert> e(n);
        std::vector<uint8_t> on(n, 0);
        double landed = 0.0;
        for (size_t i = 0; i < n; ++i) {
            auto & x = e[i];
            if (unit() < 0.3) { x.resident = true; x.gpu = 20; continue; }
            x.cpu = 300 + 400 * unit();
            if (decode && unit() < 0.8) { x.early = x.cpu * (0.5 + 0.3 * unit()); }
            if (unit() < 0.5) {
                landed += 500 + 3000 * unit();
                x.ready = landed; x.early_ready = x.early > 0.0 ? landed * (0.4 + 0.5 * unit()) : 0.0;
            }
            // equal landing times, as several prefill misses of one read
            if (!decode && i && unit() < 0.2) { x.ready = e[i - 1].ready; }
            on[i] = unit() < 0.2;
        }
        double end = 0, end_moved = 0;
        kimi_place_ring_time(kimi_place_lanes{}, e.data(), n, nullptr, &end);
        kimi_place_ring_time(kimi_place_lanes{}, e.data(), n, on.data(), &end_moved);
        same += end == reference_cpu_end(e.data(), n, nullptr) && end_moved == reference_cpu_end(e.data(), n, on.data());
        ++total;
    }
    check(same == total, "the heap schedule's CPU end is the scan's, bit for bit");
    std::printf("ring same: %zu of %zu random layers\n", same, total);
}

// Layers without early parts (grouped prefill, up to 256 experts, few reads): kimi_place_ring_best_prefix folds the CPU
// lane by landing; the same choice as the list schedule (kimi_place_ring_time) for every prefix, and the same time
// within rounding (a different order of the same sums).
static void test_ring_prefix_fold() {
    uint64_t state = 4242;
    auto unit = [&] { state = state * 6364136223846793005ull + 1442695040888963407ull; return double(state >> 11) / double(1ull << 53); };
    size_t same = 0, total = 0, moved = 0;
    for (int trial = 0; trial < 1000; ++trial) {
        const size_t n = 2 + size_t(unit() * 255);
        std::vector<kimi_place_expert> e(n);
        std::vector<size_t> c;
        double landed = 0.0;
        for (size_t i = 0; i < n; ++i) {
            auto & x = e[i];
            if (unit() < 0.3) { x.resident = true; x.gpu = 20 + 200 * unit(); continue; }
            x.cpu = 200 + 2000 * unit(); x.gpu = 30 + 400 * unit();
            if (unit() < 0.05) { landed += 1500 * unit(); x.ready = landed; x.gpu += 2800; }
            if (unit() < 0.8) { c.push_back(i); }
        }
        kimi_place_lanes l = {};
        l.resident_call = 25; l.host_call = 25; l.gpu_overhead = 100 * unit(); l.reads_end = landed;
        std::vector<uint8_t> on(n, 0);
        const size_t k_ref = kimi_place_best_prefix(c.size(), 0.0, [&](size_t k) {
            std::fill(on.begin(), on.end(), 0);
            for (size_t j = 0; j < k; ++j) { on[c[j]] = 1; }
            return kimi_place_ring_time(l, e.data(), n, on.data());
        });
        double t_ref = 0, t = 0;
        std::fill(on.begin(), on.end(), 0);
        for (size_t j = 0; j < k_ref; ++j) { on[c[j]] = 1; }
        t_ref = kimi_place_ring_time(l, e.data(), n, on.data());
        const size_t k = kimi_place_ring_best_prefix(l, e.data(), n, c.data(), c.size(), &t);
        same += k == k_ref && near(t, t_ref, 1e-12);
        moved += k > 0;
        ++total;
    }
    check(same == total, "the folded CPU lane chooses as the list schedule");
    std::printf("ring fold: %zu of %zu random prefill layers (%zu move experts)\n", same, total, moved);
}


// The GPU lane as a two-stage pipeline (B3): copies on the link into two device slots, computations on the GPU.
static void test_pipeline() {
    // link-bound (c < d): every copy back to back, the last computation after the last copy
    const double c1[4] = { 300, 350, 320, 340 }, d1[4] = { 4400, 4400, 4400, 4400 };
    check(kimi_place_pipeline(c1, d1, 4) == 4 * 4400.0 + 340, "link-bound pipeline: the copies and the last computation");
    // compute-bound (c > d): the first copy, then every computation back to back
    const double c2[4] = { 4000, 4100, 3900, 4050 }, d2[4] = { 2700, 2700, 2700, 2700 };
    check(kimi_place_pipeline(c2, d2, 4) == 2700 + 4000 + 4100 + 3900 + 4050, "compute-bound pipeline");
    // one slot: no overlap, the sum of both stages
    check(kimi_place_pipeline(c2, d2, 4, 1) == 2700 * 4 + 4000 + 4100 + 3900 + 4050, "one slot: serial");
    // in place (no copy): the computations alone
    const double d0[4] = {};
    check(kimi_place_pipeline(c1, d0, 4) == 300 + 350 + 320 + 340, "in place: compute only");
    // Mixed stages, random: between the longer stage's sum and the serial sum; the streamed form's estimate, the longer
    // sum plus min(c, d) of the first expert, is within one stage of the exact end.
    uint64_t state = 99;
    auto unit = [&] { state = state * 6364136223846793005ull + 1442695040888963407ull; return double(state >> 11) / double(1ull << 53); };
    // Random layers: between the longer stage's sum and the serial sum. Where one stage is the longer of every pair
    // (the runs: Muse's copies on an RTX 5060 Ti and an RTX 5090 against its chain, DeepSeek's hand kernel against
    // its copies), the streamed form's estimate, the longer sum plus min(c, d) of the first expert, is within one
    // stage of the exact end; where they alternate the pipeline idles, and the estimate's shortfall is printed.
    size_t bounded = 0, close = 0, total = 0;
    double worst_mixed = 0;
    for (int trial = 0; trial < 3000; ++trial) {
        const size_t n = 1 + size_t(unit() * 200);
        const int kind = trial % 3;               // 0: link-bound, 1: compute-bound, 2: mixed
        std::vector<double> c(n), d(n);
        double sc = 0, sd = 0, top = 0;
        for (size_t i = 0; i < n; ++i) {
            const double a = 100 + 2400 * unit(), b = 2600 + 2400 * unit();
            c[i] = kind == 0 ? a : kind == 1 ? b : 100 + 4900 * unit();
            d[i] = kind == 0 ? b : kind == 1 ? a : 100 + 4900 * unit();
            sc += c[i]; sd += d[i]; top = std::max(top, std::max(c[i], d[i]));
        }
        const double exact = kimi_place_pipeline(c.data(), d.data(), n);
        const double estimate = std::max(sc, sd) + std::min(c[0], d[0]);
        bounded += exact >= std::max(sc, sd) - 1e-6 && exact <= sc + sd + 1e-6;
        if (kind < 2) { close += std::fabs(exact - estimate) <= top + 1e-6; }
        else { worst_mixed = std::max(worst_mixed, (exact - estimate) / exact); }
        total += kind < 2;
    }
    check(bounded == 3000, "the pipeline lies between its longer stage and the serial sum");
    check(close == total, "one stage longer for every pair: the streamed estimate is within one stage of the pipeline");
    std::printf("pipeline: link-bound, compute-bound, serial, in place; %zu/3000 bounded, %zu/%zu estimates within a stage, "
        "alternating stages: the estimate short by at most %.1f%%\n", bounded, close, total, 100 * worst_mixed);
}

// A table and a layer's items for the planner tests: RTX 5060 Ti-like DeepSeek terms (us; MB).
static kimi_place_table table_233() {
    kimi_place_table t;
    t.cpu_one_us = 900; t.cpu_fixed_us = 520; t.cpu_route_us = 297; t.cpu_early = 0.65;
    t.vector.device = { 150, 20, 0 }; t.vector.moved = { 150, 25, 0 };
    t.batched.device = { 0, 60, 25 };
    t.token_us = 7.0;
    t.locked_us_per_mb = 1e3 / 6.91; t.pageable_us_per_mb = 1e3 / 5.83;
    t.route_us = 5448e-6 * t.locked_us_per_mb;
    t.drive_fixed_us = 12000; t.drive_us_per_mb = 1e3 / 7.0;
    return t;
}

static std::vector<kimi_place_item> layer_items(uint64_t & state, size_t n, int ntokens, double resident_share,
        double ram_share, bool locked_ring) {
    auto unit = [&] { state = state * 6364136223846793005ull + 1442695040888963407ull; return double(state >> 11) / double(1ull << 53); };
    std::vector<kimi_place_item> items(n);
    for (auto & x : items) {
        x.routes = ntokens == 1 ? 1 : 1 + int(ntokens * 0.1 * std::pow(unit(), 3.0));
        x.mb = 18.8; x.mb_early = 12.5;
        const double u = unit();
        if (u < resident_share) { x.resident = true; continue; }
        x.read = u >= resident_share + ram_share;
        x.locked = x.read ? locked_ring : unit() < 0.3;
        x.movable = true;
    }
    return items;
}

// One token: the vector engine, the one-token CPU time, page-locked copies only. The grouped path's lines and the
// pageable rate do not change the answer; a pass of one token and a decode token of the same layer are the same call.
static void test_place_one_token() {
    uint64_t state = 31337;
    size_t same = 0, no_pageable = 0, total = 0, moved = 0;
    for (int trial = 0; trial < 2000; ++trial) {
        auto t = table_233();
        if (trial % 3 == 0) { t.locked_us_per_mb = 1e3 / 40.0; t.cpu_one_us = 470; }   // RTX 5090-like
        if (trial % 3 == 1) { t.in_place = true; t.cpu_one_us = 1300; t.vector.moved = { 150, 219, 0 }; } // Spark-like
        const auto items = layer_items(state, 1 + size_t(state % 16), 1, 0.4, 0.4, trial % 2 == 0);
        const auto a = kimi_place_layer(t, 1, 8, items.data(), items.size());
        auto u = t;
        u.cpu_fixed_us *= 3; u.cpu_route_us *= 0.1; u.batched.device = { 0, 1, 1 }; u.batched.moved = { 0, 2, 2 };
        u.token_us = 1000; u.pageable_us_per_mb = 1e-3; u.route_us = 1e3;
        const auto b = kimi_place_layer(u, 1, 8, items.data(), items.size());
        same += a.gpu == b.gpu && a.time == b.time && a.time_none == b.time_none;
        bool pageable = false;
        for (size_t i = 0; i < items.size(); ++i) { pageable = pageable || (a.gpu[i] && !items[i].locked && !t.in_place); }
        no_pageable += !pageable;
        moved += !a.moved.empty();
        ++total;
    }
    check(same == total, "one token: only the vector engine's and the one-token CPU's lines count");
    check(no_pageable == total, "one token: no pageable source moves");
    std::printf("one token: %zu/%zu same answer under other grouped lines, %zu with moves\n", same, total, moved);
}

// The decode plan as it was built before the table (eb03f96..fed8444, kimi-cpu-sink.cpp plan_decode_layer), from the
// same terms: the table's one-token answer is the same, bit for bit.
static void test_place_decode_same() {
    uint64_t state = 2718;
    auto unit = [&] { state = state * 6364136223846793005ull + 1442695040888963407ull; return double(state >> 11) / double(1ull << 53); };
    size_t same = 0, total = 0, moved = 0;
    for (int trial = 0; trial < 3000; ++trial) {
        const bool in_place = trial % 4 == 0;
        const size_t n = 1 + size_t(unit() * 12);
        const double cpu = 300 + 900 * unit(), early_share = 0.5 + 0.3 * unit();
        const double resident_call = 100 + 100 * unit(), resident_b = 10 + 30 * unit();
        const double host_call = unit() < 0.5 ? 0.0 : 100 + 100 * unit(), host_b = unit() < 0.5 ? 0.0 : 20 + 300 * unit();
        const double copy_us_per_mb = in_place ? 0.0 : 10 + 150 * unit(), link_busy = unit() < 0.7 ? 0.0 : 2000 * unit();
        const double d0 = unit() < 0.3 ? 0.0 : 500 * unit(), us_per_mb = 50 + 150 * unit();
        const double cpu_moved = unit() < 0.5 ? 1.0 : 1.0 + unit();
        const int scratch = int(unit() * 3);
        std::vector<kimi_place_item> items(n);
        std::vector<uint8_t> reserved(n, 0);
        for (size_t i = 0; i < n; ++i) {
            auto & x = items[i];
            x.routes = 1; x.mb = 10 + 20 * unit(); x.mb_early = x.mb * 2 / 3;
            const double u = unit();
            if (u < 0.35) { x.resident = true; continue; }
            x.read = u > 0.7;
            x.locked = unit() < 0.6;
            x.movable = x.read || unit() < 0.85;   // the caller's movable flag (the old plan's prefetch filter)
            reserved[i] = unit() < 0.2;
            x.needs_slot = !reserved[i];
        }
        // before: the list schedule's inputs as plan_decode_layer built them
        std::vector<kimi_place_expert> e(n);
        std::vector<double> gate_land(n, 0), down_land(n, 0);
        double mb = 0, reads_end = 0;
        bool any_read = false;
        for (size_t i = 0; i < n; ++i) { if (!items[i].resident && items[i].read) { mb += items[i].mb_early; gate_land[i] = d0 + mb * us_per_mb; any_read = true; } }
        for (size_t i = 0; i < n; ++i) { if (!items[i].resident && items[i].read) { mb += items[i].mb - items[i].mb_early; down_land[i] = d0 + mb * us_per_mb; } }
        if (any_read) { reads_end = d0 + mb * us_per_mb; }
        std::vector<size_t> hits, misses;
        const double hb = host_b > 0 ? host_b : resident_b;
        for (size_t i = 0; i < n; ++i) {
            const auto & x = items[i];
            auto & y = e[i];
            if (x.resident) { y.resident = true; y.gpu = resident_b; continue; }
            y.cpu = cpu; y.early = early_share * cpu; y.gpu = hb;
            y.link = in_place ? 0.0 : x.mb * copy_us_per_mb;
            if (!x.read) { if (x.movable && (in_place || x.locked)) { hits.push_back(i); } continue; }
            y.early_ready = gate_land[i]; y.ready = down_land[i];
            if (in_place || x.locked) { misses.push_back(i); }
        }
        std::stable_sort(misses.begin(), misses.end(), [&](size_t a, size_t b) { return down_land[a] < down_land[b]; });
        std::vector<size_t> candidates(hits);
        candidates.insert(candidates.end(), misses.begin(), misses.end());
        if (!in_place) {
            size_t left = size_t(scratch);
            std::vector<size_t> kept;
            for (size_t i : candidates) { if (reserved[i]) { kept.push_back(i); } else if (left) { --left; kept.push_back(i); } }
            candidates.swap(kept);
        }
        kimi_place_lanes lanes = {};
        lanes.resident_call = resident_call; lanes.host_call = host_call > 0 ? host_call : resident_call;
        lanes.reads_end = reads_end; lanes.link_free = link_busy; lanes.cpu_moved = cpu_moved;
        double t_old = 0;
        const size_t k_old = kimi_place_ring_best_prefix(lanes, e.data(), n, candidates.data(), candidates.size(), &t_old);
        std::vector<uint8_t> none(n, 0);
        const double none_old = kimi_place_ring_time(lanes, e.data(), n, none.data());
        // now: the table
        kimi_place_table t;
        t.cpu_one_us = cpu; t.cpu_early = early_share; t.cpu_moved = cpu_moved;
        t.vector.device = { resident_call, resident_b, 0 };
        t.vector.moved = { host_call > 0 ? host_call : resident_call, hb, 0 };
        t.locked_us_per_mb = copy_us_per_mb; t.link_busy_us = link_busy; t.in_place = in_place;
        t.drive_fixed_us = d0; t.drive_us_per_mb = us_per_mb;
        t.batched.device = { 0, 1e6, 1e6 }; t.pageable_us_per_mb = 1.0;   // not a one-token layer's
        kimi_place_limits limits;
        limits.slots = size_t(scratch);
        if (copy_us_per_mb <= 0 && !in_place) { continue; }
        size_t reads = 0;
        for (const auto & x : items) { reads += !x.resident && x.read; }
        if (reads > 8) { continue; }    // decode plans a layer whose reads fit the ring (8 slots)
        const auto r = kimi_place_layer(t, 1, 8, items.data(), n, limits);
        bool ok = r.moved.size() == k_old && r.time == t_old && r.time_none == none_old;
        for (size_t j = 0; ok && j < k_old; ++j) { ok = r.moved[j] == candidates[j]; }
        same += ok;
        moved += k_old > 0;
        ++total;
    }
    check(same == total, "one token: the decode plan's choice and time, bit for bit");
    std::printf("decode same: %zu of %zu random decode layers (%zu move experts)\n", same, total, moved);
}

// The streamed form of a longer pass is M8's evaluator (kimi_prefill_best_prefix) over M8's moves with B3's transport:
// each copy the pipeline's first stage on the link, page-locked DMA or pageable staging by the GPU's worker (which
// blocks the worker, not the chains it launched), the GPU lane the chains alone; on random RTX 5060 Ti-like layers
// with a pageable ring and RAM cache (M8's sources).
static void test_place_m8_same() {
    uint64_t state = 1618;
    auto unit = [&] { state = state * 6364136223846793005ull + 1442695040888963407ull; return double(state >> 11) / double(1ull << 53); };
    size_t same = 0, total = 0, moved = 0;
    for (int trial = 0; trial < 1000; ++trial) {
        const int ntokens = 64 + int(unit() * 4096);
        const size_t n = 20 + size_t(unit() * 230);
        auto items = layer_items(state, n, ntokens, 0.2 * unit(), 0.3 * unit(), false);
        for (auto & x : items) { x.locked = false; }   // M8's ring and RAM cache: pageable
        auto t = table_233();
        t.locked_us_per_mb = t.pageable_us_per_mb;   // M8 priced every copy and row at its one link rate
        t.route_us = 5448e-6 * t.locked_us_per_mb;
        // M8, as select_prefill_transient built it (bytes, us)
        const double link = 1e3 / t.pageable_us_per_mb * 1000.0; // bytes per us
        kimi_prefill_layer l = {};
        double drive_bytes = 0, first = 0;
        for (const auto & x : items) {
            const double bytes = x.mb * 1e6;
            if (x.resident) { l.gpu += 60 + 25.0 * x.routes; l.link += (5448.0 * x.routes) / link; l.gpu_work = true; continue; }
            l.cpu += 520 + 297.0 * x.routes;
            if (x.read) { drive_bytes += bytes; if (first == 0) { first = bytes; } }
        }
        const double drive_rate = 7.0 * 1000.0;
        if (drive_bytes > 0) { l.drive = 12000 + drive_bytes / drive_rate; l.fill = first / drive_rate; }
        l.ring = 8; l.gpu_overhead = ntokens * 7.0;
        std::vector<size_t> c;
        for (size_t i = 0; i < n; ++i) { if (!items[i].resident) { c.push_back(i); } }
        std::stable_sort(c.begin(), c.end(), [&](size_t a, size_t b) {
            if (items[a].routes != items[b].routes) { return items[a].routes > items[b].routes; }
            return items[a].read < items[b].read;
        });
        std::vector<kimi_prefill_move> moves;
        for (size_t i : c) {
            const auto & x = items[i];
            const double bytes = x.mb * 1e6;
            moves.push_back({ 520 + 297.0 * x.routes, 60 + 25.0 * x.routes, (bytes + 5448.0 * x.routes) / link, bytes / link });
        }
        double t_old = 0;
        const size_t k_old = kimi_prefill_best_prefix(l, moves.data(), moves.size(), &t_old);
        const auto r = kimi_place_layer(t, ntokens, 8, items.data(), n);
        bool ok = !r.ring && r.moved.size() == k_old && near(r.time, t_old, 1e-9);
        for (size_t j = 0; ok && j < k_old; ++j) { ok = r.moved[j] == c[j]; }
        same += ok;
        moved += k_old > 0;
        ++total;
    }
    check(same == total, "a streamed pass with a pageable ring: M8's evaluator, the copies pipelined");
    std::printf("m8 same: %zu of %zu random streamed layers (%zu move experts)\n", same, total, moved);
}

// The lanes of a longer pass: a page-locked copy on the link alone, the GPU lane its compute (B3), the pipeline's fill
// once; a pageable copy on both; in place no copy at all.
static void test_place_lanes() {
    kimi_place_table t = table_233();
    t.batched.device = { 0, 120, 1.7 };        // Muse's chain on an RTX 5060 Ti, B2
    t.cpu_fixed_us = 175; t.cpu_route_us = 409;
    t.route_us = 8e-6 * t.locked_us_per_mb; t.token_us = 2.3;
    // 30 cold experts of 30.4 MB at 150 routes, a page-locked ring: every move's copy 4399 us on the link, its chain 375
    std::vector<kimi_place_item> items(30);
    for (auto & x : items) { x.routes = 150; x.mb = 30.4; x.mb_early = 20.3; x.read = true; x.locked = true; x.movable = true; }
    auto r = kimi_place_layer(t, 4221, 8, items.data(), items.size());
    const double c = 120 + 1.7 * 150, d = 30.4 * t.locked_us_per_mb;
    check(!r.ring && !r.moved.empty(), "locked: a streamed layer moves experts");
    {
        std::vector<double> cs(r.moved.size(), c), ds(r.moved.size(), d);
        check(near(r.gpu_lane, kimi_place_pipeline(cs.data(), ds.data(), cs.size()), 1e-12),
            "locked: the GPU lane is the pipeline of the copies and the chains");
        check(near(r.gpu_lane, double(r.moved.size()) * d + c, 1e-12), "locked: link-bound, the copies and the last chain");
    }
    check(near(r.link, double(r.moved.size()) * (d + 150 * 8e-6 * t.locked_us_per_mb), 1e-9), "locked: the link carries the copies");
    // the same with a pageable ring: each copy staged by the GPU's worker at the pageable rate, the pipeline's first
    // stage as a DMA copy is (the chains it launched run on); fewer moves
    for (auto & x : items) { x.locked = false; }
    auto p = kimi_place_layer(t, 4221, 8, items.data(), items.size());
    const double s = 30.4 * t.pageable_us_per_mb;
    {
        std::vector<double> cs(p.moved.size(), c), ss(p.moved.size(), s);
        check(!p.moved.empty() && near(p.gpu_lane, kimi_place_pipeline(cs.data(), ss.data(), cs.size()), 1e-9) && p.fill > 0.0,
            "pageable: the staging pipelined with the chains");
    }
    check(p.moved.size() <= r.moved.size(), "pageable: no more moves than page-locked");
    // in place where the GPU can: the in-place line (504 us at 150 routes) against the device line plus a copy
    // (375 + 5214 us): no link, no fill, the moved line
    t.in_place = true; t.batched.moved = { 0, 174, 2.2 };
    auto q = kimi_place_layer(t, 4221, 8, items.data(), items.size());
    check(q.in_place && q.link == 0.0 && near(q.fill, 0.0, 1e-9) &&
        near(q.gpu_lane, double(q.moved.size()) * (174 + 2.2 * 150), 1e-9), "in place: compute from host memory, nothing on the link");
    check(q.moved.size() >= r.moved.size(), "in place: at least as many moves as with copies");
    // where the GPU reads in place slower than it copies and computes from device memory: the copies, each move on the
    // device line (not the in-place one) after its copy
    t.batched.moved = { 0, 9000, 60 };
    for (auto & x : items) { x.locked = true; }
    auto v = kimi_place_layer(t, 4221, 8, items.data(), items.size());
    {
        std::vector<double> cs(v.moved.size(), c), ds(v.moved.size(), d);
        check(!v.in_place && !v.moved.empty() && near(v.gpu_lane, kimi_place_pipeline(cs.data(), ds.data(), cs.size()), 1e-9),
            "in place slower: copied, on the device line");
    }
    std::printf("lanes: locked %zu moves (GPU %.0f link %.0f fill %.0f), pageable %zu, in place %zu, in place slower %zu of %zu\n",
        r.moved.size(), r.gpu_lane, r.link, r.fill, p.moved.size(), q.moved.size(), v.moved.size(), items.size());

    // a longer pass's CPU line is its own calibration: decode's DRAM-sharing ratio does not price it (ring form)
    {
        kimi_place_table w = table_233();
        w.cpu_fixed_us = 3000;
        std::vector<kimi_place_item> few(4);
        for (auto & x : few) { x.routes = 40; x.mb = 18.8; x.read = true; x.locked = true; x.movable = true; }
        const auto a = kimi_place_layer(w, 64, 8, few.data(), few.size());
        w.cpu_moved = 3.0;
        const auto b = kimi_place_layer(w, 64, 8, few.data(), few.size());
        check(a.ring && !a.moved.empty() && a.gpu == b.gpu && a.time == b.time, "a longer pass: no decode DRAM ratio");
    }

    // The candidates' order: most routes first, then host memory before reads, then the items' order; limits.
    std::vector<kimi_place_item> o(6);
    const int routes[6] = { 3, 9, 9, 1, 9, 3 };
    for (size_t i = 0; i < 6; ++i) { o[i].routes = routes[i]; o[i].mb = 18.8; o[i].movable = true; o[i].locked = true; }
    o[1].read = true; o[2].read = true;      // two reads among the 9-route experts, 4 in host memory
    kimi_place_table u = table_233();
    u.cpu_fixed_us = 1e5;                    // every move pays
    auto all = kimi_place_layer(u, 64, 8, o.data(), o.size());
    const std::vector<size_t> want = { 4, 1, 2, 0, 5, 3 };
    check(all.moved == want, "candidates: by routes, host memory first, then in order");
    kimi_place_limits lim; lim.reads = 1;
    o[0].needs_slot = o[5].needs_slot = true; lim.slots = 1;
    auto limited = kimi_place_layer(u, 64, 8, o.data(), o.size(), lim);
    const std::vector<size_t> want2 = { 4, 1, 0, 3 };
    check(limited.moved == want2, "candidates: a read limit and a slot limit");
}

// Decode's candidates: every expert the GPU could take, a RAM hit its prefetch still reads among them, priced with its
// landing (its bytes yet to come, on the drive's line ahead of the layer's reads); before, such a hit was the CPU's.
static void test_place_pending() {
    // one token, in place (the Spark): 2 residents, a RAM hit in memory (H), a hit still landing (P, 10 MB to come),
    // two misses (M1, M2)
    kimi_place_table t;
    t.cpu_one_us = 1600; t.cpu_early = 0.6;
    t.vector.device = { 150, 20, 0 }; t.vector.moved = { 150, 300, 0 };
    t.in_place = true;
    t.drive_fixed_us = 500; t.drive_us_per_mb = 1e3 / 7.0;
    std::vector<kimi_place_item> items(6);
    for (auto & x : items) { x.routes = 1; x.mb = 18.8; x.mb_early = 12.5; x.movable = true; }
    items[0].resident = items[1].resident = true;
    const size_t H = 2, P = 3, M1 = 4, M2 = 5;
    items[P].pending_mb = 10.0;
    items[M1].read = items[M2].read = true;
    const auto a = kimi_place_layer(t, 1, 8, items.data(), items.size());
    // the landings: P first, then the misses' up/gate roles, then their downs
    const double r = t.drive_us_per_mb, d0 = t.drive_fixed_us;
    std::vector<kimi_place_expert> e(items.size());
    for (size_t i = 0; i < items.size(); ++i) {
        if (items[i].resident) { e[i].resident = true; e[i].gpu = 20; continue; }
        e[i].cpu = 1600; e[i].early = 0.6 * 1600; e[i].gpu = 300;
    }
    e[P].early_ready = e[P].ready = d0 + 10 * r;
    e[M1].early_ready = d0 + (10 + 12.5) * r; e[M2].early_ready = d0 + (10 + 25) * r;
    e[M1].ready = d0 + (35 + 6.3) * r; e[M2].ready = d0 + (35 + 12.6) * r;
    kimi_place_lanes lanes = {};
    lanes.resident_call = 150; lanes.host_call = 150; lanes.reads_end = e[M2].ready;
    std::vector<uint8_t> none(items.size(), 0);
    check(near(a.time_none, kimi_place_ring_time(lanes, e.data(), e.size(), none.data()), 1e-12),
        "pending: the hit lands ahead of the layer's reads, the reads after it");
    check(a.candidates == 4, "pending: the landing hit is a candidate");
    const std::vector<size_t> order = { H, P, M1, M2 };
    check(a.moved == order, "pending: every candidate moves when the CPU is the slow lane, in landing order");
    // the caller's flag still rules: a hit it keeps on the CPU is no candidate
    auto kept = items;
    kept[P].movable = false;
    const auto b = kimi_place_layer(t, 1, 8, kept.data(), kept.size());
    check(b.candidates == 3 && std::find(b.moved.begin(), b.moved.end(), P) == b.moved.end(), "pending: not movable, not moved");
    check(near(b.time_none, a.time_none, 1e-12), "pending: its landing prices the CPU too");
    check(a.time <= b.time, "pending: moving the landing hit is no slower (here the misses' last landing ends the layer)");
    {
        // without the misses the landing hit's CPU time ends the layer: moving it shortens it
        auto two = items;
        two[M1].read = two[M2].read = false; two[M1].resident = two[M2].resident = true;
        const auto x = kimi_place_layer(t, 1, 8, two.data(), two.size());
        two[P].movable = false;
        const auto y = kimi_place_layer(t, 1, 8, two.data(), two.size());
        check(std::find(x.moved.begin(), x.moved.end(), P) != x.moved.end() && x.time < y.time &&
            near(x.time, d0 + 10 * r + 150 + 300, 1e-12), "pending: the GPU takes the hit once it lands, at the host call");
    }
    // without a drive line nothing is known of its landing: in host memory, as a RAM hit
    auto u = t;
    u.drive_us_per_mb = 0;
    const auto c = kimi_place_layer(u, 1, 8, items.data(), items.size());
    auto plain = items;
    plain[P].pending_mb = 0;
    const auto d = kimi_place_layer(u, 1, 8, plain.data(), plain.size());
    check(c.moved == d.moved && c.time == d.time, "pending: no drive line, a RAM hit");
    // a discrete GPU: the landing hit moves from its page-locked RAM slot at the measured rate only, with a device target
    auto g = t;
    g.in_place = false; g.locked_us_per_mb = 1e3 / 40.0;
    auto disc = items;
    for (auto & x : disc) { x.locked = true; x.needs_slot = true; }
    disc[P].locked = false;
    kimi_place_limits lim; lim.slots = 8;
    const auto f = kimi_place_layer(g, 1, 8, disc.data(), disc.size(), lim);
    check(std::find(f.moved.begin(), f.moved.end(), P) == f.moved.end() && f.candidates == 3, "pending: pageable slot, no move at one token");
    disc[P].locked = true;
    lim.slots = 1;
    const auto h = kimi_place_layer(g, 1, 8, disc.data(), disc.size(), lim);
    check(h.candidates == 1 && h.moved.size() <= 1, "pending: a device target a move");
    // random Spark-like decode layers: the candidates gain the landing hits; the moves and the predicted time with them
    // against the old rule's (the same landings, the hits the CPU's)
    uint64_t state = 4242;
    auto unit = [&] { state = state * 6364136223846793005ull + 1442695040888963407ull; return double(state >> 11) / double(1ull << 53); };
    size_t layers = 0, gained = 0, more = 0, fewer = 0, faster = 0, slower = 0;
    double moves_old = 0, moves_new = 0;
    for (int trial = 0; trial < 3000; ++trial) {
        const size_t n = 4 + size_t(unit() * 10);
        std::vector<kimi_place_item> x(n);
        size_t landing = 0;
        for (auto & y : x) {
            y.routes = 1; y.mb = 18.8; y.mb_early = 12.5; y.movable = true;
            const double v = unit();
            if (v < 0.45) { y.resident = true; continue; }
            if (v < 0.7) { y.read = true; continue; }
            if (unit() < 0.4) { y.pending_mb = 18.8 * (1 + int(unit() * 3)) / 3.0; ++landing; }
        }
        auto old = x;
        for (auto & y : old) { if (y.pending_mb > 0) { y.movable = false; } }
        auto tt = t;
        tt.cpu_one_us = 1200 + 800 * unit(); tt.vector.moved.expert_us = 200 + 200 * unit();
        const auto now = kimi_place_layer(tt, 1, 8, x.data(), n);
        const auto was = kimi_place_layer(tt, 1, 8, old.data(), n);
        ++layers;
        gained += now.candidates == was.candidates + landing;
        moves_new += double(now.moved.size()); moves_old += double(was.moved.size());
        more += now.moved.size() > was.moved.size(); fewer += now.moved.size() < was.moved.size();
        faster += now.time < was.time; slower += now.time > was.time;
    }
    check(gained == layers, "pending: every landing hit is a candidate");
    check(moves_new >= moves_old, "pending: at least the old rule's moves on CPU-bound layers");
    std::printf("pending: %zu random layers, moves %.2f against %.2f a layer (more %zu, fewer %zu), predicted faster %zu, slower %zu\n",
        layers, moves_new / double(layers), moves_old / double(layers), more, fewer, faster, slower);
}

int main() {
    test_fit();
    test_overhead_fit();
    test_coupled();
    test_select();
    test_same_choice();
    test_ring();
    test_ring_same();
    test_ring_prefix_fold();
    test_pipeline();
    test_place_one_token();
    test_place_decode_same();
    test_place_m8_same();
    test_place_lanes();
    test_place_pending();
    if (failures) { std::fprintf(stderr, "%d failures\n", failures); return 1; }
    std::printf("KIMI_PREFILL_COST_OK\n");
    return 0;
}
