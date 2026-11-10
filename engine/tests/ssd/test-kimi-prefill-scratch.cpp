#include "gpu-services.h"
#include "explicit-fixture-inputs.h"
#include <cstring>
// Reuse one real grouped-prefill handle as expert service moves CPU -> GPU.
// Synthetic packed weights; real tier kernels; zero input is an exact oracle
// for these bias-free experts, independent of CPU/GPU rounding differences.
#define main original_sink_test_main
#include "test-kimi-cpu-sink.cpp"
#undef main

namespace {
using prefill_ptr = std::unique_ptr<kimi_cpu_prefill, decltype(&kimi_cpu_prefill_destroy)>;

size_t failures = 0, cases = 0;
void check_zero(const char * label, ggml_type type, int tokens, const std::vector<float> & values) {
    double maximum = 0;
    size_t first = values.size();
    for (size_t i = 0; i < values.size(); ++i) {
        require(std::isfinite(values[i]), "nonfinite scratch regression output");
        maximum = std::max(maximum, std::abs(double(values[i])));
        if (values[i] != 0 && first == values.size()) { first = i; }
    }
    ++cases;
    if (maximum != 0) { ++failures; }
    std::printf("SCRATCH_ZERO type=%s tokens=%d case=%s max_abs=%.9g first_nonzero=%lld status=%s\n",
        ggml_type_name(type), tokens, label, maximum,
        first == values.size() ? -1LL : static_cast<long long>(first), maximum == 0 ? "PASS" : "FAIL");
    std::fflush(stdout);
}

void run_case(fixture_inputs & inputs, ggml_type type, int tokens, int activation_kind) {
    fixture f; f.type = f.down_type = type;
    f.latent = 512; f.hidden = 256; f.experts = 4;
    make_synthetic(f, 20261004);
    const auto unit = triplet_bytes(f);
    inputs.set("vram_budget", std::to_string(8 * unit).c_str());
    inputs.set("vram_prefill", "1");
    inputs.set("ready_first", "1");
    inputs.set("tier.keep_across_requests", "1");
    inputs.set("tier.min_count", "0");
    for (const char * key : {"tier.global", "vram_exclusive",
            "vram_async", "vram_deferred",
            "tier.elastic", "early_up_gate",
            "role_major", "nonblocking_down_sched"}) {
        inputs.set(key, "0");
    }
    auto sink = make_sink(inputs, f, 4, 2);
    configure_cache(sink.get(), 0, {1}, 4);
    tensors meta(f, 1);
    char error[512] = {};
    prefill_ptr reused(kimi_cpu_prefill_create(sink.get(), error, sizeof(error)), kimi_cpu_prefill_destroy);
    require(bool(reused), error);
    constexpr int slots = 4;
    std::vector<int32_t> ids(size_t(tokens) * slots);
    std::vector<float> weights(ids.size(), 0.25f), input(size_t(tokens) * f.latent), output(input.size());
    for (size_t i = 0; i < ids.size(); ++i) { ids[i] = int(i % slots); }
    for (size_t i = 0; i < input.size(); ++i) { input[i] = 1.7f * std::sin(float(i) * 0.17f); }
    const auto nonzero_input = input;
    const kimi_cpu_sink_activation activation = {activation_kind, 0, 0, activation_kind ==
        KIMI_CPU_SINK_ACTIVATION_SWIGLU_CLAMP ? 10.0f : 0.0f};
    auto run = [&](kimi_cpu_prefill * handle) {
        kimi_cpu_prefill_stats stats = {};
        require(kimi_cpu_prefill_run_activation(handle, meta.role[0], meta.role[1], meta.role[2],
            input.data(), ids.data(), weights.data(), slots, tokens, &activation, output.data(),
            &stats, error, sizeof(error)), error);
        require(stats.stream.read_errors == 0 && stats.stream.pending_reads == 0 &&
            stats.stream.fallback_count == 0, "prefill scratch regression transport failure");
        return stats;
    };
    auto seed = [&](int lo, int hi) {
        std::vector<float> single(size_t(f.latent), 0.125f), single_out(size_t(f.latent));
        for (int expert = lo; expert < hi; ++expert) {
            begin_cache(sink.get(), 20261004, uint64_t(expert + 1));
            const float weight = 1;
            require(kimi_cpu_sink_run_activation(sink.get(), meta.role[0], meta.role[1], meta.role[2],
                single.data(), &expert, &weight, 1, &activation, single_out.data(), nullptr,
                error, sizeof(error)), error);
        }
    };
    auto stats = run(reused.get());
    require(stats.gpu_routes == 0 && stats.ssd_experts == 4, "initial call must populate CPU route scratch");
    require(std::any_of(output.begin(), output.end(), [](float v) { return std::abs(v) > 1e-5f; }),
        "scratch poison call has no signal");
    seed(0, 2);
    std::fill(input.begin(), input.end(), 0);
    stats = run(reused.get());
    require(stats.gpu_experts == 2 && stats.gpu_routes == uint64_t(2 * tokens) && stats.ssd_experts == 2,
        "CPU -> mixed call did not exercise resident GPU and CPU experts");
    check_zero("cpu_to_mixed_reused", type, tokens, output);
    prefill_ptr fresh(kimi_cpu_prefill_create(sink.get(), error, sizeof(error)), kimi_cpu_prefill_destroy);
    require(bool(fresh), error);
    run(fresh.get());
    check_zero("mixed_fresh_control", type, tokens, output);
    // Poison the remaining CPU slots, then migrate them to GPU. This covers
    // mixed -> all GPU on the same handle, including the <=64 boundary.
    input = nonzero_input;
    run(reused.get());
    seed(2, 4);
    std::fill(input.begin(), input.end(), 0);
    stats = run(reused.get());
    require(stats.gpu_experts == 4 && stats.gpu_routes == uint64_t(slots * tokens) && stats.ssd_experts == 0,
        "mixed -> GPU call did not serve all routes on GPU");
    check_zero("mixed_to_gpu_reused", type, tokens, output);
    run(fresh.get());
    check_zero("gpu_fresh_control", type, tokens, output);
    run(reused.get());
    check_zero("gpu_repeated_reused", type, tokens, output);
}
}

// An SSD-sink prefill seeds the tier's empty slots with the experts its CPU computes, and decode then reads them. The
// seeded tier's decode admits nothing, so only the seeds can put the experts on the GPU for a later pass, whose output
// matches that of a tier which took the same experts by decode admission: the same kernels over the same bytes.
void run_seed_case(fixture_inputs & inputs, ggml_type type, int tokens, int activation_kind) {
    fixture f; f.type = f.down_type = type;
    f.latent = 512; f.hidden = 256; f.experts = 4;
    make_synthetic(f, 20261110);
    const auto unit = triplet_bytes(f);
    inputs.set("vram_budget", std::to_string(8 * unit).c_str());
    inputs.set("vram_prefill", "1");
    inputs.set("ready_first", "1");
    inputs.set("tier.keep_across_requests", "1");
    // the seeded tier admits an expert at a decayed count of 4, which one decode token an expert never reaches
    inputs.set("tier.policy", "count_decay2");
    inputs.set("tier.adaptive_admission", "0");
    inputs.set("tier.min_count", "4");
    inputs.set("tier.global", "1");
    for (const char * key : {"vram_exclusive", "vram_async", "vram_deferred", "tier.elastic", "early_up_gate",
            "role_major", "nonblocking_down_sched"}) {
        inputs.set(key, "0");
    }
    tensors meta(f, 1);
    char error[512] = {};
    constexpr int slots = 4;
    std::vector<int32_t> ids(size_t(tokens) * slots);
    std::vector<float> weights(ids.size(), 0.25f), input(size_t(tokens) * f.latent);
    for (size_t i = 0; i < ids.size(); ++i) { ids[i] = int(i % slots); }
    for (size_t i = 0; i < input.size(); ++i) { input[i] = 1.7f * std::sin(float(i) * 0.17f); }
    const kimi_cpu_sink_activation activation = {activation_kind, 0, 0, activation_kind ==
        KIMI_CPU_SINK_ACTIVATION_SWIGLU_CLAMP ? 10.0f : 0.0f};
    auto run = [&](kimi_cpu_prefill * handle, std::vector<float> & output) {
        output.assign(input.size(), 0.0f);
        kimi_cpu_prefill_stats stats = {};
        require(kimi_cpu_prefill_run_activation(handle, meta.role[0], meta.role[1], meta.role[2],
            input.data(), ids.data(), weights.data(), slots, tokens, &activation, output.data(),
            &stats, error, sizeof(error)), error);
        require(stats.stream.read_errors == 0 && stats.stream.pending_reads == 0 &&
            stats.stream.fallback_count == 0, "prefill seed transport failure");
        return stats;
    };
    uint64_t token = 0;
    auto decode = [&](kimi_cpu_sink * sink, int lo, int hi) {
        std::vector<float> single(size_t(f.latent), 0.125f), single_out(size_t(f.latent));
        for (int expert = lo; expert < hi; ++expert) {
            begin_cache(sink, 20261110, ++token);
            const float weight = 1;
            require(kimi_cpu_sink_run_activation(sink, meta.role[0], meta.role[1], meta.role[2],
                single.data(), &expert, &weight, 1, &activation, single_out.data(), nullptr,
                error, sizeof(error)), error);
        }
    };
    // Seeded: a decode token gives the tier its format and admits nothing; a pass computes all four experts on the CPU
    // and seeds them from its payloads; decode reads them.
    auto seeded = make_sink(inputs, f, 4, 2);
    configure_cache(seeded.get(), 0, {1}, 4);
    prefill_ptr seeded_prefill(kimi_cpu_prefill_create(seeded.get(), error, sizeof(error)), kimi_cpu_prefill_destroy);
    require(bool(seeded_prefill), error);
    decode(seeded.get(), 0, 1);
    std::vector<float> cpu_output, seeded_output, admitted_output;
    auto stats = run(seeded_prefill.get(), cpu_output);
    require(stats.gpu_experts == 0 && stats.ssd_experts == 4, "the seeding pass must compute every expert on the CPU");
    require(kimi_cpu_sink_vram_phase_complete(seeded.get(), true, error, sizeof(error)), error);
    decode(seeded.get(), 0, 4);
    stats = run(seeded_prefill.get(), seeded_output);
    require(stats.gpu_experts == 4 && stats.ssd_experts == 0, "the seeded experts must serve the next pass on the GPU");
    // Admitted: decode admits all four, then the same pass.
    inputs.set("tier.policy", "lru");
    inputs.set("tier.min_count", "0");
    auto admitted = make_sink(inputs, f, 4, 2);
    configure_cache(admitted.get(), 0, {1}, 4);
    prefill_ptr admitted_prefill(kimi_cpu_prefill_create(admitted.get(), error, sizeof(error)), kimi_cpu_prefill_destroy);
    require(bool(admitted_prefill), error);
    decode(admitted.get(), 0, 4);
    stats = run(admitted_prefill.get(), admitted_output);
    require(stats.gpu_experts == 4 && stats.ssd_experts == 0, "the admitted experts must serve the pass on the GPU");
    const bool same = std::memcmp(seeded_output.data(), admitted_output.data(), seeded_output.size() * sizeof(float)) == 0;
    ++cases;
    if (!same) { ++failures; }
    std::printf("PREFILL_SEED type=%s tokens=%d seeded_vs_admitted=%s\n", ggml_type_name(type), tokens,
        same ? "PASS" : "FAIL");
    std::fflush(stdout);
}

// The grouped prefill's read-ahead beside the tier's batched lane: layer 1's call, given the pass's own routes as its
// prediction of layer 2's, reads layer 2's experts into the RAM cache (retained, made at load, read by a prefill), and
// layer 2's call takes them as RAM hits on either lane, the GPU's as RAM transients. The tier holds fewer experts than a layer has. Zero input is the exact oracle; a
// nonzero pass matches the graph (a token at a time, the clamped SwiGLU's scalar kernel) within the GPU's rounding,
// which a slot read wrongly would not.
void run_lookahead_case(fixture_inputs & inputs, int tokens) {
    // the read-ahead lands in the RAM cache, which admits only under a finite, readable memory envelope
    try {
        test_resources::finite_envelope();
    } catch (const test_resources::unavailable & e) {
        std::printf("PREFILL_LOOKAHEAD tokens=%d status=SKIP reason=%s\n", tokens, e.what());
        return;
    }
    fixture f; f.type = f.down_type = GGML_TYPE_Q8_0;
    f.latent = 512; f.hidden = 256; f.experts = 16;
    make_synthetic(f, 20261111);
    const auto unit = triplet_bytes(f);
    inputs.set("vram_budget", std::to_string(4 * unit).c_str());
    inputs.set("vram_prefill", "1");
    inputs.set("ready_first", "1");
    inputs.set("tier.keep_across_requests", "1");
    inputs.set("tier.policy", "lru");
    inputs.set("tier.min_count", "0");
    for (const char * key : {"tier.global", "vram_exclusive", "vram_async", "vram_deferred", "tier.elastic",
            "early_up_gate", "role_major", "nonblocking_down_sched"}) {
        inputs.set(key, "0");
    }
    const char * cache_keys[][2] = { {"prefill_lookahead", "1"}, {"place_costs", "drive_gbps=1000"},
        {"cache_policy", "count_decay2"}, {"cache_allocation", "uniform"}, {"cache_keep", "1"},
        {"cache_partial_reclaim", "1"}, {"cache_prefill_read", "1"} };
    for (const auto & k : cache_keys) { inputs.set(k[0], k[1]); }
    auto sink = make_sink(inputs, f, 4, 2);
    tensors meta1(f, 1), meta2(f, 2);
    uint64_t stride = 0;
    for (int role = 0; role < 3; ++role) {
        const auto * t = meta1.role[role];
        stride += (ggml_row_size(t->type, t->ne[0]) * uint64_t(t->ne[1]) + 8190) / 4096 * 4096;
    }
    stride = cache_slot_bytes(inputs, stride);
    for (const auto & k : cache_keys) { inputs.unset(k[0]); }
    configure_cache(sink.get(), 2 * uint64_t(f.experts) * stride, {1, 2}, f.experts);
    char error[512] = {};
    require(kimi_cpu_sink_vram_reserve(sink.get(), meta1.role, 1, 4, error, sizeof(error)), error);
    prefill_ptr prefill(kimi_cpu_prefill_create(sink.get(), error, sizeof(error)), kimi_cpu_prefill_destroy);
    require(bool(prefill), error);
    constexpr int slots = 4;
    std::vector<int32_t> ids(size_t(tokens) * slots);
    for (int t = 0; t < tokens; ++t) {
        for (int s = 0; s < slots; ++s) { ids[size_t(t) * slots + s] = int32_t((t * 3 + s * 4) % f.experts); }
    }
    std::vector<float> weights(ids.size(), 0.25f), zero(size_t(tokens) * f.latent, 0.0f), x(zero.size());
    for (size_t i = 0; i < x.size(); ++i) { x[i] = 0.3f * std::sin(float(i) * 0.23f); }
    const kimi_cpu_sink_activation swiglu = { KIMI_CPU_SINK_ACTIVATION_SWIGLU_CLAMP, 0, 0, 10.0f, nullptr, 0 };
    auto run = [&](const tensors & meta, const std::vector<float> & input, std::vector<float> & output) {
        if (&meta == &meta1) { kimi_cpu_sink_prefill_hint(sink.get(), 2, ids.data(), tokens, slots); }
        output.assign(input.size(), 0.0f);
        kimi_cpu_prefill_stats stats = {};
        require(kimi_cpu_prefill_run_activation(prefill.get(), meta.role[0], meta.role[1], meta.role[2], input.data(),
            ids.data(), weights.data(), slots, tokens, &swiglu, output.data(), &stats, error, sizeof(error)), error);
        require(stats.stream.read_errors == 0 && stats.stream.pending_reads == 0 &&
            stats.stream.read_completions == stats.stream.read_requests &&
            stats.gpu_experts + stats.ram_experts + stats.ssd_experts == uint64_t(f.experts),
            "read-ahead prefill transport or source counters");
        return stats;
    };
    uint64_t served = 0, transient = 0;
    std::vector<float> y1, y2;
    for (int pass = 0; pass < 3; ++pass) {
        run(meta1, zero, y1);
        const auto stats = run(meta2, zero, y2);
        check_zero(pass ? "lookahead_zero" : "lookahead_zero_measure", f.type, tokens, y2);
        if (pass) { served += stats.ram_experts; transient += stats.gpu_transient_experts; }
    }
    ++cases;
    if (!served) { ++failures; }
    // nonzero, against the graph
    run(meta1, x, y1);
    const auto stats = run(meta2, y1, y2);
    served += stats.ram_experts;
    std::vector<float> expected;
    for (int t = 0; t < tokens; ++t) {
        const auto row = y1.begin() + std::ptrdiff_t(t) * f.latent;
        const auto slot = ids.begin() + std::ptrdiff_t(t) * slots;
        const auto w = weights.begin() + std::ptrdiff_t(t) * slots;
        const auto y = reference_moe_graph(f, 2, std::vector<float>(row, row + f.latent), std::vector<int32_t>(slot, slot + slots),
            std::vector<float>(w, w + slots), swiglu, 1);
        expected.insert(expected.end(), y.begin(), y.end());
    }
    double scale = 0, error_max = 0;
    for (size_t i = 0; i < expected.size(); ++i) {
        require(std::isfinite(y2[i]), "nonfinite read-ahead output");
        scale = std::max(scale, std::abs(double(expected[i])));
        error_max = std::max(error_max, std::abs(double(y2[i]) - double(expected[i])));
    }
    const bool close = error_max <= 0.05 * scale;
    ++cases;
    if (!close) { ++failures; }
    std::printf("PREFILL_LOOKAHEAD tokens=%d ram_experts=%llu gpu_transient_experts=%llu last_ram_experts=%llu"
        " last_gpu_transients=%llu max_abs=%.6g scale=%.6g status=%s\n", tokens, (unsigned long long) served,
        (unsigned long long) transient, (unsigned long long) stats.ram_experts,
        (unsigned long long) stats.gpu_transient_experts, error_max, scale, served && close ? "PASS" : "FAIL");
    std::fflush(stdout);
}

int main(int argc, char ** argv) {
    fixture_inputs inputs;
    inputs.set_runtime_services(test_gpu_services());
    try {
        ggml_cpu_init();
        (void) argv;
        require(argc == 1, "usage: test-kimi-prefill-scratch");
        for (int tokens : {9, 14, 48, 64, 65}) {
            run_case(inputs, GGML_TYPE_MXFP4, tokens, KIMI_CPU_SINK_ACTIVATION_SWIGLU_CLAMP);
            run_case(inputs, GGML_TYPE_Q8_0, tokens, KIMI_CPU_SINK_ACTIVATION_SWIGLU);
            run_case(inputs, GGML_TYPE_Q8_0, tokens, KIMI_CPU_SINK_ACTIVATION_SWIGLU_CLAMP);
        }
        run_seed_case(inputs, GGML_TYPE_MXFP4, 48, KIMI_CPU_SINK_ACTIVATION_SWIGLU_CLAMP);
        run_seed_case(inputs, GGML_TYPE_Q8_0, 48, KIMI_CPU_SINK_ACTIVATION_SWIGLU);
        for (int tokens : {48, 300}) { run_lookahead_case(inputs, tokens); }
        std::printf("PREFILL_SCRATCH_%s cases=%zu failures=%zu\n", failures ? "FAIL" : "PASS", cases, failures);
        return failures ? 1 : 0;
    } catch (const std::exception & e) { std::fprintf(stderr, "%s\n", e.what()); return 1; }
}
