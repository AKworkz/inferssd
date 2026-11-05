#include "gpu-services.h"
#include "explicit-fixture-inputs.h"
// End-to-end generic-format CPU-sink test with the real VRAM tier: Q4_K/Q5_K
// as in Muse Spark UD-Q4_K_XL, Q8_0, and MXFP4 as in DeepSeek-V4.1. Synthetic
// packed weights cover non-uniform padded role sizes and mixed CPU/GPU expert
// reduction without requiring the full model. The CUDA path intentionally uses
// Q8_1 activations and is checked numerically, not bitwise.
#define main original_sink_test_main
#include "test-kimi-cpu-sink.cpp"
#undef main
#include "kimi-host-jobs-test.inc"
#include <cuda_runtime.h>

namespace {
double relative_l2(const std::vector<float> & got, const std::vector<float> & expected) {
    double error = 0.0;
    double norm = 0.0;
    for (size_t i = 0; i < got.size(); ++i) {
        const double delta = double(got[i]) - double(expected[i]);
        error += delta * delta;
        norm += double(expected[i]) * double(expected[i]);
    }
    return norm > 0.0 ? std::sqrt(error / norm) : std::sqrt(error);
}

double max_abs_error(const std::vector<float> & got, const std::vector<float> & expected) {
    double result = 0.0;
    for (size_t i = 0; i < got.size(); ++i) {
        result = std::max(result, std::abs(double(got[i]) - double(expected[i])));
    }
    return result;
}

struct mode {
    bool production;
    const char * name;
};

void set_switch(fixture_inputs & inputs, const char * name, bool enabled) {
    inputs.set(name, enabled ? "1" : "0");
}

// layer1_down: layer 1's down type where it differs from layer 2's (Muse Q4_K_S), declared to the tier at load.
void run_shape(fixture_inputs & inputs, int latent, int hidden, int experts, mode selected, size_t & checks,
        double & worst_rel, double & worst_abs,
        ggml_type up_type, ggml_type down_type, ggml_type layer1_down = GGML_TYPE_COUNT) {
    fixture f;
    f.type = up_type;
    f.down_type = down_type;
    if (layer1_down != GGML_TYPE_COUNT) { f.layer_down_types[1] = layer1_down; }
    f.latent = latent;
    f.hidden = hidden;
    f.experts = experts;
    make_synthetic(f, uint32_t(20260920 + latent + int(selected.production)));
    const uint64_t unit = triplet_bytes(f);

    inputs.set("vram_budget", "0");
    set_switch(inputs, "tier.global", false);
    set_switch(inputs, "vram_exclusive", false);
    set_switch(inputs, "vram_async", false);
    set_switch(inputs, "vram_deferred", false);
    set_switch(inputs, "early_up_gate", false);
    set_switch(inputs, "role_major", false);
    set_switch(inputs, "nonblocking_down_sched", false);
    auto cpu = make_sink(inputs, f, 9, 8);

    const uint64_t vram_bytes = selected.production ? uint64_t(2 * experts) * unit : 2 * unit;
    inputs.set("vram_budget", std::to_string(vram_bytes).c_str());
    set_switch(inputs, "tier.global", selected.production);
    set_switch(inputs, "vram_exclusive", selected.production);
    set_switch(inputs, "vram_async", selected.production);
    set_switch(inputs, "vram_deferred", selected.production);
    set_switch(inputs, "tier.keep_across_requests", true);
    set_switch(inputs, "early_up_gate", selected.production);
    set_switch(inputs, "role_major", selected.production);
    set_switch(inputs, "nonblocking_down_sched", selected.production);
    auto mixed = make_sink(inputs, f, 9, 8);

    const uint64_t ram_bytes = selected.production ? 2 * unit : 0;
    configure_cache(cpu.get(), ram_bytes, {1, 2}, experts);
    configure_cache(mixed.get(), ram_bytes, {1, 2}, experts);
    if (layer1_down != GGML_TYPE_COUNT) {
        // the load step with every routed layer's tensors: the sink declares each layer's types to the tier
        tensors first(f, 1), second(f, 2);
        const ggml_tensor * layers[] = { first.role[0], first.role[1], first.role[2], second.role[0], second.role[1], second.role[2] };
        for (kimi_cpu_sink * sink : { cpu.get(), mixed.get() }) {
            char error[512] = {};
            require(kimi_cpu_sink_vram_reserve(sink, layers, 2, 4, error, sizeof(error)), std::string("reserve: ") + error);
        }
    }
    uint64_t cpu_reads = 0;
    uint64_t mixed_reads = 0;
    const kimi_cpu_sink_activation activation = {
        KIMI_CPU_SINK_ACTIVATION_SWIGLU_CLAMP, 0.0f, 0.0f, 0.75f
    };

    for (int token = 1; token <= 4; ++token) {
        begin_cache(cpu.get(), 91, uint64_t(token));
        begin_cache(mixed.get(), 91, uint64_t(token));
        for (int layer : {1, 2}) {
            const int used = std::min(4, experts);
            std::vector<int32_t> ids;
            std::vector<float> weights;
            for (int i = 0; i < used; ++i) {
                ids.push_back((i + layer - 1) % experts);
                weights.push_back(i == 0 ? 0.4f : 0.6f / float(used - 1));
            }
            std::vector<float> input(static_cast<size_t>(latent));
            for (int i = 0; i < latent; ++i) {
                input[size_t(i)] = 1.7f * std::sin(float(i * 11 + token * 23 + layer * 5));
            }
            std::vector<float> expected(static_cast<size_t>(latent));
            std::vector<float> got(static_cast<size_t>(latent));
            tensors meta(f, layer);
            for (int arm = 0; arm < 2; ++arm) {
                kimi_cpu_sink_stats stats = {};
                char error[512] = {};
                const bool ok = kimi_cpu_sink_run_activation(arm ? mixed.get() : cpu.get(),
                    meta.role[0], meta.role[1], meta.role[2], input.data(), ids.data(), weights.data(),
                    int(ids.size()), &activation, (arm ? got : expected).data(), &stats, error, sizeof(error));
                require(ok, std::string("generic sink run failed mode=") + selected.name +
                    " token=" + std::to_string(token) + " layer=" + std::to_string(layer) +
                    " arm=" + std::to_string(arm) + ": " + error);
                require(stats.read_errors == 0 && stats.pending_reads == 0 && stats.fallback_count == 0,
                    "generic VRAM work did not drain");
                (arm ? mixed_reads : cpu_reads) += stats.read_requests;
                ++checks;
            }
            for (float value : got) {
                require(std::isfinite(value), "generic VRAM output is non-finite");
            }
            const double rel = relative_l2(got, expected);
            const double abs = max_abs_error(got, expected);
            worst_rel = std::max(worst_rel, rel);
            worst_abs = std::max(worst_abs, abs);
            require(rel <= 2e-2, "generic VRAM output exceeds 2e-2 relative L2");
            ++checks;
        }
    }
    require(mixed_reads < cpu_reads, "generic VRAM tier served no resident expert hits");
    ++checks;
    std::printf("GENERIC_VRAM_SHAPE_PASS mode=%s types=%s/%s layer1_down=%s latent=%d hidden=%d cpu_reads=%llu mixed_reads=%llu worst_rel_l2=%.6g worst_abs=%.6g\n",
        selected.name, ggml_type_name(up_type), ggml_type_name(down_type), ggml_type_name(f.down_of(1)), latent, hidden, (unsigned long long) cpu_reads,
        (unsigned long long) mixed_reads, worst_rel, worst_abs);
}

} // namespace

int main() {
    fixture_inputs inputs; inputs.set_runtime_services(test_gpu_services());
    try {
        ggml_cpu_init();
        inputs.set("ready_first", "1");
        inputs.set("cache_policy", "lru");
        inputs.set("cache_allocation", "uniform");
        inputs.set("cache_keep", "1");
        inputs.set("cache_reserve_bytes", "536870912");
        inputs.set("tier.policy", "lru");
        inputs.set("tier.min_count", "0");
        inputs.set("tier.safety_mib", "32");
        inputs.set("tier.elastic", "0");
        size_t checks = 0;
        double worst_rel = 0.0;
        double worst_abs = 0.0;
        run_shape(inputs, 512, 256, 8, {true, "production-generic"}, checks, worst_rel, worst_abs, GGML_TYPE_Q4_K, GGML_TYPE_Q5_K);
        run_shape(inputs, 512, 256, 8, {true, "production-generic"}, checks, worst_rel, worst_abs, GGML_TYPE_Q8_0, GGML_TYPE_Q8_0);
        run_shape(inputs, 512, 256, 8, {true, "production-mixed-layers"}, checks, worst_rel, worst_abs, GGML_TYPE_Q4_K,
            GGML_TYPE_Q5_K, GGML_TYPE_Q4_K);
        run_shape(inputs, 512, 256, 8, {true, "production-generic"}, checks, worst_rel, worst_abs, GGML_TYPE_MXFP4, GGML_TYPE_MXFP4);
        run_host_jobs(inputs, GGML_TYPE_Q8_0, checks);
        run_host_jobs(inputs, GGML_TYPE_MXFP4, checks);
        std::printf("REAL_GPU_GENERIC_TIER_PASS checks=%zu worst_rel_l2=%.6g worst_abs=%.6g\n",
            checks, worst_rel, worst_abs);
        return 0;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "REAL_GPU_GENERIC_TIER_FAIL: %s\n", e.what());
        return 1;
    }
}
