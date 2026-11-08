// SPDX-License-Identifier: Apache-2.0
// The GPU lane against the chain computed by hand: weights where the device cannot read them (copied), weights in
// registered host memory (in place), and a failed read, which must end the pass with an error and every other credit back.
// Then layers of different formats in one pass, each declared: a larger one after a smaller one, a supported one after
// one the device cannot run.
#include "ggml-backend.h"
#include "ggml-quants.h"
#include "gpu/provider.h"
#include "platform/memory.h"
#include "runtime-services.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace {
constexpr int latent = 320, hidden = 64, experts = 6, tokens = 96, nslots = 3;
constexpr size_t aligned = size_t(1) << 16;

[[noreturn]] void fail(const char * what) {
    std::fprintf(stderr, "test-graph-lane: %s\n", what);
    std::exit(1);
}

struct memory {
    void * data = nullptr;
    size_t bytes = 0;
    explicit memory(size_t n) : bytes((n + aligned - 1) & ~(aligned - 1)) {
        data = ssd::platform::aligned_alloc_bytes(aligned, bytes);
        if (!data) { fail("allocation"); }
    }
    ~memory() { ssd::platform::aligned_free(data); }
    memory(const memory &) = delete;
    memory & operator=(const memory &) = delete;
    char * at(size_t offset) const { return static_cast<char *>(data) + offset; }
};

std::vector<float> normal(size_t n, unsigned seed) {
    std::mt19937 engine(seed);
    std::normal_distribution<float> values(0.0f, 0.5f);
    std::vector<float> out(n);
    for (auto & v : out) { v = values(engine); }
    return out;
}

struct stage_state {
    int failing = -1, released = 0;
    static int call(void * context, int32_t request, int phase) {
        auto & self = *static_cast<stage_state *>(context);
        if (phase == 1) { ++self.released; return 1; }
        return request != self.failing;
    }
};

double multiply(ggml_type weight_type, const float * weights, int rows, int columns, const float * x,
        std::vector<float> & out, bool quantize_activations = false) {
    std::vector<float> quantized_x;
    if (quantize_activations && ggml_is_quantized(weight_type)) {
        std::vector<unsigned char> packed(ggml_row_size(GGML_TYPE_Q8_0, columns));
        ggml_quantize_chunk(GGML_TYPE_Q8_0, x, packed.data(), 0, 1, columns, nullptr);
        quantized_x.resize(columns);
        dequantize_row_q8_0(reinterpret_cast<const block_q8_0 *>(packed.data()), quantized_x.data(), columns);
        x = quantized_x.data();
    }
    out.assign(size_t(rows), 0.0f);
    for (int r = 0; r < rows; ++r) {
        double sum = 0;
        for (int c = 0; c < columns; ++c) { sum += double(weights[size_t(r) * columns + c]) * x[c]; }
        out[size_t(r)] = float(sum);
    }
    return 0;
}

// A pass's rows by hand, SwiGLU experts of these matrices.
std::vector<float> swiglu_rows(const std::vector<std::vector<float>> & up, const std::vector<std::vector<float>> & gate,
        const std::vector<std::vector<float>> & down, const std::vector<float> & input, const std::vector<int32_t> & routes,
        const std::vector<float> & route_weights, bool quantize_activations = false, ggml_type down_type = GGML_TYPE_Q8_0) {
    std::vector<float> rows(size_t(tokens) * latent, 0.0f), u, g, d, h(hidden);
    for (int token = 0; token < tokens; ++token) {
        const float * x = input.data() + size_t(token) * latent;
        for (int slot = 0; slot < nslots; ++slot) {
            const size_t route = size_t(token) * nslots + slot;
            if (routes[route] < 0) { continue; }
            multiply(GGML_TYPE_Q8_0, up[routes[route]].data(), hidden, latent, x, u, quantize_activations);
            multiply(GGML_TYPE_Q8_0, gate[routes[route]].data(), hidden, latent, x, g, quantize_activations);
            for (int i = 0; i < hidden; ++i) { h[i] = g[i] / (1.0f + std::exp(-g[i])) * u[i]; }
            multiply(down_type, down[routes[route]].data(), latent, hidden, h.data(), d, quantize_activations);
            for (int i = 0; i < latent; ++i) { rows[size_t(token) * latent + i] += d[i] * route_weights[route]; }
        }
    }
    return rows;
}
} // namespace

int main() {
    ggml_backend_load_all();
    const ssd::gpu::provider_v1 * provider = nullptr;
    ggml_backend_dev_t device = nullptr;
    int ordinal = -1;
    for (size_t r = 0; r < ggml_backend_reg_count() && !provider; ++r) {
        ggml_backend_reg_t reg = ggml_backend_reg_get(r);
        const auto get = reinterpret_cast<ssd::gpu::get_provider_v1>(ggml_backend_reg_get_proc_address(reg, "ggml_backend_ssd_provider_v1"));
        if (!get || get()->transport) { continue; }  // the CUDA tier has its own tests
        for (size_t d = 0; d < ggml_backend_reg_dev_count(reg); ++d) {
            ggml_backend_dev_t candidate = ggml_backend_reg_dev_get(reg, d);
            ggml_backend_t backend = ggml_backend_dev_init(candidate, nullptr);
            if (!backend) { continue; }
            const int number = get()->device_ordinal(backend);
            ggml_backend_free(backend);
            if (number >= 0) { provider = get(); device = candidate; ordinal = number; break; }
        }
    }
    if (!provider) {
        std::printf("GRAPH_LANE_SKIP no device the lane serves\n");
        return 0;
    }
    std::printf("device %s\n", ggml_backend_dev_description(device));

    const size_t up_bytes = ggml_row_size(GGML_TYPE_Q8_0, latent) * hidden;
    const size_t down_bytes = ggml_row_size(GGML_TYPE_Q8_0, hidden) * latent;
    const size_t expert_bytes = 2 * up_bytes + down_bytes;
    memory weights(experts * expert_bytes);
    std::vector<std::vector<float>> up(experts), gate(experts), down(experts);
    for (int e = 0; e < experts; ++e) {
        up[e] = normal(size_t(latent) * hidden, 100 + e);
        gate[e] = normal(size_t(latent) * hidden, 200 + e);
        down[e] = normal(size_t(latent) * hidden, 300 + e);
        char * base = weights.at(size_t(e) * expert_bytes);
        ggml_quantize_chunk(GGML_TYPE_Q8_0, up[e].data(), base, 0, hidden, latent, nullptr);
        ggml_quantize_chunk(GGML_TYPE_Q8_0, gate[e].data(), base + up_bytes, 0, hidden, latent, nullptr);
        ggml_quantize_chunk(GGML_TYPE_Q8_0, down[e].data(), base + 2 * up_bytes, 0, latent, hidden, nullptr);
        // What the device multiplies: the quantized values.
        const auto decoded = [&](const char * q, std::vector<float> & out, int rows, int columns) {
            for (int row = 0; row < rows; ++row) {
                dequantize_row_q8_0(reinterpret_cast<const block_q8_0 *>(q + size_t(row) * ggml_row_size(GGML_TYPE_Q8_0, columns)),
                    out.data() + size_t(row) * columns, columns);
            }
        };
        decoded(base, up[e], hidden, latent);
        decoded(base + up_bytes, gate[e], hidden, latent);
        decoded(base + 2 * up_bytes, down[e], latent, hidden);
    }
    const auto input = normal(size_t(tokens) * latent, 7);
    std::vector<int32_t> routes(size_t(tokens) * nslots);
    std::vector<float> route_weights(routes.size());
    {
        std::mt19937 engine(11);
        for (size_t i = 0; i < routes.size(); ++i) {
            // Skewed towards the first experts, with a duplicate route of a token now and then and a skipped one.
            const unsigned draw = engine();
            routes[i] = draw % 17 == 0 ? -1 : int32_t(std::min<unsigned>(draw % 12, draw % experts));
            route_weights[i] = float(draw % 100) / 100.0f;
        }
    }
    const float limit = 1.0f, beta = 1.5f, linear_beta = 2.0f;
    const auto norm_weights = normal(latent, 13);
    const float norm_eps = 1e-5f;
    const auto by_hand = [&](int kind, bool norm = false, bool quantize_activations = false) {
        std::vector<float> rows(size_t(tokens) * latent, 0.0f), u, g, d, h(hidden);
        const auto silu = [](float v) { return v / (1.0f + std::exp(-v)); };
        for (int token = 0; token < tokens; ++token) {
            const float * x = input.data() + size_t(token) * latent;
            for (int slot = 0; slot < nslots; ++slot) {
                const size_t route = size_t(token) * nslots + slot;
                if (routes[route] < 0) { continue; }
                multiply(GGML_TYPE_Q8_0, up[routes[route]].data(), hidden, latent, x, u, quantize_activations);
                multiply(GGML_TYPE_Q8_0, gate[routes[route]].data(), hidden, latent, x, g, quantize_activations);
                for (int i = 0; i < hidden; ++i) {
                    if (kind == KIMI_VRAM_ACTIVATION_SWIGLU) { h[i] = silu(g[i]) * u[i]; }
                    else if (kind == KIMI_VRAM_ACTIVATION_SWIGLU_CLAMP) { h[i] = silu(std::min(g[i], limit)) * std::clamp(u[i], -limit, limit); }
                    else { h[i] = beta * std::tanh(g[i] / beta) / (1.0f + std::exp(-g[i])) * linear_beta * std::tanh(u[i] / linear_beta); }
                }
                multiply(GGML_TYPE_Q8_0, down[routes[route]].data(), latent, hidden, h.data(), d, quantize_activations);
                if (norm) {
                    double sum = 0;
                    for (int i = 0; i < latent; ++i) { d[i] *= norm_weights[i]; sum += double(d[i]) * d[i]; }
                    const float scale = 1.0f / std::sqrt(float(sum / latent) + norm_eps);
                    for (int i = 0; i < latent; ++i) { d[i] *= scale; }
                }
                for (int i = 0; i < latent; ++i) { rows[size_t(token) * latent + i] += d[i] * route_weights[route]; }
            }
        }
        return rows;
    };
    const auto expected = by_hand(KIMI_VRAM_ACTIVATION_SWIGLU);
    const auto expected_q8 = by_hand(KIMI_VRAM_ACTIVATION_SWIGLU, false, true);

    ssd::runtime_services services;
    services.gpu = provider;
    services.gpu_device = ordinal;
    char error[512] = {};
    void * lane = provider->settings->create(nullptr, &services, error, sizeof(error));
    if (!lane) { fail(error); }
    kimi_vram_host_costs costs = {};
    provider->host->costs2(lane, &costs);
    if (!costs.in_place || !costs.in_place_only) { fail("the lane must say it reads in place, and nothing else"); }

    std::vector<kimi_vram_request> requests(experts);
    std::vector<int> slots(experts, 7);
    const int32_t layers[] = { 0 };
    const kimi_vram_expert_format format = { latent, hidden, GGML_TYPE_Q8_0, GGML_TYPE_Q8_0, experts, 1, layers };
    if (provider->prefill->prepare(lane, requests.data(), experts, &format, slots.data(), error, sizeof(error))) { fail(error); }
    for (int slot : slots) { if (slot != -1) { fail("the lane holds nothing across layers"); } }
    kimi_vram_prefill_lane info = {};
    provider->prefill->lane(lane, &info);
    if (info.engine != KIMI_VRAM_PREFILL_ENGINE_BATCHED) {
        std::printf("GRAPH_LANE_SKIP the device has no such product\n");
        provider->api->destroy(lane);
        return 0;
    }
    if (provider->prefill->transient_available(lane)) { fail("no expert may move before the pass reserved its memory"); }
    if (provider->elastic->reserve_accumulate(lane, tokens, error, sizeof(error))) { fail(error); }
    if (!provider->prefill->transient_available(lane)) { fail("the pass's memory was not reserved"); }

    kimi_vram_prefill_launch options = { { KIMI_VRAM_ACTIVATION_SWIGLU, 0.0f, 0.0f, 0.0f }, nullptr, 0.0f, 1 };
    const auto run = [&](stage_state * stage, std::vector<float> & output, kimi_vram_prefill_stats & stats, int count = experts) {
        std::vector<kimi_vram_prefill_transient> transient(count);
        for (int e = 0; e < count; ++e) {
            auto & t = transient[e];
            t.request_index = e;
            t.source = stage ? KIMI_VRAM_PREFILL_TRANSIENT_SSD : KIMI_VRAM_PREFILL_TRANSIENT_RAM;
            const char * base = weights.at(size_t(e) * expert_bytes);
            t.role[0] = base; t.role[1] = base + up_bytes; t.role[2] = base + 2 * up_bytes;
            t.stage = stage ? stage_state::call : nullptr;
            t.stage_context = stage;
        }
        output.assign(size_t(tokens) * latent, 0.0f);
        if (provider->prefill->launch(lane, input.data(), size_t(latent) * sizeof(float), routes.data(), route_weights.data(), nslots, tokens,
                transient.data(), count, &options, error, sizeof(error))) { fail(error); }
        return provider->prefill->finish(lane, output.data(), latent, &stats, error, sizeof(error));
    };
    const auto compare_with = [&](const std::vector<float> & output, const std::vector<float> & wanted,
            const std::vector<float> & wanted_q8, const char * what) {
        const auto relative_to = [&](const std::vector<float> & reference) {
            double difference = 0, norm = 0;
            for (size_t i = 0; i < reference.size(); ++i) {
                const double delta = double(output[i]) - reference[i];
                difference += delta * delta;
                norm += double(reference[i]) * reference[i];
            }
            return std::sqrt(difference / norm);
        };
        const double floating = relative_to(wanted), quantized = relative_to(wanted_q8);
        std::printf("%s: relative L2 floating %.3g, Q8 activations %.3g\n", what, floating, quantized);
        // Quantized weight kernels may read floating activations or quantize each input/hidden row for integer dots.
        // Both references use the stored, dequantized weights; unquantized weight formats never quantize activations.
        if (!(std::min(floating, quantized) < 5e-3)) { fail("the lane's rows differ from the chain by hand"); }
    };

    const auto compare = [&](const std::vector<float> & output, const char * what) { compare_with(output, expected, expected_q8, what); };

    std::vector<float> output;
    kimi_vram_prefill_stats stats = {};
    {
        const auto original_routes = routes;
        for (int row_count : {2, 3, 9}) {
            for (int token = 0; token < tokens; ++token) {
                for (int slot = 0; slot < nslots; ++slot) {
                    routes[size_t(token) * nslots + slot] = slot == 0 && token < row_count ? 0 : -1;
                }
            }
            if (row_count == 2) {
                stage_state failing_first;
                failing_first.failing = 0;
                if (!run(&failing_first, output, stats, 1)) { fail("the first failed read must fail the pass"); }
                if (stats.host_fit_experts != 0 || stats.ssd_transient_experts != 0) { fail("failed work entered calibration"); }
            }
            if (run(nullptr, output, stats, 1)) { fail(error); }
            const auto cold_output = output;
            for (float value : output) { if (!std::isfinite(value)) { fail("cold graph output is not finite"); } }
            if (stats.host_fit_experts != 0 || stats.host_fit_us != 0 || stats.ssd_transient_experts != 1 ||
                    stats.ssd_transient_routes != uint64_t(row_count) || !(stats.total_worker_us > 0)) {
                fail("cold work must execute and retain wall time without a fit observation");
            }
            if (run(nullptr, output, stats, 1)) { fail(error); }
            if (output != cold_output) { fail("calibration changed the graph's output"); }
            if (stats.host_fit_experts != 1 || !(stats.host_fit_us > 0) || stats.ssd_transient_routes != uint64_t(row_count)) {
                fail("the next successful graph must enter calibration");
            }
            std::printf("CALIBRATION_SHAPE_PASS rows=%d cold_observations=0 warm_observations=1\n", row_count);
        }
        routes = original_routes;
    }
    if (run(nullptr, output, stats)) { fail(error); }
    compare(output, "copied");
    if (stats.ssd_transient_experts != experts) { fail("an expert was not computed"); }

    options.activation = { KIMI_VRAM_ACTIVATION_SWIGLU_CLAMP, 0.0f, 0.0f, limit };
    if (run(nullptr, output, stats)) { fail(error); }
    compare_with(output, by_hand(KIMI_VRAM_ACTIVATION_SWIGLU_CLAMP), by_hand(KIMI_VRAM_ACTIVATION_SWIGLU_CLAMP, false, true), "bounded");
    options.activation = { KIMI_VRAM_ACTIVATION_SITU, beta, linear_beta, 0.0f };
    if (run(nullptr, output, stats)) { fail(error); }
    compare_with(output, by_hand(KIMI_VRAM_ACTIVATION_SITU), by_hand(KIMI_VRAM_ACTIVATION_SITU, false, true), "situ");
    options.activation = { KIMI_VRAM_ACTIVATION_SWIGLU, 0.0f, 0.0f, 0.0f };
    options.post_norm = norm_weights.data();
    options.post_norm_eps = norm_eps;
    if (run(nullptr, output, stats)) { fail(error); }
    compare_with(output, by_hand(KIMI_VRAM_ACTIVATION_SWIGLU, true), by_hand(KIMI_VRAM_ACTIVATION_SWIGLU, true, true), "post norm");
    options.post_norm = nullptr;

    if (provider->host->register_host(lane, weights.data, weights.bytes, error, sizeof(error))) {
        std::printf("the device refused this memory: %s\n", error);
    } else {
        stage_state stage;
        if (run(&stage, output, stats)) { fail(error); }
        compare(output, "in place");
        if (stage.released != experts) { fail("a ring credit did not come back"); }

        stage_state failing;
        failing.failing = 2;
        if (!run(&failing, output, stats)) { fail("a failed read must fail the pass"); }
        if (failing.released != experts - 1) { fail("a failed read must leave every other credit returned"); }
        if (provider->host->unregister_host(lane, weights.data, error, sizeof(error))) { fail(error); }
    }

    if (provider->elastic->phase_complete(lane, 1, error, sizeof(error))) { fail(error); }
    if (provider->prefill->transient_available(lane)) { fail("decode gave the memory back: no expert may move"); }
    provider->api->destroy(lane);

    // Layers of three formats in a pass, declared per layer (kimi_vram_expert_format): layer 0 of a type this device
    // does not multiply, layer 1 with a Q4_0 down, layer 2 with the larger Q8_0 one (the experts above). As the sink does,
    // each layer's pass memory is reserved before the layer is prepared, the previous layer's format still the prepared
    // one: a layer the device runs right after one it does not has the lane, and the larger layer after the smaller
    // fits the memory reserved then.
    {
        // a weight type the device refuses to multiply, by its own answer
        ggml_type refused = GGML_TYPE_COUNT;
        for (ggml_type type : { GGML_TYPE_I32, GGML_TYPE_I16, GGML_TYPE_I8, GGML_TYPE_I64, GGML_TYPE_F64 }) {
            ggml_init_params params{ ggml_tensor_overhead() * 4, nullptr, true };
            ggml_context * ctx = ggml_init(params);
            ggml_tensor * product = ggml_mul_mat(ctx, ggml_new_tensor_2d(ctx, type, latent, hidden),
                ggml_new_tensor_2d(ctx, GGML_TYPE_F32, latent, 2));
            const bool runs = ggml_backend_dev_supports_op(device, product);
            ggml_free(ctx);
            if (!runs) { refused = type; break; }
        }
        if (refused == GGML_TYPE_COUNT) { std::printf("the device multiplies every integer weight type: layer 0 runs too\n"); }
        const ggml_type unrun = refused == GGML_TYPE_COUNT ? GGML_TYPE_Q8_0 : refused;
        const size_t small_down = ggml_row_size(GGML_TYPE_Q4_0, hidden) * latent, small_bytes = 2 * up_bytes + small_down;
        memory small(experts * small_bytes);
        std::vector<std::vector<float>> small_down_rows(experts);
        for (int e = 0; e < experts; ++e) {
            char * base = small.at(size_t(e) * small_bytes);
            std::memcpy(base, weights.at(size_t(e) * expert_bytes), 2 * up_bytes); // the same up and gate
            const auto values = normal(size_t(latent) * hidden, 400 + e);
            ggml_quantize_chunk(GGML_TYPE_Q4_0, values.data(), base + 2 * up_bytes, 0, latent, hidden, nullptr);
            small_down_rows[e].resize(values.size());
            for (int row = 0; row < latent; ++row) {
                dequantize_row_q4_0(reinterpret_cast<const block_q4_0 *>(base + 2 * up_bytes + size_t(row) * ggml_row_size(GGML_TYPE_Q4_0, hidden)),
                    small_down_rows[e].data() + size_t(row) * hidden, hidden);
            }
        }
        const auto small_expected = swiglu_rows(up, gate, small_down_rows, input, routes, route_weights, false, GGML_TYPE_Q4_0);
        const auto small_expected_q8 = swiglu_rows(up, gate, small_down_rows, input, routes, route_weights, true, GGML_TYPE_Q4_0);
        void * mixed = provider->settings->create(nullptr, &services, error, sizeof(error));
        if (!mixed) { fail(error); }
        const int32_t layer_ids[] = { 0, 1, 2 };
        const int32_t up_types[] = { unrun, GGML_TYPE_Q8_0, GGML_TYPE_Q8_0 };
        const int32_t down_types[] = { unrun, GGML_TYPE_Q4_0, GGML_TYPE_Q8_0 };
        options = { { KIMI_VRAM_ACTIVATION_SWIGLU, 0.0f, 0.0f, 0.0f }, nullptr, 0.0f, 1 };
        // the sink's order for one layer: the pass reserved, the layer prepared, its lane read; true: the lane takes it
        const auto prepare_layer = [&](int layer) {
            const kimi_vram_expert_format declared = { latent, hidden, up_types[layer], down_types[layer], experts, 3,
                layer_ids, up_types, down_types };
            if (provider->elastic->reserve_accumulate(mixed, tokens, error, sizeof(error))) { fail(error); }
            if (provider->prefill->prepare(mixed, requests.data(), experts, &declared, slots.data(), error, sizeof(error))) { fail(error); }
            kimi_vram_prefill_lane layer_lane = {};
            provider->prefill->lane(mixed, &layer_lane);
            const bool taken = layer_lane.engine == KIMI_VRAM_PREFILL_ENGINE_BATCHED;
            if (taken != (provider->prefill->transient_available(mixed) != 0)) {
                fail("a layer's lane and its pass memory disagree");
            }
            return taken;
        };
        const auto run_layer = [&](const memory & set, size_t set_bytes, std::vector<float> & output) {
            std::vector<kimi_vram_prefill_transient> transient(experts);
            for (int e = 0; e < experts; ++e) {
                const char * base = set.at(size_t(e) * set_bytes);
                transient[e] = { e, KIMI_VRAM_PREFILL_TRANSIENT_RAM, { base, base + up_bytes, base + 2 * up_bytes }, nullptr, nullptr };
            }
            output.assign(size_t(tokens) * latent, 0.0f);
            kimi_vram_prefill_stats layer_stats = {};
            if (provider->prefill->launch(mixed, input.data(), size_t(latent) * sizeof(float), routes.data(), route_weights.data(),
                    nslots, tokens, transient.data(), experts, &options, error, sizeof(error))) { fail(error); }
            if (provider->prefill->finish(mixed, output.data(), latent, &layer_stats, error, sizeof(error))) { fail(error); }
            if (layer_stats.ssd_transient_experts != experts) { fail("an expert of a layer was not computed"); }
        };
        // An earlier layer taught the lane the declared formats, as the sink's first layer does: the first layer's pass
        // is reserved before any format is known, and stays on the CPU.
        {
            const kimi_vram_expert_format first = { latent, hidden, up_types[1], down_types[1], experts, 3, layer_ids,
                up_types, down_types };
            if (provider->prefill->prepare(mixed, requests.data(), experts, &first, slots.data(), error, sizeof(error))) { fail(error); }
        }
        if (prepare_layer(0) != (refused == GGML_TYPE_COUNT)) { fail("the lane took a layer of a type the device refuses"); }
        if (!prepare_layer(1)) { fail("the layer after one the device cannot run did not get the lane"); }
        run_layer(small, small_bytes, output);
        compare_with(output, small_expected, small_expected_q8, "layer of the smaller format");
        if (!prepare_layer(2)) { fail("the layer of the larger format did not get the lane"); }
        run_layer(weights, expert_bytes, output);
        compare_with(output, expected, expected_q8, "layer of the larger format after the smaller");
        if (provider->elastic->phase_complete(mixed, 1, error, sizeof(error))) { fail(error); }
        provider->api->destroy(mixed);
    }
    std::printf("GRAPH_LANE_PASS\n");
    return 0;
}
