// SPDX-License-Identifier: Apache-2.0
// The GPU lane against the chain computed by hand: weights where the device cannot read them (copied), weights in
// registered host memory (in place), and a failed read, which must end the pass with an error and every other credit back.
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

double multiply(const float * weights, int rows, int columns, const float * x, std::vector<float> & out) {
    out.assign(size_t(rows), 0.0f);
    for (int r = 0; r < rows; ++r) {
        double sum = 0;
        for (int c = 0; c < columns; ++c) { sum += double(weights[size_t(r) * columns + c]) * x[c]; }
        out[size_t(r)] = float(sum);
    }
    return 0;
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
    const auto by_hand = [&](int kind, bool norm = false) {
        std::vector<float> rows(size_t(tokens) * latent, 0.0f), u, g, d, h(hidden);
        const auto silu = [](float v) { return v / (1.0f + std::exp(-v)); };
        for (int token = 0; token < tokens; ++token) {
            const float * x = input.data() + size_t(token) * latent;
            for (int slot = 0; slot < nslots; ++slot) {
                const size_t route = size_t(token) * nslots + slot;
                if (routes[route] < 0) { continue; }
                multiply(up[routes[route]].data(), hidden, latent, x, u);
                multiply(gate[routes[route]].data(), hidden, latent, x, g);
                for (int i = 0; i < hidden; ++i) {
                    if (kind == KIMI_VRAM_ACTIVATION_SWIGLU) { h[i] = silu(g[i]) * u[i]; }
                    else if (kind == KIMI_VRAM_ACTIVATION_SWIGLU_CLAMP) { h[i] = silu(std::min(g[i], limit)) * std::clamp(u[i], -limit, limit); }
                    else { h[i] = beta * std::tanh(g[i] / beta) / (1.0f + std::exp(-g[i])) * linear_beta * std::tanh(u[i] / linear_beta); }
                }
                multiply(down[routes[route]].data(), latent, hidden, h.data(), d);
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
    const auto run = [&](stage_state * stage, std::vector<float> & output, kimi_vram_prefill_stats & stats) {
        std::vector<kimi_vram_prefill_transient> transient(experts);
        for (int e = 0; e < experts; ++e) {
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
                transient.data(), experts, &options, error, sizeof(error))) { fail(error); }
        return provider->prefill->finish(lane, output.data(), latent, &stats, error, sizeof(error));
    };
    const auto compare_with = [&](const std::vector<float> & output, const std::vector<float> & wanted, const char * what) {
        double difference = 0, norm = 0;
        for (size_t i = 0; i < wanted.size(); ++i) {
            const double delta = double(output[i]) - wanted[i];
            difference += delta * delta;
            norm += double(wanted[i]) * wanted[i];
        }
        const double relative = std::sqrt(difference / norm);
        std::printf("%s: relative L2 %.3g\n", what, relative);
        // A device's shader may decode the weights to 16-bit floats (Vulkan: 0.12% here). The CPU's own path quantizes
        // its input and hidden rows to 8 bits, about 1% from this chain.
        if (!(relative < 5e-3)) { fail("the lane's rows differ from the chain by hand"); }
    };

    const auto compare = [&](const std::vector<float> & output, const char * what) { compare_with(output, expected, what); };

    std::vector<float> output;
    kimi_vram_prefill_stats stats = {};
    if (run(nullptr, output, stats)) { fail(error); }
    compare(output, "copied");
    if (stats.ssd_transient_experts != experts) { fail("an expert was not computed"); }

    options.activation = { KIMI_VRAM_ACTIVATION_SWIGLU_CLAMP, 0.0f, 0.0f, limit };
    if (run(nullptr, output, stats)) { fail(error); }
    compare_with(output, by_hand(KIMI_VRAM_ACTIVATION_SWIGLU_CLAMP), "bounded");
    options.activation = { KIMI_VRAM_ACTIVATION_SITU, beta, linear_beta, 0.0f };
    if (run(nullptr, output, stats)) { fail(error); }
    compare_with(output, by_hand(KIMI_VRAM_ACTIVATION_SITU), "situ");
    options.activation = { KIMI_VRAM_ACTIVATION_SWIGLU, 0.0f, 0.0f, 0.0f };
    options.post_norm = norm_weights.data();
    options.post_norm_eps = norm_eps;
    if (run(nullptr, output, stats)) { fail(error); }
    compare_with(output, by_hand(KIMI_VRAM_ACTIVATION_SWIGLU, true), "post norm");
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
    std::printf("GRAPH_LANE_PASS\n");
    return 0;
}
