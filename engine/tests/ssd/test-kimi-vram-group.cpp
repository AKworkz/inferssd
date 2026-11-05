#include "gpu-services.h"
#include "explicit-fixture-inputs.h"
// The grouped prefill engine of the VRAM tier (ggml-cuda/ssd/group-compute-adapter.cu) against the CPU lane
// (kimi_cpu_prefill_run_activation) on the same packed experts.
//
// Default: three expert formats, 16 experts, top 8, at 1, 33 and 300 tokens (one expert takes every token of the 300,
// so its chain runs in two chunks). Q4_K up/gate and Q5_K down, as Muse Spark UD-Q4_K_XL, at a reduced shape (latent
// 512, hidden 512): SwiGLU unclamped and clamped, SiTU, with and without the per-expert post norm (Muse's epsilon 1e-8,
// and 1e-5). The same at hidden 768, whose down operand has a padded tail. Q8_0 up/gate and Q4_0 down at hidden 1408,
// whose down rows are not a whole number of MMQ iterations (MMVQ and the GEMM serve them). Both lanes are measured
// against the layer in doubles from the dequantized experts (each lane rounds its activations: Q8_K or Q8_0 on the CPU,
// Q8_1 on the GPU), per token: the engine's NMSE is at most 5e-4 and at most 1.5x the CPU lane's; the CPU lane's is at
// most 1e-3 and the lanes are at most 1.5e-3 apart, so the independent lane still checks what the engine and the exact
// layer both compute. Even experts compute from resident device copies; odd ones from a two-slot device arena that an
// upload stream refills, each chain waiting for its upload's event and each refill for the previous chain's done event,
// as a pipelined lane would. Where the GPU reads pageable host memory in place, two cases read every expert from it.
// Two cases split the experts between the lanes: the engine gets only the tokens with GPU routes and the sum of both
// lanes is checked. One engine serves a format's cases: each case reserves for half its rows and then for all (the
// engine grows), and releases at its end.
// Memory: what the engine must guarantee is that it gives back what it took, so that is checked from what it owns, its
// own count (device_bytes, the products' pool counters included): it does not grow after reserve, and release takes it
// to 0. The device's free memory (cudaMemGetInfo) says the same only where it counts the device's own memory and
// nothing else moves it: a discrete GPU the test owns. There it does not change between reserve and release (reserve
// loads the kernels) and release gives back what reserve took. On an integrated GPU (cudaDevAttrIntegrated, as GB10)
// it is the system's free memory: other processes and the page cache move it, and so does the driver's own memory for
// a newly mapped address range, which cudaFree leaves in place (on GB10 448 KiB when a buffer lands on a range not
// mapped before, bounded over 400 cycles); there it is printed, not checked.
//
// --time: each lane's time per expert at the given shape and types (default Muse's: latent 4096, hidden 4096,
// Q4_K/Q5_K) at 8, 26, 150 and 300 routes, and a fixed + per-route line fitted to each: the GPU chain from the engine's
// CUDA events, with the weights in device memory and, where the GPU reads pageable host memory in place, in host
// memory; the CPU lane on the sink's participants (--threads counts them and the I/O thread, as
// GGML_KIMI_CPU_SINK_THREADS does). Each lane is timed idle and busy, busy meaning with what else runs in a pass: for
// the GPU the CPU lane and, for weights in device memory, the link copying one expert after another; for the CPU the
// GPU lane and its link. One expert's copy from page-locked host memory is timed too. With --drive-gbps R the busy
// chain at 150 routes is compared with 0.8 x B_e / R, and where the GPU reads host memory in place, in place is chosen
// when its busy chain is no longer than the longer of the copy and the chain from device memory. Needs an owned GPU and
// the tier DSO preloaded; not registered with ctest.
#define main original_sink_test_main
#include "test-kimi-cpu-sink.cpp"
#undef main
#include "vram-compute-api.h"

#include <cuda_runtime.h>
#include <strings.h>

#include <atomic>

namespace {
void cuda(cudaError_t rc, const char * what) {
    if (rc != cudaSuccess) { throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(rc)); }
}

const vram_group_compute_api * engine_api() {
    const auto * api = kimi_vram_group_compute_get_api();
    require(api && api->version == 3 && api->size == sizeof(vram_group_compute_api), "grouped engine API version");
    return api;
}

struct engine_ptr {
    const vram_group_compute_api * api;
    void * p = nullptr;
    engine_ptr(const vram_group_compute_api * a, int latent, int hidden, ggml_type up, ggml_type down) : api(a) {
        char error[512] = {};
        const int32_t weight_type = up, down_type = down;
        p = api->create(0, latent, hidden, 1, &weight_type, &down_type, error, sizeof(error));
        require(p, std::string("grouped engine create: ") + error);
    }
    ~engine_ptr() { api->destroy(p); }
    void check(int rc, const char * error) const { require(rc == 0, error); }
};

// One expert's three roles, back to back, as in a tier slot.
std::vector<uint8_t> expert_bytes(const fixture & f, int layer, int expert, size_t role_bytes[3]) {
    std::vector<uint8_t> out;
    for (int role = 0; role < 3; ++role) {
        const auto & e = f.entries.at({ tensor_name(layer, role), expert });
        role_bytes[role] = e.raw_bytes;
        out.insert(out.end(), e.data.begin(), e.data.begin() + e.raw_bytes);
    }
    return out;
}

double token_nmse(const float * got, const float * expected, int latent) {
    double error = 0.0, norm = 0.0;
    for (int i = 0; i < latent; ++i) {
        const double d = double(got[i]) - double(expected[i]);
        error += d * d;
        norm += double(expected[i]) * double(expected[i]);
    }
    return norm > 0.0 ? error / norm : error;
}

// The activation of a case, for both lanes and the exact layer.
struct case_activation {
    int kind;          // KIMI_VRAM_ACTIVATION_* (the CPU sink's values are the same)
    float situ_beta, situ_linear_beta, swiglu_limit;
};

double activate_exact(const case_activation & a, double up, double gate) {
    if (a.kind == KIMI_VRAM_ACTIVATION_SWIGLU_CLAMP) {
        const double g = std::min(gate, double(a.swiglu_limit)), u = std::min(std::max(up, -double(a.swiglu_limit)), double(a.swiglu_limit));
        return g / (1.0 + std::exp(-g)) * u;
    }
    const double s = a.situ_beta * std::tanh(gate / a.situ_beta) / (1.0 + std::exp(-gate));
    return s * (a.situ_linear_beta > 0.0f ? a.situ_linear_beta * std::tanh(up / a.situ_linear_beta) : up);
}

// The layer in doubles from the dequantized experts: what both lanes approximate.
std::vector<float> exact_layer(const fixture & f, int layer, int tokens, int nslots, const std::vector<float> & input,
        const std::vector<int32_t> & ids, const std::vector<float> & weights, const case_activation & activation,
        const float * norm, float eps) {
    const int latent = f.latent, hidden = f.hidden;
    std::vector<std::vector<float>> w(size_t(f.experts) * 3);
    for (int e = 0; e < f.experts; ++e) {
        for (int role = 0; role < 3; ++role) {
            const ggml_type type = role == 2 ? f.down_type : f.type;
            const int cols = role == 2 ? hidden : latent, rows = role == 2 ? latent : hidden;
            auto & m = w[size_t(e) * 3 + role];
            m.resize(size_t(rows) * cols);
            const auto & src = f.entries.at({ tensor_name(layer, role), e }).data;
            const size_t row_bytes = ggml_row_size(type, cols);
            for (int r = 0; r < rows; ++r) {
                ggml_get_type_traits(type)->to_float(src.data() + size_t(r) * row_bytes, m.data() + size_t(r) * cols, cols);
            }
        }
    }
    std::vector<float> out(size_t(tokens) * latent, 0.0f);
    std::vector<double> h(static_cast<size_t>(hidden)), y(static_cast<size_t>(latent)), acc(static_cast<size_t>(latent));
    for (int t = 0; t < tokens; ++t) {
        std::fill(acc.begin(), acc.end(), 0.0);
        const float * x = input.data() + size_t(t) * latent;
        for (int s = 0; s < nslots; ++s) {
            const int e = ids[size_t(t) * nslots + s];
            if (e < 0) { continue; }
            const auto & up = w[size_t(e) * 3], & gate = w[size_t(e) * 3 + 1], & down = w[size_t(e) * 3 + 2];
            for (int r = 0; r < hidden; ++r) {
                double a = 0, b = 0;
                for (int k = 0; k < latent; ++k) { a += double(up[size_t(r) * latent + k]) * x[k]; b += double(gate[size_t(r) * latent + k]) * x[k]; }
                h[size_t(r)] = activate_exact(activation, a, b);
            }
            double sum = 0.0;
            for (int r = 0; r < latent; ++r) {
                double v = 0;
                for (int k = 0; k < hidden; ++k) { v += double(down[size_t(r) * hidden + k]) * h[size_t(k)]; }
                if (norm) { v *= norm[r]; }
                y[size_t(r)] = v;
                sum += v * v;
            }
            const double scale = norm ? 1.0 / std::sqrt(sum / latent + eps) : 1.0;
            for (int r = 0; r < latent; ++r) { acc[size_t(r)] += y[size_t(r)] * scale * weights[size_t(t) * nslots + s]; }
        }
        for (int r = 0; r < latent; ++r) { out[size_t(t) * latent + r] = float(acc[size_t(r)]); }
    }
    return out;
}

struct grouped_case {
    const char * name;
    int tokens;
    case_activation activation;
    bool norm;
    float eps;       // the post norm's epsilon (Muse's is 1e-8)
    bool split;      // experts 0..7 on the engine, 8..15 on the CPU lane
    bool host;       // every expert read in place from pageable host memory (only where the GPU reads it)
};

// One expert format and shape, its cases, and one engine reused over them (reserve, release, reserve again).
struct grouped_fixture {
    const char * name;
    ggml_type up, down;
    int latent, hidden;
    std::vector<grouped_case> cases;
};

size_t device_free() {
    size_t free = 0, total = 0;
    cuda(cudaMemGetInfo(&free, &total), "cudaMemGetInfo");
    return free;
}

// Whether the device's free memory counts only the device's own memory (a discrete GPU), so that it can tell what the
// engine took and gave back.
bool device_free_is_own() {
    static const bool own = [] {
        int device = 0, integrated = 0;
        cuda(cudaGetDevice(&device), "cudaGetDevice");
        cuda(cudaDeviceGetAttribute(&integrated, cudaDevAttrIntegrated, device), "cudaDevAttrIntegrated");
        return integrated == 0;
    }();
    return own;
}

// The engine's half of a layer: the experts in `gpu` (others skipped), the tokens with GPU routes as compact rows.
// Even experts read their resident device copy and odd ones go through the two-slot arena; with `host`, every expert
// is read in place from pageable host memory. Each case reserves for half its rows and routes first and then for all
// of them, so that the engine grows, as a pass whose first reserve was too small would.
std::vector<float> run_engine(const vram_group_compute_api * api, engine_ptr & engine, bool first_use, const fixture & f,
        int layer, int tokens, int nslots, const std::vector<float> & input, const std::vector<int32_t> & ids,
        const std::vector<float> & weights, const std::vector<bool> & gpu, const vram_group_activation & activation, bool host,
        size_t & checks) {
    const int latent = f.latent;
    size_t role_bytes[3] = {};
    std::vector<std::vector<uint8_t>> experts(size_t(f.experts));
    for (int e = 0; e < f.experts; ++e) { experts[size_t(e)] = expert_bytes(f, layer, e, role_bytes); }
    const size_t stride = role_bytes[0] + role_bytes[1] + role_bytes[2];
    // resident copies, the pinned host sources of the arena and the arena itself; or the pageable host copies
    void * resident = nullptr, * arena = nullptr, * pinned = nullptr;
    std::vector<uint8_t> pageable;
    if (host) {
        pageable.resize(stride * f.experts);
        for (int e = 0; e < f.experts; ++e) { std::memcpy(pageable.data() + size_t(e) * stride, experts[size_t(e)].data(), stride); }
    } else {
        cuda(cudaMalloc(&resident, stride * f.experts), "resident experts");
        cuda(cudaMalloc(&arena, 2 * stride), "arena");
        cuda(cudaMallocHost(&pinned, stride * f.experts), "pinned experts");
        for (int e = 0; e < f.experts; ++e) {
            std::memcpy(static_cast<char *>(pinned) + size_t(e) * stride, experts[size_t(e)].data(), stride);
            cuda(cudaMemcpy(static_cast<char *>(resident) + size_t(e) * stride, experts[size_t(e)].data(), stride, cudaMemcpyHostToDevice),
                "resident H2D");
        }
    }
    // compact rows and routes grouped by expert, rows ascending
    std::vector<int> row_of(size_t(tokens), -1), token_of;
    std::vector<std::vector<vram_group_route>> by_expert(size_t(f.experts));
    for (int t = 0; t < tokens; ++t) {
        for (int s = 0; s < nslots; ++s) {
            const int e = ids[size_t(t) * nslots + s];
            if (e < 0 || !gpu[size_t(e)]) { continue; }
            if (row_of[size_t(t)] < 0) { row_of[size_t(t)] = int(token_of.size()); token_of.push_back(t); }
            auto & list = by_expert[size_t(e)];
            const int row = row_of[size_t(t)];
            if (!list.empty() && list.back().row == row) { list.back().weight += weights[size_t(t) * nslots + s]; }
            else { list.push_back({ row, weights[size_t(t) * nslots + s] }); }
        }
    }
    std::vector<float> output(size_t(tokens) * latent, 0.0f);
    const int rows = int(token_of.size());
    if (!rows) { return output; }
    std::vector<vram_group_route> routes;
    std::vector<int32_t> first = { 0 };
    std::vector<int> group_expert;
    for (int e = 0; e < f.experts; ++e) {
        if (by_expert[size_t(e)].empty()) { continue; }
        routes.insert(routes.end(), by_expert[size_t(e)].begin(), by_expert[size_t(e)].end());
        first.push_back(int32_t(routes.size()));
        group_expert.push_back(e);
    }
    cudaStream_t upload = nullptr;
    cuda(cudaStreamCreateWithFlags(&upload, cudaStreamNonBlocking), "upload stream");
    cudaEvent_t ready[2] = {}, slot_done[2] = {};
    for (auto & e : ready) { cuda(cudaEventCreateWithFlags(&e, cudaEventDisableTiming), "event"); }
    const size_t free_before = device_free();
    char error[512] = {};
    require(api->device_bytes(engine.p) == 0, "the engine holds device memory before reserve");
    engine.check(api->reserve(engine.p, rows / 2 + 1, int(routes.size()) / 2 + 1, error, sizeof(error)), error);
    const size_t first_reserved = api->device_bytes(engine.p);
    engine.check(api->reserve(engine.p, rows, int(routes.size()), error, sizeof(error)), error);
    const size_t reserved = api->device_bytes(engine.p);
    require(reserved > first_reserved, "the second reserve did not grow the engine");
    const size_t free_reserved = device_free();
    float * staging = api->input_rows(engine.p);
    require(staging, "engine input rows");
    for (int r = 0; r < rows; ++r) {
        std::memcpy(staging + size_t(r) * latent, input.data() + size_t(token_of[size_t(r)]) * latent, size_t(latent) * sizeof(float));
    }
    engine.check(api->begin_layer(engine.p, 0, rows, routes.data(), first.data(), int(group_expert.size()), &activation,
        error, sizeof(error)), error);
    int arena_uses = 0;
    for (size_t g = 0; g < group_expert.size(); ++g) {
        const int e = group_expert[g];
        vram_compute_expert w = {};
        CUevent_st * wait = nullptr;
        if (host) {
            const uint8_t * base = pageable.data() + size_t(e) * stride;
            w = { base, base + role_bytes[0], base + role_bytes[0] + role_bytes[1] };
        } else if (e % 2 == 0) {
            const char * base = static_cast<const char *>(resident) + size_t(e) * stride;
            w = { base, base + role_bytes[0], base + role_bytes[0] + role_bytes[1] };
        } else {
            const int slot = arena_uses++ % 2;
            if (slot_done[slot]) { cuda(cudaStreamWaitEvent(upload, slot_done[slot], 0), "upload wait"); }
            char * base = static_cast<char *>(arena) + size_t(slot) * stride;
            cuda(cudaMemcpyAsync(base, static_cast<const char *>(pinned) + size_t(e) * stride, stride, cudaMemcpyHostToDevice, upload),
                "arena upload");
            cuda(cudaEventRecord(ready[slot], upload), "ready");
            wait = ready[slot];
            w = { base, base + role_bytes[0], base + role_bytes[0] + role_bytes[1] };
        }
        CUevent_st * done = nullptr;
        engine.check(api->compute(engine.p, int(g), &w, wait, &done, error, sizeof(error)), error);
        require(done, "compute returned no done event");
        if (!host && e % 2 == 1) { slot_done[(arena_uses - 1) % 2] = done; }
    }
    const float * rows_out = nullptr;
    vram_group_layer_stats stats = {};
    std::vector<double> group_us(group_expert.size(), -1.0);
    engine.check(api->finish_layer(engine.p, &rows_out, &stats, group_us.data(), error, sizeof(error)), error);
    require(rows_out, "finish returned no rows");
    require(stats.experts == double(group_expert.size()) && stats.routes == double(routes.size()) && stats.us > 0 &&
        stats.begin_us > 0 && stats.finish_us > 0, "engine stats do not count the layer");
    // each group's own time (API v2) adds up to the stats' sum; the span holds every chain, one after another, and
    // whatever the stream waited for between them (an arena upload)
    double group_sum = 0;
    for (double us : group_us) { require(us >= 0, "a group's time is missing"); group_sum += us; }
    require(std::fabs(group_sum - stats.us) <= 1e-6 * std::max(1.0, stats.us) && stats.span_us + 1e-3 >= stats.us,
        "the groups' times are not the layer's");
    require(api->device_bytes(engine.p) == reserved, "the engine's device memory grew after reserve");
    const size_t free_finished = device_free();
    for (int r = 0; r < rows; ++r) {
        std::memcpy(output.data() + size_t(token_of[size_t(r)]) * latent, rows_out + size_t(r) * latent, size_t(latent) * sizeof(float));
    }
    engine.check(api->release(engine.p, error, sizeof(error)), error);
    require(api->device_bytes(engine.p) == 0, "release kept device memory");
    const size_t free_after = device_free();
    // reserve takes the buffers and, on the engine's first use, the kernels' modules; nothing more until release
    const bool own = device_free_is_own();
    std::printf("[kimi_vram_group_memory] reserved_bytes=%zu grown_from=%zu reserve_took=%lld during_pass=%lld kept_after_release=%lld"
        " device_free_checked=%d\n", reserved, first_reserved, (long long) free_before - (long long) free_reserved,
        (long long) free_reserved - (long long) free_finished, (long long) free_before - (long long) free_after, int(own));
    require(!own || free_finished == free_reserved, "device memory was taken between reserve and release");
    require(!own || first_use || free_after == free_before, "release did not give back what reserve took");
    checks += 3;
    for (auto & e : ready) { (void) cudaEventDestroy(e); }
    (void) cudaStreamDestroy(upload);
    if (pinned) { (void) cudaFreeHost(pinned); }
    if (arena) { (void) cudaFree(arena); }
    if (resident) { (void) cudaFree(resident); }
    return output;
}

void run_fixture(fixture_inputs & inputs, const vram_group_compute_api * api, const grouped_fixture & spec, bool in_place, size_t & checks, double & worst,
        bool & failed) {
    fixture f;
    f.type = spec.up;
    f.down_type = spec.down;
    f.latent = spec.latent;
    f.hidden = spec.hidden;
    f.experts = 16;
    make_synthetic(f, 20261021);
    inputs.set("vram_budget", "0");
    auto sink = make_sink(inputs, f, 8, 8);
    char error[512] = {};
    std::unique_ptr<kimi_cpu_prefill, decltype(&kimi_cpu_prefill_destroy)> prefill(
        kimi_cpu_prefill_create(sink.get(), error, sizeof(error)), kimi_cpu_prefill_destroy);
    require(prefill != nullptr, error);
    tensors meta(f, 1);
    std::vector<float> norm(size_t(f.latent));
    std::mt19937 rng(20261022);
    std::uniform_real_distribution<float> unit(0.5f, 1.5f);
    for (auto & v : norm) { v = unit(rng); }
    engine_ptr engine(api, f.latent, f.hidden, f.type, f.down_type);
    bool first_use = true;
    constexpr int nslots = 8;
    for (const auto & c : spec.cases) {
        if (c.host && !in_place) {
            std::printf("[kimi_vram_group] fixture=%s case=%s skipped=the device does not read pageable host memory in place\n",
                spec.name, c.name);
            continue;
        }
        std::normal_distribution<float> sample(0.0f, 1.0f);
        std::vector<float> input(size_t(c.tokens) * f.latent);
        for (auto & v : input) { v = sample(rng); }
        std::vector<int32_t> ids(size_t(c.tokens) * nslots);
        std::vector<float> weights(ids.size());
        for (int t = 0; t < c.tokens; ++t) {
            std::vector<int32_t> order(size_t(f.experts));
            for (int e = 0; e < f.experts; ++e) { order[size_t(e)] = e; }
            std::shuffle(order.begin(), order.end(), rng);
            if (c.tokens == 300 && std::find(order.begin(), order.begin() + nslots, 0) == order.begin() + nslots) {
                order[0] = 0; // expert 0 takes every token of the 300: two chunks of its chain
            }
            float sum = 0.0f;
            for (int s = 0; s < nslots; ++s) {
                ids[size_t(t) * nslots + s] = order[size_t(s)];
                weights[size_t(t) * nslots + s] = unit(rng);
                sum += weights[size_t(t) * nslots + s];
            }
            for (int s = 0; s < nslots; ++s) { weights[size_t(t) * nslots + s] /= sum; }
        }
        const auto & a = c.activation;
        kimi_cpu_sink_activation cpu_activation = { a.kind, a.situ_beta, a.situ_linear_beta, a.swiglu_limit,
            c.norm ? norm.data() : nullptr, c.eps };
        vram_group_activation gpu_activation = { { a.kind, a.situ_beta, a.situ_linear_beta, a.swiglu_limit },
            c.norm ? norm.data() : nullptr, c.eps };
        auto cpu_run = [&](const std::vector<int32_t> & route_ids) {
            std::vector<float> out(input.size());
            kimi_cpu_prefill_stats stats = {};
            require(kimi_cpu_prefill_run_activation(prefill.get(), meta.role[0], meta.role[1], meta.role[2], input.data(),
                route_ids.data(), weights.data(), nslots, c.tokens, &cpu_activation, out.data(), &stats, error, sizeof(error)),
                std::string("CPU prefill: ") + error);
            return out;
        };
        const auto expected = cpu_run(ids);
        std::vector<bool> gpu(size_t(f.experts), true);
        std::vector<float> got;
        if (c.split) {
            for (int e = 8; e < f.experts; ++e) { gpu[size_t(e)] = false; }
            std::vector<int32_t> cpu_ids = ids;
            for (auto & e : cpu_ids) { if (e < 8) { e = -1; } }
            got = cpu_run(cpu_ids);
            const auto gpu_part = run_engine(api, engine, first_use, f, 1, c.tokens, nslots, input, ids, weights, gpu,
                gpu_activation, c.host, checks);
            for (size_t i = 0; i < got.size(); ++i) { got[i] += gpu_part[i]; }
        } else {
            got = run_engine(api, engine, first_use, f, 1, c.tokens, nslots, input, ids, weights, gpu, gpu_activation, c.host, checks);
        }
        first_use = false;
        const auto truth = exact_layer(f, 1, c.tokens, nslots, input, ids, weights, a, c.norm ? norm.data() : nullptr, c.eps);
        double lanes = 0.0, mean = 0.0, gpu_exact = 0.0, cpu_exact = 0.0;
        for (int t = 0; t < c.tokens; ++t) {
            for (int i = 0; i < f.latent; ++i) { require(std::isfinite(got[size_t(t) * f.latent + i]), "non-finite engine output"); }
            const size_t o = size_t(t) * f.latent;
            const double e = token_nmse(got.data() + o, expected.data() + o, f.latent);
            lanes = std::max(lanes, e);
            mean += e / c.tokens;
            gpu_exact = std::max(gpu_exact, token_nmse(got.data() + o, truth.data() + o, f.latent));
            cpu_exact = std::max(cpu_exact, token_nmse(expected.data() + o, truth.data() + o, f.latent));
        }
        worst = std::max(worst, gpu_exact);
        std::printf("[kimi_vram_group] fixture=%s case=%s format=%s/%s latent=%d hidden=%d experts=%d top=%d tokens=%d act=%s limit=%g norm=%d eps=%g split=%d host=%d lanes_worst_token_nmse=%.3g mean=%.3g exact_worst:gpu=%.3g,cpu=%.3g\n",
            spec.name, c.name, ggml_type_name(f.type), ggml_type_name(f.down_type), f.latent, f.hidden, f.experts, nslots, c.tokens,
            a.kind == KIMI_VRAM_ACTIVATION_SITU ? "situ" : "swiglu", double(a.swiglu_limit), int(c.norm), double(c.eps), int(c.split),
            int(c.host), lanes, mean, gpu_exact, cpu_exact);
        // The engine against the exact layer, and the CPU lane against it too, so that the independent lane still
        // checks what both the engine and the exact layer compute.
        if (gpu_exact > 5e-4 || gpu_exact > 1.5 * cpu_exact || cpu_exact > 1e-3 || lanes > 1.5e-3) {
            std::printf("[kimi_vram_group] fixture=%s case=%s FAIL\n", spec.name, c.name);
            failed = true;
        }
        ++checks;
    }
}

void run_numerics(fixture_inputs & inputs, size_t & checks, double & worst) {
    const auto * api = engine_api();
    int in_place = 0;
    cuda(cudaDeviceGetAttribute(&in_place, cudaDevAttrPageableMemoryAccessUsesHostPageTables, 0), "attribute");
    const float inf = std::numeric_limits<float>::infinity();
    const case_activation swiglu = { KIMI_VRAM_ACTIVATION_SWIGLU_CLAMP, 0.0f, 0.0f, inf };
    const case_activation clamp = { KIMI_VRAM_ACTIVATION_SWIGLU_CLAMP, 0.0f, 0.0f, 1.0f };
    const case_activation situ = { KIMI_VRAM_ACTIVATION_SITU, 4.0f, 25.0f, 0.0f };
    const float eps = 1e-8f; // Muse's post_norm_epsilon
    const grouped_fixture fixtures[] = {
        // Muse Spark UD-Q4_K_XL's routed experts at a reduced shape
        { "muse", GGML_TYPE_Q4_K, GGML_TYPE_Q5_K, 512, 512, {
            { "t1",               1, swiglu, false, eps,   false, false },
            { "t1-norm",          1, swiglu, true,  eps,   false, false },
            { "t33",             33, swiglu, false, eps,   false, false },
            { "t33-clamp",       33, clamp,  false, eps,   false, false },
            { "t33-norm",        33, swiglu, true,  eps,   false, false },
            { "t33-norm-eps1e-5",33, swiglu, true,  1e-5f, false, false },
            { "t33-clamp-norm",  33, clamp,  true,  eps,   false, false },
            { "t33-situ",        33, situ,   false, eps,   false, false },
            { "t300",           300, swiglu, false, eps,   false, false },
            { "t300-clamp",     300, clamp,  false, eps,   false, false },
            { "t300-norm",      300, swiglu, true,  eps,   false, false },
            { "t300-clamp-norm",300, clamp,  true,  eps,   false, false },
            { "t300-situ",      300, situ,   false, eps,   false, false },
            { "t33-split-norm",  33, swiglu, true,  eps,   true,  false },
            { "t300-split-clamp",300, clamp, false, eps,   true,  false },
            { "t33-host-norm",   33, swiglu, true,  eps,   false, true  },
            { "t300-host-clamp",300, clamp,  false, eps,   false, true  },
        } },
        // a hidden width that is not a multiple of 512: the down operand's padded tail
        { "hidden768", GGML_TYPE_Q4_K, GGML_TYPE_Q5_K, 512, 768, {
            { "t33-norm",        33, swiglu, true,  eps,   false, false },
            { "t300-clamp",     300, clamp,  false, eps,   false, false },
        } },
        // down rows that are not a whole number of MMQ iterations (44 blocks of 32): MMVQ and the GEMM serve them
        { "rows1408", GGML_TYPE_Q8_0, GGML_TYPE_Q4_0, 512, 1408, {
            { "t1",               1, swiglu, false, eps,   false, false },
            { "t33-norm",        33, swiglu, true,  eps,   false, false },
            { "t300-clamp",     300, clamp,  false, eps,   false, false },
        } },
    };
    bool failed = false;
    for (const auto & spec : fixtures) { run_fixture(inputs, api, spec, in_place != 0, checks, worst, failed); }
    require(!failed, "the engine is above 5e-4 or above 1.5x the CPU lane against the exact layer, or the CPU lane is above "
        "1e-3 against it, or the lanes are 1.5e-3 apart");
}

// The lane through the sink and the tier (B5): a Muse-shaped model (Q4_K/Q5_K, a per-expert post norm) whose grouped
// prefill runs on the tier's batched engine, the residents a decode step left and transients (from the ring, or in
// place where the GPU reads host memory so), against the same sink without a GPU tier. A pass of one token runs the
// vector engine with the post norm applied by the tier, as decode's rows are normalized by the CPU. Start values that
// make the GPU cheap move transients; the CPU still computes the experts the planner leaves it. mixed_layers: layer 1's
// down is Q4_K and layer 2's Q5_K (Muse Q4_K_S), declared to the tier at load, both layers' residents in its one pool
// and each pass over both layers on the one engine.
void run_tier_lane(fixture_inputs & inputs, size_t & checks, double & worst, bool mixed_layers = false) {
    fixture f;
    f.type = GGML_TYPE_Q4_K; f.down_type = GGML_TYPE_Q5_K;
    if (mixed_layers) { f.layer_down_types[1] = GGML_TYPE_Q4_K; }
    f.latent = 512; f.hidden = 512; f.experts = 16;
    make_synthetic(f, 20261023);
    inputs.set("vram_budget", "0");
    inputs.set("vram_prefill", "0");
    auto cpu = make_sink(inputs, f, 8, 4);
    inputs.set("vram_budget", std::to_string(uint64_t(mixed_layers ? 8 : 4) * triplet_bytes(f)).c_str());
    inputs.set("tier.keep_across_requests", "1");
    inputs.set("tier.global", "1");
    inputs.set("vram_prefill", "1");
    inputs.set("tier.min_count", "0");
    inputs.set("place_costs", "gpu_fixed_us=1,gpu_route_us=0.01,gpu_host_fixed_us=1,gpu_host_route_us=0.01");
    auto mixed = make_sink(inputs, f, 8, 4);
    configure_cache(cpu.get(), 0, {1, 2}, f.experts);
    configure_cache(mixed.get(), 0, {1, 2}, f.experts);
    const std::vector<int> layers = mixed_layers ? std::vector<int>{ 1, 2 } : std::vector<int>{ 1 };
    char error[512] = {};
    if (mixed_layers) {
        // the load step: every routed layer's tensors, whose types the sink declares to the tier
        tensors first(f, 1), second(f, 2);
        const ggml_tensor * experts[] = { first.role[0], first.role[1], first.role[2], second.role[0], second.role[1], second.role[2] };
        require(kimi_cpu_sink_vram_reserve(mixed.get(), experts, 2, 8, error, sizeof(error)), error);
    }
    std::vector<float> norm(size_t(f.latent));
    std::mt19937 rng(20261024);
    std::uniform_real_distribution<float> unit(0.5f, 1.5f);
    for (auto & v : norm) { v = unit(rng); }
    const kimi_cpu_sink_activation activation = { KIMI_CPU_SINK_ACTIVATION_SWIGLU_CLAMP, 0.0f, 0.0f,
        std::numeric_limits<float>::infinity(), norm.data(), 1e-5f };
    // a decode step gives the tier its format and four residents a layer
    {
        std::vector<int32_t> ids = { 0, 1, 2, 3 };
        std::vector<float> w(4, 0.25f), in(size_t(f.latent), 0.125f), out(size_t(f.latent));
        begin_cache(mixed.get(), 93, 1);
        for (int layer : layers) {
            tensors meta(f, layer);
            kimi_cpu_sink_stats st = {};
            require(kimi_cpu_sink_run_activation(mixed.get(), meta.role[0], meta.role[1], meta.role[2], in.data(), ids.data(),
                w.data(), 4, &activation, out.data(), &st, error, sizeof(error)), error);
        }
        require(kimi_cpu_sink_vram_phase_complete(mixed.get(), true, error, sizeof(error)), error);
    }
    std::unique_ptr<kimi_cpu_prefill, decltype(&kimi_cpu_prefill_destroy)> cpu_prefill(
        kimi_cpu_prefill_create(cpu.get(), error, sizeof(error)), kimi_cpu_prefill_destroy);
    require(cpu_prefill != nullptr, error);
    std::unique_ptr<kimi_cpu_prefill, decltype(&kimi_cpu_prefill_destroy)> gpu_prefill(
        kimi_cpu_prefill_create(mixed.get(), error, sizeof(error)), kimi_cpu_prefill_destroy);
    require(gpu_prefill != nullptr, error);
    constexpr int nslots = 8;
    for (const int tokens : { 48, 1, 48 }) {
        std::normal_distribution<float> sample(0.0f, 1.0f);
        std::vector<float> input(size_t(tokens) * f.latent);
        for (auto & v : input) { v = sample(rng); }
        std::vector<int32_t> ids(size_t(tokens) * nslots);
        std::vector<float> weights(ids.size());
        for (int t = 0; t < tokens; ++t) {
            std::vector<int32_t> order(size_t(f.experts));
            for (int e = 0; e < f.experts; ++e) { order[size_t(e)] = e; }
            std::shuffle(order.begin(), order.end(), rng);
            // every token takes resident expert 0 (the one-token pass too)
            if (std::find(order.begin(), order.begin() + nslots, 0) == order.begin() + nslots) { order[0] = 0; }
            float sum = 0.0f;
            for (int k = 0; k < nslots; ++k) {
                ids[size_t(t) * nslots + k] = order[size_t(k)];
                sum += weights[size_t(t) * nslots + k] = unit(rng);
            }
            for (int k = 0; k < nslots; ++k) { weights[size_t(t) * nslots + k] /= sum; }
        }
        // a pass over every layer, as a graph's: the layers of each format in turn on the tier's one engine
        for (int layer : layers) {
            tensors meta(f, layer);
            std::vector<float> expected(input.size()), got(input.size());
            kimi_cpu_prefill_stats cs = {}, gs = {};
            require(kimi_cpu_prefill_run_activation(cpu_prefill.get(), meta.role[0], meta.role[1], meta.role[2], input.data(),
                ids.data(), weights.data(), nslots, tokens, &activation, expected.data(), &cs, error, sizeof(error)), error);
            require(kimi_cpu_prefill_run_activation(gpu_prefill.get(), meta.role[0], meta.role[1], meta.role[2], input.data(),
                ids.data(), weights.data(), nslots, tokens, &activation, got.data(), &gs, error, sizeof(error)), error);
            double lanes = 0.0;
            for (int t = 0; t < tokens; ++t) {
                for (int i = 0; i < f.latent; ++i) { require(std::isfinite(got[size_t(t) * f.latent + i]), "non-finite lane output"); }
                lanes = std::max(lanes, token_nmse(got.data() + size_t(t) * f.latent, expected.data() + size_t(t) * f.latent, f.latent));
            }
            worst = std::max(worst, lanes);
            const uint64_t moved = gs.gpu_transient_experts + gs.gpu_ssd_transient_experts;
            std::printf("[kimi_vram_group_lane] layer=%d down=%s tokens=%d residents=%llu moved=%llu (ram %llu ssd %llu) "
                "cpu_ssd=%llu lanes_worst_token_nmse=%.3g\n", layer, ggml_type_name(f.down_of(layer)), tokens,
                (unsigned long long) gs.gpu_experts, (unsigned long long) moved,
                (unsigned long long) gs.gpu_transient_experts, (unsigned long long) gs.gpu_ssd_transient_experts,
                (unsigned long long) gs.ssd_experts, lanes);
            require(lanes <= 1.5e-3, "the tier's lane is more than 1.5e-3 from the CPU lane");
            require(gs.gpu_experts > 0, "the residents did not compute on the GPU");
            if (tokens > 1) { require(moved > 0, "no transient moved at a cheap GPU"); }
            require(gs.stream.pending_reads == 0, "the lane returned with live reads");
            checks += 4;
        }
    }
    inputs.unset("place_costs");
    inputs.set("vram_prefill", "0");
}

// least-squares line y = a + b x with both terms >= 0 (the engine's start values take no negative term): a line
// through the origin where the intercept would be negative, a constant where the slope would
void fit(const std::vector<double> & x, const std::vector<double> & y, double & a, double & b) {
    const double n = double(x.size());
    double sx = 0, sy = 0, sxx = 0, sxy = 0;
    for (size_t i = 0; i < x.size(); ++i) { sx += x[i]; sy += y[i]; sxx += x[i] * x[i]; sxy += x[i] * y[i]; }
    b = (n * sxy - sx * sy) / (n * sxx - sx * sx);
    a = (sy - b * sx) / n;
    if (a < 0) { a = 0; b = sxx > 0 ? sxy / sxx : 0; }
    else if (b < 0) { b = 0; a = sy / n; }
}

std::vector<int> parse_counts(const std::string & s) {
    std::vector<int> out;
    std::stringstream in(s);
    for (std::string item; std::getline(in, item, ',');) { out.push_back(std::stoi(item)); }
    return out;
}

ggml_type parse_type(const std::string & name) {
    for (int t = 0; t < GGML_TYPE_COUNT; ++t) {
        const char * n = ggml_type_name(ggml_type(t));
        if (n && strcasecmp(n, name.c_str()) == 0) { return ggml_type(t); }
    }
    throw std::runtime_error("unknown type " + name);
}

struct time_options {
    int latent = 4096, hidden = 4096, reps = 24, threads = 0;
    ggml_type up = GGML_TYPE_Q4_K, down = GGML_TYPE_Q5_K;
    std::vector<int> counts = { 8, 26, 150, 300 };
    double drive_gbps = 0.0;
    bool cpu = true;
    // the sink's threads: its participants and its I/O thread, as GGML_KIMI_CPU_SINK_THREADS counts them
    int sink_threads() const { return threads > 0 ? threads : int(std::max(2u, std::thread::hardware_concurrency())); }
    size_t expert_bytes() const { return 2 * ggml_row_size(up, latent) * hidden + ggml_row_size(down, hidden) * latent; }
};

// Runs step() over and over on its own thread until destroyed: the rest of a pass while one lane is timed.
struct background {
    std::atomic<bool> stop{false};
    std::atomic<uint64_t> rounds{0};
    std::string error;
    std::thread thread;
    template <class F> explicit background(F step) : thread([this, step]() mutable {
        try {
            while (!stop.load()) { step(); ++rounds; }
        } catch (const std::exception & e) {
            error = e.what();
        }
    }) {}
    // stops the thread; true when it ran at least once and without an error
    bool finish() {
        if (thread.joinable()) { stop = true; thread.join(); }
        return error.empty() && rounds.load() > 0;
    }
    ~background() { finish(); }
};

// The GPU lane at the timing shape: one engine and enough distinct experts that none is still in the L2 cache when it
// comes round again, in device memory or, where the GPU reads it in place, in pageable host memory.
struct gpu_lane {
    const vram_group_compute_api * api;
    const time_options & o;
    bool host;
    size_t up_bytes = 0, stride = 0;
    int distinct = 0, rotation = 0;
    char * weights = nullptr;
    std::unique_ptr<engine_ptr> engine;
    vram_group_activation activation = {};

    gpu_lane(const vram_group_compute_api * a, const time_options & options, bool host_) : api(a), o(options), host(host_) {
        up_bytes = ggml_row_size(o.up, o.latent) * o.hidden;
        stride = o.expert_bytes();
        int l2 = 0;
        cuda(cudaDeviceGetAttribute(&l2, cudaDevAttrL2CacheSize, 0), "L2 size");
        distinct = int(std::max<size_t>(4, (4 * size_t(l2) + stride - 1) / stride));
        // one expert's roles quantized once and repeated: the bytes do not change the time, the addresses do
        std::vector<uint8_t> one(stride);
        std::mt19937 rng(20261023);
        std::normal_distribution<float> sample(0.0f, 0.06f);
        for (int role = 0; role < 3; ++role) {
            const int cols = role == 2 ? o.hidden : o.latent, rows = role == 2 ? o.latent : o.hidden;
            const ggml_type type = role == 2 ? o.down : o.up;
            std::vector<float> values(size_t(cols) * rows);
            for (auto & v : values) { v = sample(rng); }
            const std::vector<float> importance(ggml_quantize_requires_imatrix(type) ? size_t(cols) : 0, 1.0f);
            ggml_quantize_chunk(type, values.data(), one.data() + size_t(role) * up_bytes, 0, rows, cols,
                importance.empty() ? nullptr : importance.data());
        }
        if (host) {
            weights = static_cast<char *>(std::malloc(stride * distinct));
            require(weights, "host experts");
            for (int e = 0; e < distinct; ++e) { std::memcpy(weights + size_t(e) * stride, one.data(), stride); }
        } else {
            cuda(cudaMalloc(reinterpret_cast<void **>(&weights), stride * distinct), "device experts");
            for (int e = 0; e < distinct; ++e) { cuda(cudaMemcpy(weights + size_t(e) * stride, one.data(), stride, cudaMemcpyHostToDevice), "H2D"); }
        }
        const int max_rows = *std::max_element(o.counts.begin(), o.counts.end());
        engine = std::make_unique<engine_ptr>(api, o.latent, o.hidden, o.up, o.down);
        char error[512] = {};
        engine->check(api->reserve(engine->p, max_rows, max_rows * o.reps, error, sizeof(error)), error);
        float * staging = api->input_rows(engine->p);
        std::normal_distribution<float> input(0.0f, 1.0f);
        for (size_t i = 0; i < size_t(max_rows) * o.latent; ++i) { staging[i] = input(rng); }
        activation = { { KIMI_VRAM_ACTIVATION_SWIGLU_CLAMP, 0.0f, 0.0f, std::numeric_limits<float>::infinity() }, nullptr, 0.0f };
    }
    ~gpu_lane() {
        engine.reset();
        if (host) { std::free(weights); } else { (void) cudaFree(weights); }
    }
    // One layer of o.reps experts of n routes each: the chains' mean time per expert from the engine's events.
    double layer(int n) {
        std::vector<vram_group_route> routes;
        std::vector<int32_t> first = { 0 };
        for (int g = 0; g < o.reps; ++g) {
            for (int r = 0; r < n; ++r) { routes.push_back({ r, 1.0f / 8 }); }
            first.push_back(int32_t(routes.size()));
        }
        char error[512] = {};
        engine->check(api->begin_layer(engine->p, 0, n, routes.data(), first.data(), o.reps, &activation, error, sizeof(error)), error);
        for (int g = 0; g < o.reps; ++g) {
            const char * base = weights + size_t(rotation++ % distinct) * stride;
            const vram_compute_expert w = { base, base + up_bytes, base + 2 * up_bytes };
            engine->check(api->compute(engine->p, g, &w, nullptr, nullptr, error, sizeof(error)), error);
        }
        const float * rows = nullptr;
        vram_group_layer_stats stats = {};
        engine->check(api->finish_layer(engine->p, &rows, &stats, nullptr, error, sizeof(error)), error);
        return stats.us / stats.experts;
    }
    // the mean of a second layer at each count (the first warms up)
    std::vector<double> times() {
        std::vector<double> us;
        for (const int n : o.counts) { layer(n); us.push_back(layer(n)); }
        return us;
    }
};

// The CPU lane at the timing shape: one expert of n routes on the sink's participants (its up/gate, hidden
// quantization and down; the input rows' quantization is per token and not counted).
struct cpu_lane {
    const time_options & o;
    fixture f;
    sink_ptr sink;
    std::unique_ptr<kimi_cpu_prefill, decltype(&kimi_cpu_prefill_destroy)> prefill{ nullptr, kimi_cpu_prefill_destroy };
    std::unique_ptr<tensors> meta;
    std::vector<float> input, output, weights;
    std::vector<int32_t> ids;
    kimi_cpu_sink_activation activation = {};

    explicit cpu_lane(fixture_inputs & inputs, const time_options & options) : o(options) {
        f.type = o.up;
        f.down_type = o.down;
        f.latent = o.latent;
        f.hidden = o.hidden;
        f.experts = 1;
        make_synthetic(f, 20261024);
        inputs.set("vram_budget", "0");
        sink = make_sink(inputs, f, o.sink_threads(), 8);
        char error[512] = {};
        prefill.reset(kimi_cpu_prefill_create(sink.get(), error, sizeof(error)));
        require(prefill != nullptr, error);
        meta = std::make_unique<tensors>(f, 1);
        activation = { KIMI_CPU_SINK_ACTIVATION_SWIGLU_CLAMP, 0.0f, 0.0f, std::numeric_limits<float>::infinity(), nullptr, 0.0f };
        const int max_rows = *std::max_element(o.counts.begin(), o.counts.end());
        input.resize(size_t(max_rows) * o.latent);
        output.resize(input.size());
        std::mt19937 rng(20261025);
        std::normal_distribution<float> sample(0.0f, 1.0f);
        for (auto & v : input) { v = sample(rng); }
        ids.assign(size_t(max_rows), 0);
        weights.assign(size_t(max_rows), 1.0f);
    }
    double run(int n) {
        char error[512] = {};
        kimi_cpu_prefill_stats stats = {};
        require(kimi_cpu_prefill_run_activation(prefill.get(), meta->role[0], meta->role[1], meta->role[2], input.data(),
            ids.data(), weights.data(), 1, n, &activation, output.data(), &stats, error, sizeof(error)), error);
        return stats.stream.up_gate_us + stats.stream.hidden_quantize_us + stats.stream.down_reduce_us;
    }
    // the mean of three runs at each count, after one that warms up
    std::vector<double> times() {
        std::vector<double> us;
        for (const int n : o.counts) {
            run(n);
            us.push_back((run(n) + run(n) + run(n)) / 3);
        }
        return us;
    }
};

// The link: one expert copied from page-locked host memory into device memory, as a transient's upload.
struct link_copy {
    size_t bytes;
    void * host = nullptr, * device = nullptr;
    cudaStream_t stream = nullptr;
    cudaEvent_t start = nullptr, end = nullptr;
    explicit link_copy(size_t bytes_) : bytes(bytes_) {
        cuda(cudaMallocHost(&host, bytes), "pinned expert");
        std::memset(host, 1, bytes);
        cuda(cudaMalloc(&device, bytes), "device expert");
        cuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "copy stream");
        cuda(cudaEventCreate(&start), "event");
        cuda(cudaEventCreate(&end), "event");
    }
    ~link_copy() {
        (void) cudaEventDestroy(start);
        (void) cudaEventDestroy(end);
        (void) cudaStreamDestroy(stream);
        (void) cudaFree(device);
        (void) cudaFreeHost(host);
    }
    double copy() {
        cuda(cudaEventRecord(start, stream), "event");
        cuda(cudaMemcpyAsync(device, host, bytes, cudaMemcpyHostToDevice, stream), "expert H2D");
        cuda(cudaEventRecord(end, stream), "event");
        cuda(cudaEventSynchronize(end), "copy");
        float ms = 0.0f;
        cuda(cudaEventElapsedTime(&ms, start, end), "event time");
        return 1e3 * double(ms);
    }
    // the mean of four copies after one
    double time() { copy(); return (copy() + copy() + copy() + copy()) / 4; }
};

// The lanes' times per expert (the GPU's chain from device memory and, where the GPU reads pageable host memory in
// place, from host memory; the CPU lane), each idle and busy. Busy is what else runs in a pass while that lane works:
// for the GPU lane the CPU lane on its participants and, when the weights come from device memory, the link copying one
// expert after another; for the CPU lane the GPU lane (in place where it can be, else from device memory, with the
// link). The drive's reads into host memory are not reproduced. Lines are fitted to each.
void run_timing(fixture_inputs & inputs, const time_options & o) {
    const auto * api = engine_api();
    const double expert_bytes = double(o.expert_bytes());
    std::vector<double> x(o.counts.begin(), o.counts.end());
    int in_place = 0;
    cuda(cudaDeviceGetAttribute(&in_place, cudaDevAttrPageableMemoryAccessUsesHostPageTables, 0), "attribute");
    const int threads = o.sink_threads();
    auto report = [&](const char * lane, const char * source, const char * load, const std::vector<double> & us) {
        double a = 0, b = 0;
        fit(x, us, a, b);
        std::string points;
        for (size_t i = 0; i < us.size(); ++i) { points += (i ? "," : "") + std::to_string(int(o.counts[i])) + ":" + std::to_string(int(std::lround(us[i]))); }
        std::printf("[kimi_vram_group_fit] lane=%s source=%s load=%s threads=%d participants=%d format=%s/%s latent=%d hidden=%d points_us=%s fixed_us=%.1f per_route_us=%.3f at150_us=%.1f\n",
            lane, source, load, threads, threads - 1, ggml_type_name(o.up), ggml_type_name(o.down), o.latent, o.hidden,
            points.c_str(), a, b, a + 150 * b);
        return a + 150 * b;
    };
    auto require_ran = [](background & load, const char * what) {
        require(load.finish(), std::string("the ") + what + " load did not run: " + load.error);
    };
    std::unique_ptr<cpu_lane> cpu;
    if (o.cpu) { cpu = std::make_unique<cpu_lane>(inputs, o); }
    link_copy link(o.expert_bytes());

    gpu_lane device(api, o, false);
    const double device_idle = report("gpu", "device", "idle", device.times());
    double device_busy = device_idle, copy_busy = 0.0;
    const double copy_idle = link.time();
    {
        std::unique_ptr<background> cpu_load, link_load;
        if (cpu) { cpu_load = std::make_unique<background>([&] { cpu->run(150); }); }
        link_load = std::make_unique<background>([&] { link.copy(); });
        device_busy = report("gpu", "device", cpu ? "busy" : "link", device.times());
        require_ran(*link_load, "link");
        if (cpu_load) {
            copy_busy = link.time();
            require_ran(*cpu_load, "CPU lane");
        }
    }
    std::printf("[kimi_vram_group_copy] bytes=%.0f idle_us=%.1f idle_gbps=%.2f busy_us=%.1f busy_gbps=%.2f\n", expert_bytes,
        copy_idle, expert_bytes / copy_idle * 1e-3, copy_busy, copy_busy > 0 ? expert_bytes / copy_busy * 1e-3 : 0.0);

    std::unique_ptr<gpu_lane> host;
    double host_idle = 0.0, host_busy = 0.0;
    if (in_place) {
        host = std::make_unique<gpu_lane>(api, o, true);
        host_idle = report("gpu", "host", "idle", host->times());
        std::unique_ptr<background> cpu_load;
        if (cpu) { cpu_load = std::make_unique<background>([&] { cpu->run(150); }); }
        host_busy = report("gpu", "host", cpu ? "busy" : "idle", host->times());
        if (cpu_load) { require_ran(*cpu_load, "CPU lane"); }
    } else {
        std::printf("[kimi_vram_group_fit] lane=gpu source=host skipped=the device does not read pageable host memory in place\n");
    }

    if (cpu) {
        report("cpu", "host", "idle", cpu->times());
        gpu_lane & lane = host ? *host : device;
        background gpu_load([&] { lane.layer(150); });
        std::unique_ptr<background> link_load;
        if (!host) { link_load = std::make_unique<background>([&] { link.copy(); }); }
        report("cpu", "host", "busy", cpu->times());
        require_ran(gpu_load, "GPU lane");
        if (link_load) { require_ran(*link_load, "link"); }
    }

    if (o.drive_gbps > 0) {
        // the design's bound on the chain at 150 routes, on the busy line
        const double bound = 0.8 * expert_bytes / (o.drive_gbps * 1e3);
        std::printf("[kimi_vram_group_gate] expert_bytes=%.0f drive_gbps=%.2f bound_us=%.1f device150_us=%.1f device150_idle_us=%.1f pass=%d",
            expert_bytes, o.drive_gbps, bound, device_busy, device_idle, int(device_busy <= bound));
        if (host) {
            // in place against the copy: with the copy overlapping the chain, a copied expert costs the longer of the
            // two per expert, an in-place one its in-place chain
            const double copy = copy_busy > 0 ? copy_busy : copy_idle;
            std::printf(" host150_us=%.1f host150_idle_us=%.1f host_pass=%d copy_us=%.1f in_place=%d", host_busy, host_idle,
                int(host_busy <= bound), copy, int(host_busy <= std::max(device_busy, copy)));
        }
        std::printf("\n");
    }
}
} // namespace

int main(int argc, char ** argv) {
    fixture_inputs inputs; inputs.set_runtime_services(test_gpu_services());
    try {
        ggml_cpu_init();
        inputs.set("ready_first", "1");
        inputs.set("vram_budget", "0");
        cuda(cudaSetDevice(0), "cudaSetDevice");
        bool timing = false;
        time_options o;
        for (int i = 1; i < argc; ++i) {
            const std::string a = argv[i];
            auto next = [&]() -> std::string { require(i + 1 < argc, "missing value for " + a); return argv[++i]; };
            if (a == "--time") { timing = true; }
            else if (a == "--latent") { o.latent = std::stoi(next()); }
            else if (a == "--hidden") { o.hidden = std::stoi(next()); }
            else if (a == "--up") { o.up = parse_type(next()); }
            else if (a == "--down") { o.down = parse_type(next()); }
            else if (a == "--routes") { o.counts = parse_counts(next()); }
            else if (a == "--reps") { o.reps = std::stoi(next()); }
            else if (a == "--threads") { o.threads = std::stoi(next()); }
            else if (a == "--drive-gbps") { o.drive_gbps = std::stod(next()); }
            else if (a == "--no-cpu") { o.cpu = false; }
            else { throw std::runtime_error("unknown argument " + a); }
        }
        if (timing) {
            run_timing(inputs, o);
            std::printf("KIMI_VRAM_GROUP_TIME_DONE\n");
            return 0;
        }
        size_t checks = 0;
        double worst = 0.0, lane_worst = 0.0;
        run_numerics(inputs, checks, worst);
        run_tier_lane(inputs, checks, lane_worst);
        run_tier_lane(inputs, checks, lane_worst, true);
        std::printf("KIMI_VRAM_GROUP_PASS checks=%zu worst_token_nmse_against_exact=%.3g lane_against_cpu=%.3g\n", checks,
            worst, lane_worst);
        return 0;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "KIMI_VRAM_GROUP_FAIL: %s\n", e.what());
        return 1;
    }
}
