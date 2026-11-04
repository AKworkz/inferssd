// SPDX-License-Identifier: Apache-2.0
// Decode microbenchmark of the SSD executor; built, not a CTest. Each decode call is one ggml graph of one sink node:
// 8 routed Q8_0 experts of hidden 2048 and expert ff 768, as inferssd-server computes it. hit: every expert a RAM-cache
// hit (compute-bound); cold: every expert read from the drive (I/O plus compute). Reports the median and spread of
// microseconds per call; empty: a call without a run, the graph and call overhead alone. --threads sizes the ggml
// threadpool; the sink adds its I/O thread.
#include "ggml.h"
#include "ggml-cpu.h"
#include "ssd/cpu-sink.h"
#include "ssd/testing.h"
#include "test-resources.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr int latent = 2048, hidden = 768, routed = 8, layer = 1;
constexpr ggml_type type = GGML_TYPE_Q8_0;

void require(bool ok, const std::string & what) { if (!ok) { throw std::runtime_error(what); } }

std::string role_name(int role) {
    static const char * roles[] = { "up", "gate", "down" };
    return "blk." + std::to_string(layer) + ".ffn_" + roles[role] + "_exps.weight";
}

struct pack {
    std::string path;
    int fd = -1, experts = 0;
    size_t role_bytes[3] = {};
    std::map<std::pair<std::string, int32_t>, uint64_t> offsets;
    ~pack() { ssd::platform::close_fd(fd); if (!path.empty()) { test_resources::remove_file(path); } }
    static bool resolve(void * user, const char * name, int32_t expert, kimi_cpu_sink_source * out) {
        auto & p = *static_cast<pack *>(user);
        const auto found = p.offsets.find({ name, expert });
        if (found == p.offsets.end()) { return false; }
        const int role = std::strstr(name, "_down_") ? 2 : std::strstr(name, "_gate_") ? 1 : 0;
        *out = {};
        out->fd_direct = p.fd;
        out->offset = found->second;
        out->nbytes = (p.role_bytes[role] + 4095) / 4096 * 4096;
        return true;
    }
};

void make_pack(pack & p, int experts) {
    p.path = test_resources::temp_file("kimi-sink-bench");
    p.experts = experts;
    std::ofstream file(p.path, std::ios::binary | std::ios::trunc);
    require(bool(file), "bench pack create");
    std::mt19937 rng(20261028);
    std::normal_distribution<float> sample(0.0f, 0.05f);
    uint64_t offset = 0;
    std::vector<float> values;
    std::vector<uint8_t> payload;
    for (int expert = 0; expert < experts; ++expert) {
        for (int role = 0; role < 3; ++role) {
            const int cols = role == 2 ? hidden : latent, rows = role == 2 ? latent : hidden;
            p.role_bytes[role] = ggml_row_size(type, cols) * rows;
            values.resize(size_t(cols) * rows);
            for (auto & v : values) { v = sample(rng); }
            payload.assign((p.role_bytes[role] + 4095) / 4096 * 4096, 0);
            require(ggml_quantize_chunk(type, values.data(), payload.data(), 0, rows, cols, nullptr) == p.role_bytes[role],
                "bench quantization");
            file.write(reinterpret_cast<const char *>(payload.data()), std::streamsize(payload.size()));
            p.offsets[{ role_name(role), expert }] = offset;
            offset += payload.size();
        }
    }
    file.close();
    require(bool(file), "bench pack write");
    p.fd = test_resources::open_direct(p.path);
}

struct call_context {
    kimi_cpu_sink * sink = nullptr;
    bool empty = false;
    ggml_tensor * role[3] = {};
    kimi_cpu_sink_stats stats = {};
    bool ok = false;
    char error[512] = {};
};

void decode_node(ggml_tensor * dst, int ith, int nth, void * userdata) {
    auto & c = *static_cast<call_context *>(userdata);
    if (ith) { kimi_cpu_sink_call_join(c.sink, ith, nth); return; }
    if (!kimi_cpu_sink_call_open(c.sink, nth, c.error, sizeof(c.error))) { std::fprintf(stderr, "%s\n", c.error); std::abort(); }
    c.ok = c.empty || kimi_cpu_sink_run(c.sink, c.role[0], c.role[1], c.role[2], static_cast<const float *>(dst->src[0]->data),
        static_cast<const int32_t *>(dst->src[1]->data), static_cast<const float *>(dst->src[2]->data), routed,
        4.0f, 25.0f, static_cast<float *>(dst->data), &c.stats, c.error, sizeof(c.error));
    kimi_cpu_sink_call_close(c.sink);
}

struct spread { double median = 0, p10 = 0, p90 = 0, min = 0, max = 0, mean = 0; };
spread summarize(std::vector<double> v) {
    spread s;
    if (v.empty()) { return s; }
    std::sort(v.begin(), v.end());
    auto at = [&](double q) { return v[std::min(v.size() - 1, size_t(q * double(v.size() - 1) + 0.5))]; };
    s.median = at(0.5); s.p10 = at(0.1); s.p90 = at(0.9); s.min = v.front(); s.max = v.back();
    for (double x : v) { s.mean += x; }
    s.mean /= double(v.size());
    return s;
}

void run_variant(pack & p, const std::string & variant, int threads, int calls, int warmup, int poll_us) {
    const bool hit = variant == "hit", empty = variant == "empty";
    ssd::cpu::testing::execution_policy policy;
    policy.ready_first = policy.early_up_gate = policy.role_major = true;
    policy.nonblocking_down_sched = policy.dynamic_rows = true;
    policy.dynamic_chunk_rows = 64;
    policy.poll_us = poll_us;
    policy.cache_policy = "count_decay2";
    policy.cache_allocation = "uniform";
    policy.cache_keep = policy.cache_prefill_read = policy.cache_partial_reclaim = policy.cache_reclaimable_file = true;
    policy.cache_reserve_bytes = 512ull << 20;
    char error[512] = {};
    kimi_cpu_sink * created = ssd::cpu::testing::create(8, policy, nullptr, pack::resolve, &p, error, sizeof(error));
    require(created, std::string("bench sink: ") + error);
    std::unique_ptr<kimi_cpu_sink, decltype(&kimi_cpu_sink_destroy)> sink(created, kimi_cpu_sink_destroy);
    const uint64_t triplet = (p.role_bytes[0] + 4095) / 4096 * 4096 * 2 + (p.role_bytes[2] + 4095) / 4096 * 4096;
    const int32_t priority = layer;
    require(kimi_cpu_sink_cache_configure(sink.get(), hit ? 2 * routed * triplet : 0, &priority, 1, 2 * routed,
        error, sizeof(error)), std::string("bench cache: ") + error);

    ggml_context * meta = ggml_init({ 4 * ggml_tensor_overhead(), nullptr, true });
    require(meta, "bench metadata");
    call_context c;
    c.sink = sink.get();
    c.empty = empty;
    for (int role = 0; role < 3; ++role) {
        c.role[role] = ggml_new_tensor_3d(meta, type, role == 2 ? hidden : latent, role == 2 ? latent : hidden, p.experts);
        ggml_set_name(c.role[role], role_name(role).c_str());
    }
    ggml_context * ctx = ggml_init({ 16 * ggml_tensor_overhead() + ggml_graph_overhead() + (size_t(1) << 20), nullptr, false });
    require(ctx, "bench graph context");
    ggml_tensor * x = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, latent);
    ggml_tensor * ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, routed);
    ggml_tensor * weights = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, routed);
    for (int i = 0; i < latent; ++i) { static_cast<float *>(x->data)[i] = std::sin(float(i) * .013f) * .5f; }
    for (int i = 0; i < routed; ++i) { static_cast<float *>(weights->data)[i] = 1.0f / routed; }
    ggml_tensor * args[] = { x, ids, weights };
    ggml_tensor * out = ggml_custom_4d(ctx, GGML_TYPE_F32, latent, 1, 1, 1, args, 3, decode_node, GGML_N_TASKS_MAX, &c);
    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, out);
    ggml_threadpool_params params = ggml_threadpool_params_default(threads);
    std::unique_ptr<ggml_threadpool, decltype(&ggml_threadpool_free)> pool(ggml_threadpool_new(&params), ggml_threadpool_free);
    require(bool(pool), "bench threadpool");
    ggml_cplan plan = ggml_graph_plan(graph, threads, pool.get());
    std::vector<uint8_t> work(plan.work_size);
    plan.work_data = work.data();

    // hit: one expert set, resident after the first call. cold: disjoint sets in turn, the cache off.
    const int sets = hit ? 1 : std::max(1, p.experts / routed);
    std::vector<std::vector<float>> first(static_cast<size_t>(sets));
    std::vector<double> wall, sink_us;
    uint64_t read_bytes = 0, hits = 0;
    for (int call = 0; call < warmup + calls; ++call) {
        const int set = call % sets;
        for (int i = 0; i < routed; ++i) { static_cast<int32_t *>(ids->data)[i] = set * routed + i; }
        require(kimi_cpu_sink_cache_begin(sink.get(), 1, uint64_t(call) + 1, true, error, sizeof(error)),
            std::string("bench token: ") + error);
        const auto start = std::chrono::steady_clock::now();
        require(ggml_graph_compute(graph, &plan) == GGML_STATUS_SUCCESS, "bench graph compute");
        const double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - start).count();
        require(c.ok, std::string("bench decode: ") + c.error);
        // the same experts and input give the same bits every call
        const auto * y = static_cast<const float *>(out->data);
        auto & seen = first[size_t(set)];
        if (seen.empty()) { seen.assign(y, y + latent); }
        require(empty || !std::memcmp(seen.data(), y, sizeof(float) * latent), "bench output changed between calls");
        if (call < warmup) { continue; }
        wall.push_back(us);
        sink_us.push_back(empty ? 0.0 : c.stats.total_us);
        read_bytes += c.stats.read_bytes;
        hits += c.stats.cache_hits;
    }
    require(!hit || hits == uint64_t(calls) * routed, "bench hit variant read from the drive");
    require(hit || empty || read_bytes == uint64_t(calls) * routed * triplet, "bench cold variant took a cache hit");
    const auto w = summarize(wall), s = summarize(sink_us);
    std::printf("bench=sink_decode variant=%s threads=%d poll_us=%d calls=%d experts=%d hidden=%d ff=%d type=q8_0"
        " median_us=%.1f p10_us=%.1f p90_us=%.1f min_us=%.1f max_us=%.1f mean_us=%.1f"
        " sink_median_us=%.1f sink_p10_us=%.1f sink_p90_us=%.1f read_mib_per_call=%.2f\n",
        variant.c_str(), threads, poll_us, calls, routed, latent, hidden, w.median, w.p10, w.p90, w.min, w.max, w.mean,
        s.median, s.p10, s.p90, double(read_bytes) / double(calls) / double(1 << 20));
    std::fflush(stdout);
    ggml_free(ctx);
    ggml_free(meta);
}

} // namespace

int main(int argc, char ** argv) {
    try {
        int threads = std::max(1, int(test_resources::process_cpus().size()) - 1), calls = 300, warmup = 30, experts = 32;
        int poll_us = 0;
        std::string variant = "both";
        for (int i = 1; i < argc; ++i) {
            auto value = [&]() { require(i + 1 < argc, "missing option value"); return std::string(argv[++i]); };
            const std::string arg = argv[i];
            if (arg == "--threads") { threads = std::stoi(value()); }
            else if (arg == "--calls") { calls = std::stoi(value()); }
            else if (arg == "--warmup") { warmup = std::stoi(value()); }
            else if (arg == "--experts") { experts = std::stoi(value()); }
            else if (arg == "--variant") { variant = value(); }
            else if (arg == "--poll-us") { poll_us = std::stoi(value()); }
            else {
                std::printf("usage: test-kimi-sink-bench [--threads N] [--calls N>=200] [--warmup N] [--experts N] "
                    "[--variant hit|cold|both|empty] [--poll-us 0..1000]\n");
                return arg == "--help" ? 0 : 2;
            }
        }
        require(threads >= 1 && threads <= 255 && calls >= 200 && warmup >= 0 && experts >= 2 * routed &&
            poll_us >= 0 && poll_us <= 10000 &&
            (variant == "hit" || variant == "cold" || variant == "both" || variant == "empty"), "invalid bench options");
        ggml_cpu_init();
        test_resources::io_ring();
        pack p;
        make_pack(p, experts);
        if (variant == "empty") { run_variant(p, "empty", threads, calls, warmup, poll_us); return 0; }
        if (variant != "cold") { run_variant(p, "hit", threads, calls, warmup, poll_us); }
        if (variant != "hit") { run_variant(p, "cold", threads, calls, warmup, poll_us); }
        return 0;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "SINK_BENCH_FAILED: %s\n", e.what());
        return 1;
    }
}
