#include "explicit-fixture-inputs.h"
// CPU-only grouped prefill correctness and real-pack transport benchmark.
// The reference is an independent one-token ggml CPU graph, never the executor.
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ssd/cpu-sink.h"
#include "ssd/cpu-prefill.h"
#include <set>
#include <limits>
#include <algorithm>
#include <cmath>
#include <chrono>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <iomanip>
#include <sstream>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>
#include "test-resources.h"

using test_resources::anonymous_bytes;

static void require(bool ok, const std::string & what) { if (!ok) { throw std::runtime_error(what); } }
static size_t aligned(size_t n) { return (n + 4095) & ~size_t(4095); }
static std::string tensor_name(int layer, int role) {
    static const char * roles[] = { "up", "gate", "down" };
    return "blk." + std::to_string(layer) + ".ffn_" + roles[role] + "_exps.weight";
}

struct entry {
    uint64_t offset = 0;
    size_t raw_bytes = 0, read_bytes = 0;
    std::vector<uint8_t> data;
};
using key = std::pair<std::string, int32_t>;
struct fixture {
    int latent = 256, hidden = 256, experts = 20, fd = -1;
    std::string path;
    bool remove_on_destroy = false;
    std::map<key, entry> entries;
    enum fault_kind { none, missing, unaligned_offset, wrong_size, past_eof, short_tail } fault = none;
    int32_t fault_expert = -1;
    int fault_role = -1;
    int resolve_calls = 0;
    ~fixture() { ssd::platform::close_fd(fd); if (remove_on_destroy) { test_resources::remove_file(path); } }
    static bool resolve(void * user, const char * name, int32_t id, kimi_cpu_sink_source * out) {
        auto & f = *static_cast<fixture *>(user);
        ++f.resolve_calls;
        auto it = f.entries.find({ name, id });
        const int role = std::strstr(name, ".ffn_up_exps.weight") ? 0 : std::strstr(name, ".ffn_gate_exps.weight") ? 1 : 2;
        const auto fault = (f.fault_expert < 0 || f.fault_expert == id) &&
                           (f.fault_role < 0 || f.fault_role == role) ? f.fault : none;
        if (fault == missing || it == f.entries.end()) { return false; }
        out->fd_direct = f.fd;
        out->offset = it->second.offset;
        out->nbytes = it->second.read_bytes;
        if (fault == unaligned_offset) { ++out->offset; }
        if (fault == wrong_size) { out->nbytes += 4096; }
        if (fault == past_eof) { out->offset = uint64_t(1) << 50; }
        if (fault == short_tail) {
            uint64_t size = 0;
            if (!ssd::platform::file_size(f.fd, size) || size < 4096) { return false; }
            out->offset = size - 4096;
        }
        return true;
    }
};

static void put_bytes(std::ofstream & f, const void * data, size_t n) {
    f.write(static_cast<const char *>(data), n); require(bool(f), "fixture write failed");
}

static void make_synthetic(fixture & f) {
    f.path = test_resources::temp_file("kimi-cpu-prefill-test"); f.remove_on_destroy = true;
    std::mt19937 rng(20260908);
    std::normal_distribution<float> sample(0.0f, 0.06f);
    // Keep the full on-disk GGMLMOEPACKv1 layout, including index and padding.
    for (int layer : { 1, 2 }) {
        for (int role = 0; role < 3; ++role) {
            const int cols = role == 2 ? f.hidden : f.latent;
            const int rows = role == 2 ? f.latent : f.hidden;
            const size_t bytes = ggml_row_size(GGML_TYPE_Q3_K, cols) * rows;
            for (int e = 0; e < f.experts; ++e) {
                std::vector<float> values(size_t(cols) * rows);
                for (auto & v : values) { v = sample(rng); }
                entry item; item.raw_bytes = bytes; item.read_bytes = aligned(bytes); item.data.resize(bytes);
                require(ggml_quantize_chunk(GGML_TYPE_Q3_K, values.data(), item.data.data(), 0, rows, cols, nullptr) == bytes,
                        "Q3_K quantization bytes mismatch");
                f.entries[{ tensor_name(layer, role), e }] = std::move(item);
            }
        }
    }
    const uint64_t n = f.entries.size();
    uint64_t off = aligned(40 + n * 152), data_start = off;
    for (auto & kv : f.entries) { kv.second.offset = off; off += kv.second.read_bytes; }
    std::ofstream file(f.path, std::ios::binary | std::ios::trunc);
    char magic[16] = {}; std::memcpy(magic, "GGMLMOEPACKv1", 12); uint32_t version = 1, header_bytes = 40;
    put_bytes(file, magic, sizeof(magic)); put_bytes(file, &version, 4); put_bytes(file, &header_bytes, 4);
    put_bytes(file, &n, 8); put_bytes(file, &data_start, 8);
    for (const auto & kv : f.entries) {
        char label[128] = {}; std::snprintf(label, sizeof(label), "%s", kv.first.first.c_str());
        int32_t expert = kv.first.second; uint32_t reserved = 0; uint64_t raw_bytes = kv.second.raw_bytes;
        put_bytes(file, label, 128); put_bytes(file, &expert, 4); put_bytes(file, &reserved, 4);
        put_bytes(file, &kv.second.offset, 8); put_bytes(file, &raw_bytes, 8);
    }
    std::vector<char> zero(4096, 0);
    auto pad_to = [&](uint64_t target) {
        uint64_t here = static_cast<uint64_t>(file.tellp());
        while (here < target) { size_t count = std::min<uint64_t>(zero.size(), target - here); put_bytes(file, zero.data(), count); here += count; }
    };
    pad_to(data_start);
    for (const auto & kv : f.entries) { pad_to(kv.second.offset); put_bytes(file, kv.second.data.data(), kv.second.raw_bytes); pad_to(kv.second.offset + kv.second.read_bytes); }
    file.close(); require(bool(file), "fixture close failed");
    f.fd = test_resources::open_direct(f.path);
}

struct tensors {
    ggml_context * ctx = nullptr;
    ggml_tensor * role[3] = {};
    tensors(const fixture & f, int layer) {
        ctx = ggml_init({ ggml_tensor_overhead() * 4, nullptr, true }); require(ctx, "metadata context failed");
        for (int r = 0; r < 3; ++r) {
            role[r] = ggml_new_tensor_3d(ctx, GGML_TYPE_Q3_K, r == 2 ? f.hidden : f.latent,
                                       r == 2 ? f.latent : f.hidden, f.experts);
            ggml_set_name(role[r], tensor_name(layer, r).c_str());
            require(role[r]->data == nullptr, "sink metadata unexpectedly has payload");
        }
    }
    ~tensors() { ggml_free(ctx); }
};

// This reference is a regular ggml CPU graph. It does not share the sink's
// conversion, dot dispatch, SiTU implementation, I/O, or reduction functions.
static std::vector<float> reference(const fixture & f, int layer, const std::vector<float> & input,
        const std::vector<int32_t> & ids, const std::vector<float> & weights, float beta, float linear_beta) {
    std::vector<float> output(f.latent, 0);
    if (std::none_of(ids.begin(), ids.end(), [](int32_t id) { return id >= 0; })) { return output; }
    const size_t max_nodes = 2048;
    ggml_context * ctx = ggml_init({ ggml_tensor_overhead() * max_nodes + ggml_graph_overhead_custom(max_nodes, false), nullptr, true });
    require(ctx, "reference context failed");
    ggml_backend_t backend = ggml_backend_cpu_init(); require(backend, "reference CPU backend failed");
    ggml_backend_cpu_set_n_threads(backend, 1);
    ggml_tensor * x = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, f.latent);
    std::vector<std::pair<ggml_tensor *, const entry *>> parameters;
    ggml_tensor * result = nullptr;
    for (size_t slot = 0; slot < ids.size(); ++slot) {
        if (ids[slot] < 0) { continue; }
        ggml_tensor * matrices[3];
        for (int role = 0; role < 3; ++role) {
            matrices[role] = ggml_new_tensor_2d(ctx, GGML_TYPE_Q3_K, role == 2 ? f.hidden : f.latent, role == 2 ? f.latent : f.hidden);
            parameters.push_back({ matrices[role], &f.entries.at({ tensor_name(layer, role), ids[slot] }) });
        }
        ggml_tensor * u = ggml_mul_mat(ctx, matrices[0], x);
        ggml_tensor * g = ggml_mul_mat(ctx, matrices[1], x);
        ggml_tensor * a = ggml_scale(ctx, ggml_tanh(ctx, ggml_scale(ctx, g, 1.0f / beta)), beta);
        a = ggml_mul(ctx, a, ggml_sigmoid(ctx, g));
        if (linear_beta > 0) { u = ggml_scale(ctx, ggml_tanh(ctx, ggml_scale(ctx, u, 1.0f / linear_beta)), linear_beta); }
        ggml_tensor * z = ggml_mul(ctx, a, u);
        ggml_tensor * y = ggml_mul_mat(ctx, matrices[2], z);
        y = ggml_scale(ctx, y, weights[slot]);
        result = result ? ggml_add(ctx, result, y) : y;
    }
    ggml_cgraph * graph = ggml_new_graph_custom(ctx, max_nodes, false); ggml_build_forward_expand(graph, result);
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend); require(buffer, "reference allocation failed");
    ggml_backend_tensor_set(x, input.data(), 0, input.size() * sizeof(float));
    for (auto & p : parameters) { ggml_backend_tensor_set(p.first, p.second->data.data(), 0, p.second->raw_bytes); }
    require(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS, "reference CPU graph failed");
    ggml_backend_tensor_get(result, output.data(), 0, output.size() * sizeof(float));
    ggml_backend_buffer_free(buffer); ggml_backend_free(backend); ggml_free(ctx);
    return output;
}

using prefill_ptr = std::unique_ptr<kimi_cpu_prefill, decltype(&kimi_cpu_prefill_destroy)>;

struct executor {
    fixture * active;
    sink_ptr sink;
    prefill_ptr prefill;
    static bool resolve(void * user, const char * name, int32_t id, kimi_cpu_sink_source * out) {
        return fixture::resolve(static_cast<executor *>(user)->active, name, id, out);
    }
    executor(fixture_inputs & inputs, fixture & f, int threads, int depth) : active(&f),
            prefill(nullptr, kimi_cpu_prefill_destroy) {
        test_resources::cpu_participants(threads - 1, 1);
        char error[512] = {};
        sink = inputs.create(threads, depth, resolve, this, error, sizeof(error));
        require(bool(sink), std::string("sink create: ") + error);
        prefill.reset(kimi_cpu_prefill_create(sink.get(), error, sizeof(error)));
        require(bool(prefill), std::string("prefill create: ") + error);
    }
};

struct batch {
    int tokens = 0, slots = 0;
    std::vector<float> x, weights;
    std::vector<int32_t> ids;
};

static batch synthetic_batch(const fixture & f, int tokens, int slots = 8) {
    batch b; b.tokens = tokens; b.slots = slots;
    b.x.resize(size_t(tokens) * f.latent); b.weights.resize(size_t(tokens) * slots); b.ids.resize(b.weights.size());
    for (size_t i = 0; i < b.x.size(); ++i) { b.x[i] = std::sin(float(i + 7) * 0.071f) * 0.3f; }
    for (int t = 0; t < tokens; ++t) {
        for (int s = 0; s < slots; ++s) {
            const size_t i = size_t(t) * slots + s;
            b.ids[i] = (t * 7 + s * 11 + 3) % f.experts;
            b.weights[i] = float((t + 3 * s) % 19 + 1) / 73.0f; // deliberately not renormalized
        }
        if (slots >= 8) {
            const size_t i = size_t(t) * slots;
            b.ids[i + 2] = b.ids[i]; // repeated token/expert pair must compute and read only once
            b.ids[i + 1] = -7; b.weights[i + 1] = 0;
            b.ids[i + 4] = -1; b.weights[i + 4] = 0;
            b.weights[i + 6] = 0; // valid zero-weight IDs still count and read
        }
    }
    return b;
}

static std::vector<float> reference_batch(fixture & f, int layer, const batch & b, float linear_beta = 25) {
    std::vector<float> expected;
    expected.reserve(size_t(b.tokens) * f.latent);
    for (int t = 0; t < b.tokens; ++t) {
        std::vector<int32_t> ids(b.ids.begin() + size_t(t) * b.slots, b.ids.begin() + size_t(t + 1) * b.slots);
        std::vector<float> weights(b.weights.begin() + size_t(t) * b.slots, b.weights.begin() + size_t(t + 1) * b.slots);
        std::vector<float> x(b.x.begin() + size_t(t) * f.latent, b.x.begin() + size_t(t + 1) * f.latent);
        // Real fixture reference weights are loaded for only one token at a time.
        // This bounds anonymous memory even when the ub64 union spans many GiB.
        std::vector<entry *> loaded;
        std::ifstream file;
        for (int32_t id : ids) {
            if (id < 0) { continue; }
            for (int role = 0; role < 3; ++role) {
                auto & e = f.entries.at({ tensor_name(layer, role), id });
                if (!e.data.empty()) { continue; }
                if (!file.is_open()) { file.open(f.path, std::ios::binary); require(bool(file), "reference pack open failed"); }
                e.data.resize(e.raw_bytes); file.seekg(e.offset);
                file.read(reinterpret_cast<char *>(e.data.data()), e.raw_bytes);
                require(bool(file), "short reference pack read"); loaded.push_back(&e);
            }
        }
        const auto y = reference(f, layer, x, ids, weights, 4, linear_beta);
        expected.insert(expected.end(), y.begin(), y.end());
        for (entry * e : loaded) { std::vector<uint8_t>().swap(e->data); }
    }
    return expected;
}

struct expected_counts {
    uint64_t experts = 0, pairs = 0, slots = 0, bytes = 0, max_group = 0;
};

static expected_counts count_routes(const fixture & f, int layer, const batch & b, bool grouped) {
    expected_counts result;
    std::map<int32_t, int> groups;
    for (int t = 0; t < b.tokens; ++t) {
        std::set<int32_t> seen;
        for (int s = 0; s < b.slots; ++s) {
            const int32_t id = b.ids[size_t(t) * b.slots + s];
            if (id < 0) { continue; }
            ++result.slots;
            if (seen.insert(id).second) { ++result.pairs; ++groups[id]; }
            if (!grouped) {
                ++result.experts;
                for (int role = 0; role < 3; ++role) { result.bytes += f.entries.at({ tensor_name(layer, role), id }).read_bytes; }
            }
        }
    }
    for (const auto & item : groups) {
        result.max_group = std::max(result.max_group, uint64_t(item.second));
        if (grouped) {
            ++result.experts;
            for (int role = 0; role < 3; ++role) { result.bytes += f.entries.at({ tensor_name(layer, role), item.first }).read_bytes; }
        }
    }
    return result;
}

struct sample_result {
    kimi_cpu_prefill_stats stats = {};
    double wall_us = 0, max_abs = 0, rel_l2 = 0;
    uint64_t hash = 1469598103934665603ULL;
};

static void add_stats(kimi_cpu_sink_stats & sum, const kimi_cpu_sink_stats & s) {
    sum.experts += s.experts; sum.read_requests += s.read_requests; sum.read_bytes += s.read_bytes;
    sum.read_errors += s.read_errors; sum.fallback_count += s.fallback_count;
    sum.ring_bytes = std::max(sum.ring_bytes, s.ring_bytes);
    sum.total_us += s.total_us; sum.io_us += s.io_us; sum.compute_us += s.compute_us; sum.wait_io_us += s.wait_io_us;
    sum.input_quantize_us += s.input_quantize_us; sum.hidden_quantize_us += s.hidden_quantize_us;
    sum.up_gate_us += s.up_gate_us; sum.down_reduce_us += s.down_reduce_us;
    sum.demand_bytes += s.demand_bytes; sum.cache_hits += s.cache_hits; sum.cache_hit_bytes += s.cache_hit_bytes;
    sum.cache_misses += s.cache_misses;
    sum.cache_admitted += s.cache_admitted; sum.cache_admitted_bytes += s.cache_admitted_bytes;
    sum.cache_copy_us += s.cache_copy_us; sum.cache_fill_read_requests += s.cache_fill_read_requests;
    sum.cache_fill_read_bytes += s.cache_fill_read_bytes; sum.ring_read_requests += s.ring_read_requests;
    sum.ring_read_bytes += s.ring_read_bytes; sum.fixed_read_requests += s.fixed_read_requests;
    sum.read_completions += s.read_completions; sum.pending_reads += s.pending_reads;
    sum.cache_new_slot_bytes += s.cache_new_slot_bytes;
}

static void check_transport(fixture_inputs & inputs, const kimi_cpu_sink_stats & s, bool succeeded, bool grouped,
        bool allow_grouped_hits = false) {
    const char * mode = inputs.get("read_fixed");
    const bool fixed = mode && !std::strcmp(mode, "1");
    require(s.cache_fill_read_requests + s.ring_read_requests == s.read_requests &&
            s.cache_fill_read_bytes + s.ring_read_bytes == s.read_bytes,
            "shared stream destination counters do not sum");
    require(s.read_completions == s.read_requests && s.pending_reads == 0,
            "shared stream returned with unconsumed CQEs");
    require(s.fixed_read_requests == (fixed ? s.ring_read_requests : 0) && s.cache_copy_us == 0,
            "shared stream READ_FIXED target or zero-copy contract failed");
    if (succeeded) {
        require(s.cache_fill_read_requests == 3 * s.cache_admitted &&
                s.cache_fill_read_bytes == s.cache_admitted_bytes && s.cache_new_slot_bytes <= s.cache_admitted_bytes,
                "shared stream direct admission counters differ");
    }
    if (grouped && !allow_grouped_hits) {
        require(s.cache_hits == 0 && s.cache_admitted == 0 && s.cache_admitted_bytes == 0 &&
                s.cache_fill_read_requests == 0 && s.cache_fill_read_bytes == 0 && s.cache_new_slot_bytes == 0,
                "prefill reused/admitted decode cache payload");
    }
}

static kimi_cpu_sink_cache_stats cache_snapshot(kimi_cpu_sink * sink, bool released = false) {
    kimi_cpu_sink_cache_stats s = {};
    kimi_cpu_sink_cache_get_stats(sink, &s);
    require(s.capacity_bytes <= s.budget_bytes && s.mapped_bytes <= s.capacity_bytes &&
            s.committed_bytes <= s.mapped_bytes && s.valid_bytes <= s.committed_bytes && s.copy_us == 0,
            "fused cache budget or zero-copy invariant failed");
    if (released) {
        require(s.mapped_bytes == 0 && s.committed_bytes == 0 && s.valid_bytes == 0,
                "prefill retained decode cache pages");
    }
    return s;
}

static sample_result run_checked(fixture_inputs & inputs, const std::string & name, fixture & f, executor & e, int layer, const batch & b,
        const std::vector<float> & expected, bool grouped = true, float linear_beta = 25,
        uint64_t expected_hits = 0, uint64_t expected_admitted = 0) {
    tensors meta(f, layer);
    sample_result r;
    require(expected.size() == size_t(b.tokens) * f.latent, "bad reference shape");
    std::vector<float> guarded(expected.size() + 2, -12345.25f);
    char error[512] = {};
    const int calls_before = f.resolve_calls;
    const auto begin = std::chrono::steady_clock::now();
    bool ok = true;
    if (grouped) {
        ok = kimi_cpu_prefill_run(e.prefill.get(), meta.role[0], meta.role[1], meta.role[2], b.x.data(),
            b.ids.data(), b.weights.data(), b.slots, b.tokens, 4, linear_beta, guarded.data() + 1,
            &r.stats, error, sizeof(error));
    } else {
        for (int t = 0; t < b.tokens && ok; ++t) {
            kimi_cpu_sink_stats s = {};
            ok = kimi_cpu_sink_run(e.sink.get(), meta.role[0], meta.role[1], meta.role[2], b.x.data() + size_t(t) * f.latent,
                b.ids.data() + size_t(t) * b.slots, b.weights.data() + size_t(t) * b.slots, b.slots,
                4, linear_beta, guarded.data() + 1 + size_t(t) * f.latent, &s, error, sizeof(error));
            add_stats(r.stats.stream, s);
        }
    }
    r.wall_us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - begin).count();
    require(ok, name + ": " + error);
    require(guarded.front() == -12345.25f && guarded.back() == -12345.25f, name + ": output canary overwritten");
    const auto counts = count_routes(f, layer, b, grouped);
    const auto & s = r.stats.stream;
    check_transport(inputs, s, true, grouped, expected_hits != 0);
    cache_snapshot(e.sink.get(), grouped && expected_hits == 0);
    require(expected_hits <= counts.experts, "invalid expected decode hit count");
    const uint64_t triplet = counts.experts ? counts.bytes / counts.experts : 0;
    require(s.experts == counts.experts && s.read_requests == 3 * (counts.experts - expected_hits) &&
            s.read_bytes == counts.bytes - expected_hits * triplet &&
            s.read_errors == 0 && s.fallback_count == 0, name + ": unique expert/read counters mismatch");
    require(s.demand_bytes == counts.bytes && s.cache_hits == expected_hits && s.cache_misses == counts.experts - expected_hits &&
            s.cache_admitted == expected_admitted && s.cache_admitted_bytes == expected_admitted * triplet,
            name + ": decode/prefill cache accounting differs");
    require(f.resolve_calls - calls_before == int(3 * counts.experts), name + ": resolver did not visit each role exactly once");
    if (grouped) {
        require(r.stats.routed_slots == counts.slots && r.stats.expert_token_pairs == counts.pairs &&
                r.stats.max_group_tokens == counts.max_group, name + ": group shape counters mismatch");
    }
    double square = 0, norm = 0;
    for (size_t i = 0; i < expected.size(); ++i) {
        const double v = guarded[i + 1], diff = v - expected[i];
        require(std::isfinite(v) && std::isfinite(expected[i]), name + ": nonfinite numerical result");
        r.max_abs = std::max(r.max_abs, std::abs(diff)); square += diff * diff; norm += double(expected[i]) * expected[i];
    }
    r.rel_l2 = std::sqrt(square / std::max(norm, 1e-30));
    const auto * bytes = reinterpret_cast<const uint8_t *>(guarded.data() + 1);
    for (size_t i = 0; i < expected.size() * sizeof(float); ++i) { r.hash ^= bytes[i]; r.hash *= 1099511628211ULL; }
    require(r.max_abs == 0, name + ": strict CPU reference equality failed: " + std::to_string(r.max_abs));
    return r;
}

static void print_sample(const std::string & name, const batch & b, const sample_result & r) {
    const auto & s = r.stats.stream;
    std::printf("case=%s tokens=%d slots=%d experts=%llu pairs=%llu max_group=%llu bytes=%llu reads=%llu "
        "wall_us=%.3f compute_us=%.3f wait_io_us=%.3f workspace_bytes=%llu max_abs=%.9g rel_l2=%.9g"
        " cache_hits=%llu cache_misses=%llu cache_admitted=%llu cache_copy_us=%.3f"
        " cache_fill_read_requests=%llu cache_fill_read_bytes=%llu ring_read_requests=%llu ring_read_bytes=%llu"
        " fixed_read_requests=%llu read_completions=%llu pending_reads=%llu cache_new_slot_bytes=%llu"
        " read_errors=%llu fallback_count=%llu status=PASS\n",
        name.c_str(), b.tokens, b.slots, (unsigned long long) s.experts, (unsigned long long) r.stats.expert_token_pairs,
        (unsigned long long) r.stats.max_group_tokens, (unsigned long long) s.read_bytes, (unsigned long long) s.read_requests,
        r.wall_us, s.compute_us, s.wait_io_us, (unsigned long long) r.stats.workspace_bytes, r.max_abs, r.rel_l2,
        (unsigned long long) s.cache_hits, (unsigned long long) s.cache_misses, (unsigned long long) s.cache_admitted, s.cache_copy_us,
        (unsigned long long) s.cache_fill_read_requests, (unsigned long long) s.cache_fill_read_bytes,
        (unsigned long long) s.ring_read_requests, (unsigned long long) s.ring_read_bytes,
        (unsigned long long) s.fixed_read_requests, (unsigned long long) s.read_completions,
        (unsigned long long) s.pending_reads, (unsigned long long) s.cache_new_slot_bytes,
        (unsigned long long) s.read_errors, (unsigned long long) s.fallback_count);
    std::fflush(stdout);
}

static void compare_case(fixture_inputs & inputs, const std::string & name, fixture & f, executor & e, int layer, const batch & b, float linear_beta = 25) {
    const auto expected = reference_batch(f, layer, b, linear_beta);
    print_sample(name, b, run_checked(inputs, name, f, e, layer, b, expected, true, linear_beta));
}

// A well-formed tensor of TYPE, so a refusal comes from the type and not from its layout.
static void retype(ggml_tensor * tensor, ggml_type type) {
    tensor->type = type;
    tensor->nb[0] = ggml_type_size(type);
    tensor->nb[1] = ggml_row_size(type, tensor->ne[0]);
    tensor->nb[2] = tensor->nb[1] * tensor->ne[1];
    tensor->nb[3] = tensor->nb[2] * tensor->ne[2];
}

static kimi_cpu_prefill_stats fail_case(fixture_inputs & inputs, const std::string & name, fixture & f, executor & e, batch b,
        int tokens_override = -1, int slots_override = -1, bool short_read = false, int metadata_fault = 0,
        bool allow_prefill_hits = false) {
    tensors meta(f, 1);
    if (metadata_fault == 1) { ++meta.role[0]->nb[1]; }
    if (metadata_fault == 2) { retype(meta.role[1], GGML_TYPE_F32); }
    if (metadata_fault == 3) { meta.role[2]->ne[0] += 256; }
    std::vector<float> output(size_t(std::max(b.tokens, 1)) * f.latent + 2, -77);
    char error[512] = {}; kimi_cpu_prefill_stats stats = {};
    const bool ok = kimi_cpu_prefill_run(e.prefill.get(), meta.role[0], meta.role[1], meta.role[2], b.x.data(),
        b.ids.data(), b.weights.data(), slots_override < 0 ? b.slots : slots_override,
        tokens_override < 0 ? b.tokens : tokens_override, 4, 25, output.data() + 1, &stats, error, sizeof(error));
    require(!ok && error[0] && stats.stream.fallback_count == 0, name + ": did not fail closed");
    if (metadata_fault == 2) {
        require(std::strstr(error, "not supported yet") && std::strstr(error, "f32"), name + ": did not name the type");
    }
    check_transport(inputs, stats.stream, false, true, allow_prefill_hits);
    cache_snapshot(e.sink.get(), true);
    require(output.front() == -77 && output.back() == -77, name + ": failed run overwrote output guard");
    if (short_read) { require(stats.stream.read_errors > 0, "short read not counted"); }
    std::printf("case=%s reads=%llu completions=%llu pending=%llu status=PASS error=%s\n", name.c_str(),
        (unsigned long long) stats.stream.read_requests, (unsigned long long) stats.stream.read_completions,
        (unsigned long long) stats.stream.pending_reads, error);
    return stats;
}

static void synthetic_tests(fixture_inputs & inputs, int threads, int depth) {
    fixture small; make_synthetic(small);
    executor e(inputs, small, threads, depth);
    for (int nt : { 1, 2, 25, 64 }) {
        auto b = synthetic_batch(small, nt, nt == 64 ? 16 : 8);
        if (nt == 64) {
            for (int t = 0; t < nt; ++t) { b.ids[size_t(t) * b.slots] = b.ids[size_t(t) * b.slots + 2] = 3; }
        }
        compare_case(inputs, "ub" + std::to_string(nt) + "_holes_duplicate_zero", small, e, 1, b);
    }
    compare_case(inputs, "ub65_slot_order", small, e, 1, synthetic_batch(small, 65, 16));
    auto full = synthetic_batch(small, 2, 16);
    compare_case(inputs, "full16_slots", small, e, 2, full);
    auto empty = synthetic_batch(small, 64);
    std::fill(empty.ids.begin(), empty.ids.end(), -7); std::fill(empty.weights.begin(), empty.weights.end(), 0);
    compare_case(inputs, "all_empty_ub64", small, e, 2, empty);
    compare_case(inputs, "no_slots_ub25", small, e, 1, synthetic_batch(small, 25, 0));
    auto mixed = synthetic_batch(small, 25);
    std::fill(mixed.ids.begin() + 8, mixed.ids.begin() + 16, -7);
    std::fill(mixed.weights.begin() + 8, mixed.weights.begin() + 16, 0);
    compare_case(inputs, "empty_token_inside_batch", small, e, 2, mixed);
    compare_case(inputs, "linear_beta_disabled", small, e, 1, synthetic_batch(small, 2), 0);
    for (int i = 0; i < 4; ++i) {
        compare_case(inputs, "layer_and_batch_reuse", small, e, i % 2 + 1, synthetic_batch(small, i % 2 ? 25 : 2));
    }
    fixture large; large.latent = 512; large.hidden = 256; make_synthetic(large);
    e.active = &large;
    compare_case(inputs, "grow_non_square_shape", large, e, 2, synthetic_batch(large, 25));
    e.active = &small;
    compare_case(inputs, "shrink_shape", small, e, 1, synthetic_batch(small, 2));
    // Exercise decode/grouped reuse of the borrowed ring and pool in both directions.
    auto one = synthetic_batch(small, 1);
    auto expected = reference_batch(small, 1, one);
    print_sample("decode_after_prefill", one, run_checked(inputs, "decode_after_prefill", small, e, 1, one, expected, false));
    compare_case(inputs, "prefill_after_decode", small, e, 2, synthetic_batch(small, 25));

    auto bad = synthetic_batch(small, 2);
    bad.ids[0] = small.experts; fail_case(inputs, "out_of_range_id", small, e, bad);
    bad = synthetic_batch(small, 2); bad.weights[0] = -1; fail_case(inputs, "negative_live_weight", small, e, bad);
    bad.weights[0] = std::numeric_limits<float>::infinity(); fail_case(inputs, "infinite_live_weight", small, e, bad);
    bad.weights[0] = std::numeric_limits<float>::quiet_NaN(); fail_case(inputs, "nan_live_weight", small, e, bad);
    bad = synthetic_batch(small, 2); bad.x[0] = std::numeric_limits<float>::infinity(); fail_case(inputs, "nonfinite_input", small, e, bad);
    bad = synthetic_batch(small, 2);
    fail_case(inputs, "zero_tokens", small, e, bad, 0);
    fail_case(inputs, "too_many_tokens", small, e, bad, 2049);
    fail_case(inputs, "too_many_slots", small, e, bad, -1, 17);
    fail_case(inputs, "noncontiguous_up_stride", small, e, bad, -1, -1, false, 1);
    fail_case(inputs, "wrong_gate_type", small, e, bad, -1, -1, false, 2);
    fail_case(inputs, "wrong_down_shape", small, e, bad, -1, -1, false, 3);
    const int direct_fd = small.fd;
    small.fd = ssd::platform::open_buffered(small.path.c_str());
    require(small.fd >= 0, "buffered test open failed");
    fail_case(inputs, "buffered_fd_rejected", small, e, bad);
    ssd::platform::close_fd(small.fd); small.fd = direct_fd;
    compare_case(inputs, "recovery_after_invalid_inputs", small, e, 2, synthetic_batch(small, 2));
    for (auto fault : { fixture::missing, fixture::unaligned_offset, fixture::wrong_size, fixture::past_eof }) {
        small.fault = fault;
        fail_case(inputs, "resolver_fault_" + std::to_string(fault), small, e, synthetic_batch(small, 25), -1, -1, fault == fixture::past_eof);
        small.fault = fixture::none;
        compare_case(inputs, "recovery_after_fault_" + std::to_string(fault), small, e, 2, synthetic_batch(small, 2));
    }
    auto late = synthetic_batch(small, 25);
    std::vector<int32_t> first_touch;
    for (int32_t id : late.ids) {
        if (id >= 0 && std::find(first_touch.begin(), first_touch.end(), id) == first_touch.end()) { first_touch.push_back(id); }
    }
    require(first_touch.size() > 2, "late fault fixture lacks valid preceding experts");
    small.fault = fixture::past_eof; small.fault_expert = first_touch.back();
    fail_case(inputs, "last_expert_short_read", small, e, late, -1, -1, true);
    small.fault = fixture::none; small.fault_expert = -1;
    compare_case(inputs, "recovery_after_last_expert_short_read", small, e, 2, synthetic_batch(small, 25));
    executor serial(inputs, small, 2, 1);
    compare_case(inputs, "one_compute_ring1", small, serial, 1, synthetic_batch(small, 25));
    std::printf("SYNTHETIC_CPU_PREFILL_OK\n");
}

static void configure_cache(executor & e, uint64_t bytes, const std::vector<int32_t> & layers, int quota) {
    char error[512] = {};
    require(kimi_cpu_sink_cache_configure(e.sink.get(), bytes, layers.data(), int(layers.size()), quota,
            error, sizeof(error)), std::string("fused cache configure: ") + error);
    require(cache_snapshot(e.sink.get(), true).budget_bytes == bytes, "fused cache budget was not retained");
}

static void begin_decode(executor & e, uint64_t epoch, uint64_t token) {
    char error[512] = {};
    require(kimi_cpu_sink_cache_begin(e.sink.get(), epoch, token, true, error, sizeof(error)),
            std::string("fused decode begin: ") + error);
}

static batch one_token(const fixture & f, const std::vector<int32_t> & ids, const std::vector<float> & weights) {
    require(ids.size() == weights.size(), "bad fused decode fixture");
    auto result = synthetic_batch(f, 1, int(ids.size()));
    result.ids = ids; result.weights = weights;
    return result;
}

static sample_result decode_cache_case(fixture_inputs & inputs, const std::string & name, fixture & f, executor & e, int layer,
        const batch & b, uint64_t hits, uint64_t admitted) {
    require(b.tokens == 1, "cache phase must represent one logical token");
    const auto expected = reference_batch(f, layer, b);
    auto result = run_checked(inputs, name, f, e, layer, b, expected, false, 25, hits, admitted);
    print_sample(name, b, result);
    return result;
}

static void require_cache_counters_unchanged(const kimi_cpu_sink_cache_stats & before, executor & e) {
    const auto after = cache_snapshot(e.sink.get(), true);
    require(after.budget_bytes == before.budget_bytes && after.capacity_bytes == before.capacity_bytes &&
            after.hits == before.hits && after.misses == before.misses && after.admitted_bytes == before.admitted_bytes,
            "grouped prefill altered decode cache policy statistics or configuration");
}

static bool address_sanitizer_build() {
#if defined(__has_feature)
#if __has_feature(address_sanitizer)
    return true;
#endif
#endif
#if defined(__SANITIZE_ADDRESS__)
    return true;
#else
    return false;
#endif
}

static void cache_retention_tests(fixture_inputs & inputs, int threads, int depth) {
    inputs.set("cache_policy", "count_decay2");
    inputs.set("cache_keep", "1");
    fixture f; make_synthetic(f);
    executor retained(inputs, f, threads, depth), baseline(inputs, f, threads, depth);
    const uint64_t unit = count_routes(f, 1, one_token(f, { 0 }, { 1 }), false).bytes;
    configure_cache(retained, 2 * unit, { 1 }, 2);
    auto selected = one_token(f, { 1, 2 }, { .4f, .6f });
    begin_decode(retained, 1, 1);
    decode_cache_case(inputs, "retain_seed", f, retained, 1, selected, 0, 2);
    const auto before = cache_snapshot(retained.sink.get());
    require(before.valid_bytes == 2 * unit, "retention seed did not fill");
    tensors meta(f, 1);
    for (int tokens : { 1, 25, 64 }) {
        auto b = synthetic_batch(f, tokens);
        std::vector<float> original(size_t(tokens) * f.latent), kept(original.size());
        kimi_cpu_prefill_stats a = {}, c = {}; char error[512] = {};
        require(kimi_cpu_prefill_run(baseline.prefill.get(), meta.role[0], meta.role[1], meta.role[2],
            b.x.data(), b.ids.data(), b.weights.data(), b.slots, b.tokens, 4, 25, original.data(), &a, error, sizeof(error)), error);
        require(kimi_cpu_prefill_run(retained.prefill.get(), meta.role[0], meta.role[1], meta.role[2],
            b.x.data(), b.ids.data(), b.weights.data(), b.slots, b.tokens, 4, 25, kept.data(), &c, error, sizeof(error)), error);
        require(!std::memcmp(original.data(), kept.data(), kept.size() * sizeof(float)), "retention changed prefill output bits");
        check_transport(inputs, c.stream, true, true);
        const auto after = cache_snapshot(retained.sink.get());
        require(after.valid_bytes == before.valid_bytes && after.hits == before.hits && after.misses == before.misses &&
            c.stream.read_bytes == a.stream.read_bytes, "prefill borrowed, admitted or released retained weights");
    }
    begin_decode(retained, 1, 1);
    selected.x[0] += .13f;
    decode_cache_case(inputs, "retain_changed_input_hits", f, retained, 1, selected, 2, 0);
    auto bad = synthetic_batch(f, 2);
    fail_case(inputs, "retain_invalid_prefill_releases", f, retained, bad, 0);
    begin_decode(retained, 1, 2);
    decode_cache_case(inputs, "retain_error_recovery_cold", f, retained, 1, selected, 0, 2);
    f.fault = fixture::past_eof; f.fault_expert = 1;
    fail_case(inputs, "retain_short_read_releases", f, retained, one_token(f, { 1 }, { 1 }), -1, -1, true);
    f.fault = fixture::none; f.fault_expert = -1;
    std::printf("CACHE_RETAIN_PREFILL_BIT_EXACT_OK tokens=1,25,64 ring=%d\n", depth);
}

static void cache_prefill_read_tests(fixture_inputs & inputs, int threads, int depth) {
    inputs.set("cache_policy", "count_decay2");
    inputs.set("cache_keep", "1");
    inputs.set("cache_prefill_read", "1");
    fixture f; make_synthetic(f);
    executor e(inputs, f, threads, depth);
    const uint64_t unit = count_routes(f, 1, one_token(f, { 0 }, { 1 }), false).bytes;
    configure_cache(e, 4 * unit, { 1 }, 4);
    const auto seed = one_token(f, { 1, 2, 3, 4 }, { .1f, .2f, .3f, .4f });
    begin_decode(e, 7, 1);
    decode_cache_case(inputs, "prefill_read_seed", f, e, 1, seed, 0, 4);
    const auto before = cache_snapshot(e.sink.get());

    const auto mixed = one_token(f, { 1, 2, 5, 6 }, { .1f, .2f, .3f, .4f });
    const auto expected = reference_batch(f, 1, mixed);
    const auto read = run_checked(inputs, "prefill_read_mixed", f, e, 1, mixed, expected, true, 25, 2, 0);
    print_sample("prefill_read_mixed", mixed, read);
    require(read.stats.ram_experts == 2 && read.stats.ram_routes == 2 && read.stats.ram_bytes == 2 * unit &&
            read.stats.ssd_experts == 2 && read.stats.ssd_routes == 2 && read.stats.ssd_bytes == 2 * unit &&
            read.stats.gpu_experts == 0 && read.stats.gpu_routes == 0,
            "read-only prefill source counters differ");
    const auto after = cache_snapshot(e.sink.get());
    require(after.valid_bytes == before.valid_bytes && after.hits == before.hits &&
            after.misses == before.misses && after.admitted_bytes == before.admitted_bytes &&
            after.evictions == before.evictions && after.prefill_hits == before.prefill_hits + 2 &&
            after.prefill_misses == before.prefill_misses + 2 &&
            after.prefill_hit_bytes == before.prefill_hit_bytes + 2 * unit &&
            after.prefill_leases == before.prefill_leases + 2,
            "read-only prefill changed decode policy state or source accounting");

    begin_decode(e, 7, 2);
    decode_cache_case(inputs, "prefill_read_decode_still_hot", f, e, 1,
        one_token(f, { 1, 2 }, { .4f, .6f }), 2, 0);

    f.fault = fixture::past_eof; f.fault_expert = 5;
    fail_case(inputs, "prefill_read_error_releases_leases", f, e,
        one_token(f, { 1, 5 }, { .5f, .5f }), -1, -1, true, 0, true);
    f.fault = fixture::none; f.fault_expert = -1;
    configure_cache(e, 2 * unit, { 1 }, 2);
    begin_decode(e, 8, 1);
    decode_cache_case(inputs, "prefill_read_error_recovery", f, e, 1,
        one_token(f, { 1, 5 }, { .5f, .5f }), 0, 2);
    std::printf("CACHE_PREFILL_READ_ONLY_OK hits=2 misses=2 lease_error_recovery=1\n");
}

static void cache_lifecycle_tests(fixture_inputs & inputs, int threads, int depth) {
    fixture f; f.experts = 32; make_synthetic(f);
    executor e(inputs, f, threads, depth);
    const uint64_t unit = count_routes(f, 1, one_token(f, { 0 }, { 1 }), false).bytes;
    require(unit == 86016, "fused fixture must include 4096-aligned Q3_K role padding");
    const auto grouped = synthetic_batch(f, 64, 16);
    // Allocate the shared ring, grouped workspace and independent reference
    // first. Later Anonymous deltas must measure release, not workspace growth.
    compare_case(inputs, "fused_prefill_workspace_warm", f, e, 1, grouped);
    const auto grouped_expected = reference_batch(f, 1, grouped);
    configure_cache(e, 64 * unit, { 1, 2 }, 32);
    std::vector<int32_t> all_ids(32);
    for (int id = 0; id < 32; ++id) { all_ids[id] = id; }
    const auto full = one_token(f, all_ids, std::vector<float>(32, 1.0f / 32));
    begin_decode(e, 1, 1);
    const auto first = decode_cache_case(inputs, "fused_decode_fill_layer1", f, e, 1, full, 0, 32);
    decode_cache_case(inputs, "fused_decode_fill_layer2", f, e, 2, full, 0, 32);
    begin_decode(e, 1, 2);
    decode_cache_case(inputs, "fused_decode_hot_layer1", f, e, 1, full, 32, 0);
    decode_cache_case(inputs, "fused_decode_hot_layer2", f, e, 2, full, 32, 0);
    const auto resident = cache_snapshot(e.sink.get());
    require(resident.valid_bytes == 64 * unit && resident.committed_bytes == 64 * unit,
            "fused physical release fixture did not fill cache");
    const uint64_t before = anonymous_bytes();
    // Deliberately do not call cache_begin(false): the public prefill API must
    // release before validating/allocating its own workspace and before stream.
    const auto prefill = run_checked(inputs, "fused_grouped_releases_hot_cache", f, e, 1, grouped, grouped_expected);
    const uint64_t after = anonymous_bytes();
    print_sample("fused_grouped_releases_hot_cache", grouped, prefill);
    require_cache_counters_unchanged(resident, e);
    std::printf("measurement=fused_prefill_physical_release payload=%llu anonymous_before=%llu anonymous_after=%llu anonymous_drop=%lld\n",
        (unsigned long long) resident.committed_bytes, (unsigned long long) before, (unsigned long long) after,
        static_cast<long long>(before) - static_cast<long long>(after));
    std::fflush(stdout);
    require(prefill.stats.stream.ring_bytes == first.stats.stream.ring_bytes,
            "decode/prefill did not retain the same bounded ring capacity");
    if (address_sanitizer_build()) {
        // ASan owns additional anonymous allocator/shadow pages, so a process
        // RSS delta is not a payload-only oracle. The normal build enforces the
        // physical gate; ASan still checks cache release, math, CQEs and UAF.
        std::printf("case=fused_prefill_physical_release payload=%llu anonymous_before=%llu anonymous_after=%llu"
            " status=SKIP reason=asan_process_rss\n",
            (unsigned long long) resident.committed_bytes, (unsigned long long) before, (unsigned long long) after);
    } else {
        require(before >= after && before - after >= resident.committed_bytes - test_resources::anonymous_noise_bytes,
                "grouped prefill zeroed cache accounting without releasing Anonymous pages");
        std::printf("case=fused_prefill_physical_release payload=%llu anonymous_before=%llu anonymous_after=%llu status=PASS\n",
            (unsigned long long) resident.committed_bytes, (unsigned long long) before, (unsigned long long) after);
    }
    begin_decode(e, 1, 3);
    decode_cache_case(inputs, "fused_decode_after_prefill_cold", f, e, 1, full, 0, 32);
    begin_decode(e, 1, 4);
    decode_cache_case(inputs, "fused_decode_after_prefill_hot", f, e, 1, full, 32, 0);

    configure_cache(e, 3 * unit, { 1 }, 3);
    begin_decode(e, 2, 1);
    decode_cache_case(inputs, "fused_mixed_seed", f, e, 1, one_token(f, { 1 }, { 1 }), 0, 1);
    begin_decode(e, 2, 2);
    const auto mixed = decode_cache_case(inputs, "fused_hit_admit_duplicate_bypass", f, e, 1,
        one_token(f, { 1, 2, 2, 3, 4, 1, 3, 5 }, { .1f, .15f, .05f, .25f, .1f, .2f, .05f, .1f }), 2, 2);
    require(mixed.stats.stream.cache_fill_read_requests == 6 && mixed.stats.stream.ring_read_requests == 12,
            "fused mixed decode did not cover direct cache and ring targets");
    compare_case(inputs, "fused_grouped_after_mixed_targets", f, e, 2, synthetic_batch(f, 25));

    const auto four = one_token(f, { 1, 2, 3, 4 }, { .1f, .2f, .3f, .4f });
    configure_cache(e, 4 * unit, { 1 }, 4);
    for (int phase = 0; phase < 3; ++phase) {
        const std::string suffix = std::to_string(phase);
        begin_decode(e, 3 + phase, 1);
        decode_cache_case(inputs, "fused_edge_cold_" + suffix, f, e, 1, four, 0, 4);
        begin_decode(e, 3 + phase, 2);
        decode_cache_case(inputs, "fused_edge_hot_" + suffix, f, e, 1, four, 4, 0);
        auto edge = synthetic_batch(f, phase == 0 ? 64 : phase == 1 ? 25 : 1, phase == 1 ? 0 : 8);
        if (phase == 0) {
            std::fill(edge.ids.begin(), edge.ids.end(), -7);
            std::fill(edge.weights.begin(), edge.weights.end(), 0);
        }
        const auto snapshot = cache_snapshot(e.sink.get());
        compare_case(inputs, phase == 0 ? "fused_empty_union_clears_cache" : phase == 1 ?
            "fused_zero_slots_clears_cache" : "fused_one_token_prefill_clears_cache", f, e, 1, edge);
        require_cache_counters_unchanged(snapshot, e);
        begin_decode(e, 3 + phase, 3);
        decode_cache_case(inputs, "fused_edge_decode_cold_" + suffix, f, e, 1, four, 0, 4);
        begin_decode(e, 3 + phase, 4);
        decode_cache_case(inputs, "fused_edge_decode_hot_" + suffix, f, e, 1, four, 4, 0);
    }

    for (int role = 0; role < 3; ++role) {
        const std::string suffix = std::to_string(role);
        begin_decode(e, 10 + role, 1);
        decode_cache_case(inputs, "fused_short_seed_" + suffix, f, e, 1, four, 0, 4);
        auto bad = synthetic_batch(f, 25);
        f.fault = fixture::short_tail; f.fault_expert = bad.ids[0]; f.fault_role = role;
        const auto failed = fail_case(inputs, "fused_prefill_short_role_" + suffix, f, e, bad, -1, -1, true);
        f.fault = fixture::none; f.fault_expert = -1; f.fault_role = -1;
        require(failed.stream.ring_read_bytes >= 4096 && failed.stream.ring_read_requests >= uint64_t(depth > 1 ? 6 : 3),
                "fused short read did not include a positive CQE and other in-flight roles");
        if (role == 2) {
            // The prefill context only borrows the sink. Destroying it after a
            // drained error must leave decode usable and allow a new borrower.
            e.prefill.reset();
            begin_decode(e, 20, 1);
            decode_cache_case(inputs, "fused_decode_after_prefill_destroy", f, e, 1, four, 0, 4);
            char error[512] = {};
            e.prefill.reset(kimi_cpu_prefill_create(e.sink.get(), error, sizeof(error)));
            require(bool(e.prefill), std::string("fused prefill recreate: ") + error);
        }
        auto empty = synthetic_batch(f, 2);
        std::fill(empty.ids.begin(), empty.ids.end(), -1);
        compare_case(inputs, "fused_short_then_empty_union_" + suffix, f, e, 1, empty);
        begin_decode(e, 30 + role, 1);
        decode_cache_case(inputs, "fused_short_decode_cold_" + suffix, f, e, 1, four, 0, 4);
        begin_decode(e, 30 + role, 2);
        decode_cache_case(inputs, "fused_short_decode_hot_" + suffix, f, e, 1, four, 4, 0);
        compare_case(inputs, "fused_short_grouped_recovery_" + suffix, f, e, 2, synthetic_batch(f, 25));
    }

    auto late = synthetic_batch(f, 25);
    std::vector<int32_t> first_touch;
    for (int32_t id : late.ids) {
        if (id >= 0 && std::find(first_touch.begin(), first_touch.end(), id) == first_touch.end()) { first_touch.push_back(id); }
    }
    begin_decode(e, 40, 1);
    decode_cache_case(inputs, "fused_late_short_seed", f, e, 1, four, 0, 4);
    f.fault = fixture::short_tail; f.fault_expert = first_touch.back(); f.fault_role = 2;
    const auto failed = fail_case(inputs, "fused_last_expert_short_read", f, e, late, -1, -1, true);
    f.fault = fixture::none; f.fault_expert = -1; f.fault_role = -1;
    require(failed.stream.read_requests == 3 * first_touch.size(), "last-expert failure did not submit the full union");
    compare_case(inputs, "fused_last_expert_short_recovery", f, e, 2, synthetic_batch(f, 2));

    for (int invalid = 0; invalid < 3; ++invalid) {
        begin_decode(e, 50 + invalid, 1);
        decode_cache_case(inputs, "fused_invalid_seed_" + std::to_string(invalid), f, e, 1, four, 0, 4);
        auto bad = synthetic_batch(f, 2);
        if (invalid == 2) { bad.x[0] = std::numeric_limits<float>::infinity(); }
        const auto result = fail_case(inputs, "fused_early_invalid_clears_cache_" + std::to_string(invalid), f, e, bad,
            invalid == 0 ? 0 : -1, -1, false, invalid == 1 ? 3 : 0);
        require(result.stream.read_requests == 0, "early prefill validation failure unexpectedly submitted IO");
        begin_decode(e, 50 + invalid, 2);
        decode_cache_case(inputs, "fused_early_invalid_decode_recovery_" + std::to_string(invalid), f, e, 1, four, 0, 4);
    }
    std::printf("FUSED_CPU_PREFILL_CACHE_OK threads=%d ring_experts=%d\n", threads, depth);
}

struct options {
    int threads = 16, ring = 8, start = 0, tokens = 64, repeats = 3, warmup = 1;
    std::string pack, prefix, jsonl, mode = "both";
};

static batch load_real(fixture & f, int & layer, const options & o) {
    batch b;
    int available = 0;
    std::ifstream route(o.prefix + ".txt");
    require(bool(route >> layer >> f.latent >> f.hidden >> available >> b.slots), "invalid batch fixture header");
    require(layer >= 1 && layer <= 92 && f.latent == 3584 && f.hidden == 3072 &&
            available >= 1 && available <= 4096 && b.slots >= 1 && b.slots <= 16,
            "invalid real batch dimensions");
    require(o.start >= 0 && o.tokens >= 1 && o.tokens <= 64 && o.start + o.tokens <= available, "fixture selection out of bounds");
    b.tokens = o.tokens; b.ids.resize(size_t(b.tokens) * b.slots); b.weights.resize(b.ids.size());
    for (int t = 0; t < available; ++t) {
        for (int s = 0; s < b.slots; ++s) {
            int32_t id; float weight;
            require(bool(route >> id >> weight) && id < 896 && std::isfinite(weight) && weight >= 0, "invalid/truncated route fixture");
            if (t >= o.start && t < o.start + o.tokens) {
                const size_t i = size_t(t - o.start) * b.slots + s;
                b.ids[i] = id; b.weights[i] = weight;
            }
        }
    }
    std::string extra; require(!(route >> extra), "unexpected trailing fixture data");
    std::ifstream input(o.prefix + ".f32", std::ios::binary | std::ios::ate);
    require(bool(input) && input.tellg() == std::streamoff(size_t(available) * f.latent * sizeof(float)), "input fixture length mismatch");
    input.seekg(size_t(o.start) * f.latent * sizeof(float)); b.x.resize(size_t(b.tokens) * f.latent);
    input.read(reinterpret_cast<char *>(b.x.data()), b.x.size() * sizeof(float)); require(bool(input), "short input fixture read");
    require(std::all_of(b.x.begin(), b.x.end(), [](float x) { return std::isfinite(x); }), "nonfinite real input");
    f.experts = 896;
    std::set<int32_t> live;
    for (int32_t id : b.ids) { if (id >= 0) { live.insert(id); } }
    require(!live.empty(), "real fixture has no experts");
    std::ifstream pack(o.pack, std::ios::binary | std::ios::ate); require(bool(pack), "real pack open failed");
    const uint64_t pack_size = uint64_t(pack.tellg()); pack.seekg(0);
    char magic[16]; uint32_t version, header; uint64_t count, data_start;
    pack.read(magic, 16); pack.read(reinterpret_cast<char *>(&version), 4); pack.read(reinterpret_cast<char *>(&header), 4);
    pack.read(reinterpret_cast<char *>(&count), 8); pack.read(reinterpret_cast<char *>(&data_start), 8);
    require(bool(pack) && !std::memcmp(magic, "GGMLMOEPACKv1", 12) && version == 1 && header == 40 &&
            count < 1000000 && data_start >= 40 + 152 * count && data_start <= pack_size, "bad pack header");
    for (uint64_t i = 0; i < count; ++i) {
        char label[129] = {}; int32_t id; uint32_t reserved; uint64_t offset, bytes;
        pack.read(label, 128); pack.read(reinterpret_cast<char *>(&id), 4); pack.read(reinterpret_cast<char *>(&reserved), 4);
        pack.read(reinterpret_cast<char *>(&offset), 8); pack.read(reinterpret_cast<char *>(&bytes), 8);
        require(bool(pack), "truncated pack index");
        if (!live.count(id)) { continue; }
        for (int role = 0; role < 3; ++role) {
            if (label != tensor_name(layer, role)) { continue; }
            require(bytes == 4730880 && offset % 4096 == 0 && offset >= data_start &&
                    offset <= pack_size && bytes <= pack_size - offset, "unexpected real role range");
            entry e; e.offset = offset; e.raw_bytes = e.read_bytes = bytes;
            require(f.entries.emplace(key { label, id }, std::move(e)).second, "duplicate pack entry");
        }
    }
    require(f.entries.size() == 3 * live.size(), "missing real expert roles");
    f.path = o.pack; f.fd = test_resources::open_direct(f.path);
    return b;
}

static void write_json(std::ostream & out, const sample_result & r, const batch & b, const options & o,
        int layer, const std::string & mode, const std::string & phase, int iteration, double reference_us) {
    const auto & s = r.stats.stream;
    out << std::setprecision(12) << "{\"schema\":1,\"mode\":\"" << mode << "\",\"phase\":\"" << phase
        << "\",\"iteration\":" << iteration << ",\"layer\":" << layer << ",\"start_token\":" << o.start
        << ",\"tokens\":" << b.tokens << ",\"slots\":" << b.slots << ",\"threads\":" << o.threads << ",\"ring_experts\":" << o.ring
        << ",\"experts\":" << s.experts << ",\"routed_slots\":" << r.stats.routed_slots
        << ",\"expert_token_pairs\":" << r.stats.expert_token_pairs << ",\"max_group_tokens\":" << r.stats.max_group_tokens
        << ",\"read_bytes\":" << s.read_bytes << ",\"read_requests\":" << s.read_requests << ",\"ring_bytes\":" << s.ring_bytes
        << ",\"workspace_bytes\":" << r.stats.workspace_bytes << ",\"wall_us\":" << r.wall_us << ",\"total_us\":" << s.total_us
        << ",\"io_us\":" << s.io_us << ",\"compute_us\":" << s.compute_us << ",\"wait_io_us\":" << s.wait_io_us
        << ",\"input_quantize_us\":" << s.input_quantize_us << ",\"hidden_quantize_us\":" << s.hidden_quantize_us
        << ",\"up_gate_us\":" << s.up_gate_us << ",\"down_reduce_us\":" << s.down_reduce_us
        << ",\"read_errors\":" << s.read_errors << ",\"fallback_count\":" << s.fallback_count
        << ",\"max_abs\":" << r.max_abs << ",\"rel_l2\":" << r.rel_l2 << ",\"output_hash\":\"" << std::hex << r.hash << std::dec
        << "\",\"reference_once_us\":" << reference_us << ",\"status\":\"PASS\"}\n";
    out.flush(); require(bool(out), "JSONL write failed");
}

static void real_test(fixture_inputs & inputs, const options & o) {
    fixture f; int layer = 0;
    auto b = load_real(f, layer, o);
    const auto begin = std::chrono::steady_clock::now();
    const auto expected = reference_batch(f, layer, b);
    const double reference_us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - begin).count();
    std::printf("reference=independent_scalar_cpu tokens=%d wall_us=%.3f excluded_from_benchmark=1\n", b.tokens, reference_us);
    std::fflush(stdout);
    executor e(inputs, f, o.threads, o.ring);
    std::ofstream json;
    if (!o.jsonl.empty()) { json.open(o.jsonl, std::ios::out | std::ios::trunc); require(bool(json), "JSONL open failed"); }
    // Reference is computed exactly once. Alternating grouped/per-token order
    // balances drift, but these are single-layer transport measurements, not TTFT.
    for (int i = 0; i < 1 + o.warmup + o.repeats; ++i) {
        const std::string phase = i == 0 ? "first_after_reference" : i <= o.warmup ? "warmup" : "steady";
        const bool first_grouped = (i % 2 == 0);
        for (bool grouped : { first_grouped, !first_grouped }) {
            const std::string mode = grouped ? "grouped" : "per-token";
            if (o.mode != "both" && o.mode != mode) { continue; }
            const auto r = run_checked(inputs, mode + "_" + phase, f, e, layer, b, expected, grouped);
            print_sample(mode + "_" + phase, b, r);
            if (json.is_open()) { write_json(json, r, b, o, layer, mode, phase, i, reference_us); }
        }
    }
    std::printf("REAL_CPU_PREFILL_OK single_layer_only=1 gpu_validation=not_run\n");
}

int main(int argc, char ** argv) {
    fixture_inputs inputs;
    try {
        options o; bool ring_set = false, cache_lifecycle = false, cache_retention = false,
            cache_prefill_read = false;
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            auto next = [&]() { require(i + 1 < argc, "missing option argument"); return std::string(argv[++i]); };
            auto number = [&]() { const std::string s = next(); size_t end = 0; const int n = std::stoi(s, &end); require(end == s.size(), "invalid integer option"); return n; };
            if (arg == "--fixture") { inputs.argument(next()); }
            else if (arg == "--threads") { o.threads = number(); }
            else if (arg == "--ring-experts") { o.ring = number(); ring_set = true; }
            else if (arg == "--real-pack") { o.pack = next(); }
            else if (arg == "--fixture-prefix") { o.prefix = next(); }
            else if (arg == "--start-token") { o.start = number(); }
            else if (arg == "--tokens") { o.tokens = number(); }
            else if (arg == "--repeats") { o.repeats = number(); }
            else if (arg == "--warmup") { o.warmup = number(); }
            else if (arg == "--mode") { o.mode = next(); }
            else if (arg == "--jsonl") { o.jsonl = next(); }
            else if (arg == "--cache-lifecycle") { cache_lifecycle = true; }
            else if (arg == "--cache-retention") { cache_retention = true; }
            else if (arg == "--cache-prefill-read") { cache_prefill_read = true; }
            else { throw std::runtime_error("usage: test-ssd-cpu-prefill [--cache-lifecycle] [--threads N --ring-experts N] [--fixture KEY=VALUE] [--real-pack FILE --fixture-prefix PREFIX --start-token N --tokens 1..64 --mode grouped|per-token|both --warmup N --repeats N --jsonl FILE]"); }
        }
        require(o.threads >= 2 && o.threads <= 128 && o.ring >= 1 && o.ring <= 32 &&
                o.repeats >= 1 && o.repeats <= 1000 && o.warmup >= 0 && o.warmup <= 1000, "invalid thread/ring/repeat count");
        require(o.mode == "both" || o.mode == "grouped" || o.mode == "per-token", "invalid benchmark mode");
        test_resources::cpu_participants(o.threads - 1, 1);
        test_resources::io_ring();
        if (cache_lifecycle || cache_retention || cache_prefill_read) { test_resources::finite_envelope(); }
        if (cache_prefill_read) {
            cache_prefill_read_tests(inputs, o.threads, ring_set ? o.ring : 2);
        } else if (cache_retention) {
            cache_retention_tests(inputs, o.threads, ring_set ? o.ring : 2);
        } else if (cache_lifecycle) {
            require(o.pack.empty() && o.prefix.empty() && o.jsonl.empty(), "cache lifecycle cannot run a real-pack benchmark");
            cache_lifecycle_tests(inputs, o.threads, ring_set ? o.ring : 2);
        } else if (o.pack.empty()) {
            require(o.prefix.empty() && o.jsonl.empty(), "real fixture/JSONL requires --real-pack");
            synthetic_tests(inputs, o.threads, ring_set ? o.ring : 2);
        } else { require(!o.prefix.empty(), "real pack requires fixture prefix"); real_test(inputs, o); }
        if (cache_lifecycle && address_sanitizer_build()) {
            throw test_resources::unavailable("physical-release RSS oracle unavailable under ASan; other cache assertions executed");
        }
        std::printf("CPU_PREFILL_TEST_OK\n"); return 0;
    } catch (const std::exception & e) { return test_resources::report("CPU_PREFILL_TEST", e); }
}
