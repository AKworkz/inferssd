#include "explicit-fixture-inputs.h"
// Independent CPU-only direct-read, read-ring and numerical coverage for the Kimi sink.
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ssd/cpu-sink.h"
#include "ssd/cpu-prefill.h"
#include "ggml-ssd/settings/settings.h"
#include "ggml-ssd/gpu/kimi-cache-location-api.h"
#include <thread>
#include <array>
#include <nlohmann/json.hpp>
#include <algorithm>
#include <cmath>
#include <chrono>
#include <cerrno>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <map>
#include <iomanip>
#include <sstream>
#include <memory>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>
#include "test-resources.h"

#if defined(__linux__)
#include <linux/capability.h>
#include <unistd.h>
#include <sys/resource.h>
#include <sys/syscall.h>
#include <sys/vfs.h>
#include <sys/wait.h>
#include "../../ggml/src/ggml-ssd/host/ledger.h"
#endif

using test_resources::anonymous_bytes;

static void require(bool ok, const std::string & what) { if (!ok) { throw std::runtime_error(what); } }
static size_t aligned(size_t n) { return (n + 4095) & ~size_t(4095); }
static std::string tensor_name(int layer, int role) {
    static const char * roles[] = { "up", "gate", "down" };
    return "blk." + std::to_string(layer) + ".ffn_" + roles[role] + "_exps.weight";
}

#if defined(__linux__)
// Linux only: the hold interposes libc's pread by symbol name and recognizes /proc/meminfo, the
// Linux ledger's system term. The cases that need it stay Linux-only.
// pread, ahead of libc's in this process (the sink's library calls it through the dynamic linker): a read of an armed
// (fd, offset) waits in it until the test lets it go, so that a prefetch read runs past windows for as long as a test
// needs, without timing assumptions. before_meminfo runs once, at the next read of /proc/meminfo: the host ledger
// reads it once a measure of the room, after the cgroup's files. Every other read passes through at once.
struct pread_hold {
    std::mutex mutex;
    std::condition_variable cv;
    int fd = -1;
    std::vector<uint64_t> offsets;
    bool armed = false;
    int waiting = 0;
    std::function<void()> before_meminfo;
    static pread_hold & get() { static pread_hold & h = *new pread_hold; return h; } // outlives every thread
};
static bool reads_meminfo(int fd) {
    char target[32] = {};
    const std::string link = "/proc/self/fd/" + std::to_string(fd);
    const ssize_t n = readlink(link.c_str(), target, sizeof(target) - 1);
    return n > 0 && !std::strcmp(target, "/proc/meminfo");
}
static ssize_t held_pread(int fd, void * buffer, size_t bytes, int64_t offset) {
    auto & h = pread_hold::get();
    std::function<void()> hook;
    {
        std::unique_lock<std::mutex> lock(h.mutex);
        if (h.before_meminfo && reads_meminfo(fd)) { hook.swap(h.before_meminfo); }
        if (h.armed && fd == h.fd && std::find(h.offsets.begin(), h.offsets.end(), uint64_t(offset)) != h.offsets.end()) {
            ++h.waiting;
            h.cv.notify_all();
            h.cv.wait(lock, [&] { return !h.armed; });
            --h.waiting;
        }
    }
    if (hook) { hook(); }
    return syscall(SYS_pread64, fd, buffer, bytes, offset);
}
// both names, whichever one the library was built to call (_FILE_OFFSET_BITS)
extern "C" ssize_t test_held_pread(int, void *, size_t, int64_t) __asm__("pread");
extern "C" ssize_t test_held_pread64(int, void *, size_t, int64_t) __asm__("pread64");
extern "C" ssize_t test_held_pread(int fd, void * buffer, size_t bytes, int64_t offset) { return held_pread(fd, buffer, bytes, offset); }
extern "C" ssize_t test_held_pread64(int fd, void * buffer, size_t bytes, int64_t offset) { return held_pread(fd, buffer, bytes, offset); }
// Holds the reads of `offsets` in `fd` for its scope; its end, or release(), lets them go.
struct pread_hold_guard {
    pread_hold_guard(int fd, std::vector<uint64_t> offsets) {
        auto & h = pread_hold::get();
        std::lock_guard<std::mutex> lock(h.mutex);
        h.fd = fd; h.offsets = std::move(offsets); h.armed = true; h.waiting = 0;
    }
    ~pread_hold_guard() { release(); }
    // until n reads wait in the hold
    void wait(int n) {
        auto & h = pread_hold::get();
        std::unique_lock<std::mutex> lock(h.mutex);
        require(h.cv.wait_for(lock, std::chrono::seconds(10), [&] { return h.waiting >= n; }),
                "the held reads did not start (" + std::to_string(h.waiting) + " of " + std::to_string(n) + ")");
    }
    void release() {
        auto & h = pread_hold::get();
        { std::lock_guard<std::mutex> lock(h.mutex); h.armed = false; }
        h.cv.notify_all();
    }
};
#endif

struct entry {
    uint64_t offset = 0;
    size_t raw_bytes = 0, read_bytes = 0;
    std::vector<uint8_t> data;
};
using key = std::pair<std::string, int32_t>;
struct fixture {
    int latent = 256, hidden = 256, experts = 20, fd = -1;
    ggml_type type = GGML_TYPE_Q3_K, down_type = GGML_TYPE_Q3_K; // up/gate and down payload types
    std::string path;
    bool remove_on_destroy = false;
    std::map<key, entry> entries;
    enum fault_kind { none, missing, unaligned_offset, wrong_size, past_eof, short_tail } fault = none;
    int32_t fault_expert = -1; // -1 faults every expert; otherwise only this ID
    int fault_role = -1;      // -1 faults every role; 0/1/2 selects up/gate/down
    int resolve_calls = 0;
    bool native_ranges = false;
    std::vector<int32_t> stored; // the experts the file holds; empty: all of them
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
        out->data_offset = 0; out->payload_nbytes = 0;
        if (f.native_ranges) {
            out->data_offset = out->offset % 4096;
            out->payload_nbytes = it->second.raw_bytes;
            out->offset -= out->data_offset;
            out->nbytes = aligned(out->data_offset + out->payload_nbytes);
        }
        if (fault == unaligned_offset) { ++out->offset; }
        if (fault == wrong_size) { out->nbytes += 4096; }
        if (fault == past_eof) { out->offset = uint64_t(1) << 50; }
        if (fault == short_tail) {
            uint64_t size = 0;
            if (!ssd::platform::file_size(f.fd, size) || size < 4096) { return false; }
            out->offset = size - 4096; // positive but incomplete CQE
        }
        return true;
    }
};

static void put_bytes(std::ofstream & f, const void * data, size_t n) {
    f.write(static_cast<const char *>(data), n); require(bool(f), "fixture write failed");
}

static void make_synthetic(fixture & f, uint32_t seed = 20260908) {
    f.path = test_resources::temp_file("kimi-cpu-sink-test"); f.remove_on_destroy = true;
    std::mt19937 rng(seed);
    std::normal_distribution<float> sample(0.0f, 0.06f);
    // Keep the full on-disk GGMLMOEPACKv1 layout, including index and padding.
    for (int layer : { 1, 2 }) {
        for (int role = 0; role < 3; ++role) {
            const int cols = role == 2 ? f.hidden : f.latent;
            const int rows = role == 2 ? f.latent : f.hidden;
            const ggml_type type = role == 2 ? f.down_type : f.type;
            const size_t bytes = ggml_row_size(type, cols) * rows;
            const std::vector<float> importance(
                ggml_quantize_requires_imatrix(type) ? size_t(cols) : 0, 1.0f);
            for (int e = 0; e < f.experts; ++e) {
                if (!f.stored.empty() && std::find(f.stored.begin(), f.stored.end(), e) == f.stored.end()) { continue; }
                std::vector<float> values(size_t(cols) * rows);
                for (auto & v : values) { v = sample(rng); }
                entry item; item.raw_bytes = bytes; item.read_bytes = aligned(bytes); item.data.resize(bytes);
                require(ggml_quantize_chunk(type, values.data(), item.data.data(), 0, rows, cols,
                        importance.empty() ? nullptr : importance.data()) == bytes,
                        "fixture quantization bytes mismatch");
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
            role[r] = ggml_new_tensor_3d(ctx, r == 2 ? f.down_type : f.type, r == 2 ? f.hidden : f.latent,
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
            matrices[role] = ggml_new_tensor_2d(ctx, role == 2 ? f.down_type : f.type, role == 2 ? f.hidden : f.latent, role == 2 ? f.latent : f.hidden);
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

static sink_ptr make_sink(fixture_inputs & inputs, fixture & f, int threads, int depth) {
    test_resources::cpu_participants(threads - 1, 1);
    char error[512] = {};
    auto sink = inputs.create(threads, depth, fixture::resolve, &f, error, sizeof(error));
    require(bool(sink), std::string("create: ") + error);
    return sink;
}

struct sample_result {
    kimi_cpu_sink_stats stats = {};
    double max_abs = 0, rel_l2 = 0, wall_us = 0, user_us = 0, system_us = 0;
    long voluntary_switches = 0, involuntary_switches = 0, minor_faults = 0, major_faults = 0, max_rss_kib = 0;
    uint64_t output_hash = 1469598103934665603ULL;
};

static void check_io_targets(fixture_inputs & inputs, const kimi_cpu_sink_stats & stats, bool succeeded) {
    require(stats.cache_fill_read_requests + stats.ring_read_requests == stats.read_requests &&
            stats.cache_fill_read_bytes + stats.ring_read_bytes == stats.read_bytes,
            "cache/ring transport counters do not sum to total IO");
    require(stats.read_completions == stats.read_requests && stats.pending_reads == 0,
            "run returned before every submitted read CQE was consumed");
    const char * mode = inputs.get("read_fixed");
    const bool fixed = mode && !std::strcmp(mode, "1");
    require(stats.fixed_read_requests == (fixed ? stats.ring_read_requests : 0),
            "READ_FIXED was used for cache payload or omitted for a bypass ring read");
    require(stats.cache_copy_us == 0, "direct cache admission performed a payload copy");
    if (succeeded) {
        // Slots keep the largest role geometry seen (native GGUF roles differ by a page across layers), so an
        // admission reads its three roles into a slot at least as large: reads fill at most the admitted stride.
        require(stats.cache_fill_read_requests == 3 * stats.cache_admitted &&
                stats.cache_fill_read_bytes <= stats.cache_admitted_bytes &&
                (stats.cache_admitted == 0) == (stats.cache_fill_read_bytes == 0),
                "successful direct cache reads differ from published admissions");
        require(stats.cache_new_slot_bytes <= stats.cache_admitted_bytes,
                "first-touch reservation exceeds successful direct admissions");
    }
}

// The RAM cache's slot for a triplet of reads: a slot that gives its own pages back (elastic or partial reclaim)
// starts on a page, which makes it larger only on pages above 4 KiB.
static uint64_t cache_slot_bytes(const fixture_inputs & inputs, uint64_t triplet) {
    const auto on = [&](const char * key) { const char * value = inputs.get(key); return value && !std::strcmp(value, "1"); };
    if (!on("cache_elastic") && !on("cache_partial_reclaim")) { return triplet; }
    const uint64_t page = ssd::platform::page_size();
    return (triplet + page - 1) / page * page;
}

static sample_result run_checked(fixture_inputs & inputs, const char * name, fixture & f, kimi_cpu_sink * sink, tensors & meta,
        const std::vector<float> & input, const std::vector<int32_t> & ids,
        const std::vector<float> & weights, const std::vector<float> & expected, int layer, float linear_beta,
        size_t expected_hits = 0, int expected_admitted = -1, std::vector<float> * exact_output = nullptr,
        const kimi_cpu_sink_activation * activation = nullptr) {
    sample_result result;
    std::vector<float> guarded(f.latent + 2, -12345.25f); char error[512] = {};
    const auto before = test_resources::usage();
    const auto begin = std::chrono::steady_clock::now();
    bool ok = activation ?
        kimi_cpu_sink_run_activation(sink, meta.role[0], meta.role[1], meta.role[2], input.data(), ids.data(),
                               weights.data(), int(ids.size()), activation, guarded.data() + 1, &result.stats, error, sizeof(error)) :
        kimi_cpu_sink_run(sink, meta.role[0], meta.role[1], meta.role[2], input.data(), ids.data(), weights.data(),
                               int(ids.size()), 4.0f, linear_beta, guarded.data() + 1, &result.stats, error, sizeof(error));
    result.wall_us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - begin).count();
    const auto after = test_resources::usage();
    result.user_us = after.user_us - before.user_us;
    result.system_us = after.system_us - before.system_us;
    result.voluntary_switches = after.voluntary_switches - before.voluntary_switches;
    result.involuntary_switches = after.involuntary_switches - before.involuntary_switches;
    result.minor_faults = after.minor_faults - before.minor_faults;
    result.major_faults = after.major_faults - before.major_faults;
    result.max_rss_kib = after.max_rss_kib;
    require(ok, std::string(name) + ": " + error);
    require(guarded.front() == -12345.25f && guarded.back() == -12345.25f, "output canary overwritten");
    if (exact_output) { exact_output->assign(guarded.begin() + 1, guarded.end() - 1); }
    double ss = 0, norm = 0; bool close_enough = true;
    for (int i = 0; i < f.latent; ++i) {
        const double value = guarded[i + 1], diff = value - expected[i];
        require(std::isfinite(value) && std::isfinite(expected[i]), "nonfinite numerical result");
        ss += diff * diff; norm += double(expected[i]) * expected[i]; result.max_abs = std::max(result.max_abs, std::abs(diff));
        close_enough &= std::abs(diff) <= 2e-5 + 2e-5 * std::abs(expected[i]);
    }
    const auto * output_bytes = reinterpret_cast<const uint8_t *>(guarded.data() + 1);
    for (size_t i = 0; i < size_t(f.latent) * sizeof(float); ++i) { result.output_hash ^= output_bytes[i]; result.output_hash *= 1099511628211ULL; }
    result.rel_l2 = std::sqrt(ss / std::max(norm, 1e-30));
    const size_t live = std::count_if(ids.begin(), ids.end(), [](int32_t id) { return id >= 0; });
    const auto & stats = result.stats;
    check_io_targets(inputs, stats, true);
    require(expected_hits <= live, "invalid test hit expectation");
    require(stats.experts == live && stats.read_requests == 3 * (live - expected_hits) && stats.fallback_count == 0 && stats.read_errors == 0,
            "expert/read/fallback counters mismatch");
    size_t expected_bytes = 0;
    for (int32_t id : ids) { if (id >= 0) { for (int role = 0; role < 3; ++role) { expected_bytes += f.entries.at({ tensor_name(layer, role), id }).read_bytes; } } }
    const size_t triplet_bytes = live ? expected_bytes / live : 0;
    require(stats.demand_bytes == expected_bytes, "demand byte count mismatch");
    require(stats.read_bytes == expected_bytes - expected_hits * triplet_bytes, "read byte count mismatch");
    require(stats.cache_hits == expected_hits && stats.cache_misses == live - expected_hits, "cache hit/miss count mismatch");
    if (expected_admitted >= 0) {
        require(stats.cache_admitted == size_t(expected_admitted) &&
                stats.cache_admitted_bytes == size_t(expected_admitted) * cache_slot_bytes(inputs, triplet_bytes),
                "cache admission count mismatch");
    }
    require(close_enough, std::string(name) + " diagnostic numerical tolerance exceeded; max_abs=" + std::to_string(result.max_abs));
    require(result.max_abs == 0, std::string(name) + " strict elementwise equality failed; max_abs=" + std::to_string(result.max_abs));
    return result;
}

static void print_sample(const char * name, const sample_result & result) {
    const auto & s = result.stats;
    std::printf("case=%s experts=%llu max_abs=%.9g rel_l2=%.9g reads=%llu bytes=%llu ring_bytes=%llu"
        " total_us=%.3f io_us=%.3f compute_us=%.3f wait_io_us=%.3f read_errors=%llu fallback_count=%llu"
        " demand_bytes=%llu cache_hits=%llu cache_misses=%llu cache_admitted=%llu cache_copy_us=%.3f"
        " cache_fill_read_requests=%llu cache_fill_read_bytes=%llu ring_read_requests=%llu ring_read_bytes=%llu"
        " fixed_read_requests=%llu read_completions=%llu pending_reads=%llu cache_new_slot_bytes=%llu status=PASS\n",
        name, (unsigned long long) s.experts, result.max_abs, result.rel_l2, (unsigned long long) s.read_requests,
        (unsigned long long) s.read_bytes, (unsigned long long) s.ring_bytes, s.total_us,
        s.io_us, s.compute_us, s.wait_io_us, (unsigned long long) s.read_errors, (unsigned long long) s.fallback_count,
        (unsigned long long) s.demand_bytes, (unsigned long long) s.cache_hits, (unsigned long long) s.cache_misses,
        (unsigned long long) s.cache_admitted, s.cache_copy_us,
        (unsigned long long) s.cache_fill_read_requests, (unsigned long long) s.cache_fill_read_bytes,
        (unsigned long long) s.ring_read_requests, (unsigned long long) s.ring_read_bytes,
        (unsigned long long) s.fixed_read_requests, (unsigned long long) s.read_completions,
        (unsigned long long) s.pending_reads, (unsigned long long) s.cache_new_slot_bytes);
    std::fflush(stdout);
}

static void compare_case(fixture_inputs & inputs, const char * name, fixture & f, kimi_cpu_sink * sink, int layer,
        const std::vector<float> & input, const std::vector<int32_t> & ids,
        const std::vector<float> & weights, float linear_beta = 25.0f) {
    tensors meta(f, layer);
    const auto expected = reference(f, layer, input, ids, weights, 4.0f, linear_beta);
    print_sample(name, run_checked(inputs, name, f, sink, meta, input, ids, weights, expected, layer, linear_beta));
}

// A well-formed tensor of TYPE, so a refusal comes from the type and not from its layout.
static void retype(ggml_tensor * tensor, ggml_type type) {
    tensor->type = type;
    tensor->nb[0] = ggml_type_size(type);
    tensor->nb[1] = ggml_row_size(type, tensor->ne[0]);
    tensor->nb[2] = tensor->nb[1] * tensor->ne[1];
    tensor->nb[3] = tensor->nb[2] * tensor->ne[2];
}

static void unsupported_type_case(fixture_inputs & inputs, fixture & f, const std::vector<float> & input) {
    auto sink = make_sink(inputs, f, 2, 1); tensors meta(f, 1);
    retype(meta.role[1], GGML_TYPE_F32);
    const int32_t id = 3; const float weight = 1; std::vector<float> output(f.latent); char error[512] = {}; kimi_cpu_sink_stats stats = {};
    const bool ok = kimi_cpu_sink_run(sink.get(), meta.role[0], meta.role[1], meta.role[2], input.data(), &id, &weight, 1,
                                      4, 25, output.data(), &stats, error, sizeof(error));
    require(!ok && std::strstr(error, "not supported yet") && std::strstr(error, "f32"), "unsupported type was not named");
}

static void fault_case(fixture_inputs & inputs, const char * name, fixture & f, fixture::fault_kind fault, const std::vector<float> & input) {
    f.fault = fault; auto sink = make_sink(inputs, f, 2, 1); tensors meta(f, 1);
    const int32_t id = 3; const float weight = 1; std::vector<float> output(f.latent); char error[512] = {}; kimi_cpu_sink_stats stats = {};
    bool ok = kimi_cpu_sink_run(sink.get(), meta.role[0], meta.role[1], meta.role[2], input.data(), &id, &weight, 1,
                               4, 25, output.data(), &stats, error, sizeof(error));
    require(!ok && error[0] && stats.fallback_count == 0, std::string(name) + " did not fail closed");
    check_io_targets(inputs, stats, false);
    if (fault == fixture::past_eof) {
        require(stats.read_errors > 0, "short read did not increment errors");
        f.fault = fixture::none;
        compare_case(inputs, "after_short_read_recovery", f, sink.get(), 1, input, { id }, { weight });
    }
    std::printf("case=%s status=PASS error=%s\n", name, error); f.fault = fixture::none;
}

static void synthetic_tests(fixture_inputs & inputs, int threads, int ring_experts) {
    fixture f; make_synthetic(f);
    std::mt19937 rng(81); std::normal_distribution<float> sample(0.0f, 0.3f);
    std::vector<float> input(f.latent); for (auto & v : input) { v = sample(rng); }
    auto sink = make_sink(inputs, f, threads, ring_experts);
    compare_case(inputs, "one_expert", f, sink.get(), 1, input, { 3 }, { 1 });
    std::vector<int32_t> ids = { 17, 0, 8, 2, 19, 4, 11, 6, 15, 1, 13, 7, 18, 5, 10, 9 };
    std::vector<float> weights(16); for (int i = 0; i < 16; ++i) { weights[i] = float(i + 1) / 136; }
    compare_case(inputs, "top16_ordered", f, sink.get(), 1, input, ids, weights);
    compare_case(inputs, "top8", f, sink.get(), 1, input, { ids.begin(), ids.begin() + 8 }, { weights.begin(), weights.begin() + 8 });
    compare_case(inputs, "mass_holes_duplicate_zero", f, sink.get(), 1, input, { 9, -1, 2, 9, -7, 13, 5, -1 }, { .2f, 0, .3f, .1f, 0, 0, .4f, 0 });
    compare_case(inputs, "empty_slots", f, sink.get(), 1, input, { -1, -3 }, { 0, 0 });
    compare_case(inputs, "linear_transform_disabled", f, sink.get(), 1, input, { 9, 0, 5, 13 }, { .1f, .2f, .3f, .4f }, 0);
    for (int repeat = 0; repeat < 8; ++repeat) {
        compare_case(inputs, "layer_generation_reuse", f, sink.get(), repeat % 2 + 1, input, ids, weights);
        input[repeat] += 0.03f;
    }
    auto one_compute = make_sink(inputs, f, 2, 1);
    compare_case(inputs, "single_compute_thread", f, one_compute.get(), 2, input, ids, weights);
    unsupported_type_case(inputs, f, input);
    for (auto test : { std::make_pair("missing_entry", fixture::missing), std::make_pair("unaligned_offset", fixture::unaligned_offset),
                       std::make_pair("invalid_range_size", fixture::wrong_size), std::make_pair("short_read_drain", fixture::past_eof) }) {
        fault_case(inputs, test.first, f, test.second, input);
    }
    std::printf("SYNTHETIC_CPU_SINK_OK\n");
}

struct real_options {
    int threads = 16, ring_experts = 8, repeats = 5, warmup = 2;
    std::string policy = "fixture", jsonl, resident_dir;
};

static std::string quote_json(const std::string & text) {
    std::ostringstream out; out << '"';
    for (unsigned char c : text) {
        if (c == '"' || c == '\\') { out << '\\' << char(c); }
        else if (c < 32) { out << "\\u" << std::hex << std::setw(4) << std::setfill('0') << unsigned(c) << std::dec; }
        else { out << char(c); }
    }
    out << '"'; return out.str();
}

static void apply_policy(std::vector<int32_t> & ids, std::vector<float> & weights, const std::string & policy) {
    require(ids.size() == weights.size(), "route/weight length mismatch");
    require(policy == "fixture" || policy == "top16" || policy == "top8" || policy == "mass085", "unknown real policy");
    if (policy == "fixture" || policy == "top16") { return; }
    require(ids.size() == 16 && std::all_of(ids.begin(), ids.end(), [](int32_t id) { return id >= 0; }),
            "top8/mass085 requires full16 input in decreasing biased selection-score order");
    require(policy == "top8" || policy == "mass085", "unknown real policy");
    if (policy == "mass085") {
        std::vector<int> order(8); for (int i = 0; i < 8; ++i) { order[i] = i; }
        std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return weights[a] > weights[b]; });
        const auto original_ids = ids; const auto original_weights = weights;
        for (int i = 0; i < 8; ++i) { ids[i] = original_ids[order[i]]; weights[i] = original_weights[order[i]]; }
    }
    float cap_sum = 0; for (int i = 0; i < 8; ++i) { cap_sum += weights[i]; }
    require(std::isfinite(cap_sum) && cap_sum > 0, "invalid top8 weight sum");
    for (int i = 0; i < 8; ++i) { weights[i] /= cap_sum; }
    int keep = 8;
    if (policy == "mass085") {
        float cumulative = 0; keep = 0;
        do { cumulative += weights[keep++]; } while (keep < 8 && (keep < 4 || cumulative < 0.85f));
    }
    float kept_sum = 0; for (int i = 0; i < keep; ++i) { kept_sum += weights[i]; }
    if (policy == "mass085") { for (int i = 0; i < keep; ++i) { weights[i] /= kept_sum; } }
    for (size_t i = keep; i < ids.size(); ++i) { ids[i] = -1; weights[i] = 0; }
}

struct unsupported_resident : std::runtime_error { using std::runtime_error::runtime_error; };

#if defined(__linux__)
// Linux only: the control copy lives on tmpfs, recognized by its statfs magic.
static void make_resident_copy(fixture & f, const std::string & directory) {
    struct statfs fs = {};
    if (statfs(directory.c_str(), &fs) || uint64_t(fs.f_type) != 0x01021994ULL) {
        throw unsupported_resident("resident control requires an existing tmpfs directory; no buffered/disk substitute");
    }
    std::string pattern = directory + "/kimi-sink-resident-XXXXXX";
    std::vector<char> name(pattern.begin(), pattern.end()); name.push_back(0);
    const int tfd = mkstemp(name.data()); require(tfd >= 0, "resident mkstemp failed"); close(tfd);
    if (f.fd >= 0) { close(f.fd); f.fd = -1; }
    f.path = name.data(); f.remove_on_destroy = true;
    std::ofstream file(f.path, std::ios::binary | std::ios::trunc); uint64_t offset = 0;
    std::vector<char> zero(4096);
    for (auto & kv : f.entries) {
        auto & e = kv.second; e.offset = offset;
        put_bytes(file, e.data.data(), e.raw_bytes);
        if (e.read_bytes > e.raw_bytes) { put_bytes(file, zero.data(), e.read_bytes - e.raw_bytes); }
        offset += e.read_bytes;
    }
    file.close(); require(bool(file), "resident copy failed");
    f.fd = test_resources::open_direct(f.path);
    std::printf("resident_control=tmpfs_O_DIRECT_accepted note=RAM_file_read_path_not_zero_IO_compute_control bytes=%llu\n", (unsigned long long) offset);
}
#endif

static void write_sample_json(std::ostream & out, const sample_result & result, const real_options & options,
        const std::string & phase, int iteration, int layer, double reference_us) {
    const auto & s = result.stats;
    std::ostringstream hash; hash << std::hex << std::setw(16) << std::setfill('0') << result.output_hash;
    out << std::setprecision(12) << "{\"schema\":1,\"kind\":\"sample\",\"phase\":" << quote_json(phase)
        << ",\"iteration\":" << iteration << ",\"layer\":" << layer
        << ",\"policy\":" << quote_json(options.policy) << ",\"threads\":" << options.threads << ",\"ring_experts\":" << options.ring_experts
        << ",\"storage\":" << quote_json(options.resident_dir.empty() ? "pack_odirect" : "tmpfs_ram_file_approximation")
        << ",\"experts\":" << s.experts << ",\"read_requests\":" << s.read_requests << ",\"read_bytes\":" << s.read_bytes
        << ",\"ring_bytes\":" << s.ring_bytes << ",\"total_us\":" << s.total_us << ",\"wall_us\":" << result.wall_us
        << ",\"io_us\":" << s.io_us << ",\"compute_us\":" << s.compute_us << ",\"wait_io_us\":" << s.wait_io_us
        << ",\"input_quantize_us\":" << s.input_quantize_us << ",\"hidden_quantize_us\":" << s.hidden_quantize_us
        << ",\"up_gate_us\":" << s.up_gate_us << ",\"down_reduce_us\":" << s.down_reduce_us
        << ",\"read_errors\":" << s.read_errors << ",\"fallback_count\":" << s.fallback_count
        << ",\"user_us\":" << result.user_us << ",\"system_us\":" << result.system_us
        << ",\"voluntary_switches\":" << result.voluntary_switches << ",\"involuntary_switches\":" << result.involuntary_switches
        << ",\"minor_faults\":" << result.minor_faults << ",\"major_faults\":" << result.major_faults << ",\"max_rss_kib\":" << result.max_rss_kib
        << ",\"max_abs\":" << result.max_abs << ",\"rel_l2\":" << result.rel_l2 << ",\"output_hash\":" << quote_json(hash.str())
        << ",\"reference_once_us\":" << reference_us << ",\"status\":\"PASS\"}\n";
    out.flush(); require(bool(out), "sample JSON output failed");
}

static void real_test(fixture_inputs & inputs, const std::string & pack, const std::string & prefix, const real_options & options) {
    fixture f; int layer, count;
    std::ifstream route(prefix + ".txt"); require(bool(route >> layer >> f.latent >> f.hidden >> count), "cannot parse route fixture");
    require(count > 0 && count <= 16 && f.latent == 3584 && f.hidden == 3072, "unexpected real fixture dimensions");
    f.experts = 896; std::vector<int32_t> ids(count); std::vector<float> weights(count);
    for (int i = 0; i < count; ++i) { require(bool(route >> ids[i] >> weights[i]), "truncated route fixture"); }
    apply_policy(ids, weights, options.policy);
    std::vector<int32_t> live_ids; for (int32_t id : ids) { if (id >= 0) { live_ids.push_back(id); } }
    std::sort(live_ids.begin(), live_ids.end()); live_ids.erase(std::unique(live_ids.begin(), live_ids.end()), live_ids.end());
    require(!live_ids.empty(), "real fixture has no live experts");
    std::vector<float> input(f.latent); std::ifstream inp(prefix + ".f32", std::ios::binary);
    inp.read(reinterpret_cast<char *>(input.data()), input.size() * sizeof(float)); require(bool(inp) && inp.peek() == EOF, "input fixture length mismatch");
    std::ifstream file(pack, std::ios::binary); require(bool(file), "pack open failed");
    char magic[16]; uint32_t version, header; uint64_t n, data_start;
    file.read(magic, 16); file.read(reinterpret_cast<char *>(&version), 4); file.read(reinterpret_cast<char *>(&header), 4);
    file.read(reinterpret_cast<char *>(&n), 8); file.read(reinterpret_cast<char *>(&data_start), 8);
    require(bool(file) && !std::memcmp(magic, "GGMLMOEPACKv1", 12) && version == 1 && header == 40 && n < 1000000 && data_start >= 40 + 152 * n, "bad pack header");
    for (uint64_t i = 0; i < n; ++i) {
        char label[129] = {}; int32_t expert; uint32_t reserved; uint64_t offset, bytes;
        file.read(label, 128); file.read(reinterpret_cast<char *>(&expert), 4); file.read(reinterpret_cast<char *>(&reserved), 4);
        file.read(reinterpret_cast<char *>(&offset), 8); file.read(reinterpret_cast<char *>(&bytes), 8); require(bool(file), "truncated pack index");
        if (std::find(live_ids.begin(), live_ids.end(), expert) == live_ids.end()) { continue; }
        for (int role = 0; role < 3; ++role) {
            if (label == tensor_name(layer, role)) {
                require(bytes == 4730880 && offset % 4096 == 0, "unexpected real role layout");
                entry item; item.offset = offset; item.raw_bytes = bytes; item.read_bytes = bytes;
                f.entries[{ label, expert }] = std::move(item);
            }
        }
    }
    require(f.entries.size() == live_ids.size() * 3, "real fixture missing role/expert");
    // Single selected layer only. The buffered reference read is independently
    // compared with the sink's subsequent O_DIRECT transport, without GPU use.
    for (auto & kv : f.entries) {
        auto & item = kv.second; item.data.resize(item.raw_bytes);
        file.seekg(item.offset); file.read(reinterpret_cast<char *>(item.data.data()), item.raw_bytes); require(bool(file), "short reference pack read");
    }
    file.close(); f.path = pack; f.fd = test_resources::open_direct(f.path);
    if (!options.resident_dir.empty()) {
#if defined(__linux__)
        make_resident_copy(f, options.resident_dir);
#else
        throw unsupported_resident("resident control requires a Linux tmpfs directory");
#endif
    }
    const auto reference_begin = std::chrono::steady_clock::now();
    const auto expected = reference(f, layer, input, ids, weights, 4.0f, 25.0f);
    const double reference_us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - reference_begin).count();
    // The ordinary CPU reference runs once before the first API sample. It never
    // runs between warmup/steady samples, and is excluded from all sink timings.
    tensors meta(f, layer); auto sink = make_sink(inputs, f, options.threads, options.ring_experts);
    std::ofstream json_file; if (!options.jsonl.empty()) { json_file.open(options.jsonl, std::ios::out | std::ios::trunc); require(bool(json_file), "cannot open sample JSONL"); }
    for (int iteration = 0; iteration < 1 + options.warmup + options.repeats; ++iteration) {
        const std::string phase = iteration == 0 ? "first_call_after_reference" : iteration <= options.warmup ? "warmup" : "steady";
        const auto result = run_checked(inputs, "real_repeat", f, sink.get(), meta, input, ids, weights, expected, layer, 25.0f);
        print_sample(phase.c_str(), result);
        if (json_file.is_open()) { write_sample_json(json_file, result, options, phase, iteration, layer, reference_us); }
    }
    std::printf("REAL_CPU_SINK_OK reference_once_us=%.3f measured_samples=%d warmup_samples=%d first_call_samples=1\n", reference_us, options.repeats, options.warmup);
}

struct fixture_selector {
    fixture * active = nullptr;
    static bool resolve(void * user, const char * name, int32_t id, kimi_cpu_sink_source * out) {
        auto * selector = static_cast<fixture_selector *>(user);
        return fixture::resolve(selector->active, name, id, out);
    }
};

#if defined(__linux__)
// Linux only: io_uring fixed buffers are charged to RLIMIT_MEMLOCK; a forked child lowers it.
static void drop_child_memlock() {
    rlimit limit = { 0, 0 };
    require(setrlimit(RLIMIT_MEMLOCK, &limit) == 0, "lowering child memlock failed");
    __user_cap_header_struct header = {}; header.version = _LINUX_CAPABILITY_VERSION_3;
    __user_cap_data_struct caps[2] = {};
    require(syscall(SYS_capget, &header, caps) == 0, "capget failed");
    caps[CAP_IPC_LOCK / 32].effective &= ~(uint32_t(1) << (CAP_IPC_LOCK % 32));
    require(syscall(SYS_capset, &header, caps) == 0, "dropping child IPC_LOCK failed");
}

static void registration_failure_child(fixture_inputs & inputs, fixture & f) {
    // Change only a child process: the parent keeps its memlock limit and caps.
    // This runs before the first sink starts any worker threads.
    std::fflush(nullptr);
    const pid_t child = fork(); require(child >= 0, "fork failed");
    if (child == 0) {
        try {
            // The kernel may charge ring setup itself to memlock. Create the
            // real ring first, then deny only the buffer registration under test.
            auto sink = make_sink(inputs, f, 2, 1);
            drop_child_memlock();
            tensors meta(f, 1);
            std::vector<float> input(f.latent, 0.1f), output(f.latent);
            int32_t id = 3; float weight = 1; char error[512] = {}; kimi_cpu_sink_stats stats = {};
            const bool ok = kimi_cpu_sink_run(sink.get(), meta.role[0], meta.role[1], meta.role[2], input.data(),
                &id, &weight, 1, 4, 25, output.data(), &stats, error, sizeof(error));
            require(!ok && std::strstr(error, "IORING_REGISTER_BUFFERS") &&
                    stats.read_requests == 0 && stats.fallback_count == 0,
                    std::string("registration failure did not fail closed: ") + error);
            check_io_targets(inputs, stats, false);
            sink.reset();
            std::printf("REGISTER_FAILURE_NO_FALLBACK_OK error=%s\n", error);
            std::fflush(nullptr);
            _exit(0);
        } catch (const std::exception & e) {
            std::fprintf(stderr, "REGISTER_FAILURE_CHILD_FAILED: %s\n", e.what());
            _exit(1);
        }
    }
    int status = 0;
    pid_t result;
    do { result = waitpid(child, &status, 0); } while (result < 0 && errno == EINTR);
    require(result == child && WIFEXITED(status) && WEXITSTATUS(status) == 0,
            "registration failure child did not pass");
}
#endif

static void registered_lifecycle_tests(fixture_inputs & inputs) {
    const char * mode = inputs.get("read_fixed");
    require(mode && !std::strcmp(mode, "1"), "run this lifecycle test with --fixture read_fixed=1");
    fixture small, large;
    make_synthetic(small);
#if defined(__linux__)
    registration_failure_child(inputs, small);
#else
    std::printf("case=registration_failure_child status=SKIP reason=linux_rlimit_memlock\n");
#endif
    large.latent = large.hidden = 512;
    make_synthetic(large);
    fixture_selector selector; selector.active = &small;
    char error[512] = {};
    sink_ptr sink = inputs.create(4, 2, fixture_selector::resolve, &selector, error, sizeof(error));
    require(bool(sink), std::string("create: ") + error);
    std::vector<float> xs(small.latent), xl(large.latent);
    for (size_t i = 0; i < xs.size(); ++i) { xs[i] = float(int(i % 23) - 11) * 0.02f; }
    for (size_t i = 0; i < xl.size(); ++i) { xl[i] = float(int(i % 29) - 14) * 0.02f; }

    compare_case(inputs, "fixed_empty_before_allocation", small, sink.get(), 1, xs, { -1 }, { 0 });
    compare_case(inputs, "fixed_initial_register", small, sink.get(), 1, xs, { 3 }, { 1 });
    compare_case(inputs, "fixed_cross_layer_reuse", small, sink.get(), 2, xs, { 3, 4, -1, 8 }, { .2f, .3f, 0, .5f });
    selector.active = &large;
    compare_case(inputs, "fixed_unregister_grow_register", large, sink.get(), 1, xl, { 3, 5 }, { .3f, .7f });
    selector.active = &small;
    compare_case(inputs, "fixed_smaller_shape_reuses_registration", small, sink.get(), 1, xs, { 9 }, { 1 });
    compare_case(inputs, "fixed_empty_keeps_registration", small, sink.get(), 2, xs, { -1, -2 }, { 0, 0 });

    tensors meta(small, 1);
    const int32_t id = 3; const float weight = 1;
    std::vector<float> output(small.latent);
    kimi_cpu_sink_stats stats = {};
    small.fault = fixture::past_eof;
    const bool ok = kimi_cpu_sink_run(sink.get(), meta.role[0], meta.role[1], meta.role[2],
        xs.data(), &id, &weight, 1, 4, 25, output.data(), &stats, error, sizeof(error));
    require(!ok && stats.read_errors > 0 && stats.fallback_count == 0, "fixed short read did not fail closed");
    check_io_targets(inputs, stats, false);
    small.fault = fixture::none;
    compare_case(inputs, "fixed_short_read_drain_recovery", small, sink.get(), 2, xs, { 3 }, { 1 });
    sink.reset(); // unregister before fixture/buffer destruction
    std::printf("REGISTERED_BUFFER_LIFECYCLE_OK expected_register_count=2 expected_unregister_count=2\n");
}

static uint64_t triplet_bytes(const fixture & f) {
    uint64_t bytes = 0;
    for (int role = 0; role < 3; ++role) { bytes += f.entries.at({ tensor_name(1, role), 0 }).read_bytes; }
    return bytes;
}

static kimi_cpu_sink_cache_stats cache_snapshot(kimi_cpu_sink * sink) {
    kimi_cpu_sink_cache_stats stats = {};
    kimi_cpu_sink_cache_get_stats(sink, &stats);
    require(stats.capacity_bytes <= stats.budget_bytes && stats.mapped_bytes <= stats.capacity_bytes &&
            stats.enabled_bytes <= stats.capacity_bytes && stats.committed_bytes <= stats.mapped_bytes &&
            stats.valid_bytes <= stats.committed_bytes && stats.valid_bytes <= stats.enabled_bytes,
            "cache physical budget invariant failed");
    require(stats.copy_us == 0, "cache snapshot recorded an admission payload copy");
    return stats;
}

static void configure_cache(kimi_cpu_sink * sink, uint64_t bytes,
        const std::vector<int32_t> & layers, int quota) {
    char error[512] = {};
    require(kimi_cpu_sink_cache_configure(sink, bytes, layers.data(), int(layers.size()), quota, error, sizeof(error)),
            std::string("cache configure: ") + error);
    const auto stats = cache_snapshot(sink);
    require(stats.budget_bytes == bytes && stats.mapped_bytes == 0 &&
            stats.committed_bytes == 0 && stats.valid_bytes == 0, "configure eagerly allocated or retained cache payload");
}

static void begin_cache(kimi_cpu_sink * sink, uint64_t epoch, uint64_t token, bool decode = true) {
    char error[512] = {};
    require(kimi_cpu_sink_cache_begin(sink, epoch, token, decode, error, sizeof(error)),
            std::string("cache begin: ") + error);
    cache_snapshot(sink);
}

static sample_result cache_case(fixture_inputs & inputs, const char * name, fixture & f, kimi_cpu_sink * sink, int layer,
        const std::vector<float> & input, const std::vector<int32_t> & ids,
        const std::vector<float> & weights, size_t hits, int admitted, float linear_beta = 25) {
    tensors meta(f, layer);
    const auto expected = reference(f, layer, input, ids, weights, 4, linear_beta);
    auto result = run_checked(inputs, name, f, sink, meta, input, ids, weights, expected, layer, linear_beta, hits, admitted);
    cache_snapshot(sink);
    print_sample(name, result);
    return result;
}

// Compare both executors on identical packed Q3 data and cache histories. The
// graph reference uses a tolerance; the scheduling change must be bit-exact.
// A synthetic asynchronous transport tests the consumer/lease contract, not
// CUDA arithmetic. Raw vectors are independently produced by the old executor.
struct location_fixture {
    std::vector<std::vector<float>> raw;
    uint32_t resident=0, pinned=0, launched=0;
    bool fail_finish=false, done=false;
    int finished_count=0,cancel_count=0,offered=0;
    std::thread work;
    ~location_fixture(){if(work.joinable()){work.join();}}
    static location_fixture & self(void *p){return *static_cast<location_fixture *>(p);}
    static int acquire(void * p,const kimi_cache_location_candidate * c,int n,kimi_cache_location_lease * out,char *,size_t){
        auto & f=self(p);require(!f.pinned,"leaked location lease between calls");
        for(int i=0;i<n;++i){require(c[i].epoch && c[i].role[0].nbytes,"invalid location key");if(f.resident&(1u<<i)){out[i]={uint64_t(i+1),c[i].epoch};f.pinned|=1u<<i;}}return 0;
    }
    static int begin(void * p,const kimi_cache_location_candidate *,int,uint32_t base,uint32_t pin,char *,size_t){
        auto & f=self(p);require(!(f.pinned&~pin),"unused VRAM leases not released before admission");require(base,"empty demand mask");f.done=false;f.launched=0;return 0;
    }
    static int launch(void * p,const kimi_cache_location_job * jobs,int count,const void *,size_t,int latent,int,float,float,char *,size_t){
        auto & f=self(p);require(!f.work.joinable(),"multiple VRAM launches");
        std::vector<kimi_cache_location_job> copy(jobs,jobs+count);
        for(auto & j:copy){require(j.lease.handle==uint64_t(j.native_slot+1)&&(f.pinned&(1u<<j.native_slot)),"GPU job lacks live lease");f.launched|=1u<<j.native_slot;}
        f.work=std::thread([&f,copy,latent]{std::this_thread::sleep_for(std::chrono::microseconds(300));for(const auto & j:copy){std::copy_n(f.raw[j.native_slot].data(),latent,j.raw_down);}});return 0;
    }
    static int admit(void *p,const kimi_cache_location_candidate *,const void *a,const void *b,const void *c,int,char *,size_t){require(a&&b&&c,"invalid upload source");self(p).offered++;return 0;}
    static int finish(void *p,char *e,size_t n){auto & f=self(p);if(f.work.joinable()){f.work.join();}f.done=true;if(f.fail_finish){std::snprintf(e,n,"synthetic GPU completion failure");return 1;}return 0;}
    static int cancel(void *p,char *,size_t){auto & f=self(p);if(f.work.joinable()){f.work.join();}f.done=true;f.cancel_count++;return 0;}
    static int release(void *p,const kimi_cache_location_lease * ls,int count,char *,size_t){auto & f=self(p);require(!f.work.joinable(),"release before GPU drain");for(int i=0;i<count;++i){const uint32_t bit=1u<<(ls[i].handle-1);require(f.pinned&bit,"double/stale lease release");f.pinned&=~bit;}return 0;}
    static int finished(void *p,kimi_cpu_sink *,char *,size_t){auto & f=self(p);require(f.done&&!f.pinned&&!f.work.joinable(),"RAM eviction callback before finish/unpin");f.finished_count++;return 0;}
    static kimi_cache_location_api api(){return {1,sizeof(kimi_cache_location_api),acquire,begin,launch,admit,finish,cancel,release,finished};}
};
static void cache_location_tests(fixture_inputs & inputs, int threads,int depth){
    ggml_cpu_init(); // Initialize FP16 lookup tables before the raw-vector oracle.
    inputs.set("ready_first", "1");
    inputs.set("early_up_gate", "0");inputs.set("cache_policy", "lru");
    inputs.set("cache_allocation", "uniform");inputs.set("read_fixed", "0");
    fixture f;make_synthetic(f);tensors meta(f,1);char error[512]={};
    std::vector<float> input(f.latent);for(int i=0;i<f.latent;++i){input[i]=std::sin(i*.037f);}
    const int32_t native[]={0,1,2,3,4,5};const float nw[]={.32f,.24f,.16f,.12f,.10f,.06f};
    const int32_t ids[]={0,1,2,3};const float weights[]={.32f/.84f,.24f/.84f,.16f/.84f,.12f/.84f};
    location_fixture transport;transport.raw.resize(6);auto api=location_fixture::api();
    for(int i=0;i<6;++i){auto old=make_sink(inputs, f,threads,depth);transport.raw[i].resize(f.latent);const float one=1;require(kimi_cpu_sink_run(old.get(),meta.role[0],meta.role[1],meta.role[2],input.data(),&native[i],&one,1,4,25,transport.raw[i].data(),nullptr,error,sizeof(error)),error);}
    for(const auto & raw:transport.raw){require(std::any_of(raw.begin(),raw.end(),[](float v){return v!=0;}),"uninitialized raw-vector oracle");}
    for(int mode=0;mode<=3;++mode) for(uint32_t resident : {0u,17u,63u}) for(bool guard : {false,true}){
        auto sink=make_sink(inputs, f,threads,depth);configure_cache(sink.get(),4*triplet_bytes(f),{1},4);begin_cache(sink.get(),1,1);
        cache_case(inputs, "location_warm",f,sink.get(),1,input,{1,3,4,5},{.4f,.3f,.2f,.1f},0,4);
        require(kimi_cpu_sink_set_cache_locations(sink.get(),&api,&transport,error,sizeof(error)),error);begin_cache(sink.get(),1,2);
        transport.resident=resident;transport.finished_count=0;
        kimi_cache_compute_config cfg{1,mode,1,2,0,.5f,0,.65f,2,0,1};kimi_cpu_sink_stats stats{};kimi_cache_compute_stats audit{};std::vector<float> actual(f.latent),expected(f.latent);
        require(kimi_cpu_sink_run_cache_compute(sink.get(),meta.role[0],meta.role[1],meta.role[2],input.data(),ids,weights,4,native,nw,6,1,guard,&cfg,4,25,actual.data(),&stats,&audit,error,sizeof(error)),error);
        require(transport.pinned==0&&transport.finished_count==1&&transport.launched==(audit.plan.final_mask&resident),"GPU final selection/leases differ");
        const uint32_t misses=audit.plan.base_mask&~resident&~(2u|8u|16u|32u);
        require(stats.read_bytes==uint64_t(__builtin_popcount(misses))*triplet_bytes(f),"GPU cache triggered SSD read or hid late read bytes");
        require(!stats.pending_reads&&stats.read_requests==stats.read_completions&&!stats.read_errors,"mixed cache drain failed");
        std::vector<int32_t> final_ids;std::vector<float> final_weights;
        for(int n=0;n<6;++n){if(audit.plan.final_mask&(1u<<n)){final_ids.push_back(n);final_weights.push_back(audit.final_weights[n]);}}
        auto old=make_sink(inputs, f,threads,depth);require(kimi_cpu_sink_run(old.get(),meta.role[0],meta.role[1],meta.role[2],input.data(),final_ids.data(),final_weights.data(),int(final_ids.size()),4,25,expected.data(),nullptr,error,sizeof(error)),error);
        require(!std::memcmp(actual.data(),expected.data(),actual.size()*sizeof(float)),"RAM+GPU fixed final set not bit exact");
        std::printf("CACHE_LOCATION_CASE mode=%d resident=%u guarded=%d final=%u GPU=%u bytes=%llu PASS\n",mode,resident,guard,audit.plan.final_mask,transport.launched,(unsigned long long)stats.read_bytes);
    }
    auto sink=make_sink(inputs, f,threads,depth);configure_cache(sink.get(),4*triplet_bytes(f),{1},4);begin_cache(sink.get(),2,1);
    require(kimi_cpu_sink_set_cache_locations(sink.get(),&api,&transport,error,sizeof(error)),error);transport.resident=17;transport.fail_finish=true;
    kimi_cache_compute_config cfg{1,1,1,2,0,.75f,0,.8f,2,0,1};kimi_cpu_sink_stats stats{};std::vector<float> actual(f.latent);
    require(!kimi_cpu_sink_run_cache_compute(sink.get(),meta.role[0],meta.role[1],meta.role[2],input.data(),ids,weights,4,native,nw,6,1,false,&cfg,4,25,actual.data(),&stats,nullptr,error,sizeof(error)),"GPU failure ignored");
    require(!transport.pinned&&transport.cancel_count==1&&!stats.pending_reads&&stats.read_requests==stats.read_completions,"GPU failure did not drain and unpin");
    require(!cache_snapshot(sink.get()).mapped_bytes,"GPU failure retained borrowed RAM");
    transport.fail_finish=false;require(kimi_cpu_sink_set_cache_locations(sink.get(),nullptr,nullptr,error,sizeof(error)),error);
    begin_cache(sink.get(),2,2);cache_case(inputs, "location_failure_recover",f,sink.get(),1,input,{1},{1},0,1);
    std::printf("CACHE_LOCATION_PASS threads=%d depth=%d\n",threads,depth);
}

static void cache_compute_tests(fixture_inputs & inputs, int threads, int depth) {
    inputs.set("ready_first", "1");
    inputs.set("early_up_gate", "0");
    inputs.set("cache_policy", "lru");
    inputs.set("cache_allocation", "uniform");
    inputs.set("read_fixed", "0");
    fixture f; make_synthetic(f); tensors meta(f,1);
    const uint64_t unit=triplet_bytes(f);
    std::vector<float> input(f.latent); for(int i=0;i<f.latent;++i){input[i]=std::sin(i*.037f);}
    const int32_t native[]={0,1,2,3,4,5}; const float nw[]={.32f,.24f,.16f,.12f,.10f,.06f};
    const int32_t ids[]={0,1,2,3}; const float weights[]={.32f/.84f,.24f/.84f,.16f/.84f,.12f/.84f};
    char error[512]={};
    for(int mode=0;mode<=3;++mode) for(int bonus : {0,1,2}) for(bool guarded : {false,true}) {
        auto sink=make_sink(inputs, f,threads,depth);
        // Four slots: all extra candidates remain pinned while two base misses
        // seek admission. Ring depth 1 exercises completion/compute credits.
        configure_cache(sink.get(),4*unit,{1},4);begin_cache(sink.get(),1,1);
        cache_case(inputs, "compensate_warm",f,sink.get(),1,input,{1,3,4,5},{.4f,.3f,.2f,.1f},0,4);
        begin_cache(sink.get(),1,2);
        kimi_cache_compute_config cfg{1,mode,bonus,2,0,.5f,0,.65f,2,0,1};
        std::vector<float> actual(f.latent),expected(f.latent);
        kimi_cpu_sink_stats st{};kimi_cache_compute_stats audit{};
        require(kimi_cpu_sink_run_cache_compute(sink.get(),meta.role[0],meta.role[1],meta.role[2],input.data(),
            ids,weights,4,native,nw,6,1.0f,guarded,&cfg,4,25,actual.data(),&st,&audit,error,sizeof(error)),error);
        require(!st.pending_reads && st.read_requests==st.read_completions && !st.read_errors,"compensate drain failed");
        if(!guarded&&mode) {
            require(audit.submitted_bytes==st.read_bytes,"all submitted SSD bytes must be charged");
            require(audit.plan.original_mask==15 && (audit.computed_mask&audit.plan.final_mask)==audit.plan.final_mask,"final set incomplete");
            const uint32_t extras=audit.plan.final_mask&~audit.plan.base_mask;
            require(!(extras&~audit.plan.cached_mask) && !(extras&~audit.plan.pin_mask),"extra lacks a resident lease");
            require(st.read_bytes==(mode==3?unit:2*unit),"shrinking failed to reduce submitted bytes");
            if(mode==2){require(audit.plan.dropped_mask==4 && audit.unused_read_bytes==unit,"deadline-zero must discard and drain low miss 2");}
        }
        std::vector<int32_t> final_ids;std::vector<float> final_weights;
        for(int n=0;n<6;++n) if(audit.plan.final_mask&(1u<<n)){final_ids.push_back(native[n]);final_weights.push_back(audit.final_weights[n]);}
        auto baseline=make_sink(inputs, f,threads,depth);
        kimi_cpu_sink_stats reference_stats{};
        require(kimi_cpu_sink_run(baseline.get(),meta.role[0],meta.role[1],meta.role[2],input.data(),final_ids.data(),
            final_weights.data(),int(final_ids.size()),4,25,expected.data(),&reference_stats,error,sizeof(error)),error);
        require(!std::memcmp(actual.data(),expected.data(),actual.size()*sizeof(float)),"explicit final-set legacy output differs bitwise");
        const auto independent=reference(f,1,input,final_ids,final_weights,4,25);
        for(size_t j=0;j<actual.size();++j){require(std::abs(actual[j]-independent[j])<1e-4f,"independent ggml graph differs");}
        std::printf("CACHE_COMPUTE_CASE mode=%d bonus=%d guarded=%d final=%u bytes=%llu bit_exact=1\n",mode,bonus,guarded,
            audit.plan.final_mask,(unsigned long long)st.read_bytes);
    }
    // A low miss that would be discarded still owns its read buffers and must
    // propagate a positive short CQE. It cannot silently become a cache bonus.
    auto broken=make_sink(inputs, f,threads,depth);configure_cache(broken.get(),4*unit,{1},4);begin_cache(broken.get(),3,1);
    f.fault=fixture::short_tail;f.fault_expert=2;f.fault_role=2;
    kimi_cache_compute_config cfg{1,2,1,2,0,.75f,0,.65f,2,0,1};
    kimi_cpu_sink_stats st{};kimi_cache_compute_stats audit{};std::vector<float> output(f.latent);
    require(!kimi_cpu_sink_run_cache_compute(broken.get(),meta.role[0],meta.role[1],meta.role[2],input.data(),
        ids,weights,4,native,nw,6,1.0f,false,&cfg,4,25,output.data(),&st,&audit,error,sizeof(error)),"unused short read was ignored");
    require(st.read_errors && !st.pending_reads && st.read_requests==st.read_completions,"failed unused read not drained");
    require(!cache_snapshot(broken.get()).mapped_bytes,"failed read retained cache targets");
    f.fault=fixture::none;f.fault_expert=-1;f.fault_role=-1;
    begin_cache(broken.get(),3,2);
    cache_case(inputs, "compensate_recover",f,broken.get(),1,input,{0},{1},0,1);
    std::printf("CACHE_COMPUTE_EXECUTOR_PASS threads=%d depth=%d\n",threads,depth);
}

static void schedule_tests(fixture_inputs & inputs, int threads, int depth, bool schedules = false, bool early_up_gate = false,
                             int poll = 0, bool role_major = false, bool nonblocking_down_sched = false) {
    fixture small, large;
    make_synthetic(small);
    large.latent = 512; large.hidden = 1024; make_synthetic(large);
    fixture_selector selector; selector.active = &small;
    const char * names[] = {
        "ready_first", "row_weights",
        "io_cpus",
        "early_up_gate", "poll_us",
        "role_major", "nonblocking_down_sched",
        "dynamic_rows", "dynamic_chunk_rows" };
    std::map<std::string, std::string> original;
    for (const char * name : names) { if (inputs.get(name)) { original[name] = inputs.get(name); } }
    // The sink pins only its own I/O threads (modes 6 and 7, on one CPU); its compute participants are the
    // caller's threads, whose placement it never changes.
    const ssd::platform::cpu_list allowed = test_resources::process_cpus();
    std::string weighted;
    for (int i = 0; i < threads - 1; ++i) { weighted += std::string(i ? "," : "") + (i < 4 ? "1" : "0.6"); }
    // Preserve original ordinary schedule IDs, independently of vector indices.
    const std::vector<int> mode_ids = schedules ? std::vector<int>{ 0, 1, 6, 7 } : std::vector<int>{ 0, 1 };
    const int modes = int(mode_ids.size());
    std::vector<sink_ptr> sinks;
    std::vector<bool> ready(modes, false);
    for (int index = 0; index < modes; ++index) {
        const int mode = mode_ids[index];
        inputs.set(names[3], role_major || nonblocking_down_sched || (early_up_gate && mode) ? "1" : "0");
        inputs.set(names[4], poll && mode ? std::to_string(poll).c_str() : "0");
        inputs.set(names[5], (role_major || nonblocking_down_sched) && mode ? "1" : "0");
        inputs.set(names[6], nonblocking_down_sched && mode ? "1" : "0");
        const bool dynamic = schedules ? mode >= 7 : nonblocking_down_sched && mode;
        inputs.set(names[7], dynamic ? "1" : "0");
        inputs.set(names[8], "16");
        if (schedules) {
            ready[index] = mode == 1 || mode == 7;
            inputs.set(names[0], ready[index] ? "1" : "0");
            inputs.set(names[1], mode == 6 ? weighted.c_str() : "");
            inputs.set(names[2], mode >= 3 ? std::to_string(allowed.back()).c_str() : "");
            // Mode 7 exercises dynamic rows through ordinary compute_phase.
        }
        char error[512] = {};
        sinks.push_back(inputs.create(threads, depth, fixture_selector::resolve, &selector, error, sizeof(error)));
        require(bool(sinks[index]), std::string("schedule create: ") + error);
        configure_cache(sinks[index].get(), 8 * triplet_bytes(large), { 1, 2 }, 4);
    }
    for (const char * name : names) {
        if (original.count(name)) { inputs.set(name, original[name].c_str()); }
        else { inputs.unset(name); }
    }
    auto check_affinity = [&] {
        require(test_resources::process_cpus() == allowed, "sink changed the caller's affinity");
    };
    check_affinity();
    uint64_t token = 0;
    auto reset = [&] {
        for (auto & sink : sinks) { begin_cache(sink.get(), 99, 0, false); }
    };
    auto check = [&](const char * name, fixture & f, const std::vector<int32_t> & ids,
                     const std::vector<float> & weights, size_t hits) {
        selector.active = &f; ++token;
        std::vector<float> input(f.latent);
        for (int i = 0; i < f.latent; ++i) { input[i] = float((i * 13 + token) % 37) * 0.013f - 0.2f; }
        tensors meta(f, 1);
        const auto expected = reference(f, 1, input, ids, weights, 4, 25);
        std::vector<std::vector<float>> actual(modes);
        std::vector<sample_result> result(modes);
        for (int index = 0; index < modes; ++index) {
            const int mode = mode_ids[index];
            begin_cache(sinks[index].get(), 99, token);
            result[index] = run_checked(inputs, name, f, sinks[index].get(), meta, input, ids, weights,
                expected, 1, 25, hits, -1, &actual[index]);
            check_affinity();
            if (schedules) {
                require(ready[index] || result[index].stats.ready_scratch_bytes == 0,
                    "disabled ready-first retained scratch");
                if (ready[index] && result[index].stats.experts) {
                    require(result[index].stats.ready_scratch_bytes >= result[index].stats.experts * f.latent * sizeof(float),
                        "ready-first did not account scratch capacity");
                }
            }
            if (nonblocking_down_sched && mode && result[index].stats.experts) {
                const auto * down_traits = ggml_get_type_traits_cpu(f.down_type);
                const size_t hidden_bytes = ggml_row_size(down_traits->vec_dot_type, f.hidden);
                require(result[index].stats.ready_scratch_bytes >= result[index].stats.experts *
                        (size_t(f.latent) * sizeof(float) + hidden_bytes),
                    "nonblocking scheduler did not retain per-expert hidden scratch");
            }
        }
        for (int index = 1; index < modes; ++index) {
            const int mode = mode_ids[index];
            require(actual[0].size() == actual[index].size() &&
                std::memcmp(actual[0].data(), actual[index].data(), actual[0].size() * sizeof(float)) == 0,
                std::string(name) + ": schedule changed output bits mode=" + std::to_string(mode));
            require(result[0].stats.read_bytes == result[index].stats.read_bytes &&
                result[0].stats.cache_admitted == result[index].stats.cache_admitted,
                "schedule changed IO/cache accounting");
            std::printf("case=%s mode=%d bit_exact=1 elements=%zu reorder=%llu scratch=%llu status=PASS\n",
                name, mode, actual[0].size(), (unsigned long long) result[index].stats.ready_out_of_order_experts,
                (unsigned long long) result[index].stats.ready_scratch_bytes);
        }
    };
    check("job_cold", small, { 3, 4 }, { .3f, .7f }, 0);
    check("job_mixed", small, { 3, 5 }, { .4f, .6f }, 1);
    check("job_all_hit_reversed_order", small, { 5, -1, 3 }, { .6f, 0, .4f }, 2);
    for (int i = 0; i < 24; ++i) {
        check("job_repeated_generation", small, { 3, 5, 4 }, { .8f, .01f, .19f }, 3);
    }
    check("job_empty", small, { -1, -2 }, { 0, 0 }, 0);
    reset();
    check("job_ring_wrap", small, { 1, 2, 3, 4, 5, 6, 7, 8 },
        { .1f, .1f, .1f, .1f, .1f, .1f, .1f, .3f }, 0);
    reset(); check("job_grow_shape", large, { 3, 4 }, { .3f, .7f }, 0);
    reset(); check("job_shrink_shape", small, { 3 }, { 1 }, 0);
    if (schedules || early_up_gate || role_major || nonblocking_down_sched || poll) {
        reset(); check("ready_warm", small, { 1, 3 }, { .4f, .6f }, 0);
        check("ready_miss_before_hit_duplicate", small, { 2, 1, 2, 4, 3 }, { .1f, .2f, .3f, .1f, .3f }, 2);
        reset();
    }
    for (auto fault : { fixture::past_eof, fixture::short_tail }) {
        small.fault = fault; small.fault_expert = 7; small.fault_role = 2;
        tensors meta(small, 1);
        const int32_t ids[] = { 3, 7, 8 }; const float weights[] = { .2f, .3f, .5f };
        std::vector<float> input(small.latent, .1f), output(small.latent);
        for (auto & sink : sinks) {
            begin_cache(sink.get(), 99, ++token);
            char error[512] = {}; kimi_cpu_sink_stats stats = {};
            require(!kimi_cpu_sink_run(sink.get(), meta.role[0], meta.role[1], meta.role[2],
                input.data(), ids, weights, 3, 4, 25, output.data(), &stats, error, sizeof(error)),
                "schedule accepted short read");
            require(error[0] && stats.read_errors && !stats.pending_reads && !stats.fallback_count,
                "schedule failed to cancel and drain");
            require(!cache_snapshot(sink.get()).valid_bytes, "schedule retained failed cache bytes");
            check_affinity();
        }
        small.fault = fixture::none; small.fault_expert = small.fault_role = -1;
        check("job_short_read_recovery", small, { 3, 7, 8 }, { .2f, .3f, .5f }, 0);
    }
    std::printf("SCHEDULE_BIT_EXACT_OK threads=%d ring_experts=%d modes=%d early_up_gate=%d role_major=%d nonblocking_down_sched=%d poll_us=%d\n",
                threads, depth, modes, early_up_gate, role_major, nonblocking_down_sched, poll);
}

static void schedule_invalid_tests(fixture_inputs & inputs) {
    fixture f;
    const std::pair<const char *, const char *> invalid[] = {
        { "ready_first", "2" },
        { "prefetch_depth", "1" },
        { "early_up_gate", "2" },
        { "role_major", "2" },
        { "nonblocking_down_sched", "2" },
        { "poll_us", "-1" },
        { "poll_us", "10001" },
        { "poll_us", "100x" },
        { "dynamic_rows", "2" },
        { "dynamic_chunk_rows", "0" },
        { "dynamic_chunk_rows", "15" },
        { "dynamic_chunk_rows", "17" },
        { "dynamic_chunk_rows", "16385" },
        { "dynamic_chunk_rows", "64x" },
        { "row_weights", "1,0,1" },
        { "row_weights", "1,nan,1" },
        { "row_weights", "1,inf,1" },
        { "row_weights", "1,1,-1" },
        { "row_weights", "1,1,1," },
        { "io_cpus", "-1" },
        { "io_cpus", "1x" },
        { "io_cpus", "0,0" },
        { "io_cpus", "1024" },
        { "decode_participants", "513" },
    };
    for (const auto & item : invalid) {
        inputs.set(item.first, item.second);
        char error[512] = {};
        sink_ptr sink = inputs.create(4, 1, fixture::resolve, &f, error, sizeof(error));
        inputs.unset(item.first);
        require(!sink && error[0], std::string("invalid schedule accepted: ") + item.first + "=" + item.second);
    }
    char error[512] = {};
    sink_ptr incompatible;
    inputs.set("dynamic_rows", "1");
    inputs.set("row_weights", "1,1,1");
    incompatible = inputs.create(4, 1, fixture::resolve, &f, error, sizeof(error));
    inputs.unset("dynamic_rows");
    inputs.unset("row_weights");
    require(!incompatible && std::strstr(error, "mutually exclusive"), "dynamic rows accepted static row weights");
    inputs.set("role_major", "1");
    incompatible = inputs.create(4, 1, fixture::resolve, &f, error, sizeof(error));
    inputs.unset("role_major");
    require(!incompatible && std::strstr(error, "EARLY_UP_GATE=1"), "role-major accepted without early up/gate");

    inputs.set("nonblocking_down_sched", "1");
    incompatible = inputs.create(4, 1, fixture::resolve, &f, error, sizeof(error));
    require(!incompatible && std::strstr(error, "ROLE_MAJOR=1"), "nonblocking scheduler accepted without role-major");
    inputs.set("role_major", "1");
    inputs.set("early_up_gate", "1");
    inputs.set("ready_first", "0");
    incompatible = inputs.create(4, 1, fixture::resolve, &f, error, sizeof(error));
    require(!incompatible && std::strstr(error, "READY_FIRST=1"), "nonblocking scheduler accepted unordered reduction");
    inputs.unset("nonblocking_down_sched");
    inputs.unset("role_major");
    inputs.unset("early_up_gate");
    inputs.unset("ready_first");
    // Inputs counted in participants are checked against each run's: three here.
    tensors meta(f, 1);
    std::vector<float> input(f.latent, .1f), output(f.latent);
    const int32_t id = 1; const float weight = 1;
    const std::pair<const char *, const char *> per_run[] = {
        { "row_weights", "1,1" },
        { "decode_participants", "4" },
    };
    for (const auto & item : per_run) {
        inputs.set(item.first, item.second);
        sink_ptr sink = inputs.create(4, 1, fixture::resolve, &f, error, sizeof(error));
        inputs.unset(item.first);
        require(bool(sink), std::string("per-run schedule rejected at creation: ") + error);
        require(!kimi_cpu_sink_run(sink.get(), meta.role[0], meta.role[1], meta.role[2], input.data(), &id, &weight, 1,
            4, 25, output.data(), nullptr, error, sizeof(error)) && error[0],
            std::string("per-run schedule accepted: ") + item.first + "=" + item.second);
        std::printf("case=per_run_%s status=PASS error=%s\n", item.first, error);
    }
    // One call at a time, of 1..512 participants.
    sink_ptr sink = inputs.create(2, 1, fixture::resolve, &f, error, sizeof(error));
    require(bool(sink), std::string("create: ") + error);
    require(!kimi_cpu_sink_call_open(sink.get(), 0, error, sizeof(error)) && error[0], "empty call accepted");
    require(!kimi_cpu_sink_call_open(sink.get(), 513, error, sizeof(error)) && error[0], "oversized call accepted");
    require(kimi_cpu_sink_call_open(sink.get(), 1, error, sizeof(error)), error);
    require(!kimi_cpu_sink_call_open(sink.get(), 1, error, sizeof(error)) && error[0], "second open call accepted");
    kimi_cpu_sink_call_close(sink.get());
    std::printf("SCHEDULE_INVALID_SETTINGS_REJECTED cases=%zu\n", sizeof(invalid) / sizeof(invalid[0]) + 2);
}

static void cache_release_test(fixture_inputs & inputs, int threads, int depth) {
    fixture f; f.experts = 32; make_synthetic(f);
    auto sink = make_sink(inputs, f, threads, depth);
    const uint64_t bytes = 64 * triplet_bytes(f);
    configure_cache(sink.get(), bytes, { 1, 2 }, 32);
    begin_cache(sink.get(), 1, 1);
    std::vector<float> input(f.latent, .13f), weights(32, 1.0f / 32);
    std::vector<int32_t> ids(32); for (int i = 0; i < 32; ++i) { ids[i] = i; }
    cache_case(inputs, "cache_physical_fill_layer1", f, sink.get(), 1, input, ids, weights, 0, 32);
    cache_case(inputs, "cache_physical_fill_layer2", f, sink.get(), 2, input, ids, weights, 0, 32);
    const auto full = cache_snapshot(sink.get());
    require(full.mapped_bytes == bytes && full.committed_bytes == bytes && full.valid_bytes == bytes,
            "physical release fixture did not fill its budget");
    const uint64_t before = anonymous_bytes();
    begin_cache(sink.get(), 1, 2, false);
    const uint64_t after = anonymous_bytes();
    const auto empty = cache_snapshot(sink.get());
    require(empty.mapped_bytes == 0 && empty.committed_bytes == 0 && empty.valid_bytes == 0 &&
            empty.capacity_bytes == full.capacity_bytes, "prefill retained payload or lost capacity configuration");
    // Dedicated arena slots must actually release pages. Allow small allocator and
    // counter noise, but not an allocator retaining the whole payload.
    require(before >= after && before - after >= bytes - test_resources::anonymous_noise_bytes,
            "prefill zeroed accounting without releasing anonymous cache pages");
    std::printf("case=cache_prefill_physical_release payload=%llu anonymous_before=%llu anonymous_after=%llu status=PASS\n",
            (unsigned long long) bytes, (unsigned long long) before, (unsigned long long) after);
    cache_case(inputs, "cache_prefill_no_admission", f, sink.get(), 1, input, { 1 }, { 1 }, 0, 0);
    begin_cache(sink.get(), 1, 3);
    cache_case(inputs, "cache_decode_after_prefill_cold", f, sink.get(), 1, input, { 1 }, { 1 }, 0, 1);
}

static void direct_cache_memlock_child(fixture_inputs & inputs, fixture & f, const std::vector<float> & input) {
    const char * mode = inputs.get("read_fixed");
    if (!mode || std::strcmp(mode, "1")) { return; }
#if !defined(__linux__)
    // Linux only: io_uring fixed buffers count against RLIMIT_MEMLOCK; other rings have no such budget.
    (void) f; (void) input;
    std::printf("case=direct_cache_memlock_child status=SKIP reason=linux_rlimit_memlock\n");
#else
    // No parent sink exists yet. Lower limits/capabilities only in this child.
    std::fflush(nullptr);
    const pid_t child = fork(); require(child >= 0, "direct cache memlock fork failed");
    if (child == 0) {
        try {
            // The kernel may charge ring setup itself to memlock. Create the
            // real ring first, then deny only the buffer registration under test.
            auto sink = make_sink(inputs, f, 2, 1);
            drop_child_memlock();
            configure_cache(sink.get(), 2 * triplet_bytes(f), { 1 }, 2);
            begin_cache(sink.get(), 1, 1);
            cache_case(inputs, "direct_cache_memlock0_all_admit", f, sink.get(), 1, input, { 1, 2 }, { .3f, .7f }, 0, 2);
            begin_cache(sink.get(), 1, 2);
            cache_case(inputs, "direct_cache_memlock0_all_hit", f, sink.get(), 1, input, { 1, 2 }, { .7f, .3f }, 2, 0);
            begin_cache(sink.get(), 1, 3);
            tensors meta(f, 1); char error[512] = {}; kimi_cpu_sink_stats stats = {};
            const int32_t ids[] = { 1, 2, 3 }; const float weights[] = { .2f, .3f, .5f };
            std::vector<float> output(f.latent);
            const bool ok = kimi_cpu_sink_run(sink.get(), meta.role[0], meta.role[1], meta.role[2],
                    input.data(), ids, weights, 3, 4, 25, output.data(), &stats, error, sizeof(error));
            require(!ok && std::strstr(error, "IORING_REGISTER_BUFFERS") && stats.read_requests == 0 &&
                    stats.fallback_count == 0, "RF bypass under memlock0 did not fail closed before submitting IO");
            check_io_targets(inputs, stats, false);
            const auto cleared = cache_snapshot(sink.get());
            require(cleared.mapped_bytes == 0 && cleared.committed_bytes == 0 && cleared.valid_bytes == 0,
                    "failed bypass registration retained direct cache targets");
            begin_cache(sink.get(), 1, 4);
            cache_case(inputs, "direct_cache_memlock0_after_registration_failure", f, sink.get(), 1, input,
                    { 1, 2 }, { .3f, .7f }, 0, 2);
            sink.reset();
            std::printf("DIRECT_CACHE_MEMLOCK0_OK cache_READ_succeeded=1 bypass_READ_FIXED_failed_closed=1\n");
            std::fflush(nullptr); _exit(0);
        } catch (const std::exception & error) {
            std::fprintf(stderr, "DIRECT_CACHE_MEMLOCK0_FAILED: %s\n", error.what());
            _exit(1);
        }
    }
    int status = 0; pid_t result;
    do { result = waitpid(child, &status, 0); } while (result < 0 && errno == EINTR);
    require(result == child && WIFEXITED(status) && WEXITSTATUS(status) == 0, "direct cache memlock child did not pass");
#endif
}

static void direct_cache_target_tests(fixture_inputs & inputs, fixture & f, const std::vector<float> & input, int threads, int depth) {
    const uint64_t unit = triplet_bytes(f);
    auto sink = make_sink(inputs, f, threads, depth);
    configure_cache(sink.get(), 8 * unit, { 1 }, 8);
    begin_cache(sink.get(), 1, 1);
    const auto cold = cache_case(inputs, "direct_cache_all_admit_over_ring_depth", f, sink.get(), 1, input,
            { 0, 1, 2, 3, 4, 5, 6, 7 }, { .02f, .03f, .05f, .1f, .1f, .15f, .2f, .35f }, 0, 8);
    require(cold.stats.cache_new_slot_bytes == 8 * unit, "cold direct reads did not reserve all first-touch pages");
    begin_cache(sink.get(), 1, 2);
    cache_case(inputs, "direct_cache_all_hit_after_fill", f, sink.get(), 1, input,
            { 7, 0, 6, 1, 5, 2, 4, 3 }, { .2f, .15f, .05f, .2f, .1f, .05f, .1f, .15f }, 8, 0);
    begin_cache(sink.get(), 1, 3);
    cache_case(inputs, "direct_cache_all_hit_then_zero_active", f, sink.get(), 1, input, { -1, -7 }, { 0, 0 }, 0, 0);

    configure_cache(sink.get(), 3 * unit, { 1 }, 3);
    begin_cache(sink.get(), 2, 1);
    cache_case(inputs, "direct_cache_mixed_warm_one", f, sink.get(), 1, input, { 1 }, { 1 }, 0, 1);
    begin_cache(sink.get(), 2, 2);
    const auto mixed = cache_case(inputs, "direct_cache_hit_admit_duplicate_bypass_order", f, sink.get(), 1, input,
            { 1, 2, 2, 3, 4, 1, 3, 5 }, { .1f, .15f, .05f, .25f, .1f, .2f, .05f, .1f }, 2, 2);
    require(mixed.stats.cache_fill_read_requests == 6 && mixed.stats.ring_read_requests == 12,
            "mixed target fixture did not exercise both cache and bypass destinations");
    begin_cache(sink.get(), 2, 3);
    cache_case(inputs, "direct_cache_mixed_next_token_resident", f, sink.get(), 1, input, { 1, 2, 3 }, { .2f, .3f, .5f }, 3, 0);

    // Each single-role positive short read targets an admitted cache slot, with
    // the other roles (and, at depth>1, another expert) submitted in the batch.
    // Completion accounting must prove kernel ownership ended before unmap.
    for (int role = 0; role < 3; ++role) {
        configure_cache(sink.get(), 4 * unit, { 1 }, 4);
        begin_cache(sink.get(), 3 + role, 1);
        tensors meta(f, 1); char error[512] = {}; kimi_cpu_sink_stats stats = {};
        const int32_t ids[] = { 1, 2, 3, 4 }; const float weights[] = { .1f, .2f, .3f, .4f };
        std::vector<float> output(f.latent);
        f.fault = fixture::short_tail; f.fault_expert = 1; f.fault_role = role;
        const bool ok = kimi_cpu_sink_run(sink.get(), meta.role[0], meta.role[1], meta.role[2],
                input.data(), ids, weights, 4, 4, 25, output.data(), &stats, error, sizeof(error));
        f.fault = fixture::none; f.fault_expert = -1; f.fault_role = -1;
        require(!ok && error[0] && stats.read_errors > 0 && stats.fallback_count == 0,
                "admitted single-role short read did not fail closed");
        check_io_targets(inputs, stats, false);
        require(stats.ring_read_requests == 0 && stats.cache_fill_read_requests >= 3 && stats.cache_fill_read_bytes >= 4096,
                "short read did not exercise the direct cache destination");
        require(stats.cache_new_slot_bytes == 4 * unit, "failed direct batch did not account for every reserved target");
        if (depth > 1) { require(stats.cache_fill_read_requests >= 6, "short read fixture did not submit a second expert"); }
        const auto cleared = cache_snapshot(sink.get());
        require(cleared.mapped_bytes == 0 && cleared.committed_bytes == 0 && cleared.valid_bytes == 0,
                "direct read failure did not unmap every cache target after draining");
        std::printf("case=direct_cache_role_short_read_drained role=%d reads=%llu completions=%llu pending=%llu status=PASS\n",
                role, (unsigned long long) stats.read_requests, (unsigned long long) stats.read_completions,
                (unsigned long long) stats.pending_reads);
        if (role == 0) {
            cache_case(inputs, "direct_cache_short_read_then_empty", f, sink.get(), 1, input, { -1 }, { 0 }, 0, 0);
            configure_cache(sink.get(), 4 * unit, { 1 }, 4);
        } else if (role == 1) {
            begin_cache(sink.get(), 3 + role, 2, false);
        } else {
            sink.reset(); // immediate destructor after the failed direct read
            sink = make_sink(inputs, f, threads, depth);
            configure_cache(sink.get(), 4 * unit, { 1 }, 4);
        }
        begin_cache(sink.get(), 10 + role, 1);
        cache_case(inputs, "direct_cache_short_read_remap_cold", f, sink.get(), 1, input,
                { 1, 2, 3, 4 }, { .1f, .2f, .3f, .4f }, 0, 4);
        begin_cache(sink.get(), 10 + role, 2);
        cache_case(inputs, "direct_cache_short_read_recovered_hits", f, sink.get(), 1, input,
                { 4, 3, 2, 1 }, { .1f, .2f, .3f, .4f }, 4, 0);
    }
    std::printf("DIRECT_CACHE_TARGETS_OK\n");
}

static void cache_policy_tests(fixture_inputs & inputs, int threads, int depth) {
    const char * policy = inputs.get("cache_policy");
    const char * allocation = inputs.get("cache_allocation");
    require(policy && allocation && std::strcmp(policy, "window"), "policy tests require a persistent policy and allocation");
    const bool recency = !std::strcmp(policy, "lru");
    const bool aging = !std::strcmp(policy, "decay") || !std::strcmp(policy, "count_decay2");
    const bool priority = !std::strcmp(allocation, "priority");
    const bool shared = !std::strcmp(allocation, "global");
    fixture f; make_synthetic(f);
    auto sink = make_sink(inputs, f, threads, depth);
    const uint64_t unit = triplet_bytes(f);
    std::vector<float> input(f.latent, .13f);

    configure_cache(sink.get(), 2 * unit, { 1 }, 2);
    begin_cache(sink.get(), 1, 1);
    cache_case(inputs, "policy_cold", f, sink.get(), 1, input, { 1, 2 }, { .4f, .6f }, 0, 2);
    begin_cache(sink.get(), 1, 2);
    cache_case(inputs, "policy_empty_preserves_history", f, sink.get(), 1, input, { -1 }, { 0 }, 0, 0);
    begin_cache(sink.get(), 1, 3);
    cache_case(inputs, "policy_miss_before_later_hit_protected", f, sink.get(), 1, input, { 3, 1 }, { .3f, .7f }, 1, 1);
    begin_cache(sink.get(), 1, 4);
    cache_case(inputs, "policy_all_residents_protected_duplicate_hit", f, sink.get(), 1, input,
        { 3, 3, 4, 1 }, { .1f, .2f, .3f, .4f }, 3, 0);
    begin_cache(sink.get(), 1, 5);
    cache_case(inputs, "policy_unlisted_layer_bypass", f, sink.get(), 2, input, { 1 }, { 1 }, 0, 0);
    begin_cache(sink.get(), 1, 6);
    cache_case(inputs, "policy_layer_skip_retains_entries", f, sink.get(), 1, input, { 1, 3 }, { .2f, .8f }, 2, 0);
    cache_case(inputs, "policy_repeated_layer_same_token_bypass", f, sink.get(), 1, input, { 1, 3 }, { .7f, .3f }, 0, 0);
    begin_cache(sink.get(), 1, 8);
    cache_case(inputs, "policy_token_jump_cold", f, sink.get(), 1, input, { 1 }, { 1 }, 0, 1);
    begin_cache(sink.get(), 2, 9);
    cache_case(inputs, "policy_epoch_cold", f, sink.get(), 1, input, { 1 }, { 1 }, 0, 1);
    begin_cache(sink.get(), 2, 10, false);
    require(cache_snapshot(sink.get()).mapped_bytes == 0, "persistent cache survived prefill");
    begin_cache(sink.get(), 2, 11);
    cache_case(inputs, "policy_prefill_cold", f, sink.get(), 1, input, { 1 }, { 1 }, 0, 1);

    configure_cache(sink.get(), 2 * unit, { 2, 1 }, 2);
    begin_cache(sink.get(), 3, 1);
    cache_case(inputs, "policy_allocation_low_priority_layer", f, sink.get(), 1, input, { 1 }, { 1 }, 0, priority ? 0 : 1);
    cache_case(inputs, "policy_layer_id_is_part_of_key", f, sink.get(), 2, input, { 1 }, { 1 }, 0, 1);
    begin_cache(sink.get(), 3, 2);
    cache_case(inputs, "policy_allocation_low_priority_hit", f, sink.get(), 1, input, { 1 }, { 1 }, priority ? 0 : 1, 0);
    cache_case(inputs, "policy_allocation_high_priority_hit", f, sink.get(), 2, input, { 1 }, { 1 }, 1, 0);
    configure_cache(sink.get(), 2 * unit, { 1, 2 }, 1);
    begin_cache(sink.get(), 4, 1);
    cache_case(inputs, "policy_global_can_borrow_other_layer_slot", f, sink.get(), 1, input, { 1, 2 }, { .4f, .6f }, 0, shared ? 2 : 1);

    configure_cache(sink.get(), 2 * unit, { 1 }, 2);
    begin_cache(sink.get(), 5, 1);
    cache_case(inputs, "policy_heat_cold", f, sink.get(), 1, input, { 1, 2 }, { .4f, .6f }, 0, 2);
    begin_cache(sink.get(), 5, 2);
    cache_case(inputs, "policy_heat_train", f, sink.get(), 1, input, { 1, 2 }, { .4f, .6f }, 2, 0);
    begin_cache(sink.get(), 5, 3);
    cache_case(inputs, "policy_cold_scan_admission", f, sink.get(), 1, input, { 3 }, { 1 }, 0, recency ? 1 : 0);
    begin_cache(sink.get(), 5, 4);
    cache_case(inputs, "policy_equal_frequency_admitted", f, sink.get(), 1, input, { 3 }, { 1 }, recency ? 1 : 0, recency ? 0 : 1);

    configure_cache(sink.get(), unit, { 1 }, 1);
    begin_cache(sink.get(), 6, 1);
    cache_case(inputs, "policy_decay_warm", f, sink.get(), 1, input, { 1 }, { 1 }, 0, 1);
    begin_cache(sink.get(), 6, 2);
    cache_case(inputs, "policy_decay_heat", f, sink.get(), 1, input, { 1 }, { 1 }, 1, 0);
    for (uint64_t token = 3; token < 32; ++token) { begin_cache(sink.get(), 6, token); }
    begin_cache(sink.get(), 6, 32);
    cache_case(inputs, "policy_decay_forgets_old_heat", f, sink.get(), 1, input, { 2 }, { 1 }, 0, recency || aging ? 1 : 0);

    // A failed direct admission must drain before unmapping, even when old
    // persistent entries would otherwise survive the next token.
    configure_cache(sink.get(), 3 * unit, { 1 }, 3);
    begin_cache(sink.get(), 7, 1);
    cache_case(inputs, "policy_fault_warm", f, sink.get(), 1, input, { 1 }, { 1 }, 0, 1);
    begin_cache(sink.get(), 7, 2);
    tensors meta(f, 1); std::vector<float> output(f.latent);
    int32_t ids[] = { 2, 1, 3 }; float weights[] = { .2f, .3f, .5f };
    char error[512] = {}; kimi_cpu_sink_stats stats = {};
    f.fault = fixture::short_tail; f.fault_expert = 2; f.fault_role = 1;
    const bool ok = kimi_cpu_sink_run(sink.get(), meta.role[0], meta.role[1], meta.role[2], input.data(),
        ids, weights, 3, 4, 25, output.data(), &stats, error, sizeof(error));
    f.fault = fixture::none; f.fault_expert = f.fault_role = -1;
    require(!ok && error[0] && stats.read_errors > 0, "persistent direct admission fault did not fail closed");
    check_io_targets(inputs, stats, false);
    require(cache_snapshot(sink.get()).mapped_bytes == 0, "persistent failed admission retained mapping");
    begin_cache(sink.get(), 7, 3);
    cache_case(inputs, "policy_fault_recovers_cold", f, sink.get(), 1, input, { 1 }, { 1 }, 0, 1);
    std::printf("CACHE_POLICY_OK policy=%s allocation=%s\n", policy, allocation);
}

static void elastic_cache_tests(fixture_inputs & inputs, int threads, int depth);

static void max_cache_tests(fixture_inputs & inputs, int threads, int depth) {
    fixture f; make_synthetic(f);
    std::vector<sink_ptr> sinks;
    for (int arm = 0; arm < 4; ++arm) {
        inputs.set("cache_policy", arm < 2 ? "lru" : "count_decay2");
        inputs.set("cache_keep", arm == 3 ? "1" : "0");
        sinks.push_back(make_sink(inputs, f, threads, depth));
        configure_cache(sinks.back().get(), arm ? 2 * triplet_bytes(f) : 0, { 1 }, 2);
    }
    tensors meta(f, 1);
    std::vector<float> input(f.latent, .17f);
    uint64_t comparisons = 0;
    auto check = [&](uint64_t epoch, uint64_t token, const std::vector<int32_t> & ids,
                     const std::vector<float> & weights, bool retained_hit = false) {
        std::vector<float> expected;
        for (size_t arm = 0; arm < sinks.size(); ++arm) {
            begin_cache(sinks[arm].get(), epoch, token);
            std::vector<float> output(f.latent);
            kimi_cpu_sink_stats stats = {}; char error[512] = {};
            require(kimi_cpu_sink_run(sinks[arm].get(), meta.role[0], meta.role[1], meta.role[2], input.data(),
                ids.data(), weights.data(), int(ids.size()), 4, 25, output.data(), &stats, error, sizeof(error)), error);
            check_io_targets(inputs, stats, true);
            if (arm == 0) { expected = output; }
            else {
                require(!std::memcmp(expected.data(), output.data(), output.size() * sizeof(float)),
                        "max-cache changed a computed output bit");
                ++comparisons;
            }
            if (retained_hit) {
                require(stats.cache_hits == (arm == 3 ? ids.size() : 0), "cross-request retained hit mismatch");
            }
            if (epoch == 2 && token == 2) { require(stats.cache_hits == 0, "changed epoch retained cache entries"); }
        }
        input[0] += .013f; // Cache weights, never reuse a previous activation/output.
    };
    check(1, 1, { 1, 2 }, { .4f, .6f });
    check(1, 2, { 1, 2 }, { .7f, .3f });
    check(1, 3, { 3 }, { 1 });
    check(1, 4, { 3 }, { 1 });
    check(1, 5, { 1, 3 }, { .2f, .8f });
    for (size_t arm = 0; arm < sinks.size(); ++arm) {
        begin_cache(sinks[arm].get(), 0, 0, false);
        require(bool(cache_snapshot(sinks[arm].get()).mapped_bytes) == (arm == 3), "wrong prefill retention state");
    }
    check(1, 1, { 3, 1 }, { .9f, .1f }, true);
    check(2, 2, { 3, 1 }, { .1f, .9f });
    for (auto & sink : sinks) {
        kimi_cpu_sink_cache_reset(sink.get());
        require(cache_snapshot(sink.get()).mapped_bytes == 0, "explicit reset retained payload");
    }
    check(2, 1, { 1, 3, 1, -1 }, { .1f, .3f, .6f, 0 });
    // A short read must drain and release even with retention enabled.
    auto & retained = sinks.back();
    begin_cache(retained.get(), 2, 2);
    f.fault = fixture::short_tail; f.fault_expert = 4; f.fault_role = 1;
    int32_t ids[] = { 4, 1 }; float weights[] = { .5f, .5f };
    std::vector<float> output(f.latent); kimi_cpu_sink_stats stats = {}; char error[512] = {};
    require(!kimi_cpu_sink_run(retained.get(), meta.role[0], meta.role[1], meta.role[2], input.data(),
        ids, weights, 2, 4, 25, output.data(), &stats, error, sizeof(error)), "retained short read succeeded");
    check_io_targets(inputs, stats, false);
    require(stats.read_errors && cache_snapshot(retained.get()).mapped_bytes == 0, "retained error did not release");
    std::printf("MAX_CACHE_BIT_EXACT_OK comparisons=%llu threads=%d ring=%d\n",
        (unsigned long long) comparisons, threads, depth);
}

static void cache_tests(fixture_inputs & inputs, int threads, int depth) {
    fixture f; make_synthetic(f);
    const uint64_t unit = triplet_bytes(f);
    std::vector<float> input(f.latent);
    for (size_t i = 0; i < input.size(); ++i) { input[i] = float(int(i % 23) - 11) * .023f; }
    direct_cache_memlock_child(inputs, f, input);
    direct_cache_target_tests(inputs, f, input, threads, depth);
    auto sink = make_sink(inputs, f, threads, depth);
    configure_cache(sink.get(), 2 * unit + 17, { 1 }, 2);
    cache_case(inputs, "cache_before_decode_bypass", f, sink.get(), 1, input, { 1, 2 }, { .4f, .6f }, 0, 0);
    require(cache_snapshot(sink.get()).mapped_bytes == 0, "nondecode run mapped cache payload");
    begin_cache(sink.get(), 1, 0, false);
    cache_case(inputs, "cache_explicit_prefill_bypass", f, sink.get(), 1, input, { 1, 2 }, { .4f, .6f }, 0, 0);
    begin_cache(sink.get(), 1, 1);
    cache_case(inputs, "cache_cold", f, sink.get(), 1, input, { 1, 2 }, { .4f, .6f }, 0, 2);
    require(cache_snapshot(sink.get()).capacity_bytes == 2 * unit, "cache rounded a partial triplet up");
    begin_cache(sink.get(), 1, 2);
    input[3] += .125f;
    cache_case(inputs, "cache_full_hit_changed_activation_weights", f, sink.get(), 1, input, { 1, 2 }, { .8f, .2f }, 2, 0);
    cache_case(inputs, "cache_repeat_same_token_bypass", f, sink.get(), 1, input, { 4, 5 }, { .3f, .7f }, 0, 0);
    begin_cache(sink.get(), 1, 3);
    cache_case(inputs, "cache_window_intersection_B", f, sink.get(), 1, input, { 2, 3 }, { .3f, .7f }, 1, 1);
    begin_cache(sink.get(), 1, 4);
    cache_case(inputs, "cache_window_intersection_C", f, sink.get(), 1, input, { 1, 3 }, { .2f, .8f }, 1, 1);
    begin_cache(sink.get(), 1, 5);
    cache_case(inputs, "cache_unselected_layer_bypass", f, sink.get(), 2, input, { 1, 2, 3, 4 }, { .1f, .2f, .3f, .4f }, 0, 0);
    cache_case(inputs, "cache_selected_survives_unselected", f, sink.get(), 1, input, { 1, 3 }, { .1f, .9f }, 2, 0);
    begin_cache(sink.get(), 1, 6);
    cache_case(inputs, "cache_skip_selected_layer_token", f, sink.get(), 2, input, { 1 }, { 1 }, 0, 0);
    begin_cache(sink.get(), 1, 7);
    cache_case(inputs, "cache_missing_previous_layer_is_cold", f, sink.get(), 1, input, { 1, 3 }, { .1f, .9f }, 0, 2);
    begin_cache(sink.get(), 1, 10);
    cache_case(inputs, "cache_nonconsecutive_token_is_cold", f, sink.get(), 1, input, { 1, 3 }, { .1f, .9f }, 0, 2);
    begin_cache(sink.get(), 2, 11);
    cache_case(inputs, "cache_new_epoch_is_cold", f, sink.get(), 1, input, { 1, 3 }, { .1f, .9f }, 0, 2);

    // Two disjoint full generations must reuse the same physical budget, never
    // allocate an unaccounted second payload while publishing the next window.
    for (uint64_t token = 12; token < 28; ++token) {
        begin_cache(sink.get(), 2, token);
        const int32_t first = token % 2 ? 5 : 7;
        cache_case(inputs, "cache_full_budget_generation_replace", f, sink.get(), 1, input,
                { first, first + 1 }, { .37f, .63f }, 0, 2);
        const auto state = cache_snapshot(sink.get());
        require(state.valid_bytes == 2 * unit && state.committed_bytes == 2 * unit && state.peak_bytes <= 2 * unit + 17,
                "window replacement exceeded one physical cache generation");
    }
    configure_cache(sink.get(), unit - 1, { 1 }, 2);
    begin_cache(sink.get(), 3, 1);
    cache_case(inputs, "cache_sub_triplet_budget_bypass", f, sink.get(), 1, input, { 1 }, { 1 }, 0, 0);
    require(cache_snapshot(sink.get()).capacity_bytes == 0, "sub-triplet budget allocated payload");
    configure_cache(sink.get(), 0, { 1 }, 2);
    begin_cache(sink.get(), 3, 2);
    cache_case(inputs, "cache_zero_budget_bypass", f, sink.get(), 1, input, { 1 }, { 1 }, 0, 0);

    // Priority quotas may end in a partial layer. No later layer may take the
    // first layer's storage simply because it is visited later in the token.
    configure_cache(sink.get(), 3 * unit, { 2, 1 }, 2);
    begin_cache(sink.get(), 4, 1);
    cache_case(inputs, "cache_priority_partial_layer_cold", f, sink.get(), 1, input, { 1, 2 }, { .4f, .6f }, 0, 1);
    cache_case(inputs, "cache_priority_full_layer_cold", f, sink.get(), 2, input, { 1, 2 }, { .4f, .6f }, 0, 2);
    begin_cache(sink.get(), 4, 2);
    cache_case(inputs, "cache_priority_partial_layer_hit", f, sink.get(), 1, input, { 1, 2 }, { .4f, .6f }, 1, 0);
    cache_case(inputs, "cache_priority_full_layer_hit", f, sink.get(), 2, input, { 1, 2 }, { .4f, .6f }, 2, 0);

    configure_cache(sink.get(), 2 * unit, { 1 }, 2);
    begin_cache(sink.get(), 5, 1);
    cache_case(inputs, "cache_duplicates_cold_not_temporal_hits", f, sink.get(), 1, input,
            { 3, -1, 3, 4, 3, -7 }, { .2f, 0, .1f, 0, .7f, 0 }, 0, 2);
    begin_cache(sink.get(), 5, 2);
    cache_case(inputs, "cache_duplicates_resident_ordered", f, sink.get(), 1, input,
            { 3, -1, 3, 4, 3, -7 }, { .2f, 0, .1f, 0, .7f, 0 }, 4, 0, 0);
    begin_cache(sink.get(), 5, 3);
    cache_case(inputs, "cache_empty_token", f, sink.get(), 1, input, { -1, -7 }, { 0, 0 }, 0, 0);
    begin_cache(sink.get(), 5, 4);
    cache_case(inputs, "cache_empty_previous_window_is_cold", f, sink.get(), 1, input, { 3 }, { 1 }, 0, 1);

    // Interleave resident and missed requests across every ring wrap. Repeated
    // misses are independent reads in this API, but publish only one entry.
    configure_cache(sink.get(), 4 * unit, { 1 }, 4);
    begin_cache(sink.get(), 6, 1);
    cache_case(inputs, "cache_ring_mix_warm", f, sink.get(), 1, input, { 1, 3 }, { .4f, .6f }, 0, 2);
    begin_cache(sink.get(), 6, 2);
    cache_case(inputs, "cache_ring_mixed_wrap", f, sink.get(), 1, input,
            { 1, 2, 3, 4, 1, -1, 2, 3, 4 }, { .1f, .2f, .15f, .05f, .1f, 0, .2f, .15f, .05f }, 4, 2);

    for (auto fault : { fixture::missing, fixture::unaligned_offset, fixture::wrong_size, fixture::past_eof }) {
        configure_cache(sink.get(), 2 * unit, { 1 }, 2);
        begin_cache(sink.get(), 7, 1);
        cache_case(inputs, "cache_fault_warm", f, sink.get(), 1, input, { 1 }, { 1 }, 0, 1);
        begin_cache(sink.get(), 7, 2);
        tensors meta(f, 1); std::vector<float> output(f.latent);
        int32_t ids[] = { 1, 2, 3 }; float weights[] = { .2f, .3f, .5f };
        char error[512] = {}; kimi_cpu_sink_stats stats = {};
        f.fault = fault; f.fault_expert = 3;
        const bool ok = kimi_cpu_sink_run(sink.get(), meta.role[0], meta.role[1], meta.role[2], input.data(),
                ids, weights, 3, 4, 25, output.data(), &stats, error, sizeof(error));
        f.fault = fixture::none; f.fault_expert = -1;
        require(!ok && error[0] && stats.fallback_count == 0, "mixed cache fault did not fail closed");
        check_io_targets(inputs, stats, false);
        if (fault == fixture::past_eof) { require(stats.read_errors > 0, "mixed short read did not report error"); }
        require(cache_snapshot(sink.get()).valid_bytes == 0, "failed run retained a cache window");
        if (fault == fixture::past_eof) {
            cache_case(inputs, "cache_short_read_then_empty_clears_io_error", f, sink.get(), 1, input,
                    { -1, -7 }, { 0, 0 }, 0, 0);
        }
        begin_cache(sink.get(), 7, 3);
        cache_case(inputs, "cache_fault_drained_cold_recovery", f, sink.get(), 1, input,
                { 1, 2, 3 }, { .2f, .3f, .5f }, 0, 2);
    }

    // A different pack with identical names/shapes/IDs has different weights.
    // Changing its epoch must invalidate all old payload, even with a reused ctx.
    fixture other; make_synthetic(other, 20260909);
    fixture_selector selector; selector.active = &f;
    char error[512] = {};
    sink_ptr identities = inputs.create(threads, depth, fixture_selector::resolve, &selector, error, sizeof(error));
    require(bool(identities), std::string("identity sink create: ") + error);
    configure_cache(identities.get(), 2 * unit, { 1 }, 2);
    begin_cache(identities.get(), 100, 1);
    cache_case(inputs, "cache_model_A_warm", f, identities.get(), 1, input, { 1, 2 }, { .4f, .6f }, 0, 2);
    selector.active = &other; begin_cache(identities.get(), 101, 2);
    cache_case(inputs, "cache_model_B_same_names_new_epoch", other, identities.get(), 1, input, { 1, 2 }, { .4f, .6f }, 0, 2);
    begin_cache(identities.get(), 101, 3);
    cache_case(inputs, "cache_model_B_resident", other, identities.get(), 1, input, { 1, 2 }, { .4f, .6f }, 2, 0);
    selector.active = &f; begin_cache(identities.get(), 102, 4);
    cache_case(inputs, "cache_model_A_return_is_cold", f, identities.get(), 1, input, { 1, 2 }, { .4f, .6f }, 0, 2);
    // Defence beyond the immutable-pack contract: fstat identity must catch a
    // caller replacing the source without changing epoch, including fd reuse.
    selector.active = &other; begin_cache(identities.get(), 102, 5);
    cache_case(inputs, "cache_same_epoch_different_pack_fd", other, identities.get(), 1, input, { 1, 2 }, { .4f, .6f }, 0, 2);
    require(f.fd != other.fd && test_resources::fd_dup2(f.fd, other.fd) == other.fd, "pack fd reuse fixture failed");
    begin_cache(identities.get(), 102, 6);
    // The resolver still returns other's fd number and identical offsets, but
    // that descriptor now names A. The independent reference uses A's bytes.
    cache_case(inputs, "cache_same_epoch_reused_fd_new_inode", f, identities.get(), 1, input, { 1, 2 }, { .4f, .6f }, 0, 2);

    const int32_t layer = 1;
    for (int quota : { 0, 897 }) {
        require(!kimi_cpu_sink_cache_configure(sink.get(), unit, &layer, 1, quota, error, sizeof(error)) && error[0],
                "invalid cache quota was accepted");
    }
    identities.reset(); sink.reset();
    cache_release_test(inputs, threads, depth);
    elastic_cache_tests(inputs, threads, depth);
    std::printf("CACHE_CPU_SINK_OK threads=%d ring_experts=%d\n", threads, depth);
}

static void cache_hitrate_tests(fixture_inputs & inputs, int threads, int depth) {
    fixture f; make_synthetic(f);
    const uint64_t unit = triplet_bytes(f);
    const char * quotas = "cache_layer_quotas";
    const char * reserve = "cache_reserve_bytes";
    inputs.set("cache_policy", "count_decay2");
    inputs.set("cache_allocation", "uniform");
    inputs.set("cache_keep", "1");
    unsigned rejected = 0;
    for (const char * value : { "", "1", "1:", ":1", "1:2,", "1:2,,2:2", "1:2,1:2", "1:-1", "1024:1", "1:897", "1:2x", "1:1:2" }) {
        inputs.set(quotas, value); char error[512] = {};
        sink_ptr s = inputs.create(threads, depth, fixture::resolve, &f, error, sizeof(error));
        require(!s && error[0], "invalid layer quota accepted"); ++rejected;
    }
    inputs.unset(quotas);
    for (const char * value : { "", "0", "536870911", "-1", "536870912x", "18446744073709551616" }) {
        inputs.set(reserve, value); char error[512] = {};
        sink_ptr s = inputs.create(threads, depth, fixture::resolve, &f, error, sizeof(error));
        require(!s && error[0], "invalid cache reserve accepted"); ++rejected;
    }
    for (const char * value : { "2147483649", "2466374752", "12884901888", "18446744073709551615" }) {
        inputs.set(reserve, value);
        auto s = make_sink(inputs, f, threads, depth);
        require(bool(s), "large model-derived cache reserve rejected");
    }
    inputs.set(reserve, "1073741824");
    for (const char * value : { "1:1", "1:1,3:3", "1:1,2:5" }) {
        inputs.set(quotas, value); auto s = make_sink(inputs, f, threads, depth);
        const int32_t ids[] = { 1, 2 }; char error[512] = {};
        require(!kimi_cpu_sink_cache_configure(s.get(), 4 * unit, ids, 2, 4, error, sizeof(error)) && error[0],
                "partial, unknown or oversized layer quota accepted"); ++rejected;
    }
    unsigned comparisons = 0;
    for (const char * profile : { "2:3,1:1", "2:4,1:0" }) {
        inputs.set(quotas, profile); auto s = make_sink(inputs, f, threads, depth);
        configure_cache(s.get(), 4 * unit + 17, { 1, 2 }, 4);
        const bool zero = std::strstr(profile, "1:0") != nullptr;
        std::vector<float> input(f.latent, .17f);
        for (uint64_t token = 1; token <= 4; ++token) {
            begin_cache(s.get(), 1, token);
            cache_case(inputs, "hitrate_layer1", f, s.get(), 1, input, { 1 }, { 1 },
                    !zero && token > 1 ? 1 : 0, !zero && token == 1 ? 1 : 0);
            cache_case(inputs, "hitrate_layer2", f, s.get(), 2, input, { 1, 2, 3 }, { .2f, .3f, .5f },
                    token > 1 ? 3 : 0, token == 1 ? 3 : 0);
            input[0] += .013f; comparisons += 2;
        }
        const auto stats = cache_snapshot(s.get());
        require(stats.capacity_bytes == 4 * unit && stats.committed_bytes == (zero ? 3 : 4) * unit && !stats.pressure_stops,
                "fixed quotas exceeded physical budget or touched unused slots");
        begin_cache(s.get(), 0, 0, false); begin_cache(s.get(), 1, 1);
        cache_case(inputs, "hitrate_retained", f, s.get(), 2, input, { 3, 1 }, { .7f, .3f }, 2, 0); ++comparisons;
        begin_cache(s.get(), 2, 2);
        cache_case(inputs, "hitrate_epoch_cold", f, s.get(), 2, input, { 3, 1 }, { .3f, .7f }, 0, 2); ++comparisons;
    }
    inputs.set(quotas, "1:1,2:2");
    auto mismatch = make_sink(inputs, f, threads, depth); configure_cache(mismatch.get(), 4 * unit, { 1, 2 }, 4);
    begin_cache(mismatch.get(), 1, 1); tensors meta(f, 1);
    std::vector<float> input(f.latent, .13f), output(f.latent);
    const int32_t id = 1; const float weight = 1; char error[512] = {}; kimi_cpu_sink_stats stats = {};
    require(!kimi_cpu_sink_run(mismatch.get(), meta.role[0], meta.role[1], meta.role[2], input.data(), &id, &weight,
            1, 4, 25, output.data(), &stats, error, sizeof(error)), "quota sum mismatch did not fail closed");
    require(error[0] && !stats.read_bytes && !stats.pending_reads && !stats.fallback_count && !cache_snapshot(mismatch.get()).mapped_bytes,
            "invalid quota geometry submitted I/O or retained a mapping"); ++rejected;
    inputs.unset(quotas); inputs.unset(reserve);
    std::printf("CACHE_HITRATE_BIT_EXACT_OK comparisons=%u rejected=%u reserve_bytes=1073741824\n", comparisons, rejected);
}

static std::vector<int32_t> all_cache_layers() {
    std::vector<int32_t> layers(256);
    for (int i = 0; i < 256; ++i) { layers[i] = i + 1; }
    return layers;
}

#if defined(__linux__)
// Linux only: plans the automatic capacity against a dedicated cgroup scope's memory.max/current.
static void cache_auto_ring_test(fixture_inputs & inputs, int threads, int depth) {
    // A dedicated small scope tests planned capacity without filling it or
    // manufacturing an OOM. The fixture and its file pages stay charged.
    std::ifstream groups("/proc/self/cgroup"); std::string line, path;
    while (std::getline(groups, line)) {
        if (line.compare(0, 3, "0::") == 0) { path = "/sys/fs/cgroup" + line.substr(3); break; }
    }
    uint64_t limit = 0;
    std::ifstream maximum(path + "/memory.max");
    test_resources::need(bool(maximum >> limit) && limit == 1024ull * 1024 * 1024,
            "--auto-cache-ring-test requires an independent 1 GiB scope");
    require(depth == 8, "cold-ring regression uses eight staging slots");
    const uint64_t reserve = 512ull * 1024 * 1024;
    inputs.set("cache_reserve_bytes", "536870912");
    inputs.set("cache_allocation", "uniform");
    inputs.set("cache_policy", "count_decay2");
    inputs.set("cache_reclaimable_file", "0");
    inputs.set("cache_keep", "0");
    inputs.set("cache_partial_reclaim", "0");
    inputs.set("cache_elastic", "0");
    inputs.set("read_fixed", "0");
    inputs.set("ready_first", "0");
    inputs.unset("cache_layer_quotas");
    fixture f; f.latent = f.hidden = 2048; f.experts = 8; make_synthetic(f);
    const uint64_t unit = triplet_bytes(f);
    std::vector<float> input(f.latent, .013f), weights(8, .125f);
    const std::vector<int32_t> ids = {0, 1, 2, 3, 4, 5, 6, 7};
    const auto expected = reference(f, 1, input, ids, weights, 4, 25);
    const auto expected_second = reference(f, 2, input, ids, weights, 4, 25);
    // More logical expert IDs reuse the same eight real fixture ranges. This
    // fills the automatic arena, without duplicating the fixture or reference.
    for (int layer : {1, 2}) {
        for (int id = 8; id < 128; ++id) {
            for (int role = 0; role < 3; ++role) {
                const auto & base = f.entries.at({tensor_name(layer, role), id % 8});
                entry alias;
                alias.offset = base.offset; alias.raw_bytes = base.raw_bytes; alias.read_bytes = base.read_bytes;
                f.entries.emplace(std::make_pair(tensor_name(layer, role), id), std::move(alias));
            }
        }
    }
    f.experts = 128;
    auto sink = make_sink(inputs, f, threads, depth);
    configure_cache(sink.get(), std::numeric_limits<uint64_t>::max(), {1, 2}, 256);
    tensors meta(f, 1);
    const auto execute = [&](const char * name, size_t hits, int admitted) {
        auto value = run_checked(inputs, name, f, sink.get(), meta, input, ids, weights, expected, 1, 25, hits, admitted);
        cache_snapshot(sink.get()); print_sample(name, value); return value;
    };
    begin_cache(sink.get(), 71, 1);
    const auto cold = execute("auto_cache_cold_ring", 0, 8);
    const auto stats = cache_snapshot(sink.get());
    require(stats.capacity_bytes > 0 && !stats.clears && !stats.pressure_stops,
            "cold-ring setup paused the cache");
    begin_cache(sink.get(), 71, 2);
    const auto hit = execute("auto_cache_warm_ring", 8, 0);
    require(cold.output_hash == hit.output_hash, "automatic cache changed an output bit");
    // Drop optional entries, then exercise every mandatory ring slot without
    // cache admissions. Evaluate the saved plan against this non-cache usage;
    // never commit an oversized plan merely to provoke a kernel OOM.
    begin_cache(sink.get(), 71, 3, false);
    const auto direct = execute("auto_cache_ring_residency", 0, 0);
    require(!cache_snapshot(sink.get()).committed_bytes && direct.stats.ring_read_bytes == depth * unit,
            "ring-residency measurement retained optional payload or skipped a ring slot");
    require(direct.output_hash == cold.output_hash, "ring/cache transport changed an output bit");
    uint64_t current = 0;
    std::ifstream usage(path + "/memory.current");
    require(bool(usage >> current), "read planned-cache usage");
    const uint64_t planned = current + stats.capacity_bytes + reserve;
    std::printf("AUTO_CACHE_RING_PLAN noncache_current=%llu prior_committed=%llu capacity=%llu ring=%llu planned=%llu limit=%llu\n",
            (unsigned long long) current, (unsigned long long) stats.committed_bytes,
            (unsigned long long) stats.capacity_bytes, (unsigned long long) cold.stats.ring_bytes,
            (unsigned long long) planned, (unsigned long long) limit);
    require(cold.stats.ring_bytes == depth * unit, "cold-ring staging geometry");
    // Allow small post-sizing metadata/workspaces, well below the eight-slot
    // ring omitted by the old auto budget. No reclaimable-file estimate used.
    require(planned <= limit + 2 * unit,
            "automatic cache capacity omitted mandatory cold-ring residency");
    const auto before_fill = cache_snapshot(sink.get());
    for (int group = 0; group < 16; ++group) {
        std::vector<int32_t> routed(8);
        for (int j = 0; j < 8; ++j) { routed[j] = group * 8 + j; }
        begin_cache(sink.get(), 72, group + 1);
        for (int layer : {1, 2}) {
            tensors model(f, layer);
            kimi_cpu_sink_stats detail = {}; char error[512] = {}; std::vector<float> output(f.latent);
            require(kimi_cpu_sink_run(sink.get(), model.role[0], model.role[1], model.role[2],
                    input.data(), routed.data(), weights.data(), 8, 4, 25, output.data(), &detail, error, sizeof(error)),
                    std::string("fill automatic cache: ") + error);
            const auto & target = layer == 1 ? expected : expected_second;
            require(std::memcmp(output.data(), target.data(), output.size() * sizeof(float)) == 0,
                    "full automatic cache changed an output bit");
            const auto filled = cache_snapshot(sink.get());
            require(filled.clears == before_fill.clears && !filled.pressure_stops,
                    "automatic cache exhausted headroom while filling its own capacity");
        }
    }
    const auto full = cache_snapshot(sink.get());
    require(full.committed_bytes == full.capacity_bytes && full.valid_bytes == full.capacity_bytes,
            "automatic capacity was not completely exercised");
    std::printf("AUTO_CACHE_FILL_OK capacity=%llu page_tables=%llu pressure_stops=%llu\n",
            (unsigned long long) full.capacity_bytes, (unsigned long long) full.page_table_bytes,
            (unsigned long long) full.pressure_stops);
    std::printf("AUTO_CACHE_RING_OK output_hash=%llu\n", (unsigned long long) hit.output_hash);
}
#endif

static void cache_large_test(fixture_inputs & inputs, int threads, int depth) {
    // Only three experts are materialized. The metadata exposes ID 895 and
    // 256 * 896 cache slots, making a > 120 GB virtual arena with < 3 MB payload
    // touched. In particular global allocation must not charge the whole arena.
    fixture f; f.latent = 512; f.hidden = 1024; f.experts = 3; make_synthetic(f);
    for (int layer : { 1, 2 }) {
        for (int role = 0; role < 3; ++role) {
            f.entries[{ tensor_name(layer, role), 895 }] = f.entries.at({ tensor_name(layer, role), 0 });
        }
    }
    f.experts = 896;
    auto sink = make_sink(inputs, f, threads, depth);
    const auto layers = all_cache_layers();
    char error[512] = {};
    const int32_t invalid_layer = 1024;
    require(!kimi_cpu_sink_cache_configure(sink.get(), UINT64_MAX, layers.data(), 257, 896, error, sizeof(error)) && error[0],
            "cache accepted out-of-bounds layer count");
    require(!kimi_cpu_sink_cache_configure(sink.get(), UINT64_MAX, &invalid_layer, 1, 896, error, sizeof(error)) && error[0],
            "cache accepted out-of-bounds layer ID");
    require(!kimi_cpu_sink_cache_configure(sink.get(), UINT64_MAX, nullptr, 1, 896, error, sizeof(error)) && error[0],
            "cache accepted null nonempty layer list");
    for (uint64_t budget : { uint64_t(100000000001), uint64_t(594000000000), UINT64_MAX }) {
        if (budget <= std::numeric_limits<size_t>::max()) { configure_cache(sink.get(), budget, layers, 896); }
    }
    configure_cache(sink.get(), 594000000000, layers, 896);
    const uint64_t unit = triplet_bytes(f), before = anonymous_bytes();
    std::vector<float> input(f.latent, .13f);
    begin_cache(sink.get(), 1, 1);
    cache_case(inputs, "cache_large_sparse_first_touch", f, sink.get(), 1, input, { 1, 2 }, { .3f, .7f }, 0, 2);
    auto stats = cache_snapshot(sink.get());
    require(stats.capacity_bytes == unit * 256 * 896 && stats.mapped_bytes == stats.capacity_bytes &&
            stats.capacity_bytes > 120000000000ull && stats.committed_bytes == 2 * unit,
            "large quota was clipped or virtual budget was eagerly committed");
    require(stats.page_table_bytes > 0 && stats.last_guard_extra_bytes > 2 * unit &&
            stats.last_guard_extra_bytes < 3 * unit && stats.peak_guard_extra_bytes < 128ull * 1024 * 1024,
            "large cache guard omitted page tables or charged unused slots: page_tables=" +
            std::to_string(stats.page_table_bytes) + " last_guard=" + std::to_string(stats.last_guard_extra_bytes) +
            " peak_guard=" + std::to_string(stats.peak_guard_extra_bytes) + " unit=" + std::to_string(unit));
    require(anonymous_bytes() < before + 128ull * 1024 * 1024, "large virtual cache prefaulted payload");
    begin_cache(sink.get(), 1, 2);
    cache_case(inputs, "cache_large_hit_and_high_id", f, sink.get(), 1, input, { 1, 895 }, { .4f, .6f }, 1, 1);
    begin_cache(sink.get(), 1, 3);
    cache_case(inputs, "cache_large_all_hits_no_new_pages", f, sink.get(), 1, input, { 895, 1 }, { .6f, .4f }, 2, 0);
    stats = cache_snapshot(sink.get());
    require(stats.committed_bytes == 3 * unit && stats.last_guard_extra_bytes == 0 && stats.pressure_stops == 0,
            "large cache hit reused or charged an untouched quota");
    begin_cache(sink.get(), 1, 4, false);
    require(cache_snapshot(sink.get()).mapped_bytes == 0, "large virtual mapping survived prefill reset");
    std::printf("CACHE_LARGE_OK capacity_bytes=%llu committed_bytes=%llu page_table_upper_bytes=%llu peak_guard_extra_bytes=%llu\n",
            (unsigned long long) stats.capacity_bytes, (unsigned long long) stats.committed_bytes,
            (unsigned long long) stats.page_table_bytes, (unsigned long long) stats.peak_guard_extra_bytes);
}

#if defined(__linux__)
// Linux only: these two lower and read cgroup v2 memory.max to create pressure.
static void cache_pressure_transition_test(fixture_inputs & inputs, int threads, int depth) {
    // Explicit opt-in, dedicated scope only. Lowering its limit leaves the
    // selected reserve minus a small, known deficit, so this exercises the
    // cache guard without OOM or pressure ballast. The same test supports an
    // unlimited child below this finite ancestor to verify ancestor accounting.
    const char * owned = std::getenv("GGML_KIMI_CPU_SINK_TEST_CGROUP");
    test_resources::need(owned && std::string(owned).find("/sys/fs/cgroup/") == 0 &&
            std::string(owned).find("/q3ks-cache-policy-large-") != std::string::npos,
            "pressure transition requires an explicitly owned q3ks-cache-policy-large-* cgroup");
    std::ifstream groups("/proc/self/cgroup"); std::string line, actual;
    while (std::getline(groups, line)) {
        if (line.compare(0, 3, "0::") == 0) { actual = "/sys/fs/cgroup" + line.substr(3); break; }
    }
    const std::string path(owned);
    require(actual == path || actual.compare(0, path.size() + 1, path + "/") == 0,
            "pressure scope must contain this process");
    std::ifstream max_file(path + "/memory.max"); std::string original;
    test_resources::need(bool(max_file >> original) && original != "max" && std::stoull(original) >= 8ull * 1024 * 1024 * 1024,
            "pressure transition requires an initially large finite limit");
    auto write_limit = [&](const std::string & value) {
        std::ofstream file(path + "/memory.max"); file << value << '\n'; file.flush();
        require(bool(file), "cannot write owned pressure-test memory.max");
    };
    const char * partial_value = inputs.get("cache_partial_reclaim");
    const bool partial = partial_value && !std::strcmp(partial_value, "1");
    fixture f; make_synthetic(f); auto sink = make_sink(inputs, f, threads, depth);
    const uint64_t unit = cache_slot_bytes(inputs, triplet_bytes(f));
    if (partial) { configure_cache(sink.get(), 4 * unit, { 1, 2 }, 2); }
    else { configure_cache(sink.get(), 594000000000, all_cache_layers(), 896); }
    std::vector<float> input(f.latent, .13f);
    begin_cache(sink.get(), 1, 1);
    if (partial) {
        cache_case(inputs, "cache_partial_pressure_warm_layer1", f, sink.get(), 1, input, { 1, 2 }, { .6f, .4f }, 0, 2);
        cache_case(inputs, "cache_partial_pressure_warm_layer2", f, sink.get(), 2, input, { 3, 4 }, { .6f, .4f }, 0, 2);
        for (uint64_t token = 2; token <= 5; ++token) {
            begin_cache(sink.get(), 1, token);
            cache_case(inputs, "cache_partial_pressure_train_hot_layer1", f, sink.get(), 1, input, { 1 }, { 1 }, 1, 0);
            cache_case(inputs, "cache_partial_pressure_train_hot_layer2", f, sink.get(), 2, input, { 3 }, { 1 }, 1, 0);
        }
    } else {
        cache_case(inputs, "cache_pressure_transition_warm", f, sink.get(), 1, input, { 1 }, { 1 }, 0, 1);
    }
    uint64_t used = 0;
    std::ifstream current(path + "/memory.current"); require(bool(current >> used), "cannot read pressure-test usage");
    const char * chosen = inputs.get("cache_reserve_bytes");
    const uint64_t reserve_bytes = chosen ? std::stoull(chosen) : 2ull * 1024 * 1024 * 1024;
    // One byte beyond a single slot makes the guard request two slots while
    // leaving almost one slot of slack after the per-layer release. That keeps
    // ordinary cgroup accounting jitter from turning the mechanism test into
    // an admission-rejection test.
    const uint64_t pressure = partial ? unit + 1 : 32 * 1024 * 1024;
    require(used <= UINT64_MAX - reserve_bytes && used + reserve_bytes > pressure,
            "pressure transition limit arithmetic overflow");
    const uint64_t reduced = used + reserve_bytes - pressure;
    require(reduced < std::stoull(original), "pressure transition would not lower limit");
    write_limit(std::to_string(reduced));
    try {
        begin_cache(sink.get(), 1, partial ? 6 : 2);
        if (partial) {
            const auto layer1 = cache_case(inputs, "cache_partial_pressure_retains_hot_layer1", f, sink.get(), 1,
                    input, { 1 }, { 1 }, 1, 0);
            const auto layer2 = cache_case(inputs, "cache_partial_pressure_retains_hot_layer2", f, sink.get(), 2,
                    input, { 3 }, { 1 }, 1, 0);
            const auto stats = cache_snapshot(sink.get());
            require(stats.pressure_stops == 0 && stats.mapped_bytes == 4 * unit &&
                    stats.enabled_bytes == 2 * unit &&
                    stats.committed_bytes == stats.enabled_bytes && stats.valid_bytes == stats.enabled_bytes &&
                    stats.guard_reclaims >= 1 && stats.released_bytes == 2 * unit &&
                    stats.clears == 0 && layer1.stats.ring_read_bytes == 0 && layer2.stats.ring_read_bytes == 0,
                    "partial pressure guard did not release one cold slot per layer while retaining both hot hits");
        } else {
            const auto result = cache_case(inputs, "cache_pressure_transition_hit_becomes_ring_miss", f, sink.get(), 1,
                    input, { 1, 2 }, { .3f, .7f }, 0, 0);
            const auto stats = cache_snapshot(sink.get());
            require(stats.pressure_stops == 1 && stats.mapped_bytes == 0 && stats.page_table_bytes == 0 &&
                    result.stats.cache_new_slot_bytes == 0 && result.stats.ring_read_bytes == 2 * unit,
                    "pressure fallback retained a cache pointer, hit counter, or new-slot reservation");
            write_limit(original);
            begin_cache(sink.get(), 1, 3, false); begin_cache(sink.get(), 1, 4);
            cache_case(inputs, "cache_pressure_transition_recovery", f, sink.get(), 1, input, { 1 }, { 1 }, 0, 1);
        }
        write_limit(original);
    } catch (...) { write_limit(original); throw; }
    std::printf("CACHE_PRESSURE_TRANSITION_OK partial_reclaim=%d ancestor=%d original_limit=%s reduced_limit=%llu\n",
            partial, actual != path, original.c_str(), (unsigned long long) reduced);
}

static void cache_pressure_test(fixture_inputs & inputs, int threads, int depth, bool unbounded = false) {
    // Run explicitly in a small dedicated scope, e.g. MemoryMax=1G. Never try
    // to manufacture pressure by filling the production 16 GiB envelope.
    std::ifstream groups("/proc/self/cgroup"); std::string line, path;
    while (std::getline(groups, line)) {
        if (line.compare(0, 3, "0::") == 0) { path = "/sys/fs/cgroup" + line.substr(3); break; }
    }
    bool constrained = false, finite = false;
    while (path.size() >= std::strlen("/sys/fs/cgroup")) {
        std::ifstream file(path + "/memory.max"); std::string limit;
        if (file >> limit && limit != "max") {
            finite = true;
            if (std::stoull(limit) <= 2ull * 1024 * 1024 * 1024) { constrained = true; }
        }
        const auto slash = path.find_last_of('/');
        if (slash == std::string::npos) { break; }
        path.resize(slash);
    }
    test_resources::need(unbounded ? !finite : constrained, unbounded ? "--cache-unbounded requires no finite cgroup ancestor" :
            "--cache-pressure requires a cgroup memory.max <= 2 GiB (use an independent 1 GiB scope)");
    fixture f; make_synthetic(f); auto sink = make_sink(inputs, f, threads, depth);
    std::vector<float> input(f.latent, .13f);
    configure_cache(sink.get(), 2 * triplet_bytes(f), { 1 }, 2);
    begin_cache(sink.get(), 1, 1);
    cache_case(inputs, "cache_pressure_bypass_correct", f, sink.get(), 1, input, { 1, 2 }, { .3f, .7f }, 0, 0);
    auto stats = cache_snapshot(sink.get());
    require(stats.pressure_stops == 1 && stats.mapped_bytes == 0 && stats.committed_bytes == 0 && stats.valid_bytes == 0,
            "2 GiB reserve did not pause before allocating payload");
    begin_cache(sink.get(), 1, 2);
    cache_case(inputs, "cache_pressure_stays_paused", f, sink.get(), 1, input, { 1, 2 }, { .3f, .7f }, 0, 0);
    require(cache_snapshot(sink.get()).pressure_stops == 1, "paused cache retried admission every token");
    begin_cache(sink.get(), 1, 3, false);
    begin_cache(sink.get(), 1, 4);
    cache_case(inputs, "cache_pressure_prefill_resets_pause", f, sink.get(), 1, input, { 1, 2 }, { .3f, .7f }, 0, 0);
    stats = cache_snapshot(sink.get());
    require(stats.pressure_stops == 2 && stats.mapped_bytes == 0, "prefill did not allow a fresh headroom check");
    configure_cache(sink.get(), 594000000000, all_cache_layers(), 896);
    begin_cache(sink.get(), 2, 1);
    const auto before = anonymous_bytes();
    cache_case(inputs, "cache_pressure_large_metadata_bypass", f, sink.get(), 1, input, { 1, 2 }, { .3f, .7f }, 0, 0);
    require(cache_snapshot(sink.get()).pressure_stops == 1 && anonymous_bytes() < before + 16ull * 1024 * 1024,
            "small finite cgroup touched large slot metadata before headroom check");
    std::printf("CACHE_PRESSURE_OK pressure_stops=%llu mapped_bytes=%llu\n",
            (unsigned long long) stats.pressure_stops, (unsigned long long) stats.mapped_bytes);
    if (unbounded) { std::printf("CACHE_UNBOUNDED_REJECT_OK\n"); }
}
#endif

// Validate the emitted records, not just the counters used to format them.
// No cgroup/cache admission is required; cache-hit transport has separate tests.
// The sink's stderr lines (trace, profile, reports) while it lives; read() restores stderr.
struct stderr_capture {
    std::string path = test_resources::temp_file("kimi-cpu-sink-stderr");
    FILE * file = std::fopen(path.c_str(), "w+b");
    int saved = -1;
    stderr_capture() {
        require(file != nullptr, "trace capture file");
        std::fflush(stderr);
        saved = test_resources::fd_dup(test_resources::fd_of(stderr));
        require(saved >= 0 && test_resources::fd_dup2(test_resources::fd_of(file), test_resources::fd_of(stderr)) >= 0,
                "redirect trace stderr");
    }
    void restore() {
        if (saved >= 0) {
            std::fflush(stderr);
            test_resources::fd_dup2(saved, test_resources::fd_of(stderr));
            test_resources::fd_close(saved);
            saved = -1;
        }
    }
    ~stderr_capture() { restore(); if (file) { std::fclose(file); } test_resources::remove_file(path); }
    std::string read() {
        restore(); std::rewind(file);
        std::string result; char buffer[4096];
        while (const size_t n = std::fread(buffer, 1, sizeof(buffer), file)) { result.append(buffer, n); }
        return result;
    }
};

static void trace_profile_test(fixture_inputs & inputs, int threads, int depth) {
    require(inputs.set("trace", "1") == 0 &&
            inputs.set("profile", "1") == 0, "set observation environment");
    fixture f; make_synthetic(f);
    auto sink = make_sink(inputs, f, threads, depth);
    std::vector<float> input(f.latent, .13f);
    const std::vector<int32_t> ids = { 1, -1, 1, 2 };
    const std::vector<float> weights = { .2f, 0, .3f, .5f };
    stderr_capture log;
    begin_cache(sink.get(), 777, 8);
    cache_case(inputs, "trace_profile_token8", f, sink.get(), 1, input, ids, weights, 0, 0);
    begin_cache(sink.get(), 777, 9);
    cache_case(inputs, "trace_profile_token9", f, sink.get(), 1, input, ids, weights, 0, 0);
    begin_cache(sink.get(), 777, 10, false);
    begin_cache(sink.get(), 778, 1);
    // Invalid input must still have an error record, without stale expert IDs.
    tensors meta(f, 1); char error[512] = {}; kimi_cpu_sink_stats stats = {};
    require(!kimi_cpu_sink_run(sink.get(), meta.role[0], meta.role[1], meta.role[2],
            nullptr, ids.data(), weights.data(), int(ids.size()), 4.0f, 0.0f,
            input.data(), &stats, error, sizeof(error)), "trace failure fixture accepted invalid input");
    std::istringstream lines(log.read());
    std::string line;
    std::vector<nlohmann::json> trace, profile;
    unsigned resets = 0;
    while (std::getline(lines, line)) {
        const std::string tp = "[kimi_cpu_sink_trace] ", pp = "[kimi_cpu_sink_profile] ";
        if (line.compare(0, tp.size(), tp) == 0) {
            auto record = nlohmann::json::parse(line.substr(tp.size()));
            if (record.at("event") == "reset") {
                require(record.at("epoch") == 777 && record.at("token") == 10, "wrong trace reset identity");
                ++resets;
            } else { trace.push_back(std::move(record)); }
        } else if (line.compare(0, pp.size(), pp) == 0) {
            profile.push_back(nlohmann::json::parse(line.substr(pp.size())));
        }
    }
    require(trace.size() == 3 && profile.size() == 3 && resets == 1, "trace/profile record count");
    for (size_t i = 0; i < trace.size(); ++i) {
        const auto & t = trace[i]; const auto & p = profile[i];
        require(t.at("schema") == 1 && p.at("schema") == 1 && t.at("layer") == 1 &&
                t.at("phase") == "decode" && t.at("has_token") == true &&
                t.at("epoch") == (i == 2 ? 778 : 777) && t.at("token") == (i == 2 ? 1 : 8 + i),
                "trace token/layer identity missing or incorrect");
        for (const char * key : { "epoch", "token", "layer", "call", "success" }) {
            require(t.at(key) == p.at(key), "trace/profile join mismatch");
        }
        require(t.at("success") == (i < 2), "trace success flag mismatch");
        if (i == 2) { require(t.at("selected_experts").empty(), "failed call reused stale requests"); continue; }
        require(t.at("selected_experts") == nlohmann::json::array({ 1, 1, 2 }) &&
                t.at("cache_hit") == nlohmann::json::array({ false, false, false }),
                "trace lost order, duplicate expert or padding semantics");
        const uint64_t unit = triplet_bytes(f);
        require(t.at("expert_bytes") == nlohmann::json::array({ unit, unit, unit }) &&
                t.at("expert_read_bytes") == t.at("expert_bytes") && t.at("read_bytes") == 3 * unit &&
                t.at("demand_bytes") == 3 * unit && t.at("hits") == 0 && t.at("misses") == 3,
                "trace byte/hit accounting mismatch");
        const double total = p.at("total_us"), compute = p.at("compute_wall_us"), wait = p.at("wait_io_us");
        const double other = p.at("other_wall_us");
        require(compute > 0 && wait >= 0 && other >= 0 && std::abs(total - compute - wait - other) < .01,
                "profile disjoint caller wall times do not sum to total");
        auto clocked = [](double us) { return test_resources::thread_clock_ticks ? us >= 0 : us > 0; };
        require(clocked(p.at("compute_thread_cpu_us").get<double>()) &&
                clocked(p.at("caller_thread_cpu_us").get<double>()) && clocked(p.at("io_thread_cpu_us").get<double>()),
                "profile CPU clocks missing");
        double worker_cpu = 0;
        for (const auto & w : p.at("workers")) {
            require(w.at("phases") == 6, "profile worker phase count");
            worker_cpu += w.at("up_gate_cpu_us").get<double>() + w.at("down_cpu_us").get<double>();
        }
        require(p.at("compute_thread_cpu_us").get<double>() + .01 >= worker_cpu,
                "profile aggregate CPU omitted worker work");
        require(p.at("io_wait_wall_us").get<double>() + p.at("io_credit_wait_wall_us").get<double>() <=
                p.at("io_job_wall_us").get<double>() + .01, "I/O component waits exceed I/O job span");
    }
    std::printf("CPU_SINK_TRACE_PROFILE_OK records=%zu resets=%u\n", trace.size(), resets);
}

// The next-layer prefetch's statistics per rank of the hint, at depth 2 and at depth 0. Token 1 makes six slots of
// each layer resident; at token 2 layer 1's run gets a hint for layer 2 (5, 6, 7, 12: 12 is cached), and layer 2
// selects 5, 7 and 12. At depth 2, 5 and 6 are read and 5 is a hit; at depth 0 nothing is read, and the counts of
// novel ranks and of novel ones selected are the same. One idle window: from layer 1's schedule (it reads nothing)
// to layer 2's start. The decode trace gives each run's hint.
static void prefetch_tests(fixture_inputs & inputs, int threads, int depth) {
    // the served scheduler (the launcher's profile), which the prefetch requires
    for (const auto & kv : { std::pair<const char *, const char *>{ "ready_first", "1" },
            { "early_up_gate", "1" },
            { "role_major", "1" }, { "nonblocking_down_sched", "1" },
            { "cache_policy", "lru" },
            { "cache_allocation", "uniform" }, { "trace", "1" } }) {
        require(inputs.set(kv.first, kv.second) == 0, "set prefetch test environment");
    }
    fixture f; make_synthetic(f);
    const uint64_t unit = triplet_bytes(f);
    std::vector<float> input(f.latent);
    for (size_t i = 0; i < input.size(); ++i) { input[i] = float(int(i % 19) - 9) * .021f; }
    // a depth up to the hint's width starts (a model's top-k may exceed 8), a deeper one is refused at start
    for (const int k : { 10, 65 }) {
        require(inputs.set("prefetch_depth", std::to_string(k).c_str()) == 0, "set prefetch depth");
        char error[512] = {};
        sink_ptr sink = inputs.create(threads, depth, fixture::resolve, &f, error, sizeof(error));
        require(k == 10 ? bool(sink) : !sink && std::strstr(error, "must be 0..64"),
                "prefetch depth " + std::to_string(k) + ": " + error);
    }
    for (const int k : { 2, 0 }) {
        require(inputs.set("prefetch_depth", std::to_string(k).c_str()) == 0, "set prefetch depth");
        stderr_capture log;
        {
            auto sink = make_sink(inputs, f, threads, depth);
            configure_cache(sink.get(), 12 * unit, { 1, 2 }, 6);
            begin_cache(sink.get(), 1, 1);
            cache_case(inputs, "prefetch_warm_layer1", f, sink.get(), 1, input, { 1, 2 }, { .4f, .6f }, 0, 2);
            cache_case(inputs, "prefetch_warm_layer2", f, sink.get(), 2, input, { 10, 11, 12, 13, 14, 15 },
                { .1f, .1f, .2f, .2f, .2f, .2f }, 0, 6);
            begin_cache(sink.get(), 1, 2);
            const int32_t hint[] = { 5, 6, 7, 12 };
            kimi_cpu_sink_prefetch_hint(sink.get(), 2, hint, 4);
            cache_case(inputs, "prefetch_hinted_layer1", f, sink.get(), 1, input, { 1, 2 }, { .4f, .6f }, 2, 0);
            cache_case(inputs, "prefetch_predicted_layer2", f, sink.get(), 2, input, { 5, 7, 12 }, { .2f, .3f, .5f },
                k ? 2 : 1, -1);
        }
        std::istringstream lines(log.read());
        std::string line, report;
        std::vector<nlohmann::json> hints;
        while (std::getline(lines, line)) {
            if (line.rfind("[kimi_cpu_sink_prefetch] reason=destroy ", 0) == 0) { report = line; }
            const std::string tp = "[kimi_cpu_sink_trace] ";
            if (line.rfind(tp, 0) == 0) {
                const auto record = nlohmann::json::parse(line.substr(tp.size()));
                if (record.at("event") == "layer") { hints.push_back(record.at("hint")); }
            }
        }
        require(!report.empty(), "no prefetch report at destroy");
        require(hints == std::vector<nlohmann::json>{ nlohmann::json::array(), nlohmann::json::array(),
                nlohmann::json::array({ 5, 6, 7, 12 }), nlohmann::json::array() }, "trace hints per run");
        auto field = [&](const std::string & key) {
            const size_t at = report.find(" " + key + "=");
            require(at != std::string::npos, "prefetch report lacks " + key);
            const size_t begin = at + key.size() + 2;
            return report.substr(begin, report.find(' ', begin) - begin);
        };
        const bool read = k > 0;
        require(field("topk") == std::to_string(k) && field("layers") == "1" && field("width") == "4" &&
                field("predicted") == (read ? "2" : "0") && field("issued") == (read ? "2" : "0") &&
                field("hits") == (read ? "1" : "0"), "prefetch report totals: " + report);
        const uint64_t reader_bytes = std::stoull(field("reader_bytes"));
        require(field("windows") == "1" && std::stod(field("window_us_p50")) >= 0 &&
                std::stod(field("window_us_mean")) >= 0 &&
                (read ? reader_bytes >= unit && reader_bytes <= 2 * unit : reader_bytes == 0) &&
                (read ? std::stod(field("reader_busy_us")) > 0 : field("reader_busy_us") == "0"),
                "prefetch report window or readers: " + report);
        require(field("predicted_r") == "1,1,1,1" && field("novel_r") == "1,1,1,0" &&
                field("novel_routed_r") == "1,0,1,0" && field("issued_r") == (read ? "1,1,0,0" : "0,0,0,0") &&
                field("used_r") == (read ? "1,0,0,0" : "0,0,0,0"), "prefetch report per rank: " + report);
        // the reads of 6 (not selected, three roles) may still run when layer 2's run closes the window: then they
        // spill, and only they; the reads of 5 are that run's own
        const uint64_t spills = std::stoull(field("spills"));
        const std::string spill_r = field("spill_us_r");
        require(field("started_r") == (read ? "1,1,0,0" : "0,0,0,0") && spills <= 3 && spill_r.rfind("0,", 0) == 0 &&
                spill_r.substr(spill_r.find(',', 2)) == ",0,0" &&
                (spills || (field("spill_us") == "0" && spill_r == "0,0,0,0")), "prefetch report spill: " + report);
        std::printf("case=prefetch_depth%d status=PASS %s\n", k, report.c_str());
    }
#if defined(__linux__)
    {
        // The reads of 6, hinted for layer 2 and not selected there, held in pread (pread_hold) for as long as a case
        // needs, without timing assumptions. A prefetch lands only in resident slots, so both layers are warmed
        // first; the hinted run of layer 1 then has no reads of its own. A case's runs end with layer 2's, which
        // closes the window on the held reads; `after` follows with the hold. The prefetch report at destroy.
        require(inputs.set("prefetch_depth", "2") == 0, "set prefetch depth");
        std::vector<uint64_t> offsets;
        for (int role = 0; role < 3; ++role) {
            kimi_cpu_sink_source source = {};
            require(fixture::resolve(&f, tensor_name(2, role).c_str(), 6, &source), "resolve the held reads");
            offsets.push_back(source.offset);
        }
        auto spill_case = [&](const char * name, const std::vector<int32_t> & ids, size_t hits, auto && after) {
            stderr_capture log;
            {
                auto sink = make_sink(inputs, f, threads, depth);
                pread_hold_guard hold(f.fd, offsets); // released before the sink's end waits for the reads
                configure_cache(sink.get(), 12 * unit, { 1, 2 }, 6);
                begin_cache(sink.get(), 1, 1);
                cache_case(inputs, "spill_warm_layer1", f, sink.get(), 1, input, { 1, 2 }, { .4f, .6f }, 0, 2);
                cache_case(inputs, "spill_warm_layer2", f, sink.get(), 2, input, { 10, 11, 12, 13, 14, 15 },
                    { .1f, .1f, .2f, .2f, .2f, .2f }, 0, 6);
                begin_cache(sink.get(), 1, 2);
                const int32_t hint[] = { 5, 6 };
                kimi_cpu_sink_prefetch_hint(sink.get(), 2, hint, 2);
                cache_case(inputs, "spill_hinted_layer1", f, sink.get(), 1, input, { 1, 2 }, { .4f, .6f }, 2, 0);
                hold.wait(3); // the readers run 6's three reads, so 5's are done
                std::vector<float> weights(ids.size(), 1.0f / float(ids.size()));
                cache_case(inputs, name, f, sink.get(), 2, input, ids, weights, hits, -1);
                after(sink.get(), hold);
            }
            std::istringstream lines(log.read());
            std::string line, report;
            while (std::getline(lines, line)) {
                if (line.rfind("[kimi_cpu_sink_prefetch] reason=destroy ", 0) == 0) { report = line; }
            }
            require(!report.empty(), std::string(name) + ": no prefetch report at destroy");
            return report;
        };
        auto field = [](const std::string & report, const std::string & key) {
            const size_t at = report.find(" " + key + "=");
            require(at != std::string::npos, "prefetch report lacks " + key);
            const size_t begin = at + key.size() + 2;
            return report.substr(begin, report.find(' ', begin) - begin);
        };
        // The reads spill when layer 2's run closes the window and stop when its own read of 7 ends (the I/O thread
        // opens the next window). The next token's begin drains every read while they still run (held 300 ms more):
        // they spill again. Each counts once.
        auto drain_held = [&](kimi_cpu_sink * sink, pread_hold_guard & hold) {
            std::thread release([&] { std::this_thread::sleep_for(std::chrono::milliseconds(300)); hold.release(); });
            try { begin_cache(sink, 1, 3); } catch (...) { release.join(); throw; }
            release.join();
        };
        const std::string twice = spill_case("spill_layer2_reads", { 5, 7 }, 1, drain_held);
        const std::string spill_r = field(twice, "spill_us_r");
        require(field(twice, "issued_r") == "1,1" && field(twice, "started_r") == "1,1" &&
                field(twice, "spills") == "3" && spill_r.rfind("0,", 0) == 0 && spill_r != "0,0",
                "a read that spilled twice: " + twice);
        std::printf("case=prefetch_spill_twice status=PASS %s\n", twice.c_str());
        // Layer 2's run takes only 5 (prefetched): no reads of its own and no hint for the next layer, as the last
        // MoE layer's run. It opens the window when it schedules, so the 300 ms the reads are held after it are not
        // spill.
        auto hold_then_release = [](kimi_cpu_sink *, pread_hold_guard & hold) {
            std::this_thread::sleep_for(std::chrono::milliseconds(300));
            hold.release();
        };
        const std::string last = spill_case("spill_layer2_last", { 5 }, 1, hold_then_release);
        require(field(last, "spills") == "3" && std::stod(field(last, "spill_us")) < 150000 &&
                std::stoull(field(last, "spill_us_r").substr(2)) < 3 * 150000,
                "reads spilled past a run without reads of its own and without a hint: " + last);
        std::printf("case=prefetch_spill_last status=PASS %s\n", last.c_str());
    }
#else
    std::printf("case=prefetch_spill_twice status=SKIP reason=linux_pread_interposition\n");
    std::printf("case=prefetch_spill_last status=SKIP reason=linux_pread_interposition\n");
#endif
    std::printf("CPU_SINK_PREFETCH_OK\n");
}
// build_moe_ffn's one-token CPU graph: MUL_MAT_ID gate/up over the full expert
// tensors, the activation op(s), MUL_MAT_ID down, MUL by the routing weights and
// ADD of per-slot views in slot order. It shares no code with the sink.
static std::vector<float> reference_moe_graph(const fixture & f, int layer, const std::vector<float> & input,
        const std::vector<int32_t> & ids, const std::vector<float> & weights,
        const kimi_cpu_sink_activation & activation, int threads) {
    const int nslots = int(ids.size());
    require(nslots > 0 && weights.size() == ids.size(), "MoE reference requires live slots");
    const size_t max_nodes = 256;
    ggml_context * ctx = ggml_init({ ggml_tensor_overhead() * max_nodes + ggml_graph_overhead_custom(max_nodes, false), nullptr, true });
    require(ctx, "MoE reference context failed");
    ggml_backend_t backend = ggml_backend_cpu_init(); require(backend, "MoE reference CPU backend failed");
    ggml_backend_cpu_set_n_threads(backend, threads);
    ggml_tensor * experts[3];
    for (int role = 0; role < 3; ++role) {
        experts[role] = ggml_new_tensor_3d(ctx, role == 2 ? f.down_type : f.type, role == 2 ? f.hidden : f.latent,
                                          role == 2 ? f.latent : f.hidden, f.experts);
    }
    ggml_tensor * x = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, f.latent, 1, 1);
    ggml_tensor * selected = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, nslots, 1);
    ggml_tensor * routing = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, nslots, 1);
    ggml_tensor * gate = ggml_mul_mat_id(ctx, experts[1], x, selected);
    ggml_tensor * up = ggml_mul_mat_id(ctx, experts[0], x, selected);
    ggml_tensor * act = nullptr;
    if (activation.kind == KIMI_CPU_SINK_ACTIVATION_SWIGLU_CLAMP) {
        act = ggml_swiglu_clamp(ctx, gate, up, activation.swiglu_limit);
    } else if (activation.kind == KIMI_CPU_SINK_ACTIVATION_SWIGLU) {
        act = ggml_swiglu_split(ctx, gate, up);
    } else {
        const float beta = activation.situ_beta, lb = activation.situ_linear_beta;
        act = ggml_scale(ctx, ggml_tanh(ctx, ggml_scale(ctx, gate, 1.0f / beta)), beta);
        act = ggml_mul(ctx, act, ggml_sigmoid(ctx, gate));
        if (lb > 0.0f) { up = ggml_scale(ctx, ggml_tanh(ctx, ggml_scale(ctx, up, 1.0f / lb)), lb); }
        act = ggml_mul(ctx, act, up);
    }
    ggml_tensor * down = ggml_mul_mat_id(ctx, experts[2], act, selected);
    ggml_tensor * norm_weight = nullptr;
    if (activation.post_exp_norm) {
        norm_weight = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, f.latent);
        down = ggml_rms_norm(ctx, ggml_mul(ctx, down, norm_weight), activation.post_exp_norm_eps);
    }
    ggml_tensor * weighted = ggml_mul(ctx, down, routing);
    ggml_cgraph * graph = ggml_new_graph_custom(ctx, max_nodes, false);
    ggml_build_forward_expand(graph, weighted);
    ggml_tensor * result = nullptr;
    for (int slot = 0; slot < nslots; ++slot) {
        ggml_tensor * view = ggml_view_2d(ctx, weighted, f.latent, 1, weighted->nb[2], size_t(slot) * weighted->nb[1]);
        ggml_build_forward_expand(graph, view);
        result = result ? ggml_add(ctx, result, view) : view;
    }
    if (nslots == 1) { result = ggml_cont(ctx, result); }
    ggml_build_forward_expand(graph, result);
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend); require(buffer, "MoE reference allocation failed");
    for (int role = 0; role < 3; ++role) {
        for (int e = 0; e < f.experts; ++e) {
            const auto & item = f.entries.at({ tensor_name(layer, role), e });
            require(item.raw_bytes == experts[role]->nb[2], "MoE reference expert stride mismatch");
            ggml_backend_tensor_set(experts[role], item.data.data(), size_t(e) * experts[role]->nb[2], item.raw_bytes);
        }
    }
    if (norm_weight) {
        ggml_backend_tensor_set(norm_weight, activation.post_exp_norm, 0, size_t(f.latent) * sizeof(float));
    }
    ggml_backend_tensor_set(x, input.data(), 0, input.size() * sizeof(float));
    ggml_backend_tensor_set(selected, ids.data(), 0, ids.size() * sizeof(int32_t));
    ggml_backend_tensor_set(routing, weights.data(), 0, weights.size() * sizeof(float));
    require(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS, "MoE reference CPU graph failed");
    std::vector<float> output(f.latent);
    ggml_backend_tensor_get(result, output.data(), 0, output.size() * sizeof(float));
    ggml_backend_buffer_free(buffer); ggml_backend_free(backend); ggml_free(ctx);
    return output;
}



static void elastic_cache_tests(fixture_inputs & inputs, int threads, int depth) {
    scoped_fixture_input policy_env(inputs, "cache_policy", "count_decay2");
    scoped_fixture_input allocation_env(inputs, "cache_allocation", "uniform");
    scoped_fixture_input keep_env(inputs, "cache_keep", "1");
    scoped_fixture_input elastic_env(inputs, "cache_elastic", "1");
    scoped_fixture_input partial_env(inputs, "cache_partial_reclaim", nullptr);
    scoped_fixture_input prefill_env(inputs, "cache_prefill_slots", "2");
    scoped_fixture_input reserve_env(inputs, "cache_reserve_bytes", "536870912");
    scoped_fixture_input quotas_env(inputs, "cache_layer_quotas", nullptr);

    fixture f; make_synthetic(f);
    const uint64_t unit = cache_slot_bytes(inputs, triplet_bytes(f));
    std::vector<float> input(f.latent, .13f);
    auto sink = make_sink(inputs, f, threads, depth);
    configure_cache(sink.get(), 6 * unit, { 1, 2 }, 3);
    begin_cache(sink.get(), 71, 1);
    cache_case(inputs, "elastic_layer1_warm", f, sink.get(), 1, input, { 1, 2, 3 }, { .5f, .3f, .2f }, 0, 3);
    cache_case(inputs, "elastic_layer2_warm", f, sink.get(), 2, input, { 4, 5, 6 }, { .5f, .3f, .2f }, 0, 3);
    for (uint64_t token = 2; token <= 5; ++token) {
        begin_cache(sink.get(), 71, token);
        cache_case(inputs, "elastic_layer1_train_hot", f, sink.get(), 1, input, { 1 }, { 1 }, 1, 0);
        cache_case(inputs, "elastic_layer2_train_hot", f, sink.get(), 2, input, { 4 }, { 1 }, 1, 0);
    }
    const auto warm = cache_snapshot(sink.get());
    require(warm.mapped_bytes == 6 * unit && warm.enabled_bytes == 6 * unit &&
            warm.committed_bytes == 6 * unit && warm.valid_bytes == 6 * unit,
            "elastic warm state did not fill the decode maximum");
    begin_cache(sink.get(), 71, 0, false);
    const auto shrunk = cache_snapshot(sink.get());
    require(shrunk.mapped_bytes == 6 * unit && shrunk.enabled_bytes == 2 * unit &&
            shrunk.committed_bytes == 2 * unit && shrunk.valid_bytes == 2 * unit &&
            shrunk.shrink_ops == 1 && shrunk.released_bytes == 4 * unit &&
            shrunk.page_table_bytes == warm.page_table_bytes && shrunk.clears == warm.clears,
            "elastic prefill did not retain one scored resident per layer");
    begin_cache(sink.get(), 71, 1);
    auto grown = cache_snapshot(sink.get());
    require(grown.enabled_bytes == 6 * unit && grown.committed_bytes == 2 * unit &&
            grown.valid_bytes == 2 * unit && grown.phase_grows == 1,
            "elastic decode grow prefaulted or lost retained entries");
    cache_case(inputs, "elastic_layer1_hot_survives", f, sink.get(), 1, input, { 1 }, { 1 }, 1, 0);
    cache_case(inputs, "elastic_layer2_hot_survives", f, sink.get(), 2, input, { 4 }, { 1 }, 1, 0);
    begin_cache(sink.get(), 71, 2);
    cache_case(inputs, "elastic_layer1_refault", f, sink.get(), 1, input, { 2, 3 }, { .6f, .4f }, 0, 2);
    cache_case(inputs, "elastic_layer2_refault", f, sink.get(), 2, input, { 5, 6 }, { .6f, .4f }, 0, 2);
    const auto refilled = cache_snapshot(sink.get());
    require(refilled.committed_bytes == 6 * unit && refilled.valid_bytes == 6 * unit &&
            refilled.refaulted_bytes == 4 * unit && refilled.page_table_bytes == warm.page_table_bytes,
            "elastic decode did not account released-slot direct-read refaults");

    {
        scoped_fixture_input zero_prefill(inputs, "cache_prefill_slots", "0");
        auto zero = make_sink(inputs, f, threads, depth);
        configure_cache(zero.get(), 2 * unit, { 1, 2 }, 1);
        begin_cache(zero.get(), 72, 1);
        cache_case(inputs, "elastic_zero_layer1_warm", f, zero.get(), 1, input, { 1 }, { 1 }, 0, 1);
        cache_case(inputs, "elastic_zero_layer2_warm", f, zero.get(), 2, input, { 4 }, { 1 }, 0, 1);
        begin_cache(zero.get(), 72, 0, false);
        const auto empty = cache_snapshot(zero.get());
        require(empty.mapped_bytes == 2 * unit && !empty.enabled_bytes && !empty.committed_bytes &&
                !empty.valid_bytes && empty.released_bytes == 2 * unit,
                "elastic zero-prefill arm unmapped or retained resident payload");
        begin_cache(zero.get(), 72, 1);
        cache_case(inputs, "elastic_zero_decode_refault", f, zero.get(), 1, input, { 1 }, { 1 }, 0, 1);
        require(cache_snapshot(zero.get()).refaulted_bytes == unit,
                "elastic zero-prefill arm did not refault through direct admission");
    }

    auto rejected = [&](const char * message) {
        char error[512] = {};
        sink_ptr value = inputs.create(threads, depth, fixture::resolve, &f, error, sizeof(error));
        require(!value && error[0], message);
    };
    {
        scoped_fixture_input bad(inputs, "cache_elastic", "2");
        rejected("invalid CACHE_ELASTIC was accepted");
    }
    {
        scoped_fixture_input off(inputs, "cache_elastic", "0");
        scoped_fixture_input no_prefill(inputs, "cache_prefill_slots", nullptr);
        scoped_fixture_input bad(inputs, "cache_partial_reclaim", "2");
        rejected("invalid CACHE_PARTIAL_RECLAIM was accepted");
    }
    {
        scoped_fixture_input missing(inputs, "cache_prefill_slots", nullptr);
        rejected("elastic cache without CACHE_PREFILL_SLOTS was accepted");
    }
    {
        scoped_fixture_input priority(inputs, "cache_allocation", "priority");
        rejected("elastic cache with priority allocation was accepted");
    }
    {
        scoped_fixture_input quotas(inputs, "cache_layer_quotas", "1:1,2:1");
        rejected("elastic cache with fixed layer quotas was accepted");
    }
    {
        scoped_fixture_input off(inputs, "cache_elastic", "0");
        scoped_fixture_input no_prefill(inputs, "cache_prefill_slots", nullptr);
        scoped_fixture_input partial(inputs, "cache_partial_reclaim", "1");
        scoped_fixture_input priority(inputs, "cache_allocation", "priority");
        rejected("partial reclaim with priority allocation was accepted");
    }
    {
        scoped_fixture_input off(inputs, "cache_elastic", "0");
        scoped_fixture_input no_prefill(inputs, "cache_prefill_slots", nullptr);
        scoped_fixture_input partial(inputs, "cache_partial_reclaim", "1");
        scoped_fixture_input quotas(inputs, "cache_layer_quotas", "1:1,2:1");
        rejected("partial reclaim with fixed layer quotas was accepted");
    }
    {
        scoped_fixture_input off(inputs, "cache_elastic", "0");
        rejected("CACHE_PREFILL_SLOTS without elastic mode was accepted");
    }
    std::printf("ELASTIC_CACHE_OK max_slots=6 prefill_slots=2 zero_slots=0 reserve_bytes=536870912\n");
}

// Strict bitwise comparison with the MoE graph at one and four threads.
static std::vector<float> activation_case(fixture_inputs & inputs, const char * name, fixture & f, kimi_cpu_sink * sink, int layer,
        const std::vector<float> & input, const std::vector<int32_t> & ids, const std::vector<float> & weights,
        const kimi_cpu_sink_activation & activation, size_t hits = 0, int admitted = -1) {
    tensors meta(f, layer);
    const auto expected = reference_moe_graph(f, layer, input, ids, weights, activation, 1);
    const auto threaded = reference_moe_graph(f, layer, input, ids, weights, activation, 4);
    require(!std::memcmp(expected.data(), threaded.data(), expected.size() * sizeof(float)),
            std::string(name) + ": MoE reference output depends on its thread count");
    std::vector<float> exact;
    const auto result = run_checked(inputs, name, f, sink, meta, input, ids, weights, expected, layer, 25.0f,
            hits, admitted, &exact, &activation);
    print_sample(name, result);
    return exact;
}

static void q2_activation_test(fixture_inputs & inputs, int threads, int depth) {
    const kimi_cpu_sink_activation clamp = {
        KIMI_CPU_SINK_ACTIVATION_SWIGLU_CLAMP, 0.0f, 0.0f, 0.75f
    };
    fixture q2;
    q2.type = GGML_TYPE_IQ2_XXS;
    q2.down_type = GGML_TYPE_Q2_K;
    q2.latent = 512;
    q2.hidden = 256;
    make_synthetic(q2, 20260920);
    require(q2.entries.begin()->second.raw_bytes % 4096 != 0,
            "Q2 fixture does not exercise logical-to-padded direct I/O");
    std::mt19937 rng(4101);
    std::normal_distribution<float> sample(0.0f, 1.5f);
    std::vector<float> input(q2.latent);
    for (auto & value : input) {
        value = sample(rng);
    }
    const std::vector<int32_t> top6 = { 17, 0, 8, 2, 19, 4 };
    const std::vector<float> weights = { .3f, .25f, .15f, .12f, .1f, .08f };
    auto sink = make_sink(inputs, q2, threads, depth);
    activation_case(inputs, "iq2_xxs_up_gate_q2k_down_swiglu_clamp", q2, sink.get(), 1,
            input, top6, weights, clamp);
    std::printf("DSV41_Q2_CPU_SINK_DECODE_OK threads=%d ring_experts=%d\n", threads, depth);
}

// The RAM cache's storage at load (kimi_cpu_sink_vram_reserve; without a unified-memory GPU, as here, the cache's
// part runs): the geometry from the tensors' strides is the one the first decode token learns, and the slots the
// ledger admits are committed and paged in, so decode commits no new slot. A sink without the load step is the
// reference: the same reads, bit-identical outputs, the same capacity, and its first token pays the new slots. The
// first decode keeps the arena whatever its epoch, through a prefill. An elastic prefill's shrink (shrink_to, as
// partial reclaim) gives load-time slots back, and decode admissions refault them. Without elastic, slots the first
// prefill gives back through the ledger come back at the first decode token as far as the room allows (a fixed and
// an automatic budget), and so do a later request's; with an automatic budget the enabled slots the load step did not
// commit take their room first. A prefill's room (its gathered pages and its buffers) is given back in one step at its
// start and taken back at its first decode token; the load step leaves the pages held then. A cache that cannot give
// room back (neither partial reclaim nor elastic) takes nothing at load.
// The room the host ledger's terms leave before its reserve (CACHE_RESERVE_BYTES, also the floor's margin off unified
// memory): the least of memory.max less memory.current over this process's cgroup and its ancestors with a finite
// limit, and MemAvailable. 0 without a finite cgroup.
#if defined(__linux__)
static uint64_t host_room() {
    std::ifstream groups("/proc/self/cgroup");
    std::string line, path;
    while (std::getline(groups, line)) {
        if (line.compare(0, 3, "0::") == 0) { path = "/sys/fs/cgroup" + line.substr(3); break; }
    }
    uint64_t room = 0;
    bool finite = false;
    for (; path.size() > std::strlen("/sys/fs/cgroup"); path.resize(path.find_last_of('/'))) {
        std::ifstream maximum(path + "/memory.max"), current(path + "/memory.current");
        std::string limit;
        uint64_t used = 0;
        if (!(maximum >> limit) || limit == "max" || !(current >> used)) { continue; }
        const uint64_t spare = std::stoull(limit) > used ? std::stoull(limit) - used : 0;
        room = finite ? std::min(room, spare) : spare;
        finite = true;
    }
    std::ifstream meminfo("/proc/meminfo");
    while (std::getline(meminfo, line)) {
        if (line.rfind("MemAvailable:", 0) == 0) { room = std::min<uint64_t>(room, std::stoull(line.substr(13)) * 1024); }
    }
    return finite ? room : 0;
}
#endif
// The ledger the sink's host arbiter measures with on this OS.
#if defined(__linux__)
using probe_ledger = kimi_host_ledger;
#else
using probe_ledger = ssd::host::native_ledger;
#endif
static void cache_load_tests(fixture_inputs & inputs, int threads, int depth) {
    scoped_fixture_input policy_env(inputs, "cache_policy", "lru");
    scoped_fixture_input allocation_env(inputs, "cache_allocation", "uniform");
    scoped_fixture_input keep_env(inputs, "cache_keep", "1");
    scoped_fixture_input partial_env(inputs, "cache_partial_reclaim", "1");
    scoped_fixture_input elastic_env(inputs, "cache_elastic", nullptr);
    scoped_fixture_input prefill_env(inputs, "cache_prefill_slots", nullptr);
    scoped_fixture_input quotas_env(inputs, "cache_layer_quotas", nullptr);
    fixture f; f.native_ranges = true; make_synthetic(f);
    uint64_t stride = 0; // a role as a decode request reserves it: its bytes and a page prefix, in whole pages
    for (int role = 0; role < 3; ++role) { stride += (f.entries.at({ tensor_name(1, role), 0 }).raw_bytes + 8190) / 4096 * 4096; }
    stride = cache_slot_bytes(inputs, stride);
    std::vector<float> input(f.latent);
    for (size_t i = 0; i < input.size(); ++i) { input[i] = float(int(i % 23) - 11) * .019f; }
    auto load = [&](kimi_cpu_sink * sink, std::string * receipt) {
        tensors meta(f, 1);
        char error[512] = {};
        stderr_capture log;
        require(kimi_cpu_sink_vram_reserve(sink, meta.role[0], meta.role[1], meta.role[2], 6, error, sizeof(error)), error);
        std::istringstream lines(log.read());
        for (std::string line; std::getline(lines, line);) {
            if (line.rfind("[kimi_cpu_sink_cache] reason=load ", 0) == 0) { *receipt = line; }
        }
    };
    auto a = make_sink(inputs, f, threads, depth), b = make_sink(inputs, f, threads, depth);
    configure_cache(a.get(), 12 * stride, { 1, 2 }, 6);
    configure_cache(b.get(), 12 * stride, { 1, 2 }, 6);
    std::string receipt;
    load(a.get(), &receipt);
    const auto loaded = cache_snapshot(a.get());
    require(loaded.capacity_bytes == 12 * stride && loaded.mapped_bytes == 12 * stride &&
            loaded.committed_bytes == 12 * stride && loaded.peak_bytes == 12 * stride && loaded.page_table_bytes > 0 &&
            !loaded.valid_bytes && !loaded.clears, "the load step did not commit every slot through the ledger");
    require(receipt.find(" slots=12 of=12 slot_bytes=" + std::to_string(stride) + " ") != std::string::npos,
            "the load receipt: " + receipt);
    require(cache_snapshot(b.get()).mapped_bytes == 0, "the reference mapped at configure");
    std::vector<float> out_a, out_b;
    // a native range's slot is its read rounded up to a page prefix: admissions are counted, not their bytes
    auto run = [&](const char * name, kimi_cpu_sink * sink, int layer, const std::vector<int32_t> & ids,
                   const std::vector<float> & weights, size_t hits, size_t admitted, std::vector<float> * out) {
        tensors meta(f, layer);
        const auto expected = reference(f, layer, input, ids, weights, 4, 25);
        const auto r = run_checked(inputs, name, f, sink, meta, input, ids, weights, expected, layer, 25, hits, -1, out);
        require(r.stats.cache_admitted == admitted && r.stats.cache_admitted_bytes == admitted * stride,
                std::string(name) + ": cache admission count mismatch");
        return r;
    };
    auto both = [&](const char * name, int layer, const std::vector<int32_t> & ids, const std::vector<float> & weights,
                    size_t hits, size_t admitted) {
        const auto x = run(name, a.get(), layer, ids, weights, hits, admitted, &out_a);
        const auto y = run(name, b.get(), layer, ids, weights, hits, admitted, &out_b);
        require(out_a == out_b, std::string(name) + ": the load step changed the output");
        require(x.stats.cache_new_slot_bytes == 0, std::string(name) + ": decode committed a slot the load step had");
        return y.stats.cache_new_slot_bytes;
    };
    begin_cache(a.get(), 9, 1, false); begin_cache(b.get(), 9, 1, false); // a prefill keeps the arena
    require(cache_snapshot(a.get()).committed_bytes == 12 * stride, "the first prefill gave back the load step's slots");
    begin_cache(a.get(), 9, 2); begin_cache(b.get(), 9, 2);               // decode in an epoch the arena never saw
    uint64_t first_touch = both("load_layer1", 1, { 1, 2 }, { .4f, .6f }, 0, 2);
    first_touch += both("load_layer2", 2, { 3, 4, 5 }, { .2f, .3f, .5f }, 0, 3);
    begin_cache(a.get(), 9, 3); begin_cache(b.get(), 9, 3);
    first_touch += both("load_layer1_next", 1, { 1, 6 }, { .5f, .5f }, 1, 1);
    const auto sa = cache_snapshot(a.get()), sb = cache_snapshot(b.get());
    require(first_touch == 6 * stride && sb.committed_bytes == 6 * stride, "the reference did not pay its new slots");
    require(sa.capacity_bytes == sb.capacity_bytes && sa.valid_bytes == sb.valid_bytes && sa.hits == sb.hits &&
            sa.misses == sb.misses && !sa.clears && sa.committed_bytes == 12 * stride,
            "the load step's geometry is not the decode's, or decode mapped again");
    {
        // an elastic prefill keeps 4 slots: the load step's others go back (MADV_DONTNEED) and come back by refault
        scoped_fixture_input elastic(inputs, "cache_elastic", "1");
        scoped_fixture_input prefill(inputs, "cache_prefill_slots", "4");
        auto c = make_sink(inputs, f, threads, depth);
        configure_cache(c.get(), 12 * stride, { 1, 2 }, 6);
        load(c.get(), &receipt);
        begin_cache(c.get(), 10, 1, false);
        const auto shrunk = cache_snapshot(c.get());
        require(shrunk.enabled_bytes == 4 * stride && shrunk.committed_bytes == 4 * stride &&
                shrunk.released_bytes == 8 * stride && shrunk.shrink_ops == 1 && shrunk.mapped_bytes == 12 * stride,
                "an elastic prefill did not give load-time slots back");
        begin_cache(c.get(), 10, 2);
        size_t refault = 0;
        for (int layer : { 1, 2 }) {
            refault += run("load_elastic_refault", c.get(), layer, { 7, 8, 9, 10 }, { .1f, .2f, .3f, .4f }, 0, 4,
                nullptr).stats.cache_new_slot_bytes;
        }
        const auto grown = cache_snapshot(c.get());
        // each layer gave back its 4 lowest slots (all equally cold) and admits into them first
        require(refault == 8 * stride && grown.refaulted_bytes == 8 * stride && grown.committed_bytes == 12 * stride,
                "decode admissions into given-back load-time slots did not refault them");
    }
    for (const bool automatic : { false, true }) {
        // an allocation larger than the cgroup during the first prefill takes every load-time slot back (the ledger's
        // partial reclaim); the room is back by the first decode token, which enables them again
        scoped_fixture_input policy(inputs, "cache_policy", automatic ? "count_decay2" : "lru");
        const std::string kind = automatic ? "automatic budget: " : "fixed budget: ";
        auto d = make_sink(inputs, f, threads, depth);
        configure_cache(d.get(), automatic ? std::numeric_limits<uint64_t>::max() : 12 * stride, { 1, 2 }, 6);
        load(d.get(), &receipt);
        require(cache_snapshot(d.get()).committed_bytes == 12 * stride, kind + "the load step did not commit 12 slots");
        begin_cache(d.get(), 11, 1, false);
        require(!kimi_cpu_sink_cache_make_room(d.get(), uint64_t(1) << 50, 0), kind + "the ledger granted 1 PiB");
        const auto given = cache_snapshot(d.get());
        require(given.enabled_bytes == 0 && given.committed_bytes == 0 && given.released_bytes == 12 * stride,
                kind + "the ledger did not take the load step's slots back");
        std::string regrow;
        {
            stderr_capture log;
            begin_cache(d.get(), 11, 2);
            std::istringstream lines(log.read());
            for (std::string line; std::getline(lines, line);) {
                if (line.rfind("[kimi_cpu_sink_cache] load_regrow=1 ", 0) == 0) { regrow = line; }
            }
        }
        const auto back = cache_snapshot(d.get());
        require(back.enabled_bytes == 12 * stride && back.capacity_bytes == 12 * stride && !back.committed_bytes &&
                !back.clears && regrow.find(" before_slots=0 after_slots=12 of=12 ") != std::string::npos,
                kind + "the first decode did not take back the given-back slots: " + regrow);
        const auto first = run("load_regrow_layer1", d.get(), 1, { 1, 2, 3 }, { .2f, .3f, .5f }, 0, 3, nullptr);
        require(first.stats.cache_new_slot_bytes == 3 * stride, kind + "regrown slots were not paged in at admission");
        // a later request's prefill gives slots back as the first's did, and its first decode token takes them back
        begin_cache(d.get(), 11, 3, false);
        require(!kimi_cpu_sink_cache_make_room(d.get(), uint64_t(1) << 50, 0), kind + "the ledger granted 1 PiB");
        require(cache_snapshot(d.get()).enabled_bytes == 0, kind + "a later prefill did not give its slots back");
        regrow.clear();
        {
            stderr_capture log;
            begin_cache(d.get(), 11, 4);
            std::istringstream lines(log.read());
            for (std::string line; std::getline(lines, line);) {
                if (line.rfind("[kimi_cpu_sink_cache] prefill_regrow=1 ", 0) == 0) { regrow = line; }
            }
        }
        require(cache_snapshot(d.get()).enabled_bytes == 12 * stride &&
                regrow.find(" before_slots=0 after_slots=12 of=12 ") != std::string::npos,
                kind + "a later request's first decode did not take its prefill's slots back: " + regrow);
        begin_cache(d.get(), 11, 5);
        require(cache_snapshot(d.get()).enabled_bytes == 12 * stride, kind + "a later decode token changed the slots");
    }
#if defined(__linux__)
    // Linux only: the drift is injected between the Linux ledger's cgroup and /proc/meminfo reads.
    {
        // An automatic budget whose room shrinks between geometry's reading of it and the load step's own, as when
        // the server allocates while it loads: the load step commits fewer slots than geometry enabled, and those owe
        // their room at admission. The first prefill gives resident slots back; at the first decode token the room
        // holds the owed slots and 1.5 more, so one given-back slot comes back, not all. Experts of 51 MiB (Q8_0
        // 4096 x 4096, no file: nothing is read) keep the process's own movements far below a slot.
        auto number = [](const std::string & line, const std::string & key) {
            const size_t at = line.find(" " + key + "=");
            require(at != std::string::npos, "the receipt lacks " + key + ": " + line);
            return uint64_t(std::stoull(line.substr(at + key.size() + 2)));
        };
        fixture big;
        big.latent = big.hidden = 4096; big.type = big.down_type = GGML_TYPE_Q8_0; big.experts = 8;
        const uint64_t role = (ggml_row_size(GGML_TYPE_Q8_0, 4096) * 4096 + 8190) / 4096 * 4096;
        const uint64_t slot = cache_slot_bytes(inputs, 3 * role);
        const uint64_t cost = slot + slot / 128; // a slot and its page tables, until geometry's receipt prices it
        const uint64_t ring = uint64_t(depth) * 3 * role, room = host_room();
        test_resources::need(room > ring + 7 * cost + (1ull << 30), "the owed slots' case needs a finite cgroup with room");
        // geometry sees 6.5 slots of room, after the read ring the load step resizes first
        scoped_fixture_input reserve_env(inputs, "cache_reserve_bytes",
            std::to_string(room - ring - 13 * cost / 2).c_str());
        const uint64_t reserve = room - ring - 13 * cost / 2;
        auto g = make_sink(inputs, big, threads, depth);
        configure_cache(g.get(), std::numeric_limits<uint64_t>::max(), { 1, 2 }, 6);
        std::vector<char> drift, pad;
        std::string capacity, loaded;
        {
            auto & h = pread_hold::get();
            // Admission can legitimately re-read the budget before the ring
            // exists. Inject between geometry's cgroup/system observations only
            // after the mandatory ring is physically resident, rather than
            // assuming a particular number of ledger reads.
            std::function<void()> inject;
            inject = [&] {
                const uint64_t current_room = host_room();
                if (current_room >= room || room - current_room < ring / 2) {
                    std::lock_guard<std::mutex> lock(h.mutex); h.before_meminfo = inject; return;
                }
                drift.assign(3 * cost / 2, 1);
            };
            { std::lock_guard<std::mutex> lock(h.mutex); h.before_meminfo = inject; }
            tensors meta(big, 1);
            char error[512] = {};
            stderr_capture log;
            const bool ok = kimi_cpu_sink_vram_reserve(g.get(), meta.role[0], meta.role[1], meta.role[2], 6, error,
                sizeof(error));
            { std::lock_guard<std::mutex> lock(h.mutex); h.before_meminfo = nullptr; }
            require(ok, error);
            std::istringstream lines(log.read());
            for (std::string line; std::getline(lines, line);) {
                if (line.rfind("[kimi_cpu_sink_cache] auto_capacity_bytes=", 0) == 0) { capacity = line; }
                if (line.rfind("[kimi_cpu_sink_cache] reason=load ", 0) == 0) { loaded = line; }
            }
        }
        require(!drift.empty() && !capacity.empty() && !loaded.empty(), "owed slots: no drift or no load: " + loaded);
        const uint64_t slot_cost = number(capacity, "slot_cost_bytes"), stride = number(loaded, "slot_bytes");
        const uint64_t metadata = number(capacity, "metadata_node_storage_bytes") + number(capacity, "metadata_guard_bytes") +
            64 * 1024;
        require(number(loaded, "slots") < number(loaded, "of"), "owed slots: the load step committed every slot: " + loaded);
        // the first prefill: a shortage of 1.5 slots takes two resident slots back
        begin_cache(g.get(), 12, 1, false);
        require(kimi_cpu_sink_cache_make_room(g.get(), host_room() - reserve + 3 * stride / 2, 0),
                "owed slots: the ledger did not make room");
        const auto given = cache_snapshot(g.get());
        const uint64_t enabled = given.enabled_bytes / stride, owed = enabled - given.committed_bytes / stride;
        const uint64_t disabled = number(loaded, "of") - enabled;
        require(owed >= 1 && disabled >= 2, "owed slots: the prefill gave back owed slots or too few: " +
                std::to_string(enabled) + " enabled, " + std::to_string(owed) + " owed");
        // the first decode token: room for the owed slots and 1.5 more
        std::vector<char>().swap(drift);
        const uint64_t want = metadata + owed * slot_cost + 3 * slot_cost / 2, now = host_room() - reserve;
        require(now > want, "owed slots: no room to set");
        pad.assign(now - want, 1);
        std::string regrow;
        {
            stderr_capture log;
            begin_cache(g.get(), 12, 2);
            std::istringstream lines(log.read());
            for (std::string line; std::getline(lines, line);) {
                if (line.rfind("[kimi_cpu_sink_cache] load_regrow=1 ", 0) == 0) { regrow = line; }
            }
        }
        require(!regrow.empty(), "owed slots: no load_regrow receipt");
        const uint64_t surplus = number(regrow, "surplus_bytes"), fit = surplus > metadata ? (surplus - metadata) / slot_cost : 0;
        const uint64_t more = std::min(disabled, fit > owed ? fit - owed : 0);
        require(number(regrow, "owed_slots") == owed && number(regrow, "before_slots") == enabled &&
                number(regrow, "after_slots") == enabled + more && std::min(disabled, fit) != more,
                "owed slots: the first decode did not price the owed slots first: " + regrow);
        std::printf("case=load_regrow_owed status=PASS %s | %s\n", loaded.c_str(), regrow.c_str());
    }
#else
    std::printf("case=load_regrow_owed status=SKIP reason=linux_ledger_injection\n");
#endif
    {
        // A prefill's room, held before it allocates anything (kimi_cpu_sink_cache_prefill_room): its pages and its
        // compute buffers take their room in one give-back at its start; its later checks keep the rest; the first
        // decode token takes the slots back. File pages count as room (CACHE_RECLAIMABLE_FILE), so the hold alone
        // keeps the pages' room. Experts of 51 MiB keep the process's own movements far below a slot.
        scoped_fixture_input file_env(inputs, "cache_reclaimable_file", "1");
        auto number = [](const std::string & line, const std::string & key) {
            const size_t at = line.find(" " + key + "=");
            require(at != std::string::npos, "the receipt lacks " + key + ": " + line);
            return uint64_t(std::stoull(line.substr(at + key.size() + 2)));
        };
        fixture big;
        big.latent = big.hidden = 4096; big.type = big.down_type = GGML_TYPE_Q8_0; big.experts = 8;
        const uint64_t role = (ggml_row_size(GGML_TYPE_Q8_0, 4096) * 4096 + 8190) / 4096 * 4096;
        const uint64_t slot = cache_slot_bytes(inputs, 3 * role);
        auto h = make_sink(inputs, big, threads, depth);
        configure_cache(h.get(), 6 * slot, { 1, 2 }, 6);
        {
            tensors meta(big, 1);
            char error[512] = {};
            require(kimi_cpu_sink_vram_reserve(h.get(), meta.role[0], meta.role[1], meta.role[2], 6, error, sizeof(error)),
                error);
        }
        require(cache_snapshot(h.get()).committed_bytes == 6 * slot, "prefill room: the load step did not commit 6 slots");
        begin_cache(h.get(), 13, 1, false);
        probe_ledger probe;
        inputs.configure(probe, false);
        const uint64_t room = probe.measure(0).surplus(), pages = room + 5 * slot / 2, host = 1 << 20;
        require(room != UINT64_MAX && room > 0, "prefill room: no finite room to measure");
        std::string line;
        {
            stderr_capture log;
            require(kimi_cpu_sink_cache_prefill_room(h.get(), pages - host, host, 0), "prefill room: refused");
            std::istringstream lines(log.read());
            for (std::string l; std::getline(lines, l);) {
                if (l.rfind("[kimi_host_ledger] event=prefill_room ", 0) == 0) { line = l; }
            }
        }
        const auto held = cache_snapshot(h.get());
        require(held.enabled_bytes == 3 * slot && held.committed_bytes == 3 * slot && held.shrink_ops == 1 &&
                held.released_bytes == 3 * slot && number(line, "slots_before") == 6 && number(line, "slots_after") == 3 &&
                number(line, "pages") == pages - host && number(line, "fits") == 1,
                "prefill room: 2.5 slots of shortage did not take 3 slots back in one step: " + line);
        begin_cache(h.get(), 13, 1, false); // a later layer's check: the hold is room already made
        require(cache_snapshot(h.get()).enabled_bytes == 3 * slot && cache_snapshot(h.get()).shrink_ops == 1,
                "prefill room: a later check gave the held room back again");
        std::string regrow; // the first request of an arena made at load
        {
            stderr_capture log;
            begin_cache(h.get(), 13, 2);
            std::istringstream lines(log.read());
            for (std::string l; std::getline(lines, l);) {
                if (l.rfind("[kimi_cpu_sink_cache] load_regrow=1 ", 0) == 0) { regrow = l; }
            }
        }
        require(cache_snapshot(h.get()).enabled_bytes == 6 * slot &&
                regrow.find(" before_slots=3 after_slots=6 of=6 ") != std::string::npos,
                "prefill room: the first decode token did not take the slots back: " + regrow);
        std::printf("case=prefill_room status=PASS %s | %s\n", line.c_str(), regrow.c_str());
    }
    {
        // The load step with a prefill's pages held (the load-time graph's): the arena has the slots the first decode
        // token would size, the slots committed leave the pages' room, and those are owed at the first decode token.
        scoped_fixture_input file_env(inputs, "cache_reclaimable_file", "1");
        auto number = [](const std::string & line, const std::string & key) {
            const size_t at = line.find(" " + key + "=");
            require(at != std::string::npos, "the receipt lacks " + key + ": " + line);
            return uint64_t(std::stoull(line.substr(at + key.size() + 2)));
        };
        fixture big;
        big.latent = big.hidden = 4096; big.type = big.down_type = GGML_TYPE_Q8_0; big.experts = 8;
        const uint64_t role = (ggml_row_size(GGML_TYPE_Q8_0, 4096) * 4096 + 8190) / 4096 * 4096;
        const uint64_t slot = cache_slot_bytes(inputs, 3 * role);
        const uint64_t cost = slot + slot / 128, ring = uint64_t(depth) * 3 * role;
        probe_ledger probe;
        inputs.configure(probe, false);
        const uint64_t room = probe.measure(0).surplus() + probe.reserve;
        require(room != UINT64_MAX && room > ring + 7 * cost + (1ull << 30), "load hold: the case needs a finite cgroup with room");
        // geometry sees 6.5 slots of room after the read ring; the hold takes 3 of them
        scoped_fixture_input reserve_env(inputs, "cache_reserve_bytes", std::to_string(room - ring - 13 * cost / 2).c_str());
        auto g = make_sink(inputs, big, threads, depth);
        configure_cache(g.get(), std::numeric_limits<uint64_t>::max(), { 1, 2 }, 6);
        std::string loaded;
        {
            tensors meta(big, 1);
            char error[512] = {};
            stderr_capture log;
            require(kimi_cpu_sink_cache_prefill_room(g.get(), 3 * cost, 0, 0), "load hold: the hold was refused");
            const bool ok = kimi_cpu_sink_vram_reserve(g.get(), meta.role[0], meta.role[1], meta.role[2], 6, error,
                sizeof(error));
            std::istringstream lines(log.read());
            for (std::string l; std::getline(lines, l);) {
                if (l.rfind("[kimi_cpu_sink_cache] reason=load ", 0) == 0) { loaded = l; }
            }
            require(ok, error);
        }
        require(number(loaded, "of") == 6 && number(loaded, "slots") == 3,
                "load hold: the arena or the committed slots did not leave the held pages: " + loaded);
        const uint64_t stride = number(loaded, "slot_bytes");
        begin_cache(g.get(), 14, 1, false);
        require(cache_snapshot(g.get()).enabled_bytes == 6 * stride, "load hold: the first prefill gave held slots back");
        begin_cache(g.get(), 14, 2);
        const auto first = cache_snapshot(g.get());
        require(first.enabled_bytes == 6 * stride && first.committed_bytes == 3 * stride,
                "load hold: the first decode token did not keep the held slots enabled and owed");
        std::printf("case=load_hold status=PASS %s\n", loaded.c_str());
    }
    for (const bool elastic : { false, true }) {
        // The load step only for a cache that can give room back to the first prefill's ledger requests: without
        // partial reclaim and without elastic, nothing at load, and the first decode token sizes and pages the cache
        // as without the hook; elastic alone takes the load step (its prefill shrinks to the prefill slots).
        scoped_fixture_input partial(inputs, "cache_partial_reclaim", nullptr);
        scoped_fixture_input elastic_env(inputs, "cache_elastic", elastic ? "1" : nullptr);
        scoped_fixture_input prefill(inputs, "cache_prefill_slots", elastic ? "4" : nullptr);
        const std::string kind = elastic ? "elastic, no partial reclaim: " : "no partial reclaim, no elastic: ";
        auto e = make_sink(inputs, f, threads, depth);
        configure_cache(e.get(), 12 * stride, { 1, 2 }, 6);
        std::string line;
        load(e.get(), &line);
        const auto at_load = cache_snapshot(e.get());
        require(elastic ? line.find(" slots=12 of=12 ") != std::string::npos && at_load.committed_bytes == 12 * stride
                        : line.empty() && !at_load.mapped_bytes && !at_load.committed_bytes,
                kind + "the load step: " + line);
        begin_cache(e.get(), 13, 1, false);
        if (elastic) { require(cache_snapshot(e.get()).committed_bytes == 4 * stride, kind + "the prefill kept more than 4 slots"); }
        begin_cache(e.get(), 13, 2);
        // both page their admissions in: new slots, or (elastic) slots the prefill gave back
        const auto first = run("load_gate_layer1", e.get(), 1, { 1, 2 }, { .4f, .6f }, 0, 2, nullptr);
        require(first.stats.cache_new_slot_bytes == 2 * stride && cache_snapshot(e.get()).capacity_bytes == 12 * stride,
                kind + "the first decode's slots");
    }
    {
        // An automatic budget without partial reclaim or elastic: the read ring is not made resident at load either (the
        // first decode charges it). Experts of 51 MiB (Q8_0 4096 x 4096, no file) make the ring's pages, a slot for each
        // of the depth's reads, show in the process's resident memory.
        scoped_fixture_input partial(inputs, "cache_partial_reclaim", nullptr);
        auto resident = [] { return test_resources::resident_bytes(); };
        fixture big;
        big.latent = big.hidden = 4096; big.type = big.down_type = GGML_TYPE_Q8_0; big.experts = 8;
        const uint64_t slot = 3 * ((ggml_row_size(GGML_TYPE_Q8_0, 4096) * 4096 + 8190) / 4096 * 4096);
        auto r = make_sink(inputs, big, threads, depth);
        configure_cache(r.get(), std::numeric_limits<uint64_t>::max(), { 1, 2 }, 6);
        tensors meta(big, 1);
        char error[512] = {};
        std::string loaded;
        const uint64_t before = resident();
        {
            stderr_capture log;
            require(kimi_cpu_sink_vram_reserve(r.get(), meta.role[0], meta.role[1], meta.role[2], 6, error, sizeof(error)), error);
            std::istringstream lines(log.read());
            for (std::string line; std::getline(lines, line);) {
                if (line.rfind("[kimi_cpu_sink_cache] reason=load ", 0) == 0) { loaded = line; }
            }
        }
        const uint64_t after = resident(), grown = after > before ? after - before : 0;
        require(loaded.empty() && !cache_snapshot(r.get()).mapped_bytes && grown < slot / 2,
                "no partial reclaim, no elastic, automatic budget: the load step took the ring or the arena (resident +" +
                std::to_string(grown) + " bytes, a slot " + std::to_string(slot) + "): " + loaded);
        std::printf("case=load_gate_ring status=PASS resident_growth=%llu slot=%llu depth=%d\n", (unsigned long long) grown,
                (unsigned long long) slot, depth);
    }
    std::printf("CACHE_LOAD_OK stride=%llu first_touch=%llu receipt=%s\n", (unsigned long long) stride,
            (unsigned long long) first_touch, receipt.c_str());
}

// An allocation outside the caches that names no reclaimer gets the RAM cache's room: the CPU backend's work buffer
// (ggml_backend_cpu_graph_compute), which a prefill grows when its graph needs more than every graph before it. The
// sink and the backend share one host client, as in a context. The cache takes the envelope's room at load (an
// automatic budget with partial reclaim) and leaves half a slot; a matrix product then needs a work buffer of three
// slots. The host arbiter runs the sink's reclaimer before it would refuse: the cache gives back the slots the
// shortage needs and no more, the ledger holds no pending debit and has granted the buffer, and the sink computes bit
// for bit after it.
static void work_buffer_reclaim_tests(fixture_inputs & inputs, int threads, int depth) {
    scoped_fixture_input policy_env(inputs, "cache_policy", "lru");
    scoped_fixture_input allocation_env(inputs, "cache_allocation", "uniform");
    scoped_fixture_input keep_env(inputs, "cache_keep", "1");
    scoped_fixture_input partial_env(inputs, "cache_partial_reclaim", "1");
    scoped_fixture_input elastic_env(inputs, "cache_elastic", nullptr);
    scoped_fixture_input quotas_env(inputs, "cache_layer_quotas", nullptr);
    auto number = [](const std::string & line, const std::string & key) {
        const size_t at = line.find(" " + key + "=");
        require(at != std::string::npos, "the receipt lacks " + key + ": " + line);
        return uint64_t(std::stoull(line.substr(at + key.size() + 2)));
    };
    // Experts of 13 MiB (Q8_0 2048 x 2048) keep the process's own movements far below a slot.
    fixture f; f.latent = f.hidden = 2048; f.type = f.down_type = GGML_TYPE_Q8_0; f.experts = 4; f.native_ranges = true;
    make_synthetic(f);
    const std::vector<int32_t> ids = { 0, 2 };
    const std::vector<float> weights = { .35f, .65f };
    std::vector<float> input(f.latent);
    for (size_t i = 0; i < input.size(); ++i) { input[i] = float(int(i % 29) - 14) * .011f; }
    const auto expected = reference(f, 1, input, ids, weights, 4, 25);
    // a role as a decode request reserves it: its bytes and a page prefix, in whole pages
    const uint64_t role = (f.entries.at({ tensor_name(1, 0), 0 }).raw_bytes + 8190) / 4096 * 4096;
    const uint64_t slot = cache_slot_bytes(inputs, 3 * role), cost = slot + slot / 128, ring = uint64_t(depth) * 3 * role;

    // F16 weights: the backend converts the F32 input to F16 into its work buffer, three slots of it.
    const int64_t k = 4096, n = int64_t(3 * slot / (2 * k));
    ggml_context * ctx = ggml_init({ ggml_tensor_overhead() * 4 + ggml_graph_overhead(), nullptr, true });
    require(ctx, "work buffer: graph context");
    ggml_tensor * a = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, k, 1), * b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k, n);
    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, ggml_mul_mat(ctx, a, b));
    ggml_backend_t backend = ggml_backend_cpu_init();
    require(backend, "work buffer: CPU backend");
    ggml_backend_cpu_set_n_threads(backend, threads);
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    require(buffer, "work buffer: graph tensors");
    ggml_backend_tensor_memset(a, 0x3c, 0, ggml_nbytes(a));
    ggml_backend_tensor_memset(b, 0x3c, 0, ggml_nbytes(b));
    const uint64_t work = ggml_graph_plan(graph, threads, nullptr).work_size;
    require(work > 5 * slot / 2, "work buffer: the product's work buffer is below the shortage it is meant to make");

    // One host client for the sink and the backend; everything the case allocates outside them is resident by now.
    auto owner = std::make_shared<ssd::host::process_arbiter>();
    auto made = owner->make_client();
    require(bool(made), made.error);
    auto client = std::move(made.value);
    const auto saved = inputs.runtime;
    struct restore { fixture_inputs & inputs; ssd::runtime_services value; ~restore() { inputs.set_runtime_services(value); } }
        restore_runtime{ inputs, saved };
    ssd::runtime_services services = saved;
    services.host = client.get();
    inputs.set_runtime_services(services);
    using attach_fn = bool (*)(ggml_backend_t, const ssd::runtime_services *) noexcept;
    const auto attach = reinterpret_cast<attach_fn>(
        ggml_backend_reg_get_proc_address(ggml_backend_cpu_reg(), "ggml_backend_cpu_set_ssd_runtime_services"));
    require(attach && attach(backend, &services), "work buffer: the backend did not take the context's services");
    probe_ledger probe;
    inputs.configure(probe, false);
    const uint64_t room = probe.measure(0).surplus() + probe.reserve;
    require(room != UINT64_MAX && room > ring + 7 * cost + (1ull << 30), "work buffer: the case needs a finite envelope with room");
    // geometry sees 6.5 slots of room after the read ring, and the load step commits what it sized
    scoped_fixture_input reserve_env(inputs, "cache_reserve_bytes", std::to_string(room - ring - 13 * cost / 2).c_str());
    auto sink = make_sink(inputs, f, threads, depth);
    configure_cache(sink.get(), std::numeric_limits<uint64_t>::max(), { 1, 2 }, 6);
    std::string loaded;
    {
        tensors meta(f, 1);
        char error[512] = {};
        stderr_capture log;
        const bool ok = kimi_cpu_sink_vram_reserve(sink.get(), meta.role[0], meta.role[1], meta.role[2], int(ids.size()),
            error, sizeof(error));
        std::istringstream lines(log.read());
        for (std::string l; std::getline(lines, l);) {
            if (l.rfind("[kimi_cpu_sink_cache] reason=load ", 0) == 0) { loaded = l; }
        }
        require(ok, error);
    }
    require(!loaded.empty() && number(loaded, "slot_bytes") == slot && number(loaded, "slots") >= 4,
            "work buffer: the load step did not take the envelope's room: " + loaded);
    begin_cache(sink.get(), 21, 1, false); // a prefill keeps the arena
    const auto before = cache_snapshot(sink.get());
    const auto left = client->measure(work, true);
    require(bool(left) && !left.value.fits(), "work buffer: the cache left room for the work buffer");
    const uint64_t shortage = left.value.shortage(), granted = client->state().value.granted;

    std::string shrink;
    ggml_status status;
    {
        stderr_capture log;
        status = ggml_backend_graph_compute(backend, graph);
        std::istringstream lines(log.read());
        for (std::string l; std::getline(lines, l);) {
            if (l.rfind("[kimi_cpu_sink_cache] cache_shrink=1 ", 0) == 0) { shrink = l; }
        }
    }
    require(status == GGML_STATUS_SUCCESS, "work buffer: the backend's work buffer was refused while the cache held the room");
    const auto after = cache_snapshot(sink.get());
    const uint64_t released = after.released_bytes - before.released_bytes;
    require(after.shrink_ops == before.shrink_ops + 1 && after.guard_reclaims == before.guard_reclaims + 1 &&
            released >= shortage && released < shortage + slot && before.committed_bytes - after.committed_bytes == released,
            "work buffer: the cache did not give back the shortage's slots in one step: shortage " + std::to_string(shortage) +
            ", released " + std::to_string(released) + ", shrinks " + std::to_string(after.shrink_ops - before.shrink_ops) +
            ", committed " + std::to_string(before.committed_bytes) + " -> " + std::to_string(after.committed_bytes) + ": " + shrink);
    const auto state = client->state();
    require(bool(state) && state.value.pending_bytes == 0 && state.value.granted == granted + work,
            "work buffer: the ledger kept a pending debit or did not grant the buffer");
    require(client->measure(0, true).value.fits(), "work buffer: the grant overdrew the envelope");

    begin_cache(sink.get(), 21, 2); // the first decode token takes back what the room allows now
    tensors meta(f, 1);
    run_checked(inputs, "work_reclaim_decode", f, sink.get(), meta, input, ids, weights, expected, 1, 25, 0, -1);
    std::printf("case=work_buffer_reclaim status=PASS work=%llu shortage=%llu released=%llu slot=%llu %s\n",
            (unsigned long long) work, (unsigned long long) shortage, (unsigned long long) released,
            (unsigned long long) slot, shrink.c_str());
    ggml_backend_buffer_free(buffer);
    ggml_backend_free(backend);
    ggml_free(ctx);
    std::printf("WORK_BUFFER_RECLAIM_OK\n");
}

static void activation_tests(fixture_inputs & inputs, int threads, int depth) {
    const kimi_cpu_sink_activation clamp10 = { KIMI_CPU_SINK_ACTIVATION_SWIGLU_CLAMP, 0.0f, 0.0f, 10.0f };
    const kimi_cpu_sink_activation clamp = { KIMI_CPU_SINK_ACTIVATION_SWIGLU_CLAMP, 0.0f, 0.0f, 0.75f };
    const kimi_cpu_sink_activation situ = { KIMI_CPU_SINK_ACTIVATION_SITU, 4.0f, 25.0f, 0.0f };
    // DeepSeek-V4.1 kernels at reduced width: MXFP4 experts with Q8_0 inputs and
    // 32-column blocks (320 and 224 are not multiples of 256).
    fixture f; f.type = f.down_type = GGML_TYPE_MXFP4; f.latent = 320; f.hidden = 224; make_synthetic(f, 20260914);
    std::mt19937 rng(4101); std::normal_distribution<float> sample(0.0f, 1.5f);
    std::vector<float> input(f.latent); for (auto & v : input) { v = sample(rng); }
    const std::vector<int32_t> top6 = { 17, 0, 8, 2, 19, 4 };
    const std::vector<float> w6 = { .3f, .25f, .15f, .12f, .1f, .08f };
    auto sink = make_sink(inputs, f, threads, depth);
    const auto wide = activation_case(inputs, "mxfp4_swiglu_clamp_top6_limit10", f, sink.get(), 1, input, top6, w6, clamp10);
    const auto narrow = activation_case(inputs, "mxfp4_swiglu_clamp_top6_limit075", f, sink.get(), 1, input, top6, w6, clamp);
    require(std::memcmp(wide.data(), narrow.data(), wide.size() * sizeof(float)) != 0,
            "the clamp limit did not change the output; the fixture does not exercise clamping");
    activation_case(inputs, "mxfp4_swiglu_clamp_one_expert", f, sink.get(), 2, input, { 3 }, { 1.5f }, clamp);
    activation_case(inputs, "mxfp4_swiglu_clamp_duplicate_ids", f, sink.get(), 2, input, { 9, 2, 9 }, { .5f, .3f, .2f }, clamp);
    const std::vector<int32_t> top16 = { 17, 0, 8, 2, 19, 4, 11, 6, 15, 1, 13, 7, 18, 5, 10, 9 };
    std::vector<float> w16(16); for (int i = 0; i < 16; ++i) { w16[i] = float(16 - i) / 136; }
    activation_case(inputs, "mxfp4_swiglu_clamp_top16", f, sink.get(), 1, input, top16, w16, clamp);
    activation_case(inputs, "mxfp4_situ", f, sink.get(), 1, input, top6, w6, situ);
    for (int repeat = 0; repeat < 4; ++repeat) {
        activation_case(inputs, "mxfp4_swiglu_clamp_layer_generation_reuse", f, sink.get(), repeat % 2 + 1, input, top6, w6, clamp);
        input[size_t(repeat)] += .25f;
    }
    const auto uncached = activation_case(inputs, "mxfp4_swiglu_clamp_uncached", f, sink.get(), 1, input, top6, w6, clamp);

    // RAM cache (count_decay2 as in the V4.1 recipe, and window): cold direct
    // admission, full and partial hits stay bit-identical to the graph.
    const uint64_t unit = triplet_bytes(f);
    for (const char * policy : { "count_decay2", "window" }) {
        scoped_fixture_input policy_env(inputs, "cache_policy", policy);
        scoped_fixture_input keep_env(inputs, "cache_keep", nullptr);
        scoped_fixture_input allocation_env(inputs, "cache_allocation", nullptr);
        const bool window = !std::strcmp(policy, "window");
        auto cached = make_sink(inputs, f, threads, depth);
        configure_cache(cached.get(), 6 * unit, { 1, 2 }, 6);
        begin_cache(cached.get(), 41, 1);
        const auto cold = activation_case(inputs, "mxfp4_cache_cold", f, cached.get(), 1, input, top6, w6, clamp, 0, 6);
        begin_cache(cached.get(), 41, 2);
        const auto hit = activation_case(inputs, "mxfp4_cache_full_hit", f, cached.get(), 1, input, top6, w6, clamp, 6, 0);
        require(!std::memcmp(cold.data(), uncached.data(), cold.size() * sizeof(float)) &&
                !std::memcmp(hit.data(), uncached.data(), hit.size() * sizeof(float)),
                "RAM cache changed an output bit");
        begin_cache(cached.get(), 41, 3);
        // count_decay2 bypasses admission of the less frequent new experts.
        activation_case(inputs, "mxfp4_cache_partial_hit", f, cached.get(), 1, input, { 17, 3, 8, 12, 19, 5 }, w6, clamp,
                3, window ? 3 : 0);
        std::printf("case=mxfp4_cache_policy policy=%s status=PASS\n", policy);
    }
    {
        // Opt-in reclaimable page-cache accounting: invalid values fail at
        // create; enabled admission and hits stay bit-identical.
        for (const char * bad : { "", "2", "yes" }) {
            scoped_fixture_input env(inputs, "cache_reclaimable_file", bad);
            char error[512] = {};
            sink_ptr rejected = inputs.create(threads, depth, fixture::resolve, &f, error, sizeof(error));
            require(!rejected && error[0], "invalid CACHE_RECLAIMABLE_FILE was accepted");
        }
        scoped_fixture_input reclaim_env(inputs, "cache_reclaimable_file", "1");
        scoped_fixture_input policy_env(inputs, "cache_policy", "count_decay2");
        scoped_fixture_input keep_env(inputs, "cache_keep", nullptr);
        scoped_fixture_input allocation_env(inputs, "cache_allocation", nullptr);
        auto cached = make_sink(inputs, f, threads, depth);
        configure_cache(cached.get(), 6 * unit, { 1, 2 }, 6);
        begin_cache(cached.get(), 43, 1);
        const auto cold = activation_case(inputs, "mxfp4_cache_reclaimable_file_cold", f, cached.get(), 1, input, top6, w6, clamp, 0, 6);
        begin_cache(cached.get(), 43, 2);
        const auto hit = activation_case(inputs, "mxfp4_cache_reclaimable_file_hit", f, cached.get(), 1, input, top6, w6, clamp, 6, 0);
        require(!std::memcmp(cold.data(), uncached.data(), cold.size() * sizeof(float)) &&
                !std::memcmp(hit.data(), uncached.data(), hit.size() * sizeof(float)),
                "reclaimable-file cache changed an output bit");
    }

    // Other trait-selected type combinations.
    fixture mixed; mixed.type = GGML_TYPE_MXFP4; mixed.down_type = GGML_TYPE_Q8_0;
    mixed.latent = 320; mixed.hidden = 224; make_synthetic(mixed, 20260915);
    auto mixed_sink = make_sink(inputs, mixed, threads, depth);
    activation_case(inputs, "mxfp4_up_gate_q8_0_down_swiglu_clamp", mixed, mixed_sink.get(), 1, input, top6, w6, clamp);
    fixture q3; make_synthetic(q3, 20260916);
    std::vector<float> q3_input(q3.latent); for (auto & v : q3_input) { v = sample(rng); }
    auto q3_sink = make_sink(inputs, q3, threads, depth);
    activation_case(inputs, "q3k_swiglu_clamp", q3, q3_sink.get(), 1, q3_input, top6, w6, clamp);
    activation_case(inputs, "q3k_situ_moe_graph", q3, q3_sink.get(), 2, q3_input, top6, w6, situ);
    q2_activation_test(inputs, threads, depth);

    // Invalid activations and up/gate type mismatches fail before any I/O.
    tensors meta(f, 1);
    std::vector<float> output(f.latent);
    const float nan = std::numeric_limits<float>::quiet_NaN(), inf = std::numeric_limits<float>::infinity();
    kimi_cpu_sink_activation unbounded = clamp;
    unbounded.swiglu_limit = inf;
    activation_case(inputs, "mxfp4_swiglu_unbounded", f, sink.get(), 1, input, top6, w6, unbounded);
    int refusals = 0;
    for (const auto & bad : { kimi_cpu_sink_activation { KIMI_CPU_SINK_ACTIVATION_SWIGLU_CLAMP, 0, 0, 0 },
                              kimi_cpu_sink_activation { KIMI_CPU_SINK_ACTIVATION_SWIGLU_CLAMP, 0, 0, 1e-7f },
                              kimi_cpu_sink_activation { KIMI_CPU_SINK_ACTIVATION_SWIGLU_CLAMP, 0, 0, -10 },
                              kimi_cpu_sink_activation { KIMI_CPU_SINK_ACTIVATION_SWIGLU_CLAMP, 0, 0, nan },
                              kimi_cpu_sink_activation { KIMI_CPU_SINK_ACTIVATION_SWIGLU_CLAMP, 0, 0, -inf },
                              kimi_cpu_sink_activation { KIMI_CPU_SINK_ACTIVATION_SITU, 0, 25, 0 },
                              kimi_cpu_sink_activation { KIMI_CPU_SINK_ACTIVATION_SITU, 4, nan, 0 },
                              kimi_cpu_sink_activation { 7, 4, 25, 10 } }) {
        kimi_cpu_sink_stats stats = {}; char error[512] = {};
        require(!kimi_cpu_sink_run_activation(sink.get(), meta.role[0], meta.role[1], meta.role[2], input.data(),
                top6.data(), w6.data(), 6, &bad, output.data(), &stats, error, sizeof(error)) && error[0] &&
                stats.read_requests == 0 && stats.fallback_count == 0, "invalid activation was accepted");
        ++refusals;
    }
    {
        kimi_cpu_sink_stats stats = {}; char error[512] = {};
        require(!kimi_cpu_sink_run_activation(sink.get(), meta.role[0], meta.role[1], meta.role[2], input.data(),
                top6.data(), w6.data(), 6, nullptr, output.data(), &stats, error, sizeof(error)) && error[0] &&
                stats.read_requests == 0, "null activation was accepted");
        ++refusals;
    }
    {
        ggml_context * ctx = ggml_init({ ggml_tensor_overhead() * 2, nullptr, true }); require(ctx, "metadata context failed");
        ggml_tensor * gate_q8 = ggml_new_tensor_3d(ctx, GGML_TYPE_Q8_0, f.latent, f.hidden, f.experts);
        ggml_set_name(gate_q8, tensor_name(1, 1).c_str());
        kimi_cpu_sink_stats stats = {}; char error[512] = {};
        const bool ok = kimi_cpu_sink_run_activation(sink.get(), meta.role[0], gate_q8, meta.role[2], input.data(),
                top6.data(), w6.data(), 6, &clamp, output.data(), &stats, error, sizeof(error));
        ggml_free(ctx);
        require(!ok && std::strstr(error, "one type") && stats.read_requests == 0, "mixed up/gate types were accepted");
        ++refusals;
    }
    {
        // Batched MXFP4 prefill uses the same clamped-SwiGLU semantics as the
        // independent graph reference for every token.
        char error[512] = {};
        kimi_cpu_prefill * prefill = kimi_cpu_prefill_create(sink.get(), error, sizeof(error));
        require(prefill, std::string("prefill create: ") + error);
        std::vector<float> batched_input = input;
        batched_input.insert(batched_input.end(), input.begin(), input.end());
        batched_input[f.latent + 3] += .25f;
        std::vector<int32_t> batched_ids = top6;
        batched_ids.insert(batched_ids.end(), top6.rbegin(), top6.rend());
        std::vector<float> batched_weights = w6;
        batched_weights.insert(batched_weights.end(), w6.rbegin(), w6.rend());
        auto expected = reference_moe_graph(f, 1, input, top6, w6, clamp, threads);
        std::vector<float> second_input(batched_input.begin() + f.latent, batched_input.end());
        std::vector<int32_t> second_ids(batched_ids.begin() + 6, batched_ids.end());
        std::vector<float> second_weights(batched_weights.begin() + 6, batched_weights.end());
        const auto second = reference_moe_graph(f, 1, second_input, second_ids, second_weights, clamp, threads);
        expected.insert(expected.end(), second.begin(), second.end());
        kimi_cpu_prefill_stats stats = {}; std::vector<float> routed(2 * f.latent);
        const bool ok = kimi_cpu_prefill_run_activation(prefill, meta.role[0], meta.role[1], meta.role[2],
                batched_input.data(), batched_ids.data(), batched_weights.data(), 6, 2, &clamp,
                routed.data(), &stats, error, sizeof(error));
        kimi_cpu_prefill_destroy(prefill);
        require(ok, std::string("MXFP4 grouped prefill: ") + error);
        require(!std::memcmp(expected.data(), routed.data(), routed.size() * sizeof(float)),
                "MXFP4 grouped prefill differs from the graph reference");
        require(stats.routed_slots == 12 && stats.stream.experts == 6 && stats.stream.read_requests == 18,
                "MXFP4 grouped prefill counters differ");
        std::printf("DSV41_MXFP4_CPU_PREFILL_OK tokens=2 experts=%llu\n",
                (unsigned long long) stats.stream.experts);
    }
    activation_case(inputs, "mxfp4_swiglu_clamp_after_refusals", f, sink.get(), 1, input, top6, w6, clamp10);
    std::printf("CPU_SINK_ACTIVATION_OK threads=%d ring_experts=%d refusals=%d\n", threads, depth, refusals);
}

// Native Muse expert semantics against the ordinary ggml MoE graph Muse builds (ggml_swiglu_split, then mul and
// rms_norm by the post norm), bit for bit. Different norm weights/epsilons and 3- and 65-token prefills catch
// normalization after the sum, missing norm weights, and the grouped slot-order sum.
static void muse_expert_norm_test(fixture_inputs & inputs, int threads, int depth, ggml_type type, bool native_ranges = false) {
    scoped_fixture_input ready(inputs, "ready_first", "1");
    fixture f; f.type = f.down_type = type; make_synthetic(f, 20261017);
    if (native_ranges) {
        // Rebuild a tiny native-weight source with mixed zero/128-byte page prefixes and
        // no EOF padding. Keep the independent reference payload unchanged.
        ssd::platform::close_fd(f.fd); f.fd = -1;
        for (auto it = f.entries.begin(); it != f.entries.end();) {
            if (it->first.first.rfind("blk.2.", 0) == 0) { it = f.entries.erase(it); }
            else { ++it; }
        }
        std::ofstream file(f.path, std::ios::binary | std::ios::trunc);
        uint64_t end = 0;
        for (auto & kv : f.entries) {
            kv.second.offset = aligned(size_t(end)) + (kv.first.second >= 10 ? 128 : 0);
            file.seekp(std::streamoff(kv.second.offset));
            file.write(reinterpret_cast<const char *>(kv.second.data.data()), std::streamsize(kv.second.data.size()));
            end = kv.second.offset + kv.second.data.size();
        }
        file.close();
        f.fd = test_resources::open_direct(f.path);
        f.native_ranges = true;
    }
    auto sink = make_sink(inputs, f, threads, depth);
    tensors meta(f, 1);
    std::vector<float> norm(f.latent), input(f.latent);
    for (int row = 0; row < f.latent; ++row) {
        norm[row] = 0.1f + float((row * 17) % 29) / 13.0f;
        input[row] = 0.17f * std::sin(float(row) * .31f);
    }
    const std::vector<int32_t> ids = { 9, 1, native_ranges ? 19 : 17, 6, 3, 12, 0, 15 };
    const std::vector<float> weights = { .21f, .19f, .15f, .13f, .11f, .09f, .07f, .05f };
    kimi_cpu_sink_activation activation = { KIMI_CPU_SINK_ACTIVATION_SWIGLU, 0.0f, 0.0f, 0.0f, norm.data(), 1e-8f };
    char error[512] = {};
    auto compare = [&](const std::vector<float> & actual, const std::vector<float> & expected) {
        require(actual.size() == expected.size(), "Muse output size");
        require(!std::memcmp(actual.data(), expected.data(), actual.size() * sizeof(float)),
            "Muse experts with their post norm differ from the graph");
        std::printf("MUSE_EXPERT_COMPARE type=%s native_ranges=%d values=%zu bit_exact=1 PASS\n", ggml_type_name(type), int(native_ranges), actual.size());
    };
    for (float eps : { 1e-8f, .2f }) {
        activation.post_exp_norm_eps = eps;
        const auto expected = reference_moe_graph(f, 1, input, ids, weights, activation, 1);
        std::vector<float> actual(f.latent);
        kimi_cpu_sink_stats stats = {};
        require(kimi_cpu_sink_run_activation(sink.get(), meta.role[0], meta.role[1], meta.role[2],
            input.data(), ids.data(), weights.data(), 8, &activation, actual.data(), &stats, error, sizeof(error)), error);
        compare(actual, expected);
        require(stats.experts == 8 && stats.read_requests == 24 && stats.pending_reads == 0 && !stats.read_errors,
            "Muse Q8 decode I/O counters");
    }
    auto * prefill = kimi_cpu_prefill_create(sink.get(), error, sizeof(error)); require(prefill, error);
    // both epsilons: one far above the mean square, which hides how the squares are rounded
    for (float eps : { 1e-8f, .2f }) {
        activation.post_exp_norm_eps = eps;
        for (int tokens : { 3, 65 }) {
            std::vector<float> x, w, expected;
            std::vector<int32_t> selected;
            for (int token = 0; token < tokens; ++token) {
                auto xt = input; auto it = ids;
                for (auto & value : xt) { value += float(token) * .003f; }
                for (auto & id : it) { id = (id + token) % f.experts; }
                auto y = reference_moe_graph(f, 1, xt, it, weights, activation, 1);
                x.insert(x.end(), xt.begin(), xt.end()); selected.insert(selected.end(), it.begin(), it.end());
                w.insert(w.end(), weights.begin(), weights.end()); expected.insert(expected.end(), y.begin(), y.end());
            }
            std::vector<float> actual(expected.size()); kimi_cpu_prefill_stats stats = {};
            require(kimi_cpu_prefill_run_activation(prefill, meta.role[0], meta.role[1], meta.role[2],
                x.data(), selected.data(), w.data(), 8, tokens, &activation, actual.data(), &stats, error, sizeof(error)), error);
            compare(actual, expected);
            require(stats.stream.pending_reads == 0 && !stats.stream.read_errors && !stats.stream.fallback_count,
                "Muse Q8 prefill I/O counters");
        }
    }
    kimi_cpu_prefill_destroy(prefill);
    if (native_ranges) {
        // Native aligned-read payloads must also survive cache admission and
        // full hits, not just direct ring consumption.
        configure_cache(sink.get(), 8 * 3 * aligned(128 + f.entries.begin()->second.raw_bytes), {1}, 8);
        for (uint64_t token : {1, 2}) {
            begin_cache(sink.get(), 91, token);
            std::vector<float> actual(f.latent); kimi_cpu_sink_stats stats = {};
            require(kimi_cpu_sink_run_activation(sink.get(), meta.role[0], meta.role[1], meta.role[2],
                input.data(), ids.data(), weights.data(), 8, &activation, actual.data(), &stats, error, sizeof(error)), error);
            compare(actual, reference_moe_graph(f, 1, input, ids, weights, activation, 1));
            if (token == 2) { require(stats.cache_hits == 8 && stats.read_requests == 0, "native offset cache hit lost its payload prefix"); }
        }
        {
            scoped_fixture_input persistent(inputs, "cache_policy", "lru");
            auto varied = make_sink(inputs, f, threads, depth);
            configure_cache(varied.get(), 16 * 3 * aligned(128 + f.entries.begin()->second.raw_bytes), {1}, 16);
            const std::vector<int32_t> aligned_ids = { 0, 1, 2, 3, 4, 5, 6, 7 };
            for (uint64_t token : {1, 2, 3}) {
                begin_cache(varied.get(), 92, token);
                const auto & selected = token == 2 ? ids : aligned_ids;
                std::vector<float> actual(f.latent); kimi_cpu_sink_stats stats = {};
                require(kimi_cpu_sink_run_activation(varied.get(), meta.role[0], meta.role[1], meta.role[2],
                    input.data(), selected.data(), weights.data(), 8, &activation, actual.data(), &stats, error, sizeof(error)), error);
                compare(actual, reference_moe_graph(f, 1, input, selected, weights, activation, 1));
                if (token == 3) { require(stats.cache_hits == 8 && stats.read_requests == 0,
                    "page-prefix variation cleared native cached payloads"); }
            }
        }
        // EOF padding may be absent, but one missing logical byte must fail. Windows shares a direct
        // descriptor for reading only, so the file is truncated between two opens.
        ssd::platform::close_fd(f.fd); f.fd = -1;
        std::filesystem::resize_file(f.path, std::filesystem::file_size(f.path) - 1);
        f.fd = test_resources::open_direct(f.path);
        auto truncated = make_sink(inputs, f, threads, depth);
        std::vector<float> actual(f.latent); kimi_cpu_sink_stats stats = {};
        require(!kimi_cpu_sink_run_activation(truncated.get(), meta.role[0], meta.role[1], meta.role[2],
            input.data(), ids.data(), weights.data(), 8, &activation, actual.data(), &stats, error, sizeof(error)) &&
            stats.read_errors > 0 && stats.pending_reads == 0, "native truncated logical payload was accepted");
    }
    std::printf("MUSE_EXPERT_POST_NORM_OK type=%s native_ranges=%d\n", ggml_type_name(type), int(native_ranges));
}

#if defined(__linux__)
// Linux only (the native ledger has test-ssd-host-budget): the host-memory ledger (ledger.h) with injected /proc and
// cgroup files: which term binds where, the open prefill's pages, the watermark boost and the refault-driven file
// reserve. The arbiter's reclaim order and margin learning are test-ssd-host-arbiter's.
static void host_ledger_tests(fixture_inputs & inputs) {
    constexpr uint64_t MiB = 1ull << 20, GiB = 1ull << 30;
    scoped_fixture_input margin_env(inputs, "cache_reserve_bytes", "536870912");
    scoped_fixture_input file_env(inputs, "cache_reclaimable_file", nullptr);
    scoped_fixture_input host_env(inputs, "host_reserve_mib", nullptr);
    scoped_fixture_input cuda_env(inputs, "device_reserve_mib", nullptr);
    scoped_fixture_input headroom_env(inputs, "host_headroom_mib", nullptr);
    const std::string scope = "/sys/fs/cgroup/user.slice/run-1.scope";
    uint64_t limit = 8 * GiB, current = 0, available = 0, file_pages = 0, refaults = 0;
    bool meminfo = true;
    auto files = [&]() {
        std::map<std::string, std::string> f;
        f["/proc/self/cgroup"] = "0::/user.slice/run-1.scope\n";
        f["/sys/fs/cgroup/user.slice/memory.max"] = "max\n";
        f[scope + "/memory.max"] = limit ? std::to_string(limit) + "\n" : "max\n";
        f[scope + "/memory.current"] = std::to_string(current) + "\n";
        f[scope + "/memory.stat"] = "anon 4096\nfile " + std::to_string(file_pages) + "\nfile_mapped 0\nactive_file 0\ninactive_file " +
            std::to_string(file_pages) + "\nworkingset_refault_anon 3\nworkingset_refault_file " + std::to_string(refaults) + "\n";
        if (meminfo) {
            f["/proc/meminfo"] = "MemTotal:       127000000 kB\nMemFree:          100000 kB\nMemAvailable:   " +
                std::to_string(available / 1024) + " kB\nBuffers:            1000 kB\n";
        }
        return f;
    };
    auto make = [&](bool unified) {
        kimi_host_ledger ledger;
        ledger.read = [&](const std::string & path, std::string & text) {
            const auto f = files();
            const auto it = f.find(path);
            if (it == f.end()) { return false; }
            text = it->second;
            return true;
        };
        inputs.configure(ledger, unified);
        return ledger;
    };
    // A discrete GPU: the cgroup term binds.
    available = 100 * GiB; current = 7 * GiB + 768 * MiB;
    auto discrete = make(false);
    require(discrete.cgroup == scope && discrete.reserve == 512 * MiB && discrete.margin == 512 * MiB &&
            discrete.explicit_reserve == 0 && discrete.floor() == 512 * MiB && !discrete.reclaimable_file,
            "discrete ledger configuration");
    auto room = discrete.measure(256 * MiB);
    require(room.cgroup == ssd::host::room_pressure && room.cgroup_shortage == 512 * MiB && room.system &&
            !room.system_shortage && room.surplus() == 0, "discrete: the cgroup term did not bind");
    require(!discrete.update_file_reserve() && discrete.file_reserve == 0, "discrete: a file reserve appeared");
    current = 1 * GiB;
    room = discrete.measure(0);
    require(room.fits() && room.surplus() == 8 * GiB - 1 * GiB - 512 * MiB, "discrete: surplus is not the cgroup's");

    // Unified memory: the GPU tier is invisible to the cgroup, so the system term binds.
    limit = 100 * GiB; current = 1 * GiB; file_pages = 300 * MiB; refaults = 10;
    available = 2 * GiB + 256 * MiB;
    auto uma = make(true);
    require(uma.update_file_reserve() && uma.file_reserve == 300 * MiB && uma.floor() == 812 * MiB,
            "unified: the file reserve was not measured");
    const uint64_t checkpoint = 1930 * MiB;
    room = uma.measure(checkpoint);
    require(room.cgroup == ssd::host::room_ok && room.system_shortage == 812 * MiB + checkpoint - available,
            "unified: the system term did not bind");

    // The file reserve only grows: one page per refault since the last measurement.
    refaults = 110; file_pages = 100 * MiB;
    require(uma.update_file_reserve() && uma.file_reserve == 300 * MiB + 100 * uma.page_size, "refaults did not grow the file reserve");
    require(!uma.update_file_reserve() && uma.file_reserve == 300 * MiB + 100 * uma.page_size, "the file reserve moved without refaults");

    // Unreadable /proc/meminfo: the system term does not bind. No finite cgroup: the cgroup term is unreadable.
    meminfo = false;
    room = uma.measure(checkpoint);
    require(!room.system && room.fits(), "an unreadable MemAvailable bound");
    meminfo = true; limit = 0; available = 100 * GiB;
    room = uma.measure(checkpoint);
    require(room.cgroup == ssd::host::room_unreadable && room.system && room.fits(),
            "an unbounded cgroup was not unreadable");

    // The cgroup term keeps the cgroup reserve; CACHE_RECLAIMABLE_FILE (read by every ledger) counts file pages as its
    // headroom.
    {
        scoped_fixture_input headroom(inputs, "host_headroom_mib", "4096");
        limit = 8 * GiB; current = 6 * GiB; available = 100 * GiB; file_pages = 0;
        auto u = make(true);
        room = u.measure(1 * GiB);
        require(u.margin == 4 * GiB && u.reserve == 512 * MiB && room.cgroup == ssd::host::room_ok && room.fits(),
                "unified: the cgroup term held the floor's margin");
        current = 7 * GiB + 512 * MiB; file_pages = 2 * GiB;
        require(!make(false).measure(1 * GiB).fits(), "file pages counted as cgroup headroom without the switch");
        scoped_fixture_input file(inputs, "cache_reclaimable_file", "1");
        auto f = make(false);
        room = f.measure(1 * GiB);
        require(f.reclaimable_file && room.fits() && room.cgroup_surplus == 8 * GiB - (current - file_pages) - 512 * MiB - 1 * GiB,
                "the ledger did not read CACHE_RECLAIMABLE_FILE");
        scoped_fixture_input bad(inputs, "cache_reclaimable_file", "2");
        bool rejected = false;
        try { (void) make(false); } catch (const std::exception &) { rejected = true; }
        require(rejected, "an invalid CACHE_RECLAIMABLE_FILE was accepted");
        current = 0; file_pages = 0;
    }

    // An open prefill's pages (the weight rows its graph gathers from the file mapping) are held in both terms from its
    // start to the first decode token: with CACHE_RECLAIMABLE_FILE they count beside the reserve whether they have
    // come in or not; without it the file pages up to them are theirs, not more use; MemAvailable needs them beside
    // the floor; on unified memory the tier keeps them free beside the allocation.
    {
        limit = 8 * GiB; available = 100 * GiB;
        {
            scoped_fixture_input file(inputs, "cache_reclaimable_file", "1");
            auto f = make(false);
            current = 6 * GiB; file_pages = 1 * GiB; // 5 GiB of anonymous and kernel memory, 1 GiB of old file pages
            room = f.measure(0);
            require(room.fits() && room.cgroup_surplus == 8 * GiB - 5 * GiB - 512 * MiB, "prefill pages: the reference room");
            f.prefill_pages = 1 * GiB;
            room = f.measure(2 * GiB);
            require(!room.fits() && room.cgroup_shortage == 512 * MiB && !room.system_shortage,
                    "prefill pages: the cgroup term did not hold the prefill's pages beside the reserve");
            current += 1 * GiB; file_pages += 1 * GiB; // they came in
            room = f.measure(2 * GiB);
            require(!room.fits() && room.cgroup_shortage == 512 * MiB, "prefill pages: counted twice once they came in");
            room = f.measure(1 * GiB + 512 * MiB);
            require(room.fits() && room.cgroup_surplus == 0, "prefill pages: the room beside them");
            f.prefill_pages = 0;
            require(f.measure(2 * GiB).fits(), "prefill pages: still held after the hold was cleared");
        }
        {
            auto n = make(false); // file pages are use
            current = 6 * GiB; file_pages = 0;
            n.prefill_pages = 1 * GiB;
            room = n.measure(0);
            require(room.fits() && room.cgroup_surplus == 8 * GiB - 6 * GiB - 512 * MiB - 1 * GiB,
                    "prefill pages: the cgroup term without reclaimable file pages");
            current += 1 * GiB; file_pages = 1 * GiB; // they came in: theirs, not more use
            room = n.measure(0);
            require(room.fits() && room.cgroup_surplus == 8 * GiB - 6 * GiB - 512 * MiB - 1 * GiB,
                    "prefill pages: counted twice without reclaimable file pages");
            current += 1 * GiB; file_pages = 2 * GiB; // file pages beyond them are use
            room = n.measure(0);
            require(!room.fits() && room.cgroup_shortage == 512 * MiB, "prefill pages: file pages beyond them were not use");
            n.prefill_pages = 0;
            current = 0; file_pages = 0;
        }
        {
            limit = 0; available = 1 * GiB + 512 * MiB; // the system term alone: the floor, 1 GiB and 256 MiB
            auto d = make(false);
            d.prefill_pages = 1 * GiB;
            room = d.measure(256 * MiB);
            require(room.cgroup == ssd::host::room_unreadable && room.system_shortage == 256 * MiB,
                    "prefill pages: MemAvailable did not hold them beside the floor");
            limit = 8 * GiB; available = 100 * GiB; current = 0; file_pages = 0; refaults = 0;
        }
    }

    // MemAvailable at the kernel's full watermark boost (kimi-memory-boost.h). The zones of a DGX Spark (GB10, 4 KiB
    // pages, 2 MiB pageblocks): DMA low 235 / high 409, Normal low 42934 / high 74638 pages without the boost, whose
    // largest boost is high * 15000 / 10000 = 111957 pages (437 MiB). /proc/zoneinfo prints low and high with the
    // boost added; a per-CPU pageset's "high:" is not the zone's.
    {
        const uint64_t page = 4096;
        auto zones = [](uint64_t dma_boost, uint64_t normal_boost, bool boost_lines = true) {
            auto zone = [&](const char * name, uint64_t boost, uint64_t low, uint64_t high, uint64_t managed) {
                std::string z = std::string("Node 0, zone ") + name + "\n  pages free     1000\n";
                if (boost_lines) { z += "        boost    " + std::to_string(boost) + "\n"; }
                z += "        min      61\n        low      " + std::to_string(low + boost) + "\n        high     " +
                    std::to_string(high + boost) + "\n        spanned  524288\n        managed  " + std::to_string(managed) +
                    "\n  pagesets\n    cpu: 0\n              count: 12\n              high:  999999\n              batch: 63\n";
                return z;
            };
            return zone("DMA", dma_boost, 235, 409, 174798) + zone("DMA32", 0, 0, 0, 0) +
                zone("Normal", normal_boost, 42934, 74638, 31726918) + zone("Movable", 0, 32, 32, 0);
        };
        const std::string meminfo_text = "MemTotal:       127606864 kB\nMemFree:        1000000 kB\nMemAvailable:   5000000 kB\n"
            "Active(file):    1048576 kB\nInactive(file):  3145728 kB\nSReclaimable:     512000 kB\nKReclaimable:     512000 kB\n"
            "HugePages_Total:       0\nHugepagesize:       2048 kB\n";
        // No boost: low sums to 43201 pages, and to 155771 at the largest boosts (DMA max(512, 613), Normal 111957).
        // The file LRU's half (524288 pages) keeps back the whole difference, the reclaimable half (64000) 20799 more.
        require(kimi_memory_boost::headroom(meminfo_text, zones(0, 0), "15000\n", page) == (112570 + 20799) * page,
                "the boost headroom without a boost");
        // Normal at its largest boost: only the DMA zone's can still grow.
        require(kimi_memory_boost::headroom(meminfo_text, zones(0, 111957), "15000\n", page) == 613 * page,
                "the boost headroom at the Normal zone's largest boost");
        require(kimi_memory_boost::headroom(meminfo_text, zones(613, 111957), "15000\n", page) == 0,
                "the boost headroom at every zone's largest boost");
        require(kimi_memory_boost::headroom(meminfo_text, zones(0, 0), "0\n", page) == 0, "a zero boost factor kept memory back");
        require(kimi_memory_boost::headroom(meminfo_text, zones(0, 0, false), "15000\n", page) == 0,
                "a kernel without watermark boosts kept memory back");
        require(kimi_memory_boost::headroom(meminfo_text, "", "15000\n", page) == 0 &&
                kimi_memory_boost::headroom("MemAvailable: 5000000 kB\n", zones(0, 0), "15000\n", page) == 0,
                "unreadable inputs kept memory back");
        std::string hugetlb = meminfo_text;
        hugetlb.replace(hugetlb.find("HugePages_Total:       0"), 24, "HugePages_Total:      64");
        require(kimi_memory_boost::headroom(hugetlb, zones(0, 0), "15000\n", page) == 0, "a hugetlbfs pool kept memory back");

        // The ledger on unified memory: at the full boost for what an allocation outside the caches takes (the
        // arbiter's take and its margin's intervals), as it is for the RAM cache's guard; a discrete GPU ignores it.
        uint64_t dma_boost = 0, normal_boost = 0;
        const uint64_t full = (112570 + 20799) * page;
        auto boosted = [&](bool unified) {
            kimi_host_ledger ledger;
            ledger.read = [&](const std::string & path, std::string & text) {
                auto f = files();
                f["/proc/meminfo"] = "MemTotal:       127000000 kB\nMemFree:          100000 kB\nMemAvailable:   " +
                    std::to_string(available / 1024) + " kB\nActive(file):    1048576 kB\nInactive(file):  3145728 kB\n"
                    "KReclaimable:     512000 kB\nHugePages_Total:       0\nHugepagesize:       2048 kB\n";
                f["/proc/zoneinfo"] = zones(dma_boost, normal_boost);
                f["/proc/sys/vm/watermark_boost_factor"] = "15000\n";
                const auto it = f.find(path);
                if (it == f.end()) { return false; }
                text = it->second;
                return true;
            };
            inputs.configure(ledger, unified);
            return ledger;
        };
        limit = 100 * GiB; current = 1 * GiB; file_pages = 300 * MiB; refaults = 10;
        auto u = boosted(true);
        require(u.update_file_reserve() && u.floor() == 812 * MiB, "boost: the file reserve was not measured");
        available = 812 * MiB + checkpoint + 64 * MiB; // fits MemAvailable as it is, not at the full boost
        uint64_t at_full = 0, headroom = 0;
        require(u.memory_available_at_full_boost(at_full, headroom) && headroom == full && at_full == available - full,
                "boost: MemAvailable at the full boost");
        room = u.measure(checkpoint);
        require(room.fits() && room.boost_headroom == 0, "boost: the RAM cache's guard counted the boost");
        room = u.measure(checkpoint, true);
        require(!room.fits() && room.boost_headroom == full && room.system_shortage == full - 64 * MiB,
                "boost: an allocation outside the caches did not count the boost");
        // A discrete GPU: no headroom anywhere.
        auto d = boosted(false);
        require(d.memory_available_at_full_boost(at_full, headroom) && headroom == 0 && at_full == available &&
                d.measure(checkpoint, true).boost_headroom == 0, "boost: a discrete GPU counted the boost");
        current = 0; file_pages = 0; refaults = 0;
    }

    std::printf("HOST_LEDGER_OK\n");
}

// Linux only: counts /proc/self/task and forks a child under RLIMIT_FSIZE.
#include "configured-runtime-gate.h"
#endif

// The configured executor without a GPU provider, as on macOS, Windows and CPU-only Linux: vram.budget auto is
// no GPU tier, and only an explicit nonzero budget is refused.
static void tierless_budget_tests() {
    fixture f;
    auto owner = std::make_shared<ssd::host::process_arbiter>();
    auto client = owner->make_client();
    require(bool(client), "host client registration");
    const ssd::runtime_services services{ nullptr, nullptr, client.value.get() };
    for (const bool automatic : { true, false }) {
        const auto settings = ssd::settings::parse("", "", { { "vram.budget", automatic ? "auto" : "1048576" } });
        require(bool(settings), settings.error);
        char error[512] = {};
        sink_ptr sink(ssd::cpu::create_configured(&settings.value, &services, fixture::resolve, &f, error, sizeof(error)));
        if (automatic) {
            require(bool(sink), std::string("vram.budget auto without a GPU provider: ") + error);
        } else {
            require(!sink && std::strstr(error, "requires an injected GPU provider"),
                "an explicit GPU tier budget without a provider was accepted");
        }
    }
    std::printf("TIERLESS_BUDGET_OK\n");
}

// The production shape of a call: graphs of sink nodes that every thread of a ggml threadpool enters, ith 0 as the
// coordinator that opens, runs and closes the call and the others as its participants, as
// llama_kimi_cpu_sink_compute does. Chained decode layers with a dense op between them, a grouped prefill and a run
// that fails inside its call, at 1, 2, an odd count and all CPUs, with every decode participant and with a fixed
// first few, bit for bit against the ggml reference graphs.
struct graph_layer {
    kimi_cpu_sink * sink = nullptr;
    kimi_cpu_prefill * prefill = nullptr;
    const tensors * meta = nullptr;
    const kimi_cpu_sink_activation * activation = nullptr; // SiTU (4, 25) when absent
    bool fail = false;
    bool ok = false;
    int nth = 0; // the participants ggml gave the call
    std::string error;
};

static void graph_layer_compute(ggml_tensor * dst, int ith, int nth, void * userdata) {
    auto & layer = *static_cast<graph_layer *>(userdata);
    if (ith) { kimi_cpu_sink_call_join(layer.sink, ith, nth); return; }
    char error[512] = {};
    if (!kimi_cpu_sink_call_open(layer.sink, nth, error, sizeof(error))) {
        std::fprintf(stderr, "graph call open: %s\n", error);
        std::abort();
    }
    layer.nth = nth;
    const ggml_tensor * x = dst->src[0], * ids = dst->src[1], * weights = dst->src[2];
    const int ntokens = int(dst->ne[1]), nslots = int(ggml_nelements(ids) / ntokens);
    const auto * id = static_cast<const int32_t *>(ids->data);
    const auto * w = static_cast<const float *>(weights->data);
    auto * out = static_cast<float *>(dst->data);
    const auto * in = static_cast<const float *>(x->data);
    const auto & m = *layer.meta;
    if (layer.prefill) {
        layer.ok = layer.activation ?
            kimi_cpu_prefill_run_activation(layer.prefill, m.role[0], m.role[1], m.role[2], in, id, w, nslots, ntokens,
                layer.activation, out, nullptr, error, sizeof(error)) :
            kimi_cpu_prefill_run(layer.prefill, m.role[0], m.role[1], m.role[2], in, id, w, nslots, ntokens,
                4.0f, 25.0f, out, nullptr, error, sizeof(error));
    } else {
        // an invalid slot count fails after the call opened: the joiners must still leave
        const int slots = layer.fail ? -1 : nslots;
        layer.ok = layer.activation ?
            kimi_cpu_sink_run_activation(layer.sink, m.role[0], m.role[1], m.role[2], in, id, w, slots,
                layer.activation, out, nullptr, error, sizeof(error)) :
            kimi_cpu_sink_run(layer.sink, m.role[0], m.role[1], m.role[2], in, id, w, slots,
                4.0f, 25.0f, out, nullptr, error, sizeof(error));
    }
    layer.error = error;
    kimi_cpu_sink_call_close(layer.sink);
}

static void threadpool_tests(fixture_inputs & inputs, int depth) {
    fixture f; make_synthetic(f);
    tensors meta1(f, 1), meta2(f, 2);
    const int cpus = int(test_resources::process_cpus().size());
    std::vector<int> counts = { 1, 2, 3 };
    if (cpus > 3) { counts.push_back(cpus); }
    const std::vector<int32_t> ids1 = { 17, 0, 8, 2, 19, 4, 11, 6 }, ids2 = { 3, 9, -1, 13, 5, 3, 1, 12 };
    const std::vector<float> w1 = { .2f, .1f, .15f, .05f, .1f, .1f, .2f, .1f }, w2 = { .3f, .1f, 0, .2f, .1f, .1f, .1f, .1f };
    std::vector<float> x(f.latent);
    for (int i = 0; i < f.latent; ++i) { x[i] = std::sin(float(i) * .37f) * .3f; }
    // decode: layer 1, x0.5, layer 2, layer 1 (three calls a graph); then the reference chain
    auto expected = reference(f, 1, x, ids1, w1, 4, 25);
    for (auto & v : expected) { v *= .5f; }
    expected = reference(f, 2, expected, ids2, w2, 4, 25);
    expected = reference(f, 1, expected, ids1, w1, 4, 25);
    constexpr int ntokens = 5, nslots = 4;
    std::vector<float> xs(size_t(ntokens) * f.latent);
    std::vector<int32_t> prefill_ids(size_t(ntokens) * nslots);
    std::vector<float> prefill_weights(prefill_ids.size()), prefill_expected;
    for (size_t i = 0; i < xs.size(); ++i) { xs[i] = std::cos(float(i) * .11f) * .25f; }
    for (int t = 0; t < ntokens; ++t) {
        for (int s = 0; s < nslots; ++s) {
            prefill_ids[size_t(t) * nslots + s] = (t * 5 + s * 7 + 1) % f.experts;
            prefill_weights[size_t(t) * nslots + s] = float(t + s + 1) / 17.0f;
        }
        const auto y = reference(f, 2, { xs.begin() + size_t(t) * f.latent, xs.begin() + size_t(t + 1) * f.latent },
            { prefill_ids.begin() + size_t(t) * nslots, prefill_ids.begin() + size_t(t + 1) * nslots },
            { prefill_weights.begin() + size_t(t) * nslots, prefill_weights.begin() + size_t(t + 1) * nslots }, 4, 25);
        prefill_expected.insert(prefill_expected.end(), y.begin(), y.end());
    }
    char error[512] = {};
    for (int fixed : { 0, 2 }) {
        scoped_fixture_input participants(inputs, "decode_participants", fixed ? std::to_string(fixed).c_str() : nullptr);
        // one participant only: the sink computes on the callers' threads
        sink_ptr sink = inputs.create(2, depth, fixture::resolve, &f, error, sizeof(error));
        require(bool(sink), std::string("threadpool create: ") + error);
        std::unique_ptr<kimi_cpu_prefill, decltype(&kimi_cpu_prefill_destroy)> prefill(
            kimi_cpu_prefill_create(sink.get(), error, sizeof(error)), kimi_cpu_prefill_destroy);
        require(bool(prefill), std::string("threadpool prefill create: ") + error);
        for (int n : counts) {
            if (fixed > n) { continue; }
            ggml_threadpool_params params = ggml_threadpool_params_default(n);
            std::unique_ptr<ggml_threadpool, decltype(&ggml_threadpool_free)> pool(ggml_threadpool_new(&params),
                ggml_threadpool_free);
            require(bool(pool), "ggml threadpool");
            auto compute = [&](ggml_cgraph * graph) {
                ggml_cplan plan = ggml_graph_plan(graph, n, pool.get());
                std::vector<uint8_t> work(plan.work_size);
                plan.work_data = work.data();
                require(ggml_graph_compute(graph, &plan) == GGML_STATUS_SUCCESS, "threadpool graph compute");
            };
            const size_t bytes = 64 * ggml_tensor_overhead() + ggml_graph_overhead() + (size_t(4) << 20);
            for (int repeat = 0; repeat < 3; ++repeat) {
                ggml_context * ctx = ggml_init({ bytes, nullptr, false });
                require(ctx, "threadpool graph context");
                auto vector = [&](ggml_type type, int64_t n0, int64_t n1, const void * data) {
                    ggml_tensor * t = ggml_new_tensor_2d(ctx, type, n0, n1);
                    std::memcpy(t->data, data, ggml_nbytes(t));
                    return t;
                };
                graph_layer layers[4];
                ggml_tensor * i1 = vector(GGML_TYPE_I32, 8, 1, ids1.data()), * i2 = vector(GGML_TYPE_I32, 8, 1, ids2.data());
                ggml_tensor * v1 = vector(GGML_TYPE_F32, 8, 1, w1.data()), * v2 = vector(GGML_TYPE_F32, 8, 1, w2.data());
                auto node = [&](graph_layer & layer, const tensors & m, ggml_tensor * in, ggml_tensor * id, ggml_tensor * w) {
                    layer.sink = sink.get(); layer.meta = &m;
                    ggml_tensor * args[] = { in, id, w };
                    return ggml_custom_4d(ctx, GGML_TYPE_F32, f.latent, in->ne[1], 1, 1, args, 3, graph_layer_compute,
                        GGML_N_TASKS_MAX, &layer);
                };
                ggml_cgraph * graph = ggml_new_graph(ctx);
                ggml_tensor * y = node(layers[0], meta1, vector(GGML_TYPE_F32, f.latent, 1, x.data()), i1, v1);
                y = ggml_scale(ctx, y, .5f);
                // a run that fails between successful ones (its own output): no participant waits for it, and the
                // next call still runs
                const bool failing = repeat == 1;
                if (failing) {
                    layers[3].fail = true;
                    ggml_build_forward_expand(graph, node(layers[3], meta2, y, i2, v2));
                }
                y = node(layers[1], meta2, y, i2, v2);
                y = node(layers[2], meta1, y, i1, v1);
                ggml_build_forward_expand(graph, y);
                compute(graph);
                for (int l = 0; l < 3; ++l) {
                    require(layers[l].ok, "threadpool decode layer: " + layers[l].error);
                    require(layers[l].nth == n, "threadpool decode call participants " + std::to_string(layers[l].nth));
                }
                require(!failing || (!layers[3].ok && !layers[3].error.empty()), "threadpool accepted an invalid run");
                require(!std::memcmp(y->data, expected.data(), expected.size() * sizeof(float)),
                    "threadpool decode differs from the reference graph threads=" + std::to_string(n));
                ggml_free(ctx);
                ctx = ggml_init({ bytes, nullptr, false });
                require(ctx, "threadpool prefill context");
                graph_layer grouped;
                grouped.prefill = prefill.get();
                ggml_tensor * p = node(grouped, meta2, vector(GGML_TYPE_F32, f.latent, ntokens, xs.data()),
                    vector(GGML_TYPE_I32, nslots, ntokens, prefill_ids.data()),
                    vector(GGML_TYPE_F32, nslots, ntokens, prefill_weights.data()));
                graph = ggml_new_graph(ctx);
                ggml_build_forward_expand(graph, p);
                compute(graph);
                require(grouped.ok, "threadpool prefill: " + grouped.error);
                require(grouped.nth == n, "threadpool prefill call participants " + std::to_string(grouped.nth));
                require(!std::memcmp(p->data, prefill_expected.data(), prefill_expected.size() * sizeof(float)),
                    "threadpool prefill differs from the reference graph threads=" + std::to_string(n));
                ggml_free(ctx);
            }
            std::printf("case=threadpool threads=%d decode_participants=%d bit_exact=1 status=PASS\n", n, fixed);
        }
    }
    std::printf("CPU_SINK_THREADPOOL_OK counts=%zu\n", counts.size());
}

// build_moe_ffn's CPU graph for SiLU-gated experts over ntokens: MUL_MAT_ID gate and up, ggml_swiglu_split,
// MUL_MAT_ID down, MUL by the routing weights and ADD of per-slot views in slot order. Its expert tensors hold the
// routed experts only, renumbered: no value the graph computes for one expert depends on another. A -1 route is
// MUL_MAT_ID's zero row. It shares no code with the sink.
static std::vector<float> reference_swiglu_graph(const fixture & f, int layer, const std::vector<float> & input,
        const std::vector<int32_t> & ids, const std::vector<float> & weights, int ntokens, int threads) {
    const int nslots = int(ids.size()) / ntokens;
    require(nslots > 0 && ids.size() == size_t(nslots) * ntokens && weights.size() == ids.size() &&
            input.size() == size_t(ntokens) * f.latent, "SwiGLU reference shape");
    std::map<int32_t, int32_t> compact;
    std::vector<int32_t> routed, selected(ids.size());
    for (size_t i = 0; i < ids.size(); ++i) {
        if (ids[i] < 0) { selected[i] = -1; continue; }
        const auto found = compact.emplace(ids[i], int32_t(routed.size()));
        if (found.second) { routed.push_back(ids[i]); }
        selected[i] = found.first->second;
    }
    require(!routed.empty(), "SwiGLU reference requires a live route");
    const size_t max_nodes = 32 + 2 * size_t(nslots);
    ggml_context * ctx = ggml_init({ ggml_tensor_overhead() * max_nodes + ggml_graph_overhead_custom(max_nodes, false), nullptr, true });
    require(ctx, "SwiGLU reference context failed");
    ggml_backend_t backend = ggml_backend_cpu_init(); require(backend, "SwiGLU reference CPU backend failed");
    ggml_backend_cpu_set_n_threads(backend, threads);
    ggml_tensor * experts[3];
    for (int role = 0; role < 3; ++role) {
        experts[role] = ggml_new_tensor_3d(ctx, role == 2 ? f.down_type : f.type, role == 2 ? f.hidden : f.latent,
                                          role == 2 ? f.latent : f.hidden, int64_t(routed.size()));
    }
    ggml_tensor * x = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, f.latent, 1, ntokens);
    ggml_tensor * selection = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, nslots, ntokens);
    ggml_tensor * routing = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, nslots, ntokens);
    ggml_tensor * gate = ggml_mul_mat_id(ctx, experts[1], x, selection);
    ggml_tensor * up = ggml_mul_mat_id(ctx, experts[0], x, selection);
    ggml_tensor * down = ggml_mul_mat_id(ctx, experts[2], ggml_swiglu_split(ctx, gate, up), selection);
    ggml_tensor * weighted = ggml_mul(ctx, down, routing);
    ggml_cgraph * graph = ggml_new_graph_custom(ctx, max_nodes, false);
    ggml_build_forward_expand(graph, weighted);
    ggml_tensor * result = nullptr;
    for (int slot = 0; slot < nslots; ++slot) {
        ggml_tensor * view = ggml_view_2d(ctx, weighted, f.latent, ntokens, weighted->nb[2], size_t(slot) * weighted->nb[1]);
        ggml_build_forward_expand(graph, view);
        result = result ? ggml_add(ctx, result, view) : view;
    }
    if (nslots == 1) { result = ggml_cont(ctx, result); }
    ggml_build_forward_expand(graph, result);
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend); require(buffer, "SwiGLU reference allocation failed");
    for (int role = 0; role < 3; ++role) {
        for (size_t e = 0; e < routed.size(); ++e) {
            const auto & item = f.entries.at({ tensor_name(layer, role), routed[e] });
            require(item.raw_bytes == experts[role]->nb[2], "SwiGLU reference expert stride mismatch");
            ggml_backend_tensor_set(experts[role], item.data.data(), e * experts[role]->nb[2], item.raw_bytes);
        }
    }
    ggml_backend_tensor_set(x, input.data(), 0, input.size() * sizeof(float));
    ggml_backend_tensor_set(selection, selected.data(), 0, selected.size() * sizeof(int32_t));
    ggml_backend_tensor_set(routing, weights.data(), 0, weights.size() * sizeof(float));
    require(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS, "SwiGLU reference CPU graph failed");
    std::vector<float> output(size_t(ntokens) * f.latent);
    ggml_backend_tensor_get(result, output.data(), 0, output.size() * sizeof(float));
    ggml_backend_buffer_free(buffer); ggml_backend_free(backend); ggml_free(ctx);
    return output;
}

static bool same_bits(const std::vector<float> & a, const std::vector<float> & b) {
    return a.size() == b.size() && !std::memcmp(a.data(), b.data(), a.size() * sizeof(float));
}

// SiLU-gated experts as qwen4exp routes them (separate up/gate/down, softmax top-10, sum-normalized weights)
// through sink nodes on a ggml threadpool of 1, 2, an odd count and all CPUs, against reference_swiglu_graph,
// compared with memcmp: two chained decode layers with a dense op between, and grouped prefills. Shapes: the model's
// (Q8_0, 2560 -> 640 -> 2560, 512 experts, of which the fixture holds 12), a narrow Q8_0 one for prefills of 2 to
// 2048 tokens, and a BF16 one whose hidden width of 70 leaves a scalar tail at every fixed vector width.
static void swiglu_tests(fixture_inputs & inputs, int depth) {
    const kimi_cpu_sink_activation swiglu = { KIMI_CPU_SINK_ACTIVATION_SWIGLU, 0.0f, 0.0f, 0.0f, nullptr, 0.0f };
    const int cpus = int(test_resources::process_cpus().size());
    std::vector<int> counts = { 1, 2, 3 };
    if (cpus > 3) { counts.push_back(cpus); }
    constexpr int nslots = 10;
    struct shape {
        const char * name;
        ggml_type type;
        int latent, hidden, experts;
        std::vector<int32_t> stored;
        std::vector<int> prefill_tokens;
    };
    const std::vector<shape> shapes = {
        { "q8_0_model", GGML_TYPE_Q8_0, 2560, 640, 512, { 0, 7, 64, 129, 200, 255, 256, 333, 400, 450, 501, 511 }, { 2, 65 } },
        { "q8_0_narrow", GGML_TYPE_Q8_0, 256, 96, 48, {}, { 2, 64, 65, 300, 2048 } },
        { "bf16_tail", GGML_TYPE_BF16, 64, 70, 24, {}, { 2, 65, 300 } },
    };
    std::mt19937 rng(20261028);
    std::normal_distribution<float> sample(0.0f, 0.3f);
    size_t discriminating = 0; // batches whose outputs tell the graph's SwiGLU kernel from the scalar clamp kernel
    // one token's route: nslots distinct experts of the fixture, softmax weights in descending order, normalized
    auto route = [&](const fixture & f, std::vector<int32_t> & ids, std::vector<float> & weights) {
        std::vector<int32_t> pool = f.stored;
        if (pool.empty()) { pool.resize(size_t(f.experts)); std::iota(pool.begin(), pool.end(), 0); }
        std::shuffle(pool.begin(), pool.end(), rng);
        std::vector<float> w(nslots);
        for (auto & v : w) { v = std::exp(4.0f * sample(rng)); }
        std::sort(w.begin(), w.end(), std::greater<float>());
        float sum = 0.0f;
        for (float v : w) { sum += v; }
        for (int slot = 0; slot < nslots; ++slot) { ids.push_back(pool[size_t(slot)]); weights.push_back(w[size_t(slot)] / sum); }
    };
    for (const auto & s : shapes) {
        fixture f;
        f.type = f.down_type = s.type; f.latent = s.latent; f.hidden = s.hidden; f.experts = s.experts; f.stored = s.stored;
        make_synthetic(f, 20261028);
        tensors meta1(f, 1), meta2(f, 2);
        std::vector<float> x(f.latent);
        for (auto & v : x) { v = s.type == GGML_TYPE_BF16 ? 5.0f * sample(rng) : sample(rng); }
        std::vector<int32_t> ids1, ids2;
        std::vector<float> w1, w2;
        route(f, ids1, w1);
        route(f, ids2, w2);
        ids2[4] = -1; w2[4] = 0.0f; // a dropped route, as mass reduction leaves one
        const auto layer1 = reference_swiglu_graph(f, 1, x, ids1, w1, 1, 1);
        require(same_bits(layer1, reference_swiglu_graph(f, 1, x, ids1, w1, 1, 4)),
            std::string(s.name) + ": SwiGLU reference output depends on its thread count");
        auto decoded = layer1;
        for (auto & v : decoded) { v *= .5f; }
        decoded = reference_swiglu_graph(f, 2, decoded, ids2, w2, 1, 1);
        struct batch { int ntokens; std::vector<float> x, weights, expected; std::vector<int32_t> ids; size_t experts; };
        std::vector<batch> batches;
        for (int ntokens : s.prefill_tokens) {
            batch b; b.ntokens = ntokens;
            b.x.resize(size_t(ntokens) * f.latent);
            for (auto & v : b.x) { v = s.type == GGML_TYPE_BF16 ? 5.0f * sample(rng) : sample(rng); }
            for (int t = 0; t < ntokens; ++t) { route(f, b.ids, b.weights); }
            b.expected = reference_swiglu_graph(f, 2, b.x, b.ids, b.weights, ntokens, 1);
            require(same_bits(b.expected, reference_swiglu_graph(f, 2, b.x, b.ids, b.weights, ntokens, 4)),
                std::string(s.name) + ": SwiGLU prefill reference output depends on its thread count");
            std::vector<int32_t> distinct(b.ids);
            std::sort(distinct.begin(), distinct.end());
            b.experts = size_t(std::unique(distinct.begin(), distinct.end()) - distinct.begin());
            batches.push_back(std::move(b));
        }

        char error[512] = {};
        // one participant: the sink computes on the threads of the graphs' calls, or alone
        sink_ptr sink = inputs.create(2, depth, fixture::resolve, &f, error, sizeof(error));
        require(bool(sink), std::string("SwiGLU sink create: ") + error);
        std::unique_ptr<kimi_cpu_prefill, decltype(&kimi_cpu_prefill_destroy)> prefill(
            kimi_cpu_prefill_create(sink.get(), error, sizeof(error)), kimi_cpu_prefill_destroy);
        require(bool(prefill), std::string("SwiGLU prefill create: ") + error);

        // the coordinator alone, with the run's I/O counters
        {
            std::vector<float> exact;
            print_sample((std::string(s.name) + "_swiglu_decode").c_str(), run_checked(inputs, s.name, f, sink.get(), meta1,
                x, ids1, w1, layer1, 1, 25.0f, 0, -1, &exact, &swiglu));
            require(same_bits(exact, layer1), std::string(s.name) + ": SwiGLU decode differs from the graph");
            const auto & b = batches.back();
            std::vector<float> routed(b.expected.size());
            kimi_cpu_prefill_stats stats = {};
            require(kimi_cpu_prefill_run_activation(prefill.get(), meta2.role[0], meta2.role[1], meta2.role[2],
                b.x.data(), b.ids.data(), b.weights.data(), nslots, b.ntokens, &swiglu, routed.data(), &stats,
                error, sizeof(error)), std::string(s.name) + " SwiGLU prefill: " + error);
            require(same_bits(routed, b.expected), std::string(s.name) + ": SwiGLU prefill differs from the graph");
            // The scalar clamp kernel at an infinite limit is SwiGLU in value only. Over a few hundred tokens
            // some hidden value it rounds otherwise survives the down input's quantization: such a batch tells
            // the graph's vector kernel from the scalar one. Which batch does depends on the vector width.
            const kimi_cpu_sink_activation clamp = { KIMI_CPU_SINK_ACTIVATION_SWIGLU_CLAMP, 0.0f, 0.0f,
                std::numeric_limits<float>::infinity(), nullptr, 0.0f };
            std::vector<float> scalar(b.expected.size());
            require(kimi_cpu_prefill_run_activation(prefill.get(), meta2.role[0], meta2.role[1], meta2.role[2],
                b.x.data(), b.ids.data(), b.weights.data(), nslots, b.ntokens, &clamp, scalar.data(), nullptr,
                error, sizeof(error)), std::string(s.name) + " scalar clamp prefill: " + error);
            size_t differing = 0;
            for (size_t i = 0; i < scalar.size(); ++i) { differing += std::memcmp(&scalar[i], &b.expected[i], sizeof(float)) != 0; }
            discriminating += differing != 0;
            std::printf("case=%s_scalar_clamp_infinity tokens=%d differing=%zu of %zu\n", s.name, b.ntokens, differing,
                scalar.size());
            require(stats.routed_slots == b.ids.size() && stats.stream.experts == b.experts &&
                stats.stream.read_requests == 3 * b.experts && !stats.stream.read_errors && !stats.stream.pending_reads,
                std::string(s.name) + ": SwiGLU prefill counters");
        }
        if (s.experts == 512) {
            // a batch whose route outputs exceed the workspace limit fails before any read
            const int ntokens = int((512ull << 20) / (uint64_t(nslots) * f.latent * sizeof(float))) + 1;
            std::vector<float> big(size_t(ntokens) * f.latent, .01f), out(big.size());
            std::vector<int32_t> ids(size_t(ntokens) * nslots);
            std::vector<float> weights(ids.size(), 1.0f / nslots);
            for (size_t i = 0; i < ids.size(); ++i) { ids[i] = s.stored[i % nslots]; }
            kimi_cpu_prefill_stats stats = {};
            require(!kimi_cpu_prefill_run_activation(prefill.get(), meta2.role[0], meta2.role[1], meta2.role[2],
                big.data(), ids.data(), weights.data(), nslots, ntokens, &swiglu, out.data(), &stats, error,
                sizeof(error)) && std::strstr(error, "route workspace") && stats.stream.read_requests == 0,
                std::string(s.name) + ": an oversized prefill batch was not refused before its reads");
            std::printf("case=%s_prefill_workspace_limit tokens=%d status=PASS error=%s\n", s.name, ntokens, error);
        }

        for (int n : counts) {
            ggml_threadpool_params params = ggml_threadpool_params_default(n);
            std::unique_ptr<ggml_threadpool, decltype(&ggml_threadpool_free)> pool(ggml_threadpool_new(&params),
                ggml_threadpool_free);
            require(bool(pool), "ggml threadpool");
            auto compute = [&](ggml_cgraph * graph) {
                ggml_cplan plan = ggml_graph_plan(graph, n, pool.get());
                std::vector<uint8_t> work(plan.work_size);
                plan.work_data = work.data();
                require(ggml_graph_compute(graph, &plan) == GGML_STATUS_SUCCESS, "threadpool graph compute");
            };
            auto context = [&](size_t floats) {
                ggml_context * ctx = ggml_init({ 64 * ggml_tensor_overhead() + ggml_graph_overhead() +
                    4 * floats * sizeof(float) + (size_t(1) << 20), nullptr, false });
                require(ctx, "threadpool graph context");
                return ctx;
            };
            auto tensor = [](ggml_context * ctx, ggml_type type, int64_t n0, int64_t n1, const void * data) {
                ggml_tensor * t = ggml_new_tensor_2d(ctx, type, n0, n1);
                std::memcpy(t->data, data, ggml_nbytes(t));
                return t;
            };
            auto node = [&](ggml_context * ctx, graph_layer & layer, const tensors & m, ggml_tensor * in, ggml_tensor * id,
                    ggml_tensor * w) {
                layer.sink = sink.get(); layer.meta = &m; layer.activation = &swiglu;
                ggml_tensor * args[] = { in, id, w };
                return ggml_custom_4d(ctx, GGML_TYPE_F32, f.latent, in->ne[1], 1, 1, args, 3, graph_layer_compute,
                    GGML_N_TASKS_MAX, &layer);
            };
            {
                ggml_context * ctx = context(size_t(4) * f.latent);
                graph_layer layers[2];
                ggml_tensor * y = node(ctx, layers[0], meta1, tensor(ctx, GGML_TYPE_F32, f.latent, 1, x.data()),
                    tensor(ctx, GGML_TYPE_I32, nslots, 1, ids1.data()), tensor(ctx, GGML_TYPE_F32, nslots, 1, w1.data()));
                y = ggml_scale(ctx, y, .5f);
                y = node(ctx, layers[1], meta2, y, tensor(ctx, GGML_TYPE_I32, nslots, 1, ids2.data()),
                    tensor(ctx, GGML_TYPE_F32, nslots, 1, w2.data()));
                ggml_cgraph * graph = ggml_new_graph(ctx);
                ggml_build_forward_expand(graph, y);
                compute(graph);
                for (const auto & layer : layers) {
                    require(layer.ok, std::string(s.name) + " threadpool SwiGLU decode: " + layer.error);
                    require(layer.nth == n, "threadpool SwiGLU decode call participants " + std::to_string(layer.nth));
                }
                require(!std::memcmp(y->data, decoded.data(), decoded.size() * sizeof(float)),
                    std::string(s.name) + ": threadpool SwiGLU decode differs from the graph threads=" + std::to_string(n));
                ggml_free(ctx);
            }
            for (const auto & b : batches) {
                ggml_context * ctx = context(b.x.size() + b.ids.size() + b.weights.size());
                graph_layer grouped;
                grouped.prefill = prefill.get();
                ggml_tensor * p = node(ctx, grouped, meta2, tensor(ctx, GGML_TYPE_F32, f.latent, b.ntokens, b.x.data()),
                    tensor(ctx, GGML_TYPE_I32, nslots, b.ntokens, b.ids.data()),
                    tensor(ctx, GGML_TYPE_F32, nslots, b.ntokens, b.weights.data()));
                ggml_cgraph * graph = ggml_new_graph(ctx);
                ggml_build_forward_expand(graph, p);
                compute(graph);
                require(grouped.ok, std::string(s.name) + " threadpool SwiGLU prefill: " + grouped.error);
                require(grouped.nth == n, "threadpool SwiGLU prefill call participants " + std::to_string(grouped.nth));
                require(!std::memcmp(p->data, b.expected.data(), b.expected.size() * sizeof(float)),
                    std::string(s.name) + ": threadpool SwiGLU prefill differs from the graph tokens=" +
                    std::to_string(b.ntokens) + " threads=" + std::to_string(n));
                ggml_free(ctx);
            }
            std::printf("case=%s_swiglu_threadpool threads=%d prefill_batches=%zu bit_exact=1 status=PASS\n",
                s.name, n, batches.size());
        }
    }
    require(discriminating, "no batch tells the graph's SwiGLU kernel from the scalar clamp kernel at an infinite limit");
    std::printf("CPU_SINK_SWIGLU_OK counts=%zu discriminating=%zu\n", counts.size(), discriminating);
}

int main(int argc, char ** argv) {
    fixture_inputs inputs;
    try {
        real_options options; bool ring_set = false, registered_lifecycle = false, cache_mode = false, cache_pressure = false;
        bool configured_runtime_mode = false;
        bool trace_profile_mode = false, max_cache_mode = false, hitrate_mode = false, elastic_cache_mode = false;
        bool schedules_mode = false, schedule_invalid_mode = false, early_up_gate_mode = false;
        bool role_major_mode = false, nonblocking_down_mode = false;
        bool poll_mode = false, poll_early_mode = false, cache_compute_mode = false, cache_location_mode = false;
        bool cache_policy_mode = false, cache_large_mode = false, cache_pressure_transition = false, cache_unbounded = false;
        bool activation_mode = false, q2_activation_mode = false, muse_q8_mode = false, muse_bf16_mode = false;
        bool auto_ring_mode = false, prefetch_mode = false, cache_load_mode = false, work_reclaim_mode = false, threadpool_mode = false;
        bool swiglu_mode = false;
        std::string pack, prefix;
        for (int i = 1; i < argc; ++i) {
            auto argument = [&]() { require(i + 1 < argc, "missing option argument"); return argv[++i]; };
            if (!std::strcmp(argv[i], "--fixture")) { inputs.argument(argument()); }
            else if (!std::strcmp(argv[i], "--threads")) { options.threads = std::atoi(argument()); }
            else if (!std::strcmp(argv[i], "--real-pack")) { pack = argument(); }
            else if (!std::strcmp(argv[i], "--fixture-prefix")) { prefix = argument(); }
            else if (!std::strcmp(argv[i], "--ring-experts")) { options.ring_experts = std::atoi(argument()); ring_set = true; }
            else if (!std::strcmp(argv[i], "--repeats")) { options.repeats = std::atoi(argument()); }
            else if (!std::strcmp(argv[i], "--warmup")) { options.warmup = std::atoi(argument()); }
            else if (!std::strcmp(argv[i], "--policy")) { options.policy = argument(); }
            else if (!std::strcmp(argv[i], "--jsonl")) { options.jsonl = argument(); }
            else if (!std::strcmp(argv[i], "--resident-dir")) { options.resident_dir = argument(); }
            else if (!std::strcmp(argv[i], "--configured-runtime")) { configured_runtime_mode = true; }
            else if (!std::strcmp(argv[i], "--trace-profile")) { trace_profile_mode = true; }
            else if (!std::strcmp(argv[i], "--prefetch-tests")) { prefetch_mode = true; }
            else if (!std::strcmp(argv[i], "--registered-lifecycle")) { registered_lifecycle = true; }
            else if (!std::strcmp(argv[i], "--early-up-gate-tests")) { early_up_gate_mode = true; }
            else if (!std::strcmp(argv[i], "--role-major-tests")) { role_major_mode = true; }
            else if (!std::strcmp(argv[i], "--nonblocking-down-tests")) { nonblocking_down_mode = true; }
            else if (!std::strcmp(argv[i], "--cache-location-tests")) { cache_location_mode = true; }
            else if (!std::strcmp(argv[i], "--cache-compute-tests")) { cache_compute_mode = true; }
            else if (!std::strcmp(argv[i], "--poll-tests")) { poll_mode = true; }
            else if (!std::strcmp(argv[i], "--poll-early-tests")) { poll_early_mode = true; }
            else if (!std::strcmp(argv[i], "--schedule-tests")) { schedules_mode = true; }
            else if (!std::strcmp(argv[i], "--schedule-invalid")) { schedule_invalid_mode = true; }
            else if (!std::strcmp(argv[i], "--cache-tests")) { cache_mode = true; }
            else if (!std::strcmp(argv[i], "--cache-pressure")) { cache_pressure = true; }
            else if (!std::strcmp(argv[i], "--cache-policy-tests")) { cache_policy_mode = true; }
            else if (!std::strcmp(argv[i], "--cache-hitrate-tests")) { hitrate_mode = true; }
            else if (!std::strcmp(argv[i], "--elastic-cache-tests")) { elastic_cache_mode = true; }
            else if (!std::strcmp(argv[i], "--cache-load-tests")) { cache_load_mode = true; }
            else if (!std::strcmp(argv[i], "--work-reclaim-tests")) { work_reclaim_mode = true; }
            else if (!std::strcmp(argv[i], "--max-cache-tests")) { max_cache_mode = true; }
            else if (!std::strcmp(argv[i], "--cache-large-tests")) { cache_large_mode = true; }
            else if (!std::strcmp(argv[i], "--cache-pressure-transition")) { cache_pressure_transition = true; }
            else if (!std::strcmp(argv[i], "--cache-unbounded")) { cache_unbounded = true; }
            else if (!std::strcmp(argv[i], "--auto-cache-ring-test")) { auto_ring_mode = true; }
            else if (!std::strcmp(argv[i], "--muse-q8-test")) { muse_q8_mode = true; }
            else if (!std::strcmp(argv[i], "--muse-bf16-test")) { muse_bf16_mode = true; }
            else if (!std::strcmp(argv[i], "--activation-tests")) { activation_mode = true; }
            else if (!std::strcmp(argv[i], "--q2-activation-test")) { q2_activation_mode = true; }
            else if (!std::strcmp(argv[i], "--threadpool-tests")) { threadpool_mode = true; }
            else if (!std::strcmp(argv[i], "--swiglu-tests")) { swiglu_mode = true; }
            else { throw std::runtime_error("usage: test-kimi-cpu-sink [--registered-lifecycle|--cache-tests|--cache-pressure] [--threads N] [--fixture KEY=VALUE] [--real-pack FILE --fixture-prefix PREFIX --ring-experts N --repeats N --warmup N --policy fixture|top16|top8|mass085 --jsonl FILE --resident-dir TMPFS]"); }
        }
        require(options.threads >= 2 && options.threads <= 128 && options.ring_experts >= 1 && options.ring_experts <= 32, "invalid thread/ring count");
        require(options.repeats >= 1 && options.repeats <= 10000 && options.warmup >= 0 && options.warmup <= 10000, "invalid repeat/warmup count");
        if (!schedule_invalid_mode) {
            const int required = registered_lifecycle ? 4 : options.threads;
            test_resources::cpu_participants(required - 1, 1);
            test_resources::io_ring();
        }
        if (cache_mode || hitrate_mode || elastic_cache_mode || cache_load_mode || work_reclaim_mode || max_cache_mode ||
                cache_location_mode || cache_compute_mode || early_up_gate_mode ||
                role_major_mode || nonblocking_down_mode || poll_mode || poll_early_mode || schedules_mode ||
                prefetch_mode || cache_policy_mode || cache_large_mode || activation_mode || muse_q8_mode || muse_bf16_mode) {
            test_resources::finite_envelope();
        }
#if defined(__linux__)
        if (configured_runtime_mode) { configured_runtime_tests(inputs); return 0; }
        if (auto_ring_mode) { cache_auto_ring_test(inputs, options.threads, ring_set ? options.ring_experts : 8); return 0; }
        if (cache_pressure_transition) { cache_pressure_transition_test(inputs, options.threads, ring_set ? options.ring_experts : 2); return 0; }
        if (cache_unbounded) { cache_pressure_test(inputs, options.threads, ring_set ? options.ring_experts : 2, true); return 0; }
        if (cache_pressure) {
            require(!cache_mode && !registered_lifecycle && pack.empty() && prefix.empty() && options.jsonl.empty() && options.resident_dir.empty(),
                    "cache tests cannot be combined with other test modes or real-pack measurement");
            cache_pressure_test(inputs, options.threads, ring_set ? options.ring_experts : 2);
            std::printf("CPU_SINK_TEST_OK\n"); return 0;
        }
#else
        if (configured_runtime_mode || auto_ring_mode || cache_pressure_transition || cache_unbounded || cache_pressure) {
            throw test_resources::unavailable("this mode reads the Linux task inventory or cgroup v2 memory limits");
        }
#endif
        if (muse_q8_mode) { muse_expert_norm_test(inputs, options.threads, ring_set ? options.ring_experts : 2, GGML_TYPE_Q8_0); return 0; }
        if (muse_bf16_mode) {
            muse_expert_norm_test(inputs, options.threads, ring_set ? options.ring_experts : 2, GGML_TYPE_BF16);
            muse_expert_norm_test(inputs, options.threads, ring_set ? options.ring_experts : 2, GGML_TYPE_BF16, true);
            return 0;
        }
        if (activation_mode) { activation_tests(inputs, options.threads, ring_set ? options.ring_experts : 2); return 0; }
        if (q2_activation_mode) { q2_activation_test(inputs, options.threads, ring_set ? options.ring_experts : 2); return 0; }
        if (cache_location_mode) { cache_location_tests(inputs, options.threads, options.ring_experts); return 0; }
        if (cache_compute_mode) { cache_compute_tests(inputs, options.threads, options.ring_experts); return 0; }
        if (hitrate_mode) { cache_hitrate_tests(inputs, options.threads, options.ring_experts); return 0; }
        if (elastic_cache_mode) { elastic_cache_tests(inputs, options.threads, ring_set ? options.ring_experts : 2); return 0; }
        if (cache_load_mode) { cache_load_tests(inputs, options.threads, ring_set ? options.ring_experts : 2); return 0; }
        if (work_reclaim_mode) { work_buffer_reclaim_tests(inputs, options.threads, ring_set ? options.ring_experts : 2); return 0; }
        if (max_cache_mode) { max_cache_tests(inputs, options.threads, options.ring_experts); return 0; }
        if (early_up_gate_mode) { schedule_tests(inputs, options.threads, options.ring_experts, false, true); return 0; }
        if (role_major_mode) { schedule_tests(inputs, options.threads, options.ring_experts, false, false, 0, true); return 0; }
        if (nonblocking_down_mode) { schedule_tests(inputs, options.threads, options.ring_experts, false, false, 0, false, true); return 0; }
        if (poll_mode || poll_early_mode) { schedule_tests(inputs, options.threads, options.ring_experts, false, poll_early_mode, 100); return 0; }
        if (schedules_mode) { schedule_tests(inputs, options.threads, options.ring_experts, true); return 0; }
        if (schedule_invalid_mode) { schedule_invalid_tests(inputs); return 0; }
        if (threadpool_mode) { threadpool_tests(inputs, ring_set ? options.ring_experts : 2); return 0; }
        if (swiglu_mode) { swiglu_tests(inputs, ring_set ? options.ring_experts : 2); return 0; }
        if (trace_profile_mode) { trace_profile_test(inputs, options.threads, ring_set ? options.ring_experts : 2); return 0; }
        if (prefetch_mode) { prefetch_tests(inputs, options.threads, ring_set ? options.ring_experts : 2); return 0; }
        if (cache_policy_mode) { cache_policy_tests(inputs, options.threads, ring_set ? options.ring_experts : 2); return 0; }
        if (cache_large_mode) { cache_large_test(inputs, options.threads, ring_set ? options.ring_experts : 2); return 0; }
        if (cache_mode) {
            require(!registered_lifecycle && pack.empty() && prefix.empty() && options.jsonl.empty() && options.resident_dir.empty(),
                    "cache tests cannot be combined with other test modes or real-pack measurement");
            cache_tests(inputs, options.threads, ring_set ? options.ring_experts : 2);
        } else if (registered_lifecycle) {
            require(pack.empty() && prefix.empty() && options.jsonl.empty() && options.resident_dir.empty(), "registered lifecycle cannot be combined with real-pack measurement");
            registered_lifecycle_tests(inputs);
        } else if (pack.empty()) {
            require(prefix.empty() && options.jsonl.empty() && options.resident_dir.empty(), "measurement options require real pack");
#if defined(__linux__)
            host_ledger_tests(inputs);
#else
            std::printf("case=host_ledger status=SKIP reason=linux_ledger_injection\n");
#endif
            tierless_budget_tests();
            synthetic_tests(inputs, options.threads, ring_set ? options.ring_experts : 2);
        }
        else { require(!prefix.empty(), "real pack requires fixture prefix"); real_test(inputs, pack, prefix, options); }
        std::printf("CPU_SINK_TEST_OK\n"); return 0;
    } catch (const unsupported_resident & e) { std::fprintf(stderr, "CPU_SINK_RESIDENT_UNSUPPORTED: %s\n", e.what()); return 77; }
      catch (const std::exception & e) { return test_resources::report("CPU_SINK_TEST", e); }
}
