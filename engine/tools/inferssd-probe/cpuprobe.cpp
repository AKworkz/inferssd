// SPDX-License-Identifier: Apache-2.0
// inferssd-cpuprobe: CPU matrix throughput per weight format, measured with ggml's CPU backend (the kernels the engine
// runs), and DRAM read bandwidth. Prints one JSON object on stdout; diagnostics go to stderr.
#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#elif defined(__APPLE__)
#include <sys/sysctl.h>
#elif defined(__linux__)
#include <sched.h>
#include <unistd.h>
#endif

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cerrno>
#include <chrono>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <fstream>
#include <iterator>
#include <map>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace {

const char * const USAGE =
    "usage: inferssd-cpuprobe [--threads 1,8,16] [--formats Q8_0,Q4_K,Q5_K,Q6_K,Q2_K,MXFP4,F16]\n"
    "                         [--rows 4096] [--cols 4096] [--tokens 1,32,512] [--min-seconds 0.5]\n"
    "                         [--cpus 0,1,2]\n"
    "\n"
    "Measures DRAM read bandwidth and ggml CPU matrix multiplication throughput per weight format and\n"
    "prints one JSON object (schema inferssd-cpuprobe/1) on stdout.\n"
    "\n"
    "  --threads LIST      thread counts (default: 1 and the logical CPUs this process may use)\n"
    "  --formats LIST      ggml weight types, case-insensitive\n"
    "  --rows N            weight rows (output features)\n"
    "  --cols N            weight columns (the reduction dimension)\n"
    "  --tokens LIST       activation columns; 1 reports weight bytes/s, more report FLOP/s\n"
    "  --min-seconds S     timed repetitions per point after one warm-up call\n"
    "  --cpus LIST         pin compute thread i to the i-th of these CPU IDs, ascending\n"
    "                      (Linux and Windows; default: unpinned)\n"
    "\n"
    "DRAM: one buffer of max(1 GiB, 8 x last-level cache) is read with each thread count (disjoint slices).\n"
    "Compute: the median time of ggml_mul_mat on the CPU backend with a persistent ggml threadpool. At\n"
    "tokens = 1 the call cycles through copies of the weight that together exceed the last-level cache,\n"
    "so every call reads the weight from DRAM as a decode step does.\n"
    "ggml_features: the instruction sets ggml's CPU kernels use in this build.\n";

struct usage_error : std::runtime_error {
    using std::runtime_error::runtime_error;
};

double now_s() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    const size_t n = v.size();
    return n % 2 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

// JSON numbers: rates and times always carry a fraction or exponent so readers see a float.
std::string num(double v) {
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%.6g", v);
    std::string s(buf);
    if (s.find_first_of(".eE") == std::string::npos) { s += ".0"; }
    return s;
}

std::string quote(const std::string & s) {
    std::string out = "\"";
    for (unsigned char c : s) {
        if (c == '"' || c == '\\') { out += '\\'; out += (char) c; }
        else if (c < 0x20) { char buf[8]; std::snprintf(buf, sizeof(buf), "\\u%04x", c); out += buf; }
        else { out += (char) c; }
    }
    return out + "\"";
}

std::string upper(std::string s) {
    for (char & c : s) { c = (char) std::toupper((unsigned char) c); }
    return s;
}

std::vector<std::string> split_list(const std::string & text) {
    std::vector<std::string> out;
    size_t start = 0;
    while (start <= text.size()) {
        const size_t end = text.find(',', start);
        const std::string item = text.substr(start, end == std::string::npos ? std::string::npos : end - start);
        if (!item.empty()) { out.push_back(item); }
        if (end == std::string::npos) { break; }
        start = end + 1;
    }
    return out;
}

long long parse_int(const std::string & option, const std::string & text, long long lo, long long hi) {
    char * end = nullptr;
    errno = 0;
    const long long v = std::strtoll(text.c_str(), &end, 10);
    if (text.empty() || *end != '\0' || errno != 0 || v < lo || v > hi) {
        throw usage_error(option + ": expected an integer in [" + std::to_string(lo) + ", " + std::to_string(hi) +
                          "], got '" + text + "'");
    }
    return v;
}

std::vector<int> parse_int_list(const std::string & option, const std::string & text, int lo, int hi) {
    std::vector<int> out;
    for (const std::string & item : split_list(text)) {
        const int v = (int) parse_int(option, item, lo, hi);
        if (std::find(out.begin(), out.end(), v) == out.end()) { out.push_back(v); }
    }
    if (out.empty()) { throw usage_error(option + ": empty list"); }
    return out;
}

// Parses "32K", "1024K", "8M" (sysfs cache sizes).
uint64_t parse_size(const std::string & text) {
    char * end = nullptr;
    const unsigned long long v = std::strtoull(text.c_str(), &end, 10);
    switch (end ? *end : '\0') {
        case 'K': return v << 10;
        case 'M': return v << 20;
        case 'G': return v << 30;
        default:  return v;
    }
}

// Logical CPUs this process may run on: the affinity mask (which a cpuset or a job narrows), else all of them.
int available_cpus() {
#if defined(__linux__)
    cpu_set_t set;
    CPU_ZERO(&set);
    if (sched_getaffinity(0, sizeof(set), &set) == 0 && CPU_COUNT(&set) > 0) { return CPU_COUNT(&set); }
#elif defined(_WIN32)
    DWORD_PTR process = 0, system = 0;
    if (GetProcessAffinityMask(GetCurrentProcess(), &process, &system) && process != 0) {
        int count = 0;
        for (; process != 0; process &= process - 1) { ++count; }
        return count;
    }
#endif
    return (int) std::max(1u, std::thread::hardware_concurrency());
}

// Total bytes of the last-level cache over all its instances; 0 when unknown.
uint64_t last_level_cache_bytes() {
#if defined(__linux__)
    const long ncpu = sysconf(_SC_NPROCESSORS_CONF);
    int level_max = 0;
    std::map<int, uint64_t> total_by_level;
    std::set<std::pair<int, std::string>> seen;
    for (long cpu = 0; cpu < ncpu; ++cpu) {
        for (int index = 0;; ++index) {
            const std::string dir =
                "/sys/devices/system/cpu/cpu" + std::to_string(cpu) + "/cache/index" + std::to_string(index) + "/";
            std::ifstream level_file(dir + "level");
            int level = 0;
            if (!(level_file >> level)) { break; }
            std::string type, size, shared;
            std::ifstream(dir + "type") >> type;
            std::ifstream(dir + "size") >> size;
            std::ifstream(dir + "shared_cpu_list") >> shared;
            if (type != "Data" && type != "Unified") { continue; }
            if (shared.empty()) { shared = std::to_string(cpu); }
            if (!seen.insert({ level, shared }).second) { continue; }
            total_by_level[level] += parse_size(size);
            level_max = std::max(level_max, level);
        }
    }
    return level_max ? total_by_level[level_max] : 0;
#elif defined(__APPLE__)
    auto sysctl_u64 = [](const std::string & name) -> uint64_t {
        unsigned char buf[8] = {};
        size_t size = sizeof(buf);
        if (sysctlbyname(name.c_str(), buf, &size, nullptr, 0) != 0) { return 0; }
        if (size == sizeof(uint32_t)) { uint32_t v; std::memcpy(&v, buf, sizeof(v)); return v; }
        if (size == sizeof(uint64_t)) { uint64_t v; std::memcpy(&v, buf, sizeof(v)); return v; }
        return 0;
    };
    if (const uint64_t l3 = sysctl_u64("hw.l3cachesize")) { return l3; }
    // Apple silicon: one L2 per cluster of each performance level (the system level cache is not reported).
    uint64_t total = 0;
    const uint64_t levels = sysctl_u64("hw.nperflevels");
    for (uint64_t i = 0; i < levels; ++i) {
        const std::string prefix = "hw.perflevel" + std::to_string(i) + ".";
        const uint64_t l2 = sysctl_u64(prefix + "l2cachesize");
        const uint64_t cpus = sysctl_u64(prefix + "logicalcpu");
        const uint64_t per_l2 = std::max<uint64_t>(1, sysctl_u64(prefix + "cpusperl2"));
        total += l2 * ((cpus + per_l2 - 1) / per_l2);
    }
    return total ? total : sysctl_u64("hw.l2cachesize");
#elif defined(_WIN32)
    DWORD length = 0;
    GetLogicalProcessorInformationEx(RelationCache, nullptr, &length);
    if (GetLastError() != ERROR_INSUFFICIENT_BUFFER || length == 0) { return 0; }
    std::vector<unsigned char> buf(length);
    auto * info = reinterpret_cast<SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX *>(buf.data());
    if (!GetLogicalProcessorInformationEx(RelationCache, info, &length)) { return 0; }
    int level_max = 0;
    std::map<int, uint64_t> total_by_level;
    for (DWORD offset = 0; offset < length;) {
        const auto * item = reinterpret_cast<const SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX *>(buf.data() + offset);
        if (item->Size == 0) { break; }
        if (item->Relationship == RelationCache &&
            (item->Cache.Type == CacheUnified || item->Cache.Type == CacheData)) {
            total_by_level[item->Cache.Level] += item->Cache.CacheSize;
            level_max = std::max<int>(level_max, item->Cache.Level);
        }
        offset += item->Size;
    }
    return level_max ? total_by_level[level_max] : 0;
#else
    return 0;
#endif
}

// ---- DRAM read bandwidth -------------------------------------------------------------------------------------------

std::atomic<uint64_t> g_sink{ 0 };

uint64_t sum_words(const uint64_t * p, size_t n) {
    uint64_t a0 = 0, a1 = 0, a2 = 0, a3 = 0;
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        a0 += p[i];
        a1 += p[i + 1];
        a2 += p[i + 2];
        a3 += p[i + 3];
    }
    for (; i < n; ++i) { a0 += p[i]; }
    return a0 + a1 + a2 + a3;
}

// Runs fn(t) on `threads` threads released together; returns the seconds from release to the last join.
template <typename Fn> double run_parallel(int threads, Fn fn) {
    std::atomic<int> ready{ 0 };
    std::atomic<bool> go{ false };
    std::vector<std::thread> pool;
    pool.reserve(threads);
    for (int t = 0; t < threads; ++t) {
        pool.emplace_back([&, t] {
            ready.fetch_add(1);
            while (!go.load(std::memory_order_acquire)) { std::this_thread::yield(); }
            fn(t);
        });
    }
    while (ready.load() < threads) { std::this_thread::yield(); }
    const double t0 = now_s();
    go.store(true, std::memory_order_release);
    for (std::thread & th : pool) { th.join(); }
    return now_s() - t0;
}

struct dram_point {
    int      threads;
    uint64_t bytes;
    double   Bps;
};

std::vector<dram_point> measure_dram(uint64_t bytes, const std::vector<int> & thread_counts, double min_seconds) {
    const size_t words = (size_t) (bytes / sizeof(uint64_t));
    std::unique_ptr<uint64_t[]> buf(new uint64_t[words]);
    const uint64_t * data = buf.get();
    // Write every page first (untouched pages would read as the shared zero page), from all threads so the pages
    // spread over the NUMA nodes the readers run on.
    const int fill_threads = *std::max_element(thread_counts.begin(), thread_counts.end());
    run_parallel(fill_threads, [&](int t) {
        const size_t b = words * t / fill_threads, e = words * (t + 1) / fill_threads;
        for (size_t i = b; i < e; ++i) { buf[i] = i * 0x9E3779B97F4A7C15ull; }
    });
    std::vector<dram_point> out;
    for (int threads : thread_counts) {
        auto pass = [&] {
            return run_parallel(threads, [&](int t) {
                const size_t b = words * t / threads, e = words * (t + 1) / threads;
                g_sink.fetch_add(sum_words(data + b, e - b), std::memory_order_relaxed);
            });
        };
        pass();
        std::vector<double> samples;
        const double start = now_s();
        do { samples.push_back(pass()); } while (now_s() - start < min_seconds);
        const double Bps = (double) (words * sizeof(uint64_t)) / median(samples);
        std::fprintf(stderr, "cpuprobe: dram threads=%d %.3g B/s (%zu passes)\n", threads, Bps, samples.size());
        out.push_back({ threads, (uint64_t) (words * sizeof(uint64_t)), Bps });
    }
    return out;
}

// ---- compute -------------------------------------------------------------------------------------------------------

uint64_t splitmix64(uint64_t x) {
    x += 0x9E3779B97F4A7C15ull;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
    return x ^ (x >> 31);
}

// Deterministic values in [-1, 1).
void fill_uniform(float * dst, size_t n, uint64_t seed) {
    const int threads = available_cpus();
    run_parallel(threads, [&](int t) {
        const size_t b = n * t / threads, e = n * (t + 1) / threads;
        for (size_t i = b; i < e; ++i) {
            dst[i] = (float) ((double) (splitmix64(seed ^ (uint64_t) i) >> 11) * 0x1.0p-52 - 1.0);
        }
    });
}

struct format_spec {
    std::string name;
    ggml_type   type;
};

format_spec parse_format(const std::string & text) {
    for (int t = 0; t < GGML_TYPE_COUNT; ++t) {
        const ggml_type type = (ggml_type) t;
        if (ggml_blck_size(type) == 0) { continue; }  // removed types
        if (upper(ggml_type_name(type)) == upper(text)) { return { upper(ggml_type_name(type)), type }; }
    }
    throw usage_error("--formats: unknown ggml type '" + text + "'");
}

// Types quantized here with all-ones importance weights: those ggml_quantize_chunk requires them for, and IQ1_M,
// which ggml_quantize_chunk handles although it has no reference row function.
bool needs_imatrix(ggml_type type) {
    return ggml_quantize_requires_imatrix(type) || type == GGML_TYPE_IQ1_M;
}

// Why the CPU backend cannot run a rows x cols weight of this type, or "" when it can.
std::string unsupported_reason(ggml_type type, int64_t cols) {
    const ggml_type_traits_cpu * cpu = ggml_get_type_traits_cpu(type);
    if (!cpu || !cpu->vec_dot) { return "no CPU kernel"; }
    const ggml_type_traits_cpu * act = ggml_get_type_traits_cpu(cpu->vec_dot_type);
    if (!act || !act->from_float) { return "no CPU activation conversion"; }
    if (type != GGML_TYPE_F32 && !ggml_get_type_traits(type)->from_float_ref && !needs_imatrix(type)) {
        return "no quantization function";
    }
    if (cols % ggml_blck_size(type) != 0) {
        return "cols (" + std::to_string(cols) + ") is not a multiple of the block size (" +
               std::to_string(ggml_blck_size(type)) + ")";
    }
    return "";
}

std::vector<uint8_t> quantize_weight(ggml_type type, int64_t rows, int64_t cols) {
    std::vector<float> src((size_t) (rows * cols));
    fill_uniform(src.data(), src.size(), 0x5EEDull);
    std::vector<uint8_t> dst(ggml_row_size(type, cols) * (size_t) rows);
    std::vector<float> ones((size_t) cols, 1.0f);
    const float * imatrix = needs_imatrix(type) ? ones.data() : nullptr;
    ggml_quantize_init(type);
    const int threads = (int) std::min<int64_t>(rows, available_cpus());
    run_parallel(threads, [&](int t) {
        const int64_t r0 = rows * t / threads, r1 = rows * (t + 1) / threads;
        if (r1 > r0) { ggml_quantize_chunk(type, src.data(), dst.data(), r0 * cols, r1 - r0, cols, imatrix); }
    });
    return dst;
}

struct compute_point {
    int    tokens;
    double us;
    double rate;  // weight bytes/s at one token, FLOP/s otherwise
};

struct compute_row {
    std::string                name;
    int                        threads;
    uint64_t                   weight_bytes;
    std::vector<compute_point> points;
};

// A threadpool of THREADS; with CPUS, strict: thread i runs on the i-th CPU in ascending order.
ggml_threadpool_t new_pool(int threads, const std::vector<int> & cpus) {
    ggml_threadpool_params tpp = ggml_threadpool_params_default(threads);
    if (!cpus.empty()) {
        for (int cpu : cpus) { tpp.cpumask[cpu] = true; }
        tpp.strict_cpu = true;
    }
    return ggml_threadpool_new(&tpp);
}

// A pinned pool also pins the calling thread to its first CPU; this gives it back the CPUs it had.
struct affinity_guard {
#if defined(__linux__)
    cpu_set_t saved;
    bool      valid;
    affinity_guard() { CPU_ZERO(&saved); valid = sched_getaffinity(0, sizeof(saved), &saved) == 0; }
    ~affinity_guard() {
        if (valid) { sched_setaffinity(0, sizeof(saved), &saved); }
    }
#elif defined(_WIN32)
    DWORD_PTR saved = 0;
    affinity_guard() {
        DWORD_PTR system = 0;
        if (!GetProcessAffinityMask(GetCurrentProcess(), &saved, &system)) { saved = 0; }
    }
    ~affinity_guard() {
        if (saved != 0) { SetThreadAffinityMask(GetCurrentThread(), saved); }
    }
#endif
};

struct ggml_context_deleter {
    void operator()(ggml_context * ctx) const { ggml_free(ctx); }
};

struct ggml_buffer_deleter {
    void operator()(ggml_backend_buffer_t buf) const { ggml_backend_buffer_free(buf); }
};

double time_calls(ggml_backend_t backend, const std::vector<ggml_cgraph *> & graphs, double min_seconds) {
    auto call = [&](size_t i) {
        const double t0 = now_s();
        const ggml_status status = ggml_backend_graph_compute(backend, graphs[i % graphs.size()]);
        const double t = now_s() - t0;
        if (status != GGML_STATUS_SUCCESS) {
            throw std::runtime_error(std::string("ggml_backend_graph_compute failed: ") +
                                     ggml_status_to_string(status));
        }
        return t;
    };
    call(0);
    std::vector<double> samples;
    const double start = now_s();
    size_t i = 1;
    do { samples.push_back(call(i++)); } while (now_s() - start < min_seconds);
    return median(samples);
}

// Runs every thread count for one format. Returns false (with *why) when the CPU backend cannot run it.
bool measure_format(ggml_backend_t backend, const format_spec & format, int64_t rows, int64_t cols,
                    const std::vector<int> & tokens, const std::vector<int> & thread_counts,
                    const std::vector<int> & cpus, double min_seconds, uint64_t rotate_bytes,
                    std::vector<compute_row> & out, std::string * why) {
    *why = unsupported_reason(format.type, cols);
    if (!why->empty()) { return false; }

    const size_t weight_bytes = ggml_row_size(format.type, cols) * (size_t) rows;
    const bool has_decode = std::find(tokens.begin(), tokens.end(), 1) != tokens.end();
    const size_t copies =
        has_decode ? std::min<size_t>(4096, std::max<size_t>(1, (rotate_bytes + weight_bytes - 1) / weight_bytes)) : 1;

    const size_t n_graphs = copies + tokens.size();
    const size_t n_tensors = 2 * copies + 2 * tokens.size();
    ggml_init_params params = {
        /*.mem_size   =*/ ggml_tensor_overhead() * n_tensors + ggml_graph_overhead_custom(4, false) * n_graphs + 4096,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    std::unique_ptr<ggml_context, ggml_context_deleter> ctx(ggml_init(params));
    if (!ctx) { throw std::runtime_error("ggml_init failed"); }

    std::vector<ggml_tensor *> weights;
    for (size_t c = 0; c < copies; ++c) { weights.push_back(ggml_new_tensor_2d(ctx.get(), format.type, cols, rows)); }
    std::vector<ggml_tensor *> inputs;
    std::vector<std::vector<ggml_cgraph *>> graphs;  // per token count
    for (int n : tokens) {
        ggml_tensor * x = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, cols, n);
        inputs.push_back(x);
        std::vector<ggml_cgraph *> per;
        for (size_t c = 0; c < (n == 1 ? copies : 1); ++c) {
            ggml_cgraph * gf = ggml_new_graph_custom(ctx.get(), 4, false);
            ggml_build_forward_expand(gf, ggml_mul_mat(ctx.get(), weights[c], x));
            per.push_back(gf);
        }
        graphs.push_back(std::move(per));
    }
    if (!ggml_backend_supports_op(backend, ggml_graph_node(graphs[0][0], -1))) {
        *why = "the CPU backend does not support MUL_MAT for this type";
        return false;
    }
    std::unique_ptr<ggml_backend_buffer, ggml_buffer_deleter> buf(ggml_backend_alloc_ctx_tensors(ctx.get(), backend));
    if (!buf) {
        throw std::runtime_error("cannot allocate " + std::to_string(copies * weight_bytes) + " weight bytes");
    }

    {
        const std::vector<uint8_t> q = quantize_weight(format.type, rows, cols);
        for (ggml_tensor * w : weights) { ggml_backend_tensor_set(w, q.data(), 0, q.size()); }
    }
    for (size_t i = 0; i < inputs.size(); ++i) {
        std::vector<float> x((size_t) ggml_nelements(inputs[i]));
        fill_uniform(x.data(), x.size(), 0xAC71ull + i);
        ggml_backend_tensor_set(inputs[i], x.data(), 0, x.size() * sizeof(float));
    }
    if (copies > 1) {
        std::fprintf(stderr, "cpuprobe: %s cycles %zu weight copies (%.0f MiB) at 1 token\n", format.name.c_str(),
                     copies, copies * weight_bytes / 1048576.0);
    }

    for (int threads : thread_counts) {
        affinity_guard affinity;
        ggml_threadpool_t pool = new_pool(threads, cpus);
        if (!pool) {
            throw std::runtime_error("ggml_threadpool_new failed for " + std::to_string(threads) + " threads");
        }
        ggml_backend_cpu_set_n_threads(backend, threads);
        ggml_backend_cpu_set_threadpool(backend, pool);
        compute_row row{ format.name, threads, (uint64_t) weight_bytes, {} };
        try {
            for (size_t i = 0; i < tokens.size(); ++i) {
                const double s = time_calls(backend, graphs[i], min_seconds);
                const double rate = tokens[i] == 1 ? weight_bytes / s : 2.0 * rows * cols * tokens[i] / s;
                row.points.push_back({ tokens[i], s * 1e6, rate });
                std::fprintf(stderr, "cpuprobe: %s threads=%d tokens=%d %.1f us %.3g %s\n", format.name.c_str(),
                             threads, tokens[i], s * 1e6, rate, tokens[i] == 1 ? "B/s" : "FLOP/s");
            }
        } catch (...) {
            ggml_backend_cpu_set_threadpool(backend, nullptr);
            ggml_threadpool_free(pool);
            throw;
        }
        ggml_backend_cpu_set_threadpool(backend, nullptr);
        ggml_threadpool_free(pool);
        out.push_back(std::move(row));
    }
    return true;
}

// The instruction sets the CPU backend's kernels use in this build (on x86 these are fixed at compile time, so a
// portable build lists fewer than the CPU has). Build options in the same list (library choices, repacking) are
// left out.
std::vector<std::string> ggml_features() {
    static const char * const BUILD_OPTIONS[] = {
        "ACCELERATE", "CPU_HBM", "KLEIDIAI", "LLAMAFILE", "OPENMP", "REPACK",
    };
    std::vector<std::string> out;
    ggml_backend_reg_t reg = ggml_backend_cpu_reg();
    auto get_features =
        (ggml_backend_get_features_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_get_features");
    if (!get_features) { return out; }
    for (const ggml_backend_feature * f = get_features(reg); f && f->name; ++f) {
        const bool option = std::any_of(std::begin(BUILD_OPTIONS), std::end(BUILD_OPTIONS),
                                        [&](const char * name) { return std::strcmp(f->name, name) == 0; });
        if (!option && std::strcmp(f->value, "1") == 0) { out.push_back(f->name); }
    }
    return out;
}

struct options {
    std::vector<int>         threads;
    std::vector<format_spec> formats;
    int64_t                  rows        = 4096;
    int64_t                  cols        = 4096;
    std::vector<int>         tokens      = { 1, 32, 512 };
    double                   min_seconds = 0.5;
    std::vector<int>         cpus;  // ascending; empty: unpinned
};

options parse_args(int argc, char ** argv) {
    options o;
    std::string formats = "Q8_0,Q4_K,Q5_K,Q6_K,Q2_K,MXFP4,F16";
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            std::fputs(USAGE, stdout);
            std::exit(0);
        }
        if (i + 1 >= argc) { throw usage_error("unknown option or missing value: " + arg); }
        const std::string value = argv[++i];
        if (arg == "--threads") {
            o.threads = parse_int_list(arg, value, 1, GGML_MAX_N_THREADS);
        } else if (arg == "--formats") {
            formats = value;
        } else if (arg == "--rows") {
            o.rows = parse_int(arg, value, 1, 1 << 20);
        } else if (arg == "--cols") {
            o.cols = parse_int(arg, value, 1, 1 << 20);
        } else if (arg == "--tokens") {
            o.tokens = parse_int_list(arg, value, 1, 1 << 16);
        } else if (arg == "--cpus") {
#if defined(__linux__)
            o.cpus = parse_int_list(arg, value, 0, GGML_MAX_N_THREADS - 1);
            std::sort(o.cpus.begin(), o.cpus.end());
            // A thread cannot be pinned outside this process's CPUs; it would run unpinned unnoticed.
            cpu_set_t allowed;
            CPU_ZERO(&allowed);
            if (sched_getaffinity(0, sizeof(allowed), &allowed) != 0) {
                throw std::runtime_error("cannot read the CPUs this process may use");
            }
            for (int cpu : o.cpus) {
                if (cpu >= CPU_SETSIZE || !CPU_ISSET(cpu, &allowed)) {
                    throw std::runtime_error("CPU " + std::to_string(cpu) + " is not available to this process");
                }
            }
#elif defined(_WIN32)
            o.cpus = parse_int_list(arg, value, 0, GGML_MAX_N_THREADS - 1);
            std::sort(o.cpus.begin(), o.cpus.end());
            // A thread cannot be pinned outside this process's CPUs; it would run unpinned unnoticed.
            DWORD_PTR allowed = 0, system = 0;
            if (!GetProcessAffinityMask(GetCurrentProcess(), &allowed, &system)) {
                throw std::runtime_error("cannot read the CPUs this process may use");
            }
            for (int cpu : o.cpus) {
                if (cpu >= (int) (8 * sizeof(allowed)) || !((allowed >> cpu) & 1)) {
                    throw std::runtime_error("CPU " + std::to_string(cpu) + " is not available to this process");
                }
            }
#else
            throw usage_error("--cpus: threads cannot be pinned to CPUs on this platform");
#endif
        } else if (arg == "--min-seconds") {
            char * end = nullptr;
            o.min_seconds = std::strtod(value.c_str(), &end);
            if (value.empty() || *end != '\0' || !(o.min_seconds >= 0.0 && o.min_seconds <= 3600.0)) {
                throw usage_error("--min-seconds: expected seconds in [0, 3600], got '" + value + "'");
            }
        } else {
            throw usage_error("unknown option: " + arg);
        }
    }
    if (o.threads.empty()) {
        o.threads = { 1 };
        const int all = std::min(GGML_MAX_N_THREADS, available_cpus());
        if (all != 1) { o.threads.push_back(all); }
    }
    for (int threads : o.threads) {
        if (!o.cpus.empty() && threads > (int) o.cpus.size()) {
            throw usage_error("--threads: " + std::to_string(threads) + " threads exceed the " +
                              std::to_string(o.cpus.size()) + " CPUs of --cpus");
        }
    }
    for (const std::string & item : split_list(formats)) {
        format_spec f = parse_format(item);
        const bool dup =
            std::any_of(o.formats.begin(), o.formats.end(), [&](const format_spec & g) { return g.type == f.type; });
        if (!dup) { o.formats.push_back(std::move(f)); }
    }
    if (o.formats.empty()) { throw usage_error("--formats: empty list"); }
    if (o.rows * o.cols > (int64_t) 1 << 30) { throw usage_error("--rows x --cols exceeds 2^30 elements"); }
    return o;
}

int run(int argc, char ** argv) {
    const options o = parse_args(argc, argv);
    ggml_cpu_init();

    const uint64_t llc = last_level_cache_bytes();
    std::fprintf(stderr, "cpuprobe: last-level cache %" PRIu64 " bytes%s\n", llc, llc ? "" : " (unknown)");

    const uint64_t dram_bytes = std::max<uint64_t>(1ull << 30, 8 * llc);
    const std::vector<dram_point> dram = measure_dram(dram_bytes, o.threads, o.min_seconds);

    ggml_backend_t backend = ggml_backend_cpu_init();
    if (!backend) { throw std::runtime_error("ggml_backend_cpu_init failed"); }
    struct backend_guard {
        ggml_backend_t b;
        ~backend_guard() { ggml_backend_free(b); }
    } guard{ backend };

    const uint64_t rotate_bytes = std::max<uint64_t>(256ull << 20, 4 * llc);
    std::vector<compute_row> compute;
    std::vector<std::pair<std::string, std::string>> skipped;
    for (const format_spec & format : o.formats) {
        std::string why;
        if (!measure_format(backend, format, o.rows, o.cols, o.tokens, o.threads, o.cpus, o.min_seconds, rotate_bytes,
                            compute, &why)) {
            std::fprintf(stderr, "cpuprobe: %s skipped: %s\n", format.name.c_str(), why.c_str());
            skipped.emplace_back(format.name, why);
        }
    }

    std::string json = "{\"schema\": \"inferssd-cpuprobe/1\",\n \"ggml_features\": [";
    const std::vector<std::string> features = ggml_features();
    for (size_t i = 0; i < features.size(); ++i) { json += (i ? ", " : "") + quote(features[i]); }
    json += "],\n \"cpus\": [";
    for (size_t i = 0; i < o.cpus.size(); ++i) { json += (i ? ", " : "") + std::to_string(o.cpus[i]); }
    json += "],\n \"dram\": [";
    for (size_t i = 0; i < dram.size(); ++i) {
        json += std::string(i ? ",\n          " : "") + "{\"threads\": " + std::to_string(dram[i].threads) +
                ", \"bytes\": " + std::to_string(dram[i].bytes) + ", \"Bps\": " + num(dram[i].Bps) + "}";
    }
    json += "],\n \"compute\": [";
    for (size_t i = 0; i < compute.size(); ++i) {
        const compute_row & r = compute[i];
        json += std::string(i ? ",\n             " : "") + "{\"format\": " + quote(r.name) +
                ", \"threads\": " + std::to_string(r.threads) + ", \"rows\": " + std::to_string(o.rows) +
                ", \"cols\": " + std::to_string(o.cols) + ", \"weight_bytes\": " + std::to_string(r.weight_bytes) +
                ", \"points\": [";
        for (size_t j = 0; j < r.points.size(); ++j) {
            const compute_point & p = r.points[j];
            json += std::string(j ? ", " : "") + "{\"tokens\": " + std::to_string(p.tokens) + ", \"us\": " + num(p.us) +
                    (p.tokens == 1 ? ", \"weight_Bps\": " : ", \"flops\": ") + num(p.rate) + "}";
        }
        json += "]}";
    }
    json += "],\n \"skipped\": [";
    for (size_t i = 0; i < skipped.size(); ++i) {
        json += std::string(i ? ", " : "") + "{\"format\": " + quote(skipped[i].first) +
                ", \"reason\": " + quote(skipped[i].second) + "}";
    }
    json += "]}\n";
    std::fputs(json.c_str(), stdout);
    std::fflush(stdout);
    return 0;
}

}  // namespace

int main(int argc, char ** argv) {
    try {
        return run(argc, argv);
    } catch (const usage_error & e) {
        std::fprintf(stderr, "inferssd-cpuprobe: %s (see --help)\n", e.what());
        return 2;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "inferssd-cpuprobe: %s\n", e.what());
        return 1;
    }
}
