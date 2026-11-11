#include "ssd/cpu-sink.h"
#include "ssd/cpu-sink-internal.h"
#include "ssd/runtime.h"
#include "ssd/testing.h"
#include "ggml-ssd/gpu/provider.h"
#include "ggml-ssd/settings/settings.h"
#include "ggml-ssd/gpu-tier-settings.h"
#include "ggml-ssd/telemetry/telemetry.h"
#include "ggml-ssd/place/runtime-policy.h"
#include "ggml-ssd/host/page-lock.h"
#include "ggml-ssd/host/arbiter.h"
#include "ggml-ssd/host/expert-cache.h"
#include "ggml-ssd/gpu/kimi-vram-tier-api.h"
#include "ggml-ssd/io/read-ring.h"
#include "ggml-ssd/platform/file.h"
#include "ggml-ssd/platform/memory.h"
#include "ggml-ssd/platform/threads.h"
#include "vec.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <bitset>
#include <chrono>
#include <climits>
#include <cmath>
#include <condition_variable>
#include <cstdarg>
#include <deque>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <new>
#include <numeric>
#include <stdexcept>
#include <string>
#include <sstream>
#include <iomanip>
#include <locale>
#include <thread>
#include <vector>

extern "C" size_t kimi_cpu_sink_source_size(void) { return sizeof(kimi_cpu_sink_source); }

static void sink_error(char * dst, size_t size, const char * message) {
    if (dst && size) {
        std::snprintf(dst, size, "%s", message);
    }
}

static ssd::result<ssd::host::allocation_ticket> sink_host_failure(ssd::status code, const char * message) noexcept {
    ssd::result<ssd::host::allocation_ticket> result;
    result.code = code;
    try { result.error = message; } catch (...) {} // allocation failure must not escape this ABI seam
    return result;
}

namespace {

using sink_clock = std::chrono::steady_clock;
static double elapsed_us(sink_clock::time_point start) {
    return std::chrono::duration<double, std::micro>(sink_clock::now() - start).count();
}

template <typename Predicate>
static void sink_poll(int microseconds, Predicate ready) {
    const auto deadline = sink_clock::now() + std::chrono::microseconds(microseconds);
    do {
        for (int i = 0; i < 32; ++i) {
            if (ready()) { return; }
            ssd::platform::cpu_relax();
        }
    } while (sink_clock::now() < deadline);
}

// A diagnostic record built whole and written with one call, so it stays
// intact beside other stderr writers on every OS. Diagnostics never throw: a
// record that cannot grow is dropped.
struct sink_record {
    std::string text;
    bool lost = false;
    GGML_ATTRIBUTE_FORMAT(2, 3) void append(const char * format, ...) noexcept {
        char local[512];
        va_list args, again;
        va_start(args, format);
        va_copy(again, args);
        const int n = std::vsnprintf(local, sizeof(local), format, args);
        va_end(args);
        try {
            if (n > 0 && size_t(n) < sizeof(local)) { text.append(local, size_t(n)); }
            else if (n > 0) {
                const size_t at = text.size();
                text.resize(at + size_t(n) + 1);
                std::vsnprintf(&text[at], size_t(n) + 1, format, again);
                text.resize(at + size_t(n));
            }
        } catch (...) { lost = true; }
        va_end(again);
    }
    void flush() noexcept {
        if (!lost && !text.empty()) { std::fwrite(text.data(), 1, text.size(), stderr); }
        text.clear();
        lost = false;
    }
};

// Saturating byte sums for the ledger's held-memory view.
static uint64_t sink_add(uint64_t a, uint64_t b) { return b > UINT64_MAX - a ? UINT64_MAX : a + b; }

using sink_ring = ssd::io::read_ring;
using sink_completion = ssd::io::completion;
template<class T> static T sink_io_value(ssd::result<T> value) {
    if (!value) { throw std::runtime_error(value.error.empty() ? "direct I/O operation failed" : value.error); }
    return std::move(value.value);
}
static void sink_io_value(ssd::result<void> value) {
    if (!value) { throw std::runtime_error(value.error.empty() ? "direct I/O operation failed" : value.error); }
}

// Direct reads land in 4096-aligned targets with 4096-padded ranges; a
// descriptor whose direct alignment does not divide 4096 is refused.
constexpr size_t sink_direct_granule = 4096;

struct sink_buffer {
    void * data = nullptr;
    size_t size = 0;
    ~sink_buffer() { ssd::platform::aligned_free(data); }
    sink_buffer() = default;
    sink_buffer(const sink_buffer &) = delete;
    sink_buffer & operator=(const sink_buffer &) = delete;
    sink_buffer(sink_buffer && other) noexcept : data(other.data), size(other.size) { other.data = nullptr; other.size = 0; }
    void reserve(size_t n) {
        if (size >= n) { return; }
        void * next = ssd::platform::aligned_alloc_bytes(sink_direct_granule, n);
        if (!next) { throw std::runtime_error("CPU sink aligned allocation failed"); }
        ssd::platform::aligned_free(data);
        data = next; size = n;
    }
};

// Whether a descriptor reads directly at the granule the ranges are padded to.
// The alignment is a device query on Windows: ask once per descriptor in a run.
struct sink_direct_check {
    int aligned = -1;
    bool operator()(int fd) {
        if (!ssd::platform::is_direct(fd)) { return false; }
        if (fd != aligned) {
            if (sink_direct_granule % ssd::platform::direct_alignment(fd) != 0) { return false; }
            aligned = fd;
        }
        return true;
    }
};

// Weight types whose CPU MUL_MAT_ID is one vec_dot per weight row against an
// input row quantized by from_float of vec_dot_type, with whole blocks per row.
static bool sink_expert_type_supported(ggml_type type, int64_t columns) {
    if (int(type) < 0 || int(type) >= int(GGML_TYPE_COUNT) || ggml_blck_size(type) <= 0) { return false; }
    const auto * traits = ggml_get_type_traits_cpu(type);
    if (!traits || !traits->vec_dot || traits->vec_dot_type == GGML_TYPE_F32) { return false; }
    const auto * input = ggml_get_type_traits_cpu(traits->vec_dot_type);
    return input && input->from_float && ggml_blck_size(traits->vec_dot_type) > 0 &&
        columns % ggml_blck_size(type) == 0 && columns % ggml_blck_size(traits->vec_dot_type) == 0;
}

struct sink_configuration {
    bool production = false;
    ssd::settings::snapshot settings;
    ssd::telemetry::debug_channels debug;
    ssd::telemetry::writer * events = nullptr;
    ssd::host::context_client * host = nullptr;
    ssd::gpu::provider_v1 gpu{};
    bool has_gpu = false;
    int gpu_device = 0;
    const ssd::store::source_view * source = nullptr;
    void * gpu_transport = nullptr;
    ssd::cpu::testing::execution_policy policy;
};

// The CPUs of the sink's own threads (cpu.io_cpus), within those this process
// may run on. Where the OS cannot pin (macOS) they are still validated.
static std::vector<int> sink_cpus(const std::vector<int> & cpus) {
    if (cpus.empty()) { return {}; }
    ssd::platform::cpu_list allowed;
    std::string why;
    if (!ssd::platform::process_cpus(allowed, &why)) { throw std::runtime_error("get sink affinity: " + why); }
    std::vector<int> result;
    for (int cpu : cpus) {
        if (!std::binary_search(allowed.begin(), allowed.end(), cpu) ||
            std::find(result.begin(), result.end(), cpu) != result.end()) {
            throw std::runtime_error("CPU setting requires distinct CPUs within the caller's allowed mask");
        }
        result.push_back(cpu);
    }
    return result;
}

static void sink_pin(std::thread & thread, const std::vector<int> & cpus) {
    if (!ssd::platform::pin_thread(thread.native_handle(), cpus)) {
        throw std::runtime_error("CPU sink affinity could not be applied exactly (no fallback)");
    }
}

// A call's participants are ggml threads.
constexpr int sink_max_participants = GGML_MAX_N_THREADS;

// Parks threads until a notify, without a lock: an event count. A waiter registers as a sleeper and rechecks its
// condition before it parks, so a notify either finds it registered or the condition already changed; a notify
// with nobody parked costs no system call. Conditions and the stores they observe are sequentially consistent.
struct sink_event {
    alignas(64) std::atomic<uint32_t> count{0};
    std::atomic<uint32_t> sleepers{0};

    // Parks once unless ready(); true if it parked.
    template <typename Ready> bool park(Ready ready) {
        const uint32_t seen = count.load();
        sleepers.fetch_add(1);
        const bool parks = !ready();
        if (parks) { ssd::platform::wait_on(count, seen); }
        sleepers.fetch_sub(1, std::memory_order_relaxed);
        return parks;
    }
    void notify() {
        count.fetch_add(1);
        if (sleepers.load()) { ssd::platform::wake_all(count); }
    }
};

// The phase word the coordinator publishes: the phase generation (bits 0-31), its active participants (32-47), its
// kind (48-50: a down phase, a parallel one, the run's last), the call's id (51-62) and whether it is open (63).
struct sink_phase {
    static constexpr uint64_t down = 1ull << 48, parallel = 1ull << 49, last = 1ull << 50, open = 1ull << 63;
    uint64_t word = 0;
    static uint64_t make(uint32_t generation, int active, uint64_t kind, uint32_t call) {
        return uint64_t(generation) | uint64_t(uint16_t(active)) << 32 | kind | uint64_t(call & 0xfffu) << 51;
    }
    uint32_t generation() const { return uint32_t(word); }
    int active() const { return int(uint16_t(word >> 32)); }
    bool is(uint64_t kind) const { return (word & kind) != 0; }
    uint32_t call() const { return uint32_t(word >> 51 & 0xfffu); }
};

// The waits at a node's boundaries, for the open at its start and for the close after the run's last phase, spin at
// least this long before they park.
constexpr int sink_node_spin_us = 20;
// How long a participant of a production sink polls for the next phase before it parks. The gaps between phases are
// a read or a GPU wait of a few milliseconds, and each park costs a wakeup the next phase waits for on every thread:
// on 60 threads of a KVM guest that took decode from 152 to 272 ms a token, where 1 ms already reaches the plateau
// and 10 ms gains nothing. Other machines gain a little more from 3 ms than from 1.
constexpr int sink_production_poll_us = 3000;

// The CPUs this process may run on; 1 when they cannot be read.
static int sink_process_cpus() {
    ssd::platform::cpu_list cpus;
    return ssd::platform::process_cpus(cpus) && !cpus.empty() ? int(cpus.size()) : 1;
}

static double sink_thread_cpu_us() { return double(ssd::platform::thread_cpu_time_ns()) * 1e-3; }

struct sink_worker_profile {
    double up_gate_wall_us = 0, down_wall_us = 0;
    double up_gate_cpu_us = 0, down_cpu_us = 0;
    double parallel_wall_us = 0, parallel_cpu_us = 0; // generic prefill callbacks
    double dispatch_us = 0, max_dispatch_us = 0;
    uint64_t phases = 0, no_work_wakeups = 0;
};

using sink_request = ssd::host::expert_request;

// Device facts are selected by L3 from this context's actual GPU backend.
static bool sink_unified_memory(const sink_configuration & value) {
    return value.has_gpu && value.gpu.device_is_unified(value.gpu_device);
}

static void sink_copy_services(sink_configuration & value, const ssd::runtime_services * services) {
    if (!services) { return; }
    value.host = services->host;
    value.source = services->source;
    value.gpu_transport = services->gpu_transport;
    value.events = services->events;
    if (services->debug) { value.debug = *services->debug; }
    if (services->gpu) {
        const auto * gpu = services->gpu;
        // The tables a tier needs are checked when it starts (sink_vram::initialize): a tier with no residency
        // serves grouped prefill alone and has fewer of them.
        if (gpu->version != 2 || gpu->size != sizeof(*gpu) || !gpu->api || !gpu->prefill || !gpu->settings || !gpu->stream ||
                !gpu->device_is_unified || !gpu->device_ordinal || !gpu->attach_services || services->gpu_device < 0) {
            throw std::runtime_error("GPU provider ABI missing or incompatible");
        }
        value.gpu = *gpu;
        value.gpu_device = services->gpu_device;
        value.has_gpu = true;
    }
}

template<class T> static T sink_host_value(ssd::result<T> result) {
    if (!result) { throw std::runtime_error(result.error); }
    return std::move(result.value);
}
static void sink_host_value(ssd::result<void> result) {
    if (!result) { throw std::runtime_error(result.error); }
}
static ssd::host::client_config sink_host_configuration(const sink_configuration & value) {
    ssd::host::client_config config;
    const auto * headroom = value.settings.integer("host.headroom_mib");
    const auto * reserve = value.settings.integer("vram.reserve_mib");
    if (value.production && (!headroom || !reserve)) { throw std::runtime_error("missing host budget settings"); }
    config.unified = sink_unified_memory(value);
    config.reserve_bytes = value.policy.cache_reserve_bytes;
    if (config.reserve_bytes < (512ull << 20)) { throw std::runtime_error("cache reserve must be at least 512 MiB"); }
    config.margin_bytes = std::max(config.reserve_bytes, uint64_t(headroom ? *headroom : 512) << 20);
    config.explicit_reserve_bytes = config.unified ? uint64_t(reserve ? *reserve : 0) << 20 : 0;
    config.reclaimable_file = value.policy.cache_reclaimable_file;
    if (const auto * budget = value.settings.integer("host.budget_mib")) { config.budget_bytes = uint64_t(*budget) << 20; }
    return config;
}
// Production borrows L3's context registration. Only the explicit legacy/test
// factory has an isolated owner; no production backend creates a budget singleton.

using sink_cache = ssd::host::expert_cache;
static ssd::host::expert_cache_options sink_cache_options(const sink_configuration & value) {
    ssd::host::expert_cache_options out;
    out.host = value.host;
    out.budget = sink_host_configuration(value);
    out.require_shared = value.production;
    const auto * tier = value.settings.vram_budget();
    out.gpu_tier = value.production ? tier && tier->tier_bytes(value.has_gpu) : value.policy.vram_budget != 0;
    const auto & policy = value.policy;
    out.cache_policy = policy.cache_policy;
    out.cache_allocation = policy.cache_allocation;
    out.cache_keep = policy.cache_keep;
    out.cache_prefill_read = policy.cache_prefill_read;
    out.cache_elastic = policy.cache_elastic;
    out.cache_partial_reclaim = policy.cache_partial_reclaim;
    out.cache_prefill_slots = policy.cache_prefill_slots;
    out.cache_layer_quotas = policy.cache_layer_quotas;
    return out;
}

#include "ssd/cpu-sink-vram.inc"

// The run lock and the thread holding it. The host arbiter runs the sink's reclaimer on the admitting thread: in a
// run of the sink (its scratch, ring and cache slots) the caches are that thread's already, and any other thread
// takes the lock first (kimi_cpu_sink::quiescent).
class sink_run_mutex {
public:
    void lock() { mutex.lock(); owner.store(std::this_thread::get_id(), std::memory_order_relaxed); }
    bool try_lock() {
        if (!mutex.try_lock()) { return false; }
        owner.store(std::this_thread::get_id(), std::memory_order_relaxed);
        return true;
    }
    void unlock() { owner.store(std::thread::id(), std::memory_order_relaxed); mutex.unlock(); }
    // only the holder ever stores its own id, and clears it before it unlocks
    bool held() const { return owner.load(std::memory_order_relaxed) == std::this_thread::get_id(); }
private:
    std::mutex mutex;
    std::atomic<std::thread::id> owner{};
};

struct sink_slot {
    sink_buffer role[3];
    enum state_type { empty, reading, ready } state = empty;
    unsigned remaining = 0;
    unsigned ready_roles = 0; // successful CQEs, protected by io_mutex
    size_t request = 0;
};

// An expert of the grouped prefill's read-ahead plan (kimi_cpu_sink::lookahead): its id, its three roles' sources as
// the planning call resolved them, and the RAM cache slot it lands in (prefetch_reserve: inflight, invisible to lookups
// and allocation, until the next layer's call publishes it). The I/O thread reads each role at `target`; its state is
// under io_mutex.
struct sink_lookahead_read {
    int32_t expert = -1;
    kimi_cpu_sink_source role[3] = {};
    int entry = -1;
    uint64_t generation = 0;
    void * target[3] = {};
    unsigned remaining = 0, ready_roles = 0;
    bool issued = false, failed = false;
};
// The CQE tags of read-ahead reads, above every ring slot's 3 * index + role: the plan's 3 * index + role from here.
constexpr uint64_t sink_lookahead_tag = uint64_t(1) << 32;

// Diagnostic only. All times use the scheduler's CLOCK_MONOTONIC microseconds.
// An I/O completion is when this process observes the CQE, not a device timestamp.
// The I/O (its thread, or a run that reads for itself) and each worker write disjoint fields; print only after
// io_done and the compute barriers. No extra synchronization is added to the data path.
struct alignas(64) sink_timeline_worker {
    uint64_t begin = 0, end = 0;
    double cpu_us = 0;
    int core = -1;
};

// The frequently modified row cursor gets a cache line of its own. alignas
// also rounds sizeof up to 64, so adjacent immutable scheduler fields cannot
// share its line.
struct alignas(64) sink_dynamic_row_counter {
    std::atomic<int> next{0};
};
static_assert(sizeof(sink_dynamic_row_counter) == 64, "dynamic row cursor must occupy one cache line");
struct sink_timeline_expert {
    std::array<uint64_t, 6> io = {}; // queue, submit begin/end, up/gate/down CQE
    std::array<uint64_t, 8> cpu = {}; // wait, up/gate, hidden quant, down: begin/end
    std::vector<sink_timeline_worker> workers; // up/gate then down, indexed by ith
};

} // namespace

// The median and the mean of a duration without keeping the samples: 32 bins per octave of 1 + us (2% wide).
struct sink_duration_histogram {
    std::array<uint64_t, 32 * 27> bins = {}; // up to 2^27 us
    uint64_t count = 0;
    double sum_us = 0;
    void add(double us) {
        us = std::max(0.0, us);
        ++bins[std::min(bins.size() - 1, size_t(32.0 * std::log2(1.0 + us)))];
        ++count;
        sum_us += us;
    }
    double median_us() const {
        uint64_t seen = 0;
        for (size_t i = 0; i < bins.size() && count; ++i) {
            if (2 * (seen += bins[i]) >= count) { return std::exp2((double(i) + 0.5) / 32.0) - 1.0; }
        }
        return 0;
    }
    double mean_us() const { return count ? sum_us / double(count) : 0; }
};

// Online miss-ratio curve of the decode expert stream. Each routed layer keeps an LRU stack of the experts it has
// seen (most recent first) and a histogram of stack distances, so the hit rate an LRU cache of any per-layer size
// would get is known while serving, for any model and without an offline trace. The k experts of one token are one
// access group: their distances are taken before any of them moves to the front. Cost: a linear scan of at most
// n_experts entries per selected expert. Explicit fixture policy enables this
// observer; production observation follows the context debug selection.
struct sink_mrc {
    bool enabled = false;
    uint64_t report_every = 256;
    std::vector<std::vector<int32_t>> stack;      // per layer
    std::vector<std::vector<uint64_t>> hist;      // per layer; bin d = stack distance d (first touches in cold)
    std::vector<int> layers;                      // layer indices seen, in first-seen order
    uint64_t tokens = 0, reported_tokens = 0;
    int first_layer = -1;
    std::vector<int> scratch;

    explicit sink_mrc(bool on) : enabled(on) {}

    void observe(int layer, const int32_t * ids, int n) {
        if (!enabled || layer < 0 || !ids || n <= 0) { return; }
        if (size_t(layer) >= stack.size()) { stack.resize(size_t(layer) + 1); hist.resize(size_t(layer) + 1); }
        if (first_layer < 0) { first_layer = layer; }
        if (layer == first_layer) { ++tokens; }
        auto & st = stack[size_t(layer)];
        auto & h = hist[size_t(layer)];
        if (st.empty() && h.empty()) { layers.push_back(layer); }
        scratch.assign(size_t(n), -1);
        for (int i = 0; i < n; ++i) {
            const int32_t e = ids[i];
            if (e < 0) { continue; }
            const auto it = std::find(st.begin(), st.end(), e);
            scratch[size_t(i)] = it == st.end() ? -1 : int(it - st.begin());
        }
        for (int i = 0; i < n; ++i) {
            if (ids[i] < 0) { continue; }
            const size_t bin = scratch[size_t(i)] < 0 ? SIZE_MAX : size_t(scratch[size_t(i)]);
            if (bin == SIZE_MAX) {
                cold_add(layer);  // first touch: a miss at every size
            } else {
                if (h.size() <= bin) { h.resize(bin + 1); }
                ++h[bin];
            }
        }
        for (int i = n - 1; i >= 0; --i) {
            const int32_t e = ids[i];
            if (e < 0) { continue; }
            const auto it = std::find(st.begin(), st.end(), e);
            if (it != st.end()) { st.erase(it); }
            st.insert(st.begin(), e);
        }
        if (tokens - reported_tokens >= report_every && layer == first_layer) { report("periodic"); }
    }

    std::vector<uint64_t> cold;
    void cold_add(int layer) {
        if (cold.size() <= size_t(layer)) { cold.resize(size_t(layer) + 1); }
        ++cold[size_t(layer)];
    }

    // hits an LRU of per-layer quota q would have had, summed over layers
    uint64_t hits_at(size_t q, int layer) const {
        const auto & h = hist[size_t(layer)];
        uint64_t sum = 0;
        for (size_t d = 0; d < h.size() && d < q; ++d) { sum += h[d]; }
        return sum;
    }
    uint64_t accesses(int layer) const {
        uint64_t sum = size_t(layer) < cold.size() ? cold[size_t(layer)] : 0;
        for (auto c : hist[size_t(layer)]) { sum += c; }
        return sum;
    }

    void report(const char * why) {
        if (!enabled || layers.empty() || tokens == reported_tokens) { return; }
        reported_tokens = tokens;
        static const size_t grid[] = { 1, 2, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512 };
        uint64_t total = 0;
        for (int l : layers) { total += accesses(l); }
        std::fprintf(stderr, "[kimi_cpu_sink_mrc] {\"schema\":1,\"why\":\"%s\",\"tokens\":%llu,\"layers\":%zu,"
                "\"accesses\":%llu,\"quota\":[", why, (unsigned long long) tokens, layers.size(),
                (unsigned long long) total);
        for (size_t i = 0; i < sizeof(grid) / sizeof(grid[0]); ++i) { std::fprintf(stderr, "%s%zu", i ? "," : "", grid[i]); }
        std::fprintf(stderr, "],\"hit_uniform\":[");
        for (size_t i = 0; i < sizeof(grid) / sizeof(grid[0]); ++i) {
            uint64_t hits = 0;
            for (int l : layers) { hits += hits_at(grid[i], l); }
            std::fprintf(stderr, "%s%.4f", i ? "," : "", total ? double(hits) / double(total) : 0.0);
        }
        std::fprintf(stderr, "],\"per_layer_hit_q16\":[");
        bool first = true;
        for (int l : layers) {
            const uint64_t a = accesses(l);
            std::fprintf(stderr, "%s%.3f", first ? "" : ",", a ? double(hits_at(16, l)) / double(a) : 0.0);
            first = false;
        }
        std::fprintf(stderr, "]}\n");
    }
};

struct kimi_cpu_sink {
    const sink_configuration configuration;
    const int ring_experts;
    kimi_cpu_sink_resolver resolver;
    void * userdata;
    sink_run_mutex run_mutex;
    sink_ring ring;
    std::vector<sink_slot> slots;
    // The read ring's buffers page-locked for the GPU tier's copies (ring_lock, kimi-host-lock.h): once per
    // allocation, undone before the buffers are reallocated or freed.
    kimi_ring_lock ring_pages;
    std::vector<sink_request> requests;
    std::vector<size_t> io_requests;
    sink_cache cache;
    sink_vram vram;
    uint64_t ledger_refusals = 0; // make_room answers that were no
    bool margin_boundary = false; // the last token boundary the margin saw was a decode token
    uint64_t margin_epoch = 0, margin_token = 0;

    // A resolved source carries its file's identity. One layer resolves dozens of roles and reading the identity takes
    // system calls, enough on Windows to show in a token's time. So it is read once per descriptor and logical token.
    // kimi_cpu_sink_cache_begin forgets it, which still catches a source replaced between tokens, at the same descriptor
    // too. run_mutex guards the list: every caller of resolve and kimi_cpu_sink_cache_begin holds it.
    std::vector<std::pair<int, ssd::platform::file_identity>> file_identities;
    bool resolve(const char * tensor, int32_t expert, kimi_cpu_sink_source * source) {
        if (!resolver(userdata, tensor, expert, source)) { return false; }
        for (const auto & known : file_identities) {
            if (known.first == source->fd_direct) { source->file = known.second; return true; }
        }
        if (!ssd::platform::identity(source->fd_direct, source->file)) { return false; }
        file_identities.emplace_back(source->fd_direct, source->file);
        return true;
    }

    // Next-layer expert prefetch (typed prefetch.depth). The graph applies the next
    // layer's router to this layer's expert input and hands its experts over as a hint, best first; the first k
    // that are neither cached nor held by the GPU tier now (its residency ABI) are read into RAM cache slots of the
    // next layer once this layer's own reads are done, so the drive works through the gap before the next router
    // runs. At any depth, 0 included, every rank of the hint is counted (prefetch_report): how often its expert was
    // novel, how often the next layer selected a novel one, and how often a read one was used.
    struct prefetch_job {
        int32_t layer = -1, expert = -1;
        int rank = -1;        // in the hint
        bool checked = false; // seen by the run of its layer (used_r)
        int entry = -1;
        uint64_t generation = 0;
        kimi_cpu_sink_source role[3] = {};
        char * target[3] = {};
        int done = 0, running = 0, queued = 0; // roles read / being read / waiting, under the prefetch mutex
        bool failed = false;
        bool activated = false; // published as a hit of a run that waits for the rest of its reads
        bool started = false;   // a read of one of its roles started (started_r)
        // The only field a run's scheduler reads without the prefetch mutex: 0 while the reads may still complete,
        // 1 once all three roles are read, 2 once the job cannot complete (a read failed or a role was dropped).
        std::atomic<int> state{0};
    };
    struct prefetch_state {
        int topk = 0;
        int32_t hint[64] = {}; // the predicted experts of layer hint_layer, best first, for the coming run
        int hint_n = 0;
        int32_t hint_layer = -1;
        std::mutex mutex;
        std::condition_variable cv;
        std::vector<std::unique_ptr<prefetch_job>> jobs; // unfinished jobs; a reader holds one only while it reads
        std::deque<std::pair<prefetch_job *, int>> queue;  // (job, role) of the next layer, read after this run's reads
        std::deque<std::pair<prefetch_job *, int>> urgent; // roles of experts the current run uses, read at once
        std::vector<std::pair<int, prefetch_job *>> pending; // (cache entry, job): hits of the current run still reading
        std::vector<std::thread> readers;
        bool gate_open = false, stop = false;
        int running = 0;
        uint64_t fail_every = 0, reads = 0; // explicit fixture fault injection
        // Explicit fixture policy can fail the n-th drain/schedule allocation.
        uint64_t throw_drain = 0, throw_schedule = 0, allocs_drain = 0, allocs_schedule = 0;
        uint64_t layers = 0, predicted = 0, issued = 0, ready = 0, cancelled = 0, failed = 0,
                 skip_vram = 0, skip_cached = 0, no_slot = 0, unresolved = 0, carried = 0, activated = 0;
        // Reads of experts the next run selected but the GPU tier came to hold since they were scheduled: not needed,
        // the tier serves them. And the converse: experts skipped because the tier held them that the next run
        // selected after the tier gave them back (a reclaim between the two runs), read on demand instead.
        uint64_t vram_late = 0, skip_vram_lost = 0;
        // Per rank of the hint: given, novel (neither held by the GPU tier nor in the RAM cache), novel and selected
        // by the next layer, read, and read and taken from the RAM cache by the next layer.
        uint64_t predicted_r[64] = {}, novel_r[64] = {}, novel_routed_r[64] = {}, issued_r[64] = {}, used_r[64] = {};
        int width = 0; // ranks of the widest hint
        // The reads that the current run selected (rank, expert), until its prepare says where it takes them from.
        std::pair<int, int32_t> selected_reads[64] = {};
        int selected_n = 0;
        // The last decode run's hint and which of its ranks were novel, until the run of the layer it predicts.
        int32_t last_layer = -1;
        int last_n = 0;
        int32_t last_hint[64] = {};
        bool last_novel[64] = {}, last_skipped[64] = {}; // last_skipped: within the depth and held by the tier
        // The drive's idle window per hinted layer: from the end of a run's own reads (its schedule, without any)
        // to the start of the run of the layer its hint predicts. The window the prefetch reads have, at any depth.
        sink_clock::time_point window_start;
        bool window_open = false;
        sink_duration_histogram window;
        // The readers: bytes read, and the wall time with at least one read running.
        uint64_t reader_bytes = 0;
        double reader_busy_us = 0;
        sink_clock::time_point busy_since;
        // What the reads cost the demand reads: the read each reader runs now, and whether it spills, i.e. runs past
        // the window's close for an expert the run that closed it does not take (an urgent read, or one of an expert
        // that run computes on the CPU, is that run's own read). spill_us is the wall time with at least one spilling
        // read; spills counts the reads that spilled, each once however many windows it runs past (spilled), spill_us_r
        // adds each one's own time past a close to the rank of its expert in the hint, and started_r counts per rank
        // the reads (experts) that began.
        struct read_state {
            prefetch_job * job = nullptr;
            bool urgent = false, spill = false, spilled = false;
            sink_clock::time_point since;
        } reading[3];
        int spilling = 0;
        double spill_us = 0;
        uint64_t spills = 0;
        sink_clock::time_point spill_since;
        uint64_t started_r[64] = {}, spill_us_r[64] = {};
    } prefetch;

    // Unified memory: the ledger's floor at every point, not only at decode token boundaries (floor_watch_loop).
    std::mutex watch_mutex;
    std::condition_variable watch_cv;
    std::thread watch_thread;
    bool watch_stop = false;

    std::mutex io_mutex;
    std::condition_variable io_cv;
    std::thread io_thread;
    bool io_stop = false, io_job = false, io_done = true, io_cancel = false, poisoned = false;
    std::string io_error;
    // A run's reads: the next of io_requests to queue, those submitted and not completed, whether one failed or the
    // run was cancelled (none is queued after that), and when they started. The I/O thread works them through
    // (io_execute), or a decode run that reads for itself (io_by_run, until they are over: io_poll). The ring has one
    // owner at a time.
    size_t io_next = 0;
    unsigned io_inflight = 0;
    bool io_failed = false, io_by_run = false;
    sink_clock::time_point io_start;
    kimi_cpu_sink_stats current = {};
    // The ring's counters when a run's own reads ended, before any read-ahead (io_mutex): a grouped-prefill call reports
    // its completions from them, as the next layer's read-ahead runs on the ring beside its tail.
    uint64_t io_completed_end = 0, io_pending_end = 0;

    // The grouped prefill's cross-layer read-ahead (prefill.lookahead). In a prompt pass the drive idles from the end
    // of a layer's own reads (io_end) to the first read of the next layer's call, while the call finishes its compute
    // and the GPU computes the next layer's attention and router. The call for layer L plans the next layer's experts
    // that its predicted routes make likely and that neither the GPU tier nor the RAM cache holds (lookahead_plan_next):
    // each takes a slot of the RAM cache for that layer, as the decode prefetch does, and once L's own reads are over the
    // I/O thread reads them into those slots through the ring (lookahead_run): reads moved earlier, none added. The call
    // for L + 1 stops it before its prepare, waits for the reads in flight and publishes what landed (lookahead_handover),
    // and its prefill read of the RAM cache takes them as hits. The read-ahead holds no memory of its own: what it read
    // stays cached, and what the next layer did not use is the coldest of what the cache holds. A plan holds as many
    // experts as the drive reads in the idle window at its calibrated rate: the window's median over the pass, the last
    // pass's for its first layers.
    struct lookahead_state {
        bool enabled = false;
        uint64_t fail_every = 0, completions = 0; // explicit fixture fault injection: every n-th completion fails
        // The plan of `layer`: set by the call of the layer before it, before it publishes its job; the I/O thread
        // reads it while `active` (io_mutex). next, inflight and batch are the I/O thread's.
        int32_t layer = -1;
        std::vector<sink_lookahead_read> plan;
        size_t next = 0;
        unsigned inflight = 0;                     // role reads in flight
        std::vector<std::pair<size_t, int>> batch; // (plan index, role) of the reads one step queues
        bool active = false, stop = false;
        int32_t job_layer = -1;                    // the grouped-prefill call of the job the I/O thread runs
        // The drive's idle window (io_mutex): from the end of a grouped-prefill call's own reads to the start of the next
        // layer's call of the same pass, the time the read-ahead has. What the next call does before its own reads (its
        // handover, prepare and plan) is not in it: the read-ahead is stopped then.
        sink_clock::time_point window_start;
        int32_t window_layer = -1;
        bool window_open = false;
        int32_t pass_layer = -1;                   // the last grouped-prefill call's layer: a pass runs in layer order
        std::vector<double> pass_windows_us;       // this pass's windows so far
        double last_pass_median_us = 0;
        // The plan in progress (io_mutex): experts issued, landed whole, failed.
        uint64_t issued = 0, landed = 0, failed = 0;
        // The cache slots the handover published for the call in progress: its prefill read counts the hits among them.
        std::vector<int> landed_entries;
        // Totals for the report at destruction.
        uint64_t layers = 0, planned = 0, issued_total = 0, landed_total = 0, failed_total = 0, used = 0, unused = 0,
                 used_bytes = 0, no_slot = 0, unresolved = 0, unlikely = 0, borrowed = 0;
        // The prediction of the next layer's routes (kimi_cpu_sink_prefill_hint): `hint_layer`'s experts, hint_k a
        // token best first over hint_tokens tokens. The plan keeps its copy (`planned_hint`) until the call of the layer it
        // predicts, which measures it against the routes the layer took.
        int32_t hint_layer = -1, planned_layer = -1;
        int hint_k = 0, hint_tokens = 0, planned_k = 0, planned_tokens = 0;
        std::vector<int32_t> hint, planned_hint;
        // Per rank of a prediction, over the prefill passes so far: the tokens predicted, and those whose prediction at
        // that rank the layer selected.
        uint64_t rank_predicted[16] = {}, rank_selected[16] = {};
    } lookahead;
    // One grouped-prefill call's receipts ([kimi_cpu_prefill_lookahead]); the call's thread only.
    struct lookahead_receipt {
        int32_t layer = -1;
        double window_us = -1, estimate_us = 0, drive_us_per_mb = 0;
        uint64_t planned = 0, issued = 0, landed = 0, failed = 0, used = 0, unused = 0, bytes = 0, next_n = 0;
        uint64_t likely = 0;  // the next layer's experts with P >= 1/2 that neither the GPU tier nor the RAM cache holds
        uint64_t no_slot = 0; // the next layer's plan stopped short: no slot of its own or of a finished layer was left
        uint64_t borrowed = 0; // the next layer's plan reads into slots of layers this pass finished
        size_t next_planned = 0;
    } lookahead_report;

    const bool ready_first;
    const bool early_up_gate;
    const bool role_major;
    const bool nonblocking_down_sched;
    const int poll_us;
    const bool dynamic_rows;
    const int dynamic_chunk_rows;
    sink_dynamic_row_counter dynamic_row_counter;
    const std::vector<double> row_weights;
    const std::vector<int> io_cpus;
    std::vector<int> up_bounds, down_bounds;
    std::vector<float> expert_output;
    std::vector<uint8_t> expert_computed;
    std::vector<uint8_t> expert_stage, expert_upload_started;
    // Measured CPU time of each decode expert of the layer, and its running mean on this machine.
    std::vector<double> expert_cpu_us;
    ssd::place::decode_learning learning;
    // The decode plan of the current layer: which experts the GPU computes, from the evaluator that also places
    // grouped prefill (kimi-prefill-cost.h, the in-ring list schedule) over the measured lines.
    struct decode_plan_state : ssd::place::decode_plan {
        sink_clock::time_point start; // executor timestamp; policy receives elapsed feedback only
    } plan;
    // Receipts: planned layers, tokens, experts the GPU computed from host memory, refused batches, and the relative
    // error of the predicted lanes' time against the measured one, in 1% bins (the last bin is 100% and more).
    struct decode_place_stats {
        uint64_t layers = 0, tokens = 0, gpu_experts = 0, refused = 0, moves_planned = 0, candidates = 0, wanted = 0;
        uint64_t pending = 0, pending_moves = 0;
        uint64_t last_token = UINT64_MAX;
        std::array<uint64_t, 101> error = {};
        double predicted_us = 0, measured_us = 0, cpu_only_us = 0;
    } place;
    // Page-locking of the RAM cache's slots for the GPU tier's copies, on a discrete GPU (kimi-host-lock.h): the slots
    // the decode plan would move an expert from, one at a time, in the drive's shadow; unlocked before their pages go
    // (sink_cache::before_discard).
    kimi_host_lock host_lock;
    // A diagnostic switch for A/B runs; on by default.
    const bool host_gpu_enabled = true;
    std::vector<uint8_t> staged_quant_hidden;
    const bool profile_enabled;
    const bool trace_enabled;
    const bool timeline_enabled;
    bool timeline_active = false;
    std::array<uint64_t, 6> timeline_call = {}; // enter, prepare end, quant end, job, I/O end, exit
    std::vector<sink_timeline_expert> timeline;
    uint64_t trace_epoch = 0, trace_token = 0, trace_call = 0;
    bool trace_has_token = false, trace_decode = false;
    int trace_layer = -1;
    sink_mrc mrc;
    double caller_cpu_start = 0;
    sink_clock::time_point phase_dispatch;
    std::vector<sink_worker_profile> worker_profiles; // one per participant of the open call
    // The call's participants (kimi_cpu_sink_call_open) take no lock. The coordinator publishes each phase, and the
    // call's open and close, as one word; joiners read it, compute their share of an active phase and park on
    // `participants` in between, and the coordinator parks on `coordinator` for completions and arrivals.
    sink_event participants, coordinator;
    alignas(64) std::atomic<uint64_t> phase{0};   // sink_phase
    alignas(64) std::atomic<uint32_t> pending{0}; // active joiners still computing the published phase
    alignas(64) std::atomic<uint32_t> joined{0};  // joiners of the open call
    // The coordinator's state, written before the phase word that publishes it: whether a call is open, its
    // participants, its id and the generation it opened at, and the latest phase generation.
    bool call_open = false;
    int call_participants = 1;
    uint32_t call_id = 0, call_generation = 0, generation = 0;
    // Participants of the run in progress: the open call's, or the coordinator alone. Set by the run's thread
    // before its first phase.
    int threads = 1;
    // Participants of a compute phase: every thread of the call for a parallel phase or row weights, else
    // run_participants.
    int active_threads = 0;
    // The participants this run's phases use (decode_threads_setup); the call thread count the decode one was
    // reported for.
    int run_participants = 0, decode_threads_for = 0;
    // What a decode graph leaves to the machine's other work (decode_threads): over decode tokens in a row, the busy
    // time of the CPUs this process may run on that the graph's threads did not take, and the wall time
    // (measure_decode_share). Each thread adds the CPU time it used since its previous call to its index
    // (mark_thread_cpu); a non-decode boundary starts a new generation, after which a thread starts over. Before any
    // measurement the reserve is the least one can give: the kernel and other threads always run, and any work at
    // all takes a CPU.
    struct decode_share {
        ssd::platform::cpu_list cpus;
        bool readable = true;
        int run = 0, reserve = 1, reported = -1;
        uint64_t epoch = 0, token = 0, busy = 0, pool = 0, wall = 0, tick = 0, gap = 0, last_call = 0;
        double other_ns = 0, wall_ns = 0, error_ns = 0;
    } share;
    std::unique_ptr<std::atomic<uint64_t>[]> thread_cpu{ new std::atomic<uint64_t>[sink_max_participants]() };
    std::atomic<uint64_t> cpu_generation{0};
    kimi_cpu_sink_parallel_fn parallel_fn = nullptr;
    void * parallel_userdata = nullptr;
    int latent = 0, hidden = 0;
    size_t row_bytes_up = 0, row_bytes_down = 0;
    // One expert's share of a compute phase: its weights and routing weight, its hidden row (and SWIGLU's u row) in
    // the phase's scratch, its output row and its quantized hidden. A phase computes the rows of every item.
    struct sink_work_item {
        size_t expert = 0;
        const char * up = nullptr, * gate = nullptr, * down = nullptr;
        float weight = 0;
        float * hidden = nullptr, * hidden_up = nullptr;
        float * output = nullptr;
        uint8_t * down_quant = nullptr;
    };
    std::vector<sink_work_item> phase_items;
    // A dynamic phase's chunks: item k's are work_chunks[k] .. work_chunks[k + 1] - 1.
    std::vector<int> work_chunks;
    float beta = 0, linear_beta = 0;
    const float * post_exp_norm = nullptr;
    int activation_kind = KIMI_CPU_SINK_ACTIVATION_SITU;
    float swiglu_limit = 0;
    float * output = nullptr;
    bool first_down = false; // without READY_FIRST: this down phase starts the slot-order sum
    // The hidden rows of a phase's items, one after another: g, or the activation.
    std::vector<float> activation;
    std::vector<float> activation_up; // SWIGLU: u, beside g in activation, until quantize_activation
    std::vector<uint8_t> quant_input, quant_hidden;
    size_t quant_hidden_stride = 0;
    // Selected from the expert tensor types by prepare(); up and gate share one.
    ggml_vec_dot_t dot_up = nullptr, dot_down = nullptr;
    ggml_from_float_t quantize_input = nullptr, quantize_hidden = nullptr;

    kimi_cpu_sink(int depth, kimi_cpu_sink_resolver resolve, void * user, const sink_configuration & input = {}) :
        configuration(input), ring_experts(depth), resolver(resolve), userdata(user), slots(depth),
        cache(sink_cache_options(configuration)),
        ready_first(configuration.policy.ready_first),
        early_up_gate(configuration.policy.early_up_gate), role_major(configuration.policy.role_major),
        nonblocking_down_sched(configuration.policy.nonblocking_down_sched), poll_us(configuration.policy.poll_us),
        dynamic_rows(configuration.policy.dynamic_rows), dynamic_chunk_rows(configuration.policy.dynamic_chunk_rows),
        row_weights(configuration.policy.row_weights),
        io_cpus(sink_cpus(configuration.policy.io_cpus)),
        profile_enabled(configuration.debug.profile), trace_enabled(configuration.debug.trace),
        timeline_enabled(configuration.debug.trace), mrc(configuration.debug.mrc), worker_profiles(1) {
        if (poll_us < 0 || poll_us > 10000 || dynamic_chunk_rows < 16 || dynamic_chunk_rows > 16384 || dynamic_chunk_rows % 16 ||
                configuration.policy.decode_participants < 0 || configuration.policy.decode_participants > sink_max_participants) {
            throw std::runtime_error("invalid explicit executor schedule");
        }
        if (configuration.policy.prefetch_depth < 0 || configuration.policy.prefetch_depth > 64) {
            throw std::runtime_error("prefetch depth must be 0..64");
        }
        if (!ssd::platform::process_cpus(share.cpus)) { share.cpus.clear(); }
        if (configuration.policy.prefetch_depth && (!ready_first || !nonblocking_down_sched)) {
            throw std::runtime_error("next-layer prefetch requires the ordered nonblocking scheduler");
        }
        for (double value : row_weights) {
            if (!std::isfinite(value) || value <= 0) { throw std::runtime_error("invalid explicit row weight"); }
        }
        if (configuration.events) {
            const auto healthy = configuration.events->flush();
            if (!healthy) { throw std::runtime_error(healthy.error); }
        }
        // Kernels are selected per call from the expert tensor types in prepare().
        if (dynamic_rows && !row_weights.empty()) {
            throw std::runtime_error("DYNAMIC_ROWS and ROW_WEIGHTS are mutually exclusive row schedulers");
        }
        vram.initialize(ready_first, configuration);
        refresh_floor("init");
        if (vram.deferred_upload) {
            cache.before_release = [](void * p) { static_cast<sink_vram *>(p)->wait_deferred(); };
            cache.before_release_arg = &vram;
        }
        cache.before_discard = [](void * p, size_t first, size_t count) { static_cast<kimi_cpu_sink *>(p)->host_unlock(first, count); };
        cache.before_discard_arg = this;
        vram.ram_slot_locked = [this](int entry) { return entry >= 0 && host_lock.is_locked(size_t(entry)); };
        lookahead.enabled = configuration.policy.prefill_lookahead;
        lookahead.fail_every = configuration.policy.lookahead_fail_every;
        // the reads into the arena (the decode prefetch's, the grouped prefill's read-ahead) land before it goes
        cache.before_unmap = [](void * p) {
            auto * sink = static_cast<kimi_cpu_sink *>(p);
            sink->prefetch_wait_reads();
            sink->lookahead_unmap();
        };
        cache.before_unmap_arg = this;
        if (role_major && !early_up_gate) {
            throw std::runtime_error("ROLE_MAJOR requires EARLY_UP_GATE=1");
        }
        if (nonblocking_down_sched && !role_major) {
            throw std::runtime_error("NONBLOCKING_DOWN_SCHED requires ROLE_MAJOR=1");
        }
        if (nonblocking_down_sched && !ready_first) {
            throw std::runtime_error("NONBLOCKING_DOWN_SCHED requires READY_FIRST=1");
        }
        sink_io_value(ring.init(static_cast<unsigned>(3 * depth), configuration.policy.read_fixed));
        try {
            io_thread = std::thread([this] {
                (void) ssd::platform::set_thread_role(ssd::platform::thread_role::io);
                io_loop();
            });
            sink_pin(io_thread, io_cpus);
            if (cache.ledger.state().unified && vram.context) {
                watch_thread = std::thread([this] {
                    (void) ssd::platform::set_thread_role(ssd::platform::thread_role::background);
                    floor_watch_loop();
                });
                sink_pin(watch_thread, io_cpus);
            }
        } catch (...) {
            stop_threads();
            throw;
        }
        std::fprintf(stderr, "[kimi_cpu_sink_schedule] ready_first=%d early_up_gate=%d role_major=%d nonblocking_down_sched=%d poll_us=%d dynamic_rows=%d dynamic_chunk_rows=%d row_weights=",
                     ready_first ? 1 : 0, early_up_gate ? 1 : 0, role_major ? 1 : 0,
                     nonblocking_down_sched ? 1 : 0, poll_us, dynamic_rows ? 1 : 0, dynamic_chunk_rows);
        for (size_t i = 0; i < row_weights.size(); ++i) { std::fprintf(stderr, "%s%.9g", i ? "," : "", row_weights[i]); }
        std::fprintf(stderr, " io_cpus=");
        for (size_t i = 0; i < io_cpus.size(); ++i) { std::fprintf(stderr, "%s%d", i ? "," : "", io_cpus[i]); }
        std::fprintf(stderr, "\n");
        try {
            prefetch_start();
            // last: from here on any admission of the context may run it
            sink_host_value(cache.ledger.client->set_reclaimer(reclaimer()));
        } catch (...) {
            prefetch_stop();
            stop_threads();
            throw;
        }
        if (prefetch.topk) { std::fprintf(stderr, "[kimi_cpu_sink_prefetch] enabled topk=%d readers=3\n", prefetch.topk); }
        if (lookahead.enabled) { std::fprintf(stderr, "[kimi_cpu_prefill_lookahead] enabled=1\n"); }
    }

    // The context's reclaimer (ssd::host::actions), which the host arbiter runs for every admission of the context
    // that does not fit: the GPU tier's deferred uploads and blocks, then the RAM cache's slots. full_boost: the
    // arbiter admits at the full watermark boost, so the cache gives back until the room fits as measured there.
    ssd::host::actions reclaimer() {
        ssd::host::actions out;
        out.collect_deferred = [this] { quiescent([this] { vram.collect_deferred(cache, KIMI_VRAM_COLLECT_IF_IDLE); return true; }); };
        out.reclaim_tier = [this](uint64_t bytes) { return quiescent([&] { return vram.reclaim_host(bytes); }); };
        out.reclaim_cache = [this](uint64_t bytes) {
            return quiescent([&] { return cache.arena && cache.reclaim_for_guard(bytes, "host_room", true); });
        };
        return out;
    }
    // Runs a reclaimer's step with the caches safe to change: as it is on the thread in a run of this sink, and on any
    // other thread under the run lock with the prefetch drained, as every entry runs.
    template<class F> bool quiescent(F && step) {
        if (run_mutex.held()) { return step(); }
        std::lock_guard<sink_run_mutex> serial(run_mutex); prefetch_drain();
        return step();
    }

    void prefetch_start() {
        prefetch.topk = configuration.policy.prefetch_depth;
        cache.staging_per_layer = std::max(1, prefetch.topk);
        if (!prefetch.topk) { return; }
        prefetch.fail_every = configuration.policy.prefetch_fail_every;
        prefetch.throw_drain = configuration.policy.prefetch_throw_drain;
        prefetch.throw_schedule = configuration.policy.prefetch_throw_schedule;
        prefetch.readers.reserve(3);
        for (int i = 0; i < 3; ++i) {
            prefetch.readers.emplace_back([this, i] {
                (void) ssd::platform::set_thread_role(ssd::platform::thread_role::io);
                prefetch_reader(i);
            });
            sink_pin(prefetch.readers.back(), io_cpus);
        }
    }
    void prefetch_stop() {
        {
            std::lock_guard<std::mutex> lock(prefetch.mutex);
            prefetch.stop = true;
            prefetch.queue.clear();
        }
        prefetch.cv.notify_all();
        for (auto & t : prefetch.readers) { if (t.joinable()) { t.join(); } }
        prefetch.readers.clear();
    }
    void prefetch_reader(int reader) {
        std::unique_lock<std::mutex> lock(prefetch.mutex);
        auto & self = prefetch.reading[reader];
        for (;;) {
            prefetch.cv.wait(lock, [&] {
                return prefetch.stop || !prefetch.urgent.empty() || (prefetch.gate_open && !prefetch.queue.empty());
            });
            if (prefetch.stop) { return; }
            const bool urgent = !prefetch.urgent.empty();
            auto & from = urgent ? prefetch.urgent : prefetch.queue;
            const auto item = from.front();
            from.pop_front();
            prefetch_job & job = *item.first;
            if (!job.started) { job.started = true; ++prefetch.started_r[job.rank]; }
            self.job = &job; self.urgent = urgent; self.spill = self.spilled = false;
            const kimi_cpu_sink_source source = job.role[item.second];
            char * target = job.target[item.second];
            if (!prefetch.running++) { prefetch.busy_since = sink_clock::now(); }
            ++job.running;
            --job.queued;
            const bool inject = prefetch.fail_every && ++prefetch.reads % prefetch.fail_every == 0;
            lock.unlock();
            bool ok = !inject;
            size_t done = 0;
            while (ok && done < source.nbytes) {
                const int64_t got = ssd::platform::pread_some(source.fd_direct, target + done, source.nbytes - done,
                    source.offset + done);
                if (got <= 0) { ok = false; break; }
                done += size_t(got);
                if (source.payload_nbytes && done >= source.data_offset + source.payload_nbytes) { break; }
            }
            lock.lock();
            if (self.spill) { prefetch_end_spill(self, sink_clock::now()); }
            self.job = nullptr;
            if (ok) { ++job.done; } else { job.failed = true; }
            prefetch.reader_bytes += done;
            if (!--prefetch.running) { prefetch.reader_busy_us += elapsed_us(prefetch.busy_since); }
            --job.running;
            prefetch_settle(job);
            prefetch.cv.notify_all();
            // a run may wait for this expert as a cache hit
            lock.unlock();
            { std::lock_guard<std::mutex> io_lock(io_mutex); }
            io_cv.notify_all();
            lock.lock();
        }
    }
    static void prefetch_alloc(uint64_t at, uint64_t & count) {
        if (at && ++count == at) { throw std::bad_alloc(); }
    }
    // Under the prefetch mutex: a spilling read ends (it completed, or a new window opened while it ran).
    void prefetch_end_spill(prefetch_state::read_state & r, sink_clock::time_point now) {
        r.spill = false;
        prefetch.spill_us_r[r.job->rank] += uint64_t(std::llround(std::chrono::duration<double, std::micro>(now - r.since).count()));
        if (!--prefetch.spilling) {
            prefetch.spill_us += std::chrono::duration<double, std::micro>(now - prefetch.spill_since).count();
        }
    }
    // Under the prefetch mutex, after any change of a job's counts: publish its terminal state once it has one.
    static void prefetch_settle(prefetch_job & job) {
        if (job.state.load() != 0) { return; }
        if (job.failed) { job.state = 2; }
        else if (job.done == 3) { job.state = 1; }
        else if (job.running == 0 && job.queued == 0) { job.state = 2; }
    }
    // When a run's own reads are done (io_end), or when a decode run without any schedules: the drive's idle window
    // starts, and the reads of the next layer may begin.
    void prefetch_open_gate() {
        {
            std::lock_guard<std::mutex> lock(prefetch.mutex);
            prefetch.window_start = sink_clock::now();
            prefetch.window_open = true;
            if (!prefetch.topk) { return; }
            prefetch.gate_open = true;
            // a read still running from before is in this window now
            for (auto & r : prefetch.reading) { if (r.spill) { prefetch_end_spill(r, prefetch.window_start); } }
        }
        prefetch.cv.notify_all();
    }
    // Every entry that holds the run mutex starts here. Reads of the next layer not started yet are dropped, and
    // jobs whose reads are over are published. A decode run (layer and ids given) does not wait: an expert it
    // computes on the CPU (it selects it and the GPU tier does not hold it) that is still being read becomes a hit
    // it waits for like one of its own reads, its remaining roles read at once; other reads go on in the background
    // into inflight slots. Every other entry waits for all reads; that full drain allocates nothing, as it runs
    // outside the entries' try blocks and in their error handling.
    void prefetch_drain(int32_t layer = -1, const int32_t * ids = nullptr, int nids = 0) {
        uint8_t held[1024] = {}; // per id: the tier holds it, so this run's prepare pins it as a hit
        if (prefetch.topk && ids && nids > 0 && nids <= 1024) { vram.held(layer, ids, nids, held); }
        std::unique_lock<std::mutex> lock(prefetch.mutex);
        prefetch_check(layer, ids, nids, held);
        if (!prefetch.topk) { return; }
        prefetch.pending.clear();
        prefetch.gate_open = false;
        auto needed = [&](const prefetch_job & job) {
            if (!ids || job.layer != layer || job.failed) { return false; }
            for (int i = 0; i < nids; ++i) { if (ids[i] == job.expert && !held[i]) { return true; } }
            return false;
        };
        // the window closes: the reads running for experts this entry does not take spill from now on (a read still
        // running from an earlier window spills again, and counts once)
        const auto closed = sink_clock::now();
        for (auto & r : prefetch.reading) {
            if (!r.job || r.urgent || r.spill || needed(*r.job)) { continue; }
            r.spill = true; r.since = closed;
            if (!r.spilled) { r.spilled = true; ++prefetch.spills; }
            if (!prefetch.spilling++) { prefetch.spill_since = closed; }
        }
        if (ids) {
            // every allocation comes before the first change, so a failure leaves the queues for the full drain
            std::deque<std::pair<prefetch_job *, int>> urgent;
            for (auto * q : { &prefetch.urgent, &prefetch.queue }) {
                for (const auto & item : *q) {
                    if (needed(*item.first)) {
                        prefetch_alloc(prefetch.throw_drain, prefetch.allocs_drain);
                        urgent.push_back(item);
                    }
                }
            }
            prefetch.pending.reserve(prefetch.jobs.size());
            for (auto * q : { &prefetch.urgent, &prefetch.queue }) {
                for (const auto & item : *q) {
                    if (!needed(*item.first)) { --item.first->queued; prefetch_settle(*item.first); }
                }
                q->clear();
            }
            prefetch.urgent.swap(urgent);
        } else {
            for (auto * q : { &prefetch.urgent, &prefetch.queue }) {
                for (const auto & item : *q) { --item.first->queued; prefetch_settle(*item.first); }
                q->clear();
            }
            prefetch.cv.wait(lock, [&] { return prefetch.running == 0; });
        }
        for (auto it = prefetch.jobs.begin(); it != prefetch.jobs.end();) {
            prefetch_job & job = **it;
            if (job.running || job.queued) {
                if (needed(job) && !job.activated && job.state.load() == 0 && job.done + job.running + job.queued == 3 &&
                        cache.prefetch_activate(job.entry, job.generation)) {
                    job.activated = true;
                    prefetch.pending.emplace_back(job.entry, &job);
                    ++prefetch.activated;
                }
                ++prefetch.carried;
                ++it;
                continue;
            }
            const bool ok = !job.failed && job.done == 3;
            if (ok) { ++prefetch.ready; } else if (job.failed) { ++prefetch.failed; } else { ++prefetch.cancelled; }
            cache.prefetch_done(job.entry, job.generation, ok);
            it = prefetch.jobs.erase(it);
        }
        const bool urgent_reads = !prefetch.urgent.empty();
        lock.unlock();
        if (urgent_reads) { prefetch.cv.notify_all(); }
    }
    // Under the prefetch mutex, as an entry starts: a decode run (layer and ids given, held[i] whether the GPU tier
    // holds ids[i]) counts the ranks of the last hint that were novel, notes the reads that it selects, and ends the
    // drive's idle window; any other entry ends the layer sequence.
    void prefetch_check(int32_t layer, const int32_t * ids, int nids, const uint8_t * held) {
        auto selected = [&](int32_t expert) { return expert >= 0 && std::find(ids, ids + nids, expert) != ids + nids; };
        auto tier_holds = [&](int32_t expert) {
            const ptrdiff_t at = std::find(ids, ids + nids, expert) - ids;
            return at < nids && at < 1024 && held[at];
        };
        if (ids && prefetch.last_layer == layer) {
            for (int i = 0; i < prefetch.last_n; ++i) {
                prefetch.novel_routed_r[i] += prefetch.last_novel[i] && selected(prefetch.last_hint[i]);
                prefetch.skip_vram_lost += prefetch.last_skipped[i] && selected(prefetch.last_hint[i]) &&
                    !tier_holds(prefetch.last_hint[i]);
            }
            if (prefetch.window_open) { prefetch.window.add(elapsed_us(prefetch.window_start)); }
        }
        if (ids) {
            prefetch.selected_n = 0;
            for (auto & job : prefetch.jobs) {
                if (job->layer != layer || job->checked) { continue; }
                job->checked = true;
                if (job->failed || !selected(job->expert)) { continue; }
                if (tier_holds(job->expert)) { ++prefetch.vram_late; continue; }
                if (prefetch.selected_n < 64) { prefetch.selected_reads[prefetch.selected_n++] = { job->rank, job->expert }; }
            }
        }
        prefetch.last_layer = -1;
        prefetch.window_open = false;
    }
    // For the run's scheduler, under the I/O mutex: whether a cache hit still waits for its prefetch reads.
    // For the run's scheduler, under the I/O mutex: the state of the prefetch behind a hit (1 for a hit that is
    // not a prefetch of this run), and whether any of them is still reading or failed.
    int prefetch_hit_state(int entry) const {
        for (const auto & p : prefetch.pending) { if (p.first == entry) { return p.second->state.load(); } }
        return 1;
    }
    bool prefetch_hits_pending() const {
        for (const auto & p : prefetch.pending) { if (p.second->state.load() == 0) { return true; } }
        return false;
    }
    bool prefetch_hit_failed() const {
        for (const auto & p : prefetch.pending) { if (p.second->state.load() == 2) { return true; } }
        return false;
    }
    // After the run's prepare: keep only the prefetched slots that are hits of its CPU experts.
    void prefetch_keep_used_hits() {
        if (prefetch.pending.empty()) { return; }
        std::vector<std::pair<int, prefetch_job *>> used;
        for (const auto & p : prefetch.pending) {
            for (const auto & r : requests) {
                if (r.cache_hit == p.first && r.vram_slot < 0) { used.push_back(p); break; }
            }
        }
        prefetch.pending.swap(used);
    }
    // After the run's prepare: a read it selected is used when the run takes it from the RAM cache.
    void prefetch_count_used() {
        for (int i = 0; i < prefetch.selected_n; ++i) {
            for (const auto & r : requests) {
                if (r.id != prefetch.selected_reads[i].second) { continue; }
                prefetch.used_r[prefetch.selected_reads[i].first] += r.cache_hit >= 0 && r.vram_slot < 0;
                break;
            }
        }
        prefetch.selected_n = 0;
    }
    // Called by the cache before it unmaps its arena, possibly in the middle of a run.
    void prefetch_wait_reads() {
        std::unique_lock<std::mutex> lock(prefetch.mutex);
        for (auto * q : { &prefetch.urgent, &prefetch.queue }) {
            for (const auto & item : *q) { --item.first->queued; prefetch_settle(*item.first); }
            q->clear();
        }
        prefetch.cv.wait(lock, [&] { return prefetch.running == 0; });
    }
    void prefetch_report(const char * reason) {
        if (!prefetch.topk && !prefetch.layers) { return; }
        sink_record out;
        auto ranks = [&](const char * name, const uint64_t * counts) {
            out.append(" %s=", name);
            for (int i = 0; i < prefetch.width; ++i) { out.append("%s%llu", i ? "," : "", (unsigned long long) counts[i]); }
        };
        uint64_t reader_bytes = 0, spills = 0, started_r[64], spill_us_r[64];
        double reader_busy_us = 0, spill_us = 0;
        {
            std::lock_guard<std::mutex> lock(prefetch.mutex);
            reader_bytes = prefetch.reader_bytes;
            reader_busy_us = prefetch.reader_busy_us;
            spill_us = prefetch.spill_us; spills = prefetch.spills;
            std::copy(prefetch.started_r, prefetch.started_r + 64, started_r);
            std::copy(prefetch.spill_us_r, prefetch.spill_us_r + 64, spill_us_r);
            // a periodic report while reads spill: their time past the close so far
            const auto now = sink_clock::now();
            if (prefetch.spilling) { spill_us += std::chrono::duration<double, std::micro>(now - prefetch.spill_since).count(); }
            for (const auto & r : prefetch.reading) {
                if (r.spill) { spill_us_r[r.job->rank] += uint64_t(std::llround(std::chrono::duration<double, std::micro>(now - r.since).count())); }
            }
        }
        out.append("[kimi_cpu_sink_prefetch] reason=%s topk=%d layers=%llu predicted=%llu skip_vram=%llu skip_cached=%llu"
                " vram_late=%llu skip_vram_lost=%llu unresolved=%llu no_slot=%llu issued=%llu ready=%llu cancelled=%llu failed=%llu carried=%llu"
                " activated=%llu hits=%llu unused=%llu windows=%llu window_us_p50=%.0f window_us_mean=%.0f"
                " reader_bytes=%llu reader_busy_us=%.0f spill_us=%.0f spills=%llu width=%d",
                reason, prefetch.topk, (unsigned long long) prefetch.layers, (unsigned long long) prefetch.predicted,
                (unsigned long long) prefetch.skip_vram, (unsigned long long) prefetch.skip_cached,
                (unsigned long long) prefetch.vram_late, (unsigned long long) prefetch.skip_vram_lost,
                (unsigned long long) prefetch.unresolved, (unsigned long long) prefetch.no_slot,
                (unsigned long long) prefetch.issued,
                (unsigned long long) prefetch.ready, (unsigned long long) prefetch.cancelled,
                (unsigned long long) prefetch.failed, (unsigned long long) prefetch.carried,
                (unsigned long long) prefetch.activated,
                (unsigned long long) cache.prefetch_hits,
                (unsigned long long) cache.prefetch_unused, (unsigned long long) prefetch.window.count,
                prefetch.window.median_us(), prefetch.window.mean_us(), (unsigned long long) reader_bytes,
                reader_busy_us, spill_us, (unsigned long long) spills, prefetch.width);
        ranks("predicted_r", prefetch.predicted_r); ranks("novel_r", prefetch.novel_r);
        ranks("novel_routed_r", prefetch.novel_routed_r); ranks("issued_r", prefetch.issued_r); ranks("used_r", prefetch.used_r);
        ranks("started_r", started_r); ranks("spill_us_r", spill_us_r);
        out.append("\n");
        out.flush();
    }
    // Called by a decode run after its own reads were published: the statistics of the hint it got for the next
    // layer, then the reads of its first k ranks. A run without reads of its own opens the drive's idle window here,
    // with a hint or without one.
    void prefetch_schedule(const ggml_tensor * up, bool demand_reads) {
        const int layer = ssd::host::routed_layer(up);
        const int nhint = prefetch.hint_n;
        prefetch.hint_n = 0;
        if (layer < 0) { return; }
        const auto policy = ssd::place::plan_prefetch(layer, prefetch.hint_layer, nhint, prefetch.topk, demand_reads);
        if (!policy.hinted) {
            // no hint for the next layer (the last MoE layer's run): nothing to read, but a run without reads of its
            // own still leaves the drive idle now, so that the reads still running from earlier stop spilling here
            if (policy.open_gate) { prefetch_open_gate(); }
            return;
        }
        ++prefetch.layers;
        if (prefetch.layers % 8192 == 0) { prefetch_report("periodic"); }
        // what the GPU tier holds of the next layer now: its prepare will pin those as hits
        uint8_t held[64] = {};
        vram.held(layer + 1, prefetch.hint, nhint, held);
        std::array<ssd::place::prefetch_rank, 64> ranks;
        for (int i = 0; i < nhint; ++i) {
            const int32_t expert = prefetch.hint[i];
            ranks[i] = ssd::place::plan_prefetch_rank(expert, held[i],
                expert >= 0 && !held[i] && cache.holds(policy.layer, expert), i, prefetch.topk);
            const bool novel = ranks[i].novel;
            ++prefetch.predicted_r[i];
            prefetch.novel_r[i] += novel;
            prefetch.last_novel[i] = novel;
            prefetch.last_skipped[i] = ranks[i].skipped;
            prefetch.last_hint[i] = expert;
        }
        prefetch.width = std::max(prefetch.width, nhint);
        prefetch.last_layer = layer + 1;
        prefetch.last_n = nhint;
        // without reads of its own, this run leaves the drive idle now
        if (policy.open_gate) { prefetch_open_gate(); }
        const int k = policy.ranks;
        if (!k) { return; }
        char names[3][GGML_MAX_NAME];
        std::snprintf(names[0], sizeof(names[0]), "blk.%d.ffn_up_exps.weight", layer + 1);
        std::snprintf(names[1], sizeof(names[1]), "blk.%d.ffn_gate_exps.weight", layer + 1);
        std::snprintf(names[2], sizeof(names[2]), "blk.%d.ffn_down_exps.weight", layer + 1);
        std::vector<std::unique_ptr<prefetch_job>> jobs;
        jobs.reserve(size_t(k));
        for (int i = 0; i < k; ++i) {
            const int32_t expert = prefetch.hint[i];
            ++prefetch.predicted;
            if (!ranks[i].request) {
                if (ranks[i].skipped) { ++prefetch.skip_vram; }
                continue;
            }
            auto job = std::make_unique<prefetch_job>(); // before the slot is reserved
            job->layer = layer + 1; job->expert = expert; job->rank = i;
            bool resolved = true;
            for (int role = 0; role < 3 && resolved; ++role) {
                resolved = resolve(names[role], expert, &job->role[role]) &&
                    job->role[role].offset % 4096 == 0 && job->role[role].nbytes % 4096 == 0;
            }
            if (!resolved) { ++prefetch.unresolved; continue; }
            job->entry = cache.prefetch_reserve(layer + 1, expert, job->role);
            if (job->entry == -2) { ++prefetch.skip_cached; continue; }
            if (job->entry < 0) { ++prefetch.no_slot; continue; }
            job->generation = cache.generation;
            for (int role = 0; role < 3; ++role) { job->target[role] = cache.data(job->entry, role); }
            jobs.push_back(std::move(job));
        }
        if (jobs.empty()) { return; }
        {
            std::lock_guard<std::mutex> lock(prefetch.mutex);
            prefetch.jobs.reserve(prefetch.jobs.size() + jobs.size());
            for (auto & job : jobs) {
                prefetch_job * queued = job.get();
                prefetch.jobs.push_back(std::move(job));
                ++prefetch.issued;
                ++prefetch.issued_r[queued->rank];
                // counted as queued, so after a failed allocation the drain settles the job as dropped
                for (int role = 0; role < 3; ++role) {
                    prefetch_alloc(prefetch.throw_schedule, prefetch.allocs_schedule);
                    prefetch.queue.emplace_back(queued, role);
                    ++queued->queued;
                }
            }
        }
        prefetch.cv.notify_all();
    }

    ~kimi_cpu_sink() {
        if (call_open) {
            std::fprintf(stderr, "[kimi_cpu_sink] destroyed inside an open call; aborting\n");
            std::abort();
        }
        cache.ledger.client->clear_reclaimer(); // before anything a reclaim on another thread uses
        end_decode_run();
        prefetch_stop();
        host_lock_stop(); // the tier and the arena are both still here
        place_report("destroy");
        prefetch_report("destroy");
        {
            const auto l = cache.ledger.state();
            std::fprintf(stderr, "[kimi_host_ledger] reason=destroy unified=%d reserve=%llu margin=%llu floor=%llu file_reserve=%llu explicit_reserve=%llu granted=%llu refused=%llu cache_cgroup_rejections=%llu cache_system_rejections=%llu margin_updates=%llu\n",
                    int(l.unified), (unsigned long long) l.reserve, (unsigned long long) l.margin, (unsigned long long) l.floor,
                    (unsigned long long) l.file_reserve, (unsigned long long) l.explicit_reserve, (unsigned long long) l.granted,
                    (unsigned long long) ledger_refusals, (unsigned long long) cache.cgroup_rejections,
                    (unsigned long long) cache.system_rejections, (unsigned long long) l.margin_updates);
        }
        cache.before_unmap = nullptr;
        cache.before_release = nullptr; // vram is destroyed before cache
        stop_threads(); // the I/O thread stops a read-ahead and waits for its reads before it returns
        ring_unlock(); // the tier is still here, and the ring's buffers too
        lookahead_summary();
        // Explicit unregister is synchronous. Do this before member destruction
        // frees slots; closing only in sink_ring's destructor would be too late.
        ring.close_ring();
    }

    // On unified memory the floor moves between token boundaries: a prefill refaults file pages (the ledger adds
    // them to its file reserve), and allocations outside the ledger take what the tier left. Decode restores it at
    // every token boundary, but a prefill passes no token boundary for minutes. Ten times a second the
    // ledger measures again and, below the floor, the RAM cache's demand and 64 MiB, the tier gives back blocks until
    // MemAvailable keeps the floor, the demand and the tier's low watermark. A sink run in progress keeps run_mutex
    // and is not waited for: the watcher acts between a decode's layers and outside the sink's runs. MemAvailable is
    // read at the full watermark boost, as the tier reads its free memory (kimi-memory-boost.h).
    void floor_watch_loop() {
        std::unique_lock<std::mutex> lock(watch_mutex);
        while (!watch_cv.wait_for(lock, std::chrono::milliseconds(100), [&] { return watch_stop; })) {
            lock.unlock();
            {
                std::unique_lock<sink_run_mutex> run(run_mutex, std::try_to_lock);
                if (run.owns_lock()) {
                    try {
                        refresh_floor("watch");
                        uint64_t available = 0, headroom = 0;
                        if (cache.ledger.memory_available_at_full_boost(available, headroom) && available < sink_add(cache.ledger.floor(),
                                cache.pending_bytes(cache.stride)) + (uint64_t(64) << 20)) {
                            vram.reclaim_host(0);
                        }
                    } catch (const std::exception & e) {
                        std::fprintf(stderr, "[kimi_host_ledger] event=watch status=fail error=%s\n", e.what());
                    }
                }
            }
            lock.lock();
        }
    }

    void stop_threads() {
        {
            std::lock_guard<std::mutex> lock(watch_mutex);
            watch_stop = true;
        }
        watch_cv.notify_all();
        if (watch_thread.joinable()) { watch_thread.join(); }
        {
            std::lock_guard<std::mutex> lock(io_mutex);
            io_stop = true;
        }
        io_cv.notify_all();
        if (io_thread.joinable()) { io_thread.join(); }
    }

    // Request i's item of a phase, the k-th of its batch: hidden scratch row k.
    sink_work_item work_item(size_t i, const void * const data[3], float weight, size_t k, float * output_row,
            uint8_t * down_quant) {
        sink_work_item w;
        w.expert = i;
        w.up = static_cast<const char *>(data[0]);
        w.gate = static_cast<const char *>(data[1]);
        w.down = static_cast<const char *>(data[2]);
        w.weight = weight;
        w.hidden = activation.data() + k * size_t(hidden);
        w.hidden_up = activation_up.empty() ? nullptr : activation_up.data() + k * size_t(hidden);
        w.output = output_row;
        w.down_quant = down_quant;
        return w;
    }

    // The rows before an item's first output cache line, which the coordinator computes.
    int dynamic_first_row(const sink_work_item & w, bool is_down) const {
        const int rows = is_down ? latent : hidden;
        const float * target = is_down ? w.output : w.hidden;
        const uintptr_t misalignment = reinterpret_cast<uintptr_t>(target) % 64;
        const int prefix = int((64 - misalignment) % 64) / int(sizeof(float));
        return std::min(rows, prefix);
    }

    void begin_dynamic_phase(bool is_down) {
        if (!dynamic_rows || parallel_fn) { return; }
        const int rows = is_down ? latent : hidden;
        work_chunks.resize(phase_items.size() + 1);
        work_chunks[0] = 0;
        for (size_t k = 0; k < phase_items.size(); ++k) {
            const int first = dynamic_first_row(phase_items[k], is_down);
            work_chunks[k + 1] = work_chunks[k] + (rows - first + dynamic_chunk_rows - 1) / dynamic_chunk_rows;
        }
        dynamic_row_counter.next.store(0, std::memory_order_relaxed);
    }

    void compute_rows(int ith, bool is_down) {
        // a phase of several experts times its workers against its first
        sink_timeline_worker * timing = timeline_active && !parallel_fn && !phase_items.empty() ?
            &timeline[phase_items.front().expert].workers[(is_down ? threads : 0) + ith] : nullptr;
        if (timing) { timing->begin = ggml_time_us(); timing->core = ssd::platform::current_cpu(); }
        const auto profile_start = profile_enabled ? sink_clock::now() : sink_clock::time_point();
        const double cpu_start = profile_enabled ? sink_thread_cpu_us() : 0.0;
        const int rows = is_down ? latent : hidden;
        const auto & bounds = is_down ? down_bounds : up_bounds;
        if (parallel_fn) {
            parallel_fn(parallel_userdata, ith, active_threads);
        } else {
            auto compute_range = [&](const sink_work_item & w, int begin, int end) {
                if (!is_down && activation_kind == KIMI_CPU_SINK_ACTIVATION_SWIGLU_CLAMP) {
                    // The scalar expression of ggml_compute_forward_swiglu_clamp_f32
                    // (gate=src0, up=src1): each row is bit-identical to the graph op.
                    const float limit = swiglu_limit;
                    for (int row = begin; row < end; ++row) {
                        float u, g;
                        dot_up(latent, &u, 0, w.up + row * row_bytes_up, 0, quant_input.data(), 0, 1);
                        dot_up(latent, &g, 0, w.gate + row * row_bytes_up, 0, quant_input.data(), 0, 1);
                        const float gate_value = std::min(g, limit);
                        const float up_value = std::clamp(u, -limit, limit);
                        w.hidden[row] = gate_value / (1.f + expf(-gate_value)) * up_value;
                    }
                } else if (!is_down && activation_kind == KIMI_CPU_SINK_ACTIVATION_SWIGLU) {
                    // g and u as they are: the coordinator gates the whole row (quantize_activation)
                    for (int row = begin; row < end; ++row) {
                        dot_up(latent, &w.hidden_up[row], 0, w.up + row * row_bytes_up, 0, quant_input.data(), 0, 1);
                        dot_up(latent, &w.hidden[row], 0, w.gate + row * row_bytes_up, 0, quant_input.data(), 0, 1);
                    }
                } else if (!is_down) {
                    for (int row = begin; row < end; ++row) {
                        float u, g;
                        dot_up(latent, &u, 0, w.up + row * row_bytes_up, 0, quant_input.data(), 0, 1);
                        dot_up(latent, &g, 0, w.gate + row * row_bytes_up, 0, quant_input.data(), 0, 1);
                        const float a = (beta * std::tanh(g * (1.0f / beta))) * (1.0f / (1.0f + std::exp(-g)));
                        if (linear_beta > 0.0f) { u = linear_beta * std::tanh(u * (1.0f / linear_beta)); }
                        w.hidden[row] = a * u;
                    }
                } else {
                    for (int row = begin; row < end; ++row) {
                        float value;
                        dot_down(hidden, &value, 0, w.down + row * row_bytes_down, 0, w.down_quant, 0, 1);
                        // Preserve the separate multiply followed by the graph's slot-order
                        // addition even when the compiler enables floating point contraction.
                        volatile float scaled = post_exp_norm ? value : value * w.weight;
                        if (ready_first || first_down) { w.output[row] = scaled; }
                        else { w.output[row] += scaled; }
                    }
                }
            };
            if (dynamic_rows) {
                // The caller owns each item's short unaligned prefix. Every claimed
                // chunk thereafter begins and ends on an output cache line.
                if (ith == 0) {
                    for (const auto & w : phase_items) {
                        const int first = dynamic_first_row(w, is_down);
                        if (first) { compute_range(w, 0, first); }
                    }
                }
                size_t k = 0;
                for (;;) {
                    const int chunk = dynamic_row_counter.next.fetch_add(1, std::memory_order_relaxed);
                    if (chunk >= work_chunks.back()) { break; }
                    while (chunk >= work_chunks[k + 1]) { ++k; } // a participant's claims only grow
                    const int begin = dynamic_first_row(phase_items[k], is_down) + (chunk - work_chunks[k]) * dynamic_chunk_rows;
                    compute_range(phase_items[k], begin, std::min(rows, begin + dynamic_chunk_rows));
                }
            } else {
                const int begin = row_weights.empty() ? rows * ith / active_threads : bounds[ith];
                const int end = row_weights.empty() ? rows * (ith + 1) / active_threads : bounds[ith + 1];
                for (const auto & w : phase_items) { compute_range(w, begin, end); }
            }
        }
        if (timing) { timing->end = ggml_time_us(); }
        if (profile_enabled) {
            auto & p = worker_profiles[ith];
            const double cpu_us = sink_thread_cpu_us() - cpu_start;
            if (timing) { timing->cpu_us = cpu_us; }
            const double wall_us = elapsed_us(profile_start);
            const double dispatch_us = std::chrono::duration<double, std::micro>(profile_start - phase_dispatch).count();
            p.dispatch_us += dispatch_us;
            p.max_dispatch_us = std::max(p.max_dispatch_us, dispatch_us);
            ++p.phases;
            if (parallel_fn) { p.parallel_wall_us += wall_us; p.parallel_cpu_us += cpu_us; }
            else if (is_down) { p.down_wall_us += wall_us; p.down_cpu_us += cpu_us; }
            else { p.up_gate_wall_us += wall_us; p.up_gate_cpu_us += cpu_us; }
        }
    }

    // The coordinator's state as the phase word; it wakes parked joiners.
    void publish_phase(int active, uint64_t kind) {
        phase.store(sink_phase::make(generation, active, kind | (call_open ? sink_phase::open : 0), call_id));
        participants.notify();
    }

    bool open_call(int nth, std::string & why) {
        if (nth < 1 || nth > sink_max_participants) { why = "CPU sink call needs 1.." + std::to_string(sink_max_participants) +
            " participants"; return false; }
        if (call_open) { why = "CPU sink call already open"; return false; }
        // Sized before any participant indexes it; the open call keeps the size.
        worker_profiles.assign(size_t(nth), sink_worker_profile());
        call_open = true;
        call_participants = nth;
        call_generation = generation;
        ++call_id;
        joined.store(0);
        publish_phase(0, 0);
        return true;
    }

    // After every joiner arrived, so none can still be on its way into this call; one that has not yet seen the
    // close tells it, or a later call, by the call's id.
    void close_call() {
        if (!call_open) { return; }
        const auto arrived = [&] { return joined.load() == uint32_t(call_participants - 1); };
        while (!arrived()) { coordinator.park(arrived); }
        call_open = false;
        publish_phase(0, 0);
    }

    // A participant of the open call (ith 1..nth-1) computes its share of every phase the run's thread publishes,
    // until the call closes. One that joins late takes up the phase in progress: no phase it is active in can finish
    // without it. Once joined, and after a phase, it polls for the next for poll_us, then parks: one that parked at
    // once would hold the call's first phase, or the close of a call that runs none, for its wakeup. It waits for the
    // open, and for the close after the run's last phase, as at a node boundary (sink_node_spin_us). A wakeup without
    // work counts into the participant's next phase profile.
    void join_call(int ith, int nth) {
        // A ggml node starts on every thread at once and its coordinator opens the call at entry. The previous call
        // closed only after this thread joined it, and ggml's barrier after that node released no thread before this
        // one left it, so the next call to open is this node's.
        sink_phase now;
        const auto opened = [&] { now.word = phase.load(); return now.is(sink_phase::open); };
        sink_poll(std::max(poll_us, sink_node_spin_us), opened);
        while (!opened()) { participants.park(opened); }
        if (nth != call_participants || ith < 1 || ith >= nth) {
            std::fprintf(stderr, "[kimi_cpu_sink] participant %d of %d joined a call of %d; aborting\n", ith, nth,
                call_participants);
            std::abort();
        }
        mark_thread_cpu(ith);
        const uint32_t call = now.call();
        uint32_t seen = call_generation;
        if (joined.fetch_add(1) + 1 == uint32_t(nth - 1)) { coordinator.notify(); }
        int poll_next = poll_us;
        uint64_t idle_wakeups = 0;
        for (;;) {
            now.word = phase.load();
            if (!now.is(sink_phase::open) || now.call() != call) { return; }
            if (now.generation() == seen) {
                const uint64_t waited = now.word;
                const auto changed = [&] { return phase.load() != waited; };
                if (poll_next) {
                    sink_poll(poll_next, changed);
                    poll_next = 0;
                } else if (participants.park(changed) && !changed()) {
                    ++idle_wakeups;
                }
                continue;
            }
            seen = now.generation();
            poll_next = now.is(sink_phase::last) ? std::max(poll_us, sink_node_spin_us) :
                now.is(sink_phase::parallel) ? 0 : poll_us;
            if (ith >= now.active()) { continue; }
            if (profile_enabled) { worker_profiles[ith].no_work_wakeups += idle_wakeups; }
            idle_wakeups = 0;
            compute_rows(ith, now.is(sink_phase::down));
            if (pending.fetch_sub(1) == 1) { coordinator.notify(); }
        }
    }

    // A run starts with the participants of the open call, or alone; per-participant state is sized for them. A
    // decode run also sets up its thread choice (decode_threads_setup).
    void begin_participants(bool decode) {
        threads = call_open ? call_participants : 1;
        if (!row_weights.empty() && row_weights.size() != size_t(threads)) {
            throw std::runtime_error("explicit row weights need one entry per participant: " +
                std::to_string(row_weights.size()) + " for " + std::to_string(threads));
        }
        run_participants = threads;
        mark_thread_cpu(0);
        if (decode) {
            const uint64_t now = uint64_t(ggml_time_us()) * 1000;
            if (share.last_call) { share.gap = std::max(share.gap, now - share.last_call); }
            share.last_call = now;
            decode_threads_setup();
        }
    }

    // This thread's CPU time since its previous call of the generation, added to its index (decode_share).
    void mark_thread_cpu(int ith) {
        thread_local const kimi_cpu_sink * owner = nullptr;
        thread_local uint64_t generation = 0, last = 0;
        const uint64_t now = ssd::platform::thread_cpu_time_ns(), current = cpu_generation.load(std::memory_order_relaxed);
        if (owner == this && generation == current && now >= last) {
            thread_cpu[size_t(ith)].fetch_add(now - last, std::memory_order_relaxed);
        }
        owner = this; generation = current; last = now;
    }

    // At a token boundary (cache_begin). Between two decode tokens in a row the CPUs' busy time less the threads' CPU
    // is the other work; over a run of such tokens the sums carry each CPU's counter up to one tick behind and each
    // thread's CPU between a boundary and its next call (at most the longest gap between calls) at both ends, so each
    // run adds that much error. The reserve follows (ssd::place::decode_reserve).
    void measure_decode_share(bool decode, uint64_t epoch, uint64_t token) {
        const bool consecutive = decode && share.run > 0 && epoch == share.epoch && token == share.token + 1;
        share.epoch = epoch; share.token = token;
        if (!consecutive) { end_decode_run(); }
        if (!decode) { cpu_generation.fetch_add(1, std::memory_order_relaxed); return; }
        uint64_t busy = 0, tick = 0;
        std::string why;
        if (!share.readable || share.cpus.empty()) { return; }
        if (!ssd::platform::cpus_busy_ns(share.cpus, busy, tick, &why)) {
            share.readable = false;
            std::fprintf(stderr, "[kimi_cpu_sink_threads] other work unmeasured: %s\n", why.c_str());
            return;
        }
        uint64_t pool = 0;
        for (int i = 0; i < sink_max_participants; ++i) { pool += thread_cpu[size_t(i)].load(std::memory_order_relaxed); }
        const uint64_t wall = uint64_t(ggml_time_us()) * 1000;
        if (consecutive) {
            share.other_ns += double(busy) - double(share.busy) - (double(pool) - double(share.pool));
            share.wall_ns += double(wall - share.wall);
        }
        share.run += 1; share.busy = busy; share.pool = pool; share.wall = wall; share.tick = tick;
        const double open = share.run > 1 ?
            double(share.cpus.size()) * double(tick) + 2.0 * double(threads) * double(share.gap) : 0.0;
        share.reserve = ssd::place::decode_reserve(share.reserve, share.other_ns, share.wall_ns, share.error_ns + open);
    }

    // A run of decode tokens ends (a non-decode or non-consecutive boundary, or the sink's end): its error joins the
    // sums', and the sums are reported.
    void end_decode_run() {
        if (share.run > 1) {
            share.error_ns += double(share.cpus.size()) * double(share.tick) + 2.0 * double(threads) * double(share.gap);
            std::fprintf(stderr, "[kimi_cpu_sink_threads] decode_share tokens=%d other_cpus=%.2f error_cpus=%.2f"
                " over_s=%.1f reserve=%d cpus=%zu\n", share.run - 1, share.other_ns / share.wall_ns,
                share.error_ns / share.wall_ns, share.wall_ns / 1e9, share.reserve, share.cpus.size());
        }
        share.run = 0; share.gap = 0; share.last_call = 0;
    }

    // The threads of a decode pool of `pool` a decode graph runs on (kimi_cpu_sink_decode_threads).
    int decode_threads(int pool) {
        if (configuration.policy.decode_participants || share.cpus.empty()) { return pool; }
        const int n = ssd::place::decode_participants(pool, int(share.cpus.size()), share.reserve);
        if (n != share.reported) {
            share.reported = n;
            std::fprintf(stderr, "[kimi_cpu_sink_threads] decode_threads=%d pool=%d cpus=%zu reserve=%d other_cpus=%.2f"
                " over_s=%.1f\n", n, pool, share.cpus.size(), share.reserve,
                share.wall_ns > 0 ? share.other_ns / share.wall_ns : 0.0, share.wall_ns / 1e9);
        }
        return n;
    }

    // A decode run's phases use every thread of its call, or the first cpu.decode_participants of them. The strategy
    // sizes the decode pool to the threads that raise the rate decode streams expert weights at, so none of them
    // only waits: each one is waited for at every call's close and at ggml's barrier after its node. With strict
    // placement ggml thread i runs on the i-th CPU of its mask.
    void decode_threads_setup() {
        const int fixed = configuration.policy.decode_participants;
        if (fixed > threads) {
            throw std::runtime_error("cpu.decode_participants=" + std::to_string(fixed) + " exceeds the call's " +
                std::to_string(threads) + " threads");
        }
        run_participants = fixed ? fixed : threads;
        if (decode_threads_for == threads) { return; }
        decode_threads_for = threads;
        std::fprintf(stderr, "[kimi_cpu_sink_threads] participants=%d decode_participants=%d\n", threads,
                run_participants);
    }

    // At load, storage before any request, so that no request waits for its page population. On unified memory the
    // GPU tier's (sink_vram::reserve), with the floor and the RAM cache's demand published first, as the first decode
    // layer publishes them before the tier sizes itself there; the RAM cache there only lands the prefetch and follows
    // at the first decode token, from the demand the tier left. Elsewhere, unified memory without a GPU tier included,
    // the RAM cache's (reserve_cache), when the cache can give room back (partial reclaim or elastic). experts: up, gate
    // and down of n routed layers; the largest expert gives both their geometry. With a GPU tier each layer's types
    // are kept first (record_expert_types).
    void reserve_tier(const ggml_tensor * const * experts, int n, int experts_used) {
        if (vram.context) { record_expert_types(experts, n); }
        const auto bytes = [&](int i) { return experts[3 * i]->nb[2] + experts[3 * i + 1]->nb[2] + experts[3 * i + 2]->nb[2]; };
        int largest = 0;
        for (int i = 1; i < n; ++i) { if (bytes(i) > bytes(largest)) { largest = i; } }
        const ggml_tensor * up = experts[3 * largest], * gate = experts[3 * largest + 1], * down = experts[3 * largest + 2];
        if (!cache.ledger.state().unified || !vram.context) { reserve_cache(up, gate, down, experts_used); }
        if (!vram.context || cache.layers.empty()) { return; }
        refresh_floor("reserve");
        if (!cache.arena) { vram.set_host_demand(cache.pending_bytes(size_t(up->nb[2] + gate->nb[2] + down->nb[2]))); }
        vram.reserve(up, gate, down, sink_vram::layer_list(cache));
    }
    // Every routed layer's expert types, kept with the cache's layers: the GPU tier is declared them with each layer it
    // is given (sink_vram::layer_list) once every configured layer has its. The tier keeps all of them in slots of one
    // geometry, so the layers must share latent, hidden and the expert count. A refusal keeps nothing.
    void record_expert_types(const ggml_tensor * const * experts, int n) {
        struct typed { size_t index; int32_t up, down; };
        std::vector<typed> found;
        const ggml_tensor * first = experts[0];
        for (int i = 0; i < n; ++i) {
            const ggml_tensor * up = experts[3 * i], * gate = experts[3 * i + 1], * down = experts[3 * i + 2];
            const int layer = ssd::host::routed_layer(up);
            if (layer < 0) {
                throw std::runtime_error(std::string("expert reserve requires canonical routed layer tensors, not ") + up->name);
            }
            if (up->type != gate->type) { throw std::runtime_error("CPU sink requires up and gate experts of one type"); }
            if (up->ne[0] != first->ne[0] || up->ne[1] != first->ne[1] || up->ne[2] != first->ne[2] ||
                    gate->ne[0] != up->ne[0] || gate->ne[1] != up->ne[1] || gate->ne[2] != up->ne[2] ||
                    down->ne[0] != up->ne[1] || down->ne[1] != up->ne[0] || down->ne[2] != up->ne[2]) {
                throw std::runtime_error("routed layer " + std::to_string(layer) + " has another expert geometry than " +
                    first->name + ": the GPU tier needs the same latent, hidden and expert count in every routed layer");
            }
            const auto at = std::find_if(cache.layers.begin(), cache.layers.end(), [&](const sink_cache::layer & l) { return l.id == layer; });
            if (at == cache.layers.end()) {
                throw std::runtime_error("expert reserve: layer " + std::to_string(layer) + " is not a configured routed layer");
            }
            found.push_back({ size_t(at - cache.layers.begin()), int32_t(up->type), int32_t(down->type) });
        }
        for (const auto & f : found) { cache.layers[f.index].up_type = f.up; cache.layers[f.index].down_type = f.down; }
    }
    // The RAM cache's arena and pages at load (sink_cache::reserve_at_load), each role as a decode request reserves
    // it (its bytes and a page prefix, in whole pages), after the read ring the first decode charges before the cache
    // sizes itself from the room left; experts_used prices the frequency nodes as the first decode layer does. A cache
    // that cannot give room back takes neither here: the ring and the arena both wait for the first decode token.
    void reserve_cache(const ggml_tensor * up, const ggml_tensor * gate, const ggml_tensor * down, int experts_used) {
        if (cache.layers.empty() || !cache.keep_across_requests || !cache.gives_room_back() || cache.arena) { return; }
        const ggml_tensor * tensors[] = { up, gate, down };
        size_t bytes[3];
        for (int role = 0; role < 3; ++role) {
            bytes[role] = (ggml_row_size(tensors[role]->type, tensors[role]->ne[0]) * tensors[role]->ne[1] + 8190) / 4096 * 4096;
        }
        if (cache.stats.budget_bytes == sink_cache::auto_budget) { reserve_ring(bytes, true); }
        double us = 0;
        // load time: nothing else runs, so every CPU pages the arena in
        cache.reserve_at_load(bytes, up, size_t(std::max(1, experts_used)), sink_process_cpus(), us);
    }
    // The ledger's floor moves with the file reserve on unified memory; the GPU tier keeps it free.
    void refresh_floor(const char * reason) {
        auto & ledger = cache.ledger;
        const uint64_t before = ledger.floor();
        if (ledger.update_file_reserve()) {
            // A higher floor is published once the tier has given back what it adds, as far as the tier can now (the
            // watcher repeats it): the file pages that came back keep coming back while the tier holds their memory.
            if (ledger.floor() > before) { vram.reclaim_host(ledger.floor() - before); }
            std::fprintf(stderr, "[kimi_host_ledger] reason=%s floor=%llu file_reserve=%llu margin=%llu explicit_reserve=%llu refaults=%llu updates=%llu\n",
                    reason, (unsigned long long) ledger.floor(), (unsigned long long) ledger.state().file_reserve,
                    (unsigned long long) ledger.state().margin, (unsigned long long) ledger.state().explicit_reserve,
                    (unsigned long long) ledger.state().refaults, (unsigned long long) ledger.state().file_updates);
        }
        vram.set_host_floor(ledger.floor());
    }
    // DRAM the engine knows it holds: GPU tier blocks on unified memory, committed RAM cache slots.
    uint64_t held_bytes() {
        uint64_t ring_bytes = 0;
        for (const auto & slot : slots) { for (const auto & role : slot.role) { ring_bytes = sink_add(ring_bytes, role.size); } }
        uint64_t scratch = sink_cache::multiply(expert_output.capacity() + activation.capacity() +
            activation_up.capacity(), sizeof(float));
        scratch = sink_cache::add(scratch, expert_computed.capacity());
        scratch = sink_cache::add(scratch, quant_input.capacity());
        scratch = sink_cache::add(scratch, quant_hidden.capacity());
        scratch = sink_cache::add(scratch, staged_quant_hidden.capacity());
        scratch = sink_cache::add(scratch, sink_cache::multiply(vram.prefill_policy_experts.capacity(),
            sizeof(ssd::place::prefill_expert)));
        return sink_add(sink_add(vram.host_bytes(), cache.stats.committed_bytes), sink_add(ring_bytes, scratch));
    }
    // At a token boundary, before anything resizes: the margin learns what the last decode token took that nobody
    // accounts (unified memory). A boundary that does not follow the previous decode token ends nothing.
    void end_margin_interval(bool decode, uint64_t epoch, uint64_t token) {
        const bool consecutive = decode && margin_boundary && epoch == margin_epoch && token == margin_token + 1;
        margin_boundary = decode; margin_epoch = epoch; margin_token = token;
        auto & ledger = cache.ledger;
        if (!ledger.state().unified) { return; }
        const uint64_t before = ledger.state().margin, held = held_bytes();
        uint64_t untracked = 0;
        if (ledger.end_interval(consecutive, held, untracked)) {
            std::fprintf(stderr, "[kimi_host_ledger] event=margin_grow token=%llu untracked=%llu margin_before=%llu margin=%llu held=%llu granted=%llu updates=%llu\n",
                    (unsigned long long) token, (unsigned long long) untracked, (unsigned long long) before,
                    (unsigned long long) ledger.state().margin, (unsigned long long) held, (unsigned long long) ledger.state().granted,
                    (unsigned long long) ledger.state().margin_updates);
        }
    }
    // After the boundary resized the GPU tier and the RAM cache: the next decode token's interval starts here.
    void begin_margin_interval(bool decode) {
        if (decode && cache.ledger.state().unified) { cache.ledger.begin_interval(held_bytes()); }
    }
    // Room for an allocation outside the expert caches (llama_host_reserve, compute buffers): `host` bytes of host
    // memory and `device` bytes of GPU memory, which count only where the GPU shares the DRAM, beside the pages the
    // ledger holds for an open prefill. The caches give back what both terms ask; the run needs the allocation, so
    // only a limit it would not fit refuses it. No run is in flight: the caller holds run_mutex and has drained the
    // prefetch.
    bool make_room(uint64_t host, uint64_t device) {
        const uint64_t extra = cache.ledger.dram_bytes(host, device);
        // nothing comes out of this DRAM (device memory of a discrete GPU) and no prefill holds room
        if (!extra && !cache.ledger.state().prefill_pages) { return true; }
        refresh_floor("make_room");
        const auto room = cache.ledger.make_room(extra);
        const bool fits = room.within_limits(extra);
        if (!fits) {
            ++ledger_refusals;
            std::fprintf(stderr, "[kimi_host_ledger] event=refused extra=%llu term=limits limits_shortage=%llu limits_readable=%d available=%llu floor=%llu\n",
                    (unsigned long long) extra, (unsigned long long) room.limits_shortage,
                    int(room.limits != ssd::host::room_unreadable), (unsigned long long) room.available,
                    (unsigned long long) room.floor);
        }
        return fits;
    }
    // Room for a prefill, before it allocates anything (kimi_cpu_sink_cache_prefill_room): the ledger holds `pages`
    // for the file pages of the weight rows its graph gathers until the first decode token, and makes room for its
    // compute buffers (`host`, `device`) beside them, so that the RAM cache gives back what both need at once.
    bool prefill_room(uint64_t pages, uint64_t host, uint64_t device) {
        cache.ledger.set_prefill_pages(pages);
        const size_t before = cache.enabled_slots();
        const uint64_t committed = cache.stats.committed_bytes;
        const bool fits = make_room(host, device);
        // (a refusal has its own receipt, with the short term)
        std::fprintf(stderr, "[kimi_host_ledger] event=prefill_room pages=%llu host=%llu device=%llu fits=%d slots_before=%zu slots_after=%zu released_bytes=%llu\n",
                (unsigned long long) pages, (unsigned long long) host, (unsigned long long) device, int(fits),
                before, cache.enabled_slots(), (unsigned long long) (committed - std::min(committed, cache.stats.committed_bytes)));
        return fits;
    }
    // The cost table a layer of `ntokens` tokens is placed from (kimi-prefill-cost.h, kimi_place_table), the same for a
    // decode token and a grouped-prefill pass: the CPU's time for a one-token expert (decode's experts and a one-token
    // pass's, their running mean) and the grouped path's line (the sink's prefill calibration); the tier's
    // vector-engine lines and copy rates (host API costs2: every call and copy it makes); the batched engine's lines
    // (the prefill calibration); the drive's line over decode's last 64 layers with reads and the newest grouped-prefill
    // layer with reads (ssd::place::prefill_reads).
    // route_bytes: what the pass's engine sends over the link for a GPU route (its lane: the batched engine's route
    // list, the hand engine's input row a route; a token's input row is in the batched engine's time a token). A
    // page-locked copy is priced only once the tier has measured one; a pageable one, and the rows, at the start
    // values until then. In place: wherever the GPU reads host memory so for one token (decode's rule), and for a
    // longer pass where its engine can (the batched engine), the evaluator then choosing per layer.
    kimi_place_table place_table(int ntokens, double route_bytes = 0.0) {
        kimi_vram_host_costs c = {};
        if (vram.context && vram.host_api) { vram.host_api->costs2(vram.context, &c); }
        const ssd::place::lane_measurement measured = {
            c.resident_call_us, c.resident_expert_us, c.host_call_us, c.host_expert_us,
            c.vector_tokens, c.copy_us_per_mb, c.pageable_us_per_mb, c.link_busy_us, c.in_place != 0, c.in_place_only != 0 };
        return ssd::place::make_table(vram.cost, vram.prior, learning, vram.learning, measured,
            vram.lane_geometry(), ntokens, route_bytes);
    }
    // For the decode plan: the MB still to land of each request whose RAM hit a prefetch still reads (the roles not
    // read yet, taken in their read order; a role being read counts whole), 0 for every other request.
    std::vector<double> prefetch_pending_mb() {
        std::vector<double> mb(requests.size(), 0.0);
        if (prefetch.pending.empty()) { return mb; }
        std::lock_guard<std::mutex> lock(prefetch.mutex);
        for (const auto & p : prefetch.pending) {
            const prefetch_job & job = *p.second;
            if (job.state.load() != 0) { continue; }
            uint64_t bytes = 0;
            for (int role = std::max(0, job.done); role < 3; ++role) { bytes += job.role[role].nbytes; }
            for (size_t i = 0; i < requests.size(); ++i) {
                if (requests[i].cache_hit == p.first) { mb[i] = double(bytes) / 1e6; }
            }
        }
        return mb;
    }
    // The layer's plan, once its reads are issued: the cost table's choice for a one-token layer (place_table,
    // kimi_place_layer), the experts as grouped prefill describes its own (sink_vram::place_items). Every expert the
    // GPU could take this layer is a candidate, priced with when its bytes are in host memory: a RAM hit at once, one
    // its prefetch still reads at that read's landing (its rest is read at once when the layer starts, ahead of the
    // layer's reads), a miss at its own; the executor hands a planned expert to the GPU once it has landed
    // (run_nonblocking_down_schedule). On a discrete GPU only from page-locked memory (the RAM slot, or the ring once
    // registered) at a measured copy rate, and with a device target (the expert's admission reservation, else one of
    // the tier's free scratch slots). A RAM hit the plan would have moved had its slot been page-locked goes to the
    // front of the registrar's queue.
    void plan_decode_layer() {
        plan.valid = false;
        plan.gpu.assign(requests.size(), 0);
        if (!host_gpu_enabled || !nonblocking_down_sched || !vram.host_ready() || !(learning.cpu_expert_us > 0) ||
                io_requests.size() > slots.size()) { return; }
        kimi_vram_host_costs c = {};
        vram.host_api->costs2(vram.context, &c);
        const bool in_place = c.in_place != 0;
        if (!in_place) { host_lock_decide(); }
        if (!in_place && !host_lock.enabled) { return; }
        auto reserved = [&](size_t i) { return i < 64 && ((c.reserved_rows >> i) & 1u); };
        auto items = vram.place_items(requests, [](size_t) { return 1; },
            [](size_t) { return true; }, [&](size_t i) { return !reserved(i); });
        const auto pending = prefetch_pending_mb();
        ssd::place::describe_decode(plan, items, pending);
        kimi_place_limits limits;
        limits.slots = size_t(std::max(0, c.slots));
        const kimi_place_table table = place_table(1);
        const auto placed = ssd::place::plan_decode(plan, table, slots.size(), items, limits);
        // Page-locking a slot is bought once, the move it allows pays per use: rent or buy (kimi_host_lock::missed). The
        // plan prices the moves it would make were the unlocked RAM hits' slots page-locked; each such slot accrues its
        // share of the time they would save, and goes to the registrar once that reaches one registration (the
        // registrar's measured mean, before the first one slot at 13.6 GB/s). Before a page-locked copy is measured a
        // copy is priced free, which is no price: nothing accrues, and only the first slot even a free copy would move
        // from is locked, whose registration measures the copy (the tier times one copy of the first range). On an
        // RTX 5090 a registration in the server takes 2-8 ms against a move's 0.1-0.4 ms: only slots the plan keeps
        // wanting are locked, and a short decode locks few.
        if (in_place) { return; }
        const auto wanted = ssd::place::plan_locks(table, slots.size(), items, limits, placed.time);
        if (!wanted.considered) { return; }
        std::vector<size_t> missed;
        std::vector<uint64_t> occupants;
        for (size_t i : wanted.requests) {
            const size_t entry = size_t(requests[i].cache_hit);
            missed.push_back(entry);
            const auto & e = cache.entries[entry];
            occupants.push_back(uint64_t(e.layer_index) << 32 | uint32_t(e.id));
        }
        place.wanted += host_lock.missed(missed, occupants, wanted.saved_us, double(cache.stride) / 13600.0,
            cache.entries.size(), c.copy_us_per_mb > 0);
    }
    // After the layer: its measured lanes' time (reads issued to the tier's finish) against the plan's, the drive's
    // and the layer's fixed time as samples, the receipts.
    void observe_decode_layer(double lanes_us, double total_us) {
        const double fixed = ssd::place::observe_layer(learning, current.read_bytes, current.io_us, lanes_us, total_us);
        if (!plan.valid) { return; }
        ++place.layers;
        if (vram.token != place.last_token) { ++place.tokens; place.last_token = vram.token; }
        place.moves_planned += plan.moves;
        place.candidates += plan.candidates;
        place.pending += plan.pending; place.pending_moves += plan.pending_moves;
        place.predicted_us += plan.predicted_us; place.measured_us += lanes_us; place.cpu_only_us += plan.cpu_only_us;
        if (lanes_us > 0) {
            const double error = std::fabs(plan.predicted_us - lanes_us) / lanes_us;
            ++place.error[std::min<size_t>(100, size_t(error * 100.0))];
        }
        if (profile_enabled) {
            std::fprintf(stderr, "[kimi_decode_place] token=%llu layer=%d resident=%zu ram=%zu miss=%zu locked=%zu candidates=%zu "
                "planned=%zu predicted_us=%.1f cpu_only_us=%.1f measured_us=%.1f fixed_us=%.1f total_us=%.1f pending=%zu "
                "pending_planned=%zu\n",
                (unsigned long long) vram.token, vram.requests.empty() ? -1 : vram.requests[0].layer,
                plan.residents, plan.ram, plan.misses, plan.locked, plan.candidates, plan.moves, plan.predicted_us,
                plan.cpu_only_us, lanes_us, fixed, total_us, plan.pending, plan.pending_moves);
        }
    }
    void place_report(const char * reason) {
        if (!place.layers) { return; }
        auto quantile = [&](double q) {
            uint64_t total = 0, seen = 0;
            for (uint64_t n : place.error) { total += n; }
            for (size_t b = 0; b < place.error.size(); ++b) { seen += place.error[b]; if (double(seen) >= q * double(total)) { return double(b); } }
            return 100.0;
        };
        std::fprintf(stderr, "[kimi_decode_place] reason=%s layers=%llu tokens=%llu gpu_experts=%llu gpu_per_token=%.2f planned=%llu "
            "candidates=%llu refused=%llu wanted=%llu error_p50_pct=%.0f error_p90_pct=%.0f predicted_ms=%.1f measured_ms=%.1f cpu_only_ms=%.1f "
            "cpu_expert_us=%.1f early=%.3f fixed_us=%.1f decode_read_gbs=%.2f cpu_alone_us=%.1f cpu_shared_us=%.1f cpu_moved=%.3f "
            "pending=%llu pending_planned=%llu\n",
            reason, (unsigned long long) place.layers, (unsigned long long) place.tokens, (unsigned long long) place.gpu_experts,
            place.tokens ? double(place.gpu_experts) / double(place.tokens) : 0.0, (unsigned long long) place.moves_planned,
            (unsigned long long) place.candidates, (unsigned long long) place.refused, (unsigned long long) place.wanted,
            quantile(0.5), quantile(0.9), place.predicted_us / 1e3, place.measured_us / 1e3, place.cpu_only_us / 1e3,
            learning.cpu_expert_us, learning.early, learning.fixed_us, learning.read_bytes_per_us / 1e3, learning.cpu_alone_us,
            learning.cpu_shared_us, learning.cpu_moved(), (unsigned long long) place.pending,
            (unsigned long long) place.pending_moves);
    }
    // B3: the read ring page-locked when a GPU lane copies from it (a grouped-prefill tier with transient device slots
    // on a GPU that does not read host memory in place). A streamed transient's upload is then DMA on the tier's upload
    // stream, which returns at once: the next upload overlaps the current computation, and the cost table prices the
    // copy on the link (kimi_place_layer). Each role buffer is one range, registered once per allocation of the ring, at
    // a quiescent point (a prefill layer before its reads and its launch; about 1.1-1.9 ms per 19-30 MB buffer, holding
    // the driver), and unregistered before the ring reallocates or frees it (reserve_ring, the destructor). A refused
    // registration leaves the ring pageable and is not retried before the next allocation.
    void ring_lock() {
        if (ring_pages.tried || !vram.context || !vram.prefill_enabled || !vram.host_api || !vram.prefill_api ||
                !vram.prefill_api->transient_available(vram.context) || slots.empty()) { return; }
        kimi_vram_host_costs c = {};
        vram.host_api->costs2(vram.context, &c);
        if (c.in_place && !c.in_place_only) { return; }  // the GPU reads the ring in place: nothing to page-lock
        if (!ring_pages.lock_pages) {
            ring_pages.lock_pages = [this](const void * p, size_t bytes, std::string * why) { return vram.register_host(p, bytes, why); };
            ring_pages.unlock_pages = [this](const void * p) { return vram.unregister_host(p); };
        }
        std::vector<std::pair<const void *, size_t>> buffers;
        size_t bytes = 0;
        for (const auto & slot : slots) {
            for (const auto & buffer : slot.role) { buffers.push_back({ buffer.data, buffer.size }); bytes += buffer.size; }
        }
        const auto start = sink_clock::now();
        std::string why;
        vram.ring_locked = ring_pages.lock(buffers, &why);
        std::fprintf(stderr, "[kimi_cpu_sink_io] ring_locked=%d ranges=%zu bytes=%zu us=%.0f%s%s\n", int(vram.ring_locked),
            ring_pages.ranges.size(), vram.ring_locked ? bytes : size_t(0), elapsed_us(start), why.empty() ? "" : " reason=",
            why.c_str());
    }
    void ring_unlock() {
        if (ring_pages.unlock_pages) { ring_pages.unlock(); }
        vram.ring_locked = false;
    }
    // Whether the registrar runs: a GPU tier whose host path copies (not in place), the host path on
    // supplied by the injected provider, a page-aligned slot stride and mapped arena. Decided once the arena exists.
    void host_lock_decide() {
        if (host_lock.decided || !cache.arena || !cache.stride || !vram.context || !vram.host_api) { return; }
        host_lock.decided = true;
        kimi_vram_host_costs c = {};
        vram.host_api->costs2(vram.context, &c);
        const bool enabled = host_gpu_enabled && !c.in_place && cache.stride % cache.page_size == 0 &&
            uintptr_t(cache.arena) % cache.page_size == 0;
        std::fprintf(stderr, "[kimi_host_register] reason=decided enabled=%d in_place=%d host_gpu=%d stride=%zu\n",
            int(enabled), int(c.in_place), int(host_gpu_enabled), cache.stride);
        if (!enabled) { return; }
        host_lock.lock_pages = [this](const void * p, size_t bytes, std::string * why) { return vram.register_host(p, bytes, why); };
        host_lock.unlock_pages = [this](const void * p) { return vram.unregister_host(p); };
        host_lock.start();
    }
    // In a decode layer's wait for reads (io_mutex held): the reads left, at the drive's decode rate, as the shadow a
    // registration may take (kimi_host_lock::offer; before the first registration its price is one slot at 13.6 GB/s).
    // Only wanted slots: a page-locked slot's admission upload is a DMA copy, cheaper, and the tier's admission rule
    // then admits more; with every slot page-locked an RTX 5090 admitted 15% more, evicted 25% more and, the RAM
    // copies of admitted experts dropped (exclusive), read 2.5% more experts from the drive.
    void host_lock_in_shadow() {
        host_lock_decide();
        if (!host_lock.enabled || !(learning.read_bytes_per_us > 0)) { return; }
        double left = 0;
        for (size_t index : io_requests) {
            const auto & request = requests[index];
            const auto & slot = slots[request.io_slot];
            for (int role = 0; role < 3; ++role) {
                if (slot.request != index || !(slot.ready_roles & (1u << role))) { left += double(request.role[role].nbytes); }
            }
        }
        host_lock.offer(left / learning.read_bytes_per_us, double(cache.stride) / 13600.0, cache.arena, cache.stride,
            cache.entries.size(), [&](size_t i) { return cache.entries[i].resident; });
    }
    // sink_cache::before_discard: unlock entries [first, first + count) before their pages go.
    void host_unlock(size_t first, size_t count) { host_lock.unlock(first, count, cache.arena, cache.stride); }
    void host_lock_stop() {
        host_lock.stop();
        if (cache.arena) { host_unlock(0, host_lock.snapshot().size()); }
        if (host_lock.decided) {
            std::lock_guard<std::mutex> lock(host_lock.mutex);
            host_lock.report(stderr, "destroy", cache.stride);
        }
        cache.before_discard = nullptr;
    }
    // What an expert the GPU tier does not hold costs this layer: computed by the faster processor that can
    // take it, and for an SSD miss no less than its share of the layer's reads. Feeds the tier's admission.
    void report_miss_cost() {
        size_t ram = 0, ssd = 0;
        for (const auto & r : requests) {
            if (r.vram_slot >= 0) { continue; }
            if (r.cache_hit >= 0) { ++ram; } else { ++ssd; }
        }
        if (!(ram + ssd) || !(learning.cpu_expert_us > 0)) { return; }
        // In place a host job's measured time is its whole cost; a discrete
        // device leaves out the copy. Gather observations before pure pricing.
        kimi_vram_host_costs costs = {};
        if (vram.host_ready()) { vram.host_api->costs2(vram.context, &costs); }
        double resident = 0, host = 0;
        if (costs.in_place) { vram.host_costs(resident, host); }
        vram.set_miss_cost(ssd::place::miss_cost(ram, ssd, learning.cpu_expert_us, current.io_us,
            costs.in_place != 0, host));
    }
    // `last`: the run dispatches no phase after this one, so its participants next wait for the close.
    void compute_phase(bool is_down, bool last = false) {
        begin_dynamic_phase(is_down);
        // Row weights fix one bound per participant, so they keep every thread.
        active_threads = parallel_fn || !row_weights.empty() ? threads : run_participants;
        pending.store(uint32_t(active_threads - 1));
        if (profile_enabled) { phase_dispatch = sink_clock::now(); }
        ++generation;
        // A phase of the coordinator alone stays unpublished unless it is the last: a participant left out skips
        // the generation.
        if (active_threads > 1 || last) {
            publish_phase(active_threads, (is_down ? sink_phase::down : 0) | (parallel_fn ? sink_phase::parallel : 0) |
                (last ? sink_phase::last : 0));
        }
        compute_rows(0, is_down);
        wait_workers();
    }

    // An item's hidden row of the last up/gate phase, quantized for down. SWIGLU gates it first with the graph's
    // kernel over the whole row: split elsewhere than at its vector width, elements would move into its scalar tail.
    void quantize_activation(const sink_work_item & w) {
        if (activation_kind == KIMI_CPU_SINK_ACTIVATION_SWIGLU) {
            ggml_vec_swiglu_f32(hidden, w.hidden, w.hidden, w.hidden_up);
        }
        quantize_hidden(w.hidden, w.down_quant, hidden);
    }

    void wait_workers() {
        const auto done = [&] { return pending.load() == 0; };
        if (poll_us && !parallel_fn) { sink_poll(poll_us, done); }
        while (!done()) { coordinator.park(done); }
    }

    // One phase in which every participant runs function(userdata, ith, nth); `last` as for compute_phase.
    void parallel_phase(kimi_cpu_sink_parallel_fn function, void * userdata, bool last) {
        parallel_fn = function;
        parallel_userdata = userdata;
        try { compute_phase(false, last); }
        catch (...) {
            wait_workers();
            parallel_fn = nullptr;
            parallel_userdata = nullptr;
            throw;
        }
        parallel_fn = nullptr;
        parallel_userdata = nullptr;
    }

    void cancel_and_wait() {
        vram.cancel();
        {
            std::unique_lock<std::mutex> lock(io_mutex);
            if (!io_done) {
                io_cancel = true;
                io_cv.notify_all();
                io_wait(lock, [&] { return io_done; });
            }
        }
        wait_workers();
    }

    void begin_observation(const ggml_tensor * up, bool decode) {
        timeline_active = timeline_enabled && decode;
        timeline_call = {};
        timeline.clear();
        if (timeline_active) { timeline_call[0] = ggml_time_us(); }
        if (!trace_enabled && !profile_enabled && !timeline_enabled && !configuration.events) { return; }
        ++trace_call;
        trace_decode = decode;
        if (!decode) { trace_has_token = false; }
        trace_layer = ssd::host::routed_layer(up);
        if (profile_enabled) {
            caller_cpu_start = sink_thread_cpu_us();
            // between phases: no participant writes its profile
            std::fill(worker_profiles.begin(), worker_profiles.end(), sink_worker_profile());
        }
    }

    void finish_observation() {
        if (!profile_enabled) { return; }
        // Called after the worker barrier and io_done acquire. These counters
        // are separate thread CPU samples, never additions to elapsed wall time.
        current.caller_cpu_us = sink_thread_cpu_us() - caller_cpu_start;
        for (size_t i = trace_decode ? 0 : 1; i < size_t(threads); ++i) {
            const auto & p = worker_profiles[i];
            current.compute_cpu_us += p.up_gate_cpu_us + p.down_cpu_us + p.parallel_cpu_us;
        }
        current.other_us = std::max(0.0, current.total_us - current.compute_us - current.wait_io_us);
    }

    ssd::result<void> publish_observation(bool success) const noexcept {
        if (!configuration.events) { return {}; }
        try {
            std::ostringstream out;
            out.imbue(std::locale::classic());
            out << std::setprecision(17) << "{\"epoch\":" << trace_epoch << ",\"token\":" << trace_token
                << ",\"has_token\":" << (trace_has_token ? "true" : "false") << ",\"layer\":" << trace_layer
                << ",\"decode\":" << (trace_decode ? "true" : "false") << ",\"success\":" << (success ? "true" : "false")
                << ",\"read_bytes\":" << current.read_bytes << ",\"read_requests\":" << current.read_requests
                << ",\"read_completions\":" << current.read_completions << ",\"pending_reads\":" << current.pending_reads
                << ",\"demand_bytes\":" << current.demand_bytes << ",\"cache_hit_bytes\":" << current.cache_hit_bytes
                << ",\"cache_hits\":" << current.cache_hits << ",\"cache_misses\":" << current.cache_misses
                << ",\"vram_hits\":" << vram.hits << ",\"wall_us\":" << current.total_us
                << ",\"wait_io_us\":" << current.wait_io_us << ",\"compute_us\":" << current.compute_us
                << ",\"compute_thread_cpu_us\":" << current.compute_cpu_us << ",\"io_thread_cpu_us\":" << current.io_cpu_us
                << ",\"caller_thread_cpu_us\":" << current.caller_cpu_us;
            // One event a layer: its counters and its routes, the controller API, independent of optional debug
            // logging. Read only after the layer's workers and readers have drained.
            out << ",\"selected_experts\":[";
            for (size_t i = 0; i < requests.size(); ++i) { out << (i ? "," : "") << requests[i].id; }
            out << "],\"vram_hit\":[";
            for (size_t i = 0; i < requests.size(); ++i) {
                out << (i ? "," : "") << (requests[i].vram_slot >= 0 ? "true" : "false");
            }
            out << "],\"cache_hit\":[";
            for (size_t i = 0; i < requests.size(); ++i) {
                out << (i ? "," : "") << (requests[i].cache_hit >= 0 ? "true" : "false");
            }
            out << "]}";
            if (!out) { return { ssd::status::resource, "telemetry serialization failed" }; }
            return configuration.events->event("ssd.place.layer", out.str());
        } catch (const std::bad_alloc &) {
            return { ssd::status::resource, "oom" };
        } catch (const std::exception & e) {
            return { ssd::status::invalid, e.what() };
        }
    }

    void print_observation(bool success) {
        if (!trace_enabled && !profile_enabled && !timeline_enabled) { return; }
        // Keep each JSON record intact even when server logging uses stderr.
        sink_record out;
        if (timeline_active) {
            out.append("[kimi_cpu_sink_timeline] {\"schema\":1,\"clock\":\"CLOCK_MONOTONIC_us\","
                "\"epoch\":%llu,\"token\":%llu,\"layer\":%d,\"call\":%llu,\"success\":%s,\"t\":[",
                (unsigned long long) trace_epoch, (unsigned long long) trace_token, trace_layer,
                (unsigned long long) trace_call, success ? "true" : "false");
            for (size_t i = 0; i < timeline_call.size(); ++i) {
                out.append("%s%llu", i ? "," : "", (unsigned long long) timeline_call[i]);
            }
            out.append("],\"ready_first\":%s,\"ready_scratch_bytes\":%llu,"
                "\"ready_out_of_order_experts\":%llu,\"ordered_reduce_us\":%.3f,\"experts\":[",
                ready_first ? "true" : "false", (unsigned long long) current.ready_scratch_bytes,
                (unsigned long long) current.ready_out_of_order_experts, current.ordered_reduce_us);
            for (size_t i = 0; i < timeline.size(); ++i) {
                const auto & t = timeline[i];
                out.append("%s{\"id\":%d,\"hit\":%s,\"io\":[", i ? "," : "",
                    requests[i].id, requests[i].cache_hit >= 0 ? "true" : "false");
                for (size_t j = 0; j < t.io.size(); ++j) {
                    out.append("%s%llu", j ? "," : "", (unsigned long long) t.io[j]);
                }
                out.append("],\"cpu\":[");
                for (size_t j = 0; j < t.cpu.size(); ++j) {
                    out.append("%s%llu", j ? "," : "", (unsigned long long) t.cpu[j]);
                }
                out.append("],\"workers\":[");
                for (size_t j = 0; j < t.workers.size(); ++j) {
                    const auto & w = t.workers[j];
                    out.append("%s[%llu,%llu,%.3f,%d]", j ? "," : "",
                        (unsigned long long) w.begin, (unsigned long long) w.end, w.cpu_us, w.core);
                }
                out.append("]}");
            }
            out.append("]}\n");
        }
        if (trace_enabled) {
            out.append("[kimi_cpu_sink_trace] {\"schema\":1,\"event\":\"layer\",\"phase\":\"%s\","
                "\"has_token\":%s,\"epoch\":%llu,\"token\":%llu,\"layer\":%d,\"call\":%llu,\"success\":%s,"
                "\"selected_experts\":[", trace_decode ? "decode" : "prefill", trace_has_token ? "true" : "false",
                (unsigned long long) trace_epoch, (unsigned long long) trace_token, trace_layer,
                (unsigned long long) trace_call, success ? "true" : "false");
            for (size_t i = 0; i < requests.size(); ++i) { out.append("%s%d", i ? "," : "", requests[i].id); }
            // the hint this decode run got for the next layer, best first
            out.append("],\"hint\":[");
            const int hint_n = trace_decode && prefetch.last_layer == trace_layer + 1 ? prefetch.last_n : 0;
            for (int i = 0; i < hint_n; ++i) { out.append("%s%d", i ? "," : "", prefetch.last_hint[i]); }
            out.append("],\"expert_bytes\":[");
            for (size_t i = 0; i < requests.size(); ++i) {
                const auto & r = requests[i];
                out.append("%s%llu", i ? "," : "", (unsigned long long)
                    (r.role[0].nbytes + r.role[1].nbytes + r.role[2].nbytes));
            }
            out.append("],\"cache_hit\":[");
            for (size_t i = 0; i < requests.size(); ++i) {
                out.append("%s%s", i ? "," : "", requests[i].cache_hit >= 0 ? "true" : "false");
            }
            out.append("],\"ram_resident_before\":[");
            for (size_t i = 0; i < requests.size(); ++i) {
                out.append("%s%s", i ? "," : "", requests[i].trace_ram_resident_before ? "true" : "false");
            }
            out.append("],\"vram_hit\":[");
            for (size_t i = 0; i < requests.size(); ++i) {
                out.append("%s%s", i ? "," : "", requests[i].vram_slot >= 0 ? "true" : "false");
            }
            out.append("],\"expert_read_bytes\":[");
            for (size_t i = 0; i < requests.size(); ++i) {
                out.append("%s%llu", i ? "," : "", (unsigned long long) requests[i].read_bytes);
            }
            out.append("],\"hits\":%llu,\"misses\":%llu,\"read_bytes\":%llu,\"demand_bytes\":%llu,"
                "\"cache_budget_bytes\":%llu,\"cache_capacity_bytes\":%llu,\"cache_max_experts_per_layer\":%d,"
                "\"cache_policy\":\"%s\",\"cache_allocation\":\"%s\",\"cache_layer_priority\":[",
                (unsigned long long) current.cache_hits, (unsigned long long) current.cache_misses,
                (unsigned long long) current.read_bytes, (unsigned long long) current.demand_bytes,
                (unsigned long long) cache.stats.budget_bytes, (unsigned long long) cache.stats.capacity_bytes,
                cache.quota, cache.policy_name(), cache.allocation_name());
            for (size_t i = 0; i < cache.layers.size(); ++i) { out.append("%s%d", i ? "," : "", cache.layers[i].id); }
            out.append("]}\n");
        }
        if (profile_enabled) {
            out.append("[kimi_cpu_sink_profile] {\"schema\":1,\"event\":\"layer\",\"phase\":\"%s\","
                "\"scope\":\"%s\",\"has_token\":%s,\"epoch\":%llu,\"token\":%llu,\"layer\":%d,\"call\":%llu,\"success\":%s,"
                "\"experts\":%llu,\"threads\":%d,\"expert_job\":%d,\"total_us\":%.3f,\"compute_wall_us\":%.3f,"
                "\"wait_io_us\":%.3f,\"other_wall_us\":%.3f,\"io_job_wall_us\":%.3f,\"io_wait_wall_us\":%.3f,"
                "\"io_credit_wait_wall_us\":%.3f,\"compute_thread_cpu_us\":%.3f,\"caller_thread_cpu_us\":%.3f,"
                "\"io_thread_cpu_us\":%.3f,\"input_quantize_us\":%.3f,\"hidden_quantize_us\":%.3f,"
                "\"up_gate_us\":%.3f,\"down_reduce_us\":%.3f,\"workers\":[",
                trace_decode ? "decode" : "prefill", trace_decode ? "sink_run" : "sink_stream",
                trace_has_token ? "true" : "false", (unsigned long long) trace_epoch,
                (unsigned long long) trace_token, trace_layer,
                (unsigned long long) trace_call, success ? "true" : "false", (unsigned long long) current.experts,
                threads + 1, 0, current.total_us, current.compute_us, current.wait_io_us,
                current.other_us, current.io_us, current.io_wait_us, current.io_credit_wait_us,
                current.compute_cpu_us, current.caller_cpu_us, current.io_cpu_us, current.input_quantize_us,
                current.hidden_quantize_us, current.up_gate_us, current.down_reduce_us);
            for (int i = 0; i < threads; ++i) {
                const auto & p = worker_profiles[i];
                out.append("%s{\"ith\":%d,\"phases\":%llu,\"up_gate_wall_us\":%.3f,\"down_wall_us\":%.3f,"
                    "\"up_gate_cpu_us\":%.3f,\"down_cpu_us\":%.3f,\"parallel_wall_us\":%.3f,\"parallel_cpu_us\":%.3f,"
                    "\"dispatch_us\":%.3f,\"max_dispatch_us\":%.3f,\"no_work_wakeups\":%llu}",
                    i ? "," : "", i, (unsigned long long) p.phases, p.up_gate_wall_us, p.down_wall_us,
                    p.up_gate_cpu_us, p.down_cpu_us, p.parallel_wall_us, p.parallel_cpu_us,
                    p.dispatch_us, p.max_dispatch_us, (unsigned long long) p.no_work_wakeups);
            }
            out.append("]}\n");
        }
        out.flush();
    }

    void account_read_bytes(const sink_completion & cqe) {
        if (cqe.user_data >= sink_lookahead_tag) { return; } // a read-ahead's: not this run's read
        if (cqe.user_data >= 3 * slots.size()) { throw std::runtime_error("invalid CPU sink CQE target"); }
        if (cqe.res <= 0) { return; }
        auto & request = requests[slots[cqe.user_data / 3].request];
        if (trace_enabled) { request.read_bytes += static_cast<uint64_t>(cqe.res); }
        current.read_bytes += static_cast<uint64_t>(cqe.res);
        if (request.cache_admit >= 0) { current.cache_fill_read_bytes += cqe.res; }
        else { current.ring_read_bytes += cqe.res; }
    }

    void io_loop() {
        std::unique_lock<std::mutex> lock(io_mutex);
        for (;;) {
            io_cv.wait(lock, [&] { return io_stop || io_job; });
            if (io_stop) { return; }
            io_job = false;
            io_start = sink_clock::now();
            lock.unlock();
            const double cpu_start = profile_enabled ? sink_thread_cpu_us() : 0.0;
            try {
                io_execute();
            } catch (const std::exception & e) {
                io_break(e);
            }
            lock.lock();
            if (profile_enabled) { current.io_cpu_us = sink_thread_cpu_us() - cpu_start; }
            const auto ended = sink_clock::now();
            io_completed_end = ring.completed();
            io_pending_end = ring.pending();
            // A grouped-prefill call's reads are over: the drive's idle window opens, and the next layer's read-ahead
            // starts unless the run failed or was cancelled. Its first reads go out before the run is told its reads
            // are over, so that the next call always finds them; the rest follow while the run computes its tail.
            if (lookahead.job_layer >= 0) {
                lookahead.window_open = true;
                lookahead.window_start = ended;
                lookahead.window_layer = lookahead.job_layer;
                lookahead.job_layer = -1;
            }
            const bool ahead = lookahead.active && !lookahead.stop && !io_stop && !io_failed && !poisoned;
            if (lookahead.active && !ahead) { lookahead.active = false; }
            if (ahead) {
                lock.unlock();
                lookahead_step();
                lock.lock();
            }
            io_end(ended);
            if (ahead && lookahead.active) {
                lock.unlock();
                lookahead_run();
                lock.lock();
            }
        }
    }

    // A broken ring cannot be reused. If it cannot even drain, end the process rather than return user buffers
    // still owned by I/O.
    void io_break(const std::exception & e) {
        try {
            sink_io_value(ring.drain(this, [](void * userdata, const sink_completion & cqe) {
                static_cast<kimi_cpu_sink *>(userdata)->account_read_bytes(cqe);
            }));
        } catch (const std::exception & drain_error) {
            std::fprintf(stderr, "[kimi_cpu_sink] cannot drain I/O after %s: %s; aborting\n",
                         e.what(), drain_error.what());
            std::abort();
        }
        ring.close_ring();
        io_inflight = 0;
        io_failed = true;
        std::lock_guard<std::mutex> error_lock(io_mutex);
        poisoned = true;
        io_error = e.what();
        ++current.read_errors;
    }

    // io_mutex held: the run's reads are over (at `ended`), and the drive's idle window opens for the next layer's
    // prefetch.
    void io_end(sink_clock::time_point ended = sink_clock::now()) {
        current.io_us = std::chrono::duration<double, std::micro>(ended - io_start).count();
        if (timeline_active) { timeline_call[4] = ggml_time_us(); }
        io_done = true;
        io_by_run = false;
        prefetch_open_gate();
        io_cv.notify_all();
    }

    void io_execute() {
        io_next = 0;
        io_inflight = 0;
        io_failed = false;
        while (io_next < io_requests.size() || io_inflight) {
            io_issue();
            const bool consumed = io_reap();
            if (io_failed && !io_inflight) { return; }
            if (io_inflight && !consumed) {
                const auto wait_start = profile_enabled ? sink_clock::now() : sink_clock::time_point();
                sink_io_value(ring.wait());
                if (profile_enabled) { current.io_wait_us += elapsed_us(wait_start); }
            } else if (!io_inflight && io_next < io_requests.size()) {
                // Always timed: the drive is idle here, and the prefill cost model takes the drive rate
                // from the read bytes over the I/O time without these waits.
                const auto wait_start = sink_clock::now();
                std::unique_lock<std::mutex> lock(io_mutex);
                io_cv.wait(lock, [&] { return io_cancel || slots[io_next % slots.size()].state == sink_slot::empty; });
                current.io_credit_wait_us += elapsed_us(wait_start);
            }
        }
    }

    // Queues the run's next reads, as many as the free ring slots take, and submits them.
    void io_issue() {
        const size_t batch_begin = io_next;
        unsigned queued = 0;
        {
            std::lock_guard<std::mutex> lock(io_mutex);
            io_failed = io_failed || io_cancel;
            auto queue_role = [&](size_t ordinal, unsigned role) {
                const size_t index = ordinal % slots.size();
                auto & request = requests[io_requests[ordinal]];
                const bool cache_fill = request.cache_admit >= 0;
                const bool external_target = cache_fill ||
                    (request.vram_ssd_transient && !request.vram_ssd_streaming);
                const bool fixed = ring.read_fixed() && !external_target;
                sink_io_value(ring.queue({request.role[role].fd_direct, request.role[role].offset, request.role[role].nbytes},
                    request.read_target[role], 3 * index + role, fixed));
                ++queued;
                ++current.read_requests;
                if (cache_fill) { ++current.cache_fill_read_requests; }
                else { ++current.ring_read_requests; }
                if (fixed) { ++current.fixed_read_requests; }
            };
            while (!io_failed && io_next < io_requests.size()) {
                const size_t index = io_next % slots.size();
                sink_slot & slot = slots[index];
                if (slot.state != sink_slot::empty) { break; }
                slot.state = sink_slot::reading;
                slot.remaining = 3;
                slot.ready_roles = 0;
                slot.request = io_requests[io_next];
                if (timeline_active) { timeline[slot.request].io[0] = ggml_time_us(); }
                if (!role_major) { for (unsigned role = 0; role < 3; ++role) { queue_role(io_next, role); } }
                ++io_next;
            }
            if (role_major) {
                // Keep one submit for the whole batch, but put every
                // expert's up/gate before any down so early-up-gate can
                // overlap the CPU work with the streamed down reads.
                for (size_t j = batch_begin; j < io_next; ++j) {
                    queue_role(j, 0);
                    queue_role(j, 1);
                }
                for (size_t j = batch_begin; j < io_next; ++j) { queue_role(j, 2); }
            }
        }
        if (!queued) { return; }
        const uint64_t submit_begin = timeline_active ? ggml_time_us() : 0;
        sink_io_value(ring.submit(queued)); io_inflight += queued;
        if (timeline_active) {
            const uint64_t submit_end = ggml_time_us();
            for (size_t j = batch_begin; j < io_next; ++j) {
                auto & t = timeline[io_requests[j]];
                t.io[1] = submit_begin; t.io[2] = submit_end;
            }
        }
    }

    // Takes every completion the ring has; false when it had none.
    bool io_reap() {
        sink_completion cqe;
        bool consumed = false;
        while (sink_io_value(ring.pop(cqe))) {
            consumed = true;
            // A run's reads never share the ring with a read-ahead (each entry stops it first): still, one is the
            // read-ahead's, never this run's.
            if (cqe.user_data >= sink_lookahead_tag) { lookahead_completion(cqe); continue; }
            if (!io_inflight || cqe.user_data >= 3 * slots.size()) {
                throw std::runtime_error("invalid CPU sink CQE");
            }
            --io_inflight;
            const size_t index = cqe.user_data / 3;
            const unsigned role = static_cast<unsigned>(cqe.user_data % 3);
            std::lock_guard<std::mutex> lock(io_mutex);
            sink_slot & slot = slots[index];
            if (timeline_active) { timeline[slot.request].io[3 + role] = ggml_time_us(); }
            const auto & range = requests[slot.request].role[role];
            const size_t expected = range.nbytes;
            const size_t required = range.payload_nbytes ? range.data_offset + range.payload_nbytes : expected;
            account_read_bytes(cqe);
            // A final GGUF payload may end before the rounded physical
            // read does. Accept only a complete logical payload, never a
            // truncated weight, and preserve the actual CQE byte count.
            if (cqe.res < 0 || size_t(cqe.res) < required || size_t(cqe.res) > expected) {
                ++current.read_errors;
                io_failed = true;
                if (io_error.empty()) {
                    io_error = "direct expert read failed/short: result=" + std::to_string(cqe.res) +
                               " expected=" + std::to_string(expected);
                }
            } else {
                slot.ready_roles |= 1u << role;
            }
            if (!--slot.remaining) { slot.state = sink_slot::ready; }
            io_cv.notify_all();
        }
        return consumed;
    }

    // A decode run's reads go out as soon as prepare has given them targets, before the input's quantization and
    // the GPU's launch. On the nonblocking schedule the run reads for itself: its coordinator submits them and takes
    // their completions while it waits for them (io_poll), so none waits for a thread to wake, and no thread sleeps
    // while they are in flight. The other schedules hand them to the I/O thread.
    void io_begin_decode() {
        {
            std::lock_guard<std::mutex> lock(io_mutex);
            io_error.clear();
            io_cancel = false;
            if (timeline_active) { timeline_call[3] = ggml_time_us(); }
            if (io_requests.empty()) { return; }
            io_done = false;
            io_by_run = nonblocking_down_sched;
            if (!io_by_run) { io_job = true; }
        }
        if (!io_by_run) { io_cv.notify_one(); return; }
        io_next = 0;
        io_inflight = 0;
        io_failed = false;
        io_start = sink_clock::now();
        io_step();
    }

    // A run that reads for itself: queues what the free ring slots take and takes the completions the ring has;
    // its reads are over once none is left to queue or in flight. True when a completion came.
    bool io_step() {
        bool consumed = false;
        try {
            io_issue();
            consumed = io_reap();
        } catch (const std::exception & e) {
            io_break(e);
        }
        if (!io_inflight && (io_failed || io_next == io_requests.size())) {
            std::lock_guard<std::mutex> lock(io_mutex);
            io_end();
        }
        return consumed;
    }

    // The coordinator of a run that reads for itself, between its looks for work: one step of its reads, and when
    // no completion came, a yield of its CPU, to a thread that shares it (the GPU tier's) or straight back to it.
    // True when a completion came.
    bool io_poll() {
        const bool consumed = io_by_run && io_step();
        if (!consumed) { std::this_thread::yield(); }
        return consumed;
    }

    // io_mutex held (lock): polls until ready(), taking the run's own reads meanwhile.
    template <class Ready> void io_poll_until(std::unique_lock<std::mutex> & lock, Ready ready) {
        while (!ready()) {
            lock.unlock();
            io_poll();
            lock.lock();
        }
    }

    // io_mutex held (lock): waits until ready(), polling when the run reads for itself, else until the I/O thread
    // notifies.
    template <class Ready> void io_wait(std::unique_lock<std::mutex> & lock, Ready ready) {
        if (io_by_run) { io_poll_until(lock, ready); }
        else { io_cv.wait(lock, ready); }
    }

    // ---- The grouped prefill's cross-layer read-ahead (lookahead_state) ----

    static double lookahead_median(std::vector<double> values) {
        if (values.empty()) { return 0; }
        const size_t mid = values.size() / 2;
        std::nth_element(values.begin(), values.begin() + mid, values.end());
        const double upper = values[mid];
        if (values.size() % 2) { return upper; }
        return (upper + *std::max_element(values.begin(), values.begin() + mid)) / 2;
    }

    // The I/O thread: queues the plan's next experts into their RAM cache slots, as many as the ring's depth takes
    // beside the reads in flight, submits them and takes the completions the ring has. The ring is the I/O thread's
    // here: a run's reads are over, and the next entry stops this before it uses the ring. False when nothing happened.
    bool lookahead_step() {
        auto & batch = lookahead.batch;
        batch.clear();
        try {
            {
                std::lock_guard<std::mutex> lock(io_mutex);
                const size_t depth = 3 * slots.size(); // the depth the layers' own reads reach
                while (!lookahead.stop && !io_stop && lookahead.next < lookahead.plan.size() &&
                        lookahead.inflight + batch.size() + 3 <= depth) {
                    auto & read = lookahead.plan[lookahead.next];
                    read.issued = true;
                    read.remaining = 3;
                    for (int role = 0; role < 3; ++role) { batch.push_back({ lookahead.next, role }); }
                    ++lookahead.next;
                    ++lookahead.issued;
                }
            }
            // the plan does not change while the read-ahead runs, and an issued read's slot is inflight in the cache
            for (const auto & item : batch) {
                const auto & read = lookahead.plan[item.first];
                const auto & source = read.role[item.second];
                sink_io_value(ring.queue({ source.fd_direct, source.offset, source.nbytes }, read.target[item.second],
                    sink_lookahead_tag + 3 * item.first + unsigned(item.second), false));
            }
            if (!batch.empty()) {
                sink_io_value(ring.submit(unsigned(batch.size())));
                lookahead.inflight += unsigned(batch.size());
            }
            sink_completion cqe;
            bool consumed = false;
            while (sink_io_value(ring.pop(cqe))) {
                consumed = true;
                if (cqe.user_data < sink_lookahead_tag) { throw std::runtime_error("a run's CQE beside the read-ahead"); }
                lookahead_completion(cqe);
            }
            return consumed || !batch.empty();
        } catch (const std::exception & e) {
            lookahead_break(e);
            return false;
        }
    }
    // The I/O thread: a read-ahead completion. A failed or short read (or an injected failure) fails its expert, whose
    // slot goes back once the next entry finishes the plan; nothing else fails, and the ring stays usable.
    void lookahead_completion(const sink_completion & cqe) {
        const uint64_t tag = cqe.user_data - sink_lookahead_tag;
        if (!lookahead.inflight || tag >= 3 * uint64_t(lookahead.plan.size())) {
            throw std::runtime_error("invalid CPU sink read-ahead CQE");
        }
        --lookahead.inflight;
        const size_t index = size_t(tag / 3);
        const int role = int(tag % 3);
        std::lock_guard<std::mutex> lock(io_mutex);
        auto & read = lookahead.plan[index];
        if (!read.issued || !read.remaining) { throw std::runtime_error("CPU sink read-ahead CQE for an expert not being read"); }
        const auto & range = read.role[role];
        const size_t expected = range.nbytes;
        const size_t required = range.payload_nbytes ? range.data_offset + range.payload_nbytes : expected;
        const bool injected = lookahead.fail_every && ++lookahead.completions % lookahead.fail_every == 0;
        if (injected || cqe.res < 0 || size_t(cqe.res) < required || size_t(cqe.res) > expected) {
            if (!read.failed) {
                std::fprintf(stderr, "[kimi_cpu_prefill_lookahead] event=read_failed layer=%d expert=%d role=%d result=%d"
                    " expected=%zu injected=%d\n", lookahead.layer, read.expert, role, int(cqe.res), expected, int(injected));
            }
            read.failed = true;
        } else {
            read.ready_roles |= 1u << role;
        }
        if (--read.remaining) { return; }
        if (read.failed || read.ready_roles != 7u) { read.failed = true; ++lookahead.failed; }
        else { ++lookahead.landed; }
    }
    // The I/O thread, after a run's reads and the read-ahead's first step: issues the rest of the plan as the ring's
    // depth allows, until the plan is read or the next call stops it, then waits for the reads in flight.
    void lookahead_run() {
        for (;;) {
            const bool progressed = lookahead_step();
            std::unique_lock<std::mutex> lock(io_mutex);
            if (!lookahead.active) { return; } // the ring broke (lookahead_break)
            const bool more = !lookahead.stop && !io_stop && lookahead.next < lookahead.plan.size();
            if (!lookahead.inflight && !more) { break; }
            if (progressed && more) { continue; }
            lock.unlock();
            try { sink_io_value(ring.wait()); }
            catch (const std::exception & e) { lookahead_break(e); return; }
        }
        std::lock_guard<std::mutex> lock(io_mutex);
        lookahead.active = false;
        io_cv.notify_all();
    }
    // The I/O thread: the ring failed under the read-ahead (not a read: a system call). As io_break: drain it, then it
    // is closed, the sink poisoned for the next entry; the call in progress, whose reads are over, is not failed.
    void lookahead_break(const std::exception & e) {
        try {
            sink_io_value(ring.drain(this, [](void *, const sink_completion &) {}));
        } catch (const std::exception & drain_error) {
            std::fprintf(stderr, "[kimi_cpu_sink] cannot drain the read-ahead after %s: %s; aborting\n", e.what(),
                drain_error.what());
            std::abort();
        }
        ring.close_ring();
        std::lock_guard<std::mutex> lock(io_mutex);
        lookahead.inflight = 0;
        for (auto & read : lookahead.plan) {
            if (read.issued && read.remaining) { read.remaining = 0; read.failed = true; ++lookahead.failed; }
        }
        poisoned = true;
        lookahead.active = false;
        std::fprintf(stderr, "[kimi_cpu_prefill_lookahead] event=ring_failed error=%s\n", e.what());
        io_cv.notify_all();
    }
    // (lock on io_mutex) The read-ahead issues nothing more, and the reads it has in flight landed or failed.
    void lookahead_stop(std::unique_lock<std::mutex> & lock) {
        if (!lookahead.active) { return; }
        lookahead.stop = true;
        io_cv.notify_all();
        io_cv.wait(lock, [&] { return !lookahead.active; });
        lookahead.stop = false;
    }
    // The read-ahead stopped (run_mutex held): each of the plan's cache slots holds the expert it read, published, or
    // goes back. For a plan of `layer`, `published` gets the slots that hold its experts now.
    void lookahead_finish(int32_t layer, std::vector<int> * published) {
        for (const auto & read : lookahead.plan) {
            const bool landed = read.issued && !read.remaining && !read.failed;
            cache.prefetch_done(read.entry, read.generation, landed, false);
            if (landed && published && lookahead.layer == layer) { published->push_back(read.entry); }
        }
        lookahead.plan.clear();
        lookahead.next = 0;
        lookahead.layer = -1;
    }
    // Every other entry that holds the run mutex and may use the ring or the cache (a phase begin, a decode run, a
    // reserve, a reset): the read-ahead stops, what it read is published, and the pass ends.
    void lookahead_quiesce() {
        {
            std::unique_lock<std::mutex> lock(io_mutex);
            lookahead_stop(lock);
            lookahead.hint_layer = lookahead.planned_layer = -1;
            lookahead.window_open = false;
            lookahead.pass_layer = -1;
        }
        lookahead_finish(-1, nullptr);
        lookahead.landed_entries.clear();
    }
    // sink_cache::before_unmap (run_mutex held): the cache drops its arena, and with it the plan's slots (its generation
    // changes); the read-ahead's reads into it land first.
    void lookahead_unmap() {
        std::unique_lock<std::mutex> lock(io_mutex);
        lookahead_stop(lock);
        lookahead.plan.clear();
        lookahead.next = 0;
        lookahead.layer = -1;
        lookahead.landed_entries.clear();
    }
    // A grouped-prefill call for `layer`, as it starts (run_mutex held, before its prepare): the read-ahead the call
    // before planned stops issuing, its reads in flight land, and the experts it read are published in the RAM cache,
    // where this call's prefill read takes them as hits. A call out of the pass's layer order starts a new pass.
    void lookahead_handover(int32_t layer) {
        auto & r = lookahead_report;
        {
            std::unique_lock<std::mutex> lock(io_mutex);
            const auto now = sink_clock::now();
            double window_us = -1;
            if (lookahead.window_open && layer >= 0 && lookahead.window_layer + 1 == layer) {
                window_us = std::chrono::duration<double, std::micro>(now - lookahead.window_start).count();
            }
            lookahead.window_open = false;
            lookahead_stop(lock);
            r = {};
            r.layer = layer;
            r.window_us = window_us;
            r.planned = lookahead.layer == layer ? lookahead.plan.size() : 0;
            r.issued = lookahead.issued; r.landed = lookahead.landed; r.failed = lookahead.failed;
            lookahead.issued = lookahead.landed = lookahead.failed = 0;
            if (layer < 0 || layer != lookahead.pass_layer + 1) {
                if (!lookahead.pass_windows_us.empty()) { lookahead.last_pass_median_us = lookahead_median(lookahead.pass_windows_us); }
                lookahead.pass_windows_us.clear();
            }
            if (window_us >= 0) { lookahead.pass_windows_us.push_back(window_us); }
            lookahead.pass_layer = layer;
        }
        lookahead.landed_entries.clear();
        lookahead_finish(layer, &lookahead.landed_entries);
    }
    // The call for `layer` of a grouped prefill, its requests prepared: the prediction the call before it planned
    // with (for this layer) against the routes this layer took, a rank at a time. A prediction of another layer or
    // pass measures nothing.
    void lookahead_measure(int32_t layer, const kimi_cpu_sink_prefill_batch & batch) {
        auto & la = lookahead;
        const bool same = la.planned_layer == layer && layer >= 0 && la.planned_tokens == batch.ntokens &&
            la.planned_k > 0 && batch.nslots > 0 && batch.route_request_index;
        la.planned_layer = -1;
        if (!same) { return; }
        const int k = std::min(la.planned_k, 16);
        for (int t = 0; t < batch.ntokens; ++t) {
            const int * routes = batch.route_request_index + size_t(t) * size_t(batch.nslots);
            for (int rank = 0; rank < k; ++rank) {
                const int32_t predicted = la.planned_hint[size_t(t) * size_t(la.planned_k) + size_t(rank)];
                bool selected = false;
                for (int s = 0; s < batch.nslots && !selected; ++s) {
                    const int request = routes[s];
                    selected = request >= 0 && size_t(request) < requests.size() && requests[size_t(request)].id == predicted;
                }
                ++la.rank_predicted[rank];
                la.rank_selected[rank] += selected;
            }
        }
    }
    // The call for `layer` of a grouped prefill (run_mutex held, the read-ahead stopped, before its reads go out): the
    // plan of the next layer's read-ahead. The prediction of the next layer's routes names each expert for some tokens
    // at some ranks; an expert named at ranks r_1..r_m is selected with P = 1 - prod(1 - q_r), q_r the rate at which
    // rank r's prediction was selected in the prefill passes so far (lookahead_measure). The plan: the next layer's
    // experts with P >= 1/2 that the GPU tier does not hold and the RAM cache's prefill read would not serve, most
    // likely first, at most N = ceil(window x drive rate / expert bytes), the experts the drive reads in the idle window
    // this pass is estimated to have, and as many as the layer's RAM cache slots take (prefetch_reserve, as the decode
    // prefetch takes one: a free slot first, else the layer's coldest). The estimate: the windows' median over the pass
    // so far, else the last pass's. No plan without a prediction of the next layer or a measured rate, before any
    // window, without a drive line, or for the last routed layer (the next one resolves no expert).
    void lookahead_plan_next(int32_t layer, const ggml_tensor * up, int ntokens, int nslots) {
        auto & r = lookahead_report;
        // what an earlier call's read-ahead borrowed went to that call's read: its slots go back to their layers
        cache.return_borrowed(layer);
        {
            std::lock_guard<std::mutex> lock(io_mutex);
            r.estimate_us = !lookahead.pass_windows_us.empty() ? lookahead_median(lookahead.pass_windows_us) :
                lookahead.last_pass_median_us;
        }
        // the prediction this plan takes, kept for the next call's measure
        const bool predicted = lookahead.hint_layer == layer + 1 && lookahead.hint_tokens == ntokens && layer >= 0;
        lookahead.planned_layer = predicted ? layer + 1 : -1;
        if (predicted) {
            lookahead.planned_hint.swap(lookahead.hint);
            lookahead.planned_k = lookahead.hint_k;
            lookahead.planned_tokens = lookahead.hint_tokens;
        }
        lookahead.hint_layer = -1;
        if (!lookahead.enabled || !predicted || nslots <= 0 || requests.empty()) { return; }
        // the drive's calibrated line (decode's layers with reads, the prefill window's, the measured prior)
        const double drive_us_per_mb = r.drive_us_per_mb = place_table(ntokens).drive_us_per_mb;
        const auto & first = requests.front();
        const double expert_mb = double(first.role[0].nbytes + first.role[1].nbytes + first.role[2].nbytes) / 1e6;
        if (!(r.estimate_us > 0) || !(drive_us_per_mb > 0) || !(expert_mb > 0)) { return; }
        r.next_n = uint64_t(std::ceil(r.estimate_us / (expert_mb * drive_us_per_mb)));
        const int32_t next = layer + 1;
        const int experts = int(up->ne[2]);
        // log(1 - P) of each expert from the ranks the prediction names it at; a rate not yet measured (a rank no
        // token has been predicted at) leaves it out
        const int k = std::min(lookahead.planned_k, 16);
        double log_miss[16];
        for (int rank = 0; rank < k; ++rank) {
            const uint64_t n = lookahead.rank_predicted[rank];
            const double q = n ? double(lookahead.rank_selected[rank]) / double(n) : 0.0;
            log_miss[rank] = q >= 1.0 ? -INFINITY : std::log1p(-q);
        }
        std::vector<double> log_unused(size_t(experts), 0.0);
        for (int t = 0; t < ntokens; ++t) {
            for (int rank = 0; rank < k; ++rank) {
                const int32_t e = lookahead.planned_hint[size_t(t) * size_t(lookahead.planned_k) + size_t(rank)];
                if (e >= 0 && e < experts) { log_unused[size_t(e)] += log_miss[rank]; }
            }
        }
        std::vector<int32_t> ids(static_cast<size_t>(experts));
        std::iota(ids.begin(), ids.end(), 0);
        std::vector<uint8_t> held(size_t(experts), 0);
        vram.held(next, ids.data(), experts, held.data());
        // A read still in flight when the next call starts delays its own reads by about the read's time, which is
        // what a used read saves: reading an expert ahead pays when P >= 1/2.
        std::vector<int32_t> likely;
        for (int e = 0; e < experts; ++e) {
            if (held[size_t(e)] || cache.prefill_holds(next, e) || !(log_unused[size_t(e)] <= std::log(0.5))) { continue; }
            likely.push_back(e);
        }
        std::stable_sort(likely.begin(), likely.end(),
            [&](int32_t a, int32_t b) { return log_unused[size_t(a)] < log_unused[size_t(b)]; });
        r.likely = likely.size();
        if (likely.empty()) { ++lookahead.unlikely; return; }
        const size_t target = size_t(std::min<uint64_t>(r.next_n, likely.size()));
        if (!target) { return; }
        char names[3][GGML_MAX_NAME];
        std::snprintf(names[0], sizeof(names[0]), "blk.%d.ffn_up_exps.weight", next);
        std::snprintf(names[1], sizeof(names[1]), "blk.%d.ffn_gate_exps.weight", next);
        std::snprintf(names[2], sizeof(names[2]), "blk.%d.ffn_down_exps.weight", next);
        std::vector<sink_lookahead_read> plan;
        plan.reserve(target);
        uint64_t unresolved = 0;
        for (size_t i = 0; i < likely.size() && plan.size() < target; ++i) {
            sink_lookahead_read read;
            read.expert = likely[i];
            bool resolved = true;
            for (int role = 0; role < 3 && resolved; ++role) {
                const auto & source = read.role[role];
                resolved = resolve(names[role], read.expert, &read.role[role]) && source.offset % 4096 == 0 &&
                    source.nbytes && source.nbytes % 4096 == 0 && source.nbytes <= size_t(std::numeric_limits<int>::max());
            }
            if (!resolved) { ++unresolved; continue; }
            read.entry = cache.prefetch_reserve(next, read.expert, read.role);
            if (read.entry == -2) { continue; }            // cached, or on its way
            if (read.entry < 0) {                           // the layer has no idle slot left: one a finished layer lends
                read.entry = cache.prefetch_borrow(next, read.expert, read.role, layer);
                if (read.entry < 0) { ++r.no_slot; break; }
                ++r.borrowed;
            }
            read.generation = cache.generation;
            for (int role = 0; role < 3; ++role) { read.target[role] = cache.data(read.entry, role); }
            plan.push_back(read);
        }
        lookahead.unresolved += unresolved;
        if (plan.empty()) { return; } // none resolves (the last routed layer's call), or no slot
        std::lock_guard<std::mutex> lock(io_mutex);
        lookahead.plan.swap(plan);
        lookahead.layer = next;
        lookahead.next = 0;
        r.next_planned = lookahead.plan.size();
    }
    // After the RAM cache's prefill read (run_mutex held): the hits among the experts the read-ahead published for this
    // layer are the reads it moved earlier; the others stay cached as the coldest of what the cache holds.
    void lookahead_count_used() {
        auto & r = lookahead_report;
        for (const auto & request : requests) {
            if (request.vram_slot >= 0 || request.cache_hit < 0 ||
                    std::find(lookahead.landed_entries.begin(), lookahead.landed_entries.end(), request.cache_hit) ==
                    lookahead.landed_entries.end()) {
                continue;
            }
            ++r.used;
            r.bytes += request.role[0].nbytes + request.role[1].nbytes + request.role[2].nbytes;
        }
        r.unused = lookahead.landed_entries.size() - std::min<uint64_t>(r.used, lookahead.landed_entries.size());
        lookahead.landed_entries.clear();
    }
    // io_mutex held, as a grouped-prefill call for `layer` publishes its I/O job: the drive's idle window since the
    // previous layer's reads ended closes, sampled when that layer was this one's predecessor in the pass, and the
    // next layer's plan goes to the I/O thread with the job. A call without a job leaves the drive idle from here.
    // True when there is a job.
    bool lookahead_publish(int32_t layer, bool reads) {
        const auto now = sink_clock::now();
        lookahead.stop = false;
        lookahead.active = !lookahead.plan.empty();
        if (!reads && !lookahead.active) {
            lookahead.window_open = layer >= 0;
            lookahead.window_start = now;
            lookahead.window_layer = layer;
            return false;
        }
        lookahead.job_layer = layer;
        return true;
    }
    // One line a grouped-prefill call: the window before its reads (-1: none sampled), the estimate its plan of the
    // next layer took, what the previous call's plan read for it and what it used, and its own plan.
    void lookahead_print() {
        const auto & r = lookahead_report;
        ++lookahead.layers;
        lookahead.planned += r.next_planned; lookahead.issued_total += r.issued; lookahead.landed_total += r.landed;
        lookahead.failed_total += r.failed; lookahead.used += r.used; lookahead.unused += r.unused;
        lookahead.used_bytes += r.bytes; lookahead.no_slot += r.no_slot; lookahead.borrowed += r.borrowed;
        std::fprintf(stderr, "[kimi_cpu_prefill_lookahead] layer=%d enabled=%d window_us=%.0f estimate_us=%.0f"
            " drive_us_per_mb=%.3f likely=%llu planned=%llu issued=%llu read=%llu failed=%llu used=%llu unused=%llu"
            " bytes=%llu next_n=%llu next_planned=%zu no_slot=%llu borrowed=%llu\n", r.layer, int(lookahead.enabled),
            r.window_us, r.estimate_us, r.drive_us_per_mb, (unsigned long long) r.likely, (unsigned long long) r.planned,
            (unsigned long long) r.issued, (unsigned long long) r.landed, (unsigned long long) r.failed,
            (unsigned long long) r.used, (unsigned long long) r.unused, (unsigned long long) r.bytes,
            (unsigned long long) r.next_n, r.next_planned, (unsigned long long) r.no_slot, (unsigned long long) r.borrowed);
    }
    void lookahead_summary() {
        if (!lookahead.enabled || !lookahead.layers) { return; }
        std::fprintf(stderr, "[kimi_cpu_prefill_lookahead] reason=destroy layers=%llu planned=%llu issued=%llu read=%llu"
            " failed=%llu used=%llu unused=%llu used_bytes=%llu no_slot=%llu unresolved=%llu unlikely=%llu borrowed=%llu\n",
            (unsigned long long) lookahead.layers, (unsigned long long) lookahead.planned,
            (unsigned long long) lookahead.issued_total, (unsigned long long) lookahead.landed_total,
            (unsigned long long) lookahead.failed_total, (unsigned long long) lookahead.used,
            (unsigned long long) lookahead.unused, (unsigned long long) lookahead.used_bytes,
            (unsigned long long) lookahead.no_slot, (unsigned long long) lookahead.unresolved,
            (unsigned long long) lookahead.unlikely, (unsigned long long) lookahead.borrowed);
        std::string rates;
        for (int rank = 0; rank < 16 && lookahead.rank_predicted[rank]; ++rank) {
            char one[32];
            std::snprintf(one, sizeof(one), "%s%.3f", rank ? "," : "",
                double(lookahead.rank_selected[rank]) / double(lookahead.rank_predicted[rank]));
            rates += one;
        }
        std::fprintf(stderr, "[kimi_cpu_prefill_lookahead] reason=destroy_prediction tokens=%llu rank_selected=%s\n",
            (unsigned long long) lookahead.rank_predicted[0], rates.c_str());
    }

    template<class T> void resize_scratch(std::vector<T> & values, size_t count) {
        if (count <= values.capacity()) { values.resize(count); return; }
        const uint64_t bytes = sink_cache::multiply(count, sizeof(T));
        auto ticket = cache.ledger.client->take(cache.allocation_cost(bytes), 0, ssd::host::requester::outside, false,
            ssd::host::allocation_class::tracked);
        if (!ticket) { throw std::runtime_error(ticket.error); }
        // Value initialization materializes the new working pages. Keep old
        // storage alive until the replacement succeeds; price the full new size.
        std::vector<T> next(count);
        std::copy(values.begin(), values.end(), next.begin());
        values.swap(next);
        ticket.value.commit();
    }
    size_t reserve_ring(const size_t * max_bytes, bool resident) {
        size_t allocated = 0;
        bool needs_resize = false;
        for (const auto & slot : slots) {
            for (int role = 0; role < 3; ++role) {
                allocated += std::max(slot.role[role].size, max_bytes[role]);
                needs_resize |= slot.role[role].size < max_bytes[role];
            }
        }
        // One alignment page per role is separate from the payload limit.
        const size_t alignment_bytes = slots.size() * 3 * 4096;
        if (allocated > 768ull * 1024 * 1024 + alignment_bytes) {
            throw std::runtime_error("CPU sink ring payload exceeds 768 MiB limit");
        }
        // Serialized runs drain previous consumers before reusing these slots.
        if (ring.pending()) { throw std::runtime_error("CPU sink reserve with pending ring I/O"); }
        // Price each full new allocation: reserve() creates it before freeing
        // the old block. Existing capacity is resident and requires no ticket.
        uint64_t growth = 0;
        for (const auto & slot : slots) { for (int role = 0; role < 3; ++role) {
            if (slot.role[role].size < max_bytes[role]) { growth = sink_cache::add(growth, max_bytes[role]); }
        } }
        auto ticket = cache.ledger.client->take(growth, 0, ssd::host::requester::outside, false,
            ssd::host::allocation_class::tracked);
        if (!ticket) { throw std::runtime_error(ticket.error); }
        if (needs_resize) { sink_io_value(ring.unregister_buffers()); ring_unlock(); }
        for (auto & slot : slots) {
            for (int role = 0; role < 3; ++role) {
                const bool new_block = slot.role[role].size < max_bytes[role];
                slot.role[role].reserve(max_bytes[role]);
                if (new_block) { ssd::host::touch_pages(slot.role[role].data, slot.role[role].size, size_t(cache.page_size)); }
                else if (resident) { std::memset(slot.role[role].data, 0, slot.role[role].size); }
            }
        }
        ticket.value.commit();
        return allocated;
    }

    void prepare(const ggml_tensor * up, const ggml_tensor * gate, const ggml_tensor * down,
                 const int32_t * ids, const float * weights, int nslots, bool use_cache = true) {
        const ggml_tensor * tensors[] = { up, gate, down };
        for (const auto * tensor : tensors) {
            if (!tensor || !ggml_is_contiguous(tensor) ||
                tensor->ne[0] <= 0 || tensor->ne[1] <= 0 || tensor->ne[2] <= 0 || tensor->ne[3] != 1 ||
                tensor->ne[0] > 16384 || tensor->ne[1] > 16384) {
                throw std::runtime_error("CPU sink requires contiguous expert metadata with valid dimensions");
            }
            if (!sink_expert_type_supported(tensor->type, tensor->ne[0])) {
                throw std::runtime_error(std::string("this quantization is not supported yet: the CPU sink cannot compute ") +
                    ggml_type_name(tensor->type) + " experts");
            }
        }
        if (up->type != gate->type) {
            throw std::runtime_error("CPU sink requires up and gate experts of one type");
        }
        if (up->ne[0] != gate->ne[0] || up->ne[1] != gate->ne[1] || up->ne[2] != gate->ne[2] ||
            down->ne[0] != up->ne[1] || down->ne[1] != up->ne[0] || down->ne[2] != up->ne[2]) {
            throw std::runtime_error("CPU sink expert shape mismatch");
        }
        latent = static_cast<int>(up->ne[0]); hidden = static_cast<int>(up->ne[1]);
        // The graph's MUL_MAT_ID kernels: vec_dot of the weight type against the
        // input quantized to its vec_dot_type (Q3_K/Q8_K, MXFP4/Q8_0, ...).
        const auto * up_traits = ggml_get_type_traits_cpu(up->type);
        const auto * down_traits = ggml_get_type_traits_cpu(down->type);
        dot_up = up_traits->vec_dot;
        dot_down = down_traits->vec_dot;
        quantize_input = ggml_get_type_traits_cpu(up_traits->vec_dot_type)->from_float;
        quantize_hidden = ggml_get_type_traits_cpu(down_traits->vec_dot_type)->from_float;
        row_bytes_up = ggml_row_size(up->type, latent);
        row_bytes_down = ggml_row_size(down->type, hidden);
        requests.clear();
        size_t max_bytes[3] = {};
        sink_direct_check direct;
        for (int slot = 0; slot < nslots; ++slot) {
            if (ids[slot] < 0) { continue; }
            if (ids[slot] >= up->ne[2] || !std::isfinite(weights[slot]) || weights[slot] < 0.0f) {
                throw std::runtime_error("invalid CPU sink expert id or weight");
            }
            sink_request request = {};
            request.weight = weights[slot];
            request.id = ids[slot];
            for (int role = 0; role < 3; ++role) {
                auto & source = request.role[role];
                if (!resolve(tensors[role]->name, ids[slot], &source)) {
                    throw std::runtime_error("CPU sink pack resolver failed for " + std::string(tensors[role]->name));
                }
                const size_t bytes = ggml_row_size(tensors[role]->type, tensors[role]->ne[0]) * tensors[role]->ne[1];
                if (!direct(source.fd_direct) || source.offset % 4096 || source.nbytes % 4096 ||
                    (source.payload_nbytes && source.payload_nbytes != bytes) || source.data_offset >= 4096 ||
                    source.data_offset > source.nbytes || bytes > source.nbytes - source.data_offset ||
                    source.nbytes > source.data_offset + bytes + 4095 ||
                    source.nbytes > static_cast<size_t>(std::numeric_limits<int>::max()) ||
                    source.offset > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) - source.nbytes) {
                    throw std::runtime_error("CPU sink requires a direct-read fd and aligned, exact padded expert range");
                }
                // A later native tensor can have another page prefix. Reserve
                // that maximum once, while submitting only the actual read.
                const size_t capacity = source.payload_nbytes ?
                    std::max(source.nbytes, (bytes + 8190) / 4096 * 4096) : source.nbytes;
                max_bytes[role] = std::max(max_bytes[role], capacity);
                current.demand_bytes += source.nbytes;
            }
            requests.push_back(request);
        }
        if (use_cache && ready_first) {
            // nslots <= 1024 and latent <= 16384: at most 64 MiB. Allocate
            // before the cache admission check so memory.current includes it.
            resize_scratch(expert_output, requests.size() * size_t(latent));
            resize_scratch(expert_computed, requests.size());
            std::fill(expert_computed.begin(), expert_computed.end(), 0);
            current.ready_scratch_bytes = expert_output.capacity() * sizeof(float) + expert_computed.capacity();
        }
        if (!row_weights.empty()) {
            const double scale = *std::max_element(row_weights.begin(), row_weights.end());
            double total = 0, cumulative = 0;
            for (double w : row_weights) { total += w / scale; }
            up_bounds.resize(threads + 1); down_bounds.resize(threads + 1);
            up_bounds[0] = down_bounds[0] = 0;
            for (int ith = 0; ith < threads; ++ith) {
                cumulative += row_weights[ith] / scale;
                up_bounds[ith + 1] = std::min(hidden, int(hidden * cumulative / total));
                down_bounds[ith + 1] = std::min(latent, int(latent * cumulative / total));
            }
            up_bounds.back() = hidden; down_bounds.back() = latent;
        }
        if (use_cache) {
            if (trace_enabled) {
                // Observe immutable RAM residency before either tier changes
                // its metadata. This does not touch or protect the entry.
                for (auto & r : requests) {
                    for (const auto & e : cache.entries) {
                        if (e.valid && cache.layers[e.layer_index].id == trace_layer &&
                                e.id == r.id && sink_cache::matches(e, r)) {
                            r.trace_ram_resident_before = true;
                            break;
                        }
                    }
                }
            }
            // The tier may size itself in this call (its first decode layer) while the RAM cache has not mapped
            // its slots yet: tell it first what the cache will commit, which on unified memory is the same DRAM.
            if (vram.context && !cache.arena) {
                vram.set_host_demand(cache.pending_bytes(max_bytes[0] + max_bytes[1] + max_bytes[2]));
            }
            if (vram.context) {
                vram.prepare(requests, up, down, latent, hidden, sink_vram::layer_list(cache), cache, post_exp_norm != nullptr);
            } else { vram.hits = 0; }
            // Auto sizing reads memory.current. After GPU prefill, the CPU
            // read ring may still be absent or only virtually allocated. Its
            // first read/registration would otherwise consume the headroom
            // just assigned to the optional cache. Charge this mandatory
            // staging before learning cache geometry, while no cache entry
            // or current I/O job exists. CPU prefill normally touched it already.
            if (cache.active && !cache.paused && !cache.stride &&
                    cache.stats.budget_bytes == sink_cache::auto_budget &&
                    std::any_of(requests.begin(), requests.end(),
                        [](const sink_request & r) { return r.vram_slot < 0; })) {
                const size_t allocated = reserve_ring(max_bytes, true);
                std::fprintf(stderr, "[kimi_cpu_sink_io] auto_cache_ring_resident=1 bytes=%zu\n", allocated);
            }
            if (vram.hits) {
                // GPU hits do not consume RAM cache quota or SSD ring credits.
                std::vector<sink_request> cpu_requests;
                for (const auto & r : requests) { if (r.vram_slot < 0) { cpu_requests.push_back(r); } }
                cache.prepare(cpu_requests, tensors, max_bytes, current);
                size_t j = 0;
                for (auto & r : requests) { if (r.vram_slot < 0) { r = cpu_requests[j++]; } }
                for (size_t i = 0; i < requests.size(); ++i) { expert_computed[i] = requests[i].vram_slot >= 0; }
            } else {
                cache.prepare(requests, tensors, max_bytes, current);
            }
            vram.pin_ram_batch(cache, requests);
        }
        io_requests.clear();
        for (size_t i = 0; i < requests.size(); ++i) {
            if (requests[i].cache_hit >= 0 || requests[i].vram_slot >= 0) { continue; }
            requests[i].io_slot = io_requests.size() % slots.size();
            io_requests.push_back(i);
        }
        current.cache_misses = io_requests.size();
        if (use_cache) {
            cache.stats.hits += current.cache_hits;
            cache.stats.misses += current.cache_misses;
        }
        const size_t allocated = reserve_ring(max_bytes, false);
        for (auto & slot : slots) {
            slot.state = sink_slot::empty;
            slot.ready_roles = 0;
        }
        bool has_ring_reads = false;
        for (size_t index : io_requests) {
            auto & request = requests[index];
            has_ring_reads |= request.cache_admit < 0;
            for (int role = 0; role < 3; ++role) {
                request.read_target[role] = request.cache_admit >= 0 ?
                    cache.data(request.cache_admit, role) : slots[request.io_slot].role[role].data;
            }
        }
        if (ring.read_fixed() && has_ring_reads && !ring.buffers_registered()) {
            std::vector<ssd::io::buffer_span> buffers;
            buffers.reserve(3 * slots.size());
            for (const auto & slot : slots) {
                for (const auto & buffer : slot.role) { buffers.push_back({ buffer.data, buffer.size }); }
            }
            sink_io_value(ring.register_buffers(buffers));
        }
        current.ring_bytes = allocated;
        current.experts = requests.size();
        // a hidden row for each expert a phase may take
        const size_t hidden_floats = std::max<size_t>(1, requests.size()) * size_t(hidden);
        resize_scratch(activation, hidden_floats);
        resize_scratch(activation_up, activation_kind == KIMI_CPU_SINK_ACTIVATION_SWIGLU ? hidden_floats : 0);
        resize_scratch(quant_input, ggml_row_size(up_traits->vec_dot_type, latent));
        quant_hidden_stride = ggml_row_size(down_traits->vec_dot_type, hidden);
        resize_scratch(quant_hidden, quant_hidden_stride);
        if (nonblocking_down_sched) {
            if (quant_hidden_stride && requests.size() > std::numeric_limits<size_t>::max() / quant_hidden_stride) {
                throw std::runtime_error("nonblocking down scheduler scratch overflow");
            }
            resize_scratch(staged_quant_hidden, requests.size() * quant_hidden_stride);
            expert_stage.assign(requests.size(), 0);
            expert_upload_started.assign(requests.size(), 0);
            for (size_t i = 0; i < requests.size(); ++i) {
                if (requests[i].vram_slot >= 0) { expert_stage[i] = 2; }
            }
            current.ready_scratch_bytes = expert_output.capacity() * sizeof(float) + expert_computed.capacity() +
                staged_quant_hidden.capacity() + expert_stage.capacity() + expert_upload_started.capacity();
        }
    }

    // Role-major makes many up/gate pairs ready before their down reads. Treat
    // those halves as independent work: never reserve the caller on one
    // expert's late down CQE while another expert's up/gate can run. Each
    // phase takes every expert ready for it, so a layer whose experts are in
    // memory runs as one up/gate and one down phase, and one whose reads
    // land one by one as it did before, an expert a phase.
    void run_nonblocking_down_schedule() {
        enum work_type { no_work, up_gate_work, down_work, gpu_work };
        const size_t cpu_experts = std::count_if(expert_stage.begin(), expert_stage.end(),
            [](uint8_t stage) { return stage != 2; });
        size_t finished = 0, down_ordinal = 0;
        // Where an expert runs: the layer's plan (plan_decode_layer). A planned expert goes to the GPU once it is whole,
        // with every other whole planned expert in one engine batch; on a discrete GPU the tier copies them first. A
        // refused batch puts the rest of the layer on the CPU.
        bool host_gpu = plan.valid && std::find(plan.gpu.begin(), plan.gpu.end(), uint8_t(1)) != plan.gpu.end();
        auto planned = [&](size_t i) { return host_gpu && i < plan.gpu.size() && plan.gpu[i]; };
        bool moved = false; // host jobs launched in this layer
        std::fill(expert_cpu_us.begin(), expert_cpu_us.end(), 0.0);
        expert_cpu_us.resize(requests.size(), 0.0);
        auto payload = [&](size_t i, int role) {
            const auto & r = requests[i];
            return static_cast<const char *>(r.cache_hit >= 0 ? cache.data(r.cache_hit, role) : r.read_target[role]) +
                r.role[role].data_offset;
        };
        std::vector<size_t> batch;           // the experts of the next phase, in request order
        std::vector<uint8_t> source_whole;   // up/gate work: the expert's three roles are in memory
        while (finished < cpu_experts) {
            work_type work = no_work;
            const auto ready_wait_start = sink_clock::now();
            const uint64_t ready_wait_us = timeline_active ? ggml_time_us() : 0;
            {
                std::unique_lock<std::mutex> lock(io_mutex);
                // The GPU's work first (its launch returns at once), then every expert whose up and gate are in memory,
                // then every expert whose down is. The GPU takes whole experts only: a partial planned one waits for
                // it rather than start here.
                auto choose = [&] {
                    work = no_work;
                    batch.clear();
                    source_whole.clear();
                    size_t gpu = requests.size();
                    for (size_t i = 0; i < requests.size(); ++i) {
                        if (expert_stage[i] != 0) { continue; }
                        const auto & r = requests[i];
                        bool pair = true, whole = true;
                        if (r.cache_hit >= 0) {
                            if (prefetch_hit_state(r.cache_hit) != 1) { continue; }
                        } else {
                            const auto & slot = slots[r.io_slot];
                            if (slot.request != i) { continue; }
                            pair = (slot.ready_roles & 3u) == 3u;
                            whole = (slot.ready_roles & 7u) == 7u;
                        }
                        if (planned(i)) {
                            if (whole && gpu == requests.size()) { gpu = i; }
                        } else if (pair) {
                            batch.push_back(i);
                            source_whole.push_back(whole);
                        }
                    }
                    if (gpu < requests.size()) {
                        batch.assign(1, gpu);
                        work = gpu_work;
                        return;
                    }
                    if (!batch.empty()) { work = up_gate_work; return; }
                    for (size_t i = 0; i < requests.size(); ++i) {
                        if (expert_stage[i] != 1) { continue; }
                        const auto & r = requests[i];
                        if (r.cache_hit >= 0 || (slots[r.io_slot].request == i && (slots[r.io_slot].ready_roles & 4u))) {
                            batch.push_back(i);
                        }
                    }
                    if (!batch.empty()) { work = down_work; }
                };
                // The coordinator does not sleep here: the reads it waits for, its own or a prefetch's, take
                // milliseconds.
                io_poll_until(lock, [&] {
                    if (!io_error.empty()) { return true; }
                    choose();
                    if (work != no_work || prefetch_hit_failed()) { return true; }
                    if (!io_done || prefetch_hits_pending()) { host_lock_in_shadow(); return false; }
                    // a prefetch read may have completed since choose() looked at its hit
                    choose();
                    return true;
                });
                current.wait_io_us += elapsed_us(ready_wait_start);
                if (!io_error.empty()) { break; }
                if (work == no_work && prefetch_hit_failed()) {
                    throw std::runtime_error("CPU sink prefetch read of a used expert failed");
                }
                if (work == no_work) {
                    for (size_t i = 0; i < requests.size(); ++i) {
                        const auto & r = requests[i];
                        std::fprintf(stderr, "[kimi_cpu_sink] stuck expert=%zu id=%d stage=%d cache_hit=%d cache_admit=%d vram_slot=%d"
                            " slot_request=%zu ready_roles=%u prefetch_pending=%d\n", i, r.id, int(expert_stage[i]), r.cache_hit,
                            r.cache_admit, r.vram_slot, slots[r.io_slot].request, unsigned(slots[r.io_slot].ready_roles),
                            r.cache_hit >= 0 ? prefetch_hit_state(r.cache_hit) : -1);
                    }
                    throw std::runtime_error("nonblocking down scheduler has no runnable work");
                }
                if (timeline_active && work == up_gate_work) {
                    for (size_t i : batch) {
                        timeline[i].cpu[0] = ready_wait_us;
                        timeline[i].cpu[1] = ggml_time_us();
                    }
                }
            }

            if (work == gpu_work) {
                const size_t expert = batch.front();
                // Every other whole planned expert goes in the same engine batch.
                std::vector<size_t> gpu_batch = { expert };
                {
                    std::lock_guard<std::mutex> lock(io_mutex);
                    for (size_t i = 0; i < requests.size(); ++i) {
                        if (i == expert || expert_stage[i] != 0 || !planned(i)) { continue; }
                        const auto & r = requests[i];
                        const bool whole = r.cache_hit >= 0 ? prefetch_hit_state(r.cache_hit) == 1 :
                            (slots[r.io_slot].request == i && (slots[r.io_slot].ready_roles & 7u) == 7u);
                        if (whole) { gpu_batch.push_back(i); }
                    }
                }
                std::vector<kimi_vram_host_job> jobs;
                std::vector<std::array<const void *, 3>> payloads;
                for (size_t i : gpu_batch) {
                    const auto & r = requests[i];
                    const std::array<const void *, 3> roles = { payload(i, 0), payload(i, 1), payload(i, 2) };
                    payloads.push_back(roles);
                    // A post-norm model weights each expert after its norm (the reduction below), as the
                    // resident tier requests do: the GPU returns the unweighted output.
                    jobs.push_back({ int32_t(i), post_exp_norm ? 1.0f : r.weight, { roles[0], roles[1], roles[2] } });
                }
                if (vram.launch_host(jobs)) {
                    place.gpu_experts += jobs.size();
                    moved = true;
                    for (size_t k = 0; k < gpu_batch.size(); ++k) {
                        const size_t i = gpu_batch[k];
                        const auto & r = requests[i];
                        // Tier admission as on the CPU path, from the same complete payload, after the launch: a discrete
                        // GPU's job may have taken the expert's reservation, and then this admit has nothing to do.
                        const bool borrow = r.cache_hit >= 0 || r.cache_admit >= 0;
                        if (vram.async_upload && borrow) { vram.admit(i, payloads[k].data(), true); }
                        else { vram.admit(i, payloads[k].data(), borrow); }
                        expert_upload_started[i] = 1;
                        cache.admit(requests[i], current);
                        expert_stage[i] = 2;
                        expert_computed[i] = 1;
                        ++finished;
                    }
                    continue;
                }
                // The tier took nothing for this layer: the CPU computes the rest of it, these experts too.
                ++place.refused;
                host_gpu = false;
                continue;
            }

            const auto compute_start = sink_clock::now();
            phase_items.clear();
            for (size_t k = 0; k < batch.size(); ++k) {
                const size_t i = batch[k];
                const auto & r = requests[i];
                const void * data[3] = { payload(i, 0), payload(i, 1), payload(i, 2) };
                const bool borrow = r.cache_hit >= 0 || r.cache_admit >= 0;
                // Up/gate work may borrow an admission's payload only once its down has landed too; down readiness
                // proves all three roles, exactly as in the blocking path.
                const bool whole = work == up_gate_work ? r.cache_hit >= 0 || (r.cache_admit >= 0 && source_whole[k]) : true;
                if (vram.async_upload && borrow && whole && !expert_upload_started[i]) {
                    vram.admit(i, data, true);
                    expert_upload_started[i] = 1;
                }
                phase_items.push_back(work_item(i, data, r.weight, k, expert_output.data() + i * size_t(latent),
                    staged_quant_hidden.data() + i * quant_hidden_stride));
            }
            if (work == up_gate_work) {
                auto phase_start = sink_clock::now();
                if (timeline_active) { for (size_t i : batch) { timeline[i].cpu[2] = ggml_time_us(); } }
                compute_phase(false);
                if (timeline_active) { for (size_t i : batch) { timeline[i].cpu[3] = ggml_time_us(); } }
                current.up_gate_us += elapsed_us(phase_start);

                phase_start = sink_clock::now();
                const double quant_cpu_start = profile_enabled ? sink_thread_cpu_us() : 0.0;
                if (timeline_active) { for (size_t i : batch) { timeline[i].cpu[4] = ggml_time_us(); } }
                for (const auto & w : phase_items) { quantize_activation(w); }
                if (timeline_active) { for (size_t i : batch) { timeline[i].cpu[5] = ggml_time_us(); } }
                current.hidden_quantize_us += elapsed_us(phase_start);
                if (profile_enabled) { current.compute_cpu_us += sink_thread_cpu_us() - quant_cpu_start; }
                const double up_gate_us = elapsed_us(compute_start);
                current.compute_us += up_gate_us;
                for (size_t i : batch) {
                    expert_cpu_us[i] += up_gate_us / double(batch.size());
                    expert_stage[i] = 1;
                }
                continue;
            }

            const auto phase_start = sink_clock::now();
            if (timeline_active) { for (size_t i : batch) { timeline[i].cpu[6] = ggml_time_us(); } }
            // the last unfinished experts' down projections end the layer's CPU work
            compute_phase(true, finished + batch.size() == cpu_experts);
            if (timeline_active) { for (size_t i : batch) { timeline[i].cpu[7] = ggml_time_us(); } }
            const double down_us = elapsed_us(phase_start);
            current.down_reduce_us += down_us;
            current.compute_us += elapsed_us(compute_start);
            for (size_t i : batch) {
                const auto & request = requests[i];
                // This machine's CPU time per decode expert, the cost the GPU placement compares against: an expert's
                // share of the phases it ran in.
                const double expert_us = expert_cpu_us[i] + down_us / double(batch.size());
                ssd::place::observe_expert(learning, { expert_us, expert_cpu_us[i], io_requests.empty(), moved });
                if (!expert_upload_started[i]) {
                    const void * data[3] = { payload(i, 0), payload(i, 1), payload(i, 2) };
                    vram.admit(i, data, request.cache_hit >= 0 || request.cache_admit >= 0);
                }
                cache.admit(request, current);
                if (request.cache_hit < 0) {
                    std::lock_guard<std::mutex> lock(io_mutex);
                    slots[request.io_slot].state = sink_slot::empty;
                }
                expert_stage[i] = 2;
                expert_computed[i] = 1;
                current.ready_out_of_order_experts += i != down_ordinal;
                ++down_ordinal;
                ++finished;
            }
        }
    }

    // io_mutex must be held. Cache hits were pinned for this whole batch by
    // prepare(). A ready ring slot may belong to a different wrap generation.
    size_t next_ready() const {
        for (size_t i = 0; i < requests.size(); ++i) {
            if (expert_computed[i]) { continue; }
            const auto & r = requests[i];
            const auto & slot = slots[r.io_slot];
            if ((r.cache_hit >= 0 && prefetch_hit_state(r.cache_hit) == 1) ||
                    (slot.state == sink_slot::ready && slot.request == i)) { return i; }
        }
        // Use partial readiness only after all fully ready experts. Otherwise
        // a partial miss could delay an already cached expert behind its down I/O.
        if (early_up_gate) {
            for (size_t i = 0; i < requests.size(); ++i) {
                const auto & slot = slots[requests[i].io_slot];
                if (!expert_computed[i] && slot.state == sink_slot::reading && slot.request == i &&
                        (slot.ready_roles & 3u) == 3u) { return i; }
            }
        }
        return requests.size();
    }

    double wait_down(size_t expert) {
        if (!early_up_gate || requests[expert].cache_hit >= 0) { return 0; }
        const auto start = sink_clock::now();
        std::unique_lock<std::mutex> lock(io_mutex);
        const auto & slot = slots[requests[expert].io_slot];
        io_cv.wait(lock, [&] {
            return (slot.request == expert && (slot.ready_roles & 4u)) || !io_error.empty();
        });
        const double elapsed = elapsed_us(start);
        current.wait_io_us += elapsed;
        // No down worker has started. Error unwinding can drain all remaining
        // reads before either cache or ring storage is released.
        if (!io_error.empty()) { throw std::runtime_error(io_error); }
        return elapsed;
    }
};

// A prefill consumer runs while the stream owns run_mutex, so its phases and
// the role buffers cannot concurrently be used by decode. The phase runs on
// every participant of the open call.
void kimi_cpu_sink_parallel(struct kimi_cpu_sink * sink,
        kimi_cpu_sink_parallel_fn function, void * userdata) {
    sink->parallel_phase(function, userdata, false);
}

// After a stream returned, on the same participants: a run of one parallel phase, the call's last before it
// closes, so its participants then wait for the close as at a node boundary.
void kimi_cpu_sink_parallel_last(struct kimi_cpu_sink * sink,
        kimi_cpu_sink_parallel_fn function, void * userdata) {
    std::lock_guard<sink_run_mutex> serial(sink->run_mutex);
    sink->begin_participants(false);
    sink->parallel_phase(function, userdata, true);
}

void kimi_cpu_sink_host_grow(struct kimi_cpu_sink * sink, uint64_t bytes, const std::function<void()> & grow) {
    if (!bytes) { grow(); return; }
    auto ticket = sink->cache.ledger.client->take(sink->cache.allocation_cost(bytes), 0, ssd::host::requester::outside,
        false, ssd::host::allocation_class::tracked);
    if (!ticket) { throw std::runtime_error(ticket.error); }
    grow();
    ticket.value.commit();
}

bool kimi_cpu_sink_stream_hybrid(struct kimi_cpu_sink * sink,
        const struct ggml_tensor * up, const struct ggml_tensor * gate,
        const struct ggml_tensor * down, const int32_t * ids, int nexperts,
        struct kimi_cpu_sink_prefill_batch * batch,
        kimi_cpu_sink_consumer_fn consumer, void * userdata,
        struct kimi_cpu_sink_stats * stats, char * error, size_t error_size) {
    sink_error(error, error_size, "");
    if (stats) { *stats = {}; }
    if (!sink) { sink_error(error, error_size, "null CPU sink"); return false; }
    std::lock_guard<sink_run_mutex> serial(sink->run_mutex); sink->prefetch_drain();
    // the previous layer's read-ahead of this one stops, and its reads in flight land: the ring is this call's
    const int32_t layer = ssd::host::routed_layer(up);
    sink->lookahead_handover(layer);
    const auto start = sink_clock::now();
    const uint64_t completed_before = sink->ring.completed();
    sink->io_completed_end = completed_before;
    sink->io_pending_end = sink->ring.pending();
    sink->current = {};
    sink->requests.clear();
    sink->begin_observation(up, false);
    try {
        sink->begin_participants(false);
        // The default remains ring-only. CACHE_PREFILL_READ may borrow retained
        // decode entries under explicit leases after the phase headroom check.
        sink->refresh_floor("prefill");
        sink->vram.begin(0, 0, false, sink->cache);
        sink->cache.begin(0, 0, false);
        if (!ids || !consumer || nexperts < 0 || nexperts > 1024 || !batch ||
                !batch->f32_input || !batch->output ||
                (batch->nslots && (!batch->route_request_index || !batch->weights)) || batch->nslots < 0 || batch->nslots > 16 ||
                batch->ntokens < 1 || (batch->nslots &&
                batch->ntokens > std::numeric_limits<int>::max() / batch->nslots)) {
            throw std::runtime_error("invalid CPU expert stream arguments");
        }
        if (sink->poisoned) { throw std::runtime_error("CPU sink ring is poisoned after an I/O failure"); }
        sink->post_exp_norm = batch->activation.post_exp_norm;
        sink->activation_kind = batch->activation.kind;
        sink->beta = batch->activation.situ_beta;
        sink->linear_beta = batch->activation.situ_linear_beta;
        sink->swiglu_limit = batch->activation.swiglu_limit;
        batch->gpu_experts = batch->gpu_routes = 0;
        batch->gpu_transient_experts = batch->gpu_transient_routes = 0;
        batch->gpu_transient_upload_bytes = 0;
        batch->gpu_transient_upload_us = batch->gpu_worker_us = 0.0;
        batch->gpu_expert_us = batch->gpu_read_wait_us = 0.0;
        batch->gpu_ssd_transient_candidates = batch->gpu_ssd_transient_experts = 0;
        batch->gpu_ssd_transient_routes = batch->gpu_ssd_transient_upload_bytes = 0;
        batch->gpu_ssd_transient_upload_us = 0.0;
        batch->ram_experts = batch->ram_routes = batch->ram_bytes = 0;
        batch->ssd_experts = batch->ssd_routes = batch->ssd_bytes = 0;
        std::vector<float> weights(static_cast<size_t>(nexperts), 1.0f);
        sink->prepare(up, gate, down, ids, weights.data(), nexperts, false);
        if (sink->requests.size() != static_cast<size_t>(nexperts)) {
            throw std::runtime_error("CPU expert stream requires nonnegative expert IDs");
        }
        // The next layer's read-ahead plan and its RAM cache slots, taken here: the read-ahead is stopped, and neither
        // this layer's reads nor its GPU work has started.
        sink->lookahead_measure(layer, *batch);
        sink->lookahead_plan_next(layer, up, batch->ntokens, batch->nslots);
        std::vector<size_t> ssd_transient_requests;
        // A GPU route's activation rows over the link: one row of the engine's input type, its token index and weight
        // (set once prepare_prefill has seen the layer's format).
        double route_bytes = 0.0;
        // Set once prepare_prefill has seen this layer's format.
        bool streaming_ssd = false;
        auto streaming_stage = +[](void * context, int32_t request_index, int phase) -> int {
            auto * owner = static_cast<kimi_cpu_sink *>(context);
            if (request_index < 0 || size_t(request_index) >= owner->requests.size()) { return 0; }
            const auto & request = owner->requests[size_t(request_index)];
            auto & slot = owner->slots[request.io_slot];
            std::unique_lock<std::mutex> lock(owner->io_mutex);
            if (phase == 0) {
                owner->io_cv.wait(lock, [&] {
                    return (slot.state == sink_slot::ready && slot.request == size_t(request_index)) ||
                        !owner->io_error.empty() || owner->io_cancel || owner->io_done;
                });
                return slot.state == sink_slot::ready && slot.request == size_t(request_index) &&
                    owner->io_error.empty() && !owner->io_cancel;
            }
            if (phase == 2) {
                return slot.state == sink_slot::ready && slot.request == size_t(request_index) &&
                    owner->io_error.empty() && !owner->io_cancel;
            }
            if (phase != 1 || slot.state != sink_slot::ready ||
                    slot.request != size_t(request_index)) { return 0; }
            slot.state = sink_slot::empty;
            lock.unlock();
            owner->io_cv.notify_all();
            return 1;
        };
        {
            // each request's routes in this pass: the tier seeds the layer's empty slots with the experts routed most
            std::vector<int32_t> routes(sink->requests.size(), 0);
            const size_t route_count = size_t(batch->nslots) * size_t(batch->ntokens);
            for (size_t route = 0; route < route_count; ++route) {
                const int request = batch->route_request_index[route];
                if (request >= 0 && size_t(request) < routes.size()) { ++routes[size_t(request)]; }
            }
            sink->vram.prefill_seed_routes.swap(routes);
        }
        batch->gpu_experts = sink->vram.prepare_prefill(sink->requests, up, down, sink->latent, sink->hidden,
            sink_vram::layer_list(sink->cache), batch->activation.post_exp_norm != nullptr);
        // the batched lane streams its SSD transients through the ring
        streaming_ssd = sink->vram.prefill_batched;
        route_bytes = double(sink->vram.prefill_lane.route_bytes);
        // no read of this layer has started and the tier is idle: a quiescent point to page-lock the ring
        if (streaming_ssd) { sink->ring_lock(); }
        if (batch->gpu_experts) {
            const size_t route_count = size_t(batch->nslots) * size_t(batch->ntokens);
            for (size_t route = 0; route < route_count; ++route) {
                const int request = batch->route_request_index[route];
                if (request >= 0 && size_t(request) < sink->requests.size() &&
                        sink->requests[size_t(request)].vram_slot >= 0) {
                    ++batch->gpu_routes;
                }
            }
        }
        // Source order is strict: retained VRAM, then a read-only leased
        // RAM entry, then a direct read. This lookup does not train or admit the
        // decode cache and an SSD miss never receives a duplicate RAM slot.
        sink->cache.prepare_prefill_read(sink->requests, up, gate, down, sink->current);
        // among the hits, the experts the previous layer's call read ahead
        sink->lookahead_count_used();
        // Streamed transients hold a ring slot and a device slot each (in place, a ring slot alone), whatever their
        // number, so they are not limited.
        const size_t staging_limit = streaming_ssd ? sink->requests.size() : sink->slots.size();
        if (streaming_ssd) {
            std::fprintf(stderr,
                "[kimi_prefill_staging] selected_cap=%zu ring_buffers=%zu streaming=1\n",
                staging_limit, sink->slots.size());
        }
        sink->vram.select_prefill_transient(
            sink->requests, *batch, sink->cache, staging_limit, sink->slots.size(),
            sink->place_table(batch->ntokens, route_bytes));

        size_t ssd_transient_count = 0;
        for (const auto & request : sink->requests) {
            if (!request.vram_ssd_transient) { continue; }
            if (request.vram_ssd_staging < 0) {
                throw std::runtime_error("SSD prefill staging index is absent");
            }
            ssd_transient_count = std::max(
                ssd_transient_count, size_t(request.vram_ssd_staging) + 1);
        }
        if (ssd_transient_count > staging_limit) {
            throw std::runtime_error("SSD prefill staging exceeds the allowed capacity");
        }
        ssd_transient_requests.assign(ssd_transient_count, std::numeric_limits<size_t>::max());
        for (size_t i = 0; i < sink->requests.size(); ++i) {
            auto & request = sink->requests[i];
            if (!request.vram_ssd_transient) { continue; }
            const size_t staging = size_t(request.vram_ssd_staging);
            if (staging >= ssd_transient_requests.size() ||
                    ssd_transient_requests[staging] != std::numeric_limits<size_t>::max()) {
                throw std::runtime_error("invalid duplicate SSD prefill staging index");
            }
            ssd_transient_requests[staging] = i;
            if (!streaming_ssd) { throw std::runtime_error("SSD transient requires a batched streaming lane"); }
            request.vram_ssd_streaming = true;
        }
        if (std::find(ssd_transient_requests.begin(), ssd_transient_requests.end(),
                std::numeric_limits<size_t>::max()) != ssd_transient_requests.end()) {
            throw std::runtime_error("non-contiguous SSD prefill staging indices");
        }

        sink->io_requests.clear();
        auto append_io_request = [&](size_t i) {
            auto & request = sink->requests[i];
            request.io_slot = sink->io_requests.size() % sink->slots.size();
            if (!request.vram_ssd_transient || request.vram_ssd_streaming) {
                for (int role = 0; role < 3; ++role) {
                    request.read_target[role] = sink->slots[request.io_slot].role[role].data;
                }
            }
            sink->io_requests.push_back(i);
        };
        std::vector<size_t> cpu_ssd_requests;
        for (size_t i = 0; i < sink->requests.size(); ++i) {
            const auto & request = sink->requests[i];
            if (request.vram_slot < 0 && request.cache_hit < 0 && !request.vram_transient) {
                cpu_ssd_requests.push_back(i);
            }
        }
        if (streaming_ssd) {
            // Issue the reads in the order the two consumers need them: by each expert's predicted start
            // on its lane (the makespan rule's lane_start_us), so that both lanes run through the whole
            // layer. Each consumer keeps its own order, so a ring credit is always returned by an earlier
            // read of a consumer that is not waiting on a later one.
            size_t gpu_next = 0, cpu_next = 0;
            while (gpu_next < ssd_transient_requests.size() || cpu_next < cpu_ssd_requests.size()) {
                const bool gpu_first = cpu_next == cpu_ssd_requests.size() ||
                    (gpu_next < ssd_transient_requests.size() &&
                     sink->requests[ssd_transient_requests[gpu_next]].lane_start_us <=
                     sink->requests[cpu_ssd_requests[cpu_next]].lane_start_us);
                append_io_request(gpu_first ? ssd_transient_requests[gpu_next++] : cpu_ssd_requests[cpu_next++]);
            }
        } else if (streaming_ssd) {
            // Keep each consumer's request order while sharing the eight
            // ring credits. CPU reads start before the GPU backlog drains.
            size_t gpu_next = 0, cpu_next = 0;
            while (gpu_next < ssd_transient_requests.size() || cpu_next < cpu_ssd_requests.size()) {
                for (int j = 0; j < 4 && gpu_next < ssd_transient_requests.size(); ++j) {
                    append_io_request(ssd_transient_requests[gpu_next++]);
                }
                for (int j = 0; j < 4 && cpu_next < cpu_ssd_requests.size(); ++j) {
                    append_io_request(cpu_ssd_requests[cpu_next++]);
                }
            }
        } else {
            // With no streamed GPU transients, only the CPU reads remain.
            for (size_t i : ssd_transient_requests) { append_io_request(i); }
            for (size_t i : cpu_ssd_requests) { append_io_request(i); }
        }
        if (streaming_ssd) {
            for (size_t i : ssd_transient_requests) {
                const auto & request = sink->requests[i];
                const void * data[3] = {
                    static_cast<const char *>(request.read_target[0]) + request.role[0].data_offset,
                    static_cast<const char *>(request.read_target[1]) + request.role[1].data_offset,
                    static_cast<const char *>(request.read_target[2]) + request.role[2].data_offset,
                };
                sink->vram.bind_prefill_ssd_transient(i, data, streaming_stage, sink);
            }
        }
        sink->current.cache_misses = sink->io_requests.size();
        sink->cache.stats.prefill_misses += sink->current.cache_misses;
        batch->ram_experts = sink->current.cache_hits;
        batch->ram_bytes = sink->current.cache_hit_bytes;
        batch->ssd_experts = sink->io_requests.size();
        const size_t route_count = size_t(batch->nslots) * size_t(batch->ntokens);
        for (size_t route = 0; route < route_count; ++route) {
            const int request = batch->route_request_index[route];
            if (request < 0 || size_t(request) >= sink->requests.size()) { continue; }
            const auto & source = sink->requests[size_t(request)];
            if (source.vram_slot >= 0) { continue; }
            if (source.cache_hit >= 0) { ++batch->ram_routes; }
            else { ++batch->ssd_routes; }
        }
        if (ssd_transient_requests.empty()) {
            sink->vram.launch_prefill(batch->f32_input, batch->f32_stride, batch->route_request_index,
                batch->weights, batch->nslots, batch->ntokens, batch->activation);
        }
        {
            std::lock_guard<std::mutex> lock(sink->io_mutex);
            sink->io_error.clear(); sink->io_cancel = false;
            // with the next layer's plan, if any: the I/O thread reads it once this layer's reads are over
            if (sink->lookahead_publish(layer, !sink->io_requests.empty())) { sink->io_done = false; sink->io_job = true; }
        }
        sink->io_cv.notify_one();
        if (streaming_ssd && !ssd_transient_requests.empty()) {
            sink->vram.launch_prefill(batch->f32_input, batch->f32_stride, batch->route_request_index,
                batch->weights, batch->nslots, batch->ntokens, batch->activation);
        }
        if (!ssd_transient_requests.empty() && !streaming_ssd) {
            // launch_prefill() borrows the staging pointers until finish. Wait
            // for every selected read, then return only the I/O credits; never
            // expose recyclable ring storage to the asynchronous GPU worker.
            for (size_t request_index : ssd_transient_requests) {
                auto & request = sink->requests[request_index];
                sink_slot & slot = sink->slots[request.io_slot];
                const auto wait_start = sink_clock::now();
                {
                    std::unique_lock<std::mutex> lock(sink->io_mutex);
                    sink->io_cv.wait(lock, [&] {
                        return (slot.state == sink_slot::ready && slot.request == request_index) ||
                            !sink->io_error.empty();
                    });
                    sink->current.wait_io_us += elapsed_us(wait_start);
                    if (!sink->io_error.empty()) { throw std::runtime_error(sink->io_error); }
                    slot.state = sink_slot::empty;
                }
                sink->io_cv.notify_one();
            }
            sink->vram.launch_prefill(batch->f32_input, batch->f32_stride, batch->route_request_index,
                batch->weights, batch->nslots, batch->ntokens, batch->activation);
        }
        for (size_t expert = 0; expert < sink->requests.size(); ++expert) {
            auto & request = sink->requests[expert];
            if (request.vram_slot >= 0 || request.vram_transient) { continue; }
            sink_slot * slot = nullptr;
            if (request.cache_hit < 0) {
                slot = &sink->slots[request.io_slot];
                const auto wait_start = sink_clock::now();
                {
                    std::unique_lock<std::mutex> lock(sink->io_mutex);
                    sink->io_cv.wait(lock, [&] {
                        return (slot->state == sink_slot::ready && slot->request == expert) ||
                            !sink->io_error.empty();
                    });
                    sink->current.wait_io_us += elapsed_us(wait_start);
                    if (!sink->io_error.empty()) { break; }
                }
            }
            const auto compute_start = sink_clock::now();
            const double cpu_start = sink->profile_enabled ? sink_thread_cpu_us() : 0.0;
            const void * data[3];
            for (int role = 0; role < 3; ++role) {
                data[role] = static_cast<const char *>(request.cache_hit >= 0 ? sink->cache.data(request.cache_hit, role) : request.read_target[role]) + request.role[role].data_offset;
            }
            // a seed's copy reads the payload beside the compute; it is done before the payload goes back
            sink->vram.seed_host(expert, data);
            try {
                consumer(userdata, expert, data[0], data[1], data[2]);
                sink->vram.seed_wait(expert);
            } catch (...) {
                try { sink->vram.seed_wait(expert); } catch (...) {}
                if (request.cache_lease) {
                    sink->cache.release_lease(request.cache_hit);
                    request.cache_lease = false;
                }
                throw;
            }
            if (request.cache_lease) {
                sink->cache.release_lease(request.cache_hit);
                request.cache_lease = false;
            }
            const double compute_us = elapsed_us(compute_start);
            sink->current.compute_us += compute_us;
            // the grouped path's line, from the passes of the batched engine (a one-token pass is priced as decode)
            sink->vram.observe_prefill_cpu(request.prefill_routes, compute_us);
            if (sink->profile_enabled) { sink->current.compute_cpu_us += sink_thread_cpu_us() - cpu_start; }
            if (slot) {
                std::lock_guard<std::mutex> lock(sink->io_mutex);
                slot->state = sink_slot::empty;
            }
            // Every waiter: with streamed GPU transients the tier's worker waits on io_cv for its read beside the
            // I/O thread waiting for this ring slot, and a single wakeup that reached the worker left both asleep.
            if (slot) { sink->io_cv.notify_all(); }
        }
        {
            const auto wait_start = sink_clock::now();
            std::unique_lock<std::mutex> lock(sink->io_mutex);
            sink->io_cv.wait(lock, [&] { return sink->io_done; });
            sink->current.wait_io_us += elapsed_us(wait_start);
            if (!sink->io_error.empty()) { throw std::runtime_error(sink->io_error); }
            // this call's reads; the next layer's read-ahead may be on the ring now
            sink->current.read_completions = sink->io_completed_end - completed_before;
            sink->current.pending_reads = sink->io_pending_end;
        }
        if (batch->gpu_experts || !sink->vram.prefill_transient.empty()) {
            sink->vram.finish_prefill(batch->output, size_t(sink->latent), *batch);
        }
        sink->vram.calibrate_prefill(sink->current.compute_us, *batch, sink->current.read_bytes,
            sink->current.io_us - sink->current.io_credit_wait_us, elapsed_us(start),
            [&] { return sink->place_table(batch->ntokens, route_bytes); });
        if (!sink->requests.empty()) {
            const auto & r = sink->requests.front();
            sink->vram.set_host_demand(sink->cache.pending_bytes(r.role[0].nbytes + r.role[1].nbytes + r.role[2].nbytes));
        }
        // CPU-consumed RAM leases were released immediately. Transient GPU
        // sources remain leased until finish drained H2D and compute.
        sink->cache.release_prefill_leases(sink->requests);
        batch->ssd_bytes = sink->current.read_bytes;
        std::fprintf(stderr,
            "[kimi_cpu_prefill_sources] tensor=%s vram_experts=%llu vram_routes=%llu "
            "ram_gpu_experts=%llu ram_gpu_routes=%llu ram_gpu_upload_bytes=%llu ram_gpu_upload_us=%.3f "
            "ssd_gpu_candidates=%llu ssd_gpu_experts=%llu ssd_gpu_routes=%llu "
            "ssd_gpu_upload_bytes=%llu ssd_gpu_upload_us=%.3f gpu_worker_us=%.3f "
            "gpu_expert_us=%.3f gpu_read_wait_us=%.3f gpu_dma_us=%.3f gpu_upload_wait_us=%.3f ring_locked=%d "
            "ram_experts=%llu ram_routes=%llu ram_bytes=%llu "
            "ssd_experts=%llu ssd_routes=%llu ssd_bytes=%llu ssd_read_us=%.1f\n",
            up->name, (unsigned long long) batch->gpu_experts,
            (unsigned long long) batch->gpu_routes,
            (unsigned long long) batch->gpu_transient_experts,
            (unsigned long long) batch->gpu_transient_routes,
            (unsigned long long) batch->gpu_transient_upload_bytes,
            batch->gpu_transient_upload_us,
            (unsigned long long) batch->gpu_ssd_transient_candidates,
            (unsigned long long) batch->gpu_ssd_transient_experts,
            (unsigned long long) batch->gpu_ssd_transient_routes,
            (unsigned long long) batch->gpu_ssd_transient_upload_bytes,
            batch->gpu_ssd_transient_upload_us,batch->gpu_worker_us,
            batch->gpu_expert_us,batch->gpu_read_wait_us,sink->vram.prefill_stats.transient_dma_us,
            sink->vram.prefill_stats.transient_wait_us,int(sink->vram.ring_locked),
            (unsigned long long) batch->ram_experts,
            (unsigned long long) batch->ram_routes,
            (unsigned long long) batch->ram_bytes, (unsigned long long) batch->ssd_experts,
            (unsigned long long) batch->ssd_routes,
            (unsigned long long) batch->ssd_bytes, sink->current.io_us - sink->current.io_credit_wait_us);
        sink->lookahead_print();
        if (sink->timeline_active) { sink->timeline_call[5] = ggml_time_us(); }
        sink->current.total_us = elapsed_us(start);
        sink->finish_observation();
        if (stats) { *stats = sink->current; }
        sink->print_observation(true);
        const auto published = sink->publish_observation(true);
        if (!published) { sink_error(error, error_size, published.error.c_str()); return false; }
        return true;
    } catch (const std::exception & e) {
        // Stop new submissions and drain before returning, including an
        // unexpected exception in a consumer. No user buffer may outlive I/O.
        sink->cancel_and_wait();
        sink->vram.cancel_prefill();
        // the read-ahead stops and the pass ends
        sink->lookahead_quiesce();
        sink->cache.release_prefill_leases(sink->requests);
        sink->cache.release();
        if (sink->timeline_active) { sink->timeline_call[5] = ggml_time_us(); }
        sink->current.total_us = elapsed_us(start);
        {
            // as the I/O thread left them after this call's reads (a published job's are over: cancel_and_wait)
            std::lock_guard<std::mutex> lock(sink->io_mutex);
            sink->current.read_completions = sink->io_completed_end - completed_before;
            sink->current.pending_reads = std::max(sink->io_pending_end, sink->ring.pending());
        }
        sink->finish_observation();
        if (stats) { *stats = sink->current; }
        sink->print_observation(false);
        const auto published = sink->publish_observation(false);
        if (!published && error && error_size) {
            std::snprintf(error, error_size, "%s; telemetry: %s", e.what(), published.error.c_str());
        } else { sink_error(error, error_size, e.what()); }
        return false;
    }
}

extern "C" struct kimi_cpu_sink * kimi_cpu_sink_create(
        int ring_experts, kimi_cpu_sink_resolver resolver, void * userdata, char * error, size_t error_size) {
    sink_error(error, error_size, "");
    try {
        if (ring_experts < 1 || ring_experts > 32 || !resolver) {
            throw std::runtime_error("CPU sink requires 1..32 ring experts and a resolver");
        }
        return new kimi_cpu_sink(ring_experts, resolver, userdata);
    } catch (const std::exception & e) {
        sink_error(error, error_size, e.what());
        return nullptr;
    }
}

extern "C" bool kimi_cpu_sink_call_open(struct kimi_cpu_sink * sink, int nth, char * error, size_t error_size) {
    sink_error(error, error_size, "");
    if (!sink) { sink_error(error, error_size, "null CPU sink"); return false; }
    try {
        std::string why;
        if (sink->open_call(nth, why)) { return true; }
        sink_error(error, error_size, why.c_str());
    } catch (const std::exception & e) { sink_error(error, error_size, e.what()); }
    return false;
}

extern "C" void kimi_cpu_sink_call_close(struct kimi_cpu_sink * sink) {
    if (sink) { sink->close_call(); }
}

extern "C" void kimi_cpu_sink_call_join(struct kimi_cpu_sink * sink, int ith, int nth) {
    if (sink) { sink->join_call(ith, nth); }
}

extern "C" int kimi_cpu_sink_decode_threads(struct kimi_cpu_sink * sink, int pool) {
    if (!sink) { return pool; }
    std::lock_guard<sink_run_mutex> serial(sink->run_mutex);
    return sink->decode_threads(pool);
}

namespace ssd {
namespace cpu {
namespace {
sink_configuration configured(const settings::snapshot * input, const runtime_services * services) {
    if (!input) { throw std::runtime_error("missing typed SSD settings"); }
    sink_configuration out;
    out.production = true;
    out.policy.poll_us = sink_production_poll_us;
    if (!services || !services->host) { throw std::runtime_error("missing process host budget client"); }
    out.settings = *input;
    sink_copy_services(out, services);
    auto integer = [&](const char * key) {
        const auto * value = out.settings.integer(key);
        if (!value) { throw std::runtime_error(std::string("missing typed ") + key); }
        return *value;
    };
    const uint64_t participants = integer("cpu.decode_participants");
    const uint64_t prefetch = integer("prefetch.depth");
    if (participants > uint64_t(sink_max_participants) || prefetch > 64 || !out.settings.cpus("cpu.io_cpus") ||
            !out.settings.vram_budget() || !out.settings.costs() || !out.settings.sources() ||
            integer("host.headroom_mib") > UINT32_MAX || integer("vram.reserve_mib") > 8192) {
        throw std::runtime_error("invalid typed SSD executor dependencies");
    }
    if (!out.has_gpu && out.settings.vram_budget()->tier_bytes(false)) {
        throw std::runtime_error("nonzero GPU tier budget requires an injected GPU provider");
    }
    // A setting of a GPU backend's expert tier would do nothing without the tier: one set explicitly is refused.
    const bool tier = out.has_gpu && out.gpu.residency;
    for (const auto & d : settings::registry()) {
        const auto * value = out.settings.find(d.key);
        if (*d.backend && !tier && value && value->origin != "default") {
            throw std::runtime_error(std::string(d.key) + " (" + value->location + ") configures the " + d.backend +
                                     " expert tier, which this engine runs without");
        }
    }
    auto & p = out.policy;
    p.ready_first = p.early_up_gate = p.role_major = p.nonblocking_down_sched = p.dynamic_rows = true;
    p.dynamic_chunk_rows = 64;
    p.decode_participants = int(participants);
    p.prefetch_depth = int(prefetch);
    p.prefill_lookahead = integer("prefill.lookahead") != 0;
    for (uint32_t cpu : *out.settings.cpus("cpu.io_cpus")) { p.io_cpus.push_back(int(cpu)); }
    p.cache_policy = "count_decay2";
    p.cache_allocation = "uniform";
    p.cache_keep = p.cache_prefill_read = p.cache_partial_reclaim = p.cache_reclaimable_file = true;
    p.cache_reserve_bytes = 512ull << 20;
    return out;
}
}  // namespace

kimi_cpu_sink * create_configured(const settings::snapshot * input, const runtime_services * services,
        kimi_cpu_sink_resolver resolver, void * userdata, char * error, size_t error_size) noexcept {
    sink_error(error, error_size, "");
    try {
        const auto value = configured(input, services);
        if (!resolver) { throw std::runtime_error("missing SSD expert resolver"); }
        return new kimi_cpu_sink(ssd::io::expert_ring_depth, resolver, userdata, value);
    } catch (const std::exception & e) {
        sink_error(error, error_size, e.what());
        return nullptr;
    }
}

namespace testing {
kimi_cpu_sink * create(int ring_experts, const execution_policy & policy,
        const runtime_services * services, kimi_cpu_sink_resolver resolver, void * userdata,
        char * error, size_t error_size) noexcept {
    sink_error(error, error_size, "");
    try {
        if (ring_experts < 1 || ring_experts > 32 || !resolver) {
            throw std::runtime_error("invalid explicit CPU test executor dimensions");
        }
        sink_configuration value;
        value.policy = policy;
        sink_copy_services(value, services);
        if (policy.vram_budget && !value.has_gpu) { throw std::runtime_error("explicit GPU fixture requires an injected provider"); }
        return new kimi_cpu_sink(ring_experts, resolver, userdata, value);
    } catch (const std::exception & e) {
        sink_error(error, error_size, e.what());
        return nullptr;
    }
}
}  // namespace testing
}  // namespace cpu
}  // namespace ssd

const ssd::telemetry::debug_channels & kimi_cpu_sink_debug(const kimi_cpu_sink * sink) noexcept {
    static const ssd::telemetry::debug_channels off;
    return sink ? sink->configuration.debug : off;
}

extern "C" void kimi_cpu_sink_destroy(struct kimi_cpu_sink * sink) { delete sink; }

extern "C" void kimi_cpu_sink_prefetch_hint(struct kimi_cpu_sink * sink, int32_t layer, const int32_t * experts, int n) {
    if (!sink || !experts || n <= 0) { return; }
    std::lock_guard<sink_run_mutex> serial(sink->run_mutex);
    sink->prefetch.hint_n = std::min(n, 64);
    std::copy(experts, experts + sink->prefetch.hint_n, sink->prefetch.hint);
    sink->prefetch.hint_layer = layer;
}

extern "C" void kimi_cpu_sink_prefill_hint(struct kimi_cpu_sink * sink, int32_t layer, const int32_t * experts,
        int ntokens, int k) {
    if (!sink || !experts || layer < 0 || ntokens < 1 || k < 1 || k > 16 ||
            ntokens > std::numeric_limits<int>::max() / k) { return; }
    std::lock_guard<sink_run_mutex> serial(sink->run_mutex);
    if (!sink->lookahead.enabled) { return; }
    auto & la = sink->lookahead;
    la.hint.assign(experts, experts + size_t(ntokens) * size_t(k));
    la.hint_layer = layer; la.hint_tokens = ntokens; la.hint_k = k;
}

extern "C" void kimi_cpu_sink_cache_reset(struct kimi_cpu_sink * sink) {
    if (!sink) { return; }
    std::lock_guard<sink_run_mutex> serial(sink->run_mutex); sink->prefetch_drain();
    sink->lookahead_quiesce();
    sink->vram.cancel();
    sink->cache.release();
    sink->cache.active = sink->cache.has_token = sink->cache.paused = false;
}

extern "C" bool kimi_cpu_sink_cache_configure(struct kimi_cpu_sink * sink, uint64_t budget,
        const int32_t * priority, int n, int quota, char * error, size_t error_size) {
    sink_error(error, error_size, "");
    if (!sink) { sink_error(error, error_size, "null CPU sink"); return false; }
    std::lock_guard<sink_run_mutex> serial(sink->run_mutex); sink->prefetch_drain();
    try { sink->vram.begin(0, 0, false, sink->cache); sink->cache.configure(budget, priority, n, quota); return true; }
    catch (const std::exception & e) { sink_error(error, error_size, e.what()); return false; }
}

// kimi_cpu_sink_cache_begin, and with `request` kimi_cpu_sink_request_begin.
static bool sink_phase_begin(struct kimi_cpu_sink * sink, uint64_t epoch, uint64_t token, bool decode, bool request,
        char * error, size_t error_size) {
    sink_error(error, error_size, "");
    if (!sink) { sink_error(error, error_size, "null CPU sink"); return false; }
    std::lock_guard<sink_run_mutex> serial(sink->run_mutex); sink->prefetch_drain();
    try {
        sink->end_margin_interval(decode, epoch, token);
        sink->measure_decode_share(decode, epoch, token);
        // A decode token ends the prompt pass: what the read-ahead read is published before the GPU tier and the RAM
        // cache change phase. Every grouped-prefill call begins a prefill phase too, which keeps the read-ahead the call
        // before it started (the cache neither discards nor unmaps a slot a read is landing in: inflight, before_unmap).
        if (decode) { sink->lookahead_quiesce(); }
        sink->refresh_floor(decode ? "decode" : "prefill");
        sink->file_identities.clear();
        sink->vram.begin(epoch, token, decode, sink->cache, request);
        sink->cache.begin(epoch, token, decode);
        if (decode) { sink->vram.drop_held(sink->cache); }
        sink->begin_margin_interval(decode);
    } catch (const std::exception & e) {
        sink->vram.cancel(); sink->cache.release();
        sink_error(error, error_size, e.what()); return false;
    }
    if (!decode) { sink->mrc.report("request"); }
    sink->trace_epoch = epoch;
    sink->trace_token = token;
    sink->trace_has_token = decode;
    sink->trace_decode = decode;
    if (sink->trace_enabled && !decode) {
        std::fprintf(stderr, "[kimi_cpu_sink_trace] {\"schema\":1,\"event\":\"reset\",\"phase\":\"prefill\","
            "\"epoch\":%llu,\"token\":%llu}\n", (unsigned long long) epoch, (unsigned long long) token);
    }
    return true;
}

extern "C" bool kimi_cpu_sink_cache_begin(struct kimi_cpu_sink * sink, uint64_t epoch,
        uint64_t token, bool decode, char * error, size_t error_size) {
    return sink_phase_begin(sink, epoch, token, decode, false, error, error_size);
}

extern "C" bool kimi_cpu_sink_request_begin(struct kimi_cpu_sink * sink, uint64_t epoch, char * error, size_t error_size) {
    return sink_phase_begin(sink, epoch, 0, false, true, error, error_size);
}

extern "C" bool kimi_cpu_sink_vram_phase_complete(
        struct kimi_cpu_sink * sink, bool decode, char * error, size_t error_size) {
    sink_error(error, error_size, "");
    if (!sink) { sink_error(error, error_size, "null CPU sink"); return false; }
    std::lock_guard<sink_run_mutex> serial(sink->run_mutex); sink->prefetch_drain();
    try {
        sink->vram.phase_complete(decode);
        return true;
    } catch (const std::exception & e) {
        sink_error(error, error_size, e.what());
        return false;
    }
}

extern "C" bool kimi_cpu_sink_vram_reserve_accumulator(
        struct kimi_cpu_sink * sink, int tokens, char * error, size_t error_size) {
    sink_error(error, error_size, "");
    if (!sink) { sink_error(error, error_size, "null CPU sink"); return false; }
    std::lock_guard<sink_run_mutex> serial(sink->run_mutex); sink->prefetch_drain();
    try { sink->vram.reserve_accumulate(tokens); return true; }
    catch (const std::exception & e) { sink_error(error, error_size, e.what()); return false; }
}

extern "C" bool kimi_cpu_sink_vram_reserve(struct kimi_cpu_sink * sink, const struct ggml_tensor * const * experts,
        int n_layers, int experts_used, char * error, size_t error_size) {
    sink_error(error, error_size, "");
    if (!sink || !experts || n_layers < 1 ||
            std::any_of(experts, experts + 3 * n_layers, [](const ggml_tensor * t) { return !t; })) {
        sink_error(error, error_size, "null CPU sink or expert tensor"); return false;
    }
    std::lock_guard<sink_run_mutex> serial(sink->run_mutex); sink->prefetch_drain();
    sink->lookahead_quiesce(); // the ring's reserve needs it idle
    try {
        sink->reserve_tier(experts, n_layers, experts_used);
        return true;
    }
    catch (const std::exception & e) { sink_error(error, error_size, e.what()); return false; }
}

extern "C" void kimi_cpu_sink_cache_get_stats(struct kimi_cpu_sink * sink, kimi_cpu_sink_cache_stats * stats) {
    if (!stats) { return; }
    *stats = {};
    if (!sink) { return; }
    std::lock_guard<sink_run_mutex> serial(sink->run_mutex);
    *stats = sink->cache.stats;
}

// The unconfigured C compatibility API has no L3 registration to borrow.
// A null sink is therefore a refusal, never a new independent production ledger.
extern "C" bool kimi_cpu_sink_cache_make_room(struct kimi_cpu_sink * sink, uint64_t host_bytes, uint64_t device_bytes) {
    if (!sink) { return false; }
    std::lock_guard<sink_run_mutex> serial(sink->run_mutex); sink->prefetch_drain();
    try { return sink->make_room(host_bytes, device_bytes); }
    catch (const std::exception & e) {
        std::fprintf(stderr, "[kimi_cpu_sink_cache] make_room failed: %s\n", e.what());
        return false;
    }
}

extern "C" bool kimi_cpu_sink_cache_prefill_room(struct kimi_cpu_sink * sink, uint64_t pages, uint64_t host_bytes,
        uint64_t device_bytes) {
    if (!sink) { return false; }
    std::lock_guard<sink_run_mutex> serial(sink->run_mutex); sink->prefetch_drain();
    try { return sink->prefill_room(pages, host_bytes, device_bytes); }
    catch (const std::exception & e) {
        std::fprintf(stderr, "[kimi_cpu_sink_cache] prefill_room failed: %s\n", e.what());
        return false;
    }
}

static bool sink_activation_valid(const kimi_cpu_sink_activation * activation) {
    if (!activation) { return false; }
    switch (activation->kind) {
        case KIMI_CPU_SINK_ACTIVATION_SITU:
            return std::isfinite(activation->situ_beta) && activation->situ_beta > 0.0f &&
                std::isfinite(activation->situ_linear_beta);
        case KIMI_CPU_SINK_ACTIVATION_SWIGLU_CLAMP:
            return !std::isnan(activation->swiglu_limit) && activation->swiglu_limit > 1e-6f;
        case KIMI_CPU_SINK_ACTIVATION_SWIGLU:
            return true;
        default:
            return false;
    }
}

namespace ssd {
namespace cpu {
result<host::allocation_ticket> take_room_configured(kimi_cpu_sink * sink, const settings::snapshot * input,
        const runtime_services * services, uint64_t host_bytes, uint64_t device_bytes, uint64_t pages) noexcept {
    if (!input || !services || !services->host) { return sink_host_failure(status::invalid, "missing host client or settings"); }
    try {
        const auto value = configured(input, services);
        if (sink) {
            if (sink->cache.ledger.client != value.host) {
                return {{}, status::invalid, "host client does not own executor"};
            }
            std::lock_guard<sink_run_mutex> serial(sink->run_mutex); sink->prefetch_drain();
            if (pages != UINT64_MAX) { sink->cache.ledger.set_prefill_pages(pages); }
            sink->refresh_floor("take_room");
            return value.host->take(host_bytes, device_bytes);
        }
        // Early graph reserve and a later sink configure the same registration
        // with observed device topology and the identical immutable settings.
        auto configured_host = value.host->configure(sink_host_configuration(value));
        if (!configured_host) { return {{}, configured_host.code, configured_host.error}; }
        if (pages != UINT64_MAX) {
            auto held = value.host->set_prefill_pages(pages);
            if (!held) { return {{}, held.code, held.error}; }
        }
        auto refresh = value.host->update_file_reserve();
        if (!refresh) { return {{}, refresh.code, refresh.error}; }
        return value.host->take(host_bytes, device_bytes);
    } catch (const std::bad_alloc &) { return sink_host_failure(status::resource, "host admission allocation failed"); }
      catch (const std::exception & e) {
          // A refused setting is thrown from configured(): its text is the reason.
          try { return sink_host_failure(status::io, (std::string("host admission failed: ") + e.what()).c_str()); }
          catch (...) { return sink_host_failure(status::io, "host admission failed"); }
      }
      catch (...) { return sink_host_failure(status::io, "host admission failed"); }
}
bool cache_make_room_configured(kimi_cpu_sink * sink, const settings::snapshot * input,
        const runtime_services * services, uint64_t host_bytes, uint64_t device_bytes) noexcept {
    auto ticket = take_room_configured(sink, input, services, host_bytes, device_bytes, UINT64_MAX);
    // The caller sees only false: name why, a refused setting included.
    if (!ticket) { std::fprintf(stderr, "[kimi_cpu_sink_cache] make_room failed: %s\n", ticket.error.c_str()); }
    return bool(ticket); // Compatibility observation only: destructor cancels the unmaterialized ticket.
}
bool cache_prefill_room_configured(kimi_cpu_sink * sink, const settings::snapshot * input,
        const runtime_services * services, uint64_t pages, uint64_t host_bytes, uint64_t device_bytes) noexcept {
    auto ticket = take_room_configured(sink, input, services, host_bytes, device_bytes, pages);
    if (!ticket) { std::fprintf(stderr, "[kimi_cpu_sink_cache] prefill_room failed: %s\n", ticket.error.c_str()); }
    return bool(ticket); // Production allocations use take_room_configured and retain the ticket.
}
}  // namespace cpu
}  // namespace ssd

extern "C" bool kimi_cpu_sink_run_activation(
        struct kimi_cpu_sink * sink, const struct ggml_tensor * up, const struct ggml_tensor * gate,
        const struct ggml_tensor * down, const float * input, const int32_t * ids, const float * weights,
        int nslots, const struct kimi_cpu_sink_activation * activation, float * output,
        struct kimi_cpu_sink_stats * stats, char * error, size_t error_size) {
    sink_error(error, error_size, "");
    if (stats) { *stats = {}; }
    if (!sink) { sink_error(error, error_size, "null CPU sink"); return false; }
    std::lock_guard<sink_run_mutex> serial(sink->run_mutex);
    sink->lookahead_quiesce(); // decode is no grouped prefill: the ring is the run's alone
    sink->cache.return_borrowed(-1); // and its layers keep their own slots
    const auto start = sink_clock::now();
    const uint64_t completed_before = sink->ring.completed();
    sink->current = {};
    sink->requests.clear();
    sink->begin_observation(up, true);
    try {
        sink->begin_participants(true);
        if (!input || !ids || !weights || !output || nslots < 0 || nslots > 1024 ||
            !sink_activation_valid(activation)) {
            throw std::runtime_error(activation && activation->kind == KIMI_CPU_SINK_ACTIVATION_SITU ?
                "invalid CPU sink input or SiTU parameters" : "invalid CPU sink input or activation parameters");
        }
        if (sink->poisoned) { throw std::runtime_error("CPU sink ring is poisoned after an I/O failure"); }
        {
            // with the ids checked: an in-flight prefetch of an expert this run uses becomes a hit it waits for
            const int layer = ssd::host::routed_layer(up);
            if (layer >= 0) {
                sink->prefetch_drain(layer, ids, nslots);
                sink->mrc.observe(layer, ids, nslots);
            } else {
                sink->prefetch_drain();
            }
        }
        sink->post_exp_norm = activation->post_exp_norm;
        if (sink->post_exp_norm && (!sink->ready_first || !std::isfinite(activation->post_exp_norm_eps) ||
                activation->post_exp_norm_eps <= 0.0f)) {
            throw std::runtime_error("per-expert post norm requires READY_FIRST=1 and positive finite epsilon");
        }
        sink->activation_kind = activation->kind;
        sink->beta = activation->situ_beta; sink->linear_beta = activation->situ_linear_beta;
        sink->swiglu_limit = activation->swiglu_limit;
        sink->prepare(up, gate, down, ids, weights, nslots);
        sink->prefetch_keep_used_hits();
        sink->prefetch_count_used();
        if (sink->timeline_active) {
            sink->timeline.resize(sink->requests.size());
            for (auto & t : sink->timeline) { t.workers.resize(2 * size_t(sink->threads)); }
            sink->timeline_call[1] = ggml_time_us();
        }
        sink->output = output;
        sink->plan.start = sink_clock::now();
        sink->io_begin_decode();
        auto compute_start = sink_clock::now();
        double quant_cpu_start = sink->profile_enabled ? sink_thread_cpu_us() : 0.0;
        sink->quantize_input(input, sink->quant_input.data(), sink->latent);
        std::fill(output, output + sink->latent, 0.0f);
        sink->current.input_quantize_us = elapsed_us(compute_start);
        if (sink->profile_enabled) { sink->current.compute_cpu_us += sink_thread_cpu_us() - quant_cpu_start; }
        sink->current.compute_us += sink->current.input_quantize_us;
        sink->vram.launch(input, size_t(sink->latent) * sizeof(float), *activation);
        if (sink->timeline_active) { sink->timeline_call[2] = ggml_time_us(); }
        sink->prefetch_schedule(up, !sink->io_requests.empty());
        // The plan reads nothing the reads change (its landing times start at plan.start): planned once they run.
        sink->plan_decode_layer();
        if (sink->nonblocking_down_sched) {
            sink->run_nonblocking_down_schedule();
        }
        const size_t blocking_experts = sink->nonblocking_down_sched ? 0 :
            sink->requests.size() - size_t(sink->vram.hits);
        for (size_t ordinal = 0; ordinal < blocking_experts; ++ordinal) {
            size_t expert = ordinal;
            const auto ready_wait_start = sink->ready_first ? sink_clock::now() : sink_clock::time_point();
            const uint64_t ready_wait_us = sink->timeline_active ? ggml_time_us() : 0;
            if (sink->ready_first) {
                std::unique_lock<std::mutex> lock(sink->io_mutex);
                sink->io_cv.wait(lock, [&] {
                    expert = sink->next_ready();
                    return expert < sink->requests.size() || !sink->io_error.empty();
                });
                sink->current.wait_io_us += elapsed_us(ready_wait_start);
                if (!sink->io_error.empty()) { break; }
                sink->expert_computed[expert] = 1;
                sink->current.ready_out_of_order_experts += expert != ordinal;
                sink->output = sink->expert_output.data() + expert * size_t(sink->latent);
                if (sink->timeline_active) {
                    sink->timeline[expert].cpu[0] = ready_wait_us;
                    sink->timeline[expert].cpu[1] = ggml_time_us();
                }
            }
            const auto & request = sink->requests[expert];
            sink_slot & slot = sink->slots[request.io_slot];
            const auto wait_start = sink_clock::now();
            if (request.cache_hit < 0 && !sink->ready_first) {
                if (sink->timeline_active) { sink->timeline[expert].cpu[0] = ggml_time_us(); }
                std::unique_lock<std::mutex> lock(sink->io_mutex);
                sink->io_cv.wait(lock, [&] {
                    return slot.state == sink_slot::ready ||
                        (sink->early_up_gate && slot.request == expert && (slot.ready_roles & 3u) == 3u) ||
                        !sink->io_error.empty();
                });
                sink->current.wait_io_us += elapsed_us(wait_start);
                if (!sink->io_error.empty()) { break; }
                if (sink->timeline_active) { sink->timeline[expert].cpu[1] = ggml_time_us(); }
            }
            compute_start = sink_clock::now();
            const void * data[3];
            for (int role = 0; role < 3; ++role) {
                data[role] = static_cast<const char *>(request.cache_hit >= 0 ? sink->cache.data(request.cache_hit, role) : request.read_target[role]) + request.role[role].data_offset;
            }
            const bool borrow_upload = request.cache_hit >= 0 || request.cache_admit >= 0;
            // A cache hit has all three roles. A cache admission is also an I/O
            // target, and early-up-gate may expose it while down is still in
            // flight. Do not let an asynchronous upload borrow that incomplete
            // payload; wait_down() supplies the next safe admission point.
            bool upload_source_complete = request.cache_hit >= 0;
            if (request.cache_admit >= 0) {
                std::lock_guard<std::mutex> lock(sink->io_mutex);
                upload_source_complete = slot.request == expert && (slot.ready_roles & 4u);
            }
            bool upload_started = sink->vram.async_upload && borrow_upload && upload_source_complete;
            if (upload_started) { sink->vram.admit(expert, data, true); }
            sink->phase_items.assign(1, sink->work_item(expert, data, request.weight, 0, sink->output, sink->quant_hidden.data()));
            double down_wait_us = 0;
            auto phase_start = sink_clock::now();
            if (sink->timeline_active) { sink->timeline[expert].cpu[2] = ggml_time_us(); }
            sink->compute_phase(false);
            if (sink->timeline_active) { sink->timeline[expert].cpu[3] = ggml_time_us(); }
            sink->current.up_gate_us += elapsed_us(phase_start);
            phase_start = sink_clock::now();
            quant_cpu_start = sink->profile_enabled ? sink_thread_cpu_us() : 0.0;
            if (sink->timeline_active) { sink->timeline[expert].cpu[4] = ggml_time_us(); }
            sink->quantize_activation(sink->phase_items.front());
            if (sink->timeline_active) { sink->timeline[expert].cpu[5] = ggml_time_us(); }
            sink->current.hidden_quantize_us += elapsed_us(phase_start);
            if (sink->profile_enabled) { sink->current.compute_cpu_us += sink_thread_cpu_us() - quant_cpu_start; }
            down_wait_us = sink->wait_down(expert);
            if (!upload_started && sink->vram.async_upload && borrow_upload) {
                // wait_down() returned only after a successful down CQE, so
                // the immutable three-role cache payload can now be borrowed
                // while CPU down computation reads it too.
                sink->vram.admit(expert, data, true);
                upload_started = true;
            }
            phase_start = sink_clock::now();
            if (sink->timeline_active) { sink->timeline[expert].cpu[6] = ggml_time_us(); }
            sink->first_down = ordinal == 0;
            sink->compute_phase(true, ordinal + 1 == blocking_experts);
            if (sink->timeline_active) { sink->timeline[expert].cpu[7] = ggml_time_us(); }
            sink->current.down_reduce_us += elapsed_us(phase_start);
            sink->current.compute_us += elapsed_us(compute_start) - down_wait_us;
            if (!upload_started) { sink->vram.admit(expert, data, borrow_upload); }
            sink->cache.admit(request, sink->current);
            if (request.cache_hit < 0) {
                std::lock_guard<std::mutex> lock(sink->io_mutex);
                slot.state = sink_slot::empty;
            }
            sink->io_cv.notify_one();
        }
        {
            const auto wait_start = sink_clock::now();
            std::unique_lock<std::mutex> lock(sink->io_mutex);
            sink->io_wait(lock, [&] { return sink->io_done; });
            sink->current.wait_io_us += elapsed_us(wait_start);
            if (!sink->io_error.empty()) { throw std::runtime_error(sink->io_error); }
        }
        ssd::place::observe_read_rate(sink->learning, sink->current.read_bytes, sink->current.io_us);
        sink->vram.finish(sink->expert_output.data(), size_t(sink->latent), sink->cache, sink->requests);
        const double lanes_us = elapsed_us(sink->plan.start);
        sink->report_miss_cost();
        sink->vram.set_host_demand(sink->cache.pending_bytes(sink->cache.stride));
        if (sink->ready_first) {
            const auto reduce_start = sink_clock::now();
            const double cpu_start = sink->profile_enabled ? sink_thread_cpu_us() : 0;
            // Each stored value is the same rounded multiply as the old path. The slot-order sum starts at the
            // first slot's product, as the graph's does: an addition to +0 would turn a -0 product into +0.
            for (size_t expert = 0; expert < sink->requests.size(); ++expert) {
                float * contribution = sink->expert_output.data() + expert * size_t(sink->latent);
                const bool first = expert == 0;
                if (activation->post_exp_norm) {
                    kimi_cpu_sink_normalize_expert(contribution, sink->latent,
                        activation->post_exp_norm, activation->post_exp_norm_eps);
                    for (int row = 0; row < sink->latent; ++row) {
                        volatile float scaled = contribution[row] * sink->requests[expert].weight;
                        output[row] = first ? scaled : output[row] + scaled;
                    }
                } else {
                    for (int row = 0; row < sink->latent; ++row) {
                        output[row] = first ? contribution[row] : output[row] + contribution[row];
                    }
                }
            }
            sink->current.ordered_reduce_us = elapsed_us(reduce_start);
            sink->current.compute_us += sink->current.ordered_reduce_us;
            if (sink->profile_enabled) { sink->current.compute_cpu_us += sink_thread_cpu_us() - cpu_start; }
        }
        if (sink->timeline_active) { sink->timeline_call[5] = ggml_time_us(); }
        sink->current.total_us = elapsed_us(start);
        sink->current.read_completions = sink->ring.completed() - completed_before;
        sink->current.pending_reads = sink->ring.pending();
        sink->observe_decode_layer(lanes_us, sink->current.total_us);
        sink->finish_observation();
        if (stats) { *stats = sink->current; }
        sink->print_observation(true);
        const auto published = sink->publish_observation(true);
        if (!published) { sink_error(error, error_size, published.error.c_str()); return false; }
        return true;
    } catch (const std::exception & e) {
        // Cache slots are kernel read targets now. Stop producing new reads and
        // wait for every published SQE before unmapping, even if the caller
        // throws while the I/O thread is waiting for a compute credit to free.
        sink->cancel_and_wait();
        sink->prefetch_drain();
        sink->cache.release();
        if (sink->timeline_active) { sink->timeline_call[5] = ggml_time_us(); }
        sink->current.total_us = elapsed_us(start);
        sink->current.read_completions = sink->ring.completed() - completed_before;
        sink->current.pending_reads = sink->ring.pending();
        sink->finish_observation();
        if (stats) { *stats = sink->current; }
        sink->print_observation(false);
        const auto published = sink->publish_observation(false);
        if (!published && error && error_size) {
            std::snprintf(error, error_size, "%s; telemetry: %s", e.what(), published.error.c_str());
        } else { sink_error(error, error_size, e.what()); }
        return false;
    }
}

extern "C" bool kimi_cpu_sink_run(
        struct kimi_cpu_sink * sink, const struct ggml_tensor * up, const struct ggml_tensor * gate,
        const struct ggml_tensor * down, const float * input, const int32_t * ids, const float * weights,
        int nslots, float situ_beta, float situ_linear_beta, float * output,
        struct kimi_cpu_sink_stats * stats, char * error, size_t error_size) {
    const kimi_cpu_sink_activation activation = { KIMI_CPU_SINK_ACTIVATION_SITU, situ_beta, situ_linear_beta, 0.0f, nullptr, 0.0f };
    return kimi_cpu_sink_run_activation(sink, up, gate, down, input, ids, weights, nslots, &activation,
        output, stats, error, error_size);
}
