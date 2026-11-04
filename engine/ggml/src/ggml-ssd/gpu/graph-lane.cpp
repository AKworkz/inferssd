// SPDX-License-Identifier: Apache-2.0
// Grouped prefill over a ggml backend, for a GPU that shares host memory: see graph-lane.h.
#include "graph-lane.h"

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml.h"
#include "host/arbiter.h"
#include "io/ring-types.h"
#include "platform/memory.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <exception>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace ssd { namespace gpu {
namespace {

using lane_clock = std::chrono::steady_clock;
double elapsed_us(lane_clock::time_point start) {
    return std::chrono::duration<double, std::micro>(lane_clock::now() - start).count();
}
void report(char * error, size_t size, const char * message) {
    if (error && size) { std::snprintf(error, size, "%s", message); }
}

// The most experts one graph takes. No more of a pass's streamed experts land at once than the sink's read ring holds;
// a larger graph would only take more working memory for copies of weights the device cannot read in place.
constexpr size_t batch_experts = size_t(ssd::io::expert_ring_depth);
// A bound on one expert's tensors in a graph: 3 weights, its rows, 3 products and up to 9 activation ops (SiTU from
// plain operations), 16 in all.
constexpr size_t tensors_per_expert = 24;
// Vulkan refuses host memory whose pointer or size is off the device's minImportedHostPointerAlignment, which ggml does
// not report (4 KiB on the GPUs we have seen); Metal maps the pages around any pointer. The lane's own working memory
// takes a bound above that.
constexpr size_t work_alignment = size_t(1) << 16;

// A range of host memory the device reads in place.
struct imported_range {
    const char * base;
    size_t bytes;
    ggml_backend_buffer_t buffer;
};

struct lane_expert {
    int32_t request;
    const void * role[3];
    int (*stage)(void *, int32_t, int);
    void * stage_context;
    std::vector<int32_t> rows;  // its tokens, ascending; a token's duplicate routes are merged
    std::vector<float> weights;
};

// The chain's line from whole graphs: a graph of c experts and r rows takes c * fixed + r * route. Least squares over
// the pass, both terms kept at or above zero; a pass's per-expert sums for the cost table are then built from the line.
struct batch_fit {
    double cc = 0, cr = 0, rr = 0, ct = 0, rt = 0;
    void add(double count, double rows, double us) {
        cc += count * count; cr += count * rows; rr += rows * rows; ct += count * us; rt += rows * us;
    }
    void line(double & fixed, double & route) const {
        const double det = cc * rr - cr * cr;
        fixed = det > 0 ? (ct * rr - rt * cr) / det : -1.0;
        route = det > 0 ? (rt * cc - ct * cr) / det : -1.0;
        if (fixed < 0) { fixed = 0; route = rr > 0 ? rt / rr : 0; }
        else if (route < 0) { route = 0; fixed = cc > 0 ? ct / cc : 0; }
    }
};

struct lane {
    ggml_backend_dev_t device = nullptr;
    ggml_backend_t backend = nullptr;
    ssd::host::context_client * host = nullptr;
    size_t tensor_alignment = 1;
    std::mutex imports_mutex;
    std::vector<imported_range> imports;
    // The rows of a pass and copies of the weights the device cannot read where they are.
    void * work = nullptr;
    size_t work_bytes = 0;
    ggml_backend_buffer_t work_buffer = nullptr;
    char * work_base = nullptr;
    int latent = 0, hidden = 0;
    ggml_type up_type = GGML_TYPE_COUNT, down_type = GGML_TYPE_COUNT;
    bool format_known = false, usable = false;
    bool pass_ready = false;  // this pass's working memory is there (reserve_pass): else every expert stays on the CPU
    // One pass, from launch to finish.
    std::thread worker;
    std::vector<lane_expert> experts;
    const char * input = nullptr;
    size_t input_stride = 0;
    kimi_vram_activation activation = {};
    std::vector<float> post_norm;  // latent weights of a per-expert post norm, or empty
    float post_norm_eps = 0.0f;
    int ntokens = 0;
    std::vector<float> accumulator;
    std::vector<uint8_t> touched;
    std::string failure;
    kimi_vram_prefill_stats stats = {};
    batch_fit fit;
    double last_fixed_us = 0, last_route_us = 0;
    uint64_t total_experts = 0, total_routes = 0, copied_bytes = 0;

    ~lane() {
        drain();
        std::fprintf(stderr, "[graph_lane] experts=%llu routes=%llu copied_bytes=%llu\n", (unsigned long long) total_experts,
            (unsigned long long) total_routes, (unsigned long long) copied_bytes);
        release_work();
        for (const auto & range : imports) { ggml_backend_buffer_free(range.buffer); }
        if (backend) { ggml_backend_free(backend); }
    }
    void drain() { if (worker.joinable()) { worker.join(); } }
    void release_work() {
        if (work_buffer) { ggml_backend_buffer_free(work_buffer); work_buffer = nullptr; }
        if (work) { ssd::platform::aligned_free(work); work = nullptr; }
        work_bytes = 0; work_base = nullptr;
        pass_ready = false;
    }
    size_t expert_bytes() const {
        return 2 * ggml_row_size(up_type, latent) * hidden + ggml_row_size(down_type, hidden) * latent;
    }
    size_t row_bytes() const { return (size_t(2) * latent + size_t(4) * hidden) * sizeof(float); }
    size_t work_needed(int rows) const {
        const size_t padding = batch_experts * tensors_per_expert * std::max<size_t>(tensor_alignment, 64);
        const size_t bytes = size_t(rows) * row_bytes() + batch_experts * expert_bytes() + padding;
        return (bytes + work_alignment - 1) & ~(work_alignment - 1);
    }
    void reserve_work(int rows) {
        const size_t needed = work_needed(rows);
        if (work_bytes >= needed) { return; }
        release_work();
        // Room from the host budget, as the sink makes for its ring.
        auto ticket = host ? host->take(needed, 0, ssd::host::requester::outside, false, ssd::host::allocation_class::tracked) :
            ssd::result<ssd::host::allocation_ticket>{};
        if (host && !ticket) { throw std::runtime_error(ticket.error); }
        work = ssd::platform::aligned_alloc_bytes(work_alignment, needed);
        if (!work) { throw std::runtime_error("the GPU lane cannot allocate its working memory"); }
        if (host) { ticket.value.commit(); }
        work_buffer = ggml_backend_dev_buffer_from_host_ptr(device, work, needed, needed);
        if (!work_buffer) { throw std::runtime_error("the GPU cannot read the lane's working memory in place"); }
        work_bytes = needed;
        work_base = static_cast<char *>(ggml_backend_buffer_get_base(work_buffer));
    }
    // Before a pass's reads, the one point where the host budget can make room: no layer holds a lease. The lane is
    // optional, so a refusal leaves this pass to the CPU and fails nothing.
    void reserve_pass(int tokens) {
        drain();
        pass_ready = false;
        if (!format_known || !usable) { return; }
        try {
            reserve_work(tokens);
            pass_ready = true;
        } catch (const std::exception & e) {
            std::fprintf(stderr, "[graph_lane] this pass stays on the CPU: %s\n", e.what());
        }
    }
    char * host_of(const ggml_tensor * tensor) const {
        return static_cast<char *>(work) + (static_cast<const char *>(tensor->data) - work_base);
    }
    // Whether the device can run this format at all, from its own answer about the chain's operations.
    void learn_format(const kimi_vram_expert_format & f) {
        if (format_known && f.latent == latent && f.hidden == hidden && f.weight_type == int(up_type) && f.down_type == int(down_type)) { return; }
        latent = f.latent; hidden = f.hidden; up_type = ggml_type(f.weight_type); down_type = ggml_type(f.down_type);
        format_known = true;
        ggml_init_params params{ggml_tensor_overhead() * tensors_per_expert, nullptr, true};
        ggml_context * ctx = ggml_init(params);
        if (!ctx) { usable = false; return; }
        ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, latent, 2);
        ggml_tensor * up = ggml_new_tensor_2d(ctx, up_type, latent, hidden);
        ggml_tensor * down = ggml_new_tensor_2d(ctx, down_type, hidden, latent);
        ggml_tensor * u = ggml_mul_mat(ctx, up, x);
        ggml_tensor * d = ggml_mul_mat(ctx, down, u);
        // The products and what every activation can be made of (activate).
        ggml_tensor * plain[] = { u, d, ggml_mul(ctx, u, u), ggml_silu(ctx, u), ggml_clamp(ctx, u, -1.0f, 1.0f),
            ggml_scale(ctx, u, 2.0f), ggml_tanh(ctx, u), ggml_sigmoid(ctx, u) };
        usable = true;
        for (ggml_tensor * op : plain) { usable = usable && ggml_backend_dev_supports_op(device, op); }
        ggml_free(ctx);
    }
    // The fused operation where the device has it, else the same math from plain operations.
    ggml_tensor * activate(ggml_context * ctx, ggml_tensor * gate, ggml_tensor * up) const {
        switch (activation.kind) {
            case KIMI_VRAM_ACTIVATION_SWIGLU: {
                ggml_tensor * fused = ggml_swiglu_split(ctx, gate, up);
                return ggml_backend_dev_supports_op(device, fused) ? fused : ggml_mul(ctx, ggml_silu(ctx, gate), up);
            }
            case KIMI_VRAM_ACTIVATION_SWIGLU_CLAMP: {
                const float limit = activation.swiglu_limit;
                ggml_tensor * fused = ggml_swiglu_clamp(ctx, gate, up, limit);
                if (ggml_backend_dev_supports_op(device, fused)) { return fused; }
                ggml_tensor * bounded = ggml_clamp(ctx, gate, -std::numeric_limits<float>::max(), limit);
                return ggml_mul(ctx, ggml_silu(ctx, bounded), ggml_clamp(ctx, up, -limit, limit));
            }
            case KIMI_VRAM_ACTIVATION_SITU: {
                const float beta = activation.situ_beta, linear = activation.situ_linear_beta;
                ggml_tensor * a = ggml_scale(ctx, ggml_tanh(ctx, ggml_scale(ctx, gate, 1.0f / beta)), beta);
                ggml_tensor * bounded = linear > 0.0f ?
                    ggml_scale(ctx, ggml_tanh(ctx, ggml_scale(ctx, up, 1.0f / linear)), linear) : up;
                return ggml_mul(ctx, ggml_mul(ctx, a, ggml_sigmoid(ctx, gate)), bounded);
            }
            default:
                throw std::runtime_error("the GPU lane has no such activation");
        }
    }
    // The range the device can read these bytes in, at an offset it allows; null: they need a copy.
    const imported_range * readable(const void * data, size_t bytes) {
        const char * pointer = static_cast<const char *>(data);
        std::lock_guard<std::mutex> lock(imports_mutex);
        for (const auto & range : imports) {
            if (pointer < range.base || pointer + bytes > range.base + range.bytes) { continue; }
            return size_t(pointer - range.base) % tensor_alignment ? nullptr : &range;
        }
        return nullptr;
    }
    // A weight matrix where it is, when the device can read it there; else a copy in working memory.
    void place_weights(ggml_tensor * tensor, const void * data, ggml_tallocr & scratch) {
        const size_t bytes = ggml_nbytes(tensor);
        if (const imported_range * range = readable(data, bytes)) {
            const size_t offset = size_t(static_cast<const char *>(data) - range->base);
            if (ggml_backend_tensor_alloc(range->buffer, tensor,
                    static_cast<char *>(ggml_backend_buffer_get_base(range->buffer)) + offset) != GGML_STATUS_SUCCESS) {
                throw std::runtime_error("the GPU lane cannot place a weight matrix in host memory");
            }
            return;
        }
        if (ggml_tallocr_alloc(&scratch, tensor) != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("the GPU lane's working memory is too small");
        }
        std::memcpy(host_of(tensor), data, bytes);
        copied_bytes += bytes;
    }
    void run_batch(size_t first, size_t count) {
        const auto start = lane_clock::now();
        const size_t nodes = count * tensors_per_expert;
        ggml_init_params params{ggml_tensor_overhead() * nodes + ggml_graph_overhead_custom(nodes, false), nullptr, true};
        ggml_context * ctx = ggml_init(params);
        if (!ctx) { throw std::runtime_error("the GPU lane cannot build a graph"); }
        struct context_free { ggml_context * c; ~context_free() { ggml_free(c); } } owner{ctx};
        ggml_cgraph * graph = ggml_new_graph_custom(ctx, nodes, false);
        ggml_tallocr scratch = ggml_tallocr_new(work_buffer);
        std::vector<ggml_tensor *> inputs(count), outputs(count);
        uint64_t rows = 0;
        for (size_t i = 0; i < count; ++i) {
            const lane_expert & expert = experts[first + i];
            rows += expert.rows.size();
            ggml_tensor * up = ggml_new_tensor_2d(ctx, up_type, latent, hidden);
            ggml_tensor * gate = ggml_new_tensor_2d(ctx, up_type, latent, hidden);
            ggml_tensor * down = ggml_new_tensor_2d(ctx, down_type, hidden, latent);
            place_weights(up, expert.role[0], scratch);
            place_weights(gate, expert.role[1], scratch);
            place_weights(down, expert.role[2], scratch);
            inputs[i] = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, latent, int64_t(expert.rows.size()));
            ggml_tensor * u = ggml_mul_mat(ctx, up, inputs[i]);
            ggml_tensor * g = ggml_mul_mat(ctx, gate, inputs[i]);
            outputs[i] = ggml_mul_mat(ctx, down, activate(ctx, g, u));
            ggml_build_forward_expand(graph, outputs[i]);
            if (ggml_tallocr_alloc(&scratch, inputs[i]) != GGML_STATUS_SUCCESS) {
                throw std::runtime_error("the GPU lane's working memory is too small");
            }
        }
        for (int n = 0; n < ggml_graph_n_nodes(graph); ++n) {
            ggml_tensor * node = ggml_graph_node(graph, n);
            // A clamp works in place: its node is a view of the row it bounds.
            const ggml_status placed = node->view_src ? ggml_backend_view_init(node) : ggml_tallocr_alloc(&scratch, node);
            if (placed != GGML_STATUS_SUCCESS) { throw std::runtime_error("the GPU lane's working memory is too small"); }
        }
        const size_t row_size = size_t(latent) * sizeof(float);
        for (size_t i = 0; i < count; ++i) {
            char * to = host_of(inputs[i]);
            for (int32_t token : experts[first + i].rows) {
                std::memcpy(to, input + size_t(token) * input_stride, row_size);
                to += row_size;
            }
        }
        const double fill_us = elapsed_us(start);
        const auto compute_start = lane_clock::now();
        if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("the GPU lane's graph failed");
        }
        const double compute_us = elapsed_us(compute_start);
        const auto add_start = lane_clock::now();
        for (size_t i = 0; i < count; ++i) {
            const lane_expert & expert = experts[first + i];
            float * from = reinterpret_cast<float *>(host_of(outputs[i]));
            for (size_t row = 0; row < expert.rows.size(); ++row, from += latent) {
                if (!post_norm.empty()) {
                    double sum = 0;
                    for (int c = 0; c < latent; ++c) { from[c] *= post_norm[size_t(c)]; sum += double(from[c]) * from[c]; }
                    const float scale = 1.0f / std::sqrt(float(sum / latent) + post_norm_eps);
                    for (int c = 0; c < latent; ++c) { from[c] *= scale; }
                }
                const size_t token = size_t(expert.rows[row]);
                float * to = accumulator.data() + token * size_t(latent);
                const float weight = expert.weights[row];
                touched[token] = 1;
                for (int c = 0; c < latent; ++c) { to[c] += from[c] * weight; }
            }
        }
        const double add_us = elapsed_us(add_start);
        fit.add(double(count), double(rows), compute_us);
        stats.engine_overhead_us += fill_us + add_us;
        stats.ssd_transient_experts += count;
        stats.ssd_transient_routes += rows;
        total_experts += count;
        total_routes += rows;
    }
    void run_pass() {
        const auto start = lane_clock::now();
        size_t next = 0;
        const size_t row_limit = size_t(ntokens);
        while (next < experts.size()) {
            lane_expert & head = experts[next];
            const auto wait_start = lane_clock::now();
            const bool ready = !head.stage || head.stage(head.stage_context, head.request, 0);
            stats.read_wait_us += elapsed_us(wait_start);
            if (!ready) {
                if (failure.empty()) { failure = "a read of the GPU lane failed"; }
                ++next;
                continue;
            }
            size_t count = 1, rows = head.rows.size();
            while (next + count < experts.size() && count < batch_experts) {
                const lane_expert & more = experts[next + count];
                if (rows + more.rows.size() > row_limit) { break; }
                if (more.stage && !more.stage(more.stage_context, more.request, 2)) { break; }
                rows += more.rows.size();
                ++count;
            }
            if (failure.empty()) {
                try {
                    run_batch(next, count);
                } catch (const std::exception & e) {
                    failure = e.what();
                }
            }
            // After a failure the reads still run to their end: the ring's credits go back, and the CPU's lane finishes.
            for (size_t i = 0; i < count; ++i) {
                const lane_expert & done = experts[next + i];
                if (done.stage) { done.stage(done.stage_context, done.request, 1); }
            }
            next += count;
        }
        double fixed = 0, route = 0;
        fit.line(fixed, route);
        last_fixed_us = fixed; last_route_us = route;
        for (const auto & expert : experts) {
            const double r = double(expert.rows.size()), us = fixed + route * r;
            stats.host_fit_experts += 1; stats.host_fit_routes += r; stats.host_fit_routes_sq += r * r;
            stats.host_fit_us += us; stats.host_fit_route_us += r * us;
        }
        stats.total_worker_us = elapsed_us(start);
    }
    int launch(const void * rows, size_t stride, const int32_t * route_request, const float * weights, int nslots, int tokens,
            const kimi_vram_prefill_transient * transient, int transient_count, const kimi_vram_prefill_launch * options,
            char * error, size_t error_size) {
        try {
            drain();
            input = static_cast<const char *>(rows);
            input_stride = stride; ntokens = tokens; activation = options->activation;
            if (options->post_norm) { post_norm.assign(options->post_norm, options->post_norm + latent); post_norm_eps = options->post_norm_eps; }
            else { post_norm.clear(); }
            failure.clear(); stats = {}; fit = {};
            if (!pass_ready || work_bytes < work_needed(tokens)) { throw std::runtime_error("the GPU lane was given a pass it did not reserve"); }
            experts.clear();
            experts.reserve(size_t(transient_count));
            std::vector<int32_t> expert_of_request;
            for (int i = 0; i < transient_count; ++i) {
                const auto & t = transient[i];
                experts.push_back({ t.request_index, { t.role[0], t.role[1], t.role[2] }, t.stage, t.stage_context, {}, {} });
                if (size_t(t.request_index) >= expert_of_request.size()) { expert_of_request.resize(size_t(t.request_index) + 1, -1); }
                expert_of_request[size_t(t.request_index)] = i;
            }
            for (int token = 0; token < tokens; ++token) {
                for (int slot = 0; slot < nslots; ++slot) {
                    const size_t route = size_t(token) * size_t(nslots) + size_t(slot);
                    const int32_t request = route_request[route];
                    if (request < 0 || size_t(request) >= expert_of_request.size() || expert_of_request[size_t(request)] < 0) { continue; }
                    auto & expert = experts[size_t(expert_of_request[size_t(request)])];
                    if (!expert.rows.empty() && expert.rows.back() == token) { expert.weights.back() += weights[route]; continue; }
                    expert.rows.push_back(token);
                    expert.weights.push_back(weights[route]);
                }
            }
            accumulator.assign(size_t(tokens) * size_t(latent), 0.0f);
            touched.assign(size_t(tokens), 0);
            worker = std::thread([this] { run_pass(); });
            return 0;
        } catch (const std::exception & e) {
            report(error, error_size, e.what());
            return 1;
        }
    }
    int finish(float * output, size_t stride, kimi_vram_prefill_stats * out, char * error, size_t error_size) {
        drain();
        const auto start = lane_clock::now();
        if (failure.empty()) {
            for (int token = 0; token < ntokens; ++token) {
                if (!touched[size_t(token)]) { continue; }
                float * to = output + size_t(token) * stride;
                const float * from = accumulator.data() + size_t(token) * size_t(latent);
                for (int c = 0; c < latent; ++c) { to[c] += from[c]; }
            }
        }
        stats.output_add_us += elapsed_us(start);
        if (out) { *out = stats; }
        experts.clear();
        if (!failure.empty()) { report(error, error_size, failure.c_str()); return 1; }
        return 0;
    }
};

lane * self(void * context) { return static_cast<lane *>(context); }

// The device of each ordinal the provider handed out.
std::mutex devices_mutex;
std::vector<ggml_backend_dev_t> devices;

// A device the lane can serve: integrated, or one that says it takes host memory. -1 for any other.
int device_ordinal(ggml_backend_t backend) {
    const ggml_backend_dev_t device = ggml_backend_get_device(backend);
    ggml_backend_dev_props props = {};
    ggml_backend_dev_get_props(device, &props);
    if (props.type != GGML_BACKEND_DEVICE_TYPE_IGPU && !props.caps.buffer_from_host_ptr) { return -1; }
    std::lock_guard<std::mutex> lock(devices_mutex);
    const auto found = std::find(devices.begin(), devices.end(), device);
    if (found != devices.end()) { return int(found - devices.begin()); }
    devices.push_back(device);
    return int(devices.size()) - 1;
}

void * create_lane(const settings::snapshot *, const runtime_services * services, char * error, size_t error_size) noexcept {
    try {
        ggml_backend_dev_t device;
        {
            std::lock_guard<std::mutex> lock(devices_mutex);
            if (!services || services->gpu_device < 0 || size_t(services->gpu_device) >= devices.size()) {
                throw std::runtime_error("the GPU lane has no such device");
            }
            device = devices[size_t(services->gpu_device)];
        }
        auto created = std::make_unique<lane>();
        created->device = device;
        created->host = services->host;
        // Its own backend: the lane's worker runs while the context's graph waits for the sink.
        created->backend = ggml_backend_dev_init(device, nullptr);
        if (!created->backend) { throw std::runtime_error("the GPU lane cannot open its device"); }
        created->tensor_alignment = ggml_backend_get_alignment(created->backend);
        return created.release();
    } catch (const std::exception & e) {
        report(error, error_size, e.what());
        return nullptr;
    }
}

// ---- The tier's tables: no residency, so the sink keeps decode off them (sink_vram::prefill_only).
void * tier_create(uint64_t, char * error, size_t error_size) { report(error, error_size, "the GPU lane starts from settings"); return nullptr; }
void tier_destroy(void * context) { delete self(context); }
int tier_begin(void *, uint64_t, uint64_t, int, char *, size_t) { return 0; }
int tier_prepare(void *, const kimi_vram_request *, int, int, int, int *, char * error, size_t error_size) { report(error, error_size, "retired"); return 1; }
int tier_launch(void *, const void *, size_t, float, float, char * error, size_t error_size) { report(error, error_size, "retired"); return 1; }
int tier_admit(void *, const kimi_vram_request *, const void *, const void *, const void *, int, char *, size_t) { return 0; }
int tier_finish(void *, float *, size_t, char *, size_t) { return 0; }
int tier_cancel(void * context, char *, size_t) { self(context)->drain(); self(context)->experts.clear(); return 0; }
const kimi_vram_tier_api tier_table = { 2, sizeof(kimi_vram_tier_api), tier_create, tier_destroy, tier_begin, tier_prepare, tier_launch, tier_admit, tier_finish, tier_cancel };

int prefill_prepare(void * context, const kimi_vram_request *, int count, const kimi_vram_expert_format * format, int * slots, char *, size_t) {
    std::fill(slots, slots + count, -1);
    self(context)->learn_format(*format);
    return 0;
}
int prefill_transient_available(void * context) { return self(context)->usable && self(context)->pass_ready; }
int prefill_launch(void * context, const void * input, size_t stride, const int32_t * route_request, const float * weights, int nslots, int ntokens,
        const kimi_vram_prefill_transient * transient, int count, const kimi_vram_prefill_launch * options, char * error, size_t error_size) {
    return self(context)->launch(input, stride, route_request, weights, nslots, ntokens, transient, count, options, error, error_size);
}
int prefill_finish(void * context, float * output, size_t stride, kimi_vram_prefill_stats * stats, char * error, size_t error_size) {
    return self(context)->finish(output, stride, stats, error, error_size);
}
int prefill_cold_active(void *) { return 0; }
int prefill_admit_cold(void *, const kimi_vram_request *, const void *, const void *, const void *, int * admitted, char * error, size_t error_size) {
    if (admitted) { *admitted = 0; }
    report(error, error_size, "retired");
    return 1;
}
void prefill_lane_info(void * context, kimi_vram_prefill_lane * out) {
    *out = {};
    out->engine = self(context)->usable ? KIMI_VRAM_PREFILL_ENGINE_BATCHED : KIMI_VRAM_PREFILL_ENGINE_NONE;
    out->f32_input = 1;
    out->in_place = 1;
    out->post_norm = 1;
}
const kimi_vram_prefill_api prefill_table = { 7, sizeof(kimi_vram_prefill_api), prefill_prepare, prefill_transient_available, prefill_launch,
    prefill_finish, tier_cancel, prefill_cold_active, prefill_admit_cold, prefill_lane_info };

int elastic_phase_complete(void * context, int decode, char *, size_t) {
    if (decode) { self(context)->drain(); self(context)->release_work(); }
    return 0;
}
int elastic_reserve_accumulate(void * context, int tokens, char *, size_t) {
    self(context)->reserve_pass(tokens);
    return 0;
}
int elastic_reserve(void * context, const kimi_vram_request *, const kimi_vram_expert_format * format, char *, size_t) {
    self(context)->learn_format(*format);
    return 0;
}
const kimi_vram_elastic_api elastic_table = { 3, sizeof(kimi_vram_elastic_api), elastic_phase_complete, elastic_reserve_accumulate, elastic_reserve };

int host_available(void *) { return 0; }
int host_launch(void *, const kimi_vram_host_job *, int, int * accepted, char *, size_t) { *accepted = 0; return 0; }
void host_costs(void *, double * resident_us, double * host_us) { *resident_us = *host_us = 0; }
int host_register(void * context, const void * pointer, size_t bytes, char * error, size_t error_size) {
    auto & l = *self(context);
    ggml_backend_buffer_t buffer = ggml_backend_dev_buffer_from_host_ptr(l.device, const_cast<void *>(pointer), bytes, bytes);
    if (!buffer) { report(error, error_size, "the device cannot read this memory in place"); return 1; }
    std::lock_guard<std::mutex> lock(l.imports_mutex);
    l.imports.push_back({ static_cast<const char *>(pointer), bytes, buffer });
    return 0;
}
int host_unregister(void * context, const void * pointer, char *, size_t) {
    auto & l = *self(context);
    l.drain();
    std::lock_guard<std::mutex> lock(l.imports_mutex);
    for (size_t i = 0; i < l.imports.size(); ++i) {
        if (l.imports[i].base != static_cast<const char *>(pointer)) { continue; }
        ggml_backend_buffer_free(l.imports[i].buffer);
        l.imports.erase(l.imports.begin() + std::ptrdiff_t(i));
        return 0;
    }
    return 0;
}
void host_costs2(void * context, kimi_vram_host_costs * costs) {
    *costs = {};
    costs->in_place = 1;
    costs->in_place_only = 1;
    // A one-expert graph: what the fit found, or until then no gain from the GPU over a CPU's expert.
    const auto & l = *self(context);
    costs->host_expert_us = l.last_fixed_us + l.last_route_us > 0 ? l.last_fixed_us + l.last_route_us : 1e6;
}
const kimi_vram_host_api host_table = { 3, sizeof(kimi_vram_host_api), host_available, host_launch, host_costs, host_register, host_unregister, host_costs2 };

const settings_api settings_table = { 2, sizeof(settings_api), sizeof(runtime_services), create_lane };

// The host ledger stays as without a GPU tier: the lane holds no expert across layers.
bool device_is_unified(int) { return false; }
bool attach_services(ggml_backend_t, const runtime_services *, char *, size_t) noexcept { return true; }
void * no_stream(ggml_backend_t) { return nullptr; }

} // namespace

const provider_v1 * graph_lane_provider() {
    static const provider_v1 provider = {
        1, sizeof(provider_v1),
        &tier_table, nullptr, nullptr, nullptr, &prefill_table, &elastic_table, nullptr,
        &host_table, nullptr, &settings_table, nullptr, nullptr,
        device_is_unified, device_ordinal, attach_services, no_stream,
    };
    return &provider;
}

}} // namespace ssd::gpu
