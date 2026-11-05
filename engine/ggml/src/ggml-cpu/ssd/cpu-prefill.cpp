#include "ssd/cpu-prefill.h"
#include "ssd/cpu-sink-internal.h"
#include "../quants.h"
#include "../vec.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using prefill_clock = std::chrono::steady_clock;
static double elapsed_us(prefill_clock::time_point start) {
    return std::chrono::duration<double, std::micro>(prefill_clock::now() - start).count();
}

static void prefill_error(char * dst, size_t size, const char * message) {
    if (dst && size) { std::snprintf(dst, size, "%s", message); }
}

static bool prefill_expert_type_supported(ggml_type type, int64_t columns) {
    if (int(type) < 0 || int(type) >= int(GGML_TYPE_COUNT) || ggml_blck_size(type) <= 0) { return false; }
    const auto * traits = ggml_get_type_traits_cpu(type);
    if (!traits || !traits->vec_dot || traits->vec_dot_type == GGML_TYPE_F32) { return false; }
    const auto * input = ggml_get_type_traits_cpu(traits->vec_dot_type);
    return input && input->from_float && ggml_blck_size(traits->vec_dot_type) > 0 &&
        columns % ggml_blck_size(type) == 0 && columns % ggml_blck_size(traits->vec_dot_type) == 0;
}

static bool prefill_activation_valid(const kimi_cpu_sink_activation * activation) {
    if (!activation || (activation->post_exp_norm &&
            (!std::isfinite(activation->post_exp_norm_eps) || activation->post_exp_norm_eps <= 0.0f))) { return false; }
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

// The route workspace a batch may take: above it a run fails before any read.
constexpr uint64_t prefill_workspace_max = 512ull << 20;

static void prefill_mxfp4_dot_pair(ggml_vec_dot_t dot, int n, float * out,
        const void * weights, const void * input0, const void * input1) {
#if defined(__x86_64__) || defined(_M_X64)
    (void) dot;
    ggml_vec_dot_mxfp4_q8_0_2x(n, out, weights, input0, input1);
#else
    dot(n, out + 0, 0, weights, 0, input0, 0, 1);
    dot(n, out + 1, 0, weights, 0, input1, 0, 1);
#endif
}

static void prefill_mxfp4_dot_quad(ggml_vec_dot_t dot, int n, float * out,
        const void * weights, const void * input0, const void * input1,
        const void * input2, const void * input3) {
#if defined(__x86_64__) || defined(_M_X64)
    (void) dot;
    ggml_vec_dot_mxfp4_q8_0_4x(n, out, weights, input0, input1, input2, input3);
#else
    dot(n, out + 0, 0, weights, 0, input0, 0, 1);
    dot(n, out + 1, 0, weights, 0, input1, 0, 1);
    dot(n, out + 2, 0, weights, 0, input2, 0, 1);
    dot(n, out + 3, 0, weights, 0, input3, 0, 1);
#endif
}

struct prefill_column {
    int token;
    std::vector<int> slots; // original flattened token/slot positions
};

struct prefill_group {
    int32_t expert;
    std::vector<prefill_column> columns;
};

} // namespace

struct kimi_cpu_prefill {
    kimi_cpu_sink * sink;
    std::mutex mutex;
    std::vector<prefill_group> groups;
    std::vector<int32_t> expert_ids;
    std::vector<int32_t> group_of_expert; // expert ID -> group, -1 while untouched
    std::vector<int32_t> route_request_index;
    std::vector<uint8_t> consumed; // groups the CPU computed; the GPU tier added the others into output
    // activation_up: SWIGLU's u beside g in activation until the chunk is gated
    std::vector<float> activation, activation_up, routed_output, post_exp_output;
    std::vector<uint8_t> quant_input, quant_hidden;
    ggml_vec_dot_t dot_up = nullptr, dot_down = nullptr;
    ggml_from_float_t quantize_input = nullptr, quantize_hidden = nullptr;
    int latent = 0, hidden = 0;
    int ntokens = 0, nslots = 0;
    size_t row_bytes_up = 0, row_bytes_down = 0;
    size_t quant_bytes_input = 0, quant_bytes_hidden = 0;
    const char * up_data = nullptr, * gate_data = nullptr, * down_data = nullptr;
    const float * weights = nullptr;
    float * output = nullptr;
    const prefill_group * active_group = nullptr;
    kimi_cpu_sink_activation activation_config = {};

    // Threads claim row chunks from a shared counter instead of a static
    // rows*ith/nth split, so a slow core (e.g. an LP-E core) no longer gates
    // every expert. Rows stay independent, so the arithmetic and the per-row
    // accumulation order are unchanged. A chunk starts at a multiple of 16 rows,
    // a multiple of every fixed ggml_vec_swiglu_f32 vector width (the predicated
    // ones have no scalar tail), and ends 16 rows later or at the row's end: gated
    // alone, it splits into vector body and scalar tail as the whole row would.
    // The final slot-order sums claim whole tokens from the same counter.
    static constexpr int row_chunk = 16;
    int column_block = 16; // 0 keeps the row-major order
    std::atomic<int> dynamic_next{0};
    bool down_phase = false;
    int mxfp4_gemm = 2; // 0: scalar, 1: two columns, 2: four columns
    int active_mxfp4_gemm = 0;
    bool group_trace = false;
    kimi_cpu_prefill_stats current = {};

    explicit kimi_cpu_prefill(kimi_cpu_sink * shared_sink) : sink(shared_sink) {
        group_trace = kimi_cpu_sink_debug(sink).trace;
        if (!sink) { throw std::runtime_error("CPU prefill needs a sink"); }
    }

    void prepare(const ggml_tensor * up, const ggml_tensor * gate, const ggml_tensor * down,
            const int32_t * ids, const float * route_weights, int nslots, int ntokens) {
        // Check before sizing any workspace. The shared stream validates the
        // exact packed ranges, direct-read descriptors, padding and tensor strides.
        const ggml_tensor * tensors[] = { up, gate, down };
        for (const auto * tensor : tensors) {
            if (!tensor || !ggml_is_contiguous(tensor) ||
                    tensor->ne[0] <= 0 || tensor->ne[0] > 16384 ||
                    tensor->ne[1] <= 0 || tensor->ne[1] > 16384 || tensor->ne[2] <= 0 || tensor->ne[3] != 1) {
                throw std::runtime_error("CPU prefill requires valid contiguous expert metadata");
            }
            if (!prefill_expert_type_supported(tensor->type, tensor->ne[0])) {
                throw std::runtime_error(std::string("this quantization is not supported yet: the CPU prefill cannot compute ") +
                    ggml_type_name(tensor->type) + " experts");
            }
        }
        if (up->type != gate->type) { throw std::runtime_error("CPU prefill requires up and gate experts of one type"); }
        if (up->ne[0] != gate->ne[0] || up->ne[1] != gate->ne[1] || up->ne[2] != gate->ne[2] ||
                down->ne[0] != up->ne[1] || down->ne[1] != up->ne[0] || down->ne[2] != up->ne[2]) {
            throw std::runtime_error("CPU prefill expert shape mismatch");
        }
        latent = static_cast<int>(up->ne[0]); hidden = static_cast<int>(up->ne[1]);
#if defined(__x86_64__) || defined(_M_X64)
        active_mxfp4_gemm = up->type == GGML_TYPE_MXFP4 && down->type == GGML_TYPE_MXFP4 ? mxfp4_gemm : 0;
#else
        active_mxfp4_gemm = 0;
#endif
        const auto * up_traits = ggml_get_type_traits_cpu(up->type);
        const auto * down_traits = ggml_get_type_traits_cpu(down->type);
        dot_up = up_traits->vec_dot;
        dot_down = down_traits->vec_dot;
        quantize_input = ggml_get_type_traits_cpu(up_traits->vec_dot_type)->from_float;
        quantize_hidden = ggml_get_type_traits_cpu(down_traits->vec_dot_type)->from_float;
        row_bytes_up = ggml_row_size(up->type, latent);
        row_bytes_down = ggml_row_size(down->type, hidden);
        quant_bytes_input = ggml_row_size(up_traits->vec_dot_type, latent);
        quant_bytes_hidden = ggml_row_size(down_traits->vec_dot_type, hidden);
        weights = route_weights;
        groups.clear(); expert_ids.clear();
        group_of_expert.assign(static_cast<size_t>(up->ne[2]), -1);
        route_request_index.assign(static_cast<size_t>(ntokens) * nslots, -1);
        // Groups in deterministic first-touch order, the order of disk streaming.
        for (int token = 0; token < ntokens; ++token) {
            for (int slot = 0; slot < nslots; ++slot) {
                const int route = token * nslots + slot;
                const int32_t expert = ids[route];
                if (expert < 0) { continue; }
                if (expert >= up->ne[2] || !std::isfinite(weights[route]) || weights[route] < 0.0f) {
                    throw std::runtime_error("invalid CPU prefill expert ID or weight");
                }
                ++current.routed_slots;
                int32_t & index = group_of_expert[size_t(expert)];
                if (index < 0) {
                    index = int32_t(groups.size());
                    groups.push_back({ expert, {} });
                    expert_ids.push_back(expert);
                }
                auto group = groups.begin() + index;
                route_request_index[size_t(route)] = index;
                // Tokens are visited in order, so repeated slots for the same
                // expert/token are adjacent in this expert's column list.
                if (group->columns.empty() || group->columns.back().token != token) {
                    group->columns.push_back({ token, {} });
                    ++current.expert_token_pairs;
                }
                group->columns.back().slots.push_back(route);
                current.max_group_tokens = std::max(current.max_group_tokens,
                        static_cast<uint64_t>(group->columns.size()));
            }
        }
        const size_t columns = static_cast<size_t>(current.max_group_tokens);
        const bool swiglu = activation_config.kind == KIMI_CPU_SINK_ACTIVATION_SWIGLU;
        // Each CPU route's weighted output, for the graph's slot-order sum per
        // token. A route the GPU tier serves is never written or read here.
        const uint64_t routed = uint64_t(ntokens) * uint64_t(nslots) * uint64_t(latent) * sizeof(float);
        const uint64_t scratch = (uint64_t(columns) * hidden * (swiglu ? 2 : 1) +
                (activation_config.post_exp_norm ? uint64_t(columns) * latent : 0)) * sizeof(float) +
                uint64_t(columns) * quant_bytes_hidden + uint64_t(ntokens) * quant_bytes_input;
        if (routed + scratch > prefill_workspace_max) {
            throw std::runtime_error("grouped CPU prefill of " + std::to_string(ntokens) + " tokens needs " +
                std::to_string((routed + scratch) >> 20) + " MiB of route workspace, above its " +
                std::to_string(prefill_workspace_max >> 20) + " MiB limit: use a smaller ubatch");
        }
        activation.resize(columns * hidden);
        activation_up.resize(swiglu ? columns * hidden : 0);
        post_exp_output.resize(activation_config.post_exp_norm ? columns * latent : 0);
        quant_hidden.resize(columns * quant_bytes_hidden);
        quant_input.resize(static_cast<size_t>(ntokens) * quant_bytes_input);
        routed_output.resize(static_cast<size_t>(ntokens) * nslots * latent);
        consumed.assign(groups.size(), 0);
        size_t workspace = (activation.capacity() + activation_up.capacity()) * sizeof(float) +
                quant_hidden.capacity() + quant_input.capacity() + routed_output.capacity() * sizeof(float) +
                post_exp_output.capacity() * sizeof(float) +
                groups.capacity() * sizeof(prefill_group) + expert_ids.capacity() * sizeof(int32_t);
        workspace += (route_request_index.capacity() + group_of_expert.capacity()) * sizeof(int32_t) +
                consumed.capacity();
        for (const auto & group : groups) {
            workspace += group.columns.capacity() * sizeof(prefill_column);
            for (const auto & column : group.columns) { workspace += column.slots.capacity() * sizeof(int); }
        }
        current.workspace_bytes = workspace;
    }

    static void parallel_rows(void * userdata, int, int) {
        auto & self = *static_cast<kimi_cpu_prefill *>(userdata);
        const auto & columns = self.active_group->columns;
        const int rows = self.down_phase ? self.latent : self.hidden;
        auto compute_range = [&](int begin, int end) {
        // Keep a weight row hot across token columns while preserving each dot's reductions.
        const size_t tile = self.active_mxfp4_gemm == 2 ? 4 : self.active_mxfp4_gemm == 1 ? 2 : 1;
        auto compute_row_columns = [&](int row, size_t first, size_t last) {
            for (size_t column = first; column < last; column += tile) {
                const size_t count = std::min(tile, last - column);
                if (!self.down_phase) {
                    float u[4], g[4];
                    const void * up = self.up_data + row * self.row_bytes_up;
                    const void * gate = self.gate_data + row * self.row_bytes_up;
                    if (count == 4) {
                        const void * input[4];
                        for (size_t offset = 0; offset < 4; ++offset) {
                            input[offset] = self.quant_input.data() + columns[column + offset].token * self.quant_bytes_input;
                        }
                        prefill_mxfp4_dot_quad(self.dot_up, self.latent, u, up,
                                input[0], input[1], input[2], input[3]);
                        prefill_mxfp4_dot_quad(self.dot_up, self.latent, g, gate,
                                input[0], input[1], input[2], input[3]);
                    } else {
                        const void * input = self.quant_input.data() + columns[column].token * self.quant_bytes_input;
                        if (count >= 2) {
                            const void * next = self.quant_input.data() + columns[column + 1].token * self.quant_bytes_input;
                            prefill_mxfp4_dot_pair(self.dot_up, self.latent, u, up, input, next);
                            prefill_mxfp4_dot_pair(self.dot_up, self.latent, g, gate, input, next);
                        }
                        if (count == 1 || count == 3) {
                            const size_t offset = count == 1 ? 0 : 2;
                            const void * last = self.quant_input.data() + columns[column + offset].token * self.quant_bytes_input;
                            self.dot_up(self.latent, u + offset, 0, up, 0, last, 0, 1);
                            self.dot_up(self.latent, g + offset, 0, gate, 0, last, 0, 1);
                        }
                    }
                    for (size_t offset = 0; offset < count; ++offset) {
                        float value;
                        if (self.activation_config.kind == KIMI_CPU_SINK_ACTIVATION_SWIGLU) {
                            // g and u as they are: the chunk is gated once complete
                            self.activation_up[(column + offset) * self.hidden + row] = u[offset];
                            value = g[offset];
                        } else if (self.activation_config.kind == KIMI_CPU_SINK_ACTIVATION_SWIGLU_CLAMP) {
                            const float limit = self.activation_config.swiglu_limit;
                            const float gate_value = std::min(g[offset], limit);
                            const float up_value = std::clamp(u[offset], -limit, limit);
                            value = gate_value / (1.0f + std::exp(-gate_value)) * up_value;
                        } else {
                            const float beta = self.activation_config.situ_beta;
                            const float linear_beta = self.activation_config.situ_linear_beta;
                            const float a = (beta * std::tanh(g[offset] * (1.0f / beta))) *
                                    (1.0f / (1.0f + std::exp(-g[offset])));
                            if (linear_beta > 0.0f) {
                                u[offset] = linear_beta * std::tanh(u[offset] * (1.0f / linear_beta));
                            }
                            value = a * u[offset];
                        }
                        self.activation[(column + offset) * self.hidden + row] = value;
                    }
                } else {
                    float values[4];
                    const auto * input = self.quant_hidden.data() + column * self.quant_bytes_hidden;
                    if (count == 4) {
                        prefill_mxfp4_dot_quad(self.dot_down, self.hidden, values,
                                self.down_data + row * self.row_bytes_down, input,
                                input + self.quant_bytes_hidden, input + 2 * self.quant_bytes_hidden,
                                input + 3 * self.quant_bytes_hidden);
                    } else if (count >= 2) {
                        prefill_mxfp4_dot_pair(self.dot_down, self.hidden, values,
                                self.down_data + row * self.row_bytes_down,
                                input, input + self.quant_bytes_hidden);
                    }
                    if (count == 1 || count == 3) {
                        const size_t offset = count == 1 ? 0 : 2;
                        self.dot_down(self.hidden, values + offset, 0, self.down_data + row * self.row_bytes_down, 0,
                                input + offset * self.quant_bytes_hidden, 0, 1);
                    }
                    for (size_t offset = 0; offset < count; ++offset) {
                        const auto & current = columns[column + offset];
                        if (self.activation_config.post_exp_norm) {
                            self.post_exp_output[(column + offset) * self.latent + row] = values[offset];
                        } else {
                            for (int route : current.slots) {
                                // Store each slot's separate multiplication before
                                // the final ordered additions. Never fuse scale/sum.
                                volatile float scaled = values[offset] * self.weights[route];
                                self.routed_output[static_cast<size_t>(route) * self.latent + row] = scaled;
                            }
                        }
                    }
                }
            }
        };
        if (self.active_mxfp4_gemm == 2 && self.column_block > 0) {
            const size_t block = static_cast<size_t>(self.column_block);
            for (size_t first = 0; first < columns.size(); first += block) {
                const size_t last = std::min(columns.size(), first + block);
                for (int row = begin; row < end; ++row) {
                    compute_row_columns(row, first, last);
                }
            }
        } else {
            for (int row = begin; row < end; ++row) {
                compute_row_columns(row, 0, columns.size());
            }
        }
        if (!self.down_phase && self.activation_config.kind == KIMI_CPU_SINK_ACTIVATION_SWIGLU) {
            // ggml_swiglu_split's kernel on this chunk of every column (row_chunk)
            for (size_t column = 0; column < columns.size(); ++column) {
                float * gate = self.activation.data() + column * self.hidden + begin;
                ggml_vec_swiglu_f32(end - begin, gate, gate, self.activation_up.data() + column * self.hidden + begin);
            }
        }
        };
        for (;;) {
            const int begin = self.dynamic_next.fetch_add(row_chunk, std::memory_order_relaxed);
            if (begin >= rows) { break; }
            compute_range(begin, std::min(rows, begin + row_chunk));
        }
    }

    static void consume(void * userdata, size_t expert, const void * up, const void * gate, const void * down) {
        auto & self = *static_cast<kimi_cpu_prefill *>(userdata);
        self.active_group = &self.groups[expert];
        self.up_data = static_cast<const char *>(up);
        self.gate_data = static_cast<const char *>(gate);
        self.down_data = static_cast<const char *>(down);
        self.consumed[expert] = 1;
        auto start = prefill_clock::now();
        self.down_phase = false;
        self.dynamic_next.store(0, std::memory_order_relaxed);
        kimi_cpu_sink_parallel(self.sink, parallel_rows, &self);
        self.current.stream.up_gate_us += elapsed_us(start);
        start = prefill_clock::now();
        for (size_t column = 0; column < self.active_group->columns.size(); ++column) {
            self.quantize_hidden(self.activation.data() + column * self.hidden,
                    self.quant_hidden.data() + column * self.quant_bytes_hidden, self.hidden);
        }
        self.current.stream.hidden_quantize_us += elapsed_us(start);
        start = prefill_clock::now();
        self.down_phase = true;
        self.dynamic_next.store(0, std::memory_order_relaxed);
        kimi_cpu_sink_parallel(self.sink, parallel_rows, &self);
        if (self.activation_config.post_exp_norm) {
            for (size_t column = 0; column < self.active_group->columns.size(); ++column) {
                float * values = self.post_exp_output.data() + column * self.latent;
                kimi_cpu_sink_normalize_expert(values, self.latent, self.activation_config.post_exp_norm,
                    self.activation_config.post_exp_norm_eps);
                const auto & current = self.active_group->columns[column];
                for (int route : current.slots) {
                    float * destination = self.routed_output.data() + size_t(route) * self.latent;
                    for (int row = 0; row < self.latent; ++row) {
                        volatile float scaled = values[row] * self.weights[route];
                        destination[row] = scaled;
                    }
                }
            }
        }
        self.current.stream.down_reduce_us += elapsed_us(start);
    }

    // Each token's CPU routes in slot order. The sum starts at the first one's
    // product, as the graph's does (an addition to +0 would turn a -0 product
    // into +0), unless GPU-served routes of the token already added theirs into
    // output. A participant claims a whole token and sums it alone, so no
    // element's order depends on the split.
    static void reduce_tokens(void * userdata, int, int) {
        auto & self = *static_cast<kimi_cpu_prefill *>(userdata);
        const int latent = self.latent, nslots = self.nslots, ntokens = self.ntokens;
        const int32_t * group = self.route_request_index.data();
        const uint8_t * consumed = self.consumed.data();
        const float * routed = self.routed_output.data();
        for (;;) {
            const int token = self.dynamic_next.fetch_add(1, std::memory_order_relaxed);
            if (token >= ntokens) { break; }
            float * destination = self.output + static_cast<size_t>(token) * latent;
            const int first = token * nslots, last = first + nslots;
            bool started = false;
            for (int route = first; route < last; ++route) { started |= group[route] >= 0 && !consumed[group[route]]; }
            for (int route = first; route < last; ++route) {
                if (group[route] < 0 || !consumed[group[route]]) { continue; }
                const float * value = routed + static_cast<size_t>(route) * latent;
                if (started) {
                    for (int row = 0; row < latent; ++row) { destination[row] += value[row]; }
                } else {
                    std::copy(value, value + latent, destination);
                    started = true;
                }
            }
        }
    }
};

extern "C" struct kimi_cpu_prefill * kimi_cpu_prefill_create(
        struct kimi_cpu_sink * sink, char * error, size_t error_size) {
    prefill_error(error, error_size, "");
    try {
        return new kimi_cpu_prefill(sink);
    } catch (const std::exception & e) {
        prefill_error(error, error_size, e.what());
        return nullptr;
    }
}

extern "C" void kimi_cpu_prefill_destroy(struct kimi_cpu_prefill * prefill) { delete prefill; }

extern "C" bool kimi_cpu_prefill_run_activation(
        struct kimi_cpu_prefill * prefill,
        const struct ggml_tensor * up, const struct ggml_tensor * gate, const struct ggml_tensor * down,
        const float * input, const int32_t * ids, const float * weights, int nslots, int ntokens,
        const struct kimi_cpu_sink_activation * activation, float * output,
        struct kimi_cpu_prefill_stats * stats, char * error, size_t error_size) {
    prefill_error(error, error_size, "");
    if (stats) { *stats = {}; }
    if (!prefill) { prefill_error(error, error_size, "null CPU prefill"); return false; }
    std::lock_guard<std::mutex> serial(prefill->mutex);
    const auto start = prefill_clock::now();
    prefill->current = {};
    try {
        // Disable decode cache use and check optional retention headroom before
        // allocating workspace, including direct calls outside graph hooks.
        if (!kimi_cpu_sink_cache_begin(prefill->sink, 0, 0, false, error, error_size)) { return false; }
        if (!input || !output || nslots < 0 || nslots > 16 || (nslots && (!ids || !weights)) ||
                ntokens < 1 || (nslots && ntokens > std::numeric_limits<int>::max() / nslots) ||
                !prefill_activation_valid(activation)) {
            throw std::runtime_error("CPU prefill requires a positive token count, 0..16 slots and a valid activation");
        }
        // The elastic tier must make room for the actual GPU accumulator before
        // the SSD ring is started. A worker allocation failure cannot strand
        // streamed GPU reads holding ring credits.
        if (!kimi_cpu_sink_vram_reserve_accumulator(prefill->sink, ntokens, error, error_size)) { return false; }
        // A lazily-created sink may have missed the context's post-reserve hook;
        // repeated layer calls are harmless. Disarm after the accumulator is ready.
        if (!kimi_cpu_sink_vram_phase_complete(prefill->sink, false, error, error_size)) { return false; }
        prefill->activation_config = *activation;
        prefill->prepare(up, gate, down, ids, weights, nslots, ntokens);
        if (prefill->active_mxfp4_gemm == 2 && prefill->group_trace) {
            std::string group_list;
            group_list.reserve(prefill->groups.size() * 8);
            for (const auto & group : prefill->groups) {
                if (!group_list.empty()) { group_list += ','; }
                group_list += std::to_string(group.expert) + ':' + std::to_string(group.columns.size());
            }
            std::fprintf(stderr, "[kimi_cpu_prefill_groups] tensor=%s ntokens=%d groups=%s\n",
                    up->name, ntokens, group_list.c_str());
        }
        // Default-off policy audit for cold-prefill admission.  Route
        // multiplicity alone is not evidence that a prompt-hot expert will be
        // useful to the following decode, so expose the exact candidates for
        // an offline prompt->decode overlap analysis without changing cache
        // heat, residency, execution order or payload lifetime.
        if (prefill->group_trace) {
                std::fprintf(stderr, "[kimi_cpu_prefill_admission_audit] tensor=%s ntokens=%d candidates=",
                    up->name, ntokens);
                bool first = true;
                for (const auto & group : prefill->groups) {
                    if (group.columns.size() < 16) { continue; }
                    std::fprintf(stderr, "%s%d:%zu", first ? "" : ",", group.expert, group.columns.size());
                    first = false;
                }
                std::fprintf(stderr, "\n");
        }
        prefill->activation_config = *activation;
        prefill->output = output;
        prefill->ntokens = ntokens;
        prefill->nslots = nslots;
        std::fill(output, output + static_cast<size_t>(ntokens) * prefill->latent, 0.0f);
        auto phase_start = prefill_clock::now();
        for (int token = 0; token < ntokens; ++token) {
            const float * column = input + static_cast<size_t>(token) * prefill->latent;
            for (int row = 0; row < prefill->latent; ++row) {
                if (!std::isfinite(column[row])) { throw std::runtime_error("nonfinite CPU prefill input"); }
            }
            prefill->quantize_input(column,
                    prefill->quant_input.data() + token * prefill->quant_bytes_input, prefill->latent);
        }
        const double input_us = elapsed_us(phase_start);
        kimi_cpu_sink_stats stream = {};
        const int32_t empty_id = 0; // keep the empty-union call's pointer valid
        kimi_cpu_sink_prefill_batch batch = {};
        batch.f32_input = input;
        batch.f32_stride = size_t(prefill->latent) * sizeof(float);
        batch.route_request_index = prefill->route_request_index.data();
        batch.weights = weights;
        batch.nslots = nslots;
        batch.ntokens = ntokens;
        batch.activation = *activation;
        batch.output = output;
        const bool ok = kimi_cpu_sink_stream_hybrid(prefill->sink, up, gate, down,
                prefill->expert_ids.empty() ? &empty_id : prefill->expert_ids.data(),
                static_cast<int>(prefill->expert_ids.size()), &batch, kimi_cpu_prefill::consume, prefill,
                &stream, error, error_size);
        prefill->current.gpu_experts = batch.gpu_experts;
        prefill->current.gpu_routes = batch.gpu_routes;
        prefill->current.gpu_transient_experts = batch.gpu_transient_experts;
        prefill->current.gpu_transient_routes = batch.gpu_transient_routes;
        prefill->current.gpu_transient_upload_bytes = batch.gpu_transient_upload_bytes;
        prefill->current.gpu_transient_upload_us = batch.gpu_transient_upload_us;
        prefill->current.gpu_ssd_transient_candidates = batch.gpu_ssd_transient_candidates;
        prefill->current.gpu_ssd_transient_experts = batch.gpu_ssd_transient_experts;
        prefill->current.gpu_ssd_transient_routes = batch.gpu_ssd_transient_routes;
        prefill->current.gpu_ssd_transient_upload_bytes = batch.gpu_ssd_transient_upload_bytes;
        prefill->current.gpu_ssd_transient_upload_us = batch.gpu_ssd_transient_upload_us;
        prefill->current.gpu_worker_us = batch.gpu_worker_us;
        prefill->current.gpu_cold_admit_candidates = batch.gpu_cold_admit_candidates;
        prefill->current.gpu_cold_admit_experts = batch.gpu_cold_admit_experts;
        prefill->current.gpu_cold_admit_routes = batch.gpu_cold_admit_routes;
        prefill->current.gpu_cold_admit_upload_bytes = batch.gpu_cold_admit_upload_bytes;
        prefill->current.gpu_cold_admit_upload_us = batch.gpu_cold_admit_upload_us;
        prefill->current.ram_experts = batch.ram_experts;
        prefill->current.ram_routes = batch.ram_routes;
        prefill->current.ram_bytes = batch.ram_bytes;
        prefill->current.ssd_experts = batch.ssd_experts;
        prefill->current.ssd_routes = batch.ssd_routes;
        prefill->current.ssd_bytes = batch.ssd_bytes;
        stream.up_gate_us = prefill->current.stream.up_gate_us;
        stream.hidden_quantize_us = prefill->current.stream.hidden_quantize_us;
        stream.down_reduce_us = prefill->current.stream.down_reduce_us;
        stream.input_quantize_us = input_us;
        stream.compute_us += input_us;
        prefill->current.stream = stream;
        if (ok) {
            // The slot-order sums, on every participant of the call, after the
            // stream returned with the GPU tier's routes in output.
            phase_start = prefill_clock::now();
            prefill->dynamic_next.store(0, std::memory_order_relaxed);
            kimi_cpu_sink_parallel_last(prefill->sink, kimi_cpu_prefill::reduce_tokens, prefill);
            const double reduce_us = elapsed_us(phase_start);
            prefill->current.stream.down_reduce_us += reduce_us;
            prefill->current.stream.compute_us += reduce_us;
        }
        prefill->current.stream.total_us = elapsed_us(start);
        if (stats) { *stats = prefill->current; }
        return ok;
    } catch (const std::exception & e) {
        kimi_cpu_sink_cache_reset(prefill->sink);
        prefill->current.stream.total_us = elapsed_us(start);
        if (stats) { *stats = prefill->current; }
        prefill_error(error, error_size, e.what());
        return false;
    }
}

extern "C" bool kimi_cpu_prefill_run(
        struct kimi_cpu_prefill * prefill,
        const struct ggml_tensor * up, const struct ggml_tensor * gate, const struct ggml_tensor * down,
        const float * input, const int32_t * ids, const float * weights, int nslots, int ntokens,
        float situ_beta, float situ_linear_beta, float * output,
        struct kimi_cpu_prefill_stats * stats, char * error, size_t error_size) {
    const kimi_cpu_sink_activation activation = {
        KIMI_CPU_SINK_ACTIVATION_SITU, situ_beta, situ_linear_beta, 0.0f, nullptr, 0.0f,
    };
    return kimi_cpu_prefill_run_activation(prefill, up, gate, down, input, ids, weights, nslots, ntokens,
            &activation, output, stats, error, error_size);
}
