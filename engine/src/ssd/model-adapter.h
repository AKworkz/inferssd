// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "llama-arch.h"

#include <string>

struct llama_hparams;

namespace llama_ssd {

// Model facts remain in L3. Neither this description nor its validator knows
// an executor's private cache, kernels, worker threads, or I/O implementation.
enum class expert_quantization { unsupported, matching_up_gate };
enum class expert_activation { unsupported, swiglu_clamp, swiglu };

struct expert_requirements {
    expert_quantization quantization = expert_quantization::unsupported;
    expert_activation activation = expert_activation::unsupported;
    float clamp = 0.0f;
    bool finite_clamp = false;
    bool require_post_norm = false;
    bool single_token_or_prefill = false;
};

[[nodiscard]] expert_requirements describe_experts(
        llm_arch architecture, const llama_hparams & hparams, int layer) noexcept;

struct expert_inputs {
    bool backend_available;
    bool runtime_available;
    const ggml_tensor * up;
    const ggml_tensor * gate;
    const ggml_tensor * down;
    const ggml_tensor * gate_up;
    const ggml_tensor * gate_up_bias;
    const ggml_tensor * gate_input_bias;
    const ggml_tensor * up_bias;
    const ggml_tensor * gate_bias;
    const ggml_tensor * down_bias;
    const ggml_tensor * up_scale;
    const ggml_tensor * gate_scale;
    const ggml_tensor * down_scale;
    int64_t n_embd;
    int64_t n_tokens;
    bool silu;
    bool prefill;
    bool weight_before_ffn;
    bool has_lora;
    const ggml_tensor * post_norm;
    float post_norm_eps;
    float routed_scale;
};

// The shared graph bridge translates this validated, model-neutral description
// to the existing executor node. Kernel/type support remains checked by the
// selected backend, at its existing initialization boundary.
struct expert_execution {
    expert_activation activation = expert_activation::unsupported;
    float clamp = 0.0f;
    bool post_norm = false;
    float post_norm_eps = 0.0f;
    float routed_scale = 1.0f;
};

// The first requirement a layer's routed experts do not meet.
enum class expert_refusal {
    none,
    unsupported_model,
    no_executor,
    layout,
    extra_parameters,
    lora,
    quantization,
    not_silu_gated,
    token_batch,
    invalid_clamp,
    missing_post_norm,
};

struct expert_validation {
    expert_refusal refusal = expert_refusal::none;
    expert_execution execution;

    [[nodiscard]] bool accepted() const noexcept { return refusal == expert_refusal::none; }
};

[[nodiscard]] expert_validation validate_experts(
        const expert_requirements & requirements, const expert_inputs & inputs) noexcept;

// States a refusal for the person running the model, with the types it concerns.
[[nodiscard]] std::string describe_refusal(const expert_inputs & inputs, expert_refusal refusal);

} // namespace llama_ssd
