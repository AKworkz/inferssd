// SPDX-License-Identifier: Apache-2.0
#include "model-adapter.h"

#include "llama-hparams.h"

#include <cmath>

namespace llama_ssd {

expert_requirements describe_experts(
        llm_arch architecture, const llama_hparams & hparams, int layer) noexcept {
    switch (architecture) {
        case LLM_ARCH_DEEPSEEK41:
            return {expert_quantization::all_mxfp4, expert_activation::swiglu_clamp,
                layer >= 0 ? hparams.swiglu_clamp_exp[layer] : 0.0f, true, false, true};
        case LLM_ARCH_MUSE_SPARK:
            return {expert_quantization::matching_up_gate, expert_activation::swiglu, 0.0f, false, true, false};
        case LLM_ARCH_QWEN35MOE:
        case LLM_ARCH_QWEN4EXP:
            return {expert_quantization::matching_up_gate, expert_activation::swiglu, 0.0f, false, false, false};
        default:
            return {};
    }
}

namespace {

bool separate_contiguous_experts(const expert_inputs & in) noexcept {
    return in.up && in.gate && in.down && !in.gate_up &&
        ggml_is_contiguous(in.up) &&
        ggml_is_contiguous(in.gate) && ggml_is_contiguous(in.down) &&
        ggml_are_same_shape(in.up, in.gate) && in.up->ne[3] == 1 &&
        in.down->ne[3] == 1 && in.down->ne[0] == in.up->ne[1] &&
        in.down->ne[1] == in.n_embd && in.up->ne[0] == in.n_embd &&
        in.down->ne[2] == in.up->ne[2];
}

bool has_extra_parameters(const expert_inputs & in) noexcept {
    return in.gate_up_bias || in.gate_input_bias ||
        in.up_bias || in.gate_bias || in.down_bias ||
        in.up_scale || in.gate_scale || in.down_scale ||
        in.weight_before_ffn;
}

bool quantization_accepted(expert_quantization rule, const expert_inputs & in) noexcept {
    switch (rule) {
        case expert_quantization::all_mxfp4:
            return in.up->type == GGML_TYPE_MXFP4 &&
                in.gate->type == GGML_TYPE_MXFP4 && in.down->type == GGML_TYPE_MXFP4;
        case expert_quantization::matching_up_gate:
            // Do not invent a model-specific type list: the selected
            // executor validates its actual vec-dot/quantization traits.
            return in.gate->type == in.up->type;
        case expert_quantization::unsupported:
            return false;
    }
    return false;
}

expert_refusal first_refusal(const expert_requirements & requirements, const expert_inputs & in) noexcept {
    if (requirements.quantization == expert_quantization::unsupported ||
            requirements.activation == expert_activation::unsupported) {
        return expert_refusal::unsupported_model;
    }
    if (!in.backend_available || !in.runtime_available) { return expert_refusal::no_executor; }
    if (!separate_contiguous_experts(in)) { return expert_refusal::layout; }
    if (has_extra_parameters(in)) { return expert_refusal::extra_parameters; }
    if (in.has_lora) { return expert_refusal::lora; }
    if (!quantization_accepted(requirements.quantization, in)) { return expert_refusal::quantization; }
    if (!in.silu) { return expert_refusal::not_silu_gated; }
    if (in.cache_compute) { return expert_refusal::compute_cache; }
    if (requirements.single_token_or_prefill && in.n_tokens != 1 && !in.prefill) {
        return expert_refusal::token_batch;
    }
    if (requirements.finite_clamp && !(std::isfinite(requirements.clamp) && requirements.clamp > 1e-6f)) {
        return expert_refusal::invalid_clamp;
    }
    if (requirements.require_post_norm && !in.post_norm) { return expert_refusal::missing_post_norm; }
    return expert_refusal::none;
}

std::string describe_quantization(const expert_inputs & in) {
    return std::string("this quantization is not supported yet: the routed experts are up=") +
        ggml_type_name(in.up->type) + " gate=" + ggml_type_name(in.gate->type) +
        " down=" + ggml_type_name(in.down->type);
}

} // namespace

expert_validation validate_experts(const expert_requirements & requirements, const expert_inputs & in) noexcept {
    const expert_refusal refusal = first_refusal(requirements, in);
    if (refusal != expert_refusal::none) { return {refusal, {}}; }
    return {expert_refusal::none, {requirements.activation, requirements.clamp, in.post_norm != nullptr,
        in.post_norm_eps, in.routed_scale == 0.0f ? 1.0f : in.routed_scale}};
}

std::string describe_refusal(const expert_inputs & in, expert_refusal refusal) {
    switch (refusal) {
        case expert_refusal::none: return {};
        case expert_refusal::unsupported_model: return "this model architecture has no SSD expert support";
        case expert_refusal::no_executor: return "the CPU backend or the SSD runtime is unavailable";
        case expert_refusal::layout:
            return "routed experts must be contiguous, separate up, gate and down tensors with matching shapes";
        case expert_refusal::extra_parameters:
            return "routed experts with biases, scales or weights applied before the layer are not supported";
        case expert_refusal::lora: return "LoRA adapters are not supported with SSD experts";
        case expert_refusal::quantization: return describe_quantization(in);
        case expert_refusal::not_silu_gated: return "routed experts must be SiLU-gated";
        case expert_refusal::compute_cache: return "the expert compute cache is not supported with SSD experts";
        case expert_refusal::token_batch: return "this model runs routed experts one token at a time outside prefill";
        case expert_refusal::invalid_clamp: return "the model's SwiGLU clamp must be finite and positive";
        case expert_refusal::missing_post_norm: return "this model requires a normalization tensor after its experts";
    }
    return "unknown expert refusal";
}

} // namespace llama_ssd
