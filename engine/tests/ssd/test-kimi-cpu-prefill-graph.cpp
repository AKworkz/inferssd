// Build the actual MoE frontend graphs using tiny metadata. No model is loaded,
// no graph is executed, and only the CPU backend is initialized.
#include "llama-graph.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "../../src/ssd/cpu-runtime.h"

#include <cstdio>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

static void require(bool ok, const char * message) {
    if (!ok) { throw std::runtime_error(message); }
}

static std::set<const ggml_tensor *> graph_dependencies(ggml_cgraph * graph) {
    std::vector<const ggml_tensor *> pending;
    for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) { pending.push_back(ggml_graph_node(graph, i)); }
    std::set<const ggml_tensor *> visited;
    while (!pending.empty()) {
        const ggml_tensor * tensor = pending.back(); pending.pop_back();
        if (!tensor || !visited.insert(tensor).second) { continue; }
        for (const auto * source : tensor->src) { if (source) { pending.push_back(source); } }
        if (tensor->view_src) { pending.push_back(tensor->view_src); }
    }
    return visited;
}

static void graph_case(ggml_backend_t cpu, int ntokens, bool prefill, bool decode, const char * label,
        ggml_type weight_type = GGML_TYPE_MXFP4, llm_arch arch = LLM_ARCH_DEEPSEEK41) {
    constexpr int latent = 256, hidden = 512, experts = 20, slots = 8;
    // Metadata-only runtimes: no pool, ring, resolver or cache is initialized.
    const auto configured = ssd::settings::parse("", "");
    require(bool(configured), "metadata fixture settings rejected");
    const ssd::runtime_services services;
    llama_kimi_cpu_sink_runtime runtime(1, 1, experts, {0}, configured.value, services, {});
    llama_kimi_cpu_sink_runtime other_runtime(1, 2, experts, {0}, configured.value, services, {});
    llm_graph_result result(2048);
    ggml_backend_t backends[] = { cpu };
    ggml_backend_buffer_type_t buffer_types[] = { ggml_backend_cpu_buffer_type() };
    std::unique_ptr<ggml_backend_sched, decltype(&ggml_backend_sched_free)> scheduler(
        ggml_backend_sched_new(backends, buffer_types, 1, 2048, false, false), ggml_backend_sched_free);
    require(bool(scheduler), "CPU scheduler allocation failed");
    llama_adapter_loras loras;
    llm_graph_params params = {};
    params.arch = arch;
    params.hparams.n_embd = latent;
    params.hparams.n_layer = 1;
    params.hparams.n_expert = experts;
    params.hparams.n_expert_used = slots;
    params.hparams.n_head_arr.fill(1);
    params.hparams.n_head_kv_arr.fill(1);
    params.hparams.n_embd_head_k_full = latent;
    params.hparams.n_embd_head_v_full = latent;
    params.hparams.swiglu_clamp_exp.fill(7.0f);
    params.cparams.n_ctx = params.cparams.n_ctx_seq = 128;
    params.cparams.n_batch = params.cparams.n_ubatch = 64;
    params.cparams.n_seq_max = 1;
    params.cparams.causal_attn = true;
    params.ubatch.b_equal_seqs = 1;
    params.ubatch.n_tokens = params.ubatch.n_seq_tokens = ntokens;
    params.ubatch.n_seqs = params.ubatch.n_seqs_unq = 1;
    params.ubatch.data = std::make_shared<llama_ubatch::data_t>();
    params.ubatch.data->seq_id_unq.push_back(0);
    params.ubatch.seq_id_unq = params.ubatch.data->seq_id_unq.data();
    params.gtype = LLM_GRAPH_TYPE_DEFAULT;
    params.sched = scheduler.get();
    params.backend_cpu = cpu;
    params.loras = &loras;
    params.n_outputs = 1;
    params.cpu_sink_prefill = prefill;
    params.cpu_sink_decode = decode;
    params.cpu_sink_runtime = &runtime;
    params.res = &result;

    require(params.allow_reuse(params), "identical graph parameters must reuse");
    auto changed = params;
    changed.cpu_sink_prefill = !changed.cpu_sink_prefill;
    require(!params.allow_reuse(changed), "prefill phase change must reject graph reuse");
    changed = params;
    changed.cpu_sink_decode = !changed.cpu_sink_decode;
    require(!params.allow_reuse(changed), "decode phase change must reject graph reuse");
    changed = params;
    changed.cpu_sink_runtime = &other_runtime;
    require(!params.allow_reuse(changed), "context runtime change must reject graph reuse");
    changed = params;
    ++changed.ubatch.n_tokens;
    require(!params.allow_reuse(changed), "token count change must reject graph reuse");

    llm_graph_context context(params);
    ggml_context * ctx = result.get_ctx();
    ggml_tensor * input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, latent, ntokens);
    ggml_tensor * logits = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, experts, ntokens);
    ggml_tensor * up = ggml_new_tensor_3d(ctx, weight_type, latent, hidden, experts);
    ggml_tensor * gate = ggml_new_tensor_3d(ctx, weight_type, latent, hidden, experts);
    ggml_tensor * down = ggml_new_tensor_3d(ctx, weight_type, hidden, latent, experts);
    ggml_set_name(up, "blk.0.ffn_up_exps.weight");
    ggml_set_name(gate, "blk.0.ffn_gate_exps.weight");
    ggml_set_name(down, "blk.0.ffn_down_exps.weight");
    // The Qwen routers read the stream as it is, with no norm of their own: the next layer's router predicts from it.
    const bool qwen = arch == LLM_ARCH_QWEN4EXP || arch == LLM_ARCH_QWEN35MOE;
    const bool predicted = qwen && decode && ntokens == 1;
    if (predicted) {
        ggml_tensor * next_router = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, latent, experts);
        context.cpu_sink_prefetch_hint = context.build_moe_prefetch_hint(input, nullptr, next_router, nullptr,
            LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX, experts, 0);
        require(context.cpu_sink_prefetch_hint, "a router without a norm must still predict the next experts");
    }
    ggml_tensor * output = context.build_moe_ffn(input, nullptr, up, gate, down, nullptr,
        experts, slots, LLM_FFN_SILU, true, 1.0f, qwen ? LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX :
        LLAMA_EXPERT_GATING_FUNC_TYPE_SIGMOID, 0, logits);
    ggml_build_forward_expand(result.get_gf(), output);
    require(output->type == GGML_TYPE_F32 && output->ne[0] == latent && output->ne[1] == ntokens &&
            output->ne[2] == 1 && output->ne[3] == 1, "MoE output shape mismatch");
    if (ntokens == 0) {
        require(ggml_nelements(output) == 0 && result.cpu_sink_nodes.empty(),
                "empty output pruning must not construct an expert executor");
        const auto dependencies = graph_dependencies(result.get_gf());
        require(!dependencies.count(up) && !dependencies.count(gate) && !dependencies.count(down),
                "empty FFN retained expert payload dependencies");
        std::printf("case=%s tokens=0 cpu_custom=0 status=PASS\n", label);
        return;
    }
    const bool use_sink = prefill || (decode && ntokens == 1);
    const auto dependencies = graph_dependencies(result.get_gf());
    if (use_sink) {
        require(output->op == GGML_OP_CUSTOM && result.cpu_sink_nodes.size() == 1, "missing CPU custom node");
        const auto & descriptor = *result.cpu_sink_nodes[0];
        require(descriptor.runtime == &runtime && descriptor.prefill == prefill,
                "custom node lost its context runtime or execution phase");
        require(qwen ? descriptor.activation.kind == KIMI_CPU_SINK_ACTIVATION_SWIGLU :
                descriptor.activation.kind == KIMI_CPU_SINK_ACTIVATION_SWIGLU_CLAMP && descriptor.activation.swiglu_limit == 7.0f,
                "custom node carries another activation than the model's");
        for (const auto * metadata : { &descriptor.up, &descriptor.gate, &descriptor.down }) {
            require(!metadata->data && !metadata->buffer && !metadata->extra && !metadata->view_src,
                    "expert descriptor retained payload ownership");
            for (const auto * source : metadata->src) {
                require(!source, "expert descriptor retained graph sources");
            }
        }
        require(ggml_backend_sched_get_tensor_backend(scheduler.get(), output) == cpu, "custom node not assigned to CPU");
        require(bool(output->src[5]) == predicted, "the custom node must carry the prediction it was given");
        if (predicted) {
            require(output->src[5]->type == GGML_TYPE_I32 && output->src[5]->ne[0] == slots,
                    "the prediction must rank as many experts as the router selects");
        }
        require(output->src[0] && output->src[1] && output->src[2] && !output->src[3], "custom node source count mismatch");
        for (int source = 0; source < 3; ++source) {
            require(ggml_is_contiguous(output->src[source]), "custom node input must be contiguous");
        }
        require(output->src[0]->type == GGML_TYPE_F32 && output->src[0]->ne[0] == latent &&
                ggml_nelements(output->src[0]) == latent * ntokens, "latent input shape mismatch");
        require(output->src[1]->type == GGML_TYPE_I32 && output->src[1]->ne[0] == slots &&
                output->src[1]->ne[1] == ntokens, "token-major expert IDs shape mismatch");
        require(output->src[2]->type == GGML_TYPE_F32 && output->src[2]->ne[0] == 1 &&
                output->src[2]->ne[1] == slots && output->src[2]->ne[2] == ntokens, "token-major weights shape mismatch");
        require(!dependencies.count(up) && !dependencies.count(gate) && !dependencies.count(down),
                "expert payload leaked into CPU sink graph dependencies");
    } else {
        require(result.cpu_sink_nodes.empty() && output->op != GGML_OP_CUSTOM, "disabled/multi-token decode used CPU sink");
        require(dependencies.count(up) && dependencies.count(gate) && dependencies.count(down),
                "legacy expert graph dependencies are missing");
    }
    require(!runtime.sink && !runtime.prefill.context && !other_runtime.sink,
            "graph construction unexpectedly initialized an executor");
    std::printf("case=%s tokens=%d cpu_custom=%d nodes=%d status=PASS\n",
        label, ntokens, use_sink ? 1 : 0, ggml_graph_n_nodes(result.get_gf()));
}

int main() {
    try {
        std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)> cpu(ggml_backend_cpu_init(), ggml_backend_free);
        require(bool(cpu), "CPU backend initialization failed");
        graph_case(cpu.get(), 0, true, false, "prefill_empty_output");
        graph_case(cpu.get(), 25, true, false, "prefill_ub25");
        graph_case(cpu.get(), 64, true, false, "prefill_ub64");
        graph_case(cpu.get(), 1, true, false, "prefill_single_token_tail");
        graph_case(cpu.get(), 1, false, true, "decode_one");
        graph_case(cpu.get(), 2, false, false, "disabled_two");
        graph_case(cpu.get(), 2, false, true, "decode_two_legacy");
        graph_case(cpu.get(), 64, true, false, "qwen4exp_prefill_ub64", GGML_TYPE_Q8_0, LLM_ARCH_QWEN4EXP);
        graph_case(cpu.get(), 1, false, true, "qwen4exp_decode_one", GGML_TYPE_Q8_0, LLM_ARCH_QWEN4EXP);
        graph_case(cpu.get(), 64, true, false, "qwen35moe_prefill_ub64", GGML_TYPE_Q8_0, LLM_ARCH_QWEN35MOE);
        graph_case(cpu.get(), 1, false, true, "qwen35moe_decode_one", GGML_TYPE_Q8_0, LLM_ARCH_QWEN35MOE);
        bool refused = false;
        try {
            graph_case(cpu.get(), 1, false, true, "unsupported_f16", GGML_TYPE_F16);
        } catch (const std::runtime_error & error) {
            const std::string message = error.what();
            refused = message.find("not supported yet") != std::string::npos && message.find("f16") != std::string::npos;
        }
        require(refused, "unsupported expert layout did not refuse with a recoverable graph error");
        std::printf("CPU_PREFILL_GRAPH_OK execution=not_run kda_validation=not_run\n");
        return 0;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "CPU_PREFILL_GRAPH_FAILED: %s\n", e.what());
        return 1;
    }
}
