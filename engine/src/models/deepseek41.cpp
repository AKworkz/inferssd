// SPDX-License-Identifier: MIT
#include "models.h"
#include "deepseek41-candidate.h"

#include "llama-engram.h"
#include "llama-impl.h"
#include "llama-kv-cache-dsv4.h"
#include "llama-mmap.h"

#include "ggml-backend.h"

#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>


// DeepSeek-V4.1.
//
// Ported from vcruz305/llama.cpp runtime/deepseek41 @ 5210c7c: src/models/deepseek41.cpp plus the
// DeepSeek-V4 helpers it inherits from upstream src/models/deepseek4.cpp.
//
// Shares almost everything with DeepSeek-V4: the hyper-connection stream, the MoE, the latent
// attention and the compressed KV cache. Three things differ.
//
// 1. The hyper-connection coefficients lag by one sublayer. A sublayer computes the mix that the
//    NEXT one consumes, so attention uses what the previous layer's FFN produced.
// 2. There is no learned hyper-connection head; the last layer's FFN mix collapses the copies.
// 3. The engram tables: n-gram keyed lookups added into the stream at a few layers.
//
// The sparse attention also differs. V4.1 compresses KV on a few source layers and the layers after
// each source read the same rows, and it derives index keys from that shared latent rather than
// from a second compressor. See build_attention_v41().
//
// Not implemented (as upstream): the two level candidate mask. It selects every block until the
// compressed length passes candidate_topk_blocks * candidate_block_size, so the context is capped
// there instead.
//
// Differences from the upstream port:
//  - no per-model classes in this tree, so the engram constants live on llama_model;
//  - hyper-connection mixing always uses plain ggml ops (the fused DSV4 HC kernels are not ported);
//  - wo_a is loaded 2D and reshaped in the graph;
//  - no per-layer input embedding extraction and no MTP graph.

namespace {

float dsv41_rope_attn_factor(float freq_scale, float ext_factor) {
    if (ext_factor == 0.0f) {
        return 1.0f;
    }

    return 1.0f / (1.0f + 0.1f*logf(1.0f/freq_scale));
}

size_t dsv41_elem_offset(const ggml_tensor * t, int64_t i) {
    return ggml_row_size(t->type, i);
}

ggml_tensor * dsv41_view_1d(ggml_context * ctx, ggml_tensor * t, int64_t ne0, int64_t i0) {
    return ggml_view_1d(ctx, t, ne0, dsv41_elem_offset(t, i0));
}

ggml_tensor * dsv41_view_2d(ggml_context * ctx, ggml_tensor * t, int64_t ne0, int64_t ne1, int64_t i0) {
    return ggml_view_2d(ctx, t, ne0, ne1, t->nb[1], dsv41_elem_offset(t, i0));
}

ggml_tensor * dsv41_hc_affine(ggml_context * ctx, ggml_tensor * x, ggml_tensor * scale, ggml_tensor * base) {
    x = ggml_mul(ctx, x, scale);
    x = ggml_add(ctx, x, base);
    return x;
}

// mean over the hyper-connection copies: [n_embd, hc, n_tokens] -> [n_embd, n_tokens]
ggml_tensor * dsv41_hc_mean(ggml_context * ctx, ggml_tensor * x) {
    const int64_t hc = x->ne[1];

    ggml_tensor * acc = ggml_view_2d(ctx, x, x->ne[0], x->ne[2], x->nb[2], 0);
    for (int64_t s = 1; s < hc; ++s) {
        acc = ggml_add(ctx, acc, ggml_view_2d(ctx, x, x->ne[0], x->ne[2], x->nb[2], s*x->nb[1]));
    }
    return ggml_scale(ctx, acc, 1.0f/hc);
}

struct dsv41_state_tensors {
    ggml_tensor * kv    = nullptr;
    ggml_tensor * score = nullptr;
};

dsv41_state_tensors dsv41_build_state_restore(
        ggml_context * ctx,
        const llm_graph_input_dsv4::comp_input & inp,
        const llama_dsv4_comp_state * state,
        int32_t il) {
    dsv41_state_tensors restored = {
        state->get_kv_all(ctx, il),
        state->get_score_all(ctx, il),
    };

    if (inp.state_restore_src_idxs == nullptr || inp.state_restore_dst_idxs == nullptr) {
        return restored;
    }

    ggml_tensor * kv_rows = ggml_get_rows(ctx, restored.kv, inp.state_restore_src_idxs);
    restored.kv = state->cpy_kv(ctx, kv_rows, inp.state_restore_dst_idxs, il);

    ggml_tensor * score_rows = ggml_get_rows(ctx, restored.score, inp.state_restore_src_idxs);
    restored.score = state->cpy_score(ctx, score_rows, inp.state_restore_dst_idxs, il);

    return restored;
}

dsv41_state_tensors dsv41_build_state_snapshot(
        ggml_context * ctx,
        const llm_graph_input_dsv4::comp_input & inp,
        const llama_dsv4_comp_state * state,
        ggml_tensor * source_kv,
        ggml_tensor * source_score,
        int32_t il) {
    if (inp.state_snapshot_src_idxs == nullptr || inp.state_snapshot_dst_idxs == nullptr ||
            source_kv == nullptr || source_score == nullptr) {
        return {};
    }

    ggml_tensor * kv_rows = ggml_get_rows(ctx, source_kv, inp.state_snapshot_src_idxs);
    ggml_tensor * kv = state->cpy_kv(ctx, kv_rows, inp.state_snapshot_dst_idxs, il);

    ggml_tensor * score_rows = ggml_get_rows(ctx, source_score, inp.state_snapshot_src_idxs);
    ggml_tensor * score = state->cpy_score(ctx, score_rows, inp.state_snapshot_dst_idxs, il);

    return { kv, score };
}

// The reference narrows the index sources after layer 20 to the candidate blocks of layer 20 (vLLM
// deepseek_v41 config: candidate_source_layer_id 20, candidate_topk_blocks 2048, candidate_block_size 8).
// The GGUF does not carry these keys. LLAMA_DSV41_CANDIDATE=0 turns the filter off.
static constexpr int32_t DSV41_CANDIDATE_SOURCE      = 20;
static constexpr int64_t DSV41_CANDIDATE_TOPK_BLOCKS = 2048;
static constexpr int64_t DSV41_CANDIDATE_BLOCK_SIZE  = 8;

void dsv41_engram_prefetch_rows(const ggml_tensor * table, const std::vector<int32_t> & rows) {
    if (!table || !table->data || !table->buffer || !ggml_backend_buffer_is_host(table->buffer)) {
        return;
    }
    // advisory only: on failure the row is read by the regular fault in get_rows
    (void) llama_prefetch_rows(table->data, table->nb[1], table->ne[1], rows.data(), rows.size());
}

void dsv41_engram_decode_op(ggml_tensor * dst, int ith, int nth, void *) {
    const ggml_tensor * table = dst->src[0];
    const ggml_tensor * rows  = dst->src[1];
    GGML_ASSERT(table != nullptr && rows != nullptr);
    GGML_ASSERT(table->type == GGML_TYPE_I8 && table->ne[0] == (int64_t) LLAMA_DSV41_ENGRAM_ROW_BYTES);
    GGML_ASSERT(rows->type == GGML_TYPE_I32 && ggml_is_contiguous(rows));
    GGML_ASSERT(dst->type == GGML_TYPE_F32 && dst->ne[0] == (int64_t) LLAMA_DSV41_ENGRAM_DIM);
    GGML_ASSERT(table->nb[0] == 1 && dst->nb[0] == sizeof(float));

    const int32_t * row_ids = static_cast<const int32_t *>(rows->data);
    const int64_t n_rows = ggml_nelements(rows);
    GGML_ASSERT(dst->ne[1] * dst->ne[2] * dst->ne[3] == n_rows);

    for (int64_t i = ith; i < n_rows; i += nth) {
        const int32_t row = row_ids[i];
        if (row < 0 || row >= table->ne[1]) {
            GGML_ABORT("DeepSeek-V4.1 Engram row %" PRId32 " is outside [0, %" PRId64 ")", row, table->ne[1]);
        }
        const uint8_t * raw = static_cast<const uint8_t *>(table->data) + (size_t) row*table->nb[1];
        float * out = reinterpret_cast<float *>(static_cast<uint8_t *>(dst->data) + (size_t) i*dst->nb[1]);
        const auto status = llama_dsv41_engram_decode_row(raw, out);
        if (status != llama_dsv41_engram_decode_status::ok) {
            GGML_ABORT("invalid e4m3_e8m0_32_row264 Engram value in row %" PRId32 " (status %u)",
                       row, (unsigned) status);
        }
    }
}

// Engram n-gram hash: each token gathers n_cols rows of this layer's table.
//   rolling_i = (t[0]*m[0]) ^ ... ^ (t[i]*m[i]);  row = rolling_i % prime[i][h] + offset[i][h]
// The hash runs host-side because ggml has no 64 bit integers and no xor. Look-back stops at the
// start of the sequence, and the compressed token map folds case and accents together first.
class llm_graph_input_engram : public llm_graph_input_i {
public:
    llm_graph_input_engram(const llama_model & model,
                           const llama_kv_cache_dsv4_raw_context * mctx,
                           int eg, bool prefetch) : model(model), mctx(mctx), eg(eg), prefetch(prefetch) {}
    virtual ~llm_graph_input_engram() = default;

    void set_input(const llama_ubatch * ubatch) override;

    bool can_reuse(const llm_graph_params & params) override {
        mctx = static_cast<const llama_kv_cache_dsv4_context *>(params.mctx)->get_raw();
        const int64_t n_cols = (model.hparams.engram_max_ngram_size - 1) * model.hparams.engram_n_head;
        return rows->ne[0] == n_cols * params.ubatch.n_tokens;
    }

    ggml_tensor * rows = nullptr;   // I32 [n_cols * n_tokens]

    const llama_model & model;

    // the predecessor tokens live in the attention KV cells (ext.tok)
    const llama_kv_cache_dsv4_raw_context * mctx;

    // which engram layer this is, so the right multipliers and buckets are used
    const int eg;
    const bool prefetch;

    // scratch, reused across set_input() calls
    std::vector<llama_token> prev;
};

void llm_graph_input_engram::set_input(const llama_ubatch * ubatch) {
    const auto & hp = model.hparams;

    const int64_t n_tokens = ubatch->n_tokens;
    const int64_t n_gram   = hp.engram_max_ngram_size;
    const int64_t n_heads  = hp.engram_n_head;
    const int64_t n_cols   = (n_gram - 1) * n_heads;
    const int64_t n_prev   = n_gram - 1;

    const uint64_t * mult   = model.engram_multipliers.data() + (size_t) eg*n_gram;
    const uint64_t * prime  = model.engram_primes.data()      + (size_t) eg*n_cols;
    const uint64_t * offset = model.engram_offsets.data()     + (size_t) eg*n_cols;

    // an image arrives as an embd batch, so ubatch->token is null; the reference gives those
    // positions no engram contribution at all, which the padding token stands in for here
    const int32_t pad = (int32_t) model.engram_pad_id;
    auto map_of = [&](llama_token t) -> uint64_t {
        if (t < 0 || (size_t) t >= model.engram_token_map.size()) {
            return (uint64_t) pad;
        }
        return (uint64_t) model.engram_token_map[t];
    };

    std::vector<int32_t> idx(n_cols * n_tokens);

    GGML_ASSERT(mctx != nullptr);

    for (int64_t i = 0; i < n_tokens; ++i) {
        // the preceding tokens would be ambiguous, see get_prev_tokens()
        GGML_ASSERT(ubatch->n_seq_id[i] == 1 && "engram n-gram lookups do not support tokens shared by multiple sequences");
    }

    // predecessors come from the KV cells (ext.tok); apply_ubatch() already stored this ubatch
    mctx->get_prev_tokens(*ubatch, n_prev, prev);

    std::vector<uint64_t> ctx(n_gram);
    for (int64_t i = 0; i < n_tokens; ++i) {
        // look-back stops at the start of the sequence; everything from there on reads as padding
        ctx[0] = ubatch->token ? map_of(ubatch->token[i]) : (uint64_t) pad;
        bool blocked = false;
        for (int64_t s = 1; s < n_gram; ++s) {
            // predecessor s positions back; prev[] is oldest-first, missing entries are LLAMA_TOKEN_NULL
            const llama_token t = blocked ? LLAMA_TOKEN_NULL : prev[i*n_prev + (n_prev - s)];
            blocked = blocked || t < 0;
            ctx[s] = blocked ? (uint64_t) pad : map_of(t);
        }

        // compressed ids stay under 2^17 and the multipliers under 2^37, so no product overflows
        uint64_t rolling = ctx[0] * mult[0];
        for (int64_t s = 1; s < n_gram; ++s) {
            rolling ^= ctx[s] * mult[s];

            for (int64_t h = 0; h < n_heads; ++h) {
                const int64_t b = (s - 1)*n_heads + h;
                idx[i*n_cols + b] = (int32_t) (rolling % prime[b] + offset[b]);
            }
        }
    }

    if (prefetch) {
        dsv41_engram_prefetch_rows(model.layers[hp.engram_layer_ids[eg]].engram_embd, idx);
    }

    ggml_backend_tensor_set(rows, idx.data(), 0, idx.size()*ggml_element_size(rows));
}

} // namespace

llm_build_deepseek41::rope_cfg llm_build_deepseek41::layer_rope_cfg(int il) const {
    // A layer that reads a compressed stream rotates with the compressor's base and YaRN; a plain
    // sliding window layer rotates with the model's base and no scaling.
    if (hparams.dsv4_compress_ratios[il] == 0) {
        return { freq_base, 1.0f, 0.0f, dsv41_rope_attn_factor(1.0f, 0.0f), 0.0f, 0.0f, 0 };
    }

    return {
        hparams.dsv4_compress_rope_base, freq_scale, ext_factor,
        dsv41_rope_attn_factor(freq_scale, ext_factor), beta_fast, beta_slow, n_ctx_orig,
    };
}

ggml_tensor * llm_build_deepseek41::build_hc_pre(
        ggml_tensor * x,
        ggml_tensor * weights,
        int           il) const {
    GGML_UNUSED(il);
    GGML_ASSERT(x->ne[0] == n_embd);
    GGML_ASSERT(x->ne[1] == hparams.dsv4_hc_mult);

    if (cparams.ssd_hc_fused) {
        return ggml_dsv4_hc_pre(ctx0, x, weights);
    }

    const int64_t hc = hparams.dsv4_hc_mult;
    const int64_t nt = x->ne[2];

    ggml_tensor * result = nullptr;
    for (int64_t ih = 0; ih < hc; ++ih) {
        ggml_tensor * xh = ggml_view_2d(ctx0, x, n_embd, nt, x->nb[2], ih*x->nb[1]);
        ggml_tensor * wh = ggml_view_2d(ctx0, weights, 1, nt, weights->nb[1], ih*weights->nb[0]);
        ggml_tensor * cur = ggml_mul(ctx0, xh, wh);
        result = result ? ggml_add(ctx0, result, cur) : cur;
    }

    return result;
}

ggml_tensor * llm_build_deepseek41::build_hc_sinkhorn(
        ggml_tensor * comb,
        int           il) const {
    GGML_UNUSED(il);

    // comb is [dst_hc, src_hc, n_tokens]. Sinkhorn follows the reference:
    // row softmax over dst, one column normalization, then repeated row/column normalization.
    comb = ggml_soft_max(ctx0, comb);

    ggml_tensor * eps = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, 1);
    eps = ggml_fill(ctx0, eps, hparams.dsv4_hc_eps);

    comb = ggml_add(ctx0, comb, eps);

    auto norm_cols = [&]() {
        ggml_tensor * comb_src_dst = ggml_cont(ctx0, ggml_permute(ctx0, comb, 1, 0, 2, 3));
        ggml_tensor * col_sum = ggml_sum_rows(ctx0, comb_src_dst);
        col_sum = ggml_add(ctx0, col_sum, eps);
        col_sum = ggml_permute(ctx0, col_sum, 1, 0, 2, 3);
        comb = ggml_div(ctx0, comb, col_sum);
    };

    auto norm_rows = [&]() {
        ggml_tensor * row_sum = ggml_sum_rows(ctx0, comb);
        row_sum = ggml_add(ctx0, row_sum, eps);
        comb = ggml_div(ctx0, comb, row_sum);
    };

    norm_cols();
    for (uint32_t i = 1; i < hparams.dsv4_hc_sinkhorn_iters; ++i) {
        norm_rows();
        norm_cols();
    }

    return comb;
}

// derive the pre / post / comb coefficients from the stream, without collapsing it
void llm_build_deepseek41::build_hc_mixes(
        ggml_tensor *  x,
        ggml_tensor *  hc_fn,
        ggml_tensor *  hc_scale,
        ggml_tensor *  hc_base,
        ggml_tensor ** pre,
        ggml_tensor ** post,
        ggml_tensor ** comb,
        int il) const {
    const int64_t hc         = hparams.dsv4_hc_mult;
    const int64_t hc_dim     = hc*n_embd;
    const int64_t hc_mix_dim = (2 + hc)*hc;
    const int64_t nt         = x->ne[2];

    GGML_ASSERT(hc == 4);
    GGML_ASSERT(hc_fn->ne[1] == hc_mix_dim);

    ggml_tensor * flat = ggml_reshape_2d(ctx0, x, hc_dim, nt);
    ggml_tensor * flat_norm = ggml_rms_norm(ctx0, flat, norm_rms_eps);
    ggml_tensor * mixes = ggml_mul_mat(ctx0, hc_fn, flat_norm);
    cb(mixes, "hc_mixes", il);

    ggml_tensor * scale_pre  = dsv41_view_1d(ctx0, hc_scale, 1, 0);
    ggml_tensor * scale_post = dsv41_view_1d(ctx0, hc_scale, 1, 1);

    ggml_tensor * base_pre  = dsv41_view_1d(ctx0, hc_base, hc, 0);
    ggml_tensor * base_post = dsv41_view_1d(ctx0, hc_base, hc, hc);

    *pre = dsv41_view_2d(ctx0, mixes, hc, nt, 0);
    *pre = dsv41_hc_affine(ctx0, *pre, scale_pre, base_pre);
    *pre = ggml_sigmoid(ctx0, *pre);
    *pre = ggml_scale_bias(ctx0, *pre, 1.0f, hparams.dsv4_hc_eps);
    cb(*pre, "hc_pre", il);

    *post = dsv41_view_2d(ctx0, mixes, hc, nt, hc);
    *post = dsv41_hc_affine(ctx0, *post, scale_post, base_post);
    *post = ggml_sigmoid(ctx0, *post);
    *post = ggml_scale(ctx0, *post, 2.0f);
    cb(*post, "hc_post", il);

    if (cparams.ssd_hc_fused) {
        *comb = ggml_dsv4_hc_comb(ctx0, mixes, hc_scale, hc_base, hparams.dsv4_hc_eps,
                (int32_t) hparams.dsv4_hc_sinkhorn_iters);
    } else {
        ggml_tensor * scale_comb = dsv41_view_1d(ctx0, hc_scale, 1, 2);
        ggml_tensor * base_comb  = dsv41_view_1d(ctx0, hc_base, hc*hc, 2*hc);

        *comb = dsv41_view_2d(ctx0, mixes, hc*hc, nt, 2*hc);
        *comb = dsv41_hc_affine(ctx0, *comb, scale_comb, base_comb);
        *comb = ggml_reshape_3d(ctx0, *comb, hc, hc, nt);
        *comb = build_hc_sinkhorn(*comb, il);
    }
    cb(*comb, "hc_comb", il);
}

ggml_tensor * llm_build_deepseek41::build_hc_post(
        ggml_tensor * x,
        ggml_tensor * residual,
        ggml_tensor * post,
        ggml_tensor * comb,
        int il) const {
    GGML_UNUSED(il);
    GGML_ASSERT(x->ne[0] == n_embd);
    GGML_ASSERT(residual->ne[1] == hparams.dsv4_hc_mult);

    if (cparams.ssd_hc_fused) {
        return ggml_dsv4_hc_post(ctx0, x, residual, post, comb);
    }

    const int64_t hc = hparams.dsv4_hc_mult;
    const int64_t nt = x->ne[1];

    ggml_tensor * out = nullptr;
    for (int64_t dst = 0; dst < hc; ++dst) {
        ggml_tensor * post_dst = ggml_view_2d(ctx0, post, 1, nt, post->nb[1], dst*post->nb[0]);
        ggml_tensor * cur = ggml_mul(ctx0, x, post_dst);

        for (int64_t src = 0; src < hc; ++src) {
            ggml_tensor * res_src = ggml_view_2d(ctx0, residual, n_embd, nt, residual->nb[2], src*residual->nb[1]);
            ggml_tensor * comb_src_dst = ggml_view_2d(ctx0, comb, 1, nt, comb->nb[2],
                    dst*comb->nb[0] + src*comb->nb[1]);
            cur = ggml_add(ctx0, cur, ggml_mul(ctx0, res_src, comb_src_dst));
        }

        cur = ggml_reshape_3d(ctx0, cur, n_embd, 1, nt);
        out = out ? ggml_concat(ctx0, out, cur, 1) : cur;
    }

    return out;
}

ggml_tensor * llm_build_deepseek41::build_inp_engram(int il) {
    const int64_t n_cols  = (hparams.engram_max_ngram_size - 1) * hparams.engram_n_head;
    const int64_t key_len = hparams.engram_key_length;

    const auto * mctx_cur = static_cast<const llama_kv_cache_dsv4_context *>(mctx);

    auto inp = std::make_unique<llm_graph_input_engram>(model, mctx_cur->get_raw(), model.engram_index(il), cparams.ssd_engram_prefetch);

    inp->rows = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_cols * n_tokens);
    ggml_set_input(inp->rows);
    ggml_tensor * rows = inp->rows;
    res->add_input(std::move(inp));

    // gather then flatten, laying the buckets out slowest, as the reference does
    ggml_tensor * emb = nullptr;
    if (hparams.engram_e4m3_e8m0_32_row264) {
        ggml_tensor * args[] = { model.layers[il].engram_embd, rows };
        emb = ggml_custom_4d(ctx0, GGML_TYPE_F32, key_len, n_cols, n_tokens, 1,
                             args, 2, dsv41_engram_decode_op, GGML_N_TASKS_MAX, nullptr);
        ggml_backend_sched_set_tensor_backend(sched, rows, backend_cpu);
        ggml_backend_sched_set_tensor_backend(sched, emb, backend_cpu);
    } else {
        emb = ggml_get_rows(ctx0, model.layers[il].engram_embd, rows);
    }
    emb = ggml_reshape_2d(ctx0, emb, key_len * n_cols, n_tokens);
    cb(emb, "engram_embd", il);

    return emb;
}

ggml_tensor * llm_build_deepseek41::build_engram(
        ggml_tensor * x,
        ggml_tensor * emb,
        int il) const {
    const int64_t hc     = hparams.dsv4_hc_mult;
    const int64_t hc_dim = hc*n_embd;
    const int64_t nt     = x->ne[2];

    // one projection makes a key per hc copy plus one value they all share
    ggml_tensor * kv = build_lora_mm(model.layers[il].engram_wkv, emb);
    cb(kv, "engram_kv", il);

    ggml_tensor * key   = ggml_cont(ctx0, ggml_view_2d(ctx0, kv, hc_dim, nt, kv->nb[1], 0));
    ggml_tensor * value = ggml_cont(ctx0, ggml_view_2d(ctx0, kv, n_embd, nt, kv->nb[1], hc_dim*kv->nb[0]));

    // The gate scales reach ggml_mul, which takes only f32, and a file quantized before the
    // quantizer learned to skip them carries them quantized. get_rows dequantizes.
    auto as_f32 = [&](ggml_tensor * w) {
        if (w->type == GGML_TYPE_F32) {
            return w;
        }
        ggml_tensor * ids = ggml_cast(ctx0, ggml_arange(ctx0, 0.0f, (float) w->ne[1], 1.0f), GGML_TYPE_I32);
        return ggml_get_rows(ctx0, w, ids);
    };

    // normalized per (token, hc copy) over n_embd, not jointly over the copies. The reference keeps
    // engram_q and engram_k apart but only ever uses their product, so applying one to each side of
    // the dot product gives the same result.
    auto grouped_norm = [&](ggml_tensor * t, ggml_tensor * w) {
        t = ggml_reshape_3d(ctx0, t, n_embd, hc, nt);
        t = ggml_rms_norm(ctx0, t, norm_rms_eps);
        t = ggml_reshape_2d(ctx0, t, hc_dim, nt);
        t = ggml_mul(ctx0, t, ggml_reshape_2d(ctx0, w, hc_dim, 1));
        return ggml_reshape_3d(ctx0, t, n_embd, hc, nt);
    };

    ggml_tensor * k = grouped_norm(key, as_f32(model.layers[il].engram_k));
    ggml_tensor * q = grouped_norm(x,   as_f32(model.layers[il].engram_q));

    ggml_tensor * s = ggml_sum_rows(ctx0, ggml_mul(ctx0, k, q));
    s = ggml_scale(ctx0, s, 1.0f/sqrtf((float) n_embd));

    // signed square root before the sigmoid, matching the training kernel. The reference uses
    // copysign, which treats +0 as positive where ggml_sgn gives 0, so a dot product of exactly
    // zero gates 0.5 here against 0.50025 there.
    ggml_tensor * mag  = ggml_sqrt(ctx0, ggml_clamp(ctx0, ggml_abs(ctx0, s), 1e-6f, INFINITY));
    ggml_tensor * gate = ggml_sigmoid(ctx0, ggml_mul(ctx0, ggml_sgn(ctx0, s), mag));
    cb(gate, "engram_gate", il);

    // the value is shared across the copies, only the gate differs
    ggml_tensor * v = ggml_reshape_3d(ctx0, value, n_embd, 1, nt);
    v = ggml_repeat_4d(ctx0, v, n_embd, hc, nt, 1);

    return ggml_add(ctx0, x, ggml_mul(ctx0, v, gate));
}

ggml_tensor * llm_build_deepseek41::build_compressed_kv_from_state(
        ggml_tensor * kv_state,
        ggml_tensor * score_state,
        ggml_tensor * state_read_idxs,
        ggml_tensor * comp_pos,
        ggml_tensor * norm,
        int64_t ratio,
        int64_t n_embd_head,
        const char * name,
        int il,
        ggml_tensor ** pre_rope) const {
    const int64_t n_embd_head_rope = hparams.n_rot();
    const int64_t n_embd_head_nope = n_embd_head - n_embd_head_rope;
    const int64_t n_blocks         = comp_pos ? comp_pos->ne[0] : 0;

    GGML_ASSERT(n_blocks > 0);
    GGML_ASSERT(state_read_idxs);
    GGML_ASSERT(state_read_idxs->ne[0] == ratio*n_blocks);
    GGML_ASSERT(n_embd_head >= n_embd_head_rope);

    ggml_tensor * kv = ggml_get_rows(ctx0, kv_state, state_read_idxs);
    kv = ggml_reshape_3d(ctx0, kv, n_embd_head, ratio, n_blocks);
    cb(kv, name, il);

    ggml_tensor * score = ggml_get_rows(ctx0, score_state, state_read_idxs);
    score = ggml_reshape_3d(ctx0, score, n_embd_head, ratio, n_blocks);
    cb(score, name, il);

    ggml_tensor * values = ggml_cont(ctx0, ggml_permute(ctx0, kv, 1, 0, 2, 3));
    ggml_tensor * scores = ggml_cont(ctx0, ggml_permute(ctx0, score, 1, 0, 2, 3));

    ggml_tensor * weights = ggml_soft_max(ctx0, scores);
    ggml_tensor * comp = ggml_mul(ctx0, values, weights);
    comp = ggml_sum_rows(ctx0, comp);
    comp = ggml_cont(ctx0, ggml_permute(ctx0, comp, 1, 0, 2, 3));
    cb(comp, name, il);

    comp = build_norm(comp, norm, nullptr, LLM_NORM_RMS, il);
    cb(comp, name, il);

    // V4.1 derives its index keys from the pooled latent before it is rotated, because the rotated
    // form is what goes into the cache.
    if (pre_rope) {
        *pre_rope = comp;
    }

    comp = ggml_rope_ext(ctx0, comp, comp_pos, nullptr, n_embd_head_rope, rope_type, n_ctx_orig,
            hparams.dsv4_compress_rope_base, freq_scale, ext_factor,
            dsv41_rope_attn_factor(freq_scale, ext_factor), beta_fast, beta_slow);
    comp = ggml_rope_set_offset(comp, n_embd_head_nope);
    cb(comp, name, il);

    return comp;
}

ggml_tensor * llm_build_deepseek41::build_top_k_mask(
        ggml_tensor * kq_mask,
        ggml_tensor * top_k,
        const char * name,
        int il) const {
    GGML_ASSERT(kq_mask);
    GGML_ASSERT(top_k);

    ggml_tensor * kq_mask_all = ggml_fill(ctx0, kq_mask, -INFINITY);
    kq_mask_all = ggml_view_4d(ctx0, kq_mask_all, 1, kq_mask_all->ne[0], kq_mask_all->ne[1], kq_mask_all->ne[3],
            kq_mask_all->nb[0], kq_mask_all->nb[1], kq_mask_all->nb[2], 0);

    ggml_tensor * top_k_3d = ggml_view_4d(ctx0, top_k, top_k->ne[0], top_k->ne[1], top_k->ne[3], 1,
            top_k->nb[1], top_k->nb[2], top_k->ne[3]*top_k->nb[3], 0);

    // f32 rows whatever the mask type: set_rows writes them into kq_mask_all in place, so it has to run
    // where kq_mask_all lives, and Metal only takes f32 rows. Left to the CPU, the scheduler copied the
    // mask before the rows landed and every compressed position stayed masked. 0 is exact in f16.
    ggml_tensor * zeros = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, 1, top_k_3d->ne[0], top_k_3d->ne[1], top_k_3d->ne[2]);
    zeros = ggml_fill(ctx0, zeros, 0.0f);

    ggml_tensor * kq_mask_top_k = ggml_set_rows(ctx0, kq_mask_all, zeros, top_k_3d);
    kq_mask_top_k = ggml_view_4d(ctx0, kq_mask_top_k,
            kq_mask_top_k->ne[1], kq_mask_top_k->ne[2], 1, kq_mask_top_k->ne[3],
            kq_mask_top_k->nb[2], kq_mask_top_k->nb[3], kq_mask_top_k->nb[3], 0);

    kq_mask_top_k = ggml_add(ctx0, kq_mask_top_k, kq_mask);
    cb(kq_mask_top_k, name, il);

    return kq_mask_top_k;
}

ggml_tensor * llm_build_deepseek41::build_raw_attention(
        llm_graph_input_dsv4_raw * inp_attn,
        ggml_tensor * q,
        ggml_tensor * kv,
        ggml_tensor * sinks,
        float kq_scale,
        int il) const {
    GGML_ASSERT(hparams.is_swa(il));

    ggml_tensor * k_rot = inp_attn->self_k_rot;

    if (k_rot) {
        q  = llama_mul_mat_hadamard(ctx0, q, k_rot);
        kv = llama_mul_mat_hadamard(ctx0, kv, k_rot);
    }

    ggml_build_forward_expand(gf, q);
    ggml_build_forward_expand(gf, kv);

    const llama_kv_cache_dsv4_raw_context * mctx_cur = inp_attn->mctx;

    ggml_build_forward_expand(gf, mctx_cur->cpy_k(ctx0, kv, inp_attn->get_k_idxs(), il));

    ggml_tensor * kq_mask = inp_attn->get_kq_mask();

    ggml_tensor * k = mctx_cur->get_k(ctx0, il);

    ggml_tensor * out = build_attn_mha(q, k, k, nullptr, kq_mask, sinks, nullptr, 0, kq_scale, il);
    if (k_rot) {
        out = llama_mul_mat_hadamard(ctx0, out, k_rot);
    }
    cb(out, "attn_raw", il);

    return out;
}

// Undo the rotation the query carried into attention, then the grouped output projection. wo_a is
// block diagonal over groups, each projecting only its own heads, hence a batched mul_mat.
ggml_tensor * llm_build_deepseek41::build_attention_tail(
        ggml_tensor * out,
        ggml_tensor * inp_pos,
        int64_t nt,
        int il) const {
    const auto & layer = model.layers[il];

    const int64_t n_embd_head      = hparams.n_embd_head_k();
    const int64_t n_embd_head_rope = hparams.n_rot();
    const int64_t n_embd_head_nope = n_embd_head - n_embd_head_rope;
    const int64_t n_groups         = hparams.dsv4_o_group_count;
    const int64_t o_lora_rank      = hparams.dsv4_o_lora_rank;
    const int64_t o_group_dim      = (n_head/n_groups)*n_embd_head;

    const rope_cfg rc = layer_rope_cfg(il);

    out = ggml_reshape_3d(ctx0, out, n_embd_head, n_head, nt);
    out = ggml_rope_ext_back(ctx0, out, inp_pos, nullptr, n_embd_head_rope, rope_type, rc.n_ctx_orig,
            rc.base, rc.scale, rc.ext_factor, rc.attn_factor, rc.beta_fast, rc.beta_slow);
    out = ggml_rope_set_offset(out, n_embd_head_nope);
    cb(out, "attn_derope", il);

    out = ggml_reshape_3d(ctx0, out, o_group_dim, n_groups, nt);
    out = ggml_permute(ctx0, out, 0, 2, 1, 3);

    // the loader keeps wo_a as the file stores it, (o_group_dim, o_lora_rank * n_groups)
    ggml_tensor * wo_a = ggml_reshape_3d(ctx0, layer.wo_a, layer.wo_a->ne[0], o_lora_rank, n_groups);

    ggml_tensor * oa = ggml_mul_mat(ctx0, wo_a, out);
    cb(oa, "attn_wo_a", il);

    oa = ggml_permute(ctx0, oa, 0, 2, 1, 3);
    oa = ggml_cont_2d(ctx0, oa, o_lora_rank*n_groups, nt);

    out = build_lora_mm(layer.wo_b, oa);
    cb(out, "attn_out", il);

    return out;
}

// Queries per indexer tile. The scores hold n_comp x n_idx_head f32 per query twice (the product and
// its permuted copy). Size tiles for the largest compressed width in the current reserve horizon,
// so the reserve graph also covers shorter contexts in the same phase.
static int64_t dsv41_indexer_tile_tokens(int64_t n_comp, int64_t n_idx_head, int64_t n_tokens) {
    constexpr int64_t budget = int64_t(256) << 20;
    const int64_t per_token = 2*n_comp*n_idx_head*int64_t(sizeof(float));
    const int64_t fit = budget/std::max<int64_t>(per_token, 1);
    if (fit >= n_tokens) {
        return n_tokens;
    }
    // whole multiples of 64 keep every tile on the same matmul path as a full ubatch
    return std::min<int64_t>(n_tokens, std::max<int64_t>(64, fit/64*64));
}

// Score this layer's queries against the shared index keys and open the best compressed positions
// in the compressed-attention mask. The keys were published by an earlier layer, so this only
// builds the query side.
ggml_tensor * llm_build_deepseek41::build_indexer_mask(
        llm_graph_input_dsv4 * inp_dsv4,
        const llm_graph_input_dsv4::comp_input & inp_comp,
        ggml_tensor * qr,
        ggml_tensor * cur,
        ggml_tensor * inp_pos,
        int64_t * n_top_k_out,
        int il) const {
    const auto & layer = model.layers[il];

    const int64_t n_idx_head      = hparams.indexer_n_head;
    const int64_t n_idx_head_dim  = hparams.indexer_head_size;
    const int64_t n_idx_head_rope = hparams.n_rot();
    const int64_t n_idx_head_nope = n_idx_head_dim - n_idx_head_rope;
    const int64_t nt              = cur->ne[1];

    GGML_ASSERT(inp_comp.kq_mask);
    GGML_ASSERT(n_idx_head_dim >= n_idx_head_rope);

    ggml_tensor * idx_q = build_lora_mm(layer.indexer_attn_q_b, qr);
    idx_q = ggml_reshape_3d(ctx0, idx_q, n_idx_head_dim, n_idx_head, nt);
    idx_q = ggml_rope_ext(ctx0, idx_q, inp_pos, nullptr, n_idx_head_rope, rope_type, n_ctx_orig,
            hparams.dsv4_compress_rope_base, freq_scale, ext_factor,
            dsv41_rope_attn_factor(freq_scale, ext_factor), beta_fast, beta_slow);
    idx_q = ggml_rope_set_offset(idx_q, n_idx_head_nope);
    cb(idx_q, "idx_q", il);

    ggml_tensor * idx_k_rot = inp_dsv4->get_lid().k_rot;
    if (idx_k_rot) {
        idx_q = llama_mul_mat_hadamard(ctx0, idx_q, idx_k_rot);
        cb(idx_q, "idx_q_rot", il);
    }

    // one weight per head, scaled so the score matches the reference's softmax_scale * n_heads**-0.5
    ggml_tensor * idx_w = build_lora_mm(layer.indexer_proj, cur);
    idx_w = ggml_scale(ctx0, idx_w, 1.0f/sqrtf(float(n_idx_head_dim*n_idx_head)));
    cb(idx_w, "idx_weights", il);

    ggml_tensor * idx_k = inp_dsv4->mctx->get_lid()->get_k(ctx0, il);
    const int64_t n_comp_capacity = idx_k->ne[2]; // keep the capacity before slicing to the current width

    const int64_t n_comp = inp_comp.kq_mask->ne[0];
    GGML_ASSERT(n_comp > 0);
    GGML_ASSERT(n_comp <= idx_k->ne[2]);

    idx_k = ggml_view_4d(ctx0, idx_k,
            idx_k->ne[0], idx_k->ne[1], n_comp, idx_k->ne[3],
            idx_k->nb[1], idx_k->nb[2], idx_k->nb[3], 0);
    cb(idx_k, "idx_k", il);

    const int64_t n_stream = idx_k->ne[3];

    const uint32_t n_top_k = n_comp < hparams.indexer_top_k ? n_comp : hparams.indexer_top_k;
    *n_top_k_out = n_top_k;

    // below topk_blocks*block_size rows every block is a candidate and the filter changes nothing
    const bool candidate = n_stream == 1 &&
        il >= DSV41_CANDIDATE_SOURCE && DSV41_CANDIDATE_SOURCE < int32_t(n_layer) &&
        hparams.dsv41_is_index_source(DSV41_CANDIDATE_SOURCE) &&
        hparams.dsv4_compress_ratios[il] == hparams.dsv4_compress_ratios[DSV41_CANDIDATE_SOURCE] &&
        n_comp > DSV41_CANDIDATE_TOPK_BLOCKS*DSV41_CANDIDATE_BLOCK_SIZE;
    if (candidate && il == DSV41_CANDIDATE_SOURCE) {
        // every query tile writes its rows of the block scores here; the candidates are picked once afterwards
        candidate_scores = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32,
                (n_comp + DSV41_CANDIDATE_BLOCK_SIZE - 1)/DSV41_CANDIDATE_BLOCK_SIZE, nt);
    }

    // the best n_top_k compressed positions for the queries in q (rows of w, mask)
    auto score_top_k = [&](ggml_tensor * q, ggml_tensor * w, ggml_tensor * mask, int64_t i0) {
        q = ggml_view_4d(ctx0, q,
                q->ne[0], q->ne[1], q->ne[2]/n_stream, n_stream,
                q->nb[1], q->nb[2], q->nb[3]/n_stream, 0);
        w = ggml_view_4d(ctx0, w,
                w->ne[0], w->ne[1]/n_stream, w->ne[2], n_stream,
                w->nb[1], w->nb[2]/n_stream, w->nb[3]/n_stream, 0);

        q = ggml_permute(ctx0, q, 0, 2, 1, 3);
        ggml_tensor * k = ggml_permute(ctx0, idx_k, 0, 2, 1, 3);

        ggml_tensor * score = ggml_mul_mat(ctx0, k, q);
        score = ggml_cont(ctx0, ggml_permute(ctx0, score, 2, 1, 0, 3));

        score = ggml_relu(ctx0, score);
        score = ggml_mul(ctx0, score, w);
        score = ggml_sum_rows(ctx0, score);
        score = ggml_cont(ctx0, ggml_permute(ctx0, score, 2, 1, 0, 3));

        // the attention mask is F16 when flash attention is on, and this score is F32. the mask only
        // ever holds 0 or -inf, so widening it is exact.
        if (mask->type != score->type) {
            mask = ggml_cast(ctx0, mask, score->type);
        }

        ggml_tensor * raw = score;
        score = ggml_add(ctx0, score, mask);
        cb(score, "idx_score", il);

        if (candidate) {
            if (il == DSV41_CANDIDATE_SOURCE) {
                ggml_tensor * block_scores = dsv41_candidate_block_scores(ctx0, raw, mask, DSV41_CANDIDATE_BLOCK_SIZE);
                ggml_tensor * rows = ggml_view_2d(ctx0, candidate_scores, candidate_scores->ne[0], block_scores->ne[1],
                        candidate_scores->nb[1], i0*candidate_scores->nb[1]);
                ggml_build_forward_expand(gf, ggml_cpy(ctx0, block_scores, rows));
            } else {
                GGML_ASSERT(candidate_blocks != nullptr && candidate_blocks->ne[1] >= i0 + score->ne[1]);
                ggml_tensor * blocks = ggml_view_2d(ctx0, candidate_blocks, candidate_blocks->ne[0], score->ne[1],
                        candidate_blocks->nb[1], i0*candidate_blocks->nb[1]);
                score = dsv41_apply_candidates(ctx0, score, blocks, DSV41_CANDIDATE_BLOCK_SIZE);
                cb(score, "idx_score_candidates", il);
            }
        }

        ggml_tensor * top_k = ggml_cont(ctx0, ggml_top_k(ctx0, score, n_top_k));
        cb(top_k, "idx_top_k", il);

        return top_k;
    };

    const int64_t ratio = hparams.dsv4_compress_ratios[il];
    const int64_t max_comp = GGML_PAD((n_ctx_reserve + ratio - 1)/ratio, 256);
    const int64_t n_comp_tile = std::max<int64_t>(n_comp, std::min<int64_t>(n_comp_capacity, max_comp));
    const int64_t n_tile = n_stream == 1 ? dsv41_indexer_tile_tokens(n_comp_tile, n_idx_head, nt) : nt;
    // the source's scores are complete once its last tile is scheduled; its readers take the candidates from here
    auto pick_candidates = [&]() {
        if (candidate && il == DSV41_CANDIDATE_SOURCE) {
            candidate_blocks = ggml_top_k(ctx0, candidate_scores, int(DSV41_CANDIDATE_TOPK_BLOCKS));
            cb(candidate_blocks, "idx_candidate_blocks", il);
            ggml_build_forward_expand(gf, candidate_blocks);
        }
    };

    if (n_tile >= nt) {
        ggml_tensor * result = build_top_k_mask(inp_comp.kq_mask, score_top_k(idx_q, idx_w, inp_comp.kq_mask, 0), "comp_top_k_mask", il);
        pick_candidates();
        return result;
    }

    ggml_tensor * kq_mask = inp_comp.kq_mask;
    ggml_tensor * out = nullptr;
    for (int64_t i0 = 0; i0 < nt; i0 += n_tile) {
        const int64_t n = std::min(n_tile, nt - i0);

        ggml_tensor * q = ggml_view_3d(ctx0, idx_q, idx_q->ne[0], idx_q->ne[1], n,
                idx_q->nb[1], idx_q->nb[2], i0*idx_q->nb[2]);
        ggml_tensor * w = ggml_view_2d(ctx0, idx_w, idx_w->ne[0], n,
                idx_w->nb[1], i0*idx_w->nb[1]);
        ggml_tensor * mask = ggml_view_4d(ctx0, kq_mask, kq_mask->ne[0], n, 1, 1,
                kq_mask->nb[1], n*kq_mask->nb[1], n*kq_mask->nb[1], i0*kq_mask->nb[1]);

        ggml_tensor * tile = build_top_k_mask(mask, score_top_k(q, w, mask, i0), "comp_top_k_mask", il);
        out = out ? ggml_concat(ctx0, out, tile, 1) : tile;
        if (candidate && il == DSV41_CANDIDATE_SOURCE) {
            // finish this tile before the next one starts, so that one tile's scores are live at a time
            ggml_build_forward_expand(gf, out);
        }
    }

    pick_candidates();
    return out;
}

// DeepSeek-V4.1 attention: a sliding window of raw KV, plus, where the layer uses one, the
// compressed positions the indexer picked, concatenated into a single masked attention.
//
// Only a source layer compresses. The layers after it read the same rows, which the KV cache
// hands them through the reuse callback, so a reader builds no compressor at all.
ggml_tensor * llm_build_deepseek41::build_attention_v41(
        llm_graph_input_dsv4 * inp_dsv4,
        ggml_tensor * cur,
        ggml_tensor * inp_pos,
        int il) const {
    const auto & layer = model.layers[il];
    llm_graph_input_dsv4_raw * inp_attn = inp_dsv4->get_raw();

    const int64_t n_embd_head      = hparams.n_embd_head_k();
    const int64_t n_embd_head_rope = hparams.n_rot();
    const int64_t nt               = cur->ne[1];
    const int64_t ratio            = hparams.dsv4_compress_ratios[il];

    GGML_ASSERT(n_embd_head == n_embd_head_v);
    GGML_ASSERT(n_head % hparams.dsv4_o_group_count == 0);

    const rope_cfg rc = layer_rope_cfg(il);

    // Query. V4 normalizes again after wq_b; V4.1 normalizes only the low rank part.
    ggml_tensor * qr = build_lora_mm(layer.wq_a, cur);
    qr = build_norm(qr, layer.attn_q_a_norm, nullptr, LLM_NORM_RMS, il);
    cb(qr, "qr", il);

    ggml_tensor * q = build_lora_mm(layer.wq_b, qr);
    q = ggml_reshape_3d(ctx0, q, n_embd_head, n_head, nt);
    q = ggml_rope_ext(ctx0, q, inp_pos, nullptr, n_embd_head_rope, rope_type, rc.n_ctx_orig,
            rc.base, rc.scale, rc.ext_factor, rc.attn_factor, rc.beta_fast, rc.beta_slow);
    q = ggml_rope_set_offset(q, n_embd_head - n_embd_head_rope);
    cb(q, "q", il);

    // the sliding window KV, which every layer keeps for itself
    ggml_tensor * kv = build_lora_mm(layer.wkv, cur);
    kv = build_norm(kv, layer.attn_kv_norm, nullptr, LLM_NORM_RMS, il);
    kv = ggml_reshape_3d(ctx0, kv, n_embd_head, 1, nt);
    kv = ggml_rope_ext(ctx0, kv, inp_pos, nullptr, n_embd_head_rope, rope_type, rc.n_ctx_orig,
            rc.base, rc.scale, rc.ext_factor, rc.attn_factor, rc.beta_fast, rc.beta_slow);
    kv = ggml_rope_set_offset(kv, n_embd_head - n_embd_head_rope);
    cb(kv, "kv", il);

    const float kq_scale = 1.0f/sqrtf(float(n_embd_head));

    ggml_tensor * out = nullptr;

    if (ratio == 0) {
        // no compressed stream, so this layer sees only its own window
        out = build_raw_attention(inp_attn, q, kv, layer.attn_sinks, kq_scale, il);

        return build_attention_tail(out, inp_pos, nt, il);
    }

    // The plan slot follows the ratio, since a plan encodes how many tokens make a row. The rows
    // themselves always live in the CSA cache, and the index keys in the indexer cache, whichever
    // plan produced them.
    const bool use_csa = (uint32_t) ratio == inp_dsv4->mctx->get_csa_state()->get_ratio();

    const auto & inp_comp = use_csa ? inp_dsv4->get_csa() : inp_dsv4->get_hca();

    const llama_dsv4_comp_state * comp_state = use_csa
        ? inp_dsv4->mctx->get_csa_state()
        : inp_dsv4->mctx->get_hca_state();

    GGML_ASSERT(inp_comp.kq_mask && "a compressed layer needs a plan for its ratio");

    if (hparams.dsv41_is_kv_source(il) && inp_comp.state_pos) {
        ggml_tensor * state_kv = build_lora_mm(layer.attn_comp_wkv, cur);
        cb(state_kv, "comp_state_kv", il);

        // At ratio 1 there is nothing to pool and the file carries no gate. The softmax below then
        // runs over a single element and returns 1.0 whatever the score holds, so the values reach
        // the cache unweighted, which is what a plain projection means.
        ggml_tensor * state_score = layer.attn_comp_wgate
            ? build_lora_mm(layer.attn_comp_wgate, cur)
            : state_kv;
        cb(state_score, "comp_state_score", il);

        const dsv41_state_tensors restored = dsv41_build_state_restore(ctx0, inp_comp, comp_state, il);

        ggml_tensor * base_kv = dsv41_view_2d(
                ctx0, restored.kv, restored.kv->ne[0], comp_state->get_n_rows(), 0);
        ggml_tensor * base_score = dsv41_view_2d(
                ctx0, restored.score, restored.score->ne[0], comp_state->get_n_rows(), 0);

        ggml_tensor * source_kv    = ggml_concat(ctx0, base_kv,    state_kv,    1);
        ggml_tensor * source_score = ggml_concat(ctx0, base_score, state_score, 1);

        // the indexer reads the latent before it is rotated, so ask for both forms at once
        ggml_tensor * latent_pre = nullptr;

        ggml_tensor * latent = build_compressed_kv_from_state(
                source_kv,
                source_score,
                inp_comp.state_read_idxs,
                inp_comp.state_write_pos,
                layer.attn_comp_norm,
                ratio,
                n_embd_head,
                "comp_kv",
                il,
                &latent_pre);

        if (hparams.dsv41_owns_index_k(il)) {
            const int64_t n_idx_head_dim  = hparams.indexer_head_size;
            const int64_t n_idx_head_rope = hparams.n_rot();

            ggml_tensor * idx_k = build_lora_mm(layer.indexer_attn_k, latent_pre);
            idx_k = build_norm(idx_k, layer.indexer_k_norm, nullptr, LLM_NORM_RMS, il);
            idx_k = ggml_rope_ext(ctx0, idx_k, inp_comp.state_write_pos, nullptr, n_idx_head_rope,
                    rope_type, n_ctx_orig, hparams.dsv4_compress_rope_base, freq_scale, ext_factor,
                    dsv41_rope_attn_factor(freq_scale, ext_factor), beta_fast, beta_slow);
            idx_k = ggml_rope_set_offset(idx_k, n_idx_head_dim - n_idx_head_rope);
            cb(idx_k, "idx_k_new", il);

            if (inp_dsv4->get_lid().k_rot) {
                idx_k = llama_mul_mat_hadamard(ctx0, idx_k, inp_dsv4->get_lid().k_rot);
            }

            ggml_build_forward_expand(gf, inp_dsv4->mctx->get_lid()->cpy_k(
                        ctx0, idx_k, inp_comp.state_write_idxs, il));
        }

        if (inp_dsv4->get_csa().k_rot) {
            latent = llama_mul_mat_hadamard(ctx0, latent, inp_dsv4->get_csa().k_rot);
            cb(latent, "comp_kv_rot", il);
        }

        ggml_build_forward_expand(gf, inp_dsv4->mctx->get_csa()->cpy_k(
                    ctx0, latent, inp_comp.state_write_idxs, il));

        // carry whatever did not complete a row into the next ubatch
        ggml_tensor * snapshot_kv    = ggml_concat(ctx0, restored.kv,    state_kv,    1);
        ggml_tensor * snapshot_score = ggml_concat(ctx0, restored.score, state_score, 1);

        const dsv41_state_tensors snapshot = dsv41_build_state_snapshot(
                ctx0, inp_comp, comp_state, snapshot_kv, snapshot_score, il);
        if (snapshot.kv != nullptr) {
            ggml_build_forward_expand(gf, snapshot.kv);
        }
        if (snapshot.score != nullptr) {
            ggml_build_forward_expand(gf, snapshot.score);
        }

        ggml_tensor * persist_kv    = ggml_get_rows(ctx0, state_kv,    inp_comp.state_persist_src_idxs);
        ggml_tensor * persist_score = ggml_get_rows(ctx0, state_score, inp_comp.state_persist_src_idxs);

        ggml_build_forward_expand(gf, comp_state->cpy_kv(
                    ctx0, persist_kv, inp_comp.state_persist_dst_idxs, il));
        ggml_build_forward_expand(gf, comp_state->cpy_score(
                    ctx0, persist_score, inp_comp.state_persist_dst_idxs, il));
    }

    // an index source picks the positions; the layers in between reuse what it picked
    ggml_tensor * top_k_mask = nullptr;
    int64_t n_top_k = 0;
    if (hparams.dsv41_is_index_source(il)) {
        top_k_mask = build_indexer_mask(inp_dsv4, inp_comp, qr, cur, inp_pos, &n_top_k, il);
        top_k_masks[il]  = top_k_mask;
        top_k_counts[il] = n_top_k;
    } else if (hparams.dsv41_topk_source[il] >= 0) {
        // the positions of a source's selection are rows of its compressed stream, so a reader may only take
        // them when it reads a stream of the same ratio from the same KV source
        const int32_t source = hparams.dsv41_topk_source[il];
        GGML_ASSERT(source >= 0 && source < il && hparams.dsv41_is_index_source(source));
        GGML_ASSERT(hparams.dsv4_compress_ratios[source] == hparams.dsv4_compress_ratios[il]);
        GGML_ASSERT(hparams.dsv41_kv_source[il] == hparams.dsv41_kv_source[source]);
        top_k_mask = top_k_masks[source];
        n_top_k    = top_k_counts[source];
        GGML_ASSERT(top_k_mask != nullptr && "an index source must be built before its readers");
        GGML_ASSERT(top_k_mask->ne[0] == inp_comp.kq_mask->ne[0] && top_k_mask->ne[1] == inp_comp.kq_mask->ne[1] &&
                    top_k_mask->ne[2] == inp_comp.kq_mask->ne[2] && top_k_mask->ne[3] == inp_comp.kq_mask->ne[3]);
    }

    ggml_tensor * k_rot = inp_attn->self_k_rot;
    if (k_rot) {
        q  = llama_mul_mat_hadamard(ctx0, q, k_rot);
        kv = llama_mul_mat_hadamard(ctx0, kv, k_rot);
    }

    ggml_build_forward_expand(gf, q);
    ggml_build_forward_expand(gf, kv);

    const llama_kv_cache_dsv4_raw_context * mctx_raw = inp_attn->mctx;

    ggml_build_forward_expand(gf, mctx_raw->cpy_k(ctx0, kv, inp_attn->get_k_idxs(), il));

    ggml_tensor * raw_k = mctx_raw->get_k(ctx0, il);
    cb(raw_k, "raw_k", il);

    ggml_tensor * comp_k = inp_dsv4->mctx->get_csa()->get_k(ctx0, il);

    const int64_t n_comp = inp_comp.kq_mask->ne[0];
    GGML_ASSERT(n_comp > 0);
    GGML_ASSERT(n_comp <= comp_k->ne[2]);

    comp_k = ggml_view_4d(ctx0, comp_k,
            comp_k->ne[0], comp_k->ne[1], n_comp, comp_k->ne[3],
            comp_k->nb[1], comp_k->nb[2], comp_k->nb[3], 0);
    cb(comp_k, "comp_k", il);

    ggml_tensor * k_all = ggml_concat(ctx0, raw_k, comp_k, 2);
    cb(k_all, "k_all", il);

    ggml_tensor * raw_mask  = inp_attn->get_kq_mask();
    ggml_tensor * comp_mask = top_k_mask ? top_k_mask : inp_comp.kq_mask;

    ggml_tensor * kq_mask = ggml_concat(ctx0, raw_mask, comp_mask, 0);
    cb(kq_mask, "kq_mask", il);

    const int64_t n_kv_max = top_k_mask
        ? std::min<int64_t>(raw_mask->ne[0], hparams.n_swa) + n_top_k
        : 0;

    out = build_attn_mha(q, k_all, k_all, nullptr, kq_mask, layer.attn_sinks,
            nullptr, n_kv_max, kq_scale, il);
    if (k_rot) {
        out = llama_mul_mat_hadamard(ctx0, out, k_rot);
    }
    cb(out, "attn_out_raw", il);

    return build_attention_tail(out, inp_pos, nt, il);
}

llm_build_deepseek41::llm_build_deepseek41(const llama_model & model, const llm_graph_params & params) :
    llm_graph_context(params), model(model) {
    ggml_tensor * cur;

    ggml_tensor * inp = build_inp_embd(model.tok_embd);
    ggml_tensor * inp_pos = build_inp_pos();
    ggml_tensor * inp_out_ids = build_inp_out_ids();
    llm_graph_input_dsv4 * inp_dsv4 = build_inp_dsv4();
    llm_graph_input_dsv4_raw * inp_attn = inp_dsv4->get_raw();
    ggml_build_forward_expand(gf, inp_attn->self_kq_mask);

    const int64_t hc = hparams.dsv4_hc_mult;
    ggml_tensor * inpL = ggml_reshape_3d(ctx0, inp, n_embd, 1, n_tokens);
    inpL = ggml_repeat_4d(ctx0, inpL, n_embd, hc, n_tokens, 1);
    cb(inpL, "hc_init", -1);

    // Layer 0 has no previous sublayer to take a mix from, so the reference hands it a one-hot that
    // selects the first copy.
    ggml_tensor * pre_mix = ggml_concat(ctx0,
            ggml_fill(ctx0, ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, 1,      n_tokens), 1.0f),
            ggml_fill(ctx0, ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, hc - 1, n_tokens), 0.0f), 0);
    cb(pre_mix, "hc_pre_init", -1);

    for (int il = 0; il < n_layer; ++il) {
        // the engram sits before the block and writes straight into the stream
        if (model.engram_index(il) >= 0) {
            inpL = build_engram(inpL, build_inp_engram(il), il);
            cb(inpL, "engram_out", il);
        }

        ggml_tensor * residual = inpL;
        ggml_tensor * attn_pre = nullptr;
        ggml_tensor * post     = nullptr;
        ggml_tensor * comb     = nullptr;

        // this sublayer's mixes are for the next one, so the collapse uses the incoming mix
        build_hc_mixes(inpL,
                model.layers[il].hc_attn_fn,
                model.layers[il].hc_attn_scale,
                model.layers[il].hc_attn_base,
                &attn_pre, &post, &comb, il);

        cur = build_hc_pre(inpL, pre_mix, il);
        cb(cur, "hc_attn_pre", il);

        cur = build_norm(cur, model.layers[il].attn_norm, nullptr, LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);

        cur = build_attention_v41(inp_dsv4, cur, inp_pos, il);

        inpL = build_hc_post(cur, residual, post, comb, il);
        cb(inpL, "hc_attn_post", il);

        residual = inpL;

        // the FFN mix is what the next layer's attention collapses with
        build_hc_mixes(inpL,
                model.layers[il].hc_ffn_fn,
                model.layers[il].hc_ffn_scale,
                model.layers[il].hc_ffn_base,
                &pre_mix, &post, &comb, il);

        cur = build_hc_pre(inpL, attn_pre, il);
        cb(cur, "hc_ffn_pre", il);
        ggml_tensor * ffn_pre = cur;

        ggml_build_forward_expand(gf, residual);
        ggml_build_forward_expand(gf, post);
        ggml_build_forward_expand(gf, comb);

        cur = build_norm(cur, model.layers[il].ffn_norm, nullptr, LLM_NORM_RMS, il);
        cb(cur, "ffn_norm", il);

        const auto & layer = model.layers[il];
        ggml_tensor * exp_probs_b = layer.ffn_exp_probs_b;

        // an embedding batch (mtmd) routes with the vision bias when the file has one
        if (ubatch.embd != nullptr && layer.ffn_exp_probs_b_vl) {
            exp_probs_b = layer.ffn_exp_probs_b_vl;
        }

        if (il + 1 < n_layer) {
            const auto & next = model.layers[il + 1];
            cpu_sink_prefetch_hint = build_moe_prefetch_hint(ffn_pre, next.ffn_norm, next.ffn_gate_inp,
                next.ffn_exp_probs_b, (llama_expert_gating_func_type) hparams.expert_gating_func, n_expert, il);
        }

        ggml_tensor * moe_out = build_moe_ffn(cur,
                layer.ffn_gate_inp,
                layer.ffn_up_exps,
                layer.ffn_gate_exps,
                layer.ffn_down_exps,
                exp_probs_b,
                n_expert, n_expert_used,
                LLM_FFN_SILU, hparams.expert_weights_norm,
                hparams.expert_weights_scale,
                (llama_expert_gating_func_type) hparams.expert_gating_func,
                il);
        cb(moe_out, "ffn_moe_out", il);
        cpu_sink_prefetch_hint = nullptr;

        ggml_tensor * ffn_shexp = build_ffn(cur,
                layer.ffn_up_shexp,   nullptr, nullptr,
                layer.ffn_gate_shexp, nullptr, nullptr,
                layer.ffn_down_shexp, nullptr, nullptr,
                nullptr, LLM_FFN_SILU, LLM_FFN_PAR, il);
        cb(ffn_shexp, "ffn_shexp", il);

        cur = ggml_add(ctx0, moe_out, ffn_shexp);
        cb(cur, "ffn_out", il);

        inpL = build_hc_post(cur, residual, post, comb, il);
        inpL = build_cvec(inpL, il);
        cb(inpL, "l_last", il);
    }

    if (inp_out_ids) {
        ggml_tensor * flat = ggml_reshape_2d(ctx0, inpL, n_embd*hc, n_tokens);
        inpL = ggml_reshape_3d(ctx0, ggml_get_rows(ctx0, flat, inp_out_ids), n_embd, hc, n_outputs);
        pre_mix = ggml_get_rows(ctx0, pre_mix, inp_out_ids);
    }

    // The last layer's FFN mix is the one nothing has consumed, and it collapses the copies here.
    cur = build_hc_pre(inpL, pre_mix, -1);
    cb(cur, "hc_out", -1);

    cur = build_norm(cur, model.output_norm, nullptr, LLM_NORM_RMS, -1);
    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    cur = ggml_mul_mat(ctx0, model.output, cur);
    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}
