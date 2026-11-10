#pragma once

#include "llama.h"
#include "llama-ext.h"
#include "llama-cparams.h"
#include "llama-graph.h"
#include "llama-adapter.h"
#include "llama-impl.h"
#include "ssd/options.h"

#include "ggml-cpp.h"
#include "ggml-opt.h"

#include <functional>
#include <map>
#include <string>
#include <vector>

struct llama_model;
class llama_batch_allocr;

class llama_io_read_i;
class llama_io_write_i;

// "memory" as in abstract memory for the context
struct llama_memory_i;
struct llama_memory_context_i;

// The prefill route of one ubatch of n_tokens tokens: the CPU sink's grouped prefill when the sink is on and the ubatch
// has at most max_tokens tokens, or when the GPU would not take a MUL_MAT_ID of that size off the CPU (the scheduler
// would compute it from the mmap'd experts on the CPU); the ggml graph on the GPU otherwise, which copies every layer's
// experts to the device.
inline bool llama_cpu_sink_prefill_route(bool sink, uint32_t max_tokens, uint32_t n_tokens, bool gpu_offloads) {
    return sink && (n_tokens <= max_tokens || !gpu_offloads);
}

// The expert copies of one prefill's graph-route passes (the [moe_prefill_copy] line): begin when the prefill starts,
// pass for each ubatch on the graph route, end when the prefill ends with the scheduler's counters then.  end returns
// the line, without a newline, and nothing for a prefill that had no graph-route pass.
struct llama_prefill_copy_meter {
    uint32_t passes = 0;
    uint64_t tokens = 0;
    ggml_backend_sched_kimi_copy_stats start = {};

    void begin(const ggml_backend_sched_kimi_copy_stats & now) { passes = 0; tokens = 0; start = now; }
    void pass(uint32_t n_tokens) { ++passes; tokens += n_tokens; }
    std::string end(const ggml_backend_sched_kimi_copy_stats & now);
};

// The copies of the weights homed on the drive in one phase of a request (the [dense_stream] line): the prefill from
// the request's start to its end, the decode from there to the request's end.  begin when the phase starts, add the
// tokens of each ubatch, end when it ends with the scheduler's counters then.  end returns the line, without a
// newline, and nothing for a phase that copied no such weight.
struct llama_dense_stream_meter {
    const char * phase = nullptr;       // nullptr: between requests
    uint64_t tokens = 0;
    ggml_backend_sched_kimi_copy_stats start = {};

    void begin(const char * p, const ggml_backend_sched_kimi_copy_stats & now) { phase = p; tokens = 0; start = now; }
    void add(uint32_t n_tokens) { tokens += n_tokens; }
    std::string end(const ggml_backend_sched_kimi_copy_stats & now);
};

// Whether one of the GPU devices gpus would compute a MUL_MAT_ID of n_tokens tokens over each of the expert tensors
// experts, weights in host memory, instead of the CPU: the scheduler's test for such an op
// (ggml_backend_sched_backend_id_from_cur), that the device supports it and offloads it, the latter by its batch
// (ne[2]).  An expert tensor the GPU does not take would be computed from the mmap'd experts on the CPU.
bool llama_gpu_offloads_mul_mat_id(const std::vector<ggml_backend_dev_t> & gpus,
        const std::vector<const ggml_tensor *> & experts, int64_t n_expert_used, uint32_t n_tokens);

// The ubatch of a request's prefill graph when a graph of the request's own size, target, does not fit (graphs sized
// to the request).  Every pass of a smaller ubatch reads its layers' experts again, so the search takes the largest
// size that fits on a route that can serve the request.  When the target takes the ggml graph: the graph's sizes,
// those above the threshold that the GPU offloads.  When none of them fits: the sink at the target, since one sink
// pass reads each expert once and a split reads the union again at the same compute; for that the request's
// threshold max_sink_tokens is raised to the target.  Then smaller sink sizes.  The search is binary, in steps of
// the granularity, at least 256 tokens and at most half the target, and ends with one token.
// sink_route(n, max_sink_tokens): whether a ubatch of n tokens takes the sink at that threshold; reserve(n):
// reserve the graph of n tokens at the threshold in max_sink_tokens.  Returns the selected ubatch, 0 if none fits.
uint32_t llama_prefill_ubatch_search(uint32_t target, uint32_t granularity, uint32_t & max_sink_tokens,
        const std::function<bool(uint32_t, uint32_t)> & sink_route, const std::function<bool(uint32_t)> & reserve);

struct llama_context {
    // init scheduler and compute buffers, reserve worst-case graphs
    llama_context(
            const llama_model & model,
                  llama_context_params params);

    ~llama_context();

    // reserve a new backend scheduler (if needed)
    // for example, when:
    //   - changing loras
    //   - changing samplers
    //   - changing attention type
    //   - etc.
    void sched_reserve();

    void synchronize();

    // Discard failed graph metadata after a C API error, without dropping
    // already processed KV state or allowing reuse of an unallocated graph.
    void discard_failed_graph() noexcept;

    // Request-scoped scheduler/cache exchange used by the default-off K3
    // grouped-prefill capacity ladder.
    bool moe_prefill_phase_workspace(bool active, uint32_t requested_tokens, uint32_t prefill_tokens);

    // whether a prefill ubatch of n_tokens runs on the CPU sink (llama_cpu_sink_prefill_route) in this context, at
    // this request's threshold or at max_tokens
    bool cpu_sink_prefill_route(uint32_t n_tokens) const;
    bool cpu_sink_prefill_route(uint32_t n_tokens, uint32_t max_tokens) const;
    // whether the scheduler would compute a MUL_MAT_ID of n_tokens over the model's expert tensors on a GPU
    // (llama_gpu_offloads_mul_mat_id)
    bool gpu_offloads_experts(uint32_t n_tokens) const;
    // the threshold: GGML_KIMI_CPU_PREFILL_MAX_TOKENS, else GGML_KIMI_CPU_PREFILL (1: every ubatch, 0: none)
    uint32_t cpu_prefill_max_tokens_get() const { return cpu_prefill_max_tokens; }

    // room in host memory for an allocation outside the RAM expert cache (llama_host_reserve)
    bool host_reserve(size_t bytes);
    ssd::result<ssd::host::allocation_ticket> host_take(size_t bytes);
    void host_note_memory_change() noexcept;


    const llama_model   & get_model()   const;
    const llama_cparams & get_cparams() const;

    ggml_backend_sched_t get_sched() const;

    uint32_t n_ctx()     const;
    uint32_t n_ctx_seq() const;
    uint32_t n_batch()   const;
    uint32_t n_ubatch()  const;
    uint32_t n_seq_max() const;

    uint32_t n_threads()       const;
    uint32_t n_threads_batch() const;

    llama_memory_t get_memory() const;

    // return true if the memory was updated
    bool memory_update(bool optimize);

    enum llama_pooling_type pooling_type() const;

    float * get_logits();
    float * get_logits_ith(int32_t i);

    float * get_embeddings();
    float * get_embeddings_ith(int32_t i);
    float * get_embeddings_seq(llama_seq_id seq_id);
    float * get_embeddings_nextn();
    float * get_embeddings_nextn_ith(int32_t i);
    float * get_embeddings_layer_inp(uint32_t lid);

    llama_token * get_sampled_tokens() const;
    llama_token   get_sampled_token_ith(int32_t idx);

    float * get_sampled_logits_ith(int32_t idx);
    size_t  get_sampled_logits_count(int32_t idx);

    float * get_sampled_probs_ith(int32_t idx);
    size_t  get_sampled_probs_count(int32_t idx);

    const llama_token * get_sampled_candidates_ith(int32_t idx);
    size_t get_sampled_candidates_count(int32_t idx);

    void attach_threadpool(
            ggml_threadpool_t threadpool,
            ggml_threadpool_t threadpool_batch);

    void detach_threadpool();

    void set_n_threads(int32_t n_threads, int32_t n_threads_batch);

    void set_abort_callback(bool (*abort_callback)(void * data), void * abort_callback_data);

    void set_embeddings (bool value);
    void set_embeddings_nextn(bool value, bool masked);
    void set_embeddings_layer_inp(uint32_t lid, bool enable);
    void set_causal_attn(bool value);
    void set_warmup(bool value);

    void set_adapters_lora(llama_adapter_lora ** adapters, size_t n_adapters, float * scales);

    bool adapters_lora_are_same(llama_adapter_lora ** adapters, size_t n_adapters, float * scales);

    bool set_adapter_cvec(
            const float * data,
                 size_t   len,
                int32_t   n_embd,
                int32_t   il_start,
                int32_t   il_end);

    // process a single ubatch with a specific graph type
    // if memory_context is provided, it will be applied first to the context's memory
    // ret contains the status of the graph computation
    // returns nullptr only if ret != GGML_STATUS_SUCCESS
    llm_graph_result * process_ubatch(
                const llama_ubatch & ubatch,
                    llm_graph_type   gtype,
            llama_memory_context_i * mctx,
                       ggml_status & ret);

    int encode(const llama_batch & batch_inp);
    int decode(const llama_batch & batch_inp);

    //
    // state save/load
    //

    size_t state_get_size();
    size_t state_get_data(      uint8_t * dst, size_t size);
    size_t state_set_data(const uint8_t * src, size_t size);

    size_t state_seq_get_size(llama_seq_id seq_id, llama_state_seq_flags flags);
    size_t state_seq_get_data(llama_seq_id seq_id,       uint8_t * dst, size_t size, llama_state_seq_flags flags);
    size_t state_seq_set_data(llama_seq_id seq_id, const uint8_t * src, size_t size, llama_state_seq_flags flags);

    // the state after the first n_tokens tokens of the sequence; -1: the current state
    uint32_t state_seq_at_align();
    bool     state_seq_can_save_at(llama_seq_id seq_id, int32_t n_tokens);
    llama_pos state_seq_pos_min_at(llama_seq_id seq_id, int32_t n_tokens);
    size_t   state_seq_get_size_at(llama_seq_id seq_id, llama_state_seq_flags flags, int32_t n_tokens);
    size_t   state_seq_get_data_at(llama_seq_id seq_id, uint8_t * dst, size_t size, llama_state_seq_flags flags, int32_t n_tokens);
    size_t   state_seq_fwrite_at  (llama_seq_id seq_id, FILE * fp, llama_state_seq_flags flags, int32_t n_tokens);
    size_t   state_seq_fread      (llama_seq_id seq_id, FILE * fp, llama_state_seq_flags flags);

    bool state_load_file(
            const char * filepath,
           llama_token * tokens_out,
                size_t   n_token_capacity,
                size_t * n_token_count_out);

    bool state_save_file(
            const char * filepath,
     const llama_token * tokens,
                size_t   n_token_count);

    size_t state_seq_load_file(
          llama_seq_id   seq_id,
            const char * filepath,
           llama_token * tokens_out,
                size_t   n_token_capacity,
                size_t * n_token_count_out);

    size_t state_seq_save_file(
          llama_seq_id   seq_id,
            const char * filepath,
     const llama_token * tokens,
                size_t   n_token_count);

    //
    // perf
    //

    llama_perf_context_data perf_get_data() const;
    void perf_reset();

    llama_memory_breakdown memory_breakdown() const;

    //
    // training
    //

    void opt_init(struct llama_model * model, struct llama_opt_params lopt_params);

    // TODO: more flexible combinations of logical/physical batch size and context size
    void opt_epoch(
            ggml_opt_dataset_t      dataset,
            ggml_opt_result_t       result_train,
            ggml_opt_result_t       result_eval,
            int64_t                 idata_split,
            ggml_opt_epoch_callback callback_train,
            ggml_opt_epoch_callback callback_eval);

    void opt_epoch_iter(
            ggml_opt_dataset_t               dataset,
            ggml_opt_result_t                result,
            const std::vector<llama_token> & tokens,
            const std::vector<llama_token> & labels_sparse,
            llama_batch                    & batch,
            ggml_opt_epoch_callback          callback,
            bool                             train,
            int64_t                          idata_in_loop,
            int64_t                          ndata_in_loop,
            int64_t                          t_loop_start);

private:
    //
    // output
    //

    // Make sure enough space is available for outputs.
    // Returns max number of outputs for which space was reserved.
    uint32_t output_reserve(int32_t n_outputs);

    void output_reorder();

    // map the output row index `i` to batch index
    int64_t output_resolve_row(int32_t i) const;

    //
    // graph
    //

public:
    uint32_t graph_max_nodes(uint32_t n_tokens) const;

    // can reuse the llm_graph_result instance of the context (for example to update a memory module)
    llm_graph_result * get_gf_res_reserve() const;

    // returns the result of ggml_backend_sched_graph_compute_async execution
    ggml_status graph_compute(ggml_cgraph * gf, bool batched, llama_memory_context_i * mctx = nullptr);

    // reserve a graph with a dummy ubatch of the specified size
    bool reserve_compute_graph(ggml_cgraph * graph, bool prefill);

    ggml_cgraph * graph_reserve(
        uint32_t n_tokens, uint32_t n_seqs, uint32_t n_outputs, const llama_memory_context_i * mctx, bool split_only = false, size_t * sizes = nullptr);

    bool set_sampler(llama_seq_id seq_id, llama_sampler * sampler);

private:
    // Async-copy enabled target layer inputs into the paired-drafter staging buffers.
    void extract_layer_inputs(const llm_graph_result * res, size_t token_offset, size_t n_tokens);

    void dense_graph_cache_begin_phase(int phase);
    void dense_graph_cache_end_phase();

    // worst-case memory context for a graph reserve that covers sched_reserve_n_ctx positions
    std::unique_ptr<llama_memory_context_i> memory_init_reserve();
    // positions currently held by the memory module (max over sequences of pos_max + 1)
    uint32_t memory_n_pos() const;
    // smallest reserve horizon (n_ubatch times a power of two, at most n_ctx) that holds n_pos
    uint32_t reserve_horizon_bucket(uint32_t n_pos) const;

    llm_graph_params graph_params(
                        llm_graph_result * res,
                      const llama_ubatch & ubatch,
            const llama_memory_context_i * mctx,
                          llm_graph_type   gtype) const;

    llm_graph_cb graph_get_cb() const;

    // TODO: read/write lora adapters and cvec
    size_t state_write_data(llama_io_write_i & io);
    size_t state_read_data (llama_io_read_i  & io);

    size_t state_seq_write_data(llama_io_write_i & io, llama_seq_id seq_id, llama_state_seq_flags flags, int32_t n_tokens = -1);
    size_t state_seq_read_data (llama_io_read_i  & io, llama_seq_id seq_id, llama_state_seq_flags flags);

    //
    // members
    //

    const llama_model & model;

    std::unique_ptr<llama_ssd_context> ssd_runtime;

    llama_cparams cparams;


    bool dense_gpu_phase_gate_enabled = false;
    bool dense_gpu_prefill_phase_active = false;
    int dense_gpu_scheduler_graph_phase = 0;
    bool cpu_sink_enabled = false;
    uint32_t cpu_prefill_max_tokens = 0; // prefill ubatches up to this size run on the CPU sink
    bool cpu_prefill_hint = false; // a sink prefill predicts each next layer's routes for its read-ahead
    // the threshold of this request: cpu_prefill_max_tokens, or the request's size when the graph does not fit
    uint32_t cpu_prefill_request_max_tokens = 0;
    // this request's prefill on the graph route: passes, their tokens, the expert-copy counters at its start
    llama_prefill_copy_meter prefill_copy;
    llama_dense_stream_meter dense_stream;
    bool cpu_prefill_actual_shape_enabled = false;
    bool cpu_prefill_actual_shape_active = false;
    bool cpu_prefill_output_sizing_enabled = false;
    uint32_t cpu_prefill_configured_ubatch = 0;
    uint32_t cpu_prefill_selected_ubatch = 0;
    uint32_t cpu_prefill_graph_granularity = 1;
    // Graph reserves cover the positions in use instead of the whole n_ctx
    // (LLAMA_RESERVE_HORIZON=0 restores full-context reserves).
    bool reserve_horizon_enabled = false;
    uint32_t sched_reserve_n_ctx = 0; // 0: reserve for the full n_ctx
    uint32_t sched_reserve_n_outputs = 0; // prompt outputs covered by the current scheduler
    uint32_t reserve_horizon_min = 0; // smallest horizon: the configured ubatch
    bool cpu_sink_decode_active = false;
    bool cpu_sink_multi_token_reported = false;
    bool cpu_sink_cache_batch_decode = false;
    std::unique_ptr<llama_kimi_cpu_sink_runtime> cpu_sink_runtime;

    llama_adapter_cvec_ptr  cvec;
    llama_adapter_loras_ptr loras;

    llama_cross cross; // TODO: tmp for handling cross-attention - need something better probably

    std::unique_ptr<llama_memory_i> memory;

    // decode output (2-dimensional array: [n_outputs][n_vocab])
    buffer_view<float> logits = {nullptr, 0};

    // embeddings output (2-dimensional array: [n_outputs][n_embd])
    // populated only when pooling_type == LLAMA_POOLING_TYPE_NONE
    buffer_view<float> embd = {nullptr, 0};

    // Hidden state used by DFlash's encoder and DSpark's confidence transport.
    buffer_view<float> embd_nextn = {nullptr, 0};

    // Per-layer target hidden states, including the final layer output at index n_layer.
    std::vector<buffer_view<float>> embd_layer_inp;

    struct sampling_info {
        // !samplers.empty() to check if any samplers are active
        std::map<llama_seq_id, llama_sampler *> samplers;

        buffer_view<float>       logits     = {nullptr, 0};
        buffer_view<llama_token> sampled    = {nullptr, 0};
        buffer_view<float>       probs      = {nullptr, 0};
        buffer_view<llama_token> candidates = {nullptr, 0};

        std::vector<uint32_t> logits_count;
        std::vector<uint32_t> probs_count;
        std::vector<uint32_t> candidates_count;

        // optimization
        std::vector<llama_token> token_ids_full_vocab;
    };

    sampling_info sampling;

    // sequence embeddings output (map of [n_embd] vectors)
    // populated only when pooling_type != LLAMA_POOLING_TYPE_NONE
    std::map<llama_seq_id, std::vector<float>> embd_seq;

    // reuse the batch_allocr to avoid unnecessary memory allocations
    std::unique_ptr<llama_batch_allocr> balloc;

    uint32_t n_outputs = 0; // number of actually-used outputs in the current ubatch or last logical batch

    std::vector<int32_t> output_ids; // map batch token positions to ids of the logits and embd buffers

    struct swap_info {
        uint32_t i0;
        uint32_t i1;
    };

    std::vector<swap_info> output_swaps;

    ggml_backend_sched_ptr sched;

    bool sched_need_reserve = true;

    ggml_backend_t backend_cpu = nullptr;
    std::vector<ggml_backend_ptr> backends;

    // training
    ggml_opt_context_t opt_ctx = nullptr;

    ggml_threadpool_t threadpool       = nullptr;
    ggml_threadpool_t threadpool_batch = nullptr;

    ggml_abort_callback abort_callback      = nullptr;
    void *              abort_callback_data = nullptr;

    std::vector<std::pair<ggml_backend_t, ggml_backend_set_n_threads_t>> set_n_threads_fns;

    // pointers and buffer types used for the compute buffer of each backend
    std::vector<ggml_backend_t>             backend_ptrs;
    std::vector<ggml_backend_buffer_type_t> backend_buft;
    std::vector<size_t>                     backend_buf_exp_size; // expected buffer sizes

    llm_graph_result_ptr gf_res_prev;
    llm_graph_result_ptr gf_res_reserve;

    // host buffer for the model output (logits and embeddings)
    ggml_backend_buffer_ptr buf_output;

    bool has_evaluated_once = false;

    // env: LLAMA_GRAPH_REUSE_DISABLE
    bool graph_reuse_disable = false;

    // perf
    mutable int64_t t_start_us  = 0;
    mutable int64_t t_load_us   = 0;
    mutable int64_t t_p_eval_us = 0;
    mutable int64_t t_eval_us   = 0;

    mutable int64_t t_compute_start_us = 0;
    mutable int64_t n_queued_tokens    = 0;

    mutable int32_t n_p_eval = 0; // number of tokens in eval calls for the prompt (with batch size > 1)
    mutable int32_t n_eval   = 0; // number of eval calls

    mutable int32_t n_reused = 0; // number of times the previous graph was reused
};
