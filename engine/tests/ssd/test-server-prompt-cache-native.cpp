// Private real-context resource admission/recovery gate; never a benchmark.
#include "llama.h"
#include "llama-ssd.h"
#include "llama-model.h" // Private fixture only: scoped enum fault injection, no payload mutation.
#include "llama-context.h" // Private fixture only: discard cached graph before fault injection.
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "inferssd/prompt-cache.h"
#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>
#include <initializer_list>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/mman.h>
#include <unistd.h>
#include <vector>
static void require(bool ok, const char * why) { if (!ok) { throw std::runtime_error(why); } }
static std::string read_text(const std::string & path) {
    std::ifstream stream(path); require(bool(stream), "resource observation unreadable");
    std::ostringstream result; result << stream.rdbuf(); require(!stream.bad(), "resource observation failed");
    return result.str();
}
static uint64_t field(const std::string & text, const std::string & key, uint64_t scale = 1) {
    std::istringstream stream(text); std::string line;
    while (std::getline(stream, line)) {
        std::istringstream row(line); std::string name; uint64_t value;
        if (row >> name && name == key) {
            require(bool(row >> value), "resource field is not an integer");
            require(value <= UINT64_MAX / scale, "resource field overflow"); return value * scale;
        }
    }
    throw std::runtime_error("resource field missing: " + key);
}
static std::string group_path() {
    std::istringstream stream(read_text("/proc/self/cgroup")); std::string line;
    while (std::getline(stream, line)) { if (line.compare(0, 3, "0::") == 0) { return "/sys/fs/cgroup" + line.substr(3); } }
    throw std::runtime_error("cgroup v2 membership missing");
}
struct observation { uint64_t current, rss, anon, available; std::string events; };
static observation observe(const std::string & group) {
    const auto status = read_text("/proc/self/status");
    return {std::stoull(read_text(group + "/memory.current")), field(status, "VmRSS:", 1024),
        field(status, "RssAnon:", 1024), field(read_text("/proc/meminfo"), "MemAvailable:", 1024),
        read_text(group + "/memory.events")};
}
static void print_observation(const char * phase, const observation & o) {
    std::printf("{\"event\":\"host.recovery.observation\",\"phase\":\"%s\",\"current\":%llu,\"rss\":%llu,\"anon\":%llu,\"available\":%llu}\n", phase,
        (unsigned long long)o.current, (unsigned long long)o.rss, (unsigned long long)o.anon, (unsigned long long)o.available);
    std::fflush(stdout);
}
struct reservation_delete { void operator()(llama_ssd_host_reservation * p) const noexcept { if (p) { llama_ssd_host_reservation_finish(p, false); } } };
using reservation = std::unique_ptr<llama_ssd_host_reservation, reservation_delete>;
static reservation take(llama_context * ctx, size_t bytes, bool expected) {
    char error[512] = {}; reservation out(llama_ssd_host_reserve(ctx, bytes, error, sizeof(error)));
    if (bool(out) != expected) { throw std::runtime_error(std::string("host reservation unexpected: ") + error); }
    if (!expected) {
        require(error[0], "refusal error missing");
        std::printf("HOST_OPAQUE_REFUSED bytes=%llu error=%s\n", (unsigned long long)bytes, error); std::fflush(stdout);
    }
    return out;
}
struct pages {
    llama_context * ctx; void * data = MAP_FAILED; size_t bytes;
    pages(llama_context * c, size_t n) : ctx(c), bytes(n) {
        data = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        require(data != MAP_FAILED, "pressure mmap failed");
    }
    void touch() {
        const long size = sysconf(_SC_PAGESIZE); require(size > 0, "page size unavailable");
        auto * p = static_cast<volatile unsigned char *>(data);
        for (size_t i = 0; i < bytes; i += size_t(size)) { p[i] = 0x5a; }
        p[bytes - 1] = 0xa5;
    }
    ~pages() { if (data != MAP_FAILED) { munmap(data, bytes); llama_ssd_host_note_memory_change(ctx); } }
};
struct backend_lifetime { backend_lifetime() { llama_backend_init(); } ~backend_lifetime() { llama_backend_free(); } };
struct batch_lifetime { llama_batch value = llama_batch_init(4, 0, 1); ~batch_lifetime() { llama_batch_free(value); } };
static void set_batch(llama_batch & batch, std::initializer_list<llama_token> tokens, int first) {
    batch.n_tokens = int32_t(tokens.size()); int i = 0;
    for (const auto token : tokens) {
        batch.token[i] = token; batch.pos[i] = first + i; batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0; batch.logits[i] = i + 1 == batch.n_tokens; ++i;
    }
}
static std::vector<float> logits(llama_context * ctx, int count) {
    const float * data = llama_get_logits_ith(ctx, -1); require(data, "missing logits");
    std::vector<float> out(data, data + count); for (float value : out) { require(std::isfinite(value), "nonfinite logits"); }
    return out;
}
int main(int argc, char ** argv) {
    if (argc != 5) { std::fprintf(stderr, "usage: test-server-prompt-cache-native MODEL ALIAS_JSON BACKEND_DIR TELEMETRY_JSONL\n"); return 2; }
    try {
        const auto group = group_path(); constexpr uint64_t limit = uint64_t(8) << 30, reserve = uint64_t(3) << 30;
        require(read_text(group + "/memory.max").find(std::to_string(limit)) == 0, "fixture requires exact 8GiB scope");
        require(std::stoull(read_text(group + "/memory.swap.max")) == 0, "fixture requires zero swap");
        backend_lifetime backend; ggml_backend_load_all_from_path(argv[3]);
        auto mp = llama_model_default_params();
        const llama_model_tensor_buft_override overrides[] = {
            {"\\.ffn_(up|down|gate|gate_up)_(ch|)exps", ggml_backend_cpu_buffer_type()},
            {"blk\\.[0-9]+\\.engram_embd\\.weight", ggml_backend_cpu_buffer_type()}, {nullptr, nullptr}};
        mp.tensor_buft_overrides = overrides; mp.n_gpu_layers = 99; mp.use_mmap = true;
        mp.use_extra_bufts = false; mp.use_direct_io = false; mp.no_host = true;
        mp.use_mlock = false; mp.check_tensors = false; mp.no_alloc = false; mp.defer_experts = false;
        using model_ptr = std::unique_ptr<llama_model, decltype(&llama_model_free)>;
        model_ptr model(llama_model_load_from_file(argv[1], mp), llama_model_free); require(bool(model), "model load failed");
        const int vocab = llama_vocab_n_tokens(llama_model_get_vocab(model.get())); require(vocab > 6, "tiny vocabulary too small");
        const llama_ssd_override values[] = {{"expert.source", argv[2]},
            {"prefetch.depth", "0"}, {"vram.budget", "0"}, {"prefill.cpu_max_tokens", "0"}};
        char error[512] = {};
        using option_ptr = std::unique_ptr<llama_ssd_options, decltype(&llama_ssd_options_free)>;
        option_ptr options(llama_ssd_options_create(nullptr, values, 4, argv[4], error, sizeof(error)), llama_ssd_options_free);
        require(bool(options), error);
        auto cp = llama_context_default_params(); cp.ssd_options = options.get();
        cp.n_ctx = 128; cp.n_batch = 4; cp.n_ubatch = 4; cp.n_seq_max = 1;
        cp.n_threads = cp.n_threads_batch = 2; cp.op_offload = false; cp.offload_kqv = true;
        cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED; cp.type_k = cp.type_v = GGML_TYPE_F16;
        cp.swa_full = true; cp.kv_unified = false; cp.samplers = nullptr; cp.n_samplers = 0;
        using context_ptr = std::unique_ptr<llama_context, decltype(&llama_free)>;
        context_ptr ctx(llama_init_from_model(model.get(), cp), llama_free); require(bool(ctx), "context create failed");
        require(llama_ssd_prefill_phase(ctx.get(), false, 0, 0), "decode phase start failed");
        batch_lifetime storage; auto & batch = storage.value; require(batch.token, "batch allocation failed");
        auto memory = llama_get_memory(ctx.get()); require(memory, "KV memory missing");
        llama_memory_clear(memory, true); set_batch(batch, {3}, 0);
        require(llama_decode(ctx.get(), batch) == 0, "clean single-token baseline failed");
        const auto expected = logits(ctx.get(), vocab);
        set_batch(batch, {4, 5}, 1); require(llama_decode(ctx.get(), batch) == 0, "two-token graph setup failed");
        (void)logits(ctx.get(), vocab); llama_memory_clear(memory, true);
        require(llama_memory_seq_pos_max(memory, 0) < 0, "KV clear failed");
        // Private internal fault injection through the real public decode C API.
        // The clean reference and failed/retried path preserve the same successful
        // two-token prefix; only the expert tensor enum changes while quiescent.
        set_batch(batch, {3, 4}, 0);
        require(llama_decode(ctx.get(), batch) == 0, "exception clean prefix failed");
        (void)logits(ctx.get(), vocab);
        set_batch(batch, {5}, 2);
        require(llama_decode(ctx.get(), batch) == 0, "exception clean suffix failed");
        const auto exception_expected = logits(ctx.get(), vocab);
        llama_memory_clear(memory, true);
        set_batch(batch, {3, 4}, 0);
        require(llama_decode(ctx.get(), batch) == 0, "exception retained prefix failed");
        (void)logits(ctx.get(), vocab);
        // DSV4 reports its current compressed-state boundary for both queries.
        require(llama_memory_seq_pos_min(memory, 0) == 1 && llama_memory_seq_pos_max(memory, 0) == 1,
            "exception setup did not retain exactly the successful prefix");
        // Decode phase uses physical one-token ubatches. Reserve before changing
        // the enum so the public decode rebuilds only after applying its memory context.
        require(ctx->get_cparams().n_ubatch == 1, "decode fixture requires physical one-token ubatches");
        ctx->discard_failed_graph();
        ctx->sched_reserve();
        require(ctx->get_cparams().n_ubatch == 1 && llama_memory_seq_pos_min(memory, 0) == 1 &&
            llama_memory_seq_pos_max(memory, 0) == 1, "pre-fault reserve changed the retained prefix or physical ubatch");
        auto * up = const_cast<ggml_tensor *>(model->get_tensor("blk.0.ffn_up_exps.weight"));
        require(up && up->type == GGML_TYPE_MXFP4, "exception fixture requires the frozen DeepSeek MXFP4 experts");
        set_batch(batch, {5}, 2); // Exact batch and position are retried after restoring the enum.
        {
            struct enum_restore {
                ggml_tensor * tensor; ggml_type type;
                ~enum_restore() { tensor->type = type; }
            } restore{up, up->type};
            up->type = GGML_TYPE_F16; // Unsupported CPU sink layout; payload/storage remain untouched.
            std::fprintf(stderr, "HOST_EXCEPTION_BEGIN enum_only=1 retained_pos_min=1 retained_pos_max=1\n");
            std::fflush(stderr);
            const int code = llama_decode(ctx.get(), batch);
            std::fprintf(stderr, "HOST_EXCEPTION_END code=%d\n", code); std::fflush(stderr);
            require(code == -3, "unsupported layout exception did not return -3 across the public C API");
            require(llama_memory_seq_pos_min(memory, 0) == 1 && llama_memory_seq_pos_max(memory, 0) == 1,
                "exception rollback lost the previous successful prefix or retained the failed position");
        }
        require(up->type == GGML_TYPE_MXFP4, "exception enum was not restored");
        require(llama_decode(ctx.get(), batch) == 0, "same context/position exception retry failed");
        const auto exception_actual = logits(ctx.get(), vocab);
        require(exception_actual.size() == exception_expected.size() &&
            !std::memcmp(exception_actual.data(), exception_expected.data(), exception_actual.size() * sizeof(float)),
            "exception retry differs from clean same-context prefix/suffix reference");
        require(llama_memory_seq_pos_min(memory, 0) == 2 && llama_memory_seq_pos_max(memory, 0) == 2,
            "exception retry did not commit the restored suffix");
        std::printf("HOST_CONTEXT_EXCEPTION_PASS internal_enum_injection=1 public_decode=1 returned_minus3=1 prefix_retained=1 same_ctx=1 same_batch=1 same_position=1 bitwise=1\n");
        std::fflush(stdout);
        set_batch(batch, {4, 5}, 3);
        require(llama_decode(ctx.get(), batch) == 0, "post-exception two-token graph setup failed");
        (void)logits(ctx.get(), vocab);
        llama_memory_clear(memory, true);
        require(llama_memory_seq_pos_max(memory, 0) < 0, "post-exception KV clear failed");
        // KV clear preserves the cached one-token graph. Reserve before applying
        // pressure so the refused public decode must take real graph admission.
        ctx->discard_failed_graph();
        ctx->sched_reserve();
        require(ctx->get_cparams().n_ubatch == 1 && llama_memory_seq_pos_max(memory, 0) < 0,
            "pre-pressure reserve changed the empty KV or physical ubatch");
        // Exercise real opaque ticket commit and cancellation before saturation.
        {
            auto ticket = take(ctx.get(), 65536, true); pages allocation(ctx.get(), 65536); allocation.touch();
            llama_ssd_host_reservation_finish(ticket.release(), true);
        }
        { auto ticket = take(ctx.get(), 65536, true); }
        { auto ticket = take(ctx.get(), 0, true); }
        { auto refused = take(ctx.get(), std::numeric_limits<size_t>::max(), false); }
        const auto before = observe(group); print_observation("before_pending", before);
        constexpr size_t chunk = size_t(128) << 20, pressure = size_t(256) << 20;
        std::vector<reservation> tickets; tickets.reserve(64); bool exhausted = false;
        for (size_t i = 0; i < 64; ++i) {
            error[0] = '\0'; reservation ticket(llama_ssd_host_reserve(ctx.get(), chunk, error, sizeof(error)));
            if (!ticket) { require(std::strstr(error, "host memory limit refused"), "pending fill failed outside the limit"); exhausted = true; break; }
            tickets.push_back(std::move(ticket));
        }
        require(exhausted && !tickets.empty(), "finite real pending budget not observed");
        { auto zero = take(ctx.get(), 0, true); }
        const auto pending = observe(group); print_observation("pending_saturated", pending);
        require(pending.current < limit - reserve - pressure - (uint64_t(64) << 20) &&
            pending.available > reserve + pressure + (uint64_t(64) << 20), "not enough actual guard reserve for bounded pressure");
        const size_t held_tickets = tickets.size(); const uint64_t held_bytes = tickets.size() * chunk;
        set_batch(batch, {3}, 0); // This exact batch is reused after the refusal, without reinitializing the context.
        {
            pages allocation(ctx.get(), pressure); allocation.touch();
            const auto touched = observe(group); print_observation("pressure_touched", touched);
            require(touched.anon >= pending.anon + pressure - (uint64_t(8) << 20), "actual anonymous first-touch growth missing");
            { auto refused = take(ctx.get(), 0, false); }
            std::fprintf(stderr, "HOST_RECOVERY_BEGIN_REJECTION tickets=%zu pending_bytes=%llu pressure_bytes=%zu\n", tickets.size(),
                (unsigned long long)(tickets.size() * chunk), pressure); std::fflush(stderr);
            const int code = llama_decode(ctx.get(), batch);
            std::fprintf(stderr, "HOST_RECOVERY_END_REJECTION code=%d\n", code); std::fflush(stderr);
            require(code == -2, "same-shape setup did not reject graph allocation with -2");
            require(llama_memory_seq_pos_max(memory, 0) < 0, "refused decode left KV positions");
        }
        tickets.clear(); llama_ssd_host_note_memory_change(ctx.get());
        { auto zero = take(ctx.get(), 0, true); }
        require(llama_decode(ctx.get(), batch) == 0, "same context/batch/position retry failed");
        const auto actual = logits(ctx.get(), vocab);
        require(actual.size() == expected.size() && !std::memcmp(actual.data(), expected.data(), actual.size() * sizeof(float)),
            "same-shape recovered logits differ from clean same-context baseline");
        const auto after = observe(group); print_observation("recovered", after);
        for (const char * name : {"max", "oom", "oom_kill", "oom_group_kill"}) {
            require(field(after.events, name) == field(before.events, name), "kernel limit/OOM event changed");
        }
        std::printf("HOST_CONTEXT_RECOVERY_PASS opaque_commit=1 opaque_cancel=1 impossible_refusal=1 zero_before=1 zero_pressure_refused=1 zero_after=1 same_ctx=1 same_batch=1 same_position=1 bitwise=1 tickets=%zu pending_bytes=%llu\n", held_tickets, (unsigned long long)held_bytes);
        // Save real serialized state only after resource refusal/recovery has
        // drained and restored admission. No synthetic context or state bytes.
        set_batch(batch, {4}, 1);
        require(llama_decode(ctx.get(), batch) == 0, "prompt cache clean suffix failed");
        const auto cache_expected = logits(ctx.get(), vocab);
        llama_memory_clear(memory, true);
        set_batch(batch, {3}, 0);
        require(llama_decode(ctx.get(), batch) == 0, "prompt cache prefix reconstruction failed");
        (void)logits(ctx.get(), vocab);
        server_prompt prompt;
        prompt.tokens = server_tokens(llama_tokens{3}, false);
        server_prompt_cache cache(0, 0);
        { auto zero = take(ctx.get(), 0, true); }
        cache.save(prompt, true, ctx.get(), 0, COMMON_CONTEXT_SEQ_RM_TYPE_FULL);
        require(cache.states.size() == 1 && !cache.states.front().data.empty(), "real cache.save did not store serialized state");
        const size_t saved_bytes = cache.states.front().data.size();
        llama_memory_clear(memory, true);
        require(llama_memory_seq_pos_max(memory, 0) < 0, "prompt cache KV clear failed");
        server_prompt restored;
        const server_tokens requested(llama_tokens{3, 4}, false);
        require(cache.load(restored, requested, 0, ctx.get(), 0, COMMON_CONTEXT_SEQ_RM_TYPE_FULL), "real cache.load did not restore prefix");
        require(restored.tokens.size() == 1 && llama_memory_seq_pos_min(memory, 0) == 0 &&
            llama_memory_seq_pos_max(memory, 0) == 0, "cache load restored wrong token/KV prefix");
        set_batch(batch, {4}, 1);
        require(llama_decode(ctx.get(), batch) == 0, "prompt cache restored suffix failed");
        const auto cache_actual = logits(ctx.get(), vocab);
        require(cache_actual.size() == cache_expected.size() && !std::memcmp(cache_actual.data(), cache_expected.data(),
            cache_actual.size() * sizeof(float)), "cache save/load suffix differs from clean same-context reference");
        cache.states.clear(); llama_ssd_host_note_memory_change(ctx.get());
        { auto zero = take(ctx.get(), 0, true); }
        const auto saved = observe(group); print_observation("prompt_cache_roundtrip", saved);
        for (const char * name : {"max", "oom", "oom_kill", "oom_group_kill"}) {
            require(field(saved.events, name) == field(before.events, name), "prompt cache kernel limit/OOM event changed");
        }
        std::printf("HOST_PROMPT_CACHE_NATIVE_PASS real_context=1 actual_save=1 actual_load=1 actual_state_bytes=%zu restored_prefix=1 suffix_bitwise=1 after_resource_recovery=1\n", saved_bytes);
        return 0;
    } catch (const std::exception & e) { std::fprintf(stderr, "HOST_CONTEXT_RECOVERY_FAILED: %s\n", e.what()); return 1; }
}
