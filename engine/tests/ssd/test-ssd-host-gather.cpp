// The page cache a prefill graph's index gathers bring in (src/llama-host-gather.h), which the host-memory ledger holds
// for the prefill: GET_ROWS of a file-mapped weight (token embeddings) and a custom op gathering rows of one by an I32
// index (DeepSeek-V4.1's Engram tables), each row at page + row bytes - 1, a table at most its own pages; weights that
// are not file-mapped, other indices and other operands are not counted.
#include "../../src/ssd/host-gather.h"

#include <cstdio>
#include <set>

static int failures = 0;
static void check(bool ok, const char * what) {
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", what); ++failures; }
}

static void decode_rows(ggml_tensor *, int, int, void *) {}

int main() {
    ggml_init_params params = { 64ull << 20, nullptr, true };
    ggml_context * ctx = ggml_init(params);
    const uint64_t page = 4096;
    const int64_t n_tokens = 4104, n_cols = 24;

    ggml_tensor * embd = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, 4096, 129280);         // 4352-byte rows
    ggml_tensor * engram = ggml_new_tensor_2d(ctx, GGML_TYPE_I8, 264, 300000000);       // 264-byte rows
    ggml_tensor * small = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 16, 4);                // 256 bytes in all
    ggml_tensor * resident = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4096, 1000);        // not file-mapped
    ggml_tensor * tokens = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_tokens);
    ggml_tensor * rows = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_cols * n_tokens);
    ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4096, n_tokens);
    std::set<const ggml_tensor *> mapped = { embd, engram, small };
    auto file_mapped = [&](const ggml_tensor * t) { return mapped.count(t) != 0; };

    ggml_cgraph * gf = ggml_new_graph_custom(ctx, 64, false);
    ggml_build_forward_expand(gf, ggml_get_rows(ctx, embd, tokens));
    ggml_tensor * args[] = { engram, rows };
    ggml_build_forward_expand(gf, ggml_custom_4d(ctx, GGML_TYPE_F32, 256, n_cols, n_tokens, 1, args, 2, decode_rows, 1, nullptr));
    const uint64_t expect_embd = uint64_t(n_tokens) * (page + 4352 - 1);
    const uint64_t expect_engram = uint64_t(n_cols * n_tokens) * (page + 264 - 1);
    const uint64_t both = llama_host_gather_bytes(gf, page, file_mapped);
    check(both == expect_embd + expect_engram, "token embeddings and an Engram table");
    std::printf("gather: embd=%llu engram=%llu total=%llu (%.1f KiB a token)\n", (unsigned long long) expect_embd,
        (unsigned long long) expect_engram, (unsigned long long) both, both / 1024.0 / n_tokens);

    // a second gather of the same table adds its rows; a view gathers from the table it views
    ggml_tensor * view = ggml_view_2d(ctx, engram, 264, 1000000, engram->nb[1], 0);
    ggml_tensor * args2[] = { view, rows };
    ggml_build_forward_expand(gf, ggml_custom_4d(ctx, GGML_TYPE_F32, 256, n_cols, n_tokens, 1, args2, 2, decode_rows, 1, nullptr));
    check(llama_host_gather_bytes(gf, page, file_mapped) == expect_embd + 2 * expect_engram, "a view of a mapped table");

    // a table costs at most its own pages and one more
    ggml_cgraph * g2 = ggml_new_graph_custom(ctx, 64, false);
    ggml_build_forward_expand(g2, ggml_get_rows(ctx, small, tokens));
    check(llama_host_gather_bytes(g2, page, file_mapped) == 2 * page, "a small table's cap");

    // not counted: a weight that is not file-mapped, an index that is not I32, a custom op whose first operand is an
    // activation (the CPU sink's expert node)
    ggml_cgraph * g3 = ggml_new_graph_custom(ctx, 64, false);
    ggml_build_forward_expand(g3, ggml_get_rows(ctx, resident, tokens));
    ggml_tensor * fidx = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_tokens);
    ggml_tensor * args3[] = { engram, fidx };
    ggml_build_forward_expand(g3, ggml_custom_4d(ctx, GGML_TYPE_F32, 256, 1, n_tokens, 1, args3, 2, decode_rows, 1, nullptr));
    ggml_tensor * args4[] = { x, rows };
    ggml_build_forward_expand(g3, ggml_custom_4d(ctx, GGML_TYPE_F32, 4096, n_tokens, 1, 1, args4, 2, decode_rows, 1, nullptr));
    check(llama_host_gather_bytes(g3, page, file_mapped) == 0, "uncounted gathers");

    check(llama_host_page_size() >= 4096, "the page size");
    ggml_free(ctx);
    if (failures) { std::fprintf(stderr, "KIMI_HOST_GATHER_FAILED failures=%d\n", failures); return 1; }
    std::printf("KIMI_HOST_GATHER_OK\n");
    return 0;
}
