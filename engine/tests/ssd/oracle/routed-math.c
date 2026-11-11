// SPDX-License-Identifier: MIT
// Explicit historical math reference; never linked into a production backend.
#include "../../../ggml/src/ggml-cpu/ggml-cpu.c"
#include "routed-math.inc"

bool inferssd_test_kimi_matmul(struct ggml_tensor * output, void * work, size_t work_size) {
    if (!output || !work || output->op != GGML_OP_MUL_MAT_ID) { return false; }
    struct ggml_threadpool_params options = ggml_threadpool_params_default(1);
    struct ggml_threadpool * pool = ggml_threadpool_new(&options);
    if (!pool) { return false; }
    // One real compute participant; this oracle calls the historical math body
    // directly rather than installing a production selector or touching env.
    atomic_store_explicit(&pool->n_graph, 1, memory_order_relaxed);
    const struct ggml_compute_params params = { 0, 1, work_size, work, pool, false };
    ggml_compute_forward_mul_mat_id_common_kimi(&params, output);
    ggml_threadpool_free(pool);
    return true;
}
