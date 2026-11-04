// SPDX-License-Identifier: Apache-2.0
// An op that writes through a view runs on the backend that holds the storage it writes. Left to the op's own
// backend, it writes to a copy (the mask stays unwritten) or into memory that backend does not own (a crash).

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-backend-impl.h"
#include "ggml-cpu.h"

#include <cstdint>
#include <cstdio>
#include <vector>

namespace {

// A second backend, as the scheduler sees a GPU: a CPU behind its own device and its own buffer type. Without a buffer
// type of its own the scheduler would merge it with the CPU backend before any placement mattered.
ggml_backend_buffer_type gpu_buffer_type;

ggml_backend_buffer_type_t gpu_get_buffer_type(ggml_backend_dev_t) { return &gpu_buffer_type; }

bool gpu_supports_every_op(ggml_backend_dev_t, const ggml_tensor *) { return true; }

// A GPU backend refuses the types it has no shader for.
bool gpu_refuses_set_rows(ggml_backend_dev_t, const ggml_tensor * op) { return op->op != GGML_OP_SET_ROWS; }

struct fake_gpu {
    ggml_backend_device device;
    ggml_backend backend;

    fake_gpu(ggml_backend_t cpu, bool (*supports_op)(ggml_backend_dev_t, const ggml_tensor *)) {
        device = *ggml_backend_get_device(cpu);
        device.iface.get_buffer_type = gpu_get_buffer_type;
        device.iface.supports_op = supports_op;
        backend = *cpu;
        backend.device = &device;
    }
};

struct assignment {
    ggml_backend_t base, out;
};

// base = repeat(-1), set_rows(base, [7, 8] at rows 1 and 3), cont: the shape of an attention mask's scatter.
bool check(const char * name, ggml_backend_t gpu, ggml_backend_t cpu, assignment given, assignment expected) {
    ggml_backend_t backends[2] = { gpu, cpu };
    ggml_backend_sched_t sched = ggml_backend_sched_new(backends, nullptr, 2, 64, false, false);

    ggml_init_params params{ ggml_tensor_overhead() * 32 + ggml_graph_overhead_custom(32, false), nullptr, true };
    ggml_context * ctx = ggml_init(params);
    ggml_tensor * seed = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
    ggml_tensor * rows = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, 2);
    ggml_tensor * idx = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 2);
    ggml_set_input(seed);
    ggml_set_input(rows);
    ggml_set_input(idx);
    ggml_tensor * base = ggml_repeat_4d(ctx, seed, 1, 6, 1, 1);
    ggml_tensor * out = ggml_set_rows(ctx, base, rows, idx);
    ggml_tensor * result = ggml_cont(ctx, out);
    ggml_set_output(result);
    ggml_cgraph * graph = ggml_new_graph_custom(ctx, 32, false);
    ggml_build_forward_expand(graph, result);

    ggml_backend_sched_set_tensor_backend(sched, base, given.base);
    ggml_backend_sched_set_tensor_backend(sched, out, given.out);

    bool ok = ggml_backend_sched_alloc_graph(sched, graph);
    if (!ok) {
        std::fprintf(stderr, "%s: allocation failed\n", name);
    }
    if (ok && ggml_backend_sched_get_tensor_backend(sched, out) != expected.out) {
        std::fprintf(stderr, "%s: the op is not on the backend of its storage\n", name);
        ok = false;
    }
    if (ok && ggml_backend_sched_get_tensor_backend(sched, base) != expected.base) {
        std::fprintf(stderr, "%s: the storage is not on the expected backend\n", name);
        ok = false;
    }
    if (ok) {
        const float seed_value = -1.0f;
        const float row_values[2] = { 7.0f, 8.0f };
        const int32_t row_ids[2] = { 1, 3 };
        ggml_backend_tensor_set(seed, &seed_value, 0, sizeof(seed_value));
        ggml_backend_tensor_set(rows, row_values, 0, sizeof(row_values));
        ggml_backend_tensor_set(idx, row_ids, 0, sizeof(row_ids));
        ok = ggml_backend_sched_graph_compute(sched, graph) == GGML_STATUS_SUCCESS;
        std::vector<float> got(6);
        ggml_backend_tensor_get(result, got.data(), 0, got.size() * sizeof(float));
        const float want[6] = { -1.0f, 7.0f, -1.0f, 8.0f, -1.0f, -1.0f };
        for (size_t i = 0; i < got.size(); i++) {
            if (got[i] != want[i]) {
                std::fprintf(stderr, "%s: row %zu is %g, expected %g\n", name, i, got[i], want[i]);
                ok = false;
            }
        }
    }
    ggml_free(ctx);
    ggml_backend_sched_free(sched);
    return ok;
}

} // namespace

int main() {
    ggml_backend_t cpu = ggml_backend_cpu_init();
    gpu_buffer_type = *ggml_backend_cpu_buffer_type();

    fake_gpu full(cpu, gpu_supports_every_op);
    fake_gpu refusing(cpu, gpu_refuses_set_rows);

    bool ok = true;
    // Both on one backend: nothing moves.
    ok = check("one backend", &full.backend, cpu, { &full.backend, &full.backend }, { &full.backend, &full.backend }) && ok;
    // The storage fell back to the CPU, the op was placed on the GPU: the op moves to its storage.
    ok = check("the op moves", &full.backend, cpu, { cpu, &full.backend }, { cpu, cpu }) && ok;
    // The storage is on the GPU, which refuses the op: the storage moves to the op's backend.
    ok = check("the storage moves", &refusing.backend, cpu, { &refusing.backend, cpu }, { cpu, cpu }) && ok;

    ggml_backend_free(cpu);
    return ok ? 0 : 1;
}
