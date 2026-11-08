// SPDX-License-Identifier: Apache-2.0
// Exercise real scheduler copies/allocator and CPU math across incompatible
// memory domains. The second domain is host-backed; this is not a GPU gate.
#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-backend-impl.h"

#include <array>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <new>
#include <stdexcept>
#include <vector>

static void require(bool ok, const char * message) { if (!ok) { throw std::runtime_error(message); } }

struct separate_domain {
    ggml_backend_buffer_type type = {};
    ggml_backend_device device = {};
    ggml_backend backend = {};
    unsigned allocations = 0, scratch_reserves = 0;
    bool fail_allocation = false, fail_scratch = false;

    separate_domain() {
        type.context = this;
        type.device = &device;
        type.iface.get_name = [](ggml_backend_buffer_type_t) { return "separate_domain"; };
        type.iface.get_alignment = [](ggml_backend_buffer_type_t) { return size_t(64); };
        type.iface.is_host = [](ggml_backend_buffer_type_t) { return false; };
        type.iface.alloc_buffer = [](ggml_backend_buffer_type_t type, size_t size) -> ggml_backend_buffer_t {
            auto & domain = *static_cast<separate_domain *>(type->context);
            if (domain.fail_allocation) { return nullptr; }
            void * data = ::operator new(size ? size : 64, std::align_val_t(64), std::nothrow);
            if (!data) { return nullptr; }
            ++domain.allocations;
            ggml_backend_buffer_i methods = {};
            methods.get_base = [](ggml_backend_buffer_t buffer) { return buffer->context; };
            methods.free_buffer = [](ggml_backend_buffer_t buffer) { ::operator delete(buffer->context, std::align_val_t(64)); };
            methods.set_tensor = [](ggml_backend_buffer_t, ggml_tensor * tensor, const void * data, size_t offset, size_t bytes) {
                std::memcpy(static_cast<char *>(tensor->data) + offset, data, bytes);
            };
            methods.get_tensor = [](ggml_backend_buffer_t, const ggml_tensor * tensor, void * data, size_t offset, size_t bytes) {
                std::memcpy(data, static_cast<const char *>(tensor->data) + offset, bytes);
            };
            return ggml_backend_buffer_init(type, methods, data, size);
        };
        device.context = this;
        device.iface.get_type = [](ggml_backend_dev_t) { return GGML_BACKEND_DEVICE_TYPE_GPU; };
        device.iface.get_buffer_type = [](ggml_backend_dev_t device) { return &static_cast<separate_domain *>(device->context)->type; };
        device.iface.supports_op = [](ggml_backend_dev_t, const ggml_tensor * tensor) { return tensor->op != GGML_OP_CUSTOM; };
        device.iface.supports_buft = [](ggml_backend_dev_t device, ggml_backend_buffer_type_t type) {
            return type == &static_cast<separate_domain *>(device->context)->type;
        };
        backend.device = &device;
        backend.context = this;
        backend.iface.get_name = [](ggml_backend_t) { return "separate_domain"; };
        backend.iface.graph_reserve = [](ggml_backend_t backend, ggml_cgraph * const *, int) {
            auto & domain = *static_cast<separate_domain *>(backend->context);
            ++domain.scratch_reserves;
            return !domain.fail_scratch;
        };
        backend.iface.graph_compute = [](ggml_backend_t, ggml_cgraph * graph) {
            auto * context = ggml_init({1 << 20, nullptr, false});
            const auto status = ggml_graph_compute_with_ctx(context, graph, 1);
            ggml_free(context);
            return status;
        };
    }
};

using scheduler_ptr = std::unique_ptr<ggml_backend_sched, decltype(&ggml_backend_sched_free)>;
struct executor {
    separate_domain domain;
    std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)> cpu {ggml_backend_cpu_init(), &ggml_backend_free};
    scheduler_ptr scheduler {nullptr, &ggml_backend_sched_free};
    explicit executor(bool parallel = false) {
        ggml_backend_t backends[] = {&domain.backend, cpu.get()};
        scheduler.reset(ggml_backend_sched_new(backends, nullptr, 2, 128, parallel, false));
        require(bool(cpu) && bool(scheduler), "backend startup failed");
        ggml_backend_sched_set_auto_reserve(scheduler.get(), false);
    }
};

struct graph_case {
    ggml_context * context = ggml_init({1 << 20, nullptr, true});
    ggml_backend_buffer_t weight_buffer = nullptr;
    ggml_tensor * x = nullptr, * a = nullptr, * b = nullptr, * output = nullptr;
    ggml_cgraph * graph = nullptr;
    explicit graph_case(executor & executor) {
        require(context, "metadata allocation failed");
        x = ggml_new_tensor_1d(context, GGML_TYPE_F32, 16);
        ggml_set_name(x, "input"); ggml_set_input(x);
        auto * weight = ggml_new_tensor_1d(context, GGML_TYPE_F32, 16);
        ggml_set_name(weight, "weight");
        weight_buffer = ggml_backend_buft_alloc_buffer(&executor.domain.type, 64);
        require(weight_buffer, "weight allocation failed");
        ggml_backend_buffer_set_usage(weight_buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        ggml_backend_tensor_alloc(weight_buffer, weight, ggml_backend_buffer_get_base(weight_buffer));
        const std::array<float, 16> ones {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
        ggml_backend_tensor_set(weight, ones.data(), 0, sizeof(ones));
        a = ggml_add(context, weight, x); ggml_set_name(a, "domain_add");
        ggml_tensor * args[] = {a};
        auto twice = +[](ggml_tensor * dst, int ith, int, void *) {
            if (ith) { return; }
            for (int i = 0; i < 16; ++i) { static_cast<float *>(dst->data)[i] = 2 * static_cast<const float *>(dst->src[0]->data)[i]; }
        };
        b = ggml_custom_4d(context, GGML_TYPE_F32, 16, 1, 1, 1, args, 1, twice, 1, nullptr);
        ggml_set_name(b, "cpu_custom");
        output = ggml_add(context, weight, b); ggml_set_name(output, "output"); ggml_set_output(output);
        graph = ggml_new_graph_custom(context, 128, false); ggml_build_forward_expand(graph, output);
        // Assignments make is_reset=false, but do not create split metadata.
        ggml_backend_sched_set_tensor_backend(executor.scheduler.get(), x, executor.cpu.get());
        ggml_backend_sched_set_tensor_backend(executor.scheduler.get(), a, &executor.domain.backend);
        ggml_backend_sched_set_tensor_backend(executor.scheduler.get(), b, executor.cpu.get());
        ggml_backend_sched_set_tensor_backend(executor.scheduler.get(), output, &executor.domain.backend);
    }
    ~graph_case() { ggml_backend_buffer_free(weight_buffer); ggml_free(context); }
    std::array<ggml_tensor *, 3> copies() const { return {a->src[1], b->src[0], output->src[1]}; }
    void compute(executor & executor) {
        std::array<float, 16> input, result;
        input.fill(1);
        ggml_backend_tensor_set(x, input.data(), 0, sizeof(input));
        require(ggml_backend_sched_graph_compute(executor.scheduler.get(), graph) == GGML_STATUS_SUCCESS, "cross-domain compute failed");
        require(ggml_backend_sched_get_n_splits(executor.scheduler.get()) == 3, "graph lost a memory-domain crossing");
        ggml_backend_tensor_get(output, result.data(), 0, sizeof(result));
        for (const float value : result) { require(value == 5, "cross-domain copies lost input data (expected 5)"); }
    }
};

static void numerical(bool parallel) {
    executor executor(parallel);
    // Preparation synchronizes (preserving the scheduler's stable copy-0
    // policy), but all configured parallel input-copy metadata must survive.
    require(ggml_backend_sched_get_n_copies(executor.scheduler.get()) == (parallel ? 4 : 1), "parallel copies were not configured");
    for (int iteration = 0; iteration < 2; ++iteration) {
        ggml_backend_sched_reset(executor.scheduler.get());
        graph_case graph(executor);
        size_t sizes[2] = {};
        const unsigned allocations = executor.domain.allocations;
        require(ggml_backend_sched_prepare_graph(executor.scheduler.get(), graph.graph, sizes), "explicit assignments prevented preparation");
        require(sizes[0] && sizes[1] && executor.domain.allocations == allocations, "probe allocated storage or lost a domain");
        const auto copies = graph.copies();
        require(ggml_backend_sched_reserve_prepared(executor.scheduler.get(), graph.graph), "prepared reservation failed");
        require(copies == graph.copies(), "reservation rewrote copy metadata");
        require(ggml_backend_sched_alloc_prepared(executor.scheduler.get(), graph.graph), "prepared allocation failed");
        require(copies == graph.copies(), "allocation rewrote copy metadata");
        graph.compute(executor);
        ggml_backend_sched_reset(executor.scheduler.get());
    }
    require(executor.domain.scratch_reserves == 2, "backend scratch was not reserved once per graph");
}

static void invalidation_and_failure() {
    for (int mode = 0; mode < 4; ++mode) {
        executor executor;
        graph_case graph(executor);
        size_t sizes[2] = {};
        require(ggml_backend_sched_prepare_graph(executor.scheduler.get(), graph.graph, sizes), "failure case preparation failed");
        if (mode == 0) {
            ggml_backend_sched_reset(executor.scheduler.get());
        } else if (mode == 1) {
            require(!ggml_backend_sched_alloc_prepared(executor.scheduler.get(), graph.graph), "allocation bypassed resource reservation");
        } else {
            executor.domain.fail_allocation = mode == 2;
            executor.domain.fail_scratch = mode == 3;
            require(!ggml_backend_sched_reserve_prepared(executor.scheduler.get(), graph.graph), "injected storage/scratch failure was accepted");
        }
        require(!ggml_backend_sched_reserve_prepared(executor.scheduler.get(), graph.graph), "cancelled transaction remained reservable");
        require(!ggml_backend_sched_alloc_prepared(executor.scheduler.get(), graph.graph), "cancelled transaction remained executable");
    }
}

int main() {
    try {
        numerical(false); numerical(true); invalidation_and_failure();
        std::cout << "prepared graph: three real copy splits, CPU custom math, parallel copies, reset/storage/scratch refusal passed\n";
    } catch (const std::exception & error) { std::cerr << error.what() << '\n'; return 1; }
}
