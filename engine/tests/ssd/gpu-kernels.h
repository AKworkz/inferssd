// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "ggml-cpu.h"
#include "ggml-cuda.h"
#include "ggml-ssd/gpu/provider.h"
#include <stdexcept>

inline const ssd::gpu::cpu_kernels_v1 * test_cpu_kernels() {
    const auto get = reinterpret_cast<ssd::gpu::get_cpu_kernels_v1>(
        ggml_backend_reg_get_proc_address(ggml_backend_cpu_reg(), "ggml_backend_cpu_ssd_kernels_v1"));
    const auto * api = get ? get() : nullptr;
    if (!ssd::gpu::valid(api)) { throw std::runtime_error("normal CPU kernels registry missing"); }
    return api;
}
inline ssd::runtime_services test_gpu_services() {
    const auto get = reinterpret_cast<ssd::gpu::get_provider_v1>(
        ggml_backend_reg_get_proc_address(ggml_backend_cuda_reg(), "ggml_backend_ssd_provider_v1"));
    const auto * api = get ? get() : nullptr;
    if (!api || api->version != 1 || api->size != sizeof(*api)) {
        throw std::runtime_error("normal CUDA provider registry missing");
    }
    ssd::runtime_services services;
    services.gpu = api;
    services.cpu = test_cpu_kernels();
    return services;
}
