// SPDX-License-Identifier: Apache-2.0
#include "ggml-ssd/gpu/provider.h"
#include "ggml-cuda.h"

extern "C" {
const kimi_vram_tier_api * kimi_vram_tier_get_api();
const kimi_vram_residency_api * kimi_vram_residency_get_api();
const kimi_vram_format_api * kimi_vram_format_get_api();
const kimi_vram_prefill_api * kimi_vram_prefill_get_api();
const kimi_vram_elastic_api * kimi_vram_elastic_get_api();
const kimi_vram_deferred_api * kimi_vram_deferred_get_api();
const kimi_vram_host_api * kimi_vram_host_get_api();
const kimi_vram_policy_api * kimi_vram_policy_get_api();
const ssd::gpu::settings_api * ggml_backend_cuda_ssd_tier_settings_get_api();
const ssd::gpu::testing::api * ggml_ssd_tier_testing_get_api();
const ssd::gpu::transport_v1 * ggml_backend_cuda_ssd_transport_v1();
bool ggml_backend_cuda_ssd_attach_services(ggml_backend_t, const ssd::runtime_services *, char *, size_t) noexcept;
int ggml_backend_cuda_ssd_device_ordinal(ggml_backend_t);
void * ggml_backend_cuda_ssd_stream(ggml_backend_t);

GGML_BACKEND_API const ssd::gpu::provider_v1 * ggml_backend_cuda_ssd_api_v1() {
    static const ssd::gpu::provider_v1 api{
        2, sizeof(ssd::gpu::provider_v1),
        kimi_vram_tier_get_api(), kimi_vram_residency_get_api(), kimi_vram_format_get_api(),
        kimi_vram_prefill_get_api(), kimi_vram_elastic_get_api(), kimi_vram_deferred_get_api(),
        kimi_vram_host_get_api(), kimi_vram_policy_get_api(),
        ggml_backend_cuda_ssd_tier_settings_get_api(), ggml_ssd_tier_testing_get_api(),
        ggml_backend_cuda_ssd_transport_v1(),
        ggml_backend_cuda_device_is_unified, ggml_backend_cuda_ssd_device_ordinal,
        ggml_backend_cuda_ssd_attach_services, ggml_backend_cuda_ssd_stream,
    };
    return &api;
}
}
