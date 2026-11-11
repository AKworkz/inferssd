// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "ggml-backend.h"
#include "vram-tier-api.h"
#include "transport.h"
#include "../gpu-tier-settings.h"
#include "../testing/tier.h"

namespace ssd { namespace gpu {
// Same-build L1 contract. L3 borrows the immutable table from the normal backend
// registry, retains that registry, and attaches context-owned services before use.
struct provider_v1 {
    uint32_t version; // 2: no cache-location view
    size_t size;
    const kimi_vram_tier_api * api;
    const kimi_vram_residency_api * residency;
    const kimi_vram_format_api * format;
    const kimi_vram_prefill_api * prefill;
    const kimi_vram_elastic_api * elastic;
    const kimi_vram_deferred_api * deferred;
    const kimi_vram_host_api * host;
    const kimi_vram_policy_api * policy;
    const settings_api * settings;
    const testing::api * testing;
    const transport_v1 * transport;
    bool (*device_is_unified)(int);
    int (*device_ordinal)(ggml_backend_t);
    bool (*attach_services)(ggml_backend_t, const runtime_services *, char *, size_t) noexcept;
    void * (*stream)(ggml_backend_t);
};
using get_provider_v1 = const provider_v1 * (*)();
}} // namespace ssd::gpu
