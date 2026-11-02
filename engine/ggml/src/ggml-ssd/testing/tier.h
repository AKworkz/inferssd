// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstddef>
#include <cstdint>

namespace ssd {
namespace gpu {
struct cpu_kernels_v1;
namespace testing {
// Explicit construction inputs for real CUDA lifecycle fixtures. Product
// callers receive no policy parser: their factory derives the fixed profile
// only from the typed settings snapshot and copied observation channels.
struct tier_options {
    bool fixture = true, mxfp4_gpu_hidden = false;
    uint64_t safety_mib = 512;
    bool async_upload = false, memory_trace = false, keep_across_requests = false, timing = false;
    bool elastic = false, demand_reclaim = false, deferred_upload = false;
    size_t safety_blocks = 0, margin_mib = 32, max_slots = 0;
    uint64_t fail_malloc_at = 0, fail_free_at = 0;
    bool mxfp4_batch = true, watermark = true;
    size_t low_mib = 128, high_mib = 448;
    bool global = false, count_policy = true, adaptive_admission = true;
    double min_count = 0, half_life = 2;
};
struct api {
    uint32_t version;
    size_t size;
    void * (*create)(uint64_t, const tier_options *, const cpu_kernels_v1 *, char *, size_t) noexcept;
};
using get_api = const api * (*)();
}  // namespace testing
}  // namespace gpu
}  // namespace ssd
