// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "../runtime-services.h"
#include <cstddef>
#include <cstdint>

namespace ssd { namespace gpu {
// Existing transport operations with an explicit context owner. CUDA pointers
// are opaque here; L1 owns no CUDA resources. All source descriptors are borrowed.
struct transport_v1 {
    uint32_t version;
    size_t size;
    size_t runtime_services_size;
    void * (*create)(const runtime_services *, int device, char *, size_t) noexcept;
    void (*quiesce)(void *) noexcept;
    void (*destroy)(void *) noexcept;
    bool (*dense_has)(void *, const char *, size_t, size_t *) noexcept;
    bool (*dense_prepare)(void *, size_t *) noexcept;
    bool (*dense_read_ahead)(void *, const char *, size_t) noexcept;
    bool (*dense_copy)(void *, const char *, void *, size_t, void *, bool *) noexcept;
};
}} // namespace ssd::gpu
