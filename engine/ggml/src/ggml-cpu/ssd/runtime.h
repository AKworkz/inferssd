// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "cpu-sink.h"
#include "../../ggml-ssd/runtime-services.h"
#include "../../ggml-ssd/host/arbiter.h"

namespace ssd {
namespace settings {
class snapshot;
}
namespace cpu {
// Copy settings/debug before starting any thread; the read ring has eight
// expert slots. The resolver and writer remain borrowed until sink destruction.
[[nodiscard]] GGML_BACKEND_API kimi_cpu_sink * create_configured(
    const settings::snapshot * settings, const runtime_services * services,
    kimi_cpu_sink_resolver resolver, void * userdata, char * error, size_t error_size) noexcept;

// With no sink, apply the same context settings to the early graph's ledger.
[[nodiscard]] GGML_BACKEND_API bool cache_make_room_configured(
    kimi_cpu_sink * sink, const settings::snapshot * settings, const runtime_services * services,
    uint64_t host_bytes, uint64_t device_bytes) noexcept;
[[nodiscard]] GGML_BACKEND_API bool cache_prefill_room_configured(
    kimi_cpu_sink * sink, const settings::snapshot * settings, const runtime_services * services,
    uint64_t pages, uint64_t host_bytes, uint64_t device_bytes) noexcept;
// UINT64_MAX keeps the context prefill hold; 0 clears it. Success is the
// result status, including for a zero-byte allocation; keep ticket through first touch.
[[nodiscard]] GGML_BACKEND_API result<host::allocation_ticket> take_room_configured(
    kimi_cpu_sink * sink, const settings::snapshot * settings, const runtime_services * services,
    uint64_t host_bytes, uint64_t device_bytes, uint64_t prefill_pages) noexcept;
}  // namespace cpu
}  // namespace ssd
