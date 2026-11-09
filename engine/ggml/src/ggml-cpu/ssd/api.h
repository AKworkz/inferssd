// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "cpu-prefill.h"
#include "runtime.h"
#include "../../ggml-ssd/host/arbiter.h"
#include <cstddef>
#include <cstdint>

// A backend owns this immutable table and every callback it contains. Consumers
// borrow it only while that backend stays loaded, just as for its other procs.
namespace ssd { namespace cpu {
struct api_v2 {
    uint32_t version;   // 5: request_begin
    size_t size;
    decltype(&kimi_cpu_sink_source_size) source_size;
    decltype(&create_configured) create;
    decltype(&kimi_cpu_sink_run) run;
    decltype(&kimi_cpu_sink_run_activation) run_activation;
    decltype(&kimi_cpu_sink_destroy) destroy;
    decltype(&kimi_cpu_sink_call_open) call_open;
    decltype(&kimi_cpu_sink_call_close) call_close;
    decltype(&kimi_cpu_sink_call_join) call_join;
    decltype(&kimi_cpu_sink_decode_threads) decode_threads;
    decltype(&kimi_cpu_sink_cache_configure) cache_configure;
    decltype(&kimi_cpu_sink_cache_begin) cache_begin;
    decltype(&kimi_cpu_sink_request_begin) request_begin;
    decltype(&kimi_cpu_sink_vram_phase_complete) vram_phase_complete;
    decltype(&kimi_cpu_sink_cache_get_stats) cache_get_stats;
    decltype(&cache_make_room_configured) cache_make_room;
    decltype(&cache_prefill_room_configured) cache_prefill_room;
    decltype(&kimi_cpu_sink_prefetch_hint) prefetch_hint;
    decltype(&kimi_cpu_sink_vram_reserve) vram_reserve;
    decltype(&kimi_cpu_prefill_create) prefill_create;
    decltype(&kimi_cpu_prefill_run) prefill_run;
    decltype(&kimi_cpu_prefill_run_activation) prefill_run_activation;
    decltype(&kimi_cpu_prefill_destroy) prefill_destroy;
    ssd::result<ssd::host::allocation_ticket> (*take_room)(
        kimi_cpu_sink *, const ssd::settings::snapshot *, const ssd::runtime_services *,
        uint64_t, uint64_t, uint64_t) noexcept;
    bool (*attach_services)(ggml_backend_t, const ssd::runtime_services *) noexcept;
    size_t runtime_services_size;
};
using get_api_v2 = const api_v2 * (*)();

// Check the prefix before accessing members from a possibly different build.
inline bool valid(const api_v2 * api) noexcept {
    return api && api->version == 5 && api->size == sizeof(api_v2) &&
        api->runtime_services_size == sizeof(runtime_services) &&
        api->source_size &&
        api->create &&
        api->run &&
        api->run_activation &&
        api->destroy &&
        api->call_open && api->call_close && api->call_join && api->decode_threads &&
        api->cache_configure &&
        api->cache_begin && api->request_begin &&
        api->vram_phase_complete &&
        api->cache_get_stats &&
        api->cache_make_room &&
        api->cache_prefill_room &&
        api->prefetch_hint &&
        api->vram_reserve &&
        api->prefill_create &&
        api->prefill_run &&
        api->prefill_run_activation &&
        api->prefill_destroy &&
        api->take_room && api->attach_services;
}
} } // namespace ssd::cpu
