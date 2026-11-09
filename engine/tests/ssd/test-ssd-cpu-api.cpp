// SPDX-License-Identifier: Apache-2.0
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ssd/api.h"
#include <cstdio>

// Exercise the loaded CPU registry, including builds which hide the individual
// executor exports. Invalid tables must be rejected before any allocation/run.
int main() {
    const auto registry = ggml_backend_cpu_reg();
    const auto get = reinterpret_cast<ssd::cpu::get_api_v2>(
        ggml_backend_reg_get_proc_address(registry, "ggml_backend_cpu_ssd_api_v2"));
    const auto * api = get ? get() : nullptr;
    if (!ssd::cpu::valid(api) || api->source_size() != sizeof(kimi_cpu_sink_source)) {
        std::fputs("loaded CPU executor table is absent or incompatible\n", stderr);
        return 1;
    }
    auto changed = *api;
    changed.version = 1;
    if (ssd::cpu::valid(&changed)) { return 2; }
    changed = *api;
    changed.size -= sizeof(void *);
    if (ssd::cpu::valid(&changed)) { return 3; }
    changed = *api;
    changed.cache_make_room = nullptr;
    if (ssd::cpu::valid(&changed)) { return 4; }
    changed = *api;
    changed.prefill_destroy = nullptr;
    if (ssd::cpu::valid(&changed) || ssd::cpu::valid(nullptr)) { return 5; }
    changed = *api;
    --changed.runtime_services_size;
    if (ssd::cpu::valid(&changed)) { return 6; }
    changed = *api;
    changed.decode_threads = nullptr;
    if (ssd::cpu::valid(&changed)) { return 7; }
    changed = *api;
    changed.request_begin = nullptr;
    if (ssd::cpu::valid(&changed)) { return 8; }
    std::puts("CPU_EXECUTOR_REGISTRY_ABI_PASS");
    return 0;
}
