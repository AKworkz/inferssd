// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "../gpu/source.h"
#include <cstddef>
#include <cstdint>
namespace ssd { namespace host {
struct expert_request {
    kimi_cpu_sink_source role[3];
    float weight;
    int32_t id = -1;
    int cache_hit = -1;
    int cache_admit = -1;
    int vram_slot = -1;
    bool vram_transient = false;
    bool vram_ssd_transient = false;
    bool vram_ssd_streaming = false;
    int vram_ssd_staging = -1;
    double lane_start_us = 0.0; // MXFP4 grouped prefill: predicted start on its lane (CPU or GPU)
    int prefill_routes = 0;     // grouped prefill: the expert's distinct tokens
    bool cache_lease = false;
    bool trace_ram_resident_before = false;
    size_t io_slot = 0;
    uint64_t read_bytes = 0; // positive CQE bytes; only the I/O thread writes
    void * read_target[3] = {}; // fixed before publishing the I/O job
};
}} // namespace ssd::host
