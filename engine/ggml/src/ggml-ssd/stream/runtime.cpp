// SPDX-License-Identifier: Apache-2.0
#include "runtime.h"
#include "../gpu/provider.h"
#include "../gpu/transport.h"
#include "../telemetry/telemetry.h"
#include <string>

namespace ssd { namespace stream {
const ggml_backend_sched_ssd_hooks runtime::hooks_{
    1, sizeof(ggml_backend_sched_ssd_hooks),
    [](void * p, const ggml_tensor * t) { return static_cast<runtime *>(p)->drive_home(t); },
    [](void * p, ggml_backend_t b, const ggml_tensor * src, ggml_tensor * dst,
            const ggml_tensor * next, uint64_t wait) {
        return static_cast<runtime *>(p)->copy_dense(b, src, dst, next, wait);
    },
    [](void * p, int phase) { auto & r = *static_cast<runtime *>(p);
        std::lock_guard<std::mutex> lock(r.mutex_); r.phase_ = phase >= 0 && phase <= 2 ? phase : 0; },
    [](void * p, ggml_backend_sched_kimi_copy_stats * out) { auto & r = *static_cast<runtime *>(p);
        std::lock_guard<std::mutex> lock(r.mutex_); *out = r.stats_; },
    [](void *) {} // Source geometry is immutable; counters survive graph replacement.
};
void runtime::attach(ggml_backend_sched_t scheduler) {
    ggml_backend_sched_set_ssd_runtime(scheduler, &hooks_, this);
}

bool runtime::drive_home(const ggml_tensor * tensor) const noexcept {
    const auto * gpu = services_.gpu;
    return gpu && services_.gpu_transport && tensor &&
        gpu->transport->dense_has(services_.gpu_transport, tensor->name, ggml_nbytes(tensor), nullptr);
}

bool runtime::publish(const char * name, uint64_t bytes, uint64_t read_us, uint64_t reads) noexcept {
    if (!services_.events) { return true; }
    try {
        return bool(services_.events->event(name, "{\"bytes\":" + std::to_string(bytes) +
            ",\"read_us\":" + std::to_string(read_us) + ",\"reads\":" + std::to_string(reads) +
            ",\"phase\":" + std::to_string(phase_) + "}"));
    } catch (...) { return false; }
}

bool runtime::copy_dense(ggml_backend_t backend, const ggml_tensor * source, ggml_tensor * destination,
        const ggml_tensor * next, uint64_t wait_us) noexcept {
    const auto * gpu = services_.gpu;
    if (!gpu || !services_.gpu_transport || gpu->device_ordinal(backend) != services_.gpu_device) { return false; }
    bool hit = false;
    const int64_t start = ggml_time_us();
    const bool copied = gpu->transport->dense_copy(services_.gpu_transport, source->name,
        destination->data, ggml_nbytes(destination), gpu->stream(backend), &hit);
    const uint64_t elapsed = uint64_t(ggml_time_us() - start);
    if (!copied) { return false; }
    // Retain the original single-weight lookahead and its transport ordering.
    if (next && !gpu->transport->dense_read_ahead(services_.gpu_transport, next->name, ggml_nbytes(next))) {
        return false;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    ++stats_.dense_tensors;
    stats_.dense_bytes += ggml_nbytes(source);
    stats_.dense_read_us += elapsed;
    stats_.dense_first_us += hit ? 0 : elapsed;
    stats_.dense_wait_us += wait_us;
    stats_.dense_readahead += hit ? 1 : 0;
    return publish("ssd.stream.dense", ggml_nbytes(source), elapsed, 1);
}
}}
