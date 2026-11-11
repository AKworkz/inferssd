// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "ggml-backend.h"
#include "../runtime-services.h"
#include <mutex>

namespace ssd { namespace stream {
// One request/context owner; the scheduler borrows the table and this object.
// No CUDA symbols, source descriptors, or mutable process-wide policy live here.
class runtime {
public:
    explicit runtime(const runtime_services & services) : services_(services) {}
    void attach(ggml_backend_sched_t scheduler);
private:
    static const ggml_backend_sched_ssd_hooks hooks_;
    const runtime_services & services_;
    std::mutex mutex_;
    ggml_backend_sched_kimi_copy_stats stats_{};
    int phase_ = 0;
    bool drive_home(const ggml_tensor *) const noexcept;
    bool copy_dense(ggml_backend_t, const ggml_tensor *, ggml_tensor *, const ggml_tensor *, uint64_t) noexcept;
    bool publish(const char *, uint64_t, uint64_t, uint64_t) noexcept;
};
}}
