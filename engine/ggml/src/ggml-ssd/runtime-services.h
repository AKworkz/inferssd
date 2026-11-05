// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstddef>

namespace ssd {
namespace store { struct source_view; }
namespace gpu { struct provider_v1; }
namespace host { class context_client; }
namespace telemetry {
class writer;
struct debug_channels;
}

// The runtime owns these services. Executors copy the debug selection and borrow
// the writer only until their workers and publishers have all joined.
struct runtime_services {
    telemetry::writer * events = nullptr;
    const telemetry::debug_channels * debug = nullptr;
    host::context_client * host = nullptr;
    const gpu::provider_v1 * gpu = nullptr;
    int gpu_device = 0;
    const store::source_view * source = nullptr;
    void * gpu_transport = nullptr;
    size_t gpu_reserve_bytes = 0;
};
}  // namespace ssd
