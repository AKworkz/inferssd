// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "llama-ssd.h"
#include "telemetry.h"
#include "ggml-ssd/settings/settings.h"
#include "ggml-ssd/telemetry/telemetry.h"
#include "ggml-ssd/runtime-services.h"
#include "ggml-ssd/store/source.h"
#include "ggml-ssd/host/arbiter.h"
#include "ggml-ssd/stream/runtime.h"

#include <memory>

struct llama_ssd_options {
    ssd::settings::snapshot settings;
    ssd::telemetry::debug_channels debug;
    std::string telemetry_file;
};

struct llama_model;
struct llama_ssd_source_delete {
    void (*close)(void *) noexcept = nullptr;
    void operator()(void * source) const noexcept { if (source) { close(source); } }
};

// Declare this owner before executors in llama_context. Reverse destruction
// then joins every borrowed publisher before closing the event stream, also
// when construction of a later member throws.
struct llama_ssd_context {
    const ssd::settings::snapshot settings;
    const ssd::telemetry::debug_channels debug;
    std::shared_ptr<ssd::host::process_arbiter> host_owner;
    std::unique_ptr<ssd::host::context_client> host;
    std::unique_ptr<void, llama_ssd_source_delete> source_owner;
    ssd::store::source_view sources;
    std::unique_ptr<ssd::telemetry::writer> events;
    llama_ssd_telemetry diagnostics;
    ssd::runtime_services services;
    std::unique_ptr<void, llama_ssd_source_delete> gpu_transport;
    ssd::stream::runtime stream;

    explicit llama_ssd_context(const llama_ssd_options & options);
    void attach_backend(ggml_backend_t backend);
    void prepare_dense(const llama_model & model);
};

struct llama_ssd_host_reservation {
    ssd::host::allocation_ticket ticket;
};
