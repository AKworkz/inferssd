// SPDX-License-Identifier: Apache-2.0
#include "options.h"
#include "llama-model.h"
#include "ggml-ssd/gpu/provider.h"
#include "ggml-ssd/gpu/transport.h"
#if defined(LLAMA_SSD_STORE)
#include "ggml-ssd/store/index.h"
#endif

#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <mutex>
#include <limits>
#if defined(__APPLE__)
#include <crt_externs.h>
#elif defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
extern char ** environ;
#endif

namespace {
// One factory in the llama DSO. Backend copies of core code never create a
// process singleton; every executor borrows this context's registration.
std::shared_ptr<ssd::host::process_arbiter> host_owner() {
    static std::mutex mutex;
    static std::weak_ptr<ssd::host::process_arbiter> current;
    std::lock_guard<std::mutex> lock(mutex);
    auto owner = current.lock();
    if (!owner) { owner = std::make_shared<ssd::host::process_arbiter>(); current = owner; }
    return owner;
}

void error_text(char * output, size_t size, const char * message) noexcept {
    if (output && size) { std::snprintf(output, size, "%s", message); }
}

// Inspect names only. Never copy or report environment values.
ssd::result<void> check_environment() {
    std::vector<std::string> names;
#if defined(_WIN32)
    struct environment {
        char * value = GetEnvironmentStringsA();
        ~environment() { if (value) { FreeEnvironmentStringsA(value); } }
    } block;
    if (!block.value) { return {ssd::status::io, "cannot enumerate environment names"}; }
    for (const char * entry = block.value; *entry; entry += std::strlen(entry) + 1) {
        const char * equals = std::strchr(entry, '=');
        if (equals && equals != entry) { names.emplace_back(entry, size_t(equals - entry)); }
    }
#else
#if defined(__APPLE__)
    char ** entries = *_NSGetEnviron();
#else
    char ** entries = environ;
#endif
    for (; entries && *entries; ++entries) {
        const char * equals = std::strchr(*entries, '=');
        if (equals) { names.emplace_back(*entries, size_t(equals - *entries)); }
    }
#endif
    return ssd::settings::reject_retired_environment(names);
}
} // namespace

llama_ssd_options * llama_ssd_options_create(const char * file,
        const llama_ssd_override * overrides, size_t count, const char * telemetry,
        char * error, size_t error_size) {
    error_text(error, error_size, "");
    try {
        if (count && !overrides) { throw std::invalid_argument("missing SSD overrides"); }
        const auto environment = check_environment();
        if (!environment) { error_text(error, error_size, environment.error.c_str()); return nullptr; }
        std::vector<ssd::settings::override_value> values;
        for (size_t i = 0; i < count; ++i) {
            if (!overrides[i].key || !overrides[i].value) { throw std::invalid_argument("null SSD override"); }
            values.push_back({overrides[i].key, overrides[i].value});
        }
#if defined(__APPLE__)
        const char * platform = "darwin";
#elif defined(_WIN32)
        const char * platform = "windows";
#else
        const char * platform = "linux";
#endif
        auto parsed = ssd::settings::load(file ? file : "", values, platform);
        if (!parsed) { error_text(error, error_size, parsed.error.c_str()); return nullptr; }
        auto debug = ssd::telemetry::read_debug_environment();
        if (!debug) { error_text(error, error_size, debug.error.c_str()); return nullptr; }
        return new llama_ssd_options{std::move(parsed.value), std::move(debug.value), telemetry ? telemetry : ""};
    } catch (const std::exception & exception) {
        error_text(error, error_size, exception.what());
    } catch (...) {
        error_text(error, error_size, "SSD options failed");
    }
    return nullptr;
}

void llama_ssd_options_free(llama_ssd_options * options) { delete options; }

bool llama_ssd_check_environment(char * error, size_t size) {
    error_text(error, size, "");
    try {
        const auto checked = check_environment();
        if (checked) { return true; }
        error_text(error, size, checked.error.c_str());
    } catch (const std::exception & exception) {
        error_text(error, size, exception.what());
    } catch (...) {
        error_text(error, size, "environment name check failed");
    }
    return false;
}

size_t llama_ssd_schema_json(char * output, size_t size) {
    if (output && size) { output[0] = '\0'; }
    const auto schema = ssd::settings::schema_json();
    if (!schema || schema.value.size() == SIZE_MAX) { return 0; }
    const size_t required = schema.value.size() + 1;
    if (output && size >= required) { std::memcpy(output, schema.value.c_str(), required); }
    return required;
}

llama_ssd_context::llama_ssd_context(const llama_ssd_options & options)
    : settings(options.settings), debug(options.debug), stream(services) {
    if (!settings.sources()->empty()) {
        host_owner = ::host_owner();
        auto registered = host_owner->make_client();
        if (!registered) { throw std::runtime_error(registered.error); }
        host = std::move(registered.value);
#if defined(LLAMA_SSD_STORE)
        auto opened = ssd::store::source_index::open(*settings.sources());
        if (!opened) { throw std::runtime_error(opened.error); }
        sources = opened.value->view();
        source_owner = std::unique_ptr<void, llama_ssd_source_delete>(opened.value.release(),
            llama_ssd_source_delete{[](void * value) noexcept { delete static_cast<ssd::store::source_index *>(value); }});
#else
        throw std::runtime_error("SSD experts require LLAMA_SSD_STORE=ON");
#endif
    }
    if (!options.telemetry_file.empty()) {
        auto opened = ssd::telemetry::writer::open(options.telemetry_file);
        if (!opened) { throw std::runtime_error(opened.error); }
        events = std::move(opened.value);
        const auto published = events->settings(settings);
        if (!published) { throw std::runtime_error(published.error); }
        const auto contract = events->event("ssd.controller.begin", "{\"schema\":1}");
        if (!contract) { throw std::runtime_error(contract.error); }
    }
    const auto diagnostics_bound = diagnostics.bind(debug, events.get());
    if (!diagnostics_bound) { throw std::runtime_error(diagnostics_bound.error); }
    const auto effective = settings.json();
    if (!effective) { throw std::runtime_error(effective.error); }
    std::fprintf(stderr, "[ssd.settings] %s\n", effective.value.c_str());
    services = {events.get(), &debug, host.get()};
    services.source = &sources;
    const auto * reserve_mib = settings.integer("vram.reserve_mib");
    constexpr size_t mib = size_t{1} << 20;
    if (!reserve_mib || *reserve_mib > std::numeric_limits<size_t>::max() / mib) {
        throw std::runtime_error("CUDA allocation reserve is not representable in bytes");
    }
    services.gpu_reserve_bytes = static_cast<size_t>(*reserve_mib) * mib;
}

void llama_ssd_context::attach_backend(ggml_backend_t backend) {
    if (!sources.resolve) { return; }
    const auto registry = ggml_backend_dev_backend_reg(ggml_backend_get_device(backend));
    const auto get = reinterpret_cast<ssd::gpu::get_provider_v1>(
        ggml_backend_reg_get_proc_address(registry, "ggml_backend_ssd_provider_v1"));
    // A backend without the provider (the CPU) has no GPU tier: a GPU there runs
    // the dense layers, and the CPU sink every routed expert.
    if (!get) { return; }
    if (services.gpu) { throw std::runtime_error("one GPU expert executor per context is supported"); }
    const auto * provider = get();
    if (!provider || provider->version != 1 || provider->size != sizeof(*provider) ||
            !provider->device_is_unified || !provider->device_ordinal || !provider->attach_services || !provider->stream) {
        throw std::runtime_error("GPU backend has no compatible SSD provider table");
    }
    const auto * transport = provider->transport;
    if (transport && (transport->version != 1 || transport->size != sizeof(*transport) ||
            transport->runtime_services_size != sizeof(services) || !transport->create ||
            !transport->quiesce || !transport->destroy || !transport->copy_ranges ||
            !transport->dense_has || !transport->dense_prepare || !transport->dense_read_ahead ||
            !transport->dense_copy || !transport->seed_register || !transport->seed_unregister ||
            !transport->seed_offer || !transport->seed_fence)) {
        throw std::runtime_error("GPU backend has no compatible context transport table");
    }
    const int device = provider->device_ordinal(backend);
    // This device of the backend has no expert executor (the graph lane needs host memory it can read).
    if (device < 0 && !transport) { return; }
    if (device < 0) { throw std::runtime_error("SSD GPU device is unavailable"); }
    services.gpu = provider;
    services.gpu_device = device;
    char error[512] {};
    if (transport) {
        gpu_transport = std::unique_ptr<void, llama_ssd_source_delete>(
            transport->create(&services, device, error, sizeof(error)),
            llama_ssd_source_delete{transport->destroy});
        if (!gpu_transport) { throw std::runtime_error(error[0] ? error : "cannot create context GPU transport"); }
        services.gpu_transport = gpu_transport.get();
    }
    if (!provider->attach_services(backend, &services, error, sizeof(error))) {
        throw std::runtime_error(error[0] ? error : "cannot attach context GPU transport");
    }
}

void llama_ssd_context::prepare_dense(const llama_model & model) {
    if (!services.gpu || !gpu_transport) { return; }
    size_t tensors = 0, bytes = 0;
    for (const auto & named : model.tensors_by_name) {
        const auto * tensor = named.second;
        if (ggml_n_dims(tensor) <= 2 && model.file_mapped(tensor) &&
                services.gpu->transport->dense_has(gpu_transport.get(), tensor->name, ggml_nbytes(tensor), nullptr)) {
            ++tensors;
        }
    }
    if (tensors && !services.gpu->transport->dense_prepare(gpu_transport.get(), &bytes)) {
        throw std::runtime_error("cannot reserve the context dense weight staging buffers");
    }
    if (events) {
        const auto published = events->event("ssd.stream.prepare",
            "{\"dense_tensors\":" + std::to_string(tensors) + ",\"pinned_bytes\":" + std::to_string(bytes) + "}");
        if (!published) { throw std::runtime_error(published.error); }
    }
}
