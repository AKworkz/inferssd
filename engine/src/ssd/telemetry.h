// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "ggml-ssd/telemetry/telemetry.h"

#include <atomic>
#include <cstring>

// L3 diagnostics borrow the context's writer. All mutable observation state is
// owned here and is destroyed before that writer; model values are never changed.
static_assert(sizeof(float) == 4, "logits diagnostics require the model F32 representation");

class llama_ssd_telemetry {
public:
    [[nodiscard]] ssd::result<void> bind(const ssd::telemetry::debug_channels & debug,
                                       ssd::telemetry::writer * writer) noexcept {
        if ((debug.trace || debug.profile) && !writer) {
            return fail(ssd::status::invalid, "trace/profile requires an owned telemetry writer");
        }
        writer_ = writer;
        trace_ = debug.trace;
        profile_ = debug.profile;
        return {};
    }

    bool profile_enabled() const noexcept { return profile_; }
    bool good() const noexcept { return !failed_.load(std::memory_order_relaxed); }

    [[nodiscard]] ssd::result<void> logits(const float * values, int32_t vocabulary,
                                          int32_t output_index) noexcept {
        if (!good()) { return fail(ssd::status::io, "context telemetry previously failed"); }
        if (!trace_) { return {}; }
        if (!values || vocabulary <= 0 || size_t(vocabulary) > SIZE_MAX / (sizeof(float) * 2)) {
            return fail(ssd::status::invalid, "invalid logits trace extent");
        }
        try {
            std::lock_guard<std::mutex> guard(mutex_);
            const size_t bytes = size_t(vocabulary) * sizeof(float);
            const auto * data = reinterpret_cast<const unsigned char *>(values);
            uint64_t hash = 1469598103934665603ull;
            for (size_t i = 0; i < bytes; ++i) { hash = (hash ^ data[i]) * 1099511628211ull; }
            if (have_hash_ && hash == last_hash_ && vocabulary == last_vocabulary_) { return {}; }
            static constexpr char digits[] = "0123456789abcdef";
            std::string hex(bytes * 2, '\0');
            for (size_t i = 0; i < bytes; ++i) {
                hex[2 * i] = digits[data[i] >> 4];
                hex[2 * i + 1] = digits[data[i] & 15];
            }
            const uint16_t endian = 1;
            const auto published = writer_->event("ssd.context.logits",
                "{\"output_index\":" + std::to_string(output_index) +
                ",\"vocabulary\":" + std::to_string(vocabulary) +
                ",\"hash_legacy_fnv64\":\"" + std::to_string(hash) +
                "\",\"encoding\":\"f32_native_hex\",\"byteorder\":\"" +
                (*reinterpret_cast<const unsigned char *>(&endian) ? "little" : "big") +
                "\",\"data\":\"" + hex + "\"}");
            if (!published) { failed_.store(true, std::memory_order_relaxed); return published; }
            have_hash_ = true;
            last_hash_ = hash;
            last_vocabulary_ = vocabulary;
            return {};
        } catch (const std::bad_alloc &) {
            return fail(ssd::status::resource, "oom");
        } catch (...) {
            return fail(ssd::status::io, "context logits trace failed");
        }
    }

    [[nodiscard]] ssd::result<void> graph(bool submit, uint64_t elapsed_us,
                                         uint32_t tokens, int status) noexcept {
        if (!good()) { return fail(ssd::status::io, "context telemetry previously failed"); }
        if (!profile_) { return {}; }
        try {
            const auto published = writer_->event(submit ? "ssd.context.graph_submit" : "ssd.context.graph_sync",
                "{\"elapsed_us\":" + std::to_string(elapsed_us) +
                (submit ? "" : ",\"tokens\":" + std::to_string(tokens)) + ",\"status\":" + std::to_string(status) + "}");
            if (!published) { failed_.store(true, std::memory_order_relaxed); }
            return published;
        } catch (const std::bad_alloc &) {
            return fail(ssd::status::resource, "oom");
        } catch (...) {
            return fail(ssd::status::io, "context graph trace failed");
        }
    }
    // Production controller observations are separate from trace/profile channels.
    [[nodiscard]] ssd::result<void> prefill_graph(uint32_t requested, uint32_t configured,
                                                uint32_t selected) noexcept {
        if (!good()) { return fail(ssd::status::io, "context telemetry previously failed"); }
        if (!writer_) { return {}; }
        try {
            const auto published = writer_->event("ssd.prefill.graph",
                "{\"requested_tokens\":" + std::to_string(requested) +
                ",\"configured_max\":" + std::to_string(configured) +
                ",\"selected_ubatch\":" + std::to_string(selected) + "}");
            if (!published) { failed_.store(true, std::memory_order_relaxed); }
            return published;
        } catch (...) { return fail(ssd::status::resource, "prefill observation failed"); }
    }

private:
    [[nodiscard]] ssd::result<void> fail(ssd::status code, const char * error) noexcept {
        failed_.store(true, std::memory_order_relaxed);
        try { return {code, error}; } catch (...) { return {code, {}}; }
    }
    ssd::telemetry::writer * writer_ = nullptr;
    bool trace_ = false, profile_ = false;
    std::atomic<bool> failed_{false};
    std::mutex mutex_;
    bool have_hash_ = false;
    uint64_t last_hash_ = 0;
    int32_t last_vocabulary_ = 0;
};
