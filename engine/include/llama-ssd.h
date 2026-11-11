// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "llama.h"

#ifdef __cplusplus
extern "C" {
#endif

// Immutable launch options. Creation copies the file, overrides and path;
// context creation copies the options again. The caller can then free them.
struct llama_ssd_options;
struct llama_ssd_override {
    const char * key;
    const char * value;
};

LLAMA_API struct llama_ssd_options * llama_ssd_options_create(
    const char * settings_file, const struct llama_ssd_override * overrides,
    size_t count, const char * telemetry_file, char * error, size_t error_size);
LLAMA_API void llama_ssd_options_free(struct llama_ssd_options * options);
// Startup guard, including callers which did not request SSD execution.
LLAMA_API bool llama_ssd_check_environment(char * error, size_t error_size);

// Returns required bytes including NUL, or zero on failure. A small output
// buffer is left empty; callers can first query with (NULL, 0).
LLAMA_API size_t llama_ssd_schema_json(char * output, size_t output_size);

// Request lifecycle transition; counts are tokens after prompt-cache reuse.
// Failure is reported without allowing C++ exceptions across the C boundary.
LLAMA_API bool llama_ssd_prefill_phase(struct llama_context * ctx, bool active,
    uint32_t prompt_tokens, uint32_t prefill_tokens);

// Once the load's passes are done and before any request: the expert caches filled from the drive (the GPU tier's
// empty slots and the RAM cache's), each routed layer's experts taken in their order, since no request has named one.
// What no request uses is as good as an empty slot to what allocates later. A context without SSD experts returns
// true and does nothing; false reports a failure, after which the caches fill as requests run.
LLAMA_API bool llama_ssd_preload(struct llama_context * ctx);

// Keep this reservation alive until allocation and first touch finish. NULL
// means refusal. A context without the Linux SSD executor returns a valid no-op
// handle. Finish exactly once before destroying the context; false cancels an
// unmaterialized allocation.
struct llama_ssd_host_reservation;
LLAMA_API struct llama_ssd_host_reservation * llama_ssd_host_reserve(
    struct llama_context * ctx, size_t bytes, char * error, size_t error_size);
LLAMA_API void llama_ssd_host_reservation_finish(
    struct llama_ssd_host_reservation * reservation, bool materialized);
// A known external free/replacement invalidates the current memory-learning
// interval. Contexts without a Linux SSD registration treat this as a no-op.
LLAMA_API void llama_ssd_host_note_memory_change(struct llama_context * ctx);

#ifdef __cplusplus
}
#endif
