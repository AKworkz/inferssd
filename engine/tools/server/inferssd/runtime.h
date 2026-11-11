// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "llama-ssd.h"
#include "../server-common.h"
#include <cstring>

// Correctness lifecycle uses the normal C API for every context. Contexts
// without SSD services are handled by that API; no environment or weak lookup
// selects the phase transition.
inline bool server_moe_workspace_transition(
        llama_context * ctx,
        const char * label,
        int64_t task_id,
        uint32_t prompt_tokens,
        uint32_t prefill_tokens) {
    const bool begin = label && std::strcmp(label, "request_begin") == 0;
    const bool end = label && (std::strcmp(label, "prefill_done") == 0 ||
        std::strcmp(label, "request_end") == 0);
    if (!begin && !end) { return true; }
    if (llama_ssd_prefill_phase(ctx, begin, begin ? prompt_tokens : 0, begin ? prefill_tokens : 0)) {
        return true;
    }
    SRV_WRN("SSD phase transition failed: label=%s task=%lld\n", label, (long long)task_id);
    return false;
}

// At load, once the load's passes are done (llama_ssd_preload). A failure leaves the caches to fill as requests run.
inline void server_ssd_preload(llama_context * ctx) {
    if (!llama_ssd_preload(ctx)) { SRV_WRN("%s", "SSD expert preload failed; the caches fill as requests run\n"); }
}

inline bool server_moe_phase_report(
        llama_context * ctx,
        const char * label,
        int64_t task_id,
        uint32_t prompt_tokens = 0,
        bool transition_workspace = true) {
    return !transition_workspace || server_moe_workspace_transition(
        ctx, label, task_id, prompt_tokens, prompt_tokens);
}
