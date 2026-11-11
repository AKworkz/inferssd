#pragma once
#include "ggml-ssd/gpu/vram-tier-api.h"
// The routed experts' activation on the GPU, shared by the tier's engines: SwiGLU, SwiGLU with its clamp (an
// infinite limit leaves it unclamped), or SiTU with its optional linear soft cap. The CPU sink computes the same
// (compute_rows).
static __device__ __forceinline__ float kimi_vram_activate(float up, float gate, int kind, float situ_beta,
        float situ_linear_beta, float swiglu_limit) {
    if (kind == KIMI_VRAM_ACTIVATION_SWIGLU) {
        return gate / (1.0f + expf(-gate)) * up;
    }
    if (kind == KIMI_VRAM_ACTIVATION_SWIGLU_CLAMP) {
        const float g = fminf(gate, swiglu_limit);
        const float u = fminf(fmaxf(up, -swiglu_limit), swiglu_limit);
        return g / (1.0f + expf(-g)) * u;
    }
    const float a = (situ_beta * tanhf(gate / situ_beta)) / (1.0f + expf(-gate));
    const float u = situ_linear_beta > 0.0f ? situ_linear_beta * tanhf(up / situ_linear_beta) : up;
    return a * u;
}
