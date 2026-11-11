// Synthetic Q4_K transport geometry; included after vram-tier.cpp.
// Explicit owner metadata keeps the old 92-layer quota shape without the legacy parser.
#pragma once
#include <array>
static inline const kimi_vram_expert_format & lifecycle_format() {
    static const std::array<int32_t, 92> layers = [] {
        std::array<int32_t, 92> values{};
        for (size_t i = 0; i < values.size(); ++i) { values[i] = int32_t(i + 1); }
        return values;
    }();
    static const kimi_vram_expert_format format = {
        256, 256, int32_t(GGML_TYPE_Q4_K), int32_t(GGML_TYPE_Q4_K),
        256, int32_t(layers.size()), layers.data()
    };
    return format;
}
static inline size_t lifecycle_role_bytes() {
    return (ggml_row_size(GGML_TYPE_Q4_K, 256) * size_t(256) + 4095) & ~size_t(4095);
}
static inline void lifecycle_prepare(tier & target, const kimi_vram_request * requests, int count, int * slots) {
    const auto & format = lifecycle_format();
    target.prepare_format(requests, count, &format, slots);
}
static inline void lifecycle_launch(tier & target, const std::vector<float> & input) {
    const kimi_vram_activation activation = {KIMI_VRAM_ACTIVATION_SITU, 4.0f, 25.0f, 0.0f};
    target.launch_activation(input.data(), input.size() * sizeof(float), &activation);
}
