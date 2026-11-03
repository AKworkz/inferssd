// SPDX-License-Identifier: MIT
#include "llama-model-loader.h"
#include "llama-hparams.h"
#include <algorithm>
#include <regex>
#include <stdexcept>
#include <cstdio>

void llama_coalesce_ranges(std::vector<llama_file_range> & ranges) {
    if (ranges.empty()) {
        return;
    }

    std::sort(ranges.begin(), ranges.end(), [] (const llama_file_range & a, const llama_file_range & b) {
        return a.first < b.first || (a.first == b.first && a.last < b.last);
    });

    std::vector<llama_file_range> merged;
    merged.reserve(ranges.size());
    for (const auto & range : ranges) {
        if (range.first >= range.last) {
            continue;
        }
        if (merged.empty() || range.first > merged.back().last) {
            merged.push_back(range);
        } else {
            merged.back().last = std::max(merged.back().last, range.last);
        }
    }
    ranges = std::move(merged);
}

bool llama_is_expert_tensor_name(const std::string & name, uint32_t & layer) {
    int parsed_layer = -1;
    if (sscanf(name.c_str(), "blk.%d.", &parsed_layer) != 1 || parsed_layer < 0) {
        return false;
    }

    static const std::regex expert_re(R"(^blk\.([0-9]+)\.ffn_(up|down|gate|gate_up)_(ch)?exps\.)");
    if (!std::regex_search(name, expert_re)) {
        return false;
    }

    layer = (uint32_t) parsed_layer;
    return true;
}

void llama_model_loader::build_expert_tensor_index(const llama_hparams & hparams) {
    expert_tensor_index = {};

    if (hparams.n_layer == 0 || hparams.n_expert == 0 || files.empty()) {
        return;
    }

    expert_tensor_index.file_ranges.resize(files.size());

    size_t deferred_bytes = 0;
    for (const auto & it : weights_map) {
        const std::string & name = it.first;
        const llama_tensor_weight & weight = it.second;

        uint32_t layer = 0;
        if (!llama_is_expert_tensor_name(name, layer)) {
            continue;
        }
        if (layer >= hparams.n_layer) {
            throw std::runtime_error(format("expert tensor '%s' has invalid layer index %u", name.c_str(), layer));
        }
        if (weight.idx >= expert_tensor_index.file_ranges.size()) {
            throw std::runtime_error(format("expert tensor '%s' has invalid file index %u", name.c_str(), (unsigned) weight.idx));
        }

        const size_t tensor_bytes = ggml_nbytes(weight.tensor);
        deferred_bytes += tensor_bytes;
        expert_tensor_index.file_ranges[weight.idx].push_back({ weight.offs, weight.offs + tensor_bytes });
    }

    for (auto & ranges : expert_tensor_index.file_ranges) {
        llama_coalesce_ranges(ranges);
    }

    expert_tensor_index.deferred_bytes = deferred_bytes;
    expert_tensor_index.dense_bytes = n_bytes > deferred_bytes ? n_bytes - deferred_bytes : 0;
}

bool llama_model_loader::mapping_range_encloses_experts(ggml_context * ctx) const {
    for (uint32_t idx = 0; idx < files.size(); ++idx) {
        size_t first = 0;
        size_t last = 0;
        void * addr = nullptr;
        get_mapping_range(&first, &last, &addr, int(idx), ctx);
        if (first >= last) {
            continue;
        }
        for (const auto & [name, weight] : weights_map) {
            uint32_t layer = 0;
            if (weight.idx == idx && weight.offs < last && first < weight.offs + ggml_nbytes(weight.tensor) &&
                    llama_is_expert_tensor_name(name, layer) && ggml_get_tensor(ctx, name.c_str()) == nullptr) {
                return true;
            }
        }
    }
    return false;
}

bool llama_model_loader::should_defer_expert_mmaps() const {
    return defer_experts && use_mmap && !no_alloc && !expert_tensor_index.empty();
}

void llama_model_loader::drop_mmap_expert_pages() const {
    if (!use_mmap || mappings.empty() || expert_tensor_index.file_ranges.empty()) {
        return;
    }

    const size_t n_range_sets = std::min(mappings.size(), expert_tensor_index.file_ranges.size());
    for (size_t idx = 0; idx < n_range_sets; ++idx) {
        for (const auto & range : expert_tensor_index.file_ranges[idx]) {
            mappings[idx]->dontneed_fragment(range.first, range.last);
        }
    }
}

