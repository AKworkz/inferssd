// SPDX-License-Identifier: MIT
#pragma once
#include "../chat.h"
#include "../chat-auto-parser.h"
#include <functional>
#include <optional>

namespace inferssd { namespace chat {
using json = nlohmann::ordered_json;
// Borrow the existing common helpers; no renderer/iterator copies or state.
using render = std::string (*)(const common_chat_template &, const autoparser::generation_params &,
    const std::optional<json> &, const std::optional<json> &, const std::optional<json> &,
    std::vector<llama_tokenize_input_span> *);
using for_each = void (*)(const json &, const std::function<void(const json &)> &);
using for_each_parameter = void (*)(const json &,
    const std::function<void(const std::string &, const json &, bool)> &);
struct services { render apply; for_each functions; for_each_parameter parameters; };
void configure_muse_caps(common_chat_template *, common_chat_template *);
common_chat_params deepseek(const common_chat_template &, const autoparser::generation_params &,
    bool v41, services);
common_chat_params muse(const common_chat_template &, const autoparser::generation_params &, services);
}} // namespace inferssd::chat
