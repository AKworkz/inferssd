// SPDX-License-Identifier: MIT
#include "chat-templates.h"
#include "../chat-peg-parser.h"
#include "../common.h"
#include "../json-schema-to-grammar.h"
#include "../log.h"
#include <algorithm>
#include <string>
#include <vector>

namespace inferssd { namespace chat {
void configure_muse_caps(common_chat_template * default_template, common_chat_template * tool_use_template) {
    for (auto * tmpl : { default_template, tool_use_template }) {
        if (tmpl && tmpl->source().find("to=commentary") != std::string::npos &&
                tmpl->source().find("<|eom|>") != std::string::npos) {
            // Macro-heavy Muse templates defeat this fork's dynamic capability
            // probe. These capabilities are exercised in the Muse chat tests.
            auto & caps = tmpl->caps;
            caps.supports_string_content = caps.supports_typed_content = true;
            caps.supports_tools = caps.supports_tool_calls = caps.supports_parallel_tool_calls = true;
            caps.supports_system_role = caps.supports_preserve_reasoning = caps.supports_object_arguments = true;
        }
    }
}

// DeepSeek DSML tool calls. V3.2 writes <｜DSML｜function_calls>, <｜DSML｜invoke and <｜DSML｜parameter; V4.1 (v41)
// names the block " calls" and puts a space before every tag name: <｜DSML｜ calls>, <｜DSML｜ invoke, <｜DSML｜ parameter.
common_chat_params deepseek(const common_chat_template &    tmpl,
                                                                 const autoparser::generation_params & inputs,
                                                                 bool                                  v41, services shared) {
    const auto common_chat_template_direct_apply_impl = shared.apply;
    const auto foreach_function = shared.functions;
    common_chat_params data;

    data.prompt            = common_chat_template_direct_apply_impl(tmpl, inputs, std::nullopt, std::nullopt, std::nullopt, &data.prompt_input_spans);
    data.format            = COMMON_CHAT_FORMAT_PEG_NATIVE;
    data.supports_thinking = true;
    data.thinking_start_tag = "<think>";
    data.thinking_end_tag   = "</think>";
    data.preserved_tokens  = {
        "｜DSML｜",
        "<think>",
        "</think>",
    };

    auto has_tools           = inputs.tools.is_array() && !inputs.tools.empty();
    auto has_response_format = !inputs.json_schema.is_null() && inputs.json_schema.is_object();
    auto extract_reasoning   = inputs.reasoning_format != COMMON_REASONING_FORMAT_NONE;
    auto include_grammar     = has_response_format || (has_tools && inputs.tool_choice != COMMON_CHAT_TOOL_CHOICE_NONE);

    const std::string DSML         = "｜DSML｜";
    const std::string THINK_START  = "<think>";
    const std::string THINK_END    = "</think>";
    const std::string CALLS_TAG    = v41 ? " calls" : "function_calls";
    const std::string INVOKE_TAG   = v41 ? " invoke" : "invoke";
    const std::string PARAM_TAG    = v41 ? " parameter" : "parameter";
    const std::string FC_START     = "<" + DSML + CALLS_TAG + ">";
    const std::string FC_END       = "</" + DSML + CALLS_TAG + ">";
    const std::string INVOKE_START = "<" + DSML + INVOKE_TAG;
    const std::string INVOKE_END   = "</" + DSML + INVOKE_TAG + ">";
    const std::string PARAM_START  = "<" + DSML + PARAM_TAG;
    const std::string PARAM_END    = "</" + DSML + PARAM_TAG + ">";

    auto parser = build_chat_peg_parser([&](common_chat_peg_builder & p) {
        auto generation_prompt = p.prefix(inputs.generation_prompt, THINK_START);
        auto end = p.end();

        auto reasoning = p.eps();
        if (extract_reasoning && inputs.enable_thinking) {
            reasoning = p.optional(THINK_START + p.reasoning(p.until(THINK_END)) + THINK_END);
        } else if (extract_reasoning) {
            // Thinking disabled but reasoning extraction requested: the generation prompt
            // contains an empty <think></think> pair that must still be consumed.
            reasoning = p.optional(p.literal(THINK_START) + p.until(THINK_END) + p.literal(THINK_END));
        }

        if (has_response_format) {
            auto response_format = p.rule("response-format",
                p.literal("```json") + p.space() +
                p.content(p.schema(p.json(), "response-format-schema", inputs.json_schema)) +
                p.space() + p.literal("```"));
            return generation_prompt + reasoning + response_format + end;
        }

        if (!has_tools || inputs.tool_choice == COMMON_CHAT_TOOL_CHOICE_NONE) {
            return generation_prompt + reasoning + p.content(p.rest()) + end;
        }

        auto tool_choice = p.choice();
        foreach_function(inputs.tools, [&](const json & tool) {
            const auto & function = tool.at("function");
            std::string  name     = function.at("name");
            auto params   = function.contains("parameters") ? function.at("parameters") : json::object();
            const auto & props    = params.contains("properties") ? params.at("properties") : json::object();

            std::set<std::string> required;
            if (params.contains("required")) {
                params.at("required").get_to(required);
            }

            auto schema_info = common_schema_info();
            schema_info.resolve_refs(params);

            std::vector<common_peg_parser> required_parsers;
            std::vector<common_peg_parser> optional_parsers;
            for (const auto & [param_name, param_schema] : props.items()) {
                bool is_required = required.find(param_name) != required.end();
                bool is_string   = schema_info.resolves_to_string(param_schema);

                auto arg = p.tool_arg(
                    p.tool_arg_open(
                        p.literal(PARAM_START + " name=\"") +
                        p.tool_arg_name(p.literal(param_name)) +
                        p.literal("\" string=\"" + std::string(is_string ? "true" : "false") + "\">")) +
                    (is_string
                         ? (v41 ? p.tool_arg_raw_string_value(p.until(PARAM_END)) : p.tool_arg_string_value(p.until(PARAM_END)))
                         : p.tool_arg_json_value(p.schema(p.json(),
                                                          "tool-" + name + "-arg-" + param_name + "-schema",
                                                          param_schema, false))) +
                    p.tool_arg_close(p.literal(PARAM_END)));

                auto named_arg = p.rule("tool-" + name + "-arg-" + param_name, arg);
                if (is_required) {
                    required_parsers.push_back(named_arg);
                } else {
                    optional_parsers.push_back(named_arg);
                }
            }

            common_peg_parser args_seq = p.eps();
            for (size_t i = 0; i < required_parsers.size(); i++) {
                if (i > 0) {
                    args_seq = args_seq + p.space();
                }
                args_seq = args_seq + required_parsers[i];
            }

            if (!optional_parsers.empty()) {
                common_peg_parser any_opt = p.choice();
                for (const auto & opt : optional_parsers) {
                    any_opt |= opt;
                }
                args_seq = args_seq + p.repeat(p.space() + any_opt, 0, -1);
            }

            common_peg_parser invoke_body = args_seq;
            auto func_parser = p.tool(
                p.tool_open(p.literal(INVOKE_START + " name=\"") +
                            p.tool_name(p.literal(name)) + p.literal("\">\n")) +
                invoke_body + p.space() +
                p.tool_close(p.literal(INVOKE_END)));

            tool_choice |= p.rule("tool-" + name, func_parser);
        });

        auto require_tools = inputs.tool_choice == COMMON_CHAT_TOOL_CHOICE_REQUIRED;

        common_peg_parser tool_calls = p.eps();
        if (inputs.parallel_tool_calls) {
            tool_calls = p.trigger_rule("tool-call",
                p.literal(FC_START) + p.space() + tool_choice +
                p.zero_or_more(p.space() + tool_choice) + p.space() + p.literal(FC_END));
        } else {
            tool_calls = p.trigger_rule("tool-call",
                p.literal(FC_START) + p.space() + tool_choice + p.space() + p.literal(FC_END));
        }

        if (!require_tools) {
            tool_calls = p.optional(tool_calls);
        }

        // V4.1 writes the block after "\n\n", which its own parser counts as part of the start marker. Leaving it out
        // of the content keeps a re-rendered turn identical to the generated one, and holding it back while streaming
        // keeps streamed content from ending in newlines that turn out to precede a tool call.
        auto content_before_tools = v41
            ? p.content(p.until_one_of({ "\n\n" + FC_START, FC_START })) + p.optional(p.literal("\n\n"))
            : p.content(p.until(FC_START));
        return generation_prompt + reasoning + content_before_tools + tool_calls + end;
    });

    data.parser = parser.save();

    if (include_grammar) {
        data.grammar_lazy = !(has_response_format || (has_tools && inputs.tool_choice == COMMON_CHAT_TOOL_CHOICE_REQUIRED));
        data.grammar      = build_grammar([&](const common_grammar_builder & builder) {
            foreach_function(inputs.tools, [&](const json & tool) {
                const auto & function = tool.at("function");
                auto         schema   = function.contains("parameters") ? function.at("parameters") : json::object();
                builder.resolve_refs(schema);
            });
            if (has_response_format) {
                auto schema = inputs.json_schema;
                builder.resolve_refs(schema);
            }
            parser.build_grammar(builder, data.grammar_lazy);
        });

        data.grammar_triggers = {
            { COMMON_GRAMMAR_TRIGGER_TYPE_WORD, FC_START },
        };
    }

    return data;
}

// An assistant turn is rendered as one or more messages, each
// "<|start|>assistant to=<recipient><|message|>{content}{END}" where END is
// <|eom|> (more messages follow) or <|eot|> (end of turn):
//   - chain-of-thought:  to=self, terminated by <|eom|>
//   - user facing aside: to=commentary, terminated by <|eom|>
//   - final answer:      to=user, terminated by <|eot|>
// A commentary message carries content the user sees while the assistant keeps
// working, typically before or between tool calls.
// The generation prompt is just "<|start|>assistant"; the model emits its own
// " to=...<|message|>".
common_chat_params muse(const common_chat_template &          tmpl,
                                                               const autoparser::generation_params & inputs, services shared) {
    const auto common_chat_template_direct_apply_impl = shared.apply;
    const auto foreach_function = shared.functions;
    const auto foreach_parameter = shared.parameters;
    common_chat_params data;

    // Muse always thinks: its template has no switch to turn thinking off, only reasoning_effort.
    data.prompt            = common_chat_template_direct_apply_impl(tmpl, inputs, std::nullopt, std::nullopt, std::nullopt, &data.prompt_input_spans);
    data.generation_prompt = "<|start|>assistant";
    data.format            = COMMON_CHAT_FORMAT_PEG_NATIVE;
    data.supports_thinking = true;

    data.preserved_tokens = {
        "<|start|>", "<|message|>", "<|eom|>", "<|eot|>",
        // ATEM tool-call markup emitted on " to=<tool>" turns.
        "<atem:function_calls>", "<atem:invoke", "<atem:parameter", "</atem:parameter>",
        "</atem:invoke>", "</atem:function_calls>",
    };

    auto extract_reasoning = inputs.reasoning_format != COMMON_REASONING_FORMAT_NONE;

    auto has_tools = inputs.tools.is_array() && !inputs.tools.empty();
    // Constrained grammar whenever tools are offered.
    auto include_grammar = has_tools && inputs.tool_choice != COMMON_CHAT_TOOL_CHOICE_NONE;

    auto parser = build_chat_peg_parser([&](common_chat_peg_builder & p) {
        auto start = p.rule("start", p.literal("<|start|>assistant"));

        if (extract_reasoning) {
            p.rule("analysis", p.literal(" to=self<|message|>") + p.reasoning(p.until("<|eom|>")) + p.literal("<|eom|>"));
        } else {
            p.rule("analysis", p.literal(" to=self<|message|>") + p.content(p.until("<|eom|>")) + p.literal("<|eom|>"));
        }
        auto analysis = p.ref("analysis");

        // Commentary is user facing content emitted mid turn, so it reads as
        // content and always ends on <|eom|>.
        auto preamble = p.rule("preamble", p.literal(" to=commentary<|message|>") +
                                               p.content(p.until("<|eom|>")) + p.literal("<|eom|>"));

        auto aside = p.rule("aside", analysis | preamble);

        auto recipient  = p.optional(p.literal(" to=user"));
        auto final_msg  = p.rule("final", recipient + p.literal("<|message|>") +
                                              p.content(p.until_one_of({ "<|eot|>", "<|eom|>" })));

        if (has_tools && inputs.tool_choice != COMMON_CHAT_TOOL_CHOICE_NONE) {
            auto string_value = p.tool_arg_string_value(p.until("</atem:parameter>")) +
                p.tool_arg_close(p.literal("</atem:parameter>"));

            auto tool_choice = p.choice();
            foreach_function(inputs.tools, [&](const json & tool) {
                const auto &      function = tool.at("function");
                const std::string name     = function.at("name");

                common_schema_info schema_info;
                auto params = function.value("parameters", json::object());
                schema_info.resolve_refs(params);
                std::vector<common_peg_parser> arg_rules;
                foreach_parameter(function, [&](const std::string & param_name, const json & param_schema, bool) {
                    auto value_parser = p.eps();
                    if (schema_info.resolves_to_string(param_schema)) {
                        value_parser = string_value;
                    } else {
                        value_parser = p.tool_arg_json_value(
                                p.schema(p.json(), "tool-" + name + "-arg-" + param_name + "-schema", param_schema))
                            + p.tool_arg_close(p.literal("</atem:parameter>"));
                    }

                    arg_rules.push_back(p.tool_arg(
                        p.tool_arg_open(p.literal("<atem:parameter name=\"") + p.tool_arg_name(p.literal(param_name)) + p.literal("\">")) +
                        value_parser));
                });

                auto args = p.eps();
                if (!arg_rules.empty()) {
                    args = p.zero_or_more(p.choice(arg_rules) + p.space());
                }

                auto tool_parser = p.tool(
                    p.tool_open(p.literal(" to=") + p.until("<|message|>") +
                                p.literal("<|message|><atem:function_calls>") + p.space() +
                                p.literal("<atem:invoke name=\"") + p.tool_name(p.literal(name)) + p.literal("\">") + p.space())
                    << p.tool_args(args)
                    << p.tool_close(p.literal("</atem:invoke>") + p.space() + p.literal("</atem:function_calls>")));

                tool_choice |= p.rule("tool-" + name, tool_parser);
            });

            auto tool_calls = inputs.parallel_tool_calls
                ? p.trigger_rule("tool-call", tool_choice + p.zero_or_more(p.literal("<|eom|>") + start + tool_choice))
                : p.trigger_rule("tool-call", tool_choice);


            if (inputs.tool_choice == COMMON_CHAT_TOOL_CHOICE_REQUIRED) {
                return p.zero_or_more(start + aside) + start + tool_calls;
            }
            auto trailing_calls = p.optional(p.literal("<|eom|>") + start + tool_calls);
            return p.zero_or_more(start + aside) + start + (tool_calls | (final_msg + trailing_calls));
        }

        return p.zero_or_more(start + aside) + start + final_msg;
    });

    data.parser = parser.save();

    if (include_grammar) {
        data.grammar_lazy = inputs.tool_choice != COMMON_CHAT_TOOL_CHOICE_REQUIRED;
        data.grammar      = build_grammar([&](const common_grammar_builder & builder) {
            foreach_function(inputs.tools, [&](const json & tool) {
                auto schema = tool.at("function").value("parameters", json::object());
                builder.resolve_refs(schema);
            });
            parser.build_grammar(builder, data.grammar_lazy);
        });
        data.grammar_triggers = {
            { COMMON_GRAMMAR_TRIGGER_TYPE_PATTERN,
              "<\\|start\\|>assistant( to=(?!self<\\|message\\|>)(?!commentary<\\|message\\|>)(?!user<\\|message\\|>)[^<]*?<\\|message\\|>)" },
        };
    }

    return data;
}

}} // namespace inferssd::chat
