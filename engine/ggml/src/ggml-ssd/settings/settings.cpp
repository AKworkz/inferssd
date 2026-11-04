// SPDX-License-Identifier: Apache-2.0
#include "settings.h"

#include "json.h"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <locale>
#include <set>
#include <sstream>
#include <type_traits>
#include <utility>

namespace ssd {
namespace settings {
namespace {
template <class T> result<T> failure(status code, const std::string & message) {
    return { T{}, code, message };
}

template <> result<void> failure<void>(status code, const std::string & message) {
    return { code, message };
}

std::string trim(const std::string & text) {
    const size_t begin = text.find_first_not_of(" \t\r\n");
    return begin == std::string::npos ? "" : text.substr(begin, text.find_last_not_of(" \t\r\n") - begin + 1);
}

result<uint64_t> number(const std::string & text, uint64_t low, uint64_t high) {
    uint64_t value = 0;
    if (text.empty() || text[0] < '0' || text[0] > '9') {
        return failure<uint64_t>(status::invalid, "expected unsigned decimal integer");
    }
    const auto read = std::from_chars(text.data(), text.data() + text.size(), value);
    if (read.ec != std::errc() || read.ptr != text.data() + text.size() || value < low || value > high) {
        return failure<uint64_t>(status::invalid,
                                 "integer outside " + std::to_string(low) + ".." + std::to_string(high));
    }
    return { value, status::ok, {} };
}

result<std::vector<uint32_t>> cpu_list(const std::string & text) {
    std::vector<uint32_t> out;
    if (text.empty()) {
        return { out, status::ok, {} };
    }
    std::set<uint32_t> seen;
    size_t             at = 0;
    while (at <= text.size()) {
        const size_t      end  = text.find(',', at);
        const std::string item = text.substr(at, end == std::string::npos ? end : end - at);
        const size_t      dash = item.find('-');
        const auto        low  = number(item.substr(0, dash), 0, 2147483647);
        const auto        high = dash == std::string::npos ? low : number(item.substr(dash + 1), 0, 2147483647);
        if (!low || !high || high.value < low.value || high.value - low.value > 65535 ||
            out.size() + high.value - low.value + 1 > 65536) {
            return failure<std::vector<uint32_t>>(status::invalid, "invalid or excessive CPU list");
        }
        for (uint64_t cpu = low.value; cpu <= high.value; ++cpu) {
            if (!seen.insert(uint32_t(cpu)).second) {
                return failure<std::vector<uint32_t>>(status::invalid, "duplicate CPU ID");
            }
            out.push_back(uint32_t(cpu));
        }
        if (end == std::string::npos) {
            break;
        }
        at = end + 1;
    }
    return { std::move(out), status::ok, {} };
}

const std::set<std::string> & cost_names() {
    static const std::set<std::string> names = { "cpu_fixed_us",   "cpu_route_us",      "gpu_fixed_us",
                                                 "gpu_route_us",   "gpu_host_fixed_us", "gpu_host_route_us",
                                                 "link_gbps",      "pageable_gbps",     "drive_gbps",
                                                 "drive_fixed_us", "gpu_token_us" };
    return names;
}

result<cost_table> cost_values(const std::string & literal) {
    cost_table  out;
    std::string words = literal;
    std::replace(words.begin(), words.end(), ',', ' ');
    std::istringstream stream(words);
    std::string        word;
    while (stream >> word) {
        const size_t      equal = word.find('=');
        const std::string key   = word.substr(0, equal);
        if (equal == std::string::npos || !cost_names().count(key) || out.count(key)) {
            return failure<cost_table>(status::invalid, "unknown or duplicate cost term " + key);
        }
        std::istringstream numeric(word.substr(equal + 1));
        numeric.imbue(std::locale::classic());
        double v = 0;
        numeric >> std::noskipws >> v;
        if (!numeric || !numeric.eof() || !std::isfinite(v) || v < 0 || v > 1e6 || (key == "link_gbps" && v == 0)) {
            return failure<cost_table>(status::invalid, "invalid cost term " + key);
        }
        out.emplace(key, v);
    }
    return { std::move(out), status::ok, {} };
}

result<value> typed(const descriptor & d, std::string literal) {
    literal = trim(literal);
    if (!literal.empty() && literal.front() == '"' && d.type != kind::paths) {
        if (d.type == kind::integer) {
            return failure<value>(status::invalid, "integer must not be quoted");
        }
        const auto parsed = json::parse(literal);
        if (!parsed) {
            return failure<value>(parsed.code, parsed.error);
        }
        if (parsed.value.type != 's') {
            return failure<value>(status::invalid, "expected string literal");
        }
        literal = parsed.value.text;
        if (d.type == kind::budget && literal != "auto") {
            return failure<value>(status::invalid, "byte budget must not be quoted");
        }
    }
    if (d.type == kind::integer) {
        const auto n = number(literal, d.minimum, d.maximum);
        if (!n) {
            return failure<value>(n.code, n.error);
        }
        return { value(n.value), status::ok, {} };
    }
    if (d.type == kind::budget) {
        if (literal == "auto") {
            return { value(budget{ true, 0 }), status::ok, {} };
        }
        const auto n = number(literal, d.minimum, d.maximum);
        if (!n) {
            return failure<value>(n.code, n.error);
        }
        return { value(budget{ false, n.value }), status::ok, {} };
    }
    if (d.type == kind::cpus) {
        const auto ids = cpu_list(literal);
        if (!ids) {
            return failure<value>(ids.code, ids.error);
        }
        return { value(ids.value), status::ok, {} };
    }
    if (d.type == kind::costs) {
        const auto costs = cost_values(literal);
        if (!costs) {
            return failure<value>(costs.code, costs.error);
        }
        return { value(costs.value), status::ok, {} };
    }
    const auto paths = json::parse(literal);
    if (!paths) {
        return failure<value>(paths.code, paths.error);
    }
    if (paths.value.type != 'a') {
        return failure<value>(status::invalid, "expected JSON array of source paths");
    }
    std::vector<std::string> out;
    for (const auto & path : paths.value.array) {
        if (path.type != 's' || path.text.empty() || path.text.find('\0') != std::string::npos) {
            return failure<value>(status::invalid, "source paths must be nonempty strings without NUL");
        }
        out.push_back(path.text);
    }
    return { value(std::move(out)), status::ok, {} };
}

const descriptor * describe(const std::string & key) {
    for (const auto & d : registry()) {
        if (key == d.key) {
            return &d;
        }
    }
    return nullptr;
}

std::string value_json(const value & v) {
    return std::visit(
        [](const auto & item) {
            using T = std::decay_t<decltype(item)>;
            if constexpr (std::is_same_v<T, uint64_t>) {
                return std::to_string(item);
            } else if constexpr (std::is_same_v<T, budget>) {
                return item.automatic ? json::quote("auto") : std::to_string(item.bytes);
            } else if constexpr (std::is_same_v<T, cost_table>) {
                std::ostringstream out;
                out.imbue(std::locale::classic());
                out << std::setprecision(17) << '{';
                bool comma = false;
                for (const auto & pair : item) {
                    if (comma) {
                        out << ',';
                    }
                    comma = true;
                    out << json::quote(pair.first) << ':' << pair.second;
                }
                out << '}';
                return out.str();
            } else {
                std::string out   = "[";
                bool        comma = false;
                for (const auto & part : item) {
                    if (comma) {
                        out += ',';
                    }
                    comma = true;
                    if constexpr (std::is_same_v<T, std::vector<std::string>>) {
                        out += json::quote(part);
                    } else {
                        out += std::to_string(part);
                    }
                }
                return out + ']';
            }
        },
        v);
}

template <class Apply>
result<void> read_settings_file(const std::string & text, const std::string & source, Apply apply) {
    if (!text.empty() || !source.empty()) {
        std::istringstream stream(text);
        std::string        line;
        size_t             line_number = 1;
        if (!std::getline(stream, line) || trim(line) != "inferssd-settings 1") {
            return failure<void>(status::invalid, source + ":1: expected inferssd-settings 1");
        }
        std::set<std::string> seen;
        while (std::getline(stream, line)) {
            ++line_number;
            bool   quoted = false, escape = false;
            size_t comment = std::string::npos;
            for (size_t i = 0; i < line.size(); ++i) {
                if (escape) {
                    escape = false;
                    continue;
                }
                if (quoted && line[i] == '\\') {
                    escape = true;
                    continue;
                }
                if (line[i] == '"') {
                    quoted = !quoted;
                }
                if (!quoted && line[i] == '#') {
                    comment = i;
                    break;
                }
            }
            const std::string body = trim(line.substr(0, comment));
            if (body.empty()) {
                continue;
            }
            const auto        equal = body.find('=');
            const std::string key   = trim(body.substr(0, equal));
            if (equal == std::string::npos || !seen.insert(key).second) {
                return failure<void>(status::invalid, source + ':' + std::to_string(line_number) +
                                                          ": missing '=' or duplicate key " + key);
            }
            const auto done =
                apply(key, trim(body.substr(equal + 1)), "file", source + ':' + std::to_string(line_number),
                      comment == std::string::npos ? "" : line.substr(comment + 1));
            if (!done) {
                return failure<void>(done.code, done.error);
            }
        }
    }
    return {};
}

// The settings platforms with their scope bits, in schema order.
constexpr std::pair<uint32_t, const char *> platform_names[] = { { scope_linux, "linux" },
                                                                 { scope_darwin, "darwin" },
                                                                 { scope_windows, "windows" } };

// The scope bit of a settings platform; 0 when unknown.
uint32_t platform_bit(const std::string & platform) {
    for (const auto & item : platform_names) {
        if (platform == item.second) {
            return item.first;
        }
    }
    return 0;
}

// "all", or the platform names joined by commas: the schema's and the errors' spelling of a scope.
std::string scope_name(uint32_t scope) {
    if (scope == scope_all) {
        return "all";
    }
    std::string out;
    for (const auto & item : platform_names) {
        if (scope & item.first) {
            out += (out.empty() ? "" : ",") + std::string(item.second);
        }
    }
    return out;
}

}  // namespace

const std::vector<descriptor> & registry() {
    static const std::vector<descriptor> entries = {
#define SSD_SETTING(key, type, low, high, literal, unit, producer, scope, backend, description) \
    { key, kind::type, low##ULL, high##ULL, literal, unit, producer, scope, backend, description },
#include "registry.def"
#undef SSD_SETTING
    };
    return entries;
}

const setting * snapshot::find(const std::string & key) const noexcept {
    const auto it = values_.find(key);
    return it == values_.end() ? nullptr : &it->second;
}

const uint64_t * snapshot::integer(const std::string & key) const noexcept {
    const auto * v = find(key);
    return v ? std::get_if<uint64_t>(&v->data) : nullptr;
}

const std::vector<std::string> * snapshot::sources() const noexcept {
    const auto * v = find("expert.source");
    return v ? std::get_if<std::vector<std::string>>(&v->data) : nullptr;
}

const std::vector<uint32_t> * snapshot::cpus(const std::string & key) const noexcept {
    const auto * v = find(key);
    return v ? std::get_if<std::vector<uint32_t>>(&v->data) : nullptr;
}

const cost_table * snapshot::costs() const noexcept {
    const auto * v = find("place.costs");
    return v ? std::get_if<cost_table>(&v->data) : nullptr;
}

const budget * snapshot::vram_budget() const noexcept {
    const auto * v = find("vram.budget");
    return v ? std::get_if<budget>(&v->data) : nullptr;
}

result<snapshot> parse(const std::string &                 text,
                       const std::string &                 source,
                       const std::vector<override_value> & overrides,
                       const std::string &                 platform) noexcept {
    std::string where;
    try {
        where = source;
        if (!json::valid_utf8(text) || !json::valid_utf8(source)) {
            return failure<snapshot>(status::invalid, "settings source/text must be UTF-8");
        }
        const uint32_t target = platform_bit(platform);
        if (!target) {
            return failure<snapshot>(status::invalid, "unknown settings platform " + platform);
        }
        snapshot out;
        for (const auto & d : registry()) {
            const auto v = typed(d, d.default_literal);
            if (!v) {
                return failure<snapshot>(v.code, std::string(d.key) + " default: " + v.error);
            }
            out.values_.emplace(d.key, setting{ v.value, "default", "registry", "" });
        }
        auto apply = [&](const std::string & key, const std::string & literal, const std::string & origin,
                         const std::string & location, const std::string & receipt) -> result<void> {
            where          = key + " (" + location + ")";
            const auto * d = describe(key);
            if (!d) {
                return { status::invalid, where + ": unknown key" };
            }
            if (!(d->scope & target)) {
                return { status::invalid, where + ": setting is only supported on " + scope_name(d->scope) };
            }
            const auto v = typed(*d, literal);
            if (!v) {
                return { v.code, where + ": " + v.error };
            }
            out.values_[key] = setting{ v.value, origin, location, receipt };
            return {};
        };
        const auto read = read_settings_file(text, source, apply);
        if (!read) {
            return failure<snapshot>(read.code, read.error);
        }
        for (const auto & item : overrides) {
            const auto done = apply(item.key, item.literal, "cli", cli_option(item.key), "");
            if (!done) {
                return failure<snapshot>(done.code, done.error);
            }
        }
        return { std::move(out), status::ok, {} };
    } catch (const std::bad_alloc &) {
        return failure<snapshot>(status::resource, "oom");
    } catch (const std::exception & error) {
        return failure<snapshot>(status::invalid, where + ": " + error.what());
    }
}

result<snapshot> load(const std::string &                 file,
                      const std::vector<override_value> & overrides,
                      const std::string &                 platform) noexcept {
    try {
        if (file.empty()) {
            return parse("", "", overrides, platform);
        }
        std::ifstream stream(file, std::ios::binary);
        if (!stream) {
            return failure<snapshot>(status::io, file + ": cannot open settings file");
        }
        const std::string text((std::istreambuf_iterator<char>(stream)), std::istreambuf_iterator<char>());
        if (stream.bad()) {
            return failure<snapshot>(status::io, file + ": cannot read settings file");
        }
        return parse(text, file, overrides, platform);
    } catch (const std::bad_alloc &) {
        return failure<snapshot>(status::resource, "oom");
    } catch (const std::exception & error) {
        return failure<snapshot>(status::io, error.what());
    }
}

std::string cli_option(const std::string & key) {
    std::string out = "--ssd-" + key;
    std::replace(out.begin(), out.end(), '.', '-');
    std::replace(out.begin(), out.end(), '_', '-');
    return out;
}

result<bool> cli_override(const std::string &           option,
                          const std::string &           literal,
                          std::vector<override_value> & overrides) noexcept {
    try {
        for (const auto & d : registry()) {
            if (option == cli_option(d.key)) {
                overrides.push_back({ d.key, literal });
                return { true, status::ok, {} };
            }
        }
        if (option.compare(0, 6, "--ssd-") == 0) {
            return failure<bool>(status::invalid, "unknown setting option " + option);
        }
        return { false, status::ok, {} };
    } catch (const std::bad_alloc &) {
        return failure<bool>(status::resource, "oom");
    }
}

result<std::string> snapshot::json() const noexcept {
    try {
        std::string values = "{", sources = "{";
        bool        comma = false;
        for (const auto & d : registry()) {
            const auto * item = find(d.key);
            if (!item) {
                return failure<std::string>(status::invalid, "incomplete settings snapshot");
            }
            if (comma) {
                values += ',';
                sources += ',';
            }
            comma                 = true;
            const std::string key = json::quote(d.key);
            values += key + ':' + value_json(item->data);
            sources += key + ":{\"origin\":" + json::quote(item->origin) +
                       ",\"location\":" + json::quote(item->location) + ",\"receipt\":" + json::quote(item->receipt) +
                       '}';
        }
        return { "{\"values\":" + values + "},\"sources\":" + sources + "}}", status::ok, {} };
    } catch (const std::bad_alloc &) {
        return failure<std::string>(status::resource, "oom");
    }
}

result<std::string> schema_json() noexcept {
    try {
        std::string out =
            "{\"$schema\":\"https://json-schema.org/draft/2020-12/schema\",\"title\":\"InferSSD settings "
            "v1\",\"type\":\"object\",\"additionalProperties\":false,\"properties\":{";
        bool comma = false;
        for (const auto & d : registry()) {
            if (comma) {
                out += ',';
            }
            comma             = true;
            const char * type = d.type == kind::integer ? "integer" :
                                d.type == kind::budget  ? "budget" :
                                d.type == kind::costs   ? "costs" :
                                d.type == kind::paths   ? "paths" :
                                                          "cpus";

            out += json::quote(d.key) + ":{\"description\":" + json::quote(d.description) +
                   ",\"kind\":" + json::quote(type) + ",\"unit\":" + json::quote(d.unit) +
                   ",\"producer\":" + json::quote(d.producer) + ",\"scope\":" + json::quote(scope_name(d.scope)) +
                   (*d.backend ? ",\"backend\":" + json::quote(d.backend) : std::string()) +
                   ",\"cli\":" + json::quote(cli_option(d.key)) +
                   ",\"default_literal\":" + json::quote(d.default_literal);
            if (d.type == kind::integer) {
                out += ",\"type\":\"integer\",\"minimum\":" + std::to_string(d.minimum) +
                       ",\"maximum\":" + std::to_string(d.maximum);
            } else if (d.type == kind::budget) {
                out += ",\"oneOf\":[{\"const\":\"auto\"},{\"type\":\"integer\",\"minimum\":0,\"maximum\":" +
                       std::to_string(d.maximum) + "}]";
            } else if (d.type == kind::costs) {
                out += ",\"type\":\"object\",\"additionalProperties\":false,\"properties\":{";
                bool separator = false;
                for (const auto & name : cost_names()) {
                    if (separator) {
                        out += ',';
                    }
                    separator = true;
                    out += json::quote(name) + ":{\"type\":\"number\",\"maximum\":1000000," +
                           (name == "link_gbps" ? "\"exclusiveMinimum\":0}" : "\"minimum\":0}");
                }
                out += '}';
            } else if (d.type == kind::paths) {
                out += ",\"type\":\"array\",\"items\":{\"type\":\"string\",\"minLength\":1}";
            } else {
                out +=
                    ",\"type\":\"array\",\"uniqueItems\":true,\"maxItems\":65536,\"items\":{\"type\":\"integer\","
                    "\"minimum\":0,\"maximum\":2147483647}";
            }
            const auto v = typed(d, d.default_literal);
            out += ",\"default\":" + value_json(v.value) + '}';
        }
        return { out + "}}", status::ok, {} };
    } catch (const std::bad_alloc &) {
        return failure<std::string>(status::resource, "oom");
    } catch (const std::exception & e) {
        return failure<std::string>(status::invalid, e.what());
    }
}

result<std::string> help() noexcept {
    try {
        std::string out;
        for (const auto & d : registry()) {
            out += cli_option(d.key) + " VALUE  " + d.description + " (default " + d.default_literal + ")\n";
        }
        return { out, status::ok, {} };
    } catch (const std::bad_alloc &) {
        return failure<std::string>(status::resource, "oom");
    }
}

result<void> reject_retired_environment(const std::vector<std::string> & names) noexcept {
    try {
        for (const auto & name : names) {
            bool retired = false;
            for (const char * prefix :
                 { "GGML_KIMI_", "GGML_MOE_", "LLAMA_K3_", "LLAMA_DSV41_", "DS4_", "SSD_LLAMA_MAC_" }) {
                retired |= name.compare(0, std::char_traits<char>::length(prefix), prefix) == 0;
            }
            if (name.compare(0, 4, "TASK") == 0) {
                size_t i = 4;
                while (i < name.size() && name[i] >= '0' && name[i] <= '9') {
                    ++i;
                }
                retired |= i > 4 && i < name.size() && name[i] == '_';
            }
            if (retired) {
                return { status::invalid,
                         "retired environment variable " + name + "; use typed settings, see docs/configuration.md" };
            }
        }
        return {};
    } catch (const std::bad_alloc &) {
        return { status::resource, "oom" };
    }
}

}  // namespace settings
}  // namespace ssd
