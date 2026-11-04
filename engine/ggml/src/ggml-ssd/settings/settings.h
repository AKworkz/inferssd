// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "result.h"

#include <cstdint>
#include <map>
#include <string>
#include <variant>
#include <vector>

namespace ssd {
namespace settings {

enum class kind { integer, paths, cpus, costs, budget };

// The platforms a setting applies to, one bit each: the registry's scope column.
enum platform_scope : uint32_t {
    scope_linux   = 1u << 0,
    scope_darwin  = 1u << 1,
    scope_windows = 1u << 2,
    scope_all     = scope_linux | scope_darwin | scope_windows,
};

struct budget {
    bool     automatic;
    uint64_t bytes;

    // The GPU tier's byte budget: auto is all a GPU provider admits, and no
    // tier without one; an explicit budget stays as given.
    uint64_t tier_bytes(bool provider) const noexcept { return automatic ? (provider ? UINT64_MAX : 0) : bytes; }
};

using cost_table = std::map<std::string, double>;
using value      = std::variant<uint64_t, std::vector<std::string>, std::vector<uint32_t>, cost_table, budget>;

struct descriptor {
    const char * key;
    kind         type;
    uint64_t     minimum, maximum;
    const char * default_literal;
    const char * unit;
    const char * producer;
    uint32_t     scope;    // platform_scope bits
    const char * backend;  // the GPU backend whose expert tier the setting configures, or ""
    const char * description;
};

struct setting {
    value       data;
    std::string origin;    // cli | file | default
    std::string location;  // filename:line or command option
    std::string receipt;   // exact trailing file comment, without '#'
};

struct override_value {
    std::string key, literal;
};

// Copyable value object. All public accessors are const; contexts own a copy.
class snapshot {
  public:
    const setting *                   find(const std::string & key) const noexcept;
    const uint64_t *                  integer(const std::string & key) const noexcept;
    const std::vector<std::string> *  sources() const noexcept;
    const std::vector<uint32_t> *     cpus(const std::string & key) const noexcept;
    const cost_table *                costs() const noexcept;
    const budget *                    vram_budget() const noexcept;
    [[nodiscard]] result<std::string> json() const noexcept;  // values + source receipts
  private:
    std::map<std::string, setting> values_;
    friend result<snapshot>        parse(const std::string &,
                                         const std::string &,
                                         const std::vector<override_value> &,
                                         const std::string &) noexcept;
};

const std::vector<descriptor> &   registry();
[[nodiscard]] result<std::string> schema_json() noexcept;
[[nodiscard]] result<std::string> help() noexcept;
std::string                       cli_option(const std::string & key);
// Return false for a non-SSD option; reject unknown --ssd-* names explicitly.
[[nodiscard]] result<bool>        cli_override(const std::string &           option,
                                               const std::string &           literal,
                                               std::vector<override_value> & overrides) noexcept;
// Empty file_text and source together mean no settings file. A supplied file must start with
// 'inferssd-settings 1'. Unknown keys, duplicate file keys and invalid dependencies fail. The platform is
// "linux", "darwin" or "windows".
[[nodiscard]] result<snapshot>    parse(const std::string &                 file_text,
                                        const std::string &                 source,
                                        const std::vector<override_value> & overrides = {},
                                        const std::string &                 platform  = "linux") noexcept;
[[nodiscard]] result<snapshot>    load(const std::string &                 file,
                                       const std::vector<override_value> & overrides = {},
                                       const std::string &                 platform  = "linux") noexcept;
// Startup guard takes explicitly enumerated environment names; it never calls getenv.
[[nodiscard]] result<void>        reject_retired_environment(const std::vector<std::string> & names) noexcept;

}  // namespace settings
}  // namespace ssd
