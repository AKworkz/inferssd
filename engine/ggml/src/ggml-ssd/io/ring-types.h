// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstddef>
#include <cstdint>

namespace ssd { namespace io {
// The experts a CPU sink keeps in flight on its read ring, three role reads each.
constexpr int expert_ring_depth = 8;
// The request and completion shapes every read ring shares, whatever the OS.
// fd_direct comes from platform::open_direct.
struct read_range { int fd_direct; uint64_t offset; size_t nbytes; };
// res is the byte count read, or a negative errno code.
struct completion { uint64_t user_data; int32_t res; uint32_t flags; };
// A target registered for fixed reads; its index in the registered vector is
// the tag a fixed read passes.
struct buffer_span { void * base; size_t len; };
}} // namespace ssd::io
