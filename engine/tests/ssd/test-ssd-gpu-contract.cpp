// SPDX-License-Identifier: Apache-2.0
#undef NDEBUG
#include "ggml-ssd/gpu/provider.h"
#include <cassert>

int main() {
    ssd::platform::file_identity file;
    file.device = 7; file.id_lo = 11; file.id_hi = 29; file.size = 4096;
    file.mtime_ns = 13000000017; file.ctime_ns = -1000000000 + 23; // before the epoch: -1 s + 23 ns
    const auto source = kimi_vram_source_identity(file, 32, 64);
    assert(source.device == 7 && source.inode == 11 && source.inode_hi == 29 && source.size == 4096 &&
           source.offset == 32 && source.bytes == 64);
    // Seconds and nanoseconds as POSIX stat splits them.
    assert(source.mtime_s == 13 && source.mtime_ns == 17 && source.ctime_s == -1 && source.ctime_ns == 23);
    file.ctime_ns = -1; // 1 ns before the epoch
    const auto early = kimi_vram_source_identity(file, 32, 64);
    assert(early.ctime_s == -1 && early.ctime_ns == 999999999);
}
