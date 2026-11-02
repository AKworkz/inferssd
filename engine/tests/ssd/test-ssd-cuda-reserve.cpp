// SPDX-License-Identifier: Apache-2.0
#include "ggml-cuda/ssd/reserve-budget.h"
#include "ggml-cuda/ssd/reserve-registry.h"
#include "ggml-ssd/runtime-services.h"

#include <cstdio>
#include <cstdlib>

static void require(bool condition) {
    if (!condition) { std::abort(); }
}

// A process on a 32 GiB card with 24 GiB free at its start, keeping 1 GiB and growing a cache until 1 GiB plus 64 MiB
// is left free, as the tier does.
static void reserve_is_kept_against_what_this_process_holds() {
    constexpr size_t mib = size_t{1} << 20;
    const size_t reserve = 1024 * mib, margin = 64 * mib, line = reserve + margin;
    ssd::gpu::reserve_budget budget;

    // Without a reserve the answer is the card's free memory, whatever is held or was seen.
    budget.hold(9000 * mib);
    budget.anchor(24000 * mib);
    require(budget.free_bytes(5 * mib, 0) == 5 * mib);
    require(budget.free_bytes(30000 * mib, 0) == 30000 * mib);

    // The start reads as the card.
    require(budget.free_bytes(24000 * mib, reserve) == 24000 * mib);
    const size_t cache = 24000 * mib - line;
    budget.hold(cache);
    require(budget.free_bytes(line, reserve) == line);

    // Another process takes the whole reserve: nothing changes for this one.
    budget.observe(margin);
    require(budget.free_bytes(margin, reserve) == line);
    // It also takes the margin: the answer is capped, and is still not below the reserve, so no block goes.
    budget.observe(0);
    require(budget.free_bytes(0, reserve) == reserve);
    // It releases all of that: the reserve is back, and nothing came free above it to grow into.
    budget.observe(line);
    require(budget.free_bytes(line, reserve) == line);

    // A compute buffer swaps with cache blocks: giving the blocks back makes the room, against this process only.
    budget.release(3000 * mib);
    budget.observe(margin + 3000 * mib);
    require(budget.free_bytes(margin + 3000 * mib, reserve) == line + 3000 * mib);
    budget.hold(3000 * mib);
    require(budget.free_bytes(margin, reserve) == line);

    // Memory this process takes beyond the margin reads below the reserve even with the card's free memory above it.
    budget.hold(100 * mib);
    require(budget.free_bytes(line - 100 * mib, reserve) == line - 100 * mib);
    budget.release(100 * mib);

    // A process that held memory at the start exits: what is above the line stays reachable after it is taken up.
    budget.observe(line + 4000 * mib);
    require(budget.free_bytes(line + 4000 * mib, reserve) == line + 4000 * mib);
    budget.hold(4000 * mib);
    require(budget.free_bytes(line, reserve) == line);
    budget.observe(margin);
    require(budget.free_bytes(margin, reserve) == line);

    // Allocations take whole device pages.
    require(ssd::gpu::device_pages(0) == 0 && ssd::gpu::device_pages(1) == 2 * mib);
    require(ssd::gpu::device_pages(2 * mib) == 2 * mib && ssd::gpu::device_pages(2 * mib + 1) == 4 * mib);
    require(ssd::gpu::device_pages(286 * mib + 7 * (mib / 8)) == 288 * mib);

    // anchor starts over from the card.
    budget.anchor(line);
    require(budget.reachable() == budget.held() + line);
}

int main() {
    reserve_is_kept_against_what_this_process_holds();
    ssd::gpu::reserve_registry<2> registry;
    const int first = 1, second = 2, other = 3, zero = 4;
    constexpr size_t mib = size_t{1} << 20;
    require(registry.maximum(0) == 0 && registry.maximum(1) == 0);
    require(registry.add(&first, 0, 1024 * mib) == ssd::status::ok);
    require(registry.add(&second, 0, 2048 * mib) == ssd::status::ok);
    require(registry.maximum(0) == 2048 * mib);
    require(registry.add(&other, 1, 4096 * mib) == ssd::status::ok);
    require(registry.maximum(0) == 2048 * mib && registry.maximum(1) == 4096 * mib);
    // Idempotent attach must not replace an owner's immutable registration.
    require(registry.add(&first, 0, 1024 * mib) == ssd::status::ok);
    require(registry.add(&first, 0, 3072 * mib) == ssd::status::invalid);
    require(registry.add(&first, 1, 1024 * mib) == ssd::status::invalid);
    require(registry.maximum(0) == 2048 * mib && registry.maximum(1) == 4096 * mib);
    require(registry.remove(&second) == ssd::status::ok && registry.maximum(0) == 1024 * mib);
    require(registry.remove(&second) == ssd::status::ok && registry.maximum(0) == 1024 * mib);
    require(registry.add(&zero, 0, 0) == ssd::status::ok);
    require(registry.remove(&first) == ssd::status::ok && registry.maximum(0) == 0);
    require(registry.remove(&zero) == ssd::status::ok && registry.maximum(0) == 0);
    require(registry.remove(&other) == ssd::status::ok && registry.maximum(1) == 0);
    require(registry.add(nullptr, 0, 1) == ssd::status::invalid);
    require(registry.add(&first, -1, 1) == ssd::status::invalid);
    require(registry.add(&first, 2, 1) == ssd::status::invalid);
    require(registry.remove(nullptr) == ssd::status::invalid);
    require(registry.maximum(-1) == 0 && registry.maximum(2) == 0);
    const ssd::runtime_services services{};
    require(services.gpu_reserve_bytes == 0);
    std::puts("SSD_CUDA_CONTEXT_RESERVE_PASS");
}
