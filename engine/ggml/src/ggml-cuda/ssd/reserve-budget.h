// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <algorithm>
#include <atomic>
#include <cstddef>

namespace ssd { namespace gpu {

// What an allocation takes from a discrete card: it is mapped in 2 MiB pages, so a tier block of 286.875 MiB takes 288.
[[nodiscard]] inline size_t device_pages(size_t bytes) noexcept {
    constexpr size_t page = size_t{2} << 20;
    return (bytes + page - 1) & ~(page - 1);
}

// Keeps a VRAM reserve on a discrete GPU against what this process holds. The card's free memory cannot say whose
// memory is whose: kept on that, a reserve would make this process give way to every other process's growth.
//
// `reachable` is the most the card has offered this process, what it held plus what was free. What this process has
// left is that less what it holds now, which another process taking memory does not change. One releasing memory
// raises `reachable`, so this process may use what came free above the reserve. Device memory nothing here counts
// (the driver's own, graph instantiation, kernels) comes out of the margin kept above the reserve.
//
// Without a reserve the answer is the card's free memory, and nothing here has any effect.
class reserve_budget {
  public:
    void hold(size_t bytes) noexcept { held_.fetch_add(bytes, std::memory_order_relaxed); }
    void release(size_t bytes) noexcept { held_.fetch_sub(bytes, std::memory_order_relaxed); }
    // Starts from what the card offers now.
    void anchor(size_t card_free) noexcept { reachable_.store(held() + card_free, std::memory_order_relaxed); }
    // Takes in what other processes released. Only where this process is not allocating or freeing: memory it is
    // halfway through either would count twice, and stay counted.
    void observe(size_t card_free) noexcept {
        const size_t now = held() + card_free;
        size_t seen = reachable_.load(std::memory_order_relaxed);
        while (seen < now && !reachable_.compare_exchange_weak(seen, now, std::memory_order_relaxed)) {}
    }
    // Free memory as this process should read it under `reserve`. Others may use up the reserve; past that the card
    // has nothing for this process to grow into, so the answer never exceeds what is really free plus the reserve.
    [[nodiscard]] size_t free_bytes(size_t card_free, size_t reserve) const noexcept {
        if (!reserve) { return card_free; }
        const size_t own = held();
        const size_t most = std::max(reachable(), own + card_free);
        return std::min(most - own, card_free + reserve);
    }
    [[nodiscard]] size_t held() const noexcept { return held_.load(std::memory_order_relaxed); }
    [[nodiscard]] size_t reachable() const noexcept { return reachable_.load(std::memory_order_relaxed); }

  private:
    std::atomic<size_t> held_{0}, reachable_{0};
};

}} // namespace ssd::gpu
