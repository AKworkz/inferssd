// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "../settings/result.h"
#include "room.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <string>

namespace ssd { namespace host {

// Tracked bytes also appear in held_bytes() at interval boundaries. External
// allocations (compute/checkpoints) appear only in the separate grant counter.
enum class allocation_class { external, tracked };
struct client_config {
    bool unified = false;
    uint64_t reserve_bytes = 512ull << 20;
    uint64_t margin_bytes = 512ull << 20;
    uint64_t explicit_reserve_bytes = 0;
    bool reclaimable_file = true;
    // host.budget_mib in bytes: the macOS/Windows envelope against process memory; 0 derives it at startup.
    // Linux ignores it: its envelope is the cgroup's memory.max.
    uint64_t budget_bytes = 0;
};

// What the arbiter may ask of a context's expert caches before it refuses an
// admission of that context, in this order. The executor that owns the caches
// registers them with its context (context_client::set_reclaimer), so every
// take() and make_room() of the context runs them and no allocation has to name
// them. Only the requesting context's reclaimer runs, synchronously on the
// admitting thread and without the arbiter mutex: each step makes the caches
// safe to change itself.
struct actions {
    // Publishes the GPU tier's finished deferred uploads and returns their RAM leases.
    std::function<void()> collect_deferred;
    // Unified memory: the GPU tier gives blocks back for `bytes` more in both terms (false when it could not).
    std::function<bool(uint64_t bytes)> reclaim_tier;
    // The RAM cache gives slots back until room for `bytes` fits (false when it cannot).
    std::function<bool(uint64_t bytes)> reclaim_cache;
};
// Who an admission is for, which decides what the reclaimer gives back for it.
// An allocation outside the caches (compute and work buffers, scratch,
// checkpoints) has the GPU tier give room first on unified memory while a term
// is short, as the tier grows back into memory that is free again at a later
// decode token, then the RAM cache. The RAM cache's own growth has its colder
// slots give room. The GPU tier's growth takes free room only: it keeps the
// floor and the RAM cache's demand free itself.
enum class requester { outside, ram_cache, gpu_tier };

struct snapshot {
    bool configured = false, unified = false, reclaimable_file = false;
    uint64_t client_id = 0, reserve = 0, margin = 0, explicit_reserve = 0, floor = 0;
    uint64_t prefill_pages = 0, all_prefill_pages = 0, pending_bytes = 0, device_held = 0, all_device_held = 0;
    uint64_t file_reserve = 0, refaults = 0, file_updates = 0;
    uint64_t granted = 0, margin_updates = 0, overlap_skips = 0;
    std::string cgroup;
};

class process_arbiter;
class context_client;
namespace detail { struct client_state; }

// A successful take debits process pending bytes before returning. Keep this
// ticket until allocation AND required first touch complete. Commit releases
// pending accounting to the measured OS usage; cancellation also rolls back
// the unmaterialized grant. Both operations are idempotent and noexcept.
class allocation_ticket {
  public:
    allocation_ticket() noexcept = default;
    ~allocation_ticket();
    allocation_ticket(allocation_ticket &&) noexcept;
    allocation_ticket & operator=(allocation_ticket &&) noexcept;
    allocation_ticket(const allocation_ticket &) = delete;
    allocation_ticket & operator=(const allocation_ticket &) = delete;
    explicit operator bool() const noexcept { return id_ != 0; }
    [[nodiscard]] const room & observation() const noexcept { return observation_; }
    void commit() noexcept;
    void cancel() noexcept;
  private:
    friend class context_client;
    allocation_ticket(std::shared_ptr<process_arbiter>, std::shared_ptr<detail::client_state>,
                      uint64_t id, room observation) noexcept;
    void finish(bool materialized) noexcept;
    std::shared_ptr<process_arbiter> owner_;
    std::shared_ptr<detail::client_state> client_;
    uint64_t id_ = 0;
    room observation_;
};

// One strong owner belongs to L3. The factory obtaining that owner must live in
// one llama translation unit, never in an inline/header singleton or backend.
class process_arbiter : public std::enable_shared_from_this<process_arbiter> {
  public:
    using reader = std::function<bool(const std::string &, std::string &)>;
    explicit process_arbiter(reader injected_reader = {}) noexcept;
    ~process_arbiter();
    process_arbiter(const process_arbiter &) = delete;
    process_arbiter & operator=(const process_arbiter &) = delete;
    [[nodiscard]] result<std::unique_ptr<context_client>> make_client() noexcept;
  private:
    friend class context_client;
    friend class allocation_ticket;
    struct impl;
    std::unique_ptr<impl> impl_;
    status init_status_ = status::ok;
    std::string init_error_; // what the reader threw at construction, if it said
};

class context_client {
  public:
    ~context_client();
    context_client(const context_client &) = delete;
    context_client & operator=(const context_client &) = delete;
    // Configure before every admission path, including null-sink graph reserve.
    // Repeating the identical configuration is harmless; changing it returns
    // invalid. Clients must agree on the owner's physical DRAM/reclaim domain.
    [[nodiscard]] result<void> configure(const client_config &) noexcept;
    [[nodiscard]] result<snapshot> state() const noexcept;
    [[nodiscard]] result<void> set_prefill_pages(uint64_t) noexcept;
    // Unified memory: the device memory this context's caches hold in the host's DRAM (the GPU tier's blocks and
    // buffers), which the cgroup does not charge to the process. The envelope counts every context's as used, so the
    // GPU tier and the RAM cache share it. The owner publishes its total before it commits an allocation's ticket and
    // after it frees. Off unified memory device memory is not this DRAM and the value is ignored.
    [[nodiscard]] result<void> set_device_held(uint64_t) noexcept;
    [[nodiscard]] result<void> clear_prefill_pages() noexcept { return set_prefill_pages(0); }
    [[nodiscard]] result<uint64_t> dram_bytes(uint64_t host, uint64_t device) const noexcept;
    [[nodiscard]] result<room> measure(uint64_t extra, bool full_boost = false) const noexcept;
    [[nodiscard]] result<room> project_decode_capacity(uint64_t extra = 0, bool full_boost = false) const noexcept;
    // The reclaimer every admission of this context runs before it refuses. One
    // per context, registered by the executor that owns the caches once it can
    // run it; clearing waits for the reclaims running on other threads, so the
    // executor must not hold anything they wait for.
    [[nodiscard]] result<void> set_reclaimer(const actions &) noexcept;
    void clear_reclaimer() noexcept;
    // Observation/reclaim compatibility seam. It does not reserve a future
    // allocation. Production allocations must keep the returned take ticket.
    [[nodiscard]] result<room> make_room(uint64_t extra) noexcept;
    [[nodiscard]] result<allocation_ticket> take(uint64_t host, uint64_t device = 0,
                           requester who = requester::outside, bool require_finite_cgroup = false,
                           allocation_class classification = allocation_class::external) noexcept;
    [[nodiscard]] result<bool> update_file_reserve() noexcept;
    // The margin learns growth nobody accounts (unified memory), one decode token at a time. `held` is the bytes the
    // engine knows it holds in this DRAM (GPU tier blocks, committed RAM cache slots). An interval begins after a
    // decode token boundary has resized the caches and ends at the next boundary before it resizes anything, so the
    // boundaries' own resizes (and a MemAvailable that lags them) fall outside every interval, while a block the OOM
    // reclaimer gives a ggml allocation inside a token pairs with that allocation. Within the interval, a drop in
    // MemAvailable beyond what `held` and this context's committed grants explain was taken by something untracked:
    // CUDA graphs instantiated for a new shape, pool and driver allocations. When one interval's untracked drop
    // exceeds the margin, the margin becomes it; it never shrinks. An interval another client's admission or a
    // ticket still open overlaps learns nothing. MemAvailable is read at the full watermark boost, so that a boost the
    // kernel raises or drops within the token is not growth. consecutive=false (a prefill or a skipped token came
    // between) discards the open interval; end_interval is true when the margin grew.
    [[nodiscard]] result<void> begin_interval(uint64_t held) noexcept;
    [[nodiscard]] result<bool> end_interval(bool consecutive, uint64_t held, uint64_t & untracked) noexcept;
    // Call on known external frees/replacements: invalidate this actor and
    // peers for the current interval; cumulative grants cancel at the next boundary.
    [[nodiscard]] result<void> note_known_memory_change() noexcept;
    // Rejects new operations, then waits for in-flight reclaimers and tickets.
    // Executors must drain before L3 destroys this registration.
    void close() noexcept;
  private:
    friend class process_arbiter;
    context_client(std::shared_ptr<process_arbiter>, std::shared_ptr<detail::client_state>) noexcept;
    std::shared_ptr<process_arbiter> owner_;
    std::shared_ptr<detail::client_state> state_;
};

}} // namespace ssd::host
