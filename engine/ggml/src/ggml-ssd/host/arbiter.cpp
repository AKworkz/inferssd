// SPDX-License-Identifier: Apache-2.0
#include "arbiter.h"
#include <algorithm>
#include <condition_variable>
#include <map>
#include <mutex>
#include <new>
#include <stdexcept>
#include <utility>

// One arbiter on every OS; only the budget sources differ.
#if defined(__linux__)
#include "ledger.h" // cgroup v2 and /proc
namespace ssd { namespace host { using host_ledger = ::kimi_host_ledger; } }
#else
#include "ledger-native.h" // job objects or host.budget_mib, and the OS memory status
namespace ssd { namespace host { using host_ledger = native_ledger; } }
#endif

namespace ssd { namespace host {
namespace {
template<class T> result<T> passed(T value) { return {std::move(value), status::ok, {}}; }
template<class T> result<T> failed(status code, const char * message) { return {T{}, code, message}; }
template<> result<void> failed<void>(status code, const char * message) { return {code, message}; }
template<class T, class F> result<T> guarded(F && call) noexcept {
    try { return call(); }
    catch (const std::bad_alloc &) { return failed<T>(status::resource, "host allocation"); }
    catch (const std::bad_weak_ptr &) { return failed<T>(status::invalid, "host ownership"); }
    catch (const std::exception & error) {
        // The error the run ends in names what the callback threw, a reclaimer's fault for one.
        try { return failed<T>(status::io, (std::string("host callback: ") + error.what()).c_str()); }
        catch (...) { return failed<T>(status::io, "host callback"); }
    }
    catch (...) { return failed<T>(status::io, "host callback"); }
}
bool same_config(const client_config & a, const client_config & b) noexcept {
    return a.unified == b.unified && a.reserve_bytes == b.reserve_bytes &&
        a.margin_bytes == b.margin_bytes && a.explicit_reserve_bytes == b.explicit_reserve_bytes &&
        a.reclaimable_file == b.reclaimable_file && a.budget_bytes == b.budget_bytes;
}
}
namespace detail {
struct client_state {
    uint64_t id = 0;
    bool configured = false, closing = false, closed = false;
    client_config config;
    uint64_t prefill_pages = 0, committed_grants = 0, pending = 0, external_pending = 0, device_held = 0;
    size_t operations = 0, tickets = 0;
    bool interval_open = false, interval_overlap = false;
    uint64_t interval_available = 0, interval_known = 0, overlap_skips = 0;
    std::shared_ptr<const actions> reclaimer; // set_reclaimer
    size_t reclaims = 0;                      // reclaimer runs in progress, outside the mutex
};
}
struct process_arbiter::impl {
    std::mutex mutex;
    std::condition_variable changed;
    host_ledger budget;
    uint64_t next_client = 1, next_ticket = 1, pending = 0, margin_updates = 0;
    bool domain_set = false, domain_unified = false, domain_reclaimable = true;
    std::map<uint64_t, std::shared_ptr<detail::client_state>> clients;
    struct pending_ticket {
        std::shared_ptr<detail::client_state> client;
        uint64_t bytes;
        allocation_class classification;
    };
    std::map<uint64_t, pending_ticket> tickets;

    explicit impl(reader injected) {
#if defined(__linux__)
        // The Linux ledger reads /proc and cgroup v2 files, through the injected reader in tests.
        budget.read = std::move(injected);
        budget.locate();
#else
        (void) injected; // the native ledger asks the OS, not files
#endif
    }
    static const char * invalid(const detail::client_state & state, bool configured = true) noexcept {
        if (state.closing || state.closed) { return "host context is closing"; }
        if (configured && !state.configured) { return "host context is not configured"; }
        return nullptr;
    }
    struct operation {
        impl & owner;
        std::shared_ptr<detail::client_state> state;
        const char * error = nullptr;
        operation(impl & owner_, std::shared_ptr<detail::client_state> state_, bool configured = true)
            : owner(owner_), state(std::move(state_)) {
            std::lock_guard<std::mutex> lock(owner.mutex);
            error = invalid(*state, configured);
            if (!error) { ++state->operations; }
        }
        ~operation() {
            if (!error) {
                std::lock_guard<std::mutex> lock(owner.mutex);
                --state->operations; owner.changed.notify_all();
            }
        }
    };
    void invalidate_peers(uint64_t actor) noexcept {
        for (const auto & pair : clients) {
            auto & state = *pair.second;
            if (state.id != actor && state.interval_open) { state.interval_overlap = true; }
        }
    }
    uint64_t pages(uint64_t excluding = 0) const noexcept {
        uint64_t total = 0;
        for (const auto & pair : clients) {
            if (pair.first != excluding) { total = add(total, pair.second->prefill_pages); }
        }
        return total;
    }
    // What the cgroup does not charge: the caches' device memory, on unified memory only.
    uint64_t held_device() const noexcept {
        uint64_t total = 0;
        if (!budget.unified) { return 0; }
        for (const auto & pair : clients) { total = add(total, pair.second->device_held); }
        return total;
    }
    void refresh_config() noexcept {
        uint64_t reserve = 512ull << 20, margin = reserve, explicit_reserve = 0, envelope = 0;
        bool unified = domain_unified, reclaimable = domain_reclaimable;
        for (const auto & pair : clients) {
            if (!pair.second->configured) { continue; }
            const auto & config = pair.second->config;
            unified = config.unified; reclaimable = config.reclaimable_file;
            reserve = std::max(reserve, config.reserve_bytes);
            margin = std::max(margin, config.margin_bytes);
            explicit_reserve = std::max(explicit_reserve, config.unified ? config.explicit_reserve_bytes : 0);
            // The tightest explicit envelope binds every client of the process.
            if (config.budget_bytes) { envelope = envelope ? std::min(envelope, config.budget_bytes) : config.budget_bytes; }
        }
        budget.unified = unified; budget.reclaimable_file = reclaimable;
        budget.reserve = reserve; budget.margin = std::max(budget.margin, std::max(margin, reserve));
        budget.explicit_reserve = explicit_reserve;
#if defined(__linux__)
        (void) envelope; // the cgroup's memory.max is the Linux envelope
#else
        budget.budget = envelope;
#endif
    }
    room measure(uint64_t extra, bool full_boost, uint64_t excluding = 0) const {
        // Copies share only the cached descriptor reader. Projection never
        // mutates published holds or the owner's learning state.
        auto view = budget;
        view.prefill_pages = pages(excluding);
#if defined(__linux__)
        view.device_held = held_device(); // the native ledgers' envelopes charge device memory to the process
#endif
        return view.measure(add(extra, pending), full_boost);
    }
    // An admission of `extra` bytes that `observed` found short: the context's reclaimer gives back what `who` may
    // ask of it, its steps run without the mutex. False when there is nothing to run.
    result<bool> reclaim(uint64_t extra, room observed, const std::shared_ptr<detail::client_state> & state, requester who) {
        // clear_reclaimer waits for the count; the callbacks go before it drops
        struct running {
            impl & owner; detail::client_state & state; std::shared_ptr<const actions> callbacks;
            ~running() {
                callbacks.reset();
                std::lock_guard<std::mutex> lock(owner.mutex);
                --state.reclaims; owner.changed.notify_all();
            }
        };
        std::shared_ptr<const actions> registered;
        bool unified = false;
        {
            std::lock_guard<std::mutex> lock(mutex);
            if (const auto * error = invalid(*state)) { return failed<bool>(status::invalid, error); }
            if (who == requester::gpu_tier || !state->reclaimer) { return passed(false); }
            registered = state->reclaimer; ++state->reclaims; unified = budget.unified;
        }
        const running done{ *this, *state, std::move(registered) };
        const actions & callbacks = *done.callbacks;
        if (unified && who == requester::outside && callbacks.reclaim_tier) {
            if (callbacks.collect_deferred) { callbacks.collect_deferred(); }
            uint64_t need;
            {
                std::lock_guard<std::mutex> lock(mutex);
                if (const auto * error = invalid(*state)) { return failed<bool>(status::invalid, error); }
                need = add(add(extra, pages()), pending);
            }
            callbacks.reclaim_tier(need);
            std::lock_guard<std::mutex> lock(mutex);
            if (const auto * error = invalid(*state)) { return failed<bool>(status::invalid, error); }
            observed = measure(extra, true);
        }
        if (!observed.fits() && callbacks.reclaim_cache) { callbacks.reclaim_cache(extra); }
        return passed(true);
    }
};
process_arbiter::process_arbiter(reader injected) noexcept {
    try { impl_.reset(new impl(std::move(injected))); }
    catch (const std::bad_alloc &) { init_status_ = status::resource; }
    catch (const std::exception & error) {
        init_status_ = status::io;
        try { init_error_ = error.what(); } catch (...) {}
    }
    catch (...) { init_status_ = status::io; }
}
process_arbiter::~process_arbiter() = default;
result<std::unique_ptr<context_client>> process_arbiter::make_client() noexcept {
    return guarded<std::unique_ptr<context_client>>([&]() -> result<std::unique_ptr<context_client>> {
        if (!impl_) {
            return failed<std::unique_ptr<context_client>>(init_status_, init_error_.empty() ?
                "host owner initialization" : ("host owner initialization: " + init_error_).c_str());
        }
        auto owner = shared_from_this();
        auto state = std::make_shared<detail::client_state>();
        auto client = std::unique_ptr<context_client>(new context_client(std::move(owner), state));
        {
            std::lock_guard<std::mutex> lock(impl_->mutex);
            if (!impl_->next_client) { return failed<std::unique_ptr<context_client>>(status::resource, "host identity exhausted"); }
            state->id = impl_->next_client++;
            impl_->invalidate_peers(state->id); impl_->clients.emplace(state->id, state);
        }
        return passed(std::move(client));
    });
}
context_client::context_client(std::shared_ptr<process_arbiter> owner, std::shared_ptr<detail::client_state> state) noexcept
    : owner_(std::move(owner)), state_(std::move(state)) {}
context_client::~context_client() { close(); }
result<void> context_client::configure(const client_config & config) noexcept {
    return guarded<void>([&]() -> result<void> {
        auto & owner = *owner_->impl_;
        std::lock_guard<std::mutex> lock(owner.mutex);
        if (const auto * error = owner.invalid(*state_, false)) { return {status::invalid, error}; }
        if (state_->configured) {
            if (!same_config(state_->config, config)) { return {status::invalid, "host context configuration changed"}; }
            return {};
        }
        if (config.reserve_bytes < (512ull << 20)) { return {status::invalid, "host reserve is below 512 MiB"}; }
        if (owner.domain_set && (owner.domain_unified != config.unified ||
                owner.domain_reclaimable != config.reclaimable_file)) {
            return {status::invalid, "host clients disagree on the physical budget domain"};
        }
        owner.domain_set = true; owner.domain_unified = config.unified; owner.domain_reclaimable = config.reclaimable_file;
        state_->config = config; state_->configured = true;
        owner.refresh_config(); owner.invalidate_peers(state_->id);
        return {};
    });
}
result<snapshot> context_client::state() const noexcept {
    return guarded<snapshot>([&]() -> result<snapshot> {
        auto & owner = *owner_->impl_;
        process_arbiter::impl::operation operation(owner, state_, false);
        if (operation.error) { return failed<snapshot>(status::invalid, operation.error); }
        std::lock_guard<std::mutex> lock(owner.mutex);
        snapshot out;
        out.configured = state_->configured; out.client_id = state_->id;
        out.unified = owner.budget.unified; out.reclaimable_file = owner.budget.reclaimable_file;
        out.reserve = owner.budget.reserve; out.margin = owner.budget.margin;
        out.explicit_reserve = owner.budget.explicit_reserve; out.floor = owner.budget.floor();
        out.prefill_pages = state_->prefill_pages; out.all_prefill_pages = owner.pages(); out.pending_bytes = owner.pending;
        out.device_held = state_->device_held; out.all_device_held = owner.held_device();
        out.file_reserve = owner.budget.file_reserve; out.refaults = owner.budget.refaults;
        out.file_updates = owner.budget.file_updates; out.granted = add(state_->committed_grants, state_->external_pending);
        out.margin_updates = owner.margin_updates; out.overlap_skips = state_->overlap_skips; out.cgroup = owner.budget.cgroup;
        return passed(std::move(out));
    });
}
result<void> context_client::set_prefill_pages(uint64_t pages) noexcept {
    return guarded<void>([&]() -> result<void> {
        auto & owner = *owner_->impl_; process_arbiter::impl::operation operation(owner, state_);
        if (operation.error) { return {status::invalid, operation.error}; }
        std::lock_guard<std::mutex> lock(owner.mutex);
        if (state_->prefill_pages != pages) { state_->prefill_pages = pages; owner.invalidate_peers(state_->id); }
        return {};
    });
}
result<void> context_client::set_device_held(uint64_t bytes) noexcept {
    return guarded<void>([&]() -> result<void> {
        auto & owner = *owner_->impl_; process_arbiter::impl::operation operation(owner, state_);
        if (operation.error) { return {status::invalid, operation.error}; }
        std::lock_guard<std::mutex> lock(owner.mutex);
        const uint64_t held = state_->config.unified ? bytes : 0;
        if (state_->device_held != held) { state_->device_held = held; owner.invalidate_peers(state_->id); }
        return {};
    });
}
result<uint64_t> context_client::dram_bytes(uint64_t host, uint64_t device) const noexcept {
    return guarded<uint64_t>([&]() -> result<uint64_t> {
        auto & owner = *owner_->impl_; process_arbiter::impl::operation operation(owner, state_);
        if (operation.error) { return failed<uint64_t>(status::invalid, operation.error); }
        std::lock_guard<std::mutex> lock(owner.mutex);
        if (state_->config.unified && device > UINT64_MAX - host) { return failed<uint64_t>(status::resource, "host byte overflow"); }
        return passed(add(host, state_->config.unified ? device : 0));
    });
}
result<room> context_client::measure(uint64_t extra, bool full_boost) const noexcept {
    return guarded<room>([&]() -> result<room> {
        auto & owner = *owner_->impl_; process_arbiter::impl::operation operation(owner, state_);
        if (operation.error) { return failed<room>(status::invalid, operation.error); }
        std::lock_guard<std::mutex> lock(owner.mutex); return passed(owner.measure(extra, full_boost));
    });
}
result<room> context_client::project_decode_capacity(uint64_t extra, bool full_boost) const noexcept {
    return guarded<room>([&]() -> result<room> {
        auto & owner = *owner_->impl_; process_arbiter::impl::operation operation(owner, state_);
        if (operation.error) { return failed<room>(status::invalid, operation.error); }
        std::lock_guard<std::mutex> lock(owner.mutex); return passed(owner.measure(extra, full_boost, state_->id));
    });
}
result<void> context_client::set_reclaimer(const actions & callbacks) noexcept {
    return guarded<void>([&]() -> result<void> {
        auto & owner = *owner_->impl_;
        auto registered = std::make_shared<const actions>(callbacks);
        std::lock_guard<std::mutex> lock(owner.mutex);
        if (const auto * error = owner.invalid(*state_, false)) { return {status::invalid, error}; }
        if (state_->reclaimer) { return {status::invalid, "host context already has a reclaimer"}; }
        state_->reclaimer = std::move(registered);
        return {};
    });
}
void context_client::clear_reclaimer() noexcept {
    auto & owner = *owner_->impl_; std::unique_lock<std::mutex> lock(owner.mutex);
    state_->reclaimer.reset();
    owner.changed.wait(lock, [&] { return !state_->reclaims; });
}
result<room> context_client::make_room(uint64_t extra) noexcept {
    return guarded<room>([&]() -> result<room> {
        auto & owner = *owner_->impl_; process_arbiter::impl::operation operation(owner, state_);
        if (operation.error) { return failed<room>(status::invalid, operation.error); }
        const auto measured = [&]() -> result<room> {
            std::lock_guard<std::mutex> lock(owner.mutex);
            if (const auto * error = owner.invalid(*state_)) { return failed<room>(status::invalid, error); }
            return passed(owner.measure(extra, true));
        };
        auto observed = measured();
        if (!observed || observed.value.fits()) { return observed; }
        const auto reclaimed = owner.reclaim(extra, observed.value, state_, requester::outside);
        if (!reclaimed) { return failed<room>(reclaimed.code, reclaimed.error.c_str()); }
        return reclaimed.value ? measured() : observed;
    });
}
result<allocation_ticket> context_client::take(uint64_t host, uint64_t device, requester who,
        bool finite, allocation_class classification) noexcept {
    return guarded<allocation_ticket>([&]() -> result<allocation_ticket> {
        auto & owner = *owner_->impl_; process_arbiter::impl::operation operation(owner, state_);
        if (operation.error) { return failed<allocation_ticket>(status::invalid, operation.error); }
        if (classification != allocation_class::external && classification != allocation_class::tracked) {
            return failed<allocation_ticket>(status::invalid, "host allocation class");
        }
        if (who != requester::outside && who != requester::ram_cache && who != requester::gpu_tier) {
            return failed<allocation_ticket>(status::invalid, "host requester");
        }
        if (state_->config.unified && device > UINT64_MAX - host) {
            return failed<allocation_ticket>(status::resource, "host byte overflow");
        }
        const uint64_t extra = add(host, state_->config.unified ? device : 0);
        // The room measured under the mutex, and the ticket issued under the same lock when it fits.
        const auto admit = [&]() -> result<allocation_ticket> {
            std::lock_guard<std::mutex> lock(owner.mutex);
            if (const auto * error = owner.invalid(*state_)) { return failed<allocation_ticket>(status::invalid, error); }
            auto observed = owner.measure(extra, true);
            const bool readable = observed.system || observed.cgroup != room_unreadable;
            const bool missing_cgroup = observed.cgroup == room_unreadable && (finite || !owner.budget.unbounded());
            // Nothing asked for needs no room: a shortage must not stop a run over memory it already holds.
            if ((extra && !observed.fits()) || !readable || missing_cgroup) {
                const auto code = !readable || missing_cgroup ? status::io : status::resource;
                return {allocation_ticket({}, {}, 0, observed), code, "host budget refused"};
            }
            const uint64_t held_pages = owner.pages();
            if (extra == UINT64_MAX || held_pages == UINT64_MAX || extra > UINT64_MAX - owner.pending ||
                    held_pages > UINT64_MAX - owner.pending - extra || !owner.next_ticket) {
                return failed<allocation_ticket>(status::resource, "host pending overflow");
            }
            const uint64_t id = owner.next_ticket++;
            owner.tickets.emplace(id, process_arbiter::impl::pending_ticket{state_, extra, classification});
            owner.pending += extra; state_->pending += extra;
            if (classification == allocation_class::external) { state_->external_pending += extra; }
            ++state_->tickets; owner.invalidate_peers(state_->id);
            return passed(allocation_ticket(owner_, state_, id, observed));
        };
        // One measurement when the room is there. A shortage runs the context's reclaimer, then measures again.
        auto ticket = admit();
        if (ticket.code != status::resource) { return ticket; }
        const auto reclaimed = owner.reclaim(extra, ticket.value.observation(), state_, who);
        if (!reclaimed) { return failed<allocation_ticket>(reclaimed.code, reclaimed.error.c_str()); }
        return reclaimed.value ? admit() : std::move(ticket);
    });
}
result<bool> context_client::update_file_reserve() noexcept {
    return guarded<bool>([&]() -> result<bool> {
        auto & owner = *owner_->impl_; process_arbiter::impl::operation operation(owner, state_);
        if (operation.error) { return failed<bool>(status::invalid, operation.error); }
        std::lock_guard<std::mutex> lock(owner.mutex);
        const bool changed = owner.budget.update_file_reserve();
        if (changed) { owner.invalidate_peers(state_->id); }
        return passed(changed);
    });
}
result<void> context_client::begin_interval(uint64_t held) noexcept {
    return guarded<void>([&]() -> result<void> {
        auto & owner = *owner_->impl_; process_arbiter::impl::operation operation(owner, state_);
        if (operation.error) { return {status::invalid, operation.error}; }
        std::lock_guard<std::mutex> lock(owner.mutex); uint64_t headroom = 0;
        // A ticket issued before this boundary may still be touching pages.
        // Its pending/held value is not an observation of resident memory.
        state_->interval_overlap = owner.pending != 0;
        for (const auto & pair : owner.clients) {
            if (pair.first != state_->id && pair.second->interval_open) { state_->interval_overlap = true; }
        }
        owner.invalidate_peers(state_->id);
        state_->interval_open = state_->config.unified && owner.budget.memory_available_at_full_boost(state_->interval_available, headroom);
        state_->interval_known = add(held, add(state_->committed_grants, state_->external_pending));
        return {};
    });
}
result<bool> context_client::end_interval(bool consecutive, uint64_t held, uint64_t & untracked) noexcept {
    untracked = 0;
    return guarded<bool>([&]() -> result<bool> {
        auto & owner = *owner_->impl_; process_arbiter::impl::operation operation(owner, state_);
        if (operation.error) { return failed<bool>(status::invalid, operation.error); }
        std::lock_guard<std::mutex> lock(owner.mutex);
        const bool open = state_->interval_open; state_->interval_open = false;
        if (!state_->config.unified || !open || !consecutive) { return passed(false); }
        if (state_->interval_overlap || state_->tickets) { ++state_->overlap_skips; return passed(false); }
        uint64_t available = 0, headroom = 0;
        if (!owner.budget.memory_available_at_full_boost(available, headroom)) { return passed(false); }
        const uint64_t lhs = add(state_->interval_available, state_->interval_known);
        const uint64_t rhs = add(available, add(held, add(state_->committed_grants, state_->external_pending)));
        untracked = lhs > rhs ? lhs - rhs : 0;
        if (untracked <= owner.budget.margin) { return passed(false); }
        owner.budget.margin = untracked; ++owner.margin_updates; return passed(true);
    });
}
result<void> context_client::note_known_memory_change() noexcept {
    return guarded<void>([&]() -> result<void> {
        auto & owner = *owner_->impl_; process_arbiter::impl::operation operation(owner, state_);
        if (operation.error) { return {status::invalid, operation.error}; }
        std::lock_guard<std::mutex> lock(owner.mutex);
        // A known free/replacement can mask simultaneous driver growth in this
        // actor too; skip this interval instead of learning from a mixed sample.
        if (state_->interval_open) { state_->interval_overlap = true; }
        owner.invalidate_peers(state_->id); return {};
    });
}
void context_client::close() noexcept {
    auto & owner = *owner_->impl_; std::unique_lock<std::mutex> lock(owner.mutex);
    if (state_->closed) { return; }
    state_->closing = true;
    owner.changed.wait(lock, [&] { return !state_->operations && !state_->tickets; });
    state_->interval_open = false; state_->prefill_pages = 0; state_->device_held = 0; state_->reclaimer.reset();
    state_->closed = true;
    owner.invalidate_peers(state_->id); owner.clients.erase(state_->id); owner.refresh_config();
}
allocation_ticket::allocation_ticket(std::shared_ptr<process_arbiter> owner,
        std::shared_ptr<detail::client_state> client, uint64_t id, room observed) noexcept
    : owner_(std::move(owner)), client_(std::move(client)), id_(id), observation_(observed) {}
allocation_ticket::~allocation_ticket() { cancel(); }
allocation_ticket::allocation_ticket(allocation_ticket && other) noexcept
    : owner_(std::move(other.owner_)), client_(std::move(other.client_)),
      id_(std::exchange(other.id_, 0)), observation_(other.observation_) {}
allocation_ticket & allocation_ticket::operator=(allocation_ticket && other) noexcept {
    if (this != &other) {
        cancel(); owner_ = std::move(other.owner_); client_ = std::move(other.client_);
        id_ = std::exchange(other.id_, 0); observation_ = other.observation_;
    }
    return *this;
}
void allocation_ticket::commit() noexcept { finish(true); }
void allocation_ticket::cancel() noexcept { finish(false); }
void allocation_ticket::finish(bool materialized) noexcept {
    if (!id_) { return; }
    // Pin the owner until after its mutex is released.
    auto owner = std::move(owner_); auto client = std::move(client_);
    const uint64_t id = std::exchange(id_, 0);
    {
        auto & bookkeeping = *owner->impl_; std::lock_guard<std::mutex> lock(bookkeeping.mutex);
        const auto found = bookkeeping.tickets.find(id);
        if (found == bookkeeping.tickets.end() || found->second.client != client) { std::terminate(); }
        const uint64_t bytes = found->second.bytes;
        bookkeeping.pending -= bytes; client->pending -= bytes; --client->tickets;
        if (found->second.classification == allocation_class::external) {
            client->external_pending -= bytes;
            if (materialized) { client->committed_grants = add(client->committed_grants, bytes); }
        }
        bookkeeping.tickets.erase(found); bookkeeping.invalidate_peers(client->id); bookkeeping.changed.notify_all();
    }
}
}} // namespace ssd::host
