// SPDX-License-Identifier: Apache-2.0
#include "ggml-ssd/gpu/transport.h"
#include "ggml-ssd/store/source.h"
#include "ggml-ssd/host/arbiter.h"
#include "ggml-ssd/io/read-ring.h"
#include "ggml-ssd/platform/file.h"
#include "ggml-cuda.h"
#include <cuda_runtime_api.h>
#include <algorithm>
#include <atomic>
#include <stdexcept>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <climits>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace {
constexpr size_t alignment = 4096, stage_slots = 16, scheduler_refill = 16;
[[noreturn]] void fatal(const char * reason) noexcept {
    std::fprintf(stderr, "[ssd.transport] cannot drain owned work: %s\n", reason); std::abort();
}
uint64_t align_up_u64(uint64_t value, uint64_t a) { return (value + a - 1) / a * a; }
struct extent { int fd = -1; uint64_t offset = 0, file_size = 0; size_t bytes = 0; };

// Pinned host slots that this OS's read ring (the one the CPU sink reads with) fills from the drive and a stream
// copies into the device. The slots are cut from one pinned block at the granule direct reads need, which the pinned
// allocator does not promise for a small block. A read's tag is its slot. A slot's event records its last copy, which
// finishes before the slot is read into again.
struct pinned_stage {
    struct slot { char * host = nullptr; cudaEvent_t done = nullptr; bool copying = false; };
    enum class taken { one, none, failed };
    std::unique_ptr<ssd::io::read_ring> ring;
    void * block = nullptr; // the pinned allocation the slots are cut from
    std::vector<slot> slots;
    size_t slot_bytes = 0, next = 0;
    unsigned queued = 0;
    bool failed = false; // the pinned memory could not be had: not asked for again

    // slot_count slots of slot_size bytes, a multiple of the granule, and a ring as deep, admitted by the host budget
    // first and touched inside its reservation
    bool allocate(ssd::host::context_client * host, size_t slot_size, size_t slot_count) {
        if (!slot_size || slot_size % alignment || !slot_count || slot_count > UINT_MAX ||
                slot_count > (SIZE_MAX - alignment) / slot_size || failed) return false;
        if (slot_bytes >= slot_size && slots.size() == slot_count) return true;
        release(host);
        const size_t bytes = slot_size * slot_count + alignment - 1;
        auto ticket = host->take(bytes);
        if (!ticket) return false;
        (void) ggml_cuda_admit_host_alloc(bytes); // unified memory: the reserve first, as for device memory
        if (cudaHostAlloc(&block, bytes, cudaHostAllocDefault) != cudaSuccess) {
            block = nullptr;
            failed = true;
            return false;
        }
        char * first = static_cast<char *>(block) + (alignment - reinterpret_cast<uintptr_t>(block) % alignment) % alignment;
        slots.resize(slot_count);
        for (size_t i = 0; i < slot_count; ++i) {
            slots[i].host = first + i * slot_size;
            if (cudaEventCreateWithFlags(&slots[i].done, cudaEventDisableTiming) != cudaSuccess) {
                release(host);
                failed = true;
                return false;
            }
        }
        slot_bytes = slot_size;
        ring.reset(new ssd::io::read_ring);
        if (!ring->init(unsigned(slot_count), false)) {
            release(host);
            return false;
        }
        std::memset(first, 0, slot_size * slot_count);
        ticket.value.commit();
        return true;
    }
    void release(ssd::host::context_client * host) noexcept {
        const bool allocated = block != nullptr;
        if (!drain()) fatal("staging reads");
        ring.reset();
        for (auto & s : slots) {
            if (s.copying && cudaEventSynchronize(s.done) != cudaSuccess) fatal("staging copy");
            if (s.done && cudaEventDestroy(s.done) != cudaSuccess) fatal("staging release");
        }
        if (block && cudaFreeHost(block) != cudaSuccess) fatal("staging release");
        block = nullptr;
        slots.clear();
        slot_bytes = next = 0;
        if (allocated && host) (void)host->note_known_memory_change();
    }
    bool ready(size_t index) {
        slot & s = slots[index];
        if (s.copying && cudaEventSynchronize(s.done) != cudaSuccess) return false;
        s.copying = false;
        return true;
    }
    // queues a read of bytes at offset of fd into the slot, once the slot's last copy finished
    bool read(size_t index, int fd, uint64_t offset, size_t bytes) {
        if (!ring || index >= slots.size() || !bytes || bytes > slot_bytes || bytes > size_t(INT32_MAX) || !ready(index)) {
            return false;
        }
        if (!ring->queue({fd, offset, bytes}, slots[index].host, index, false)) return false;
        ++queued;
        return true;
    }
    bool submit() {
        const unsigned count = queued;
        queued = 0;
        return !count || bool(ring->submit(count));
    }
    // a completion ready now or, with wait, the next to arrive; only wait with reads in flight
    taken take(ssd::io::completion & out, bool wait) {
        for (;;) {
            const auto popped = ring->pop(out);
            if (!popped) return taken::failed;
            if (popped.value) return taken::one;
            if (!wait) return taken::none;
            if (!ring->wait()) return taken::failed;
        }
    }
    bool copy(size_t index, void * dst, size_t offset, size_t bytes, cudaStream_t stream) {
        slot & s = slots[index];
        if (cudaMemcpyAsync(dst, s.host + offset, bytes, cudaMemcpyHostToDevice, stream) != cudaSuccess ||
                cudaEventRecord(s.done, stream) != cudaSuccess) {
            return false;
        }
        s.copying = true;
        return true;
    }
    // every read submitted and reaped, its bytes ignored
    bool drain() noexcept {
        queued = 0;
        return !ring || bool(ring->drain(nullptr, [](void *, const ssd::io::completion &) {}));
    }
};
struct transport_owner;
struct dense_stage_half {
    std::string tensor;
    size_t n_blocks = 0, block_bytes = 0;
    std::vector<size_t> prefix, expect;
    std::vector<uint8_t> done;
    size_t in_flight = 0;
    bool failed = false;
};
struct dense_stage_state {
    transport_owner * owner = nullptr;
    pinned_stage stage;
    size_t blocks_per_half = 0, need_blocks = 0, need_slot = 0;
    bool failed = false;
    dense_stage_half half[2];
    int last = 1;
};
struct transport_owner {
    ssd::store::source_view source;
    ssd::host::context_client * host;
    int device;
    std::mutex mutex, seed_mutex;
    std::atomic<bool> closed{false};
    pinned_stage stage;
    dense_stage_state dense;
    std::unordered_map<int, uint64_t> source_sizes;
    uint64_t reads = 0, bytes = 0;
    ssd::gpu::seed_offer_fn seed = nullptr;
    void * seed_data = nullptr;
    cudaEvent_t ready = nullptr, pending = nullptr, last_seed_done = nullptr;
    transport_owner(const ssd::runtime_services & services, int ordinal) : source(*services.source), host(services.host), device(ordinal) { dense.owner = this; }
    bool resolve(const char * name, int expert, size_t bytes, int64_t experts, extent & out) {
        size_t logical = 0;
        if (!source.resolve(source.userdata, name, expert, bytes, experts, &out.fd, &out.offset, &logical) || logical != bytes || out.fd < 0) return false;
        auto found = source_sizes.find(out.fd);
        if (found == source_sizes.end()) {
            // a descriptor that reads directly at the granule the reads are padded to, as the sink's
            uint64_t size = 0;
            if (!ssd::platform::is_direct(out.fd) || alignment % ssd::platform::direct_alignment(out.fd) != 0 ||
                    !ssd::platform::file_size(out.fd, size)) return false;
            found = source_sizes.emplace(out.fd, size).first;
        }
        out.file_size = found->second; out.bytes = bytes;
        return out.offset <= out.file_size && bytes <= out.file_size - out.offset;
    }
};
struct copy_job { void * dst; extent source; };
struct read_plan { size_t job; uint64_t offset; size_t read_sz, prefix, expect; };

// The scheduler's fixed production profile: no coalescing or sorting; one first batch as deep as the stage, then
// refills of the slots that completions free, the last freed first.
bool copy_jobs(transport_owner & owner, const std::vector<copy_job> & jobs, cudaStream_t stream) {
    if (jobs.empty()) return true;
    std::vector<read_plan> plans; plans.reserve(jobs.size());
    size_t maximum = 0;
    for (size_t i = 0; i < jobs.size(); ++i) {
        const auto & source = jobs[i].source;
        const uint64_t offset = source.offset / alignment * alignment;
        const size_t prefix = size_t(source.offset - offset);
        if (source.bytes > UINT_MAX - prefix - alignment + 1) return false;
        const size_t read = size_t(align_up_u64(prefix + source.bytes, alignment));
        const size_t expected = size_t(std::min<uint64_t>(read, source.file_size - offset));
        if (expected < prefix + source.bytes || read > INT_MAX) return false;
        plans.push_back({i, offset, read, prefix, expected}); maximum = std::max(maximum, read);
    }
    auto & stage = owner.stage;
    if (!stage.allocate(owner.host, maximum, stage_slots)) return false;
    const size_t depth = stage.slots.size();
    struct drain_guard {
        pinned_stage & stage; cudaStream_t stream; bool success = false;
        ~drain_guard() {
            if (!stage.drain()) fatal("copy read queue");
            if (!success && cudaStreamSynchronize(stream) != cudaSuccess) fatal("failed copy stream");
        }
    } drain{stage, stream};
    std::vector<size_t> plan_of(depth, SIZE_MAX); // the plan each slot reads
    std::vector<size_t> reusable; reusable.reserve(depth);
    size_t next = 0, inflight = 0, completed = 0;
    auto submit_one = [&](size_t slot) {
        if (!stage.read(slot, jobs[plans[next].job].source.fd, plans[next].offset, plans[next].read_sz)) return false;
        plan_of[slot] = next++; ++inflight;
        return true;
    };
    auto complete = [&](const ssd::io::completion & completion) {
        const size_t slot = size_t(completion.user_data);
        if (slot >= depth || plan_of[slot] == SIZE_MAX) return false;
        const auto & plan = plans[plan_of[slot]]; const auto & job = jobs[plan.job];
        plan_of[slot] = SIZE_MAX;
        if (completion.res < 0 || size_t(completion.res) != plan.expect ||
                !stage.copy(slot, job.dst, plan.prefix, job.source.bytes, stream)) return false;
        ++owner.reads; owner.bytes += job.source.bytes;
        ++completed; --inflight; reusable.push_back(slot);
        return true;
    };
    for (size_t i = 0, first = std::min(plans.size(), depth); i < first; ++i) {
        if (!submit_one(stage.next++ % depth)) return false;
    }
    if (!stage.submit()) return false;
    while (completed < plans.size()) {
        ssd::io::completion completion;
        if (stage.take(completion, true) != pinned_stage::taken::one || !complete(completion)) return false;
        while (completed < plans.size() && inflight > 0) {
            const auto ready = stage.take(completion, false);
            if (ready == pinned_stage::taken::none) break;
            if (ready == pinned_stage::taken::failed || !complete(completion)) return false;
        }
        size_t submitted = 0;
        while (next < plans.size() && inflight < depth && !reusable.empty() && submitted < std::min(scheduler_refill, depth)) {
            const size_t slot = reusable.back(); reusable.pop_back();
            if (!submit_one(slot)) return false;
            ++submitted;
        }
        if (submitted && !stage.submit()) return false;
    }
    drain.success = true;
    return true;
}

static size_t dense_slot_bytes(size_t block_bytes) {
    const size_t a = alignment;
    return (size_t)align_up_u64((uint64_t)block_bytes + a, a);
}

// the blocks of a matrix the index holds as n_blocks x block_bytes == nbytes, n >= 1; 0 when it holds none
static size_t dense_index_blocks(transport_owner & owner, const char * name, size_t bytes, size_t * block) {
    return owner.source.dense_blocks(owner.source.userdata, name, bytes, block);
}

// the ring failed: every read in flight is reaped unmatched (no read lands in a slot later) and its half failed
static void dense_stage_abort(dense_stage_state &s) {
    if (!s.stage.drain()) fatal("dense read queue");
    for (dense_stage_half &half : s.half) {
        half.failed = half.failed || half.in_flight > 0;
        half.in_flight = 0;
    }
}

// takes one completion of the stage's ring and marks its block (a failed or short read fails its half); false when
// the ring gives none (dense_stage_abort)
static bool dense_stage_reap_one(dense_stage_state &s) {
    ssd::io::completion c;
    if (s.stage.take(c, true) != pinned_stage::taken::one) {
        dense_stage_abort(s);
        return false;
    }
    const size_t slot = (size_t)c.user_data;
    if (s.blocks_per_half == 0 || slot >= 2 * s.blocks_per_half) return true;
    dense_stage_half &half = s.half[slot / s.blocks_per_half];
    const size_t i = slot % s.blocks_per_half;
    if (half.in_flight > 0) --half.in_flight;
    if (i >= half.n_blocks || c.res < 0 || (size_t)c.res != half.expect[i]) {
        half.failed = true;
    } else {
        half.done[i] = 1;
    }
    return true;
}

// no read in flight in half h; false when one of its reads failed
static bool dense_stage_drain(dense_stage_state &s, int h) {
    while (s.half[h].in_flight > 0 && dense_stage_reap_one(s)) {
    }
    return !s.half[h].failed;
}

// the stage at the size of the largest matrix asked about: every read finished, every slot's copy done; a stage that
// could not be allocated is not tried again
static bool dense_stage_ensure(dense_stage_state &s, size_t n_blocks, size_t slot_bytes) {
    s.need_blocks = std::max(s.need_blocks, n_blocks);
    s.need_slot = std::max(s.need_slot, slot_bytes);
    if (s.blocks_per_half >= n_blocks && s.stage.slot_bytes >= slot_bytes) {
        return true;
    }
    if (s.failed) {
        return false;
    }
    for (int h = 0; h < 2; ++h) {
        if (!s.stage.slots.empty()) (void)dense_stage_drain(s, h);
        s.half[h] = dense_stage_half();
    }
    s.stage.release(s.owner->host);
    if (!s.stage.allocate(s.owner->host, s.need_slot, 2 * s.need_blocks)) {
        std::fprintf(stderr, "[moe_stream_batch] dense stage: init failed for 2 x %zu slots of %.2f MiB\n",
                     s.need_blocks, s.need_slot / (1024.0 * 1024.0));
        s.stage.release(s.owner->host);
        s.blocks_per_half = 0;
        s.failed = true;
        return false;
    }
    s.blocks_per_half = s.need_blocks;
    std::fprintf(stderr, "[moe_stream_batch] dense stage: 2 x %zu slots of %.2f MiB pinned\n",
                 s.blocks_per_half, s.stage.slot_bytes / (1024.0 * 1024.0));
    return true;
}

// submits the reads of a matrix into half h, its slots' earlier copies finished first
static bool dense_stage_submit(dense_stage_state &s, int h, const char *tensor_name, size_t n_blocks,
                               size_t block_bytes) {
    dense_stage_half &half = s.half[h];
    (void)dense_stage_drain(s, h);
    half = dense_stage_half();
    half.tensor = tensor_name;
    half.n_blocks = n_blocks;
    half.block_bytes = block_bytes;
    half.prefix.assign(n_blocks, 0);
    half.expect.assign(n_blocks, 0);
    half.done.assign(n_blocks, 0);
    const size_t a = alignment;
    for (size_t i = 0; i < n_blocks; ++i) {
        extent src;
        if (!s.owner->resolve(tensor_name, int(i), block_bytes, int64_t(n_blocks), src) || src.fd < 0) {
            half.failed = true;
            break;
        }
        const uint64_t off = src.offset / a * a;
        half.prefix[i] = (size_t)(src.offset - off);
        const size_t read_sz = (size_t)align_up_u64(half.prefix[i] + block_bytes, a);
        half.expect[i] = src.file_size > off && src.file_size - off < read_sz ? (size_t)(src.file_size - off) : read_sz;
        if (half.expect[i] < half.prefix[i] + block_bytes ||
                !s.stage.read((size_t)h * s.blocks_per_half + i, src.fd, off, read_sz)) {
            half.failed = true;
            break;
        }
        ++half.in_flight;
    }
    if (!s.stage.submit()) {
        dense_stage_abort(s);
    }
    if (half.failed) {
        (void)dense_stage_drain(s, h);
        half = dense_stage_half();
        return false;
    }
    return true;
}

bool dense_has_impl(transport_owner & owner, const char *tensor_name, size_t nbytes, size_t *block_bytes) {
    size_t bb = 0;
    const size_t n = dense_index_blocks(owner, tensor_name, nbytes, &bb);
    if (n == 0) return false;
    std::lock_guard<std::mutex> lock(owner.mutex);
    if (owner.closed || cudaSetDevice(owner.device) != cudaSuccess) return false;
    owner.dense.need_blocks = std::max(owner.dense.need_blocks, n);
    owner.dense.need_slot = std::max(owner.dense.need_slot, dense_slot_bytes(bb));
    if (block_bytes) *block_bytes = bb;
    return true;
}

// the stage for the largest matrix asked about so far, allocated now (the load receipt, once every weight homed on the
// drive was asked about): its pinned bytes; false when it cannot be
bool dense_prepare_impl(transport_owner & owner, size_t *pinned_bytes) {
    if (pinned_bytes) *pinned_bytes = 0;
    std::lock_guard<std::mutex> lock(owner.mutex);
    if (owner.closed || cudaSetDevice(owner.device) != cudaSuccess) return false;
    dense_stage_state &s = owner.dense;
    if (s.need_blocks == 0 || !dense_stage_ensure(s, s.need_blocks, s.need_slot)) {
        return false;
    }
    if (pinned_bytes) *pinned_bytes = 2 * s.blocks_per_half * s.stage.slot_bytes;
    return true;
}

bool dense_read_ahead_impl(transport_owner & owner, const char *tensor_name, size_t nbytes) {
    size_t bb = 0;
    const size_t n = dense_index_blocks(owner, tensor_name, nbytes, &bb);
    if (n == 0) return false;
    std::lock_guard<std::mutex> lock(owner.mutex);
    if (owner.closed || cudaSetDevice(owner.device) != cudaSuccess) return false;
    dense_stage_state &s = owner.dense;
    for (const dense_stage_half &h : s.half) {
        if (h.tensor == tensor_name) return true;
    }
    if (!dense_stage_ensure(s, n, dense_slot_bytes(bb))) return false;
    return dense_stage_submit(s, 1 - s.last, tensor_name, n, bb);
}

bool dense_copy_impl(transport_owner & owner,
        const char *tensor_name,
        void *device_dst,
        size_t nbytes,
        void *stream_ptr,
        bool *read_ahead_hit) {
    if (read_ahead_hit) *read_ahead_hit = false;
    size_t bb = 0;
    const size_t n = dense_index_blocks(owner, tensor_name, nbytes, &bb);
    if (n == 0 || !device_dst || !stream_ptr) return false;
    std::lock_guard<std::mutex> lock(owner.mutex);
    if (owner.closed || cudaSetDevice(owner.device) != cudaSuccess) return false;
    dense_stage_state &s = owner.dense;
    if (!dense_stage_ensure(s, n, dense_slot_bytes(bb))) return false;
    int h = s.half[0].tensor == tensor_name ? 0 : s.half[1].tensor == tensor_name ? 1 : -1;
    if (h >= 0 && (s.half[h].n_blocks != n || s.half[h].block_bytes != bb || s.half[h].failed)) {
        (void)dense_stage_drain(s, h);
        s.half[h] = dense_stage_half();
        h = -1;
    }
    if (read_ahead_hit) *read_ahead_hit = h >= 0;
    if (h < 0) {
        h = 1 - s.last;
        if (!dense_stage_submit(s, h, tensor_name, n, bb)) return false;
    }
    dense_stage_half &half = s.half[h];
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    bool ok = true;
    for (size_t i = 0; i < n && ok; ++i) {
        while (!half.done[i] && !half.failed && dense_stage_reap_one(s)) {
        }
        ok = half.done[i] && s.stage.copy((size_t)h * s.blocks_per_half + i, (char *)device_dst + i * bb,
                                          half.prefix[i], bb, stream);
    }
    (void)dense_stage_drain(s, h);
    half = dense_stage_half();
    s.last = h;
    if (!ok && cudaStreamSynchronize(stream) != cudaSuccess) fatal("dense failure stream");
    return ok;
}

void quiesce(void * value) noexcept {
    if (!value) return;
    auto & owner = *static_cast<transport_owner *>(value);
    std::lock_guard<std::mutex> lock(owner.mutex);
    if (owner.closed) return;
    owner.closed = true;
    if (cudaSetDevice(owner.device) != cudaSuccess) fatal("device selection");
    owner.stage.release(owner.host);
    dense_stage_abort(owner.dense);
    owner.dense.stage.release(owner.host);
    for (auto & half : owner.dense.half) half = dense_stage_half();
    std::lock_guard<std::mutex> seed_lock(owner.seed_mutex);
    if (owner.last_seed_done && cudaEventSynchronize(owner.last_seed_done) != cudaSuccess) fatal("last seed copy");
    if (owner.ready && cudaEventSynchronize(owner.ready) != cudaSuccess) fatal("seed source");
    if (owner.ready && cudaEventDestroy(owner.ready) != cudaSuccess) fatal("seed event");
    owner.pending = owner.ready = owner.last_seed_done = nullptr;
}
void destroy(void * value) noexcept {
    if (!value) return;
    quiesce(value);
    auto * owner = static_cast<transport_owner *>(value);
    if (owner->seed) fatal("tier still registered during owner destruction");
    delete owner;
}
void * create(const ssd::runtime_services * services, int device, char * error, size_t size) noexcept {
    try {
        if (!services || !services->host || !services->source || !services->source->resolve ||
                !services->source->dense_blocks || device < 0) throw std::runtime_error("invalid context transport services");
        return new transport_owner(*services, device);
    } catch (const std::exception & failure) {
        if (error && size) std::snprintf(error, size, "%s", failure.what());
        return nullptr;
    } catch (...) {
        if (error && size) std::snprintf(error, size, "transport construction failed");
        return nullptr;
    }
}
bool copy_ranges(void * value, const char * name, const int32_t * first, const int32_t * last, size_t count,
        int64_t experts, size_t expert_bytes, size_t padding, bool cache_bypass, void * destination, size_t capacity, void * stream_ptr,
        uint64_t * copied, uint64_t * reads, uint64_t * bytes, uint64_t * lookups, uint64_t * hits) noexcept {
    if (copied) *copied = 0;
    if (reads) *reads = 0;
    if (bytes) *bytes = 0;
    if (lookups) *lookups = 0;
    if (hits) *hits = 0;
    (void)cache_bypass; // The retired process-global cache producer is never consulted.
    if (!value || !name || !name[0] || !first || !last || !count || !expert_bytes || experts <= 0 || experts > INT32_MAX ||
            !destination || !stream_ptr || uint64_t(experts) > SIZE_MAX / expert_bytes || size_t(experts) * expert_bytes > capacity) return false;
    auto & owner = *static_cast<transport_owner *>(value);
    try {
        std::lock_guard<std::mutex> lock(owner.mutex);
        if (owner.closed || cudaSetDevice(owner.device) != cudaSuccess) return false;
        std::vector<copy_job> jobs;
        int32_t previous = -1;
        for (size_t range = 0; range < count; ++range) {
            if (first[range] < 0 || last[range] < first[range] || last[range] >= experts || first[range] <= previous) return false;
            previous = last[range];
            for (int32_t expert = first[range]; expert <= last[range]; ++expert) {
                extent source;
                if (!owner.resolve(name, expert, expert_bytes, experts, source)) return false;
                jobs.push_back({static_cast<char *>(destination) + size_t(expert) * expert_bytes, source});
            }
        }
        const uint64_t before_reads = owner.reads, before_bytes = owner.bytes;
        const auto stream = reinterpret_cast<cudaStream_t>(stream_ptr);
        bool ok = copy_jobs(owner, jobs, stream);
        if (ok && padding) {
            for (size_t range = 0; range < count; ++range) {
                if (last[range] >= experts - 1) continue;
                const size_t offset = (size_t(last[range]) + 1) * expert_bytes;
                if (offset > capacity || padding > capacity - offset ||
                        cudaMemsetAsync(static_cast<char *>(destination) + offset, 0, padding, stream) != cudaSuccess) { ok = false; break; }
            }
        }
        if (!ok && cudaStreamSynchronize(stream) != cudaSuccess) fatal("copy failure stream");
        if (copied && ok) *copied = jobs.size();
        if (reads) *reads = owner.reads - before_reads;
        if (bytes) *bytes = owner.bytes - before_bytes;
        return ok;
    } catch (...) { quiesce(value); return false; }
}
bool dense_has(void * p, const char * name, size_t bytes, size_t * block) noexcept {
    if (!p) return false;
    try { return dense_has_impl(*static_cast<transport_owner *>(p), name, bytes, block); }
    catch (...) { quiesce(p); return false; }
}
bool dense_prepare(void * p, size_t * bytes) noexcept {
    if (!p) return false;
    try { return dense_prepare_impl(*static_cast<transport_owner *>(p), bytes); }
    catch (...) { quiesce(p); return false; }
}
bool dense_read_ahead(void * p, const char * name, size_t bytes) noexcept {
    if (!p) return false;
    try { return dense_read_ahead_impl(*static_cast<transport_owner *>(p), name, bytes); }
    catch (...) { quiesce(p); return false; }
}
bool dense_copy(void * p, const char * name, void * dst, size_t bytes, void * stream, bool * hit) noexcept {
    if (!p) return false;
    try { return dense_copy_impl(*static_cast<transport_owner *>(p), name, dst, bytes, stream, hit); }
    catch (...) { quiesce(p); return false; }
}
bool seed_register(void * p, ssd::gpu::seed_offer_fn callback, void * data) noexcept {
    if (!p || !callback || !data) return false;
    auto & owner = *static_cast<transport_owner *>(p);
    std::lock_guard<std::mutex> lock(owner.seed_mutex);
    if (owner.closed) return false;
    if (owner.seed) return owner.seed == callback && owner.seed_data == data;
    owner.seed = callback; owner.seed_data = data; return true;
}
bool seed_unregister(void * p, ssd::gpu::seed_offer_fn callback, void * data) noexcept {
    if (!p) return false;
    auto & owner = *static_cast<transport_owner *>(p);
    std::lock_guard<std::mutex> lock(owner.seed_mutex);
    if (owner.seed != callback || owner.seed_data != data) return false;
    if (owner.last_seed_done && cudaEventSynchronize(owner.last_seed_done) != cudaSuccess) fatal("seed unregister");
    owner.seed = nullptr; owner.seed_data = nullptr; owner.pending = owner.last_seed_done = nullptr; return true;
}
bool seed_offer(void * p, const char * name, const void * data, size_t bytes, int64_t experts,
        const int32_t * counts, void * stream_ptr) noexcept {
    if (!p || !name || !data || !bytes || experts <= 0 || !counts || !stream_ptr) return false;
    auto & owner = *static_cast<transport_owner *>(p);
    std::lock_guard<std::mutex> lock(owner.seed_mutex);
    if (owner.closed) return false;
    if (!owner.seed) return true;
    if (cudaSetDevice(owner.device) != cudaSuccess) return false;
    if (!owner.ready && cudaEventCreateWithFlags(&owner.ready, cudaEventDisableTiming) != cudaSuccess) return false;
    if (cudaEventRecord(owner.ready, reinterpret_cast<cudaStream_t>(stream_ptr)) != cudaSuccess) return false;
    void * done = nullptr;
    const int result = owner.seed(owner.seed_data, name, data, bytes, experts, counts, owner.ready, &done);
    if (result < 0) return false;
    if (result > 0 && done) owner.pending = owner.last_seed_done = static_cast<cudaEvent_t>(done);
    return true;
}
bool seed_fence(void * p, void * stream_ptr) noexcept {
    if (!p || !stream_ptr) return false;
    auto & owner = *static_cast<transport_owner *>(p);
    std::lock_guard<std::mutex> lock(owner.seed_mutex);
    if (owner.closed) return false;
    if (owner.pending) {
        if (cudaSetDevice(owner.device) != cudaSuccess || cudaStreamWaitEvent(reinterpret_cast<cudaStream_t>(stream_ptr), owner.pending, 0) != cudaSuccess) return false;
        owner.pending = nullptr;
    }
    return true;
}
} // namespace
extern "C" const ssd::gpu::transport_v1 * ggml_backend_cuda_ssd_transport_v1() {
    static const ssd::gpu::transport_v1 api{1, sizeof(ssd::gpu::transport_v1), sizeof(ssd::runtime_services),
        create, quiesce, destroy, copy_ranges, dense_has, dense_prepare, dense_read_ahead, dense_copy,
        seed_register, seed_unregister, seed_offer, seed_fence};
    return &api;
}
