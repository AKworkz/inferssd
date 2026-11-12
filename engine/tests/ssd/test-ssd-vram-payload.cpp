// SPDX-License-Identifier: Apache-2.0
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#endif
#include "explicit-fixture-inputs.h"
// CUDA test of experts whose roles are not a whole number of 4 KiB pages, named as the CPU sink names them: by their
// payload (cpu-runtime.h resolve_source sets payload_nbytes to the GGUF tensor's nb[2], and the RAM cache keys a role
// by it). Layer 1 is Q4_K up/gate with a Q6_K down, as DeepSeek-V4.1-Flash Q4_K_M lays out its experts (the 256 x 256
// down is 53760 bytes, 57344 as a slot pads it); layer 2 is MXFP4 throughout (34816 bytes per role, 36864 padded).
// Every role lives in its own host allocation of exactly its payload, followed by an inaccessible page, so a copy of
// the padded length faults. Admissions, deferred uploads and host jobs copy the payload and zero the slot's padding,
// and the GPU's rows match the experts computed on the CPU. A format's first request fixes how its roles are named: a
// later padded name of a payload-named role is refused. Experts preloaded at load from payload-only host memory land
// the same way and are hits for the first decode token.
#include "ggml-cuda.h"
#include "gpu-services.h"
#include "vram-tier.cpp"
#include <cmath>
#include <cstring>
#include <memory>
#include <random>
#ifdef _WIN32
#include <windows.h>
#else
#include <sys/mman.h>
#include <unistd.h>
#endif

namespace {
int assertions = 0;
void require(bool ok, const std::string & what) { ++assertions; if (!ok) throw std::runtime_error(what); }

constexpr int latent = 256, hidden = 256, experts = 8;
const int32_t layer_ids[2] = { 1, 2 };
const int32_t up_types[2] = { GGML_TYPE_Q4_K, GGML_TYPE_MXFP4 }, down_types[2] = { GGML_TYPE_Q6_K, GGML_TYPE_MXFP4 };
ggml_type type_of(int layer, int role) { return ggml_type(role == 2 ? down_types[layer - 1] : up_types[layer - 1]); }
// A role's logical bytes (what the sink names and reads) and its bytes as a slot lays it out.
size_t payload(int layer, int role) {
    return ggml_row_size(type_of(layer, role), role == 2 ? hidden : latent) * size_t(role == 2 ? latent : hidden);
}
size_t padded(int layer, int role) { return (payload(layer, role) + 4095) & ~size_t(4095); }
size_t slot_offset(int layer, int role) {
    size_t at = 0;
    for (int i = 0; i < role; ++i) { at += padded(layer, i); }
    return at;
}
size_t expert_bytes(int layer) { return slot_offset(layer, 3); }
kimi_vram_expert_format declared(int layer) {
    return { latent, hidden, up_types[layer - 1], down_types[layer - 1], experts, 2, layer_ids, up_types, down_types };
}
// The request the sink builds: each role by its payload, at its own offset of one file.
kimi_vram_request key(int layer, int expert) {
    kimi_vram_request r = {}; r.layer = layer; r.expert = expert; r.weight = .5f;
    uint64_t offset = (uint64_t(layer) * experts + uint64_t(expert)) << 20;
    for (int i = 0; i < 3; ++i) { r.role[i] = { 1, 2, uint64_t(1) << 32, offset, payload(layer, i), 3, 4, 5, 6 }; offset += payload(layer, i); }
    return r;
}

size_t page_size() {
#ifdef _WIN32
    SYSTEM_INFO info; GetSystemInfo(&info); return size_t(info.dwPageSize);
#else
    return size_t(sysconf(_SC_PAGESIZE));
#endif
}
// `bytes` that end where an inaccessible page begins: reading past them faults.
struct guarded {
    char * base = nullptr, * data = nullptr;
    size_t span = 0;
    explicit guarded(size_t bytes) {
        const size_t page = page_size(), body = (bytes + page - 1) / page * page;
        span = body + page;
#ifdef _WIN32
        base = static_cast<char *>(VirtualAlloc(nullptr, span, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE));
        require(base != nullptr, "guarded allocation");
        DWORD previous = 0;
        require(VirtualProtect(base + body, page, PAGE_NOACCESS, &previous) != 0, "guard page");
#else
        void * mapped = mmap(nullptr, span, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        require(mapped != MAP_FAILED, "guarded mmap");
        base = static_cast<char *>(mapped);
        require(mprotect(base + body, page, PROT_NONE) == 0, "guard page");
#endif
        data = base + body - bytes;
    }
    ~guarded() {
#ifdef _WIN32
        VirtualFree(base, 0, MEM_RELEASE);
#else
        munmap(base, span);
#endif
    }
    guarded(const guarded &) = delete;
    guarded & operator=(const guarded &) = delete;
};

// Every expert's roles, each in its own guarded payload, and the values its quantized blocks decode to.
struct store {
    std::vector<std::unique_ptr<guarded>> roles[2][experts];
    std::vector<float> dense[2][experts][3];
    store() {
        std::mt19937 rng(20261109);
        std::normal_distribution<float> sample(0.0f, 0.06f);
        for (int layer = 1; layer <= 2; ++layer) {
            for (int e = 0; e < experts; ++e) {
                for (int role = 0; role < 3; ++role) {
                    const ggml_type type = type_of(layer, role);
                    const int rows = role == 2 ? latent : hidden, columns = role == 2 ? hidden : latent;
                    std::vector<float> values(size_t(rows) * columns);
                    for (auto & v : values) { v = sample(rng); }
                    roles[layer - 1][e].push_back(std::make_unique<guarded>(payload(layer, role)));
                    char * at = roles[layer - 1][e].back()->data;
                    require(ggml_quantize_chunk(type, values.data(), at, 0, rows, columns, nullptr) == payload(layer, role),
                        "quantized role bytes");
                    auto & out = dense[layer - 1][e][role];
                    out.resize(values.size());
                    ggml_get_type_traits(type)->to_float(at, out.data(), int64_t(out.size()));
                }
            }
        }
    }
    const void * role_at(int layer, int e, int role) const { return roles[layer - 1][e][size_t(role)]->data; }
    // the expert's row for input x, SwiGLU, times the request's weight, in double precision
    std::vector<float> row(int layer, int e, const std::vector<float> & x) const {
        const auto & m = dense[layer - 1][e];
        std::vector<double> h(static_cast<size_t>(hidden));
        for (int r = 0; r < hidden; ++r) {
            double u = 0, g = 0;
            for (int c = 0; c < latent; ++c) { u += double(m[0][size_t(r) * latent + c]) * x[size_t(c)]; g += double(m[1][size_t(r) * latent + c]) * x[size_t(c)]; }
            h[size_t(r)] = g / (1.0 + std::exp(-g)) * u;
        }
        std::vector<float> y(static_cast<size_t>(latent));
        for (int r = 0; r < latent; ++r) {
            double d = 0;
            for (int c = 0; c < hidden; ++c) { d += double(m[2][size_t(r) * hidden + c]) * h[size_t(c)]; }
            y[size_t(r)] = float(.5 * d);
        }
        return y;
    }
};

std::vector<float> input(int layer, int token) {
    std::vector<float> x(static_cast<size_t>(latent));
    for (int i = 0; i < latent; ++i) { x[size_t(i)] = std::sin(0.37f * float(i) + float(layer * 5 + token * 11)); }
    return x;
}
void compare(const store & s, int layer, const std::vector<int> & ids, const std::vector<bool> & on_gpu,
        const std::vector<float> & x, const std::vector<float> & rows, const char * what) {
    for (size_t j = 0; j < ids.size(); ++j) {
        if (!on_gpu[j]) { continue; }
        const auto wanted = s.row(layer, ids[j], x);
        double error = 0, norm = 0;
        for (int i = 0; i < latent; ++i) {
            const double d = double(rows[j * latent + size_t(i)]) - wanted[size_t(i)];
            error += d * d; norm += double(wanted[size_t(i)]) * wanted[size_t(i)];
        }
        const double relative = std::sqrt(error / norm);
        require(std::isfinite(relative) && relative < 2e-2, std::string(what) + ": layer " + std::to_string(layer) +
            " expert " + std::to_string(ids[j]) + " is " + std::to_string(relative) + " from its CPU row");
    }
}
// A slot as `layer` lays it out: each role's payload equal to the expert's, the rest of the role zero.
void check_slot(tier & t, const store & s, int slot, int layer, int e, const char * what) {
    std::vector<char> copied(expert_bytes(layer));
    cuda_check(cudaMemcpy(copied.data(), t.slot_base(size_t(slot)), copied.size(), cudaMemcpyDeviceToHost));
    for (int role = 0; role < 3; ++role) {
        const char * at = copied.data() + slot_offset(layer, role);
        require(!std::memcmp(at, s.role_at(layer, e, role), payload(layer, role)),
            std::string(what) + ": a role's payload differs from its expert's");
        bool zero = true;
        for (size_t i = payload(layer, role); i < padded(layer, role); ++i) { zero = zero && at[i] == 0; }
        require(zero, std::string(what) + ": a role's padding is not zero");
    }
}
// Once, after the first prepare allocates the slots and before anything is copied into them: every slot's bytes set
// to a pattern, so that padding a copy does not zero shows. Later copies land on the pattern or on another expert's
// bytes (an eviction gives a layer 1 slot to a layer 2 expert, whose roles lie elsewhere).
bool scribble_pending = true;
void scribble(tier & t) {
    if (!scribble_pending) { return; }
    scribble_pending = false;
    for (size_t slot = 0; slot < t.table->slots.size(); ++slot) {
        require(!t.table->slots[slot].valid, "a slot holds an expert before the first admission");
        cuda_check(cudaMemset(t.slot_base(slot), 0xa5, t.stride));
    }
    // the tier's copy streams do not wait for the default stream's memsets
    cuda_check(cudaDeviceSynchronize());
}
const kimi_vram_activation swiglu = { KIMI_VRAM_ACTIVATION_SWIGLU, 0.0f, 0.0f, 0.0f };
// At load, as the sink fills the tier from the drive (cpu-sink.cpp preload): reserve takes the storage for a request
// named by its payload, and preload copies experts no request has named from host memory that holds only their
// payloads into empty slots. Each slot holds the payloads with zero padding, a padded name of a payload-named role is
// refused, and the first decode token's request of a preloaded expert, keyed as the sink keys it, is a hit.
std::vector<int> layer_step(tier & t, const store & s, int layer, int token, const std::vector<int> & ids, const char * what,
        bool deferred = false);
void preload_case(fixture_inputs & inputs, const store & s) {
    if (ggml_backend_cuda_device_is_unified(0)) { std::printf("KIMI_VRAM_PAYLOAD preload case: unified memory, skipped\n"); return; }
    inputs.set("tier.keep_across_requests", "1");
    inputs.set("tier.elastic", "1");
    inputs.set("tier.demand_reclaim", "0");
    inputs.set("tier.margin_mib", "0");
    tier u(tier::automatic_budget, inputs.tier_policy());
    const auto f = declared(1);
    u.reserve(key(1, 0), f);
    require(u.has_storage() && u.table && u.growth_deferred, "the automatic tier did not take its storage at load");
    const size_t empty = u.preload_slots();
    require(empty >= 2, "no empty slots to preload");
    scribble_pending = true;
    scribble(u);
    const int preloaded[2][2] = { { 1, 4 }, { 2, 6 } };
    int slots[2] = { -1, -1 };
    for (int i = 0; i < 2; ++i) {
        const int layer = preloaded[i][0], e = preloaded[i][1];
        const void * data[3] = { s.role_at(layer, e, 0), s.role_at(layer, e, 1), s.role_at(layer, e, 2) };
        require(u.preload(key(layer, e), data), "a payload-named expert was not preloaded");
        slots[i] = u.table->find(key(layer, e), [](const vram_slot_table::entry & x) { return x.valid; });
        require(slots[i] >= 0 && u.table->slots[size_t(slots[i])].preloaded, "the preloaded expert is not a preloaded entry");
        check_slot(u, s, slots[i], layer, e, "a preloaded expert");
    }
    require(u.preload_slots() == empty - 2, "a preload did not take one empty slot an expert");
    {
        auto r = key(1, 5);
        r.role[2].bytes = padded(1, 2);
        const void * data[3] = { s.role_at(1, 5, 0), s.role_at(1, 5, 1), s.role_at(1, 5, 2) };
        std::string refused;
        try { u.preload(r, data); } catch (const std::runtime_error & e) { refused = e.what(); }
        require(refused == "preloaded expert has another layout than its layer's format",
            "a padded preload of a payload-named role was not refused: " + refused);
    }
    u.begin(61, 1, true);
    require(layer_step(u, s, 1, 1, { 4 }, "a preloaded expert")[0] == slots[0], "the preloaded layer 1 expert is not a hit");
    require(layer_step(u, s, 2, 1, { 6 }, "a preloaded expert")[0] == slots[1], "the preloaded layer 2 expert is not a hit");
    u.release();
}
// One decode layer as the sink runs it: prepare, launch, the misses' admissions (deferred where asked), finish.
std::vector<int> layer_step(tier & t, const store & s, int layer, int token, const std::vector<int> & ids, const char * what,
        bool deferred) {
    std::vector<kimi_vram_request> r;
    for (int e : ids) { r.push_back(key(layer, e)); }
    std::vector<int> slots(ids.size(), -1);
    const auto f = declared(layer);
    t.prepare_format(r.data(), int(r.size()), &f, slots.data());
    scribble(t);
    const auto x = input(layer, token);
    t.launch_activation(x.data(), x.size() * sizeof(float), &swiglu);
    for (size_t j = 0; j < ids.size(); ++j) {
        if (slots[j] >= 0) { continue; }
        const void * data[3] = { s.role_at(layer, ids[j], 0), s.role_at(layer, ids[j], 1), s.role_at(layer, ids[j], 2) };
        if (deferred) { int queued = 0; t.admit_deferred(r[j], data, &queued); require(queued == 1, "the deferred upload was not queued"); }
        else { t.admit(r[j], data, false); }
    }
    std::vector<float> rows(ids.size() * latent, 0.0f);
    t.finish(rows.data(), latent);
    std::vector<bool> hits;
    for (int slot : slots) { hits.push_back(slot >= 0); }
    compare(s, layer, ids, hits, x, rows, what);
    return slots;
}
}

int main() {
    fixture_inputs inputs; inputs.set_runtime_services(test_gpu_services()); try {
    ggml_cpu_init();
    inputs.set("vram_async", "1");
    inputs.set("vram_deferred", "1");
    inputs.set("tier.policy", "lru");
    inputs.set("tier.min_count", "0");
    inputs.set("tier.global", "1");
    for (int layer = 1; layer <= 2; ++layer) {
        require(payload(layer, 2) % 4096 != 0, "the fixture's downs must not be whole pages");
    }
    const store s;
    const size_t larger = (std::max)(expert_bytes(1), expert_bytes(2));
    tier t(4 * larger, inputs.tier_policy());

    // Token 1: every request is an admission; the first prepare takes both formats from roles named by their payload.
    t.begin(41, 1, true);
    layer_step(t, s, 1, 1, { 0, 1 }, "first admissions");
    layer_step(t, s, 2, 1, { 0 }, "first admissions");
    require(t.formats.size() == 2 && t.stride == larger, "the tier did not take both formats");
    // Token 2: the residents are hits, each slot holds its expert's payloads with zero padding, the rows are right.
    t.begin(41, 2, true);
    const auto first = layer_step(t, s, 1, 2, { 0, 1 }, "residents of layer 1");
    const auto second = layer_step(t, s, 2, 2, { 0 }, "residents of layer 2");
    require(first[0] >= 0 && first[1] >= 0 && second[0] >= 0, "the residents were not hits");
    check_slot(t, s, first[0], 1, 0, "a layer 1 admission");
    check_slot(t, s, first[1], 1, 1, "a layer 1 admission");
    check_slot(t, s, second[0], 2, 0, "a layer 2 admission");

    // Token 3: a deferred upload copies its payloads and zeroes the padding; token 4: it is a hit.
    t.begin(41, 3, true);
    const auto deferred = layer_step(t, s, 1, 3, { 2 }, "deferred admission", true);
    const int target = t.table->admissions[0];
    {
        kimi_vram_request collected[4] = {};
        int32_t aborted[4] = {};
        int count = 0;
        t.collect(-1, collected, aborted, 4, &count);
        require(count == 1 && !aborted[0] && deferred[0] < 0 && target >= 0, "the deferred upload was not collected");
        check_slot(t, s, target, 1, 2, "a deferred upload");
    }
    t.begin(41, 4, true);
    require(layer_step(t, s, 1, 4, { 2 }, "a deferred upload")[0] == target, "the deferred upload is not a hit");

    // Token 5: a host job of layer 2 whose roles are page-locked payloads: on a discrete GPU it is copied into its
    // admission reservation (payloads only) and computed there; token 6: the copy is a hit with zero padding.
    {
        char error[512] = {};
        for (int role = 0; role < 3; ++role) {
            require(host_register(&t, s.role_at(2, 3, role), payload(2, role), error, sizeof(error)) == 0,
                std::string("host register: ") + error);
        }
        t.begin(41, 5, true);
        const auto r = key(2, 3);
        int slot = -1;
        const auto f = declared(2);
        t.prepare_format(&r, 1, &f, &slot);
        require(slot < 0, "the host job's expert is already resident");
        const auto x = input(2, 5);
        t.launch_activation(x.data(), x.size() * sizeof(float), &swiglu);
        const kimi_vram_host_job job = { 0, .5f, { s.role_at(2, 3, 0), s.role_at(2, 3, 1), s.role_at(2, 3, 2) } };
        int accepted = 0;
        t.launch_host(&job, 1, &accepted);
        require(accepted == 1, "the host job was refused");
        std::vector<float> rows(static_cast<size_t>(latent), 0.0f);
        t.finish(rows.data(), latent);
        compare(s, 2, { 3 }, { true }, x, rows, "a host job");
        if (!t.host_readable) {
            t.begin(41, 6, true);
            const int copy = layer_step(t, s, 2, 6, { 3 }, "a host job's copy")[0];
            require(copy >= 0, "the host job's copy was not published");
            check_slot(t, s, copy, 2, 3, "a host job's copy");
        }
        for (int role = 0; role < 3; ++role) {
            require(host_unregister(&t, s.role_at(2, 3, role), error, sizeof(error)) == 0, std::string("host unregister: ") + error);
        }
    }
    // The first request named the format's roles by their payload; one that names a role padded is refused (its key
    // would miss every resident and seed of the expert).
    {
        t.begin(41, 7, true);
        auto r = key(1, 0);
        r.role[2].bytes = padded(1, 2);
        int slot = -1;
        const auto f = declared(1);
        std::string refused;
        try { t.prepare_format(&r, 1, &f, &slot); } catch (const std::runtime_error & e) { refused = e.what(); }
        require(refused == "VRAM role bytes changed", "a padded name of a payload-named role was not refused: " + refused);
    }
    t.release();
    preload_case(inputs, s);
    std::printf("KIMI_VRAM_PAYLOAD_PASS assertions=%d layer1=%s/%s down=%zu/%zu layer2=%s/%s role=%zu/%zu\n", assertions,
        ggml_type_name(type_of(1, 0)), ggml_type_name(type_of(1, 2)), payload(1, 2), padded(1, 2),
        ggml_type_name(type_of(2, 0)), ggml_type_name(type_of(2, 2)), payload(2, 0), padded(2, 0));
    return 0;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "KIMI_VRAM_PAYLOAD_FAIL: %s\n", e.what());
        return 1;
    }
}
