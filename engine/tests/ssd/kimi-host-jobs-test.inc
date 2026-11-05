#include "explicit-fixture-inputs.h"
// The tier's host path (kimi_vram_host_api v3) through its C API, used by test-kimi-vram-generic (Q8_0 and MXFP4).
// Included after test-kimi-cpu-sink.cpp (require, ggml).
#pragma once
#include "ggml-ssd/gpu/kimi-vram-tier-api.h"
#include <dlfcn.h>
#include <functional>
#include <sys/mman.h>

// The host path of the tier (kimi_vram_host_api v3) through its C API, on experts of one type:
// a host job computes the same rows as the same expert resident (same kernel, same bytes, bit for bit); on a discrete
// GPU a pageable source is refused, a page-locked one is copied into the expert's admission reservation, which finish
// publishes (the next prepare hits it), or into a scratch slot the tier allocates at the token boundary after the first
// registration; that registration's range is copied once, timed, when no copy has been; a batch wider than the free
// scratch slots is refused; the costs report measured lines.
inline void run_host_jobs(fixture_inputs & inputs, ggml_type type, size_t & checks) {
    const auto * tier_api = reinterpret_cast<kimi_vram_tier_get_api_fn>(dlsym(RTLD_DEFAULT, "kimi_vram_tier_get_api"))();
    const auto * format_api = reinterpret_cast<kimi_vram_format_get_api_fn>(dlsym(RTLD_DEFAULT, "kimi_vram_format_get_api"))();
    const auto * host_api = reinterpret_cast<kimi_vram_host_get_api_fn>(dlsym(RTLD_DEFAULT, "kimi_vram_host_get_api"))();
    require(tier_api && format_api && host_api && host_api->version == 3 && host_api->costs2 && host_api->register_host,
        "host API v2 missing from the tier DSO");
    const int latent = 512, hidden = 256, n_experts = 8;
    auto pad = [](size_t b) { return (b + 4095) & ~size_t(4095); };
    const size_t up = pad(ggml_row_size(type, latent) * hidden), down = pad(ggml_row_size(type, hidden) * latent);
    const size_t stride = 2 * up + down;
    inputs.set("vram_async", "0");
    inputs.set("vram_deferred", "0");
    inputs.set("tier.global", "0");
    inputs.set("tier.elastic", "0");
    inputs.set("tier.keep_across_requests", "1");
    inputs.set("tier.policy", "lru");
    inputs.set("tier.min_count", "0");
    char error[512] = {};
    const auto get_fixture = reinterpret_cast<ssd::gpu::testing::get_api>(
        dlsym(RTLD_DEFAULT, "ggml_ssd_tier_testing_get_api"));
    require(get_fixture != nullptr, "explicit tier fixture factory missing");
    const auto * fixture_api = get_fixture();
    require(fixture_api && fixture_api->version == 3 &&
        fixture_api->size == sizeof(ssd::gpu::testing::api) && fixture_api->create,
        "explicit tier fixture factory ABI mismatch");
    const auto options = inputs.tier_policy();
    const auto create_tier = [&] { return fixture_api->create(2 * stride, &options, error, sizeof(error)); };
    void * t = create_tier();
    require(t != nullptr, std::string("tier create: ") + error);
    // five experts in host memory, blocks with small scales, in an anonymous mapping like the RAM cache's
    const int count = 5;
    char * host = static_cast<char *>(mmap(nullptr, count * stride, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0));
    require(host != MAP_FAILED, "host experts mmap");
    uint32_t state = 20261022;
    auto next = [&] { state = state * 1664525u + 1013904223u; return state >> 8; };
    for (int e = 0; e < count; ++e) {
        for (size_t role = 0; role < 3; ++role) {
            char * base = host + size_t(e) * stride + (role ? up : 0) + (role == 2 ? up : 0);
            const size_t rows = role == 2 ? size_t(latent) : size_t(hidden), columns = role == 2 ? size_t(hidden) : size_t(latent);
            for (size_t r = 0; r < rows; ++r) {
                char * row = base + r * ggml_row_size(type, columns);
                for (size_t b = 0; b < columns / 32; ++b) {
                    if (type == GGML_TYPE_MXFP4) { // block_mxfp4: an E8M0 scale, 16 bytes of 4-bit values
                        row[17 * b] = char(116 + next() % 5);
                        for (int k = 0; k < 16; ++k) { row[17 * b + 1 + k] = char(next() & 0xff); }
                        continue;
                    }
                    // block_q8_0: an fp16 scale, 32 int8 values
                    const ggml_fp16_t d = ggml_fp32_to_fp16(0.002f + 0.001f * float(next() % 7));
                    std::memcpy(row + 34 * b, &d, 2);
                    for (int k = 0; k < 32; ++k) { row[34 * b + 2 + k] = char(int(next() % 255) - 127); }
                }
            }
        }
    }
    auto key = [&](int expert) {
        kimi_vram_request r = {};
        r.layer = 1; r.expert = expert; r.weight = 0.5f;
        const size_t bytes[3] = { up, up, down };
        uint64_t offset = 0;
        for (int i = 0; i < 3; ++i) { r.role[i] = { 9, 99, 1u << 30, uint64_t(expert) * stride + offset, bytes[i], 1, 2, 3, 4 }; offset += bytes[i]; }
        return r;
    };
    auto payload = [&](int expert, int role) -> const void * {
        return host + size_t(expert) * stride + (role >= 1 ? up : 0) + (role == 2 ? up : 0);
    };
    const int32_t layers[1] = { 1 };
    const kimi_vram_expert_format format = { latent, hidden, int(type), int(type), n_experts, 1, layers };
    const kimi_vram_activation act = { KIMI_VRAM_ACTIVATION_SWIGLU_CLAMP, 0, 0, 7.0f };
    std::vector<float> input(latent);
    for (int i = 0; i < latent; ++i) { input[size_t(i)] = std::sin(0.37f * float(i)); }
    // the engine's input row: F32
    std::vector<uint8_t> row_in(input.size() * sizeof(float));
    std::memcpy(row_in.data(), input.data(), row_in.size());
    uint64_t token = 0;
    // one layer: begin, prepare the experts, launch, then `between` (host jobs, admissions), finish; the rows
    struct layer_result { std::vector<int> slots; std::vector<float> rows; };
    auto layer = [&](const std::vector<int> & experts, const std::function<void(const std::vector<int> &)> & between) {
        require(tier_api->begin(t, 1, ++token, 1, error, sizeof(error)) == 0, std::string("begin: ") + error);
        std::vector<kimi_vram_request> r;
        for (int e : experts) { r.push_back(key(e)); }
        layer_result out;
        out.slots.assign(experts.size(), -1);
        require(format_api->prepare(t, r.data(), int(r.size()), &format, out.slots.data(), error, sizeof(error)) == 0,
            std::string("prepare: ") + error);
        require(format_api->launch(t, row_in.data(), row_in.size(), &act, error, sizeof(error)) == 0,
            std::string("launch: ") + error);
        between(out.slots);
        out.rows.assign(experts.size() * size_t(latent), 0.0f);
        require(tier_api->finish(t, out.rows.data(), size_t(latent), error, sizeof(error)) == 0, std::string("finish: ") + error);
        return out;
    };
    auto job = [&](int row, int expert) {
        return kimi_vram_host_job{ row, 0.5f, { payload(expert, 0), payload(expert, 1), payload(expert, 2) } };
    };
    auto launch = [&](const std::vector<kimi_vram_host_job> & jobs) {
        int accepted = -1;
        require(host_api->launch(t, jobs.data(), int(jobs.size()), &accepted, error, sizeof(error)) == 0, std::string("host launch: ") + error);
        return accepted;
    };
    auto admit = [&](int row, int expert) {
        const kimi_vram_request r = key(expert);
        require(tier_api->admit(t, &r, payload(expert, 0), payload(expert, 1), payload(expert, 2), 0, error, sizeof(error)) == 0,
            std::string("admit: ") + error);
        (void) row;
    };
    // In place or not: a tier reads the device's attributes at its first layer, so ask one that has run a layer (before
    // that every tier reports in_place=0, also on a unified device).
    kimi_vram_host_costs c = {};
    bool in_place = false;
    {
        void * main_tier = t;
        t = create_tier();
        require(t != nullptr, std::string("tier create: ") + error);
        layer({ 0 }, [&](const std::vector<int> &) {});
        host_api->costs2(t, &c);
        in_place = c.in_place != 0;
        tier_api->destroy(t);
        t = main_tier;
    }
    if (!in_place) {
        // Before any page-locked copy (a tier of its own): the first registration asks for the scratch slots; the next
        // decode token boundary copies its range into one, timed, and the boundary after reads the time. No job ran.
        void * main_tier = t;
        t = create_tier();
        require(t != nullptr, std::string("tier create: ") + error);
        layer({ 0 }, [&](const std::vector<int> &) {});
        require(host_api->register_host(t, host, count * stride, error, sizeof(error)) == 0, std::string("register: ") + error);
        host_api->costs2(t, &c);
        require(c.copy_us_per_mb == 0 && c.slots == 2, "costs: no copy measured before the probe");
        layer({ 0 }, [&](const std::vector<int> &) {});
        layer({ 0 }, [&](const std::vector<int> &) {});
        host_api->costs2(t, &c);
        require(c.copy_us_per_mb > 0 && c.slots == 2 && c.host_call_us == 0, "the first registration did not measure a copy");
        const double probe_us_per_mb = c.copy_us_per_mb;
        require(host_api->unregister_host(t, host, error, sizeof(error)) == 0, std::string("unregister: ") + error);
        tier_api->destroy(t);
        // A range unregistered before its probe started is not copied.
        t = create_tier();
        require(t != nullptr, std::string("tier create: ") + error);
        layer({ 0 }, [&](const std::vector<int> &) {});
        require(host_api->register_host(t, host, count * stride, error, sizeof(error)) == 0, std::string("register: ") + error);
        require(host_api->unregister_host(t, host, error, sizeof(error)) == 0, std::string("unregister: ") + error);
        layer({ 0 }, [&](const std::vector<int> &) {});
        layer({ 0 }, [&](const std::vector<int> &) {});
        host_api->costs2(t, &c);
        require(c.copy_us_per_mb == 0, "a probe copied from an unregistered range");
        tier_api->destroy(t);
        t = main_tier;
        checks += 4;
        std::printf("HOST_PROBE_PASS type=%s copy_us_per_mb=%.2f\n", ggml_type_name(type), probe_us_per_mb);
    }
    // token 1: expert 0 misses and is admitted (resident from token 2)
    layer({ 0 }, [&](const std::vector<int> &) { admit(0, 0); });
    // token 2: 0 hits; 1 misses with a reservation. Pageable: refused (discrete). Page-locked: accepted.
    std::vector<float> host_row;
    auto r2 = layer({ 0, 1 }, [&](const std::vector<int> & slots) {
        require(slots[0] >= 0 && slots[1] < 0, "token 2: expert 0 resident, expert 1 not");
        if (!in_place) { require(launch({ job(1, 1) }) == 0, "a pageable host job was accepted on a discrete GPU"); ++checks; }
        host_api->costs2(t, &c);
        require(in_place || ((c.reserved_rows & 2u) && !(c.reserved_rows & 1u) && c.slots == 2),
            "costs: row 1 holds a reservation, two scratch slots before they are allocated");
        require(host_api->register_host(t, host, count * stride, error, sizeof(error)) == 0, std::string("register: ") + error);
        require(launch({ job(1, 1) }) == 1, "a page-locked host job was refused");
        host_api->costs2(t, &c);
        require(in_place || !(c.reserved_rows & 2u), "costs: the job took row 1's reservation");
        checks += 2;
        admit(1, 1); // the job took the reservation: this admit has nothing left to do
    });
    host_row.assign(r2.rows.begin() + latent, r2.rows.begin() + 2 * latent);
    // token 3: expert 1 is resident now (the job's copy was its admission on a discrete GPU, the admit otherwise)
    auto r3 = layer({ 1 }, [&](const std::vector<int> &) {});
    require(r3.slots[0] >= 0, "the host job's reservation was not published");
    require(!std::memcmp(host_row.data(), r3.rows.data(), size_t(latent) * sizeof(float)),
        "a host job's rows differ from the same expert resident");
    checks += 2;
    if (!in_place) {
        // tokens 4-5: 0 and 1 hit and lock both slots, so 2 has no reservation: it needs a scratch slot. The
        // registration at token 2 asked for two, allocated at token 3's boundary.
        layer({ 0, 1, 2 }, [&](const std::vector<int> & slots) {
            require(slots[0] >= 0 && slots[1] >= 0 && slots[2] < 0, "token 4 residency");
            require(launch({ job(2, 2) }) == 1, "a job without reservation was refused a scratch slot");
        });
        auto r5 = layer({ 0, 1, 2, 3, 4 }, [&](const std::vector<int> &) {
            host_api->costs2(t, &c);
            require(c.slots == 2, "costs: two free scratch slots");
            require(launch({ job(2, 2), job(3, 3), job(4, 4) }) == 0, "three jobs in two scratch slots were accepted");
            require(launch({ job(2, 2), job(3, 3) }) == 1, "two jobs in two scratch slots were refused");
            host_api->costs2(t, &c);
            require(c.slots == 0, "costs: no free scratch slot left in the layer");
        });
        checks += 2;
        // tokens 6-7: expert 2 resident (admitted), its rows from the tier against token 5's scratch rows
        layer({ 2 }, [&](const std::vector<int> &) { admit(0, 2); });
        auto r7 = layer({ 2 }, [&](const std::vector<int> &) {});
        require(r7.slots[0] >= 0 && !std::memcmp(r5.rows.data() + 2 * latent, r7.rows.data(), size_t(latent) * sizeof(float)),
            "a scratch-slot job's rows differ from the same expert resident");
        checks += 4;
    }
    host_api->costs2(t, &c);
    require(c.resident_call_us > 0 && c.host_call_us > 0 && (in_place || (c.copy_us_per_mb > 0 && c.slots == 2)),
        "host costs not measured");
    require(host_api->unregister_host(t, host, error, sizeof(error)) == 0, std::string("unregister: ") + error);
    checks += 2;
    std::printf("HOST_JOBS_PASS type=%s in_place=%d resident_call_us=%.1f resident_expert_us=%.1f host_call_us=%.1f host_expert_us=%.1f copy_us_per_mb=%.2f slots=%d\n",
        ggml_type_name(type), int(in_place), c.resident_call_us, c.resident_expert_us, c.host_call_us, c.host_expert_us, c.copy_us_per_mb, c.slots);
    tier_api->destroy(t);
    munmap(host, count * stride);
}
