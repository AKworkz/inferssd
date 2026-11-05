#include "explicit-fixture-inputs.h"
// CUDA test of a tier whose layers' experts have different formats, declared per layer: as Muse Spark Q4_K_S's (down
// Q4_K in most layers, Q5_K in some), layer 1 Q4_K/Q4_K and layer 2 Q4_K/Q5_K; and MXFP4 beside them, layer 1
// MXFP4/MXFP4 and layer 2 Q4_K/Q5_K. Slots are the larger expert's; both layers' residents share the one pool, an
// eviction gives a slot to an expert of the other format, a deferred upload copies its own layer's roles while a layer
// of the other format is prepared, and a host job is copied as its layer lays it out; each layer's rows match its
// experts computed on the CPU. Declarations that change and calls that differ from theirs are refused.
#include "gpu-services.h"
#include "vram-tier.cpp"
#include <cmath>
#include <cstring>
#include <random>
#include <sys/mman.h>

namespace {
std::mutex gate_mutex;
std::condition_variable gate_cv;
bool held = false, entered = false;
bool upload_gate(void *) {
    std::unique_lock<std::mutex> lock(gate_mutex);
    entered = true; gate_cv.notify_all();
    gate_cv.wait(lock, [] { return !held; });
    return true;
}
void hold() { std::lock_guard<std::mutex> lock(gate_mutex); held = true; entered = false; }
void open_gate() { std::lock_guard<std::mutex> lock(gate_mutex); held = false; gate_cv.notify_all(); }
bool copy_started() {
    std::unique_lock<std::mutex> lock(gate_mutex);
    return gate_cv.wait_for(lock, std::chrono::seconds(5), [] { return entered; });
}
int assertions = 0;
void require(bool ok, const std::string & what) { ++assertions; if (!ok) throw std::runtime_error(what); }

constexpr int latent = 256, hidden = 256, experts = 8;
const int32_t layer_ids[2] = { 1, 2 };
// the model's layers' types (run sets them)
int32_t up_types[2] = {}, down_types[2] = {};
size_t role_bytes(int layer, int role) {
    const ggml_type type = ggml_type(role == 2 ? down_types[layer - 1] : up_types[layer - 1]);
    return (ggml_row_size(type, role == 2 ? hidden : latent) * size_t(role == 2 ? latent : hidden) + 4095) & ~size_t(4095);
}
size_t expert_bytes(int layer) { return role_bytes(layer, 0) + role_bytes(layer, 1) + role_bytes(layer, 2); }
// the declaration every call makes, with the types of the call's layer
kimi_vram_expert_format declared(int layer) {
    return { latent, hidden, up_types[layer - 1], down_types[layer - 1], experts, 2, layer_ids, up_types, down_types };
}
kimi_vram_request key(int layer, int expert) {
    kimi_vram_request r = {}; r.layer = layer; r.expert = expert; r.weight = .5f;
    uint64_t offset = (uint64_t(layer) * experts + uint64_t(expert)) << 20;
    for (int i = 0; i < 3; ++i) { r.role[i] = { 1, 2, uint64_t(1) << 32, offset, role_bytes(layer, i), 3, 4, 5, 6 }; offset += role_bytes(layer, i); }
    return r;
}

// Every expert of both layers, its roles back to back as the pack pads them, in page-aligned host memory (a host job
// copies from it once registered), and the values its quantized blocks decode to.
struct store {
    size_t place = 0, bytes = 0;
    char * base = nullptr;
    std::vector<float> dense[2][experts][3];
    store() {
        place = std::max(expert_bytes(1), expert_bytes(2));
        bytes = size_t(2 * experts) * place;
        void * mapped = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        require(mapped != MAP_FAILED, "expert store mmap");
        base = static_cast<char *>(mapped);
        std::mt19937 rng(20261103);
        std::normal_distribution<float> sample(0.0f, 0.06f);
        for (int layer = 1; layer <= 2; ++layer) {
            for (int e = 0; e < experts; ++e) {
                for (int role = 0; role < 3; ++role) {
                    const ggml_type type = ggml_type(role == 2 ? down_types[layer - 1] : up_types[layer - 1]);
                    const int rows = role == 2 ? latent : hidden, columns = role == 2 ? hidden : latent;
                    std::vector<float> values(size_t(rows) * columns);
                    for (auto & v : values) { v = sample(rng); }
                    char * at = const_cast<char *>(static_cast<const char *>(role_at(layer, e, role)));
                    require(ggml_quantize_chunk(type, values.data(), at, 0, rows, columns, nullptr) == ggml_row_size(type, columns) * rows,
                        "quantized expert bytes");
                    auto & out = dense[layer - 1][e][role];
                    out.resize(values.size());
                    ggml_get_type_traits(type)->to_float(at, out.data(), int64_t(out.size()));
                }
            }
        }
    }
    ~store() { munmap(base, bytes); }
    const char * expert(int layer, int e) const { return base + (size_t(layer - 1) * experts + size_t(e)) * place; }
    const void * role_at(int layer, int e, int role) const {
        return expert(layer, e) + (role >= 1 ? role_bytes(layer, 0) : 0) + (role == 2 ? role_bytes(layer, 1) : 0);
    }
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
// the GPU's rows (only its hits' and its host jobs' are written) against the experts' rows on the CPU
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
const kimi_vram_activation swiglu = { KIMI_VRAM_ACTIVATION_SWIGLU, 0.0f, 0.0f, 0.0f };
// One decode layer as the sink runs it: prepare, launch, the misses' admissions (deferred where asked), finish.
// Returns the slots prepare gave.
std::vector<int> layer_step(tier & t, const store & s, int layer, int token, const std::vector<int> & ids, const char * what,
        bool deferred = false) {
    std::vector<kimi_vram_request> r;
    for (int e : ids) { r.push_back(key(layer, e)); }
    std::vector<int> slots(ids.size(), -1);
    const auto f = declared(layer);
    t.prepare_format(r.data(), int(r.size()), &f, slots.data());
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
// A refused call through the C ABI: its error names `wanted`, and the tier is released.
void refused(tier & t, int layer, const kimi_vram_expert_format & f, const char * wanted) {
    const auto r = key(layer, 0);
    int slot = -1;
    char error[512] = {};
    t.begin(77, 1, true);
    require(format_prepare(&t, &r, 1, &f, &slot, error, sizeof(error)) != 0 && std::strstr(error, wanted),
        std::string("not refused with \"") + wanted + "\": " + error);
    require(!t.table && !t.arena && t.blocks.empty(), "a refusal left the tier's storage");
}

void run(fixture_inputs & inputs, ggml_type first_layer) {
    up_types[0] = down_types[0] = first_layer;
    up_types[1] = GGML_TYPE_Q4_K; down_types[1] = GGML_TYPE_Q5_K;
    const store s;
    const size_t larger = expert_bytes(2);
    require(larger > expert_bytes(1), "the fixture's second layer must have the larger expert");
    tier t(3 * larger, inputs.tier_policy()); // three slots, one pool for both layers
    struct cleanup { ~cleanup() { open_gate(); } } cleanup_on_error; // before the tier waits for a held upload
    t.deferred_hook = upload_gate;

    // Token 1: the declaration gives two formats and slots of the larger expert; every request is an admission.
    t.begin(41, 1, true);
    layer_step(t, s, 1, 1, { 0, 1 }, "first admissions");
    require(t.formats.size() == 2 && t.stride == larger && t.table && t.table->slots.size() == 3,
        "the tier did not take two formats and slots of the larger expert");
    require(t.capacity_bytes == 3 * larger, "the arena is not three slots of the larger expert");
    layer_step(t, s, 2, 1, { 0 }, "first admissions");
    // Token 2: residents of both layers in the one pool, each computed by its own format's engine.
    t.begin(41, 2, true);
    const auto first = layer_step(t, s, 1, 2, { 0, 1 }, "residents of layer 1");
    const auto second = layer_step(t, s, 2, 2, { 0 }, "residents of layer 2");
    require(first[0] >= 0 && first[1] >= 0 && second[0] >= 0, "the residents of both layers were not hits");
    {
        std::vector<char> copied(larger);
        cuda_check(cudaMemcpy(copied.data(), t.slot_base(size_t(second[0])), larger, cudaMemcpyDeviceToHost));
        require(!std::memcmp(copied.data(), s.expert(2, 0), larger), "a layer 2 resident holds other bytes than its expert");
    }
    // Token 3: a layer 2 expert takes the least recent slot, a layer 1 expert's; token 4: it is a hit there.
    t.begin(41, 3, true);
    layer_step(t, s, 2, 3, { 1 }, "eviction");
    require(t.table->slots[size_t(first[0])].valid && t.table->slots[size_t(first[0])].key.layer == 2,
        "a layer 2 expert did not take the least recent slot, a layer 1 expert's");
    t.begin(41, 4, true);
    layer_step(t, s, 1, 4, { 1 }, "a layer 1 resident");
    require(layer_step(t, s, 2, 4, { 1 }, "a layer 2 expert in a layer 1 slot")[0] == first[0],
        "the layer 2 expert was not a hit in the former layer 1 slot");
    // Token 5: a layer 1 expert takes the least recent slot, now the first layer 2 expert's; token 6: a hit there.
    t.begin(41, 5, true);
    layer_step(t, s, 1, 5, { 0 }, "eviction");
    require(t.table->slots[size_t(second[0])].valid && t.table->slots[size_t(second[0])].key.layer == 1,
        "a layer 1 expert did not take the least recent slot, a layer 2 expert's");
    t.begin(41, 6, true);
    require(layer_step(t, s, 1, 6, { 0 }, "a layer 1 expert in a layer 2 slot")[0] == second[0],
        "the layer 1 expert was not a hit in the former layer 2 slot");

    // Token 7: a deferred layer 2 upload is held while a layer 1 layer is prepared and computed; it copies its own
    // layer's roles, the whole of the larger expert.
    t.begin(41, 7, true);
    hold();
    const auto deferred = layer_step(t, s, 2, 7, { 2 }, "deferred admission", true);
    require(copy_started(), "the deferred upload did not start");
    const int target = t.table->admissions[0];
    layer_step(t, s, 1, 7, { 0 }, "a layer 1 hit while a layer 2 upload is in flight");
    require(t.now().down_type == down_types[0], "the tier did not select layer 1's format");
    open_gate();
    {
        kimi_vram_request collected[4] = {};
        int32_t aborted[4] = {};
        int count = 0;
        t.collect(-1, collected, aborted, 4, &count);
        require(count == 1 && !aborted[0] && collected[0].layer == 2 && deferred[0] < 0 && target >= 0,
            "the deferred layer 2 upload was not collected");
        std::vector<char> copied(larger);
        cuda_check(cudaMemcpy(copied.data(), t.slot_base(size_t(target)), larger, cudaMemcpyDeviceToHost));
        require(!std::memcmp(copied.data(), s.expert(2, 2), larger),
            "the deferred upload did not copy its own layer's roles");
    }
    t.begin(41, 8, true);
    require(layer_step(t, s, 2, 8, { 2 }, "a deferred layer 2 upload")[0] == target, "the deferred upload is not a hit");

    // Token 9: a layer 2 host job, copied into the expert's admission reservation on a discrete GPU (from page-locked
    // memory) and computed in place where the GPU reads host memory; token 10 on a discrete GPU: the copy is a hit.
    {
        char error[512] = {};
        require(host_register(&t, s.base, s.bytes, error, sizeof(error)) == 0, std::string("host register: ") + error);
        t.begin(41, 9, true);
        const auto r = key(2, 3);
        int slot = -1;
        const auto f = declared(2);
        t.prepare_format(&r, 1, &f, &slot);
        require(slot < 0, "the host job's expert is already resident");
        const auto x = input(2, 9);
        t.launch_activation(x.data(), x.size() * sizeof(float), &swiglu);
        const kimi_vram_host_job job = { 0, .5f, { s.role_at(2, 3, 0), s.role_at(2, 3, 1), s.role_at(2, 3, 2) } };
        int accepted = 0;
        t.launch_host(&job, 1, &accepted);
        require(accepted == 1, "the host job was refused");
        std::vector<float> rows(static_cast<size_t>(latent), 0.0f);
        t.finish(rows.data(), latent);
        compare(s, 2, { 3 }, { true }, x, rows, "a host job");
        if (!t.host_readable) {
            t.begin(41, 10, true);
            require(layer_step(t, s, 2, 10, { 3 }, "a host job's copy")[0] >= 0, "the host job's copy was not published");
        }
        require(host_unregister(&t, s.base, error, sizeof(error)) == 0, std::string("host unregister: ") + error);
    }
    t.release();

    // Refusals. A later call whose declaration differs, a call whose types differ from its layer's declaration, a
    // geometry that differs from the first call's.
    {
        tier u(3 * larger, inputs.tier_policy());
        u.begin(77, 1, true);
        layer_step(u, s, 1, 1, { 0 }, "a declared layer");
        const int32_t uniform_down[2] = { down_types[0], down_types[0] };
        auto changed = declared(1); changed.layer_down_types = uniform_down;
        refused(u, 1, changed, "VRAM expert format changed");
        auto wrong = declared(2); wrong.down_type = GGML_TYPE_Q4_K;
        refused(u, 2, wrong, "differs from its declaration");
        auto wider = declared(1); wider.hidden = 2 * hidden;
        refused(u, 1, wider, "geometry changed");
    }
    std::printf("KIMI_VRAM_MIXED_MODEL_PASS layer1=%s/%s layer2=%s/%s stride=%zu layer1_expert=%zu layer2_expert=%zu\n",
        ggml_type_name(ggml_type(up_types[0])), ggml_type_name(ggml_type(down_types[0])),
        ggml_type_name(ggml_type(up_types[1])), ggml_type_name(ggml_type(down_types[1])), larger, expert_bytes(1),
        expert_bytes(2));
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
    run(inputs, GGML_TYPE_Q4_K);
    run(inputs, GGML_TYPE_MXFP4);
    std::printf("KIMI_VRAM_MIXED_PASS assertions=%d\n", assertions);
    return 0;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "KIMI_VRAM_MIXED_FAIL: %s\n", e.what());
        return 1;
    }
}
