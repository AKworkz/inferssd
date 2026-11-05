// SPDX-License-Identifier: Apache-2.0
// Context-owned dense transport on a synthetic direct-I/O file. The original
// 1/64-token, reshape, resident-buffer and bitwise device-reference cases remain.
// Invalid source ranges fail at startup; a later short read fails the graph and
// cannot fall back to reading the mapped weight.
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ssd/options.h"
#include "ggml-ssd/gpu/provider.h"
#include "ggml-ssd/gpu/transport.h"
#include <memory>
#include <fstream>
#include <iterator>

#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(__linux__)
#include "ggml-cuda.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

namespace {

void require(bool ok, const std::string & message) {
    if (!ok) {
        throw std::runtime_error(message);
    }
}

constexpr int64_t K = 512;          // columns: 16 Q8_0 blocks of 34 bytes, 544 bytes a row
constexpr int64_t M = 256;          // rows
constexpr int n_blocks = 4;         // 64 rows a block, 34816 bytes: not a multiple of the page
constexpr int n_drive = 4;          // w0..w2 directly, w3 through a reshape
constexpr uint64_t base = 4096 + 96;   // the first matrix's offset: 32-byte aligned as in a GGUF, not page aligned

struct temp_file {
    std::string path;
    int fd = -1;
    explicit temp_file(const char * tag) {
        const char * tmp = std::getenv("TMPDIR");
        std::string pattern = std::string(tmp && *tmp ? tmp : "/tmp") + "/inferssd-dense-stream-" + tag + "-XXXXXX";
        std::vector<char> buf(pattern.begin(), pattern.end());
        buf.push_back('\0');
        fd = mkstemp(buf.data());
        require(fd >= 0, "mkstemp failed");
        path = buf.data();
    }
    ~temp_file() {
        if (fd >= 0) {
            close(fd);
        }
        unlink(path.c_str());
    }
    void write(const void * data, size_t bytes, uint64_t offset) const {
        const uint8_t * p = static_cast<const uint8_t *>(data);
        for (size_t done = 0; done < bytes;) {
            const ssize_t n = pwrite(fd, p + done, bytes - done, off_t(offset + done));
            require(n > 0 || errno == EINTR, "pwrite failed");
            done += n > 0 ? size_t(n) : 0;
        }
    }
};

struct copy_delta {
    uint64_t tensors, bytes, readahead, fallbacks;
};

copy_delta copies_since(ggml_backend_sched_t sched, const ggml_backend_sched_kimi_copy_stats & a) {
    ggml_backend_sched_kimi_copy_stats b = {};
    ggml_backend_sched_kimi_copy_stats_get(sched, &b);
    return { b.dense_tensors - a.dense_tensors, b.dense_bytes - a.dense_bytes, b.dense_readahead - a.dense_readahead,
             b.dense_fallbacks - a.dense_fallbacks };
}

// out = w0 x + w1 x + w2 x + reshape(w3) x3 (as 2-D) + bad x + r x: w0..w3 and bad on the drive, r resident; side =
// ram x, ram a weight in a plain CPU buffer that the index holds too
struct graph_out {
    std::vector<float> out, side;
    int splits = 0;
    ggml_status status = GGML_STATUS_SUCCESS;
};

graph_out run_graph(ggml_backend_sched_t sched, const std::vector<ggml_tensor *> & w,
                    ggml_tensor * bad, ggml_tensor * r, ggml_tensor * ram, int n_tokens,
                    const std::vector<float> & x_data, bool expect_failure = false) {
    ggml_init_params params = { ggml_tensor_overhead() * 64 + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(params);
    ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, n_tokens);
    ggml_set_input(x);
    ggml_tensor * out = ggml_mul_mat(ctx, w[0], x);
    out = ggml_add(ctx, out, ggml_mul_mat(ctx, w[1], x));
    out = ggml_add(ctx, out, ggml_mul_mat(ctx, w[2], x));
    // w3 [K, M] seen as [K, M/2, 2] against x repeated as [K, n, 2]: [M/2, n, 2], laid out as [M, n] again
    ggml_tensor * x3 = ggml_cont(ctx, ggml_repeat(ctx, ggml_reshape_3d(ctx, x, K, n_tokens, 1),
                                                  ggml_new_tensor_3d(ctx, GGML_TYPE_F32, K, n_tokens, 2)));
    ggml_tensor * y3 = ggml_mul_mat(ctx, ggml_reshape_3d(ctx, w[3], K, M / 2, 2), x3);
    out = ggml_add(ctx, out, ggml_reshape_2d(ctx, ggml_cont(ctx, ggml_permute(ctx, y3, 0, 2, 1, 3)), M, n_tokens));
    out = ggml_add(ctx, out, ggml_mul_mat(ctx, bad, x));
    out = ggml_add(ctx, out, ggml_mul_mat(ctx, r, x));
    ggml_set_output(out);
    ggml_tensor * side = ram ? ggml_mul_mat(ctx, ram, x) : nullptr;
    if (side) {
        ggml_set_output(side);
    }
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);
    if (side) {
        ggml_build_forward_expand(gf, side);
    }
    ggml_backend_sched_reset(sched);
    require(ggml_backend_sched_alloc_graph(sched, gf), "graph allocation failed");
    ggml_backend_tensor_set(x, x_data.data(), 0, sizeof(float) * K * n_tokens);
    graph_out g;
    g.status = ggml_backend_sched_graph_compute(sched, gf);
    if (expect_failure) {
        require(g.status != GGML_STATUS_SUCCESS, "failed direct read was accepted or fell back to mmap");
        ggml_backend_sched_synchronize(sched);
        ggml_backend_sched_reset(sched);
        ggml_free(ctx);
        return g;
    }
    require(g.status == GGML_STATUS_SUCCESS, "graph compute failed");
    g.out.resize(size_t(M) * n_tokens);
    ggml_backend_tensor_get(out, g.out.data(), 0, sizeof(float) * g.out.size());
    if (side) {
        g.side.resize(size_t(M) * n_tokens);
        ggml_backend_tensor_get(side, g.side.data(), 0, sizeof(float) * g.side.size());
    }
    g.splits = ggml_backend_sched_get_n_splits(sched);
    ggml_free(ctx);
    return g;
}

void run() {
    const size_t row_bytes = ggml_row_size(GGML_TYPE_Q8_0, K);
    const size_t nbytes = row_bytes * M;
    const size_t block_bytes = nbytes / n_blocks;
    const int n_mats = n_drive + 3;     // w0..w3, bad, r, ram
    const char * names[n_mats] = { "blk.0.attn_q.weight", "blk.0.attn_gate.weight", "blk.0.attn_output.weight",
                                   "blk.1.attn_output_a.weight", "blk.1.attn_q.weight", "blk.2.attn_q.weight",
                                   "blk.3.attn_q.weight" };
    std::mt19937 rng(20261021);
    std::normal_distribution<float> normal(0.0f, 1.0f);
    std::vector<std::vector<uint8_t>> q(n_mats, std::vector<uint8_t>(nbytes));
    for (auto & m : q) {
        std::vector<float> f(size_t(K) * M);
        for (float & v : f) {
            v = normal(rng);
        }
        ggml_quantize_chunk(GGML_TYPE_Q8_0, f.data(), m.data(), 0, M, K, nullptr);
    }
    // the data file, as a GGUF holds the weights: each matrix 32-byte aligned, none on a page boundary
    temp_file data("data");
    std::vector<uint64_t> offset(n_mats);
    uint64_t cursor = base;
    for (int i = 0; i < n_mats; ++i) {
        offset[i] = cursor;
        data.write(q[i].data(), nbytes, cursor);
        cursor += (nbytes + 32 + 31) / 32 * 32;
    }
    require(fsync(data.fd) == 0, "fsync failed");
    // A separate invalid alias must be rejected before allocating a transport or starting a graph.
    temp_file short_file("short");
    short_file.write(q[n_drive].data(), block_bytes / 2, 0);
    // The valid index holds every drive matrix and the GPU/plain-CPU resident weights too.
    temp_file alias1("alias1"), alias2("alias2");
    auto rows = [&](int i, const std::string & src, uint64_t at) {
        std::string s;
        for (int b = 0; b < n_blocks; ++b) {
            s += std::string(names[i]) + "\t" + std::to_string(b) + "\t" + std::to_string(block_bytes) + "\t" + src +
                 "\t" + std::to_string(at + uint64_t(b) * block_bytes) + "\n";
        }
        return s;
    };
    const std::string header = "tensor\texpert_idx\tnbytes\tsource_path\toffset\n";
    std::string t1 = header, t2 = header;
    for (int i = 0; i < n_drive; ++i) {
        t1 += rows(i, data.path, offset[i]);
    }
    t1 += rows(n_drive, data.path, offset[n_drive]);
    t1 += rows(n_drive + 1, data.path, offset[n_drive + 1]);
    t1 += rows(n_drive + 2, data.path, offset[n_drive + 2]);
    t2 += rows(n_drive, short_file.path, 0);
    alias1.write(t1.data(), t1.size(), 0);
    alias2.write(t2.data(), t2.size(), 0);
    const auto invalid_settings = ssd::settings::parse("", "", {{"expert.source", "[\"" + alias2.path + "\"]"}});
    require(bool(invalid_settings), invalid_settings.error);
    bool refused = false;
    try { llama_ssd_context invalid(llama_ssd_options{invalid_settings.value, {}, {}}); }
    catch (const std::runtime_error &) { refused = true; }
    require(refused, "short source index was accepted at L3 startup");
    const auto settings = ssd::settings::parse("", "", {{"expert.source", "[\"" + alias1.path + "\"]"}});
    require(bool(settings), settings.error);
    temp_file telemetry("events");
    close(telemetry.fd); telemetry.fd = -1;
    require(unlink(telemetry.path.c_str()) == 0, "unlink empty event placeholder failed");
    const llama_ssd_options options{settings.value, {}, telemetry.path};
    // This is L3's actual owner/factory, with a real source_index, writer and
    // registry injection. It outlives both borrowed backend registrations.
    auto runtime = std::make_unique<llama_ssd_context>(options);
    using backend_owner = std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)>;
    backend_owner gpu_owned(ggml_backend_cuda_init(0), ggml_backend_free);
    backend_owner cpu_owned(ggml_backend_cpu_init(), ggml_backend_free);
    ggml_backend_t gpu = gpu_owned.get(), cpu = cpu_owned.get();
    require(gpu && cpu, "backend init failed");
    runtime->attach_backend(gpu);
    ssd::host::client_config budget;
    budget.unified = runtime->services.gpu->device_is_unified(runtime->services.gpu_device);
    require(bool(runtime->host->configure(budget)), "host budget configuration failed");
    const auto * transport = runtime->services.gpu->transport;
    void * transport_owner = runtime->services.gpu_transport;

    // the drive homes: the mapping of the data file, a CPU buffer of mapped weights (as llama's loader makes it)
    const size_t map_bytes = size_t(cursor);
    void * map = mmap(nullptr, map_bytes, PROT_READ, MAP_SHARED, data.fd, 0);
    require(map != MAP_FAILED, "mmap failed");
    ggml_backend_buffer_t mapped = ggml_backend_cpu_buffer_from_ptr(map, map_bytes);
    ggml_backend_buffer_set_usage(mapped, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    ggml_init_params wparams = { ggml_tensor_overhead() * 16, nullptr, true };
    ggml_context * wctx = ggml_init(wparams);
    std::vector<ggml_tensor *> home(n_drive + 1);
    for (int i = 0; i <= n_drive; ++i) {
        home[i] = ggml_new_tensor_2d(wctx, GGML_TYPE_Q8_0, K, M);
        ggml_set_name(home[i], names[i]);
        require(ggml_backend_tensor_alloc(mapped, home[i], (char *) map + offset[i]) == GGML_STATUS_SUCCESS, "alloc");
    }
    // the resident weight, and the same weights in device memory for the reference
    ggml_context * gctx = ggml_init(wparams);
    std::vector<ggml_tensor *> dev(n_drive + 2);
    for (int i = 0; i < n_drive + 2; ++i) {
        dev[i] = ggml_new_tensor_2d(gctx, GGML_TYPE_Q8_0, K, M);
        ggml_set_name(dev[i], names[i]);
    }
    ggml_backend_buffer_t dbuf = ggml_backend_alloc_ctx_tensors(gctx, gpu);
    ggml_backend_buffer_set_usage(dbuf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    for (int i = 0; i < n_drive + 2; ++i) {
        ggml_backend_tensor_set(dev[i], q[i].data(), 0, nbytes);
    }
    // a weight in a plain CPU buffer (not the mapping) that the index holds: the rule does not take it
    ggml_context * cctx = ggml_init(wparams);
    ggml_tensor * ram = ggml_new_tensor_2d(cctx, GGML_TYPE_Q8_0, K, M);
    ggml_set_name(ram, names[n_drive + 2]);
    ggml_backend_buffer_t cbuf = ggml_backend_alloc_ctx_tensors(cctx, cpu);
    ggml_backend_buffer_set_usage(cbuf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    ggml_backend_tensor_set(ram, q[n_drive + 2].data(), 0, nbytes);

    // what the load receipt asks: the index holds the four and 'bad' in blocks; then the stage for the largest
    for (int i = 0; i <= n_drive; ++i) {
        size_t bb = 0;
        require(transport->dense_has(transport_owner, names[i], nbytes, &bb) && bb == block_bytes,
                std::string("the index does not hold ") + names[i] + " in its blocks");
    }
    require(!transport->dense_has(transport_owner, names[0], nbytes + row_bytes, nullptr), "a size the index lacks was held");
    size_t stage = 0;
    require(transport->dense_prepare(transport_owner, &stage) && stage >= 2 * n_blocks * block_bytes, "stage not prepared");

    ggml_backend_t backends[2] = { gpu, cpu };
    using sched_owner = std::unique_ptr<ggml_backend_sched, decltype(&ggml_backend_sched_free)>;
    sched_owner scheduler(ggml_backend_sched_new(backends, nullptr, 2, 2048, false, true), ggml_backend_sched_free);
    ggml_backend_sched_t sched = scheduler.get();
    require(sched, "scheduler creation failed");
    runtime->stream.attach(sched);
    for (int n_tokens : { 1, 64 }) {
        std::vector<float> x(size_t(K) * n_tokens);
        for (float & v : x) {
            v = normal(rng);
        }
        const std::vector<ggml_tensor *> ref_w = { dev[0], dev[1], dev[2], dev[3] };
        const graph_out ref = run_graph(sched, ref_w, dev[n_drive], dev[n_drive + 1], nullptr, n_tokens, x);
        for (int pass = 0; pass < 2; ++pass) {
            ggml_backend_sched_kimi_copy_stats before = {};
            ggml_backend_sched_kimi_copy_stats_get(sched, &before);
            const std::vector<ggml_tensor *> drive_w = { home[0], home[1], home[2], home[3] };
            const graph_out got = run_graph(sched, drive_w, home[n_drive], dev[n_drive + 1], ram, n_tokens, x);
            const copy_delta d = copies_since(sched, before);
            require(std::memcmp(got.out.data(), ref.out.data(), sizeof(float) * got.out.size()) == 0,
                    "the drive homes' product differs from the device weights' at " + std::to_string(n_tokens) +
                    " tokens");
            // All five mapped matrices use transport; every one after the first was read ahead.
            // Indexed device-resident and ordinary CPU weights are never drive copies.
            require(d.tensors == n_drive + 1 && d.bytes == (n_drive + 1) * nbytes,
                    "copies " + std::to_string(d.tensors) + " of " + std::to_string(d.bytes) + " bytes");
            require(d.fallbacks == 0, "fallbacks " + std::to_string(d.fallbacks));
            require(d.readahead == n_drive, "read ahead " + std::to_string(d.readahead));
            // one copy target at a time: at one token each drive home's copy is a GPU split of its own (the reshape of
            // w3 too, which has no buffer when the graph is split), then the CPU's split of 'ram'
            require(n_tokens > 1 || got.splits == n_drive + 2, "splits " + std::to_string(got.splits));
            std::printf("tokens=%d pass=%d copies=%llu readahead=%llu fallbacks=%llu splits=%d\n", n_tokens, pass,
                        (unsigned long long) d.tensors, (unsigned long long) d.readahead,
                        (unsigned long long) d.fallbacks, got.splits);
        }
    }
    // The index was valid at startup. Shrink its payload later: completion is
    // short/EOF, and a failed graph must not read the mapped source as fallback.
    require(ftruncate(data.fd, off_t(offset[0] + block_bytes / 2)) == 0, "truncate source failed");
    std::vector<float> failed_x(K, 1.0f);
    const std::vector<ggml_tensor *> failed_w = {home[0], home[1], home[2], home[3]};
    const auto failed = run_graph(sched, failed_w, home[n_drive], dev[n_drive + 1], ram, 1, failed_x, true);
    require(failed.status != GGML_STATUS_SUCCESS, "short read did not reject the graph");
    ggml_backend_sched_kimi_copy_stats failed_stats = {};
    ggml_backend_sched_kimi_copy_stats_get(sched, &failed_stats);
    require(failed_stats.dense_fallbacks == 0, "failed source used mmap fallback");
    scheduler.reset();
    ggml_backend_buffer_free(cbuf);
    ggml_backend_buffer_free(dbuf);
    ggml_backend_buffer_free(mapped);
    ggml_free(cctx);
    ggml_free(gctx);
    ggml_free(wctx);
    munmap(map, map_bytes);
    cpu_owned.reset();
    gpu_owned.reset(); // backend quiesces transport before destroying its CUDA streams
    runtime.reset(); // closes source/writer only after borrowed users are gone
    std::ifstream events(telemetry.path);
    const std::string published{std::istreambuf_iterator<char>(events), {}};
    require(published.find("\"ev\":\"ssd.stream.dense\"") != std::string::npos,
        "real context writer did not receive the dense copy events");
    std::printf("SSD_DENSE_STREAM_OK stage=%zu\n", stage);
}

} // namespace

int main() {
    try {
        run();
        return 0;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "SSD_DENSE_STREAM_FAILED: %s\n", e.what());
        return 1;
    }
}

#else
int main() {
    return 77;
}
#endif
