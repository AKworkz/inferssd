#include "ggml-backend.h"
#include "ggml-cuda.h"
#include "../../ggml/src/ggml-ssd/host/budget-linux.h"
#include <cuda_runtime_api.h>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <unistd.h>

static constexpr size_t MIB = 1024 * 1024;
static constexpr size_t RESERVE = 576 * MIB;

struct cache_fixture {
    void * ptr = nullptr;
    size_t bytes = 0;
    int calls = 0;
};

static void check(bool ok, const char * message) {
    if (!ok) {
        std::fprintf(stderr, "%s\n", message);
        std::exit(1);
    }
}

static size_t free_memory() {
    size_t available = 0, total = 0;
    check(cudaMemGetInfo(&available, &total) == cudaSuccess, "memory query failed");
    return available;
}

static int reclaim(void * userdata, size_t, size_t * freed) {
    auto * cache = static_cast<cache_fixture *>(userdata);
    ++cache->calls;
    *freed = 0;
    if (!cache->ptr) {
        return 0;
    }
    check(cudaFree(cache->ptr) == cudaSuccess, "cache release failed");
    cache->ptr = nullptr;
    *freed = cache->bytes;
    return 1;
}

static std::string slurp(const char * path) {
    std::ifstream file(path);
    std::stringstream text;
    text << file.rdbuf();
    return text.str();
}

// A unified-memory GPU (DGX Spark): ggml's free device memory is MemAvailable as it reads at the kernel's full
// watermark boost. Nothing large is allocated here: on unified memory the discrete test below would fill the DRAM.
static int unified_tests() {
    const std::string meminfo = slurp("/proc/meminfo");
    uint64_t available = 0;
    check(kimi_memory_boost::meminfo_field(meminfo, "MemAvailable", available), "MemAvailable unreadable");
    const uint64_t headroom = kimi_memory_boost::headroom(meminfo, slurp("/proc/zoneinfo"),
        slurp("/proc/sys/vm/watermark_boost_factor"), uint64_t(sysconf(_SC_PAGESIZE)));
    size_t free = 0, total = 0;
    ggml_backend_cuda_get_device_memory(0, &free, &total);
    const uint64_t expected = available - headroom, slack = 64 * MIB; // two reads of a moving MemAvailable
    std::printf("unified: MemAvailable %.0f MiB, boost headroom %.0f MiB, ggml free %.0f MiB\n",
        available / double(MIB), headroom / double(MIB), free / double(MIB));
    check(free + slack >= expected && free <= expected + slack, "ggml's free memory is not MemAvailable at the full boost");
    std::puts("CUDA unified free memory at the full watermark boost: PASS");

    // Pinned host memory comes out of the same DRAM: with a reserve that leaves 8 MiB, a 256 MiB pinned buffer makes
    // its room first through the OOM reclaimer (a stand-in cache holding 512 MiB), as a device allocation would, and
    // is still pinned. The reserve is published, not filled: nothing large stays allocated.
    cache_fixture cache;
    cache.bytes = 512 * MIB;
    check(cudaMalloc(&cache.ptr, cache.bytes) == cudaSuccess, "cache allocation failed");
    check(ggml_cuda_oom_reclaimer_register(0, reclaim, &cache), "register failed");
    ggml_backend_cuda_get_device_memory(0, &free, &total);
    check(free > 1024 * MIB, "too little free memory for the pinned test");
    ggml_cuda_set_memory_reserve(0, free - 8 * MIB);
    auto * host_buft = ggml_backend_cuda_host_buffer_type();
    auto * pinned = ggml_backend_buft_alloc_buffer(host_buft, 256 * MIB);
    ggml_cuda_set_memory_reserve(0, 0);
    check(pinned != nullptr && ggml_backend_buffer_get_type(pinned) == host_buft, "the pinned buffer was not allocated pinned");
    check(cache.calls == 1 && cache.ptr == nullptr, "pinned host memory did not make its room first");
    ggml_backend_buffer_free(pinned);
    // With the reserve cleared a small pinned allocation is admitted without the reclaimer.
    check(ggml_cuda_admit_host_alloc(1 * MIB) && cache.calls == 1, "a small pinned allocation found no room");
    check(ggml_cuda_oom_reclaimer_unregister(0, reclaim, &cache), "unregister failed");
    std::puts("CUDA unified pinned host memory admitted before allocation: PASS");
    return 0;
}

int main() {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
        std::puts("No CUDA device; skipping reserve test");
        return 77;
    }
    auto * buft = ggml_backend_cuda_buffer_type(0);
    check(buft != nullptr, "CUDA device required");
    if (ggml_backend_cuda_device_is_unified(0)) {
        return unified_tests();
    }
    const size_t available = free_memory();
    if (available <= RESERVE + 128 * MIB) {
        std::puts("Insufficient free memory; skipping reserve test");
        return 77;
    }

    ggml_cuda_set_memory_reserve(0, RESERVE);

    // This fits physically, but must fail without consuming the configured reserve.
    auto * rejected = ggml_backend_buft_alloc_buffer(buft, available - RESERVE + 64 * MIB);
    check(rejected == nullptr, "allocation consumed reserved memory");
    check(free_memory() >= RESERVE, "reserve lost after rejected allocation");

    // Simulate an elastic expert cache leaving only 8 MiB of allocatable room.
    cache_fixture cache;
    cache.bytes = free_memory() - RESERVE - 8 * MIB;
    check(cudaMalloc(&cache.ptr, cache.bytes) == cudaSuccess, "cache allocation failed");
    check(ggml_cuda_oom_reclaimer_register(0, reclaim, &cache), "register failed");
    auto * buffer = ggml_backend_buft_alloc_buffer(buft, 64 * MIB);
    check(buffer != nullptr, "allocation failed after reclaim");
    check(cache.calls == 1 && cache.ptr == nullptr, "cache was not reclaimed proactively");
    check(free_memory() >= RESERVE, "allocation consumed reserve after reclaim");
    ggml_backend_buffer_free(buffer);
    check(ggml_cuda_oom_reclaimer_unregister(0, reclaim, &cache), "unregister failed");
    ggml_cuda_set_memory_reserve(0, 0);
    std::puts("CUDA reserve rejection and proactive cache reclaim: PASS");
}
