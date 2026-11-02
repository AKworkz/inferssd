#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#ifdef  __cplusplus
extern "C" {
#endif

#ifdef GGML_USE_HIP
#define GGML_CUDA_NAME "ROCm"
#define GGML_CUBLAS_NAME "hipBLAS"
#elif defined(GGML_USE_MUSA)
#define GGML_CUDA_NAME "MUSA"
#define GGML_CUBLAS_NAME "muBLAS"
#else
#define GGML_CUDA_NAME "CUDA"
#define GGML_CUBLAS_NAME "cuBLAS"
#endif
#define GGML_CUDA_MAX_DEVICES       16

// backend API
GGML_BACKEND_API ggml_backend_t ggml_backend_cuda_init(int device);

GGML_BACKEND_API bool ggml_backend_is_cuda(ggml_backend_t backend);

// device buffer
GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_cuda_buffer_type(int device);

// conduct allreduce operation between devices
GGML_BACKEND_API bool ggml_backend_cuda_allreduce_tensor(ggml_backend_t * backends, struct ggml_tensor ** tensors, size_t n_backends);

// split tensor buffer that splits matrices by rows across multiple devices
GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_cuda_split_buffer_type(int main_device, const float * tensor_split);

// pinned host buffer for use with the CPU backend for faster copies between CPU and GPU
GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_cuda_host_buffer_type(void);

GGML_BACKEND_API int  ggml_backend_cuda_get_device_count(void);
GGML_BACKEND_API void ggml_backend_cuda_get_device_description(int device, char * description, size_t description_size);
// Free memory a new device allocation can get: on a unified-memory GPU, where device allocations come out of
// system memory and cudaMemGetInfo reports only MemFree, MemAvailable as it reads at the kernel's full
// watermark boost (MemAvailable keeps the boosted low watermarks back).
GGML_BACKEND_API void ggml_backend_cuda_get_device_memory(int device, size_t * free, size_t * total);
GGML_BACKEND_API bool ggml_backend_cuda_device_is_unified(int device);

GGML_BACKEND_API bool ggml_backend_cuda_register_host_buffer(void * buffer, size_t size);
GGML_BACKEND_API void ggml_backend_cuda_unregister_host_buffer(void * buffer);

// Optional phase-scoped recovery for direct backend-buffer cudaMalloc OOMs.
// A callback returns 1 after freeing one block, 0 when nothing is reclaimable,
// GGML_CUDA_RECLAIM_LATER when it cannot free now but will at its next quiescent
// point, and -1 on a fail-closed release error. Registration is exclusive per device.
#define GGML_CUDA_RECLAIM_LATER 2
typedef int (*ggml_cuda_oom_reclaimer_fn)(void * userdata, size_t requested_bytes, size_t * freed_bytes);
GGML_BACKEND_API bool ggml_cuda_oom_reclaimer_register(
        int device, ggml_cuda_oom_reclaimer_fn callback, void * userdata);
GGML_BACKEND_API bool ggml_cuda_oom_reclaimer_unregister(
        int device, ggml_cuda_oom_reclaimer_fn callback, void * userdata);
// Memory that device allocations must leave free, set at run time by a cache that can give memory back
// through the OOM reclaimer (on unified memory, the host's own working set).
GGML_BACKEND_API void ggml_cuda_set_memory_reserve(int device, size_t bytes);
// A context's VRAM reserve on a discrete GPU is kept against what this process holds (ssd/reserve-budget.h). Device
// memory allocated outside ggml's buffers reports what it takes with hold and release. anchor starts from what the card
// offers now. free_memory is the device's free memory as a cache growing into it should read it, and takes in what
// other processes released, so call it only where this process is not allocating or freeing.
GGML_BACKEND_API void ggml_cuda_reserve_hold(int device, size_t bytes);
GGML_BACKEND_API void ggml_cuda_reserve_release(int device, size_t bytes);
GGML_BACKEND_API void ggml_cuda_reserve_anchor(int device);
GGML_BACKEND_API void ggml_cuda_reserve_free_memory(int device, size_t * free, size_t * total);
GGML_BACKEND_API void ggml_cuda_reserve_state(int device, size_t * held, size_t * reachable);
// Pinned host memory about to be allocated for the current device: on a unified-memory GPU it comes out of the same
// DRAM, so the reserve is made first (the OOM reclaimer gives memory back). False when no room could be made; the
// caller may allocate anyway. True on a discrete GPU.
GGML_BACKEND_API bool ggml_cuda_admit_host_alloc(size_t size);
// Deterministic test hook; the next count backend-buffer allocations report
// OOM before cudaMalloc and exercise the same reclaim-and-retry loop.
GGML_BACKEND_API bool ggml_cuda_oom_reclaimer_inject_failures(int device, uint64_t count);

GGML_BACKEND_API ggml_backend_reg_t ggml_backend_cuda_reg(void);

#ifdef  __cplusplus
}
#endif
