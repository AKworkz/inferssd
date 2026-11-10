// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "ggml-cuda.h"
#include <cuda_runtime_api.h>

// Private normal-link boundaries inside ggml-cuda. Definitions are backend
// kernels/services in this same library, never runtime-discovered DSOs.
extern "C" {
bool ggml_cuda_moe_stream_mmvf_dev(int, const void *, int64_t, int64_t, size_t,
    const float *, float *, int64_t, cudaStream_t);
bool ggml_cuda_moe_stream_mmvq_quantize_dev(int, const float *, int64_t, int64_t, void *, cudaStream_t);
bool ggml_cuda_moe_stream_mmvq_q8_dev(int, const void *, int64_t, int64_t, size_t, const void *, float *, cudaStream_t);
bool ggml_cuda_moe_stream_mmvq_supports(int);
bool ggml_cuda_moe_stream_mm_supports(int);
void * ggml_cuda_moe_stream_mm_create(int);
void ggml_cuda_moe_stream_mm_destroy(void *);
void ggml_cuda_moe_stream_mm_release(void *);
bool ggml_cuda_moe_stream_mm_reserve(void *, int, int64_t, int64_t, int64_t);
size_t ggml_cuda_moe_stream_mm_operand_bytes(void *, int, int64_t, int64_t);
bool ggml_cuda_moe_stream_mm_quantize(void *, int, const float *, const int32_t *, int64_t, int64_t, int64_t, void *, cudaStream_t);
bool ggml_cuda_moe_stream_mm(void *, int, const void *, int64_t, int64_t, const void *, int64_t, float *, cudaStream_t);
size_t ggml_cuda_moe_stream_mm_device_bytes(void *);
// cudaMalloc for device memory outside ggml's buffers, admitted against the VRAM reserve and retried after the OOM
// reclaimer gives memory back, as a backend buffer's allocation is. Freed with cudaFree.
cudaError_t ggml_cuda_malloc_reclaiming(void ** ptr, size_t size, int device);
}
