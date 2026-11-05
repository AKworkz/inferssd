# Expert VRAM tier

This optional CUDA component caches routed expert weights and runs explicit
expert formats for DeepSeek and Muse. The `kimi-vram-tier` object target links
normally into `ggml-cuda`. No component DSO or preload is used.

The owner supplies tensor types, geometry, expert count and layer IDs through
`kimi_vram_format_api`; it supplies source file identities and ranges through
`kimi_vram_request`. The tier shares admission policy, asynchronous and deferred
uploads, residency leases, elastic blocks and failure cleanup across formats.
A borrowed upload keeps its source pinned until collect/finish or failure drain.
Reclaim refuses live compute/residency leases and drains queued copies before
freeing their blocks. The CPU sink owns the request and phase boundaries.

Every expert format, MXFP4 included, runs on the CUDA backend's own kernels:
decode on the generic format adapter (MMVQ), grouped prefill on the batched
adapter (MMQ, MMVQ or a dequantizing GEMM, as ggml chooses). A model may mix
formats across layers; slots take the largest expert. The results follow
ggml's CUDA backend, not the CPU sink bit for bit: on Blackwell, MMQ takes
MXFP4 with FP4 activations, as ggml's own `MUL_MAT` does. The GPU backend does
not link the CPU backend.

The former implicit Q3_K transport path has been removed. The internal
transport ABI remains version 2 with its existing field order; implicit
`prepare`/`launch` slots return an explicit unsupported-format error and drain
their context. Use the explicit format extension. Generic upstream Q3_K
quantizers and kernels remain part of ggml.

GPU lifecycle and numerical tests require an owned device and the matching
normal backend. They are built as explicit targets and are not automatically
run by CTest. License and origin notes for this directory are in
[THIRD_PARTY.md](THIRD_PARTY.md).
