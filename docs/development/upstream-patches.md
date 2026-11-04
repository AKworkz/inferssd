# Upstream patches

InferSSD bundles a llama.cpp snapshot under `engine/`. The unmodified snapshot
is commit `8884f8314ce8ea96d62cd602b6bbf7fc25758a09` in the history of `main`. Its attribution and license notices are listed in
[NOTICE](../../NOTICE). The InferSSD modules are described in
[How it works](../architecture.md). Relocating the snapshot under `engine/`
preserves its source attribution and license files.

| Integration group | Reason and boundary |
| --- | --- |
| ggml scheduler and allocator | Borrowed SSD copy hooks; graph buffer measurement/admission; first touch; backend scratch reservation; a pass that keeps an in-place op with the backend holding its storage. No core implementation is linked into ggml-base. |
| CPU/CUDA registries | Normal private provider discovery and context-service attachment. CPU and CUDA do not link each other. |
| Backend operation files | New model operations and narrow native kernel entry points stay with their upstream operation/architecture files. Strict MXFP4 math is compiled separately with contraction and FTZ disabled. |
| llama context | Runtime assembly, request phases, resource-aware graph reservation and failure recovery. Source, cache and telemetry owners reside in `src/ssd`. |
| Model loading/mapping | Deferred expert ranges and compact Metal allocations where a GGUF interleaves the dense tensors with experts held elsewhere; fixed complete-page advice. Source/range helpers reside in `src/ssd`. |
| Model/architecture support | DeepSeek V4.1, Engram, Muse and Qwen3.8-Flash-Next (qwen4exp) architecture, metadata, graph and KV operations follow upstream model/op organization. They are intentional model exceptions. |
| Common/server | CLI dispatch, rendering/parser hooks, prompt-cache/checkpoint and lifecycle calls. Owned implementations reside in `common/inferssd` and `tools/server/inferssd`. |
| State serialization | Virtual/C API entry points call positional/range codecs under `src/inferssd`. |

The inventory below lists every upstream file that differs from the snapshot.
It includes model and operator additions as well as integration hooks; review
those two kinds of change separately when assessing patch growth.

Regenerate the inventory with `python3 scripts/inferssd/patch-surface.py` and
replace the block between the inventory markers. The script compares each file
of snapshot commit `8884f83` (override with `--base`) in the `ggml`, `src`, `include`, `common` and `tools/server` trees with the
matching file under `engine/`. Relocation into `engine/`, files added by
InferSSD and upstream documentation, which is not bundled, are not counted.

<!-- BEGIN UPSTREAM INVENTORY -->
| Upstream path (under engine/) | Added | Removed |
| --- | ---: | ---: |
| `common/CMakeLists.txt` | 16 | 22 |
| `common/arg.cpp` | 61 | 3 |
| `common/build-info.cpp.in` | 13 | 2 |
| `common/build-info.h` | 2 | 0 |
| `common/chat-auto-parser-generator.cpp` | 1 | 1 |
| `common/chat-peg-parser.cpp` | 4 | 2 |
| `common/chat-peg-parser.h` | 3 | 0 |
| `common/chat.cpp` | 175 | 199 |
| `common/chat.h` | 34 | 3 |
| `common/common.cpp` | 3 | 1 |
| `common/common.h` | 13 | 2 |
| `common/debug.cpp` | 16 | 13 |
| `common/download.cpp` | 2 | 2 |
| `common/hf-cache.cpp` | 1 | 1 |
| `common/jinja/caps.cpp` | 1 | 1 |
| `common/jinja/parser.cpp` | 10 | 2 |
| `common/jinja/runtime.cpp` | 6 | 1 |
| `common/jinja/string.cpp` | 16 | 5 |
| `common/jinja/string.h` | 5 | 6 |
| `common/jinja/value.cpp` | 170 | 133 |
| `common/jinja/value.h` | 17 | 18 |
| `common/peg-parser.cpp` | 16 | 0 |
| `common/peg-parser.h` | 4 | 0 |
| `common/reasoning-budget.cpp` | 0 | 264 |
| `common/reasoning-budget.h` | 1 | 53 |
| `common/sampling.cpp` | 78 | 12 |
| `common/sampling.h` | 11 | 2 |
| `common/speculative.cpp` | 23 | 0 |
| `common/speculative.h` | 6 | 0 |
| `ggml/CMakeLists.txt` | 1 | 0 |
| `ggml/include/ggml-alloc.h` | 4 | 0 |
| `ggml/include/ggml-backend.h` | 58 | 0 |
| `ggml/include/ggml-cuda.h` | 34 | 0 |
| `ggml/include/ggml-rpc.h` | 2 | 2 |
| `ggml/include/ggml.h` | 158 | 1 |
| `ggml/src/CMakeLists.txt` | 2 | 0 |
| `ggml/src/ggml-alloc.c` | 38 | 2 |
| `ggml/src/ggml-backend-impl.h` | 5 | 0 |
| `ggml/src/ggml-backend-meta.cpp` | 29 | 1 |
| `ggml/src/ggml-backend.cpp` | 413 | 64 |
| `ggml/src/ggml-blas/ggml-blas.cpp` | 1 | 0 |
| `ggml/src/ggml-cann/ggml-cann.cpp` | 1 | 0 |
| `ggml/src/ggml-common.h` | 1 | 0 |
| `ggml/src/ggml-cpu/CMakeLists.txt` | 8 | 0 |
| `ggml/src/ggml-cpu/arch/arm/quants.c` | 0 | 1 |
| `ggml/src/ggml-cpu/arch/x86/quants.c` | 138 | 7 |
| `ggml/src/ggml-cpu/ggml-cpu.c` | 3708 | 3650 |
| `ggml/src/ggml-cpu/ggml-cpu.cpp` | 144 | 6 |
| `ggml/src/ggml-cpu/ops.cpp` | 751 | 12 |
| `ggml/src/ggml-cpu/ops.h` | 6 | 0 |
| `ggml/src/ggml-cpu/quants.c` | 1 | 0 |
| `ggml/src/ggml-cpu/quants.h` | 7 | 0 |
| `ggml/src/ggml-cpu/unary-ops.cpp` | 271 | 1 |
| `ggml/src/ggml-cpu/unary-ops.h` | 3 | 0 |
| `ggml/src/ggml-cuda/CMakeLists.txt` | 9 | 0 |
| `ggml/src/ggml-cuda/argsort.cu` | 93 | 71 |
| `ggml/src/ggml-cuda/argsort.cuh` | 6 | 0 |
| `ggml/src/ggml-cuda/binbcast.cu` | 12 | 2 |
| `ggml/src/ggml-cuda/common.cuh` | 163 | 8 |
| `ggml/src/ggml-cuda/concat.cu` | 69 | 46 |
| `ggml/src/ggml-cuda/cpy.cu` | 1 | 1 |
| `ggml/src/ggml-cuda/fattn-common.cuh` | 35 | 7 |
| `ggml/src/ggml-cuda/fattn.cu` | 10 | 0 |
| `ggml/src/ggml-cuda/fattn.cuh` | 3 | 0 |
| `ggml/src/ggml-cuda/ggml-cuda.cu` | 1498 | 132 |
| `ggml/src/ggml-cuda/mmq.cu` | 54 | 3 |
| `ggml/src/ggml-cuda/mmq.cuh` | 28 | 10 |
| `ggml/src/ggml-cuda/mmvf.cu` | 22 | 0 |
| `ggml/src/ggml-cuda/mmvq.cu` | 258 | 40 |
| `ggml/src/ggml-cuda/mmvq.cuh` | 12 | 0 |
| `ggml/src/ggml-cuda/norm.cu` | 15 | 0 |
| `ggml/src/ggml-cuda/quantize.cu` | 238 | 21 |
| `ggml/src/ggml-cuda/quantize.cuh` | 18 | 0 |
| `ggml/src/ggml-cuda/rope.cu` | 93 | 59 |
| `ggml/src/ggml-cuda/set-rows.cu` | 65 | 4 |
| `ggml/src/ggml-cuda/ssm-conv.cu` | 11 | 3 |
| `ggml/src/ggml-cuda/top-k.cu` | 250 | 0 |
| `ggml/src/ggml-cuda/top-k.cuh` | 9 | 0 |
| `ggml/src/ggml-cuda/unary.cu` | 377 | 0 |
| `ggml/src/ggml-cuda/unary.cuh` | 16 | 0 |
| `ggml/src/ggml-cuda/vecdotq.cuh` | 2 | 1 |
| `ggml/src/ggml-hexagon/ggml-hexagon.cpp` | 1 | 0 |
| `ggml/src/ggml-metal/CMakeLists.txt` | 2 | 0 |
| `ggml/src/ggml-metal/ggml-metal-device.cpp` | 12 | 0 |
| `ggml/src/ggml-metal/ggml-metal-device.h` | 1 | 0 |
| `ggml/src/ggml-metal/ggml-metal-device.m` | 8 | 1 |
| `ggml/src/ggml-metal/ggml-metal-impl.h` | 47 | 0 |
| `ggml/src/ggml-metal/ggml-metal-ops.cpp` | 136 | 0 |
| `ggml/src/ggml-metal/ggml-metal-ops.h` | 3 | 0 |
| `ggml/src/ggml-metal/ggml-metal.cpp` | 5 | 0 |
| `ggml/src/ggml-metal/ggml-metal.metal` | 133 | 0 |
| `ggml/src/ggml-opencl/ggml-opencl.cpp` | 1 | 0 |
| `ggml/src/ggml-openvino/ggml-openvino.cpp` | 1 | 0 |
| `ggml/src/ggml-quants.c` | 5 | 0 |
| `ggml/src/ggml-rpc/ggml-rpc.cpp` | 1 | 0 |
| `ggml/src/ggml-sycl/ggml-sycl.cpp` | 1 | 0 |
| `ggml/src/ggml-virtgpu/ggml-backend.cpp` | 1 | 0 |
| `ggml/src/ggml-vulkan/CMakeLists.txt` | 2 | 0 |
| `ggml/src/ggml-vulkan/ggml-vulkan.cpp` | 119 | 3 |
| `ggml/src/ggml-vulkan/vulkan-shaders/vulkan-shaders-gen.cpp` | 4 | 0 |
| `ggml/src/ggml-webgpu/ggml-webgpu.cpp` | 1 | 0 |
| `ggml/src/ggml-zdnn/ggml-zdnn.cpp` | 1 | 0 |
| `ggml/src/ggml-zendnn/ggml-zendnn.cpp` | 1 | 0 |
| `ggml/src/ggml.c` | 374 | 8 |
| `ggml/src/gguf.cpp` | 7 | 0 |
| `include/llama.h` | 88 | 1 |
| `src/CMakeLists.txt` | 20 | 0 |
| `src/llama-arch.cpp` | 168 | 2 |
| `src/llama-arch.h` | 109 | 2 |
| `src/llama-context.cpp` | 1316 | 45 |
| `src/llama-context.h` | 148 | 2 |
| `src/llama-cparams.h` | 9 | 0 |
| `src/llama-ext.h` | 16 | 0 |
| `src/llama-graph.cpp` | 760 | 50 |
| `src/llama-graph.h` | 158 | 5 |
| `src/llama-hparams.cpp` | 21 | 0 |
| `src/llama-hparams.h` | 73 | 1 |
| `src/llama-impl.h` | 35 | 0 |
| `src/llama-kv-cache-iswa.cpp` | 41 | 3 |
| `src/llama-kv-cache-iswa.h` | 28 | 0 |
| `src/llama-kv-cache.cpp` | 327 | 132 |
| `src/llama-kv-cache.h` | 70 | 4 |
| `src/llama-kv-cells.h` | 140 | 25 |
| `src/llama-memory-recurrent.cpp` | 41 | 1 |
| `src/llama-memory-recurrent.h` | 2 | 0 |
| `src/llama-memory.h` | 70 | 0 |
| `src/llama-mmap.cpp` | 132 | 3 |
| `src/llama-mmap.h` | 7 | 0 |
| `src/llama-model-loader.cpp` | 164 | 7 |
| `src/llama-model-loader.h` | 39 | 0 |
| `src/llama-model-saver.cpp` | 9 | 2 |
| `src/llama-model.cpp` | 1077 | 39 |
| `src/llama-model.h` | 107 | 0 |
| `src/llama-quant.cpp` | 22 | 1 |
| `src/llama-sampler.cpp` | 1 | 0 |
| `src/llama-vocab.cpp` | 67 | 8 |
| `src/llama-vocab.h` | 5 | 0 |
| `src/llama.cpp` | 4 | 1 |
| `src/models/deepseek2.cpp` | 7 | 2 |
| `src/models/kimi-linear.cpp` | 2 | 2 |
| `src/models/models.h` | 165 | 0 |
| `src/models/qwen35.cpp` | 2 | 2 |
| `src/models/qwen35moe.cpp` | 8 | 2 |
| `src/models/qwen3next.cpp` | 2 | 2 |
| `tools/server/CMakeLists.txt` | 13 | 5 |
| `tools/server/server-chat.cpp` | 66 | 16 |
| `tools/server/server-common.cpp` | 288 | 5 |
| `tools/server/server-common.h` | 22 | 1 |
| `tools/server/server-context.cpp` | 685 | 127 |
| `tools/server/server-task.cpp` | 195 | 183 |
| `tools/server/server-task.h` | 23 | 80 |
| `tools/server/tests/utils.py` | 2 | 2 |

Total: 17480 added, 5678 removed across 152 upstream paths.
<!-- END UPSTREAM INVENTORY -->
