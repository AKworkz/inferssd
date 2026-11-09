# Changelog

## 0.1.4 - 2026-11-10

This release adds Qwen3.8 Flash Next and Gemma 4 E4B, and publishes checked Mac and Windows packages together from the same tagged commit.

### Highlights

- **Precompiled packages in one release:** Apple silicon macOS and Windows x64 packages, manifests and version-pinned installers are built and checked before either platform is published. Installation needs no Git, Python, compiler or SDK. Linux retains its source installer ([#162](https://github.com/wici-ai/inferssd/pull/162)).
- **Qwen3.8 Flash Next:** the tested IQ4_XS checkpoint uses SSD expert offload through the shared launch flow ([#157](https://github.com/wici-ai/inferssd/pull/157)).
- **Gemma 4 E4B:** the tested QAT Q4_0 checkpoint uses the ordinary RAM inference path and is admitted only when its estimated memory fits the available RAM and host reserve ([#157](https://github.com/wici-ai/inferssd/pull/157)).
- **Windows package:** includes its own Python, native CPU/Vulkan engine, probes, Vulkan loader and runtime DLLs. It requires Windows 11 x64 and AVX2; GPU execution uses the normal graphics driver. macOS requires Apple silicon and macOS 15 or later ([#162](https://github.com/wici-ai/inferssd/pull/162)).
- **Explicit model capabilities:** `inferssd --version --json` reports its supported model profiles, so clients can check the capabilities they need across branch and release builds ([#157](https://github.com/wici-ai/inferssd/pull/157)).

### Install or upgrade

On macOS or Linux:

```sh
curl -fsSL https://github.com/wici-ai/inferssd/releases/download/v0.1.4/install.sh | sh
```

On Windows, in PowerShell:

```powershell
irm https://github.com/wici-ai/inferssd/releases/download/v0.1.4/install.ps1 | iex
```

Run the same command to upgrade. Downloaded models are preserved.

## 0.1.3 - 2026-11-06

This release decodes faster where experts run on the CPU or a CUDA GPU, puts Apple silicon's GPU to work on prompts, and lets a client turn thinking off or choose how much the model thinks, as the model's own chat template allows.

### Highlights

- **Faster decode:** on DGX Spark, DeepSeek V4.1 Flash decodes at about 7.8 tok/s. CPU threads poll instead of sleeping between phases, a phase runs every expert ready for it, a decode layer submits its own reads, and the CUDA expert tier keeps models that mix expert formats ([#129](https://github.com/wici-ai/inferssd/pull/129), [#137](https://github.com/wici-ai/inferssd/pull/137), [#139](https://github.com/wici-ai/inferssd/pull/139), [#140](https://github.com/wici-ai/inferssd/pull/140), [#141](https://github.com/wici-ai/inferssd/pull/141), [#143](https://github.com/wici-ai/inferssd/pull/143)).
- **Prompts on Apple silicon:** the GPU now computes routed experts during prompt processing, over Metal and over Vulkan on integrated GPUs. On an M5 Max, DeepSeek V4.1 prefills a 4,104-token prompt at about 35 tok/s. Where the GPU takes part, a reply can differ between runs of one prompt, because the CPU and the GPU compute an expert slightly differently ([#132](https://github.com/wici-ai/inferssd/pull/132)).
- **Thinking controls:** `/props` reports what the loaded model's chat template offers for thinking: whether it turns thinking off, and its own `reasoning_effort` values. A request may use exactly those. DeepSeek V4.1 takes `none`, `low`, `medium`, `high` (default) and `max`; Muse always thinks and takes `low`, `medium`, `high` and `xhigh` (default); Qwen3.6 takes `none` ([#142](https://github.com/wici-ai/inferssd/pull/142)).

### Changes

- **The models' own chat templates:** DeepSeek V4.1 and Muse now run the chat templates their makers publish. DeepSeek's prompts match its official encoder, tool definitions and images included ([#142](https://github.com/wici-ai/inferssd/pull/142)).
- **`reasoning_effort`:** a value the model's template does not offer gets a 400 that names the ones it does. DeepSeek no longer takes `minimal` or `xhigh`, and Muse can no longer turn thinking off. `inferssd cli` uses the least thinking a model offers ([#142](https://github.com/wici-ai/inferssd/pull/142)).
- **GPU backends by hardware:** on Linux the installer builds for the machine's GPUs: CUDA for NVIDIA, Vulkan for AMD and Intel when `glslc` is installed, and the CPU otherwise. It no longer refuses a machine without `nvcc`. A Linux machine with an AMD or Intel display controller profiles itself again on its first launch ([#130](https://github.com/wici-ai/inferssd/pull/130)).
- **CUDA expert tier:** a model whose experts mix quantization formats, MXFP4 included, keeps the tier on instead of running every expert on the CPU ([#137](https://github.com/wici-ai/inferssd/pull/137), [#139](https://github.com/wici-ai/inferssd/pull/139)).
- **No liburing:** CUDA builds no longer need it ([#138](https://github.com/wici-ai/inferssd/pull/138)).
- **`vram.budget` on every platform:** 0 keeps every expert on the CPU ([#132](https://github.com/wici-ai/inferssd/pull/132)).
- **Threads on macOS:** where threads cannot be pinned, compute leaves one CPU to the I/O thread ([#133](https://github.com/wici-ai/inferssd/pull/133)).

### Fixes

- **Memory pressure on shared-memory GPUs:** a run no longer stops when another process takes memory during decode on a machine whose GPU shares it, and a memory error says why ([#128](https://github.com/wici-ai/inferssd/pull/128)).
- **In-place ops:** an op that writes through a view runs on its storage's backend; on Vulkan it crashed ([#129](https://github.com/wici-ai/inferssd/pull/129)).

### Install or upgrade

```sh
curl -fsSL https://github.com/wici-ai/inferssd/releases/download/v0.1.3/install.sh | sh
```

Run the same command to upgrade from an earlier version. Downloaded models are not touched.

## 0.1.2 - 2026-11-04

This release adds Qwen3.6-35B-A3B, estimates each supported model's decode speed on your machine, and trims the command to `serve`, `cli` and `profile`.

### Highlights

- **Qwen3.6-35B-A3B:** Unsloth's UD-Q4_K_XL runs with its routed experts streamed from SSD. Start it with `inferssd serve -hf unsloth/Qwen3.6-35B-A3B-GGUF`; its one file is about 21 GiB ([#125](https://github.com/wici-ai/inferssd/pull/125)).
- **Decode speed estimates:** `inferssd profile --model <models directory>` estimates the decode speed of each supported model on this machine before you download it, and `--json` prints the profile and the estimates as one JSON object ([#122](https://github.com/wici-ai/inferssd/pull/122)).
- **`inferssd --version`** prints the installed version ([#122](https://github.com/wici-ai/inferssd/pull/122)).

### Changes

- **Three commands:** `inferssd` now has `serve`, `cli` and `profile`. The launch, settings, planning and calibration commands are gone; `serve` and `cli` do what the launch did ([#121](https://github.com/wici-ai/inferssd/pull/121)).
- **Measuring an empty models directory:** given a directory with no file of 1 GiB, `inferssd profile` measures its drive in a 4 GiB test file, which it removes ([#122](https://github.com/wici-ai/inferssd/pull/122)).
- **Busy GPUs:** a launch no longer refuses a GPU that other processes use; it keeps a 1024 MiB reserve for them, counted against this process's own memory ([#119](https://github.com/wici-ai/inferssd/pull/119)). Thanks to @xNoVoSx for the report and for testing the fix ([#108](https://github.com/wici-ai/inferssd/issues/108)).
- **Metal memory:** where a GGUF interleaves the dense tensors with the experts, as Muse's and Qwen3.6's do, Metal holds only the dense weights instead of pinning the whole file ([#125](https://github.com/wici-ai/inferssd/pull/125)).

### Fixes

- **macOS with a decimal comma:** `inferssd` no longer stops with `Darwin memory counters unavailable` under a locale such as German or French ([#115](https://github.com/wici-ai/inferssd/pull/115)).
- **Unsupported expert formats:** a GGUF whose routed experts use a quantization the engine does not run is refused with the tensor types named ([#116](https://github.com/wici-ai/inferssd/pull/116)). Thanks to @okayali1976-cell for the report ([#114](https://github.com/wici-ai/inferssd/issues/114)).
- **Finding the installed engine:** `cli`, `serve` and `profile` look for it in the same places ([#104](https://github.com/wici-ai/inferssd/pull/104)).
- **Apple GPU in the machine profile:** a Mac's profile now names its GPU. Mac profiles measure once more after the upgrade. Thanks to @HamedShams ([#102](https://github.com/wici-ai/inferssd/pull/102)).

### Install or upgrade

```sh
curl -fsSL https://github.com/wici-ai/inferssd/releases/download/v0.1.2/install.sh | sh
```

Run the same command to upgrade from v0.1.0 or v0.1.1. Downloaded models are not touched. `-hf` now accepts the three tested repositories in the [README](https://github.com/wici-ai/inferssd/blob/v0.1.2/README.md#models).

## 0.1.1 - 2026-10-31

This release fixes the Linux source install with recent CUDA toolkits.

### Fixes

- **Build with CCCL 3.4 and later:** the installer failed to compile with CCCL 3.4.0 and later, including the CCCL 3.4.3 bundled with CUDA 13.4.3 and CCCL 3.5.0 on Arch Linux. Thanks to @rokam for the report ([#95](https://github.com/wici-ai/inferssd/issues/95)).
- **`inferssd profile` after a one-command install:** the command now finds the installed probes instead of reporting them as missing. Thanks to @HamedShams for the report ([#97](https://github.com/wici-ai/inferssd/issues/97)).

### Install or upgrade

```sh
curl -fsSL https://github.com/wici-ai/inferssd/releases/download/v0.1.1/install.sh | sh
```

Run the same command to upgrade from v0.1.0. Downloaded models are not touched. The [known limitations of v0.1.0](https://github.com/wici-ai/inferssd/releases/tag/v0.1.0#known-limitations) still apply.

## 0.1.0 - 2026-10-30

First public release of InferSSD, which runs mixture-of-experts models larger than your RAM or VRAM from SSD. InferSSD builds on llama.cpp, reads GGUF weights directly, and serves browser chat and an OpenAI-compatible API.

### Highlights

- **Models:** DeepSeek-V4.1-Flash-1017 (Q8_0) and Muse-Spark-Open (UD-Q4_K_XL), with routed experts streamed from SSD through bounded RAM and VRAM caches. See the [benchmarks](https://github.com/wici-ai/inferssd/blob/v0.1.0/README.md#performance-benchmarks).
- **Platforms:** macOS on Apple Silicon with Metal, and Linux with NVIDIA CUDA on x86_64 and ARM64. Windows support is in progress.
- **One-command install and launch:** the installer sets up a private environment, and `inferssd cli` and `inferssd serve` download a tested model and start it.
- **Automatic setup:** the first launch measures the machine, and the engine sizes its caches from the memory available.

### Install

```sh
curl -fsSL https://github.com/wici-ai/inferssd/releases/download/v0.1.0/install.sh | sh
```

This release installs from source, so the engine is compiled for your machine's CPU and GPU. Prebuilt packages are not available yet. Building takes about 2 to 12 minutes, depending on the machine.

The installer needs Git, Python 3.9 or later, and a C/C++ compiler. macOS also needs the Xcode Command Line Tools; Linux also needs the CUDA toolkit, `liburing` headers, and systemd with cgroup v2.

Then start a model. The first run downloads about 470 GiB of model files.

```sh
# Chat in your terminal
inferssd cli -hf mrfakename/DeepSeek-V4.1-Flash-1017-GGUF

# Or start the web UI and an OpenAI-compatible API at http://127.0.0.1:8080
inferssd serve -hf unsloth/Muse-Spark-Open-816B-A42B-GGUF
```

### Known limitations

- `-hf` accepts only the two tested repositories above. Use `-m` for local GGUF files.
- If the CUDA runtime is outside the system library path, the first launch cannot measure the GPU, and prefill runs on the CPU.

### License

InferSSD is licensed under Apache-2.0. Bundled llama.cpp/ggml and other third-party components keep their original licenses and notices.
