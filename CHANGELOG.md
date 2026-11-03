# Changelog

## 0.1.2 - 2026-11-04

This release adds Qwen3.6-35B-A3B, estimates each supported model's decode speed on your machine, and trims the command to `serve`, `cli` and `profile`.

### Highlights

- **Qwen3.6-35B-A3B:** Unsloth's UD-Q4_K_XL runs with its routed experts streamed from SSD. Start it with `inferssd serve -hf unsloth/Qwen3.6-35B-A3B-GGUF`; its one file is about 21 GiB ([#125](https://github.com/wici-ai/inferssd/pull/125)).
- **Decode speed estimates:** `inferssd profile --model <models directory>` estimates the decode speed of each supported model on this machine before you download it, and `--json` prints the profile and the estimates as one JSON object ([#122](https://github.com/wici-ai/inferssd/pull/122)).
- **`inferssd --version`** prints the installed version ([#122](https://github.com/wici-ai/inferssd/pull/122)).

### Changes

- **Three commands:** `inferssd` now has `serve`, `cli` and `profile`. The launch, settings, planning and calibration commands are gone; `serve` and `cli` do what the launch did ([#121](https://github.com/wici-ai/inferssd/pull/121)).
- **Measuring an empty models directory:** given a directory with no file of 1 GiB, `inferssd profile` measures its drive in a 4 GiB test file, which it removes ([#122](https://github.com/wici-ai/inferssd/pull/122)).
- **Busy GPUs:** a launch no longer refuses a GPU that other processes use; it keeps a 1024 MiB reserve for them, counted against this process's own memory ([#119](https://github.com/wici-ai/inferssd/pull/119)).
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
- **Automatic setup:** the first launch measures the machine, and the engine sizes its caches from the memory available. There is no memory setting to choose.

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
