<!-- SPDX-License-Identifier: Apache-2.0 -->
# Quick start

[Back to README](../README.md)

Both platforms use the `main` branch of `wici-ai/inferssd`. Use the
[one-command installation](#one-command-installation), or build manually for
**[Apple Silicon](#apple-silicon)** / **[Linux / NVIDIA](#linux--nvidia)** and
[prepare a launch](#prepare-a-launch).

## Before you start

- Install Git, CMake, Ninja, a C/C++ compiler, and Python 3.9 or later. macOS also needs the Xcode Command Line Tools; Linux needs `liburing` development files, and a compatible CUDA toolkit on a machine with an NVIDIA GPU.
- Choose a [tested GGUF checkpoint](../README.md#models). `cli -hf` and `serve -hf` download all shards of its tested quantization. For manual `launch`, obtain a complete local copy and keep all shards together.
- Plan storage for the checkpoint. The benchmarked DeepSeek GGUF files take about **473 GiB**; an optional DeepSeek expert pack adds about **269 GiB**, and a Muse pack about **450 GiB**. The framework can index GGUF experts directly, so a pack is optional and must match its source checkpoint.
- Choose CPU, cache and process-memory budgets for your machine. The README benchmark settings describe their recorded runs, not universal deployment defaults.

## One-command installation

```sh
curl -fsSL https://github.com/wici-ai/inferssd/releases/latest/download/install.sh | sh
```

This is a source installer, not a prebuilt binary release. It checks for a C/C++
compiler, Git, Python 3.9+ with `venv`, and the platform toolchain. On macOS,
the Xcode Command Line Tools must be installed. On Linux, install `liburing`
headers. The engine is built for the machine's GPUs, as the
[machine profile](machine-profile.md) lists them: for an NVIDIA GPU with CUDA, which
needs `nvcc` on PATH, under `/usr/local/cuda`, or through `CUDACXX`; for an AMD or
Intel GPU with Vulkan, when the Vulkan SDK's `glslc` is on PATH; and for the CPU
alone otherwise.
The [Linux runtime requirements](#linux-requirements) still apply.

CMake, Ninja, Hugging Face support and the framework are installed into a private
virtual environment under `~/.local/share/inferssd/releases`. The installer
builds the server and machine probes, creates the engine descriptor, and installs
`~/.local/bin/inferssd`. If that directory is not on PATH, run the export printed
by the installer and add it to your shell profile. It does not edit shell profiles
or install system packages.

```sh
inferssd cli -hf mrfakename/DeepSeek-V4.1-Flash-1017-GGUF
inferssd serve -hf mrfakename/DeepSeek-V4.1-Flash-1017-GGUF

# Muse selects UD-Q4_K_XL automatically
inferssd cli -hf unsloth/Muse-Spark-Open-816B-A42B-GGUF

# One response, then stop the owned server
inferssd cli -hf mrfakename/DeepSeek-V4.1-Flash-1017-GGUF -p 'Explain MoE.' -n 256

# Existing local weights, including fully offline launches after installation
inferssd serve -m /path/on/ssd/model-00001-of-00010.gguf
```

`-hf` accepts these two README repositories and their tested quantizations only;
explicit `:Q8_0` and `:UD-Q4_K_XL` suffixes are accepted. It fetches repository
metadata, pins the resolved commit, checks the complete shard set and disk space,
then downloads just those GGUF files. It never downloads every quantization.
The first shard's local directory is printed; use `-m` with that path to avoid
contacting Hugging Face on subsequent launches. Model access uses the standard
Hugging Face token store or `HF_TOKEN`. `--revision` pins a model revision;
`--model-dir` chooses storage. Keep all shards together and do not edit them.

Downloads use the Hub client's local metadata and resume support. The disk check
conservatively reserves the full size of unfinished shards plus 1 GiB, so a
partially downloaded shard may require more free space than its remaining bytes.
The source alias index references the GGUF payloads without an extra expert pack.

Both commands use the owned runtime, machine measurement, model profile
selection and chat templates. They need no memory settings: the engine
uses the memory it finds available and keeps its own margin for the host. The
runtime enforces its Linux cgroup requirements.

Use `--context`, `--settings`, `--set`, `--machine` and `--reprofile` to override
launch choices. `--resources` selects a manually built engine descriptor. The
server listens on loopback port 8080 (`--port` changes it). Terminal chat keeps
conversation history; `/clear` starts a new conversation, `/exit` or EOF stops
the service. Ctrl+C also stops it, including during startup or generation.
Each launch keeps its receipts under `~/.local/share/inferssd/state/runs`.

`inferssd --version` prints the installed version. Rerun the installer to upgrade. Builds use a new release directory and publish
the command only after build and schema checks; a failed build keeps the previous
command. Older releases and model downloads are retained. For a branch or tag,
download the installer and set `INFERSSD_REF` when running it. Optional
`INFERSSD_HOME`, `INFERSSD_BIN_DIR`, and `INFERSSD_JOBS` select the installation
root, command directory and compilation parallelism (default 4).

To uninstall, stop active sessions, remove the `inferssd` command from the chosen
bin directory, and remove the installation's `releases`, `current` and `state`
entries. Keep `models` to preserve downloaded weights. Prompt caches and machine
measurements use the framework's existing cache locations; see
[machine profiles](machine-profile.md) and [configuration](configuration.md).

## Manual source installation

```sh
git clone --branch main https://github.com/wici-ai/inferssd.git
cd inferssd
python3 -m venv .venv
. .venv/bin/activate
python3 -m pip install ./framework
```

## Apple Silicon

Install the command-line developer tools, then CMake and Ninja (for example with Homebrew):

```sh
xcode-select --install
brew install cmake ninja
```

Build the Metal backend and describe the resulting deployment:

```sh
cmake --preset macos-metal
cmake --build build-macos-metal --target inferssd-server inferssd-cpuprobe inferssd-ssdprobe --parallel 4
python3 scripts/engine-resources.py \
  --build build-macos-metal --output engine-resources.json
```

A launch applies static typed settings on both platforms. Routed experts run on the CPU from SSD, while dense layers use Metal; unified memory does not need to hold the full expert set. See [Apple Silicon deployment](mac-launcher.md) for cache and thread settings.

## Muse on Mac

Use the same Metal build and point `-m` at the first Muse GGUF shard; the launch selects the `muse` profile from the GGUF's architecture. The deployment descriptor selects `engine/models/templates/muse-spark.jinja`, which replaces the chat template embedded in the GGUF. Keep the complete shard set and use a separate source index or matching pack for each checkpoint.

## Linux / NVIDIA

Install `liburing-dev` on Debian/Ubuntu, or your distribution's equivalent. The launcher also has [system requirements](#linux-requirements). Ensure the CUDA compiler is available to CMake; set `CMAKE_CUDA_ARCHITECTURES` if your toolchain needs an explicit target for your GPU.

```sh
cmake --preset linux-cuda
cmake --build build-linux-cuda --parallel 4 \
  --target inferssd-server inferssd-cpuprobe inferssd-ssdprobe inferssd-memprobe
python3 scripts/engine-resources.py \
  --build build-linux-cuda --output engine-resources.json
```

Without an NVIDIA GPU, build `--preset linux-cpu` and leave out `inferssd-memprobe`; add
`-DGGML_VULKAN=ON` with the Vulkan SDK to run an AMD or Intel GPU through Vulkan.
The [Linux resource prerequisites](testing.md#linux-resource-prerequisites) describe I/O, cgroup and pinned-memory requirements.

### Linux requirements

On Linux the launch runs the server in a systemd scope whose memory limit is the policy's `memory_budget_bytes` and which forbids the server to swap; other programs may still use swap. This needs:

- systemd as the init system. Most containers, and WSL without systemd enabled, do not provide it.
- The cgroup v2 unified hierarchy. It is the default on current distributions, for example Ubuntu 22.04 and later, Debian 11 and later, Fedora and RHEL 9.
- When you do not launch as root: a systemd user manager for your account (a login session, or `loginctl enable-linger`), with the memory controller delegated to it (`Delegate=memory` in a `user@.service` drop-in if your distribution does not already delegate it).

The launcher checks these before it starts the server and reports the first one that is missing.

## Prepare a launch

A source build runs with `inferssd serve` or `inferssd cli`, like an installed one. Point `--resources` at the descriptor written above and `-m` at the model's first GGUF shard.

The first launch on a machine starts by measuring it: CPUs, memory, GPUs and the drive that holds the model. This takes one to three minutes and prints its progress; the result is saved, and later launches reuse it. Each launch then chooses thread counts, CPU placement and similar machine-dependent settings from that measurement and the model. These choices do not change the model's output. The launch finds the measuring tools next to `inferssd-server`, which is why the build commands above build them too. See [machine profile](machine-profile.md) for what is measured, `--reprofile` and `--machine`.

Settings are optional. A settings file fixes every value it lists, so list only the values you want to fix; the rest are chosen for your machine at each launch. The [configuration guide](configuration.md) lists every setting and its default. Pass your file with `--settings settings.ssd`, or one value with `--set key=value`.

For DeepSeek:

```sh
inferssd serve \
  --resources engine-resources.json \
  -m /path/to/model-00001-of-00010.gguf
```

For Muse, Qwen3.8-Flash-Next or Qwen3.6-35B-A3B, select that model's first GGUF shard instead. The launch picks the model profile from the GGUF's `general.architecture`: `dsv` for DeepSeek-V4.1 (with its own sampling defaults, temperature 1.0, top-p 0.95), `muse` for Muse Spark, `qwen4exp` for Qwen3.8-Flash-Next and `qwen35moe` for Qwen3.6-35B-A3B; the two Qwen profiles keep the chat template embedded in their GGUF. The context defaults to 8192 tokens (`--context` sets it) and the batch sizes to what the launch chooses for this machine. The server listens on `127.0.0.1:8080`; pass `--port` to use another loopback port. Each launch keeps its logs and receipts under `~/.local/share/inferssd/state/runs`, and uses the whole machine: the engine keeps its own margin for the host.

If `expert.source` is omitted, the framework builds an alias index from the GGUF metadata without duplicating weight payloads. Use the [deployment resources guide](engine-resources.md) when moving binaries and templates outside the checkout.

To create a v1 pack, run the bundled builder on the shards you load. It needs NumPy and PyYAML for the bundled GGUF reader (`python3 -m pip install numpy pyyaml`):

```sh
python3 engine/tools/expert-pack/create-moe-expert-pack.py \
  '/path/to/model-*-of-00010.gguf' -o /path/to/model.expert-pack --align 4096
python3 engine/tools/expert-pack/create-moe-expert-pack.py --inspect -o /path/to/model.expert-pack
```

To launch with the pack, set `expert.source = ["/path/to/model.expert-pack"]` in the settings file. The v1 format records no checkpoint fingerprint, so rebuild the pack whenever the shards change.

Open **http://127.0.0.1:8080** after the server is ready, or point an OpenAI-compatible client at `http://127.0.0.1:8080/v1`. Stop the owned launcher with **Ctrl+C**.
