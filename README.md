<h1 align="center">Large MoE models, powered by your SSD</h1>

<p align="center">
  <a href="https://wici.ai">
    <img src="docs/assets/wicissdinfer.png" width="600" alt="InferSSD">
  </a>
</p>

<p align="center">
  <a href="https://wici.ai/inferssd"><b>Website</b></a> &nbsp;·&nbsp;
  <a href="https://wici.ai/research"><b>Blog</b></a> &nbsp;·&nbsp;
  <a href="https://join.slack.com/t/wici-ai/shared_invite/zt-4f2e7ob2s-9nUg35Phqyqt4EI6qUjw5w"><b>Slack</b></a> &nbsp;·&nbsp;
  <a href="LICENSE">Apache-2.0</a>
</p>
<p align="center">macOS / Metal &nbsp;·&nbsp; Linux / CUDA &nbsp;·&nbsp; Windows (in progress)</p>

**Run models larger than your RAM or VRAM.** InferSSD builds on [llama.cpp](https://github.com/ggml-org/llama.cpp) to run large mixture-of-experts (MoE) models from SSD, using standard GGUF files, with browser chat and an OpenAI-compatible API. It supports **DeepSeek-V4.1-Flash-1017**, **Muse-Spark-Open**, **Qwen3.6-35B-A3B** and **Qwen3.8 Flash Next**, plus ordinary RAM inference for **Gemma4 E4B**.

## Models

These are the tested GGUF files, community conversions of the named model releases. DeepSeek's Q8_0 keeps the official bit widths: routed experts stay native 4-bit MXFP4, and FP8 tensors become 8-bit Q8_0. See [weight-format details](docs/benchmark.md#checkpoint-formats).

| Model | Parameter count | GGUF reference |
| --- | --- | --- |
| DeepSeek-V4.1-Flash-1017 | 763B total / 8–16B active | [mrfakename · Q8_0](https://huggingface.co/mrfakename/DeepSeek-V4.1-Flash-1017-GGUF) |
| Muse-Spark-Open | 816B total / 42B active | [Unsloth · UD-Q4_K_XL](https://huggingface.co/unsloth/Muse-Spark-Open-816B-A42B-GGUF) |
| Qwen3.6-35B-A3B | 35B total / 3B active | [Unsloth · UD-Q4_K_XL](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF) |
| Qwen3.8 Flash Next | 177B total / A3B | [Bartowski · IQ4_XS](https://huggingface.co/bartowski/Qwen3.8-Flash-Next-GGUF) |
| Gemma4 E4B | 7.46B total / E4B | [Google · QAT Q4_0](https://huggingface.co/google/gemma-4-E4B-it-qat-q4_0-gguf) |

Gemma uses the ordinary native loader without SSD expert offload. Before loading, the shared
launcher checks that weights, KV cache and working memory fit the live available RAM, the
configured budget and headroom. An insufficient fit is rejected. macOS and Windows use the
same model selection and launch entry points.

## Performance benchmarks

**4K prompt tier · 256 output tokens · decode measured over tokens 129–256.** All rates are in **token/s**; higher is faster. The multiple next to each **InferSSD** rate is its speedup over llama.cpp on the same hardware. On the MacBook Pro, Colibri runs the official Safetensors on CPU; llama.cpp and InferSSD run the GGUF with Metal. See [measurement details](docs/benchmark.md).

### Test hardware

| Hardware | CPU (cores) | GPU | Memory | SSD |
| --- | --- | --- | --- | --- |
| MacBook&nbsp;Pro | Apple&nbsp;M5&nbsp;Max&nbsp;(18) | 40-core | 48&nbsp;GB&nbsp;unified | 2&nbsp;TB&nbsp;internal |
| RTX&nbsp;5060&nbsp;Ti | Intel&nbsp;Core&nbsp;3&nbsp;100U&nbsp;(6) | RTX&nbsp;5060&nbsp;Ti | 8&nbsp;GB&nbsp;+&nbsp;16&nbsp;GB&nbsp;VRAM | 2&nbsp;TB&nbsp;PCIe&nbsp;4&nbsp;×4 |
| RTX&nbsp;5090 | Intel&nbsp;Core&nbsp;Ultra&nbsp;5&nbsp;230F&nbsp;(10) | RTX&nbsp;5090 | 32&nbsp;GB&nbsp;+&nbsp;32&nbsp;GB&nbsp;VRAM | 2&nbsp;TB&nbsp;PCIe&nbsp;5&nbsp;×4 |
| DGX&nbsp;Spark | NVIDIA&nbsp;GB10&nbsp;(20) | Blackwell | 128&nbsp;GB&nbsp;unified | 4&nbsp;TB&nbsp;PCIe&nbsp;5&nbsp;×4 |

Unified memory is shared by the CPU and GPU. SSD capacities are nominal. See [hardware details](docs/benchmark.md#hardware-details) for SSD models, CPU core layouts, and configuration notes.

### DeepSeek-V4.1-Flash-1017

<table>
<thead>
<tr><th rowspan="2">Hardware</th><th colspan="3">Prefill (token/s)</th><th colspan="3">Decode (token/s)</th></tr>
<tr><th>llama.cpp</th><th>Colibri</th><th>InferSSD</th><th>llama.cpp</th><th>Colibri</th><th>InferSSD</th></tr>
</thead>
<tbody>
<tr><td>MacBook&nbsp;Pro</td><td align="right">5.929</td><td align="right">0.708</td><td align="right"><b>28.766</b>&nbsp;(4.85×)</td><td align="right">0.873</td><td align="right">0.437</td><td align="right"><b>2.874</b>&nbsp;(3.29×)</td></tr>
<tr><td>RTX 5060 Ti</td><td align="right">6.950</td><td align="right">7.020</td><td align="right"><b>75.100</b>&nbsp;(10.81×)</td><td align="right">0.242</td><td align="right">0.900</td><td align="right"><b>2.550</b>&nbsp;(10.54×)</td></tr>
<tr><td>RTX 5090</td><td align="right">8.060</td><td align="right">39.210</td><td align="right"><b>164.500</b>&nbsp;(20.41×)</td><td align="right">0.867</td><td align="right">1.878</td><td align="right"><b>7.270</b>&nbsp;(8.39×)</td></tr>
<tr><td>DGX Spark</td><td align="right">2.730</td><td align="right">0.826</td><td align="right"><b>127.500</b>&nbsp;(46.70×)</td><td align="right">1.608</td><td align="right">0.356</td><td align="right"><b>7.260</b>&nbsp;(4.51×)</td></tr>
</tbody>
</table>

### Muse-Spark-Open

<table>
<thead>
<tr><th rowspan="2">Hardware</th><th colspan="3">Prefill (token/s)</th><th colspan="3">Decode (token/s)</th></tr>
<tr><th>llama.cpp</th><th>Colibri</th><th>InferSSD</th><th>llama.cpp</th><th>Colibri</th><th>InferSSD</th></tr>
</thead>
<tbody>
<tr><td>MacBook&nbsp;Pro</td><td align="right">3.140</td><td align="right">N/S</td><td align="right"><b>14.503</b>&nbsp;(4.62×)</td><td align="right">0.153</td><td align="right">N/S</td><td align="right"><b>1.097</b>&nbsp;(7.17×)</td></tr>
<tr><td>RTX 5060 Ti</td><td align="right">3.410</td><td align="right">N/S</td><td align="right"><b>52.100</b>&nbsp;(15.28×)</td><td align="right">0.059</td><td align="right">N/S</td><td align="right"><b>0.219</b>&nbsp;(3.71×)</td></tr>
<tr><td>RTX 5090</td><td align="right">4.000</td><td align="right">N/S</td><td align="right"><b>106.600</b>&nbsp;(26.65×)</td><td align="right">0.155</td><td align="right">N/S</td><td align="right"><b>1.520</b>&nbsp;(9.81×)</td></tr>
<tr><td>DGX Spark</td><td align="right">1.740</td><td align="right">N/S</td><td align="right"><b>83.900</b>&nbsp;(48.22×)</td><td align="right">0.120</td><td align="right">N/S</td><td align="right"><b>2.140</b>&nbsp;(17.83×)</td></tr>
</tbody>
</table>

**N/S:** Muse is not currently supported by Colibri.

## Get started

Install on Apple Silicon macOS or on Linux with an NVIDIA GPU:

```sh
curl -fsSL https://github.com/wici-ai/inferssd/releases/latest/download/install.sh | sh
```

Each numbered release includes checked prebuilt packages for Apple Silicon macOS 15 or later and Windows 11 x64, built from the same tagged commit. Its installers download that exact release without developer tools; see [the macOS package](docs/macos-package.md) and [Windows package](docs/windows-package.md). Linux builds that release from source for the machine's CPU and GPU, which takes about 2 to 12 minutes. The installer on `main` downloads the latest published package:

```sh
curl -fsSL https://raw.githubusercontent.com/wici-ai/inferssd/main/scripts/install.sh | sh
```

The installer works in a private environment and uses no `sudo`. A source build needs Git, Python 3.9 or later, and a C/C++ compiler. On macOS it also needs the Xcode Command Line Tools; Linux also needs systemd with cgroup v2, and the CUDA toolkit on a machine with an NVIDIA GPU. Follow the printed PATH instruction if it shows one.

Then start a model:

```sh
# Chat in your terminal
inferssd cli -hf mrfakename/DeepSeek-V4.1-Flash-1017-GGUF

# Or start the web UI and an OpenAI-compatible API at http://127.0.0.1:8080
inferssd serve -hf mrfakename/DeepSeek-V4.1-Flash-1017-GGUF
```

For Muse, use `unsloth/Muse-Spark-Open-816B-A42B-GGUF`; for Qwen3.6, `unsloth/Qwen3.6-35B-A3B-GGUF`; for Flash Next, `bartowski/Qwen3.8-Flash-Next-GGUF`; for Gemma, `google/gemma-4-E4B-it-qat-q4_0-gguf`. With `serve`, open `http://127.0.0.1:8080` in your browser, or point any OpenAI-compatible client at `http://127.0.0.1:8080/v1`. Add `--host 0.0.0.0` to reach it from other devices on your network; the API has no authentication, so anyone who can reach the port can use it.

The first run downloads the model files (about 470 GiB for DeepSeek or Muse, 21 GiB for Qwen3.6) to `~/.local/share/inferssd/models` and measures your machine. Use `--model-dir /path/on/ssd` to choose the drive, or `-m /path/to/first-shard.gguf` for files you already have. There is no memory setting to choose: InferSSD uses the memory available. Stop with **Ctrl+C**, or `/exit` in chat.

See the [quick start](docs/quickstart.md#one-command-installation) for upgrades and uninstalling.

## Versions

InferSSD has a release number, `0.1.3` in `framework/pyproject.toml`, and every build has a
label of its own, `<yymmdd>.<build>.<commit>`: the day (UTC) of the commit it was built from, the
branch's commit count up to that commit, and the commit. The installer records the three in
`engine-resources.json` beside the build, from the clone's Git metadata, and so does a local
build that runs `scripts/engine-resources.py`; a source archive has no Git metadata and records
none. The [macOS](docs/macos-package.md) and [Windows](docs/windows-package.md) packages carry those of the same tagged commit in each numbered release
and are named by that label. `inferssd --version` prints the release and the label, `inferssd --version --json` every
field:

```sh
$ inferssd --version
inferssd 0.1.3 (261106.202.03af979)
$ inferssd --version --json
{"version": "0.1.3", "build": 202, "revision": "03af97…", "date": "2026-11-06", "branch": "main", "label": "261106.202.03af979"}
```

The engine only says what it is. Whether a build is one it can work with is for whatever runs
the engine to decide; the WiCi AI desktop app, for one, installs `main`, offers its newer
builds as updates, and keeps a floor of its own on `build`. Nothing here refers to it.

## Further reading

- [How it works](docs/architecture.md)
- [Benchmark methodology](docs/benchmark.md)

## Contributing

Contributions are welcome, including hardware results and improvements to expert I/O, caching, and batching. See the [contributing guide](CONTRIBUTING.md).

## Acknowledgments and license

Built on [llama.cpp and ggml](https://github.com/ggml-org/llama.cpp), with thanks to their contributors and the model-adaptation community.

InferSSD is licensed under [Apache-2.0](LICENSE); bundled llama.cpp/ggml and other third-party components keep their original licenses and notices; see [NOTICE](NOTICE). Model weights remain subject to their publishers' licenses.
