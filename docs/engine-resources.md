# Engine deployment resources

`inferssd serve` and `cli` use the engine the installer set up. The installed
framework does not discover a source checkout: pass `--resources` to use a
deployment descriptor, for example one generated from a source build.

A descriptor is ordinary JSON. Relative paths resolve against its own directory:

```json
{
  "schema": "inferssd-engine 1",
  "binary": "bin/inferssd-server",
  "revision": "the-deployed-source-revision",
  "templates": {
    "dsv": "templates/deepseek-ai-DeepSeek-V4.1.jinja",
    "muse": "templates/muse-spark.jinja"
  },
  "probes": {},
  "capabilities": {
    "settings": "inferssd-settings 1",
    "telemetry": 1,
    "platform": "linux",
    "gpu_backends": ["CUDA"]
  }
}
```

Use `darwin` for an Apple Silicon deployment and `windows` for a Windows one.
`gpu_backends` lists the GPU backends the engine was built with, of `CUDA`,
`Metal` and `Vulkan`; an empty list is a CPU-only build. The launch takes the CUDA
expert tier's settings only from an engine that has `CUDA`. Without the field,
the launch reads the backends from the `CMakeCache.txt` of the build holding the
server; when neither says, the launch strategy takes them to be those of the
machine's GPUs. The `qwen4exp` and `qwen35moe` profiles need no template: they
launch with the chat template in their GGUF. Missing selected resources or incompatible
contract versions fail before engine startup. The descriptor is an operator's
explicit deployment contract, not a claim that untrusted binaries were verified.
Normal process ownership, memory policy and runtime evidence checks still apply.

From a source checkout, generate the descriptor after building:

```sh
python3 scripts/engine-resources.py --build build-linux-cuda --output engine-resources.json
```

This inspects the schema without running a model, and reads the GPU backends from
the build's `CMakeCache.txt`. It describes existing paths;
for a standalone deployment, copy the executable and its runtime libraries plus
templates, then use paths relative to the deployed descriptor. The framework
wheel needs only its standard-library Python runtime. Existing source indexing
uses packaged code and reads GGUF metadata directly, without importing engine
source files or requiring a separate GGUF Python installation.

`--resources /deployment/engine.json` works from any current directory; see the
[settings](configuration.md) for what a launch can be given. The optional `probes`
map names `io`, `memory`, `ssd` and `cpu` executable resources, which
`inferssd profile` and the first launch run to measure the machine.
