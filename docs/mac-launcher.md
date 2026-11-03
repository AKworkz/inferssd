# Apple Silicon deployment

Apple Silicon uses the same `inferssd serve` and `cli` commands and typed
settings contract as Linux, and the same static settings for each launch.

```sh
cmake --preset macos-metal
cmake --build build-macos-metal --target inferssd-server --parallel 4
python3 -m pip install ./framework
python3 scripts/engine-resources.py --build build-macos-metal --output engine-resources.json
inferssd serve --help
```

Use `--resources engine-resources.json` and `-m` pointing to the first GGUF
shard. All shards must be present. The launch picks the `dsv`, `muse`,
`qwen4exp` or `qwen35moe` profile from the GGUF's architecture, and each launch writes to a new
run directory. Settings are optional. See the
[quick start](quickstart.md#prepare-a-launch).

The framework checks GGUF metadata and creates an alias index without duplicating
model weights. It selects the matching chat template from the deployment
resources; `qwen4exp` and `qwen35moe` keep the template embedded in their GGUF. The server runs
the same SSD executor as on Linux: Metal computes the dense layers and the CPU
sink reads and computes every routed expert, with its RAM cache sized within
the process envelope (`host.budget_mib`, or the memory available at startup;
see [configuration](configuration.md)). There is no GPU expert tier on macOS.
The launch chooses the decode `-t` and prefill `-tb` thread counts; macOS cannot
pin threads to CPUs, and `cpu.io_cpus` is only validated. Model-specific memory envelopes need
measured memory evidence and are not inferred from machine RAM alone.

Earlier environment-driven launch scripts are not supported interfaces. Copies
kept under `framework/tests/fixtures` are test inputs only; do not run them as
production commands. See [deployment resources](engine-resources.md) for a wheel
installation that works without a checkout and
[measurement tools](../bench/README.md) for separate numerical and
performance evidence.
