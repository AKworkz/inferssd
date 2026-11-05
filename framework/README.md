# InferSSD framework tools

The package supplies the `inferssd` command, typed settings, source metadata and
an owned runtime that supervises one local server. The runtime uses only the
Python standard library.

For download-and-run usage, `inferssd cli -hf REPO` opens terminal chat and
`inferssd serve -hf REPO` starts the API. These commands prepare model files and
a default resource policy, then use the owned runtime. Install the optional
Hub client with `python3 -m pip install './framework[hf]'`, or use the
[one-command installer](../docs/quickstart.md#one-command-installation), which also
builds the native engine. `-m FILE` uses local weights without the Hub dependency.

## Commands

| Command | Input | Output |
| --- | --- | --- |
| `inferssd cli` | Tested Hugging Face repository or local GGUF | Streaming terminal chat; owned server stops on exit |
| `inferssd serve` | Tested Hugging Face repository or local GGUF | OpenAI-compatible API with automatic resource policy |
| `inferssd profile` | This machine, optional model paths and the native probes | Machine characteristics file and a summary per section; see [machine profile](../docs/machine-profile.md) |

## Settings and launch choices

`serve` and `cli` take `--settings FILE` for the versioned registry-defined
configuration and repeated `--set key=literal` for explicit overrides. CLI values take precedence over file
values, then the launch's own choices for this machine, then registry defaults. Comments following a file value are preserved as
source receipts. The generated native file contains file/generated values;
CLI-origin values become the matching registry-generated native `--ssd-*` option
so native effective telemetry preserves that priority. There is no environment
translation. Retired settings/launcher environment names fail by name without
printing their values.

CPU threads are server arguments, not settings. The SSD executor computes in the
server's ggml threadpools and adds only its I/O, prefetch-reader and watch
threads, which `cpu.io_cpus` places. The launch chooses the decode `-t` and
prefill `-tb` counts and, on Linux, the CPUs they run on.

An executed launch measures the machine before its first launch and reuses the
result afterwards (`--reprofile` measures again; `--machine PATH` uses a given
characteristics file and measures nothing); see [machine profile](../docs/machine-profile.md).
From that file, the model's GGUF header and the CPUs, GPUs and memory it finds
at that moment, the launch chooses thread counts, CPU placement, `place.costs`
and, for an engine with CUDA but no usable CUDA GPU, `vram.budget = 0` (on macOS also
`host.budget_mib` from the runtime policy). With an engine that has CUDA and one usable
CUDA GPU it
also chooses `prefill.cpu_max_tokens`, the longest prompt pass the CPU prefills
no slower than the GPU graph, and prefills each prompt in one pass (`-b` and
`-ub` as large as the context); on a GPU that shares host memory it keeps
`host.headroom_mib = 4096`, and on macOS a quarter of the machine's memory. It fills only settings left at their registry
defaults and thread options and sizes not given, so explicit values always win.
The choices are written to the generated `settings.ssd` with `source=strategy`
receipts and to `strategy.json`, with the values your settings kept and the
choices left out because they contradicted them. The server keeps one context
checkpoint (`--ctx-checkpoints 1`) unless you set `--ctx-checkpoints` or
`LLAMA_ARG_CTX_CHECKPOINTS`.

The model profile (chat template and model settings) follows the GGUF's
`general.architecture`. The `dsv` profile also sets
DeepSeek-V4.1's own sampling defaults (`--temp 1.0 --top-p 0.95 --top-k 0
--min-p 0`), which a request's own values or your extra server arguments
replace.

The prompt cache lives on disk, one directory per model under
`inferssd/prompts` in your user cache directory, and survives restarts
(`--cache-disk`, 16 GiB per model). With the `dsv` profile the DeepSeek-V4.1
template keeps earlier turns' thinking (`drop_thinking` false), so a
conversation's history renders as the server cached it.

```text
inferssd-settings 1
prefetch.depth = 4 # measured predictor depth
host.headroom_mib = 3072 # operator resource reserve
vram.budget = auto # use admitted capacity
```

A model must be a local GGUF or the first shard of a complete set. An omitted
`expert.source` generates a strict header-validated alias from that set; explicit
sources must be nonempty existing alias paths. Native source validation remains
a separate boundary. `serve` and `cli` download a tested model with `-hf`, or take
a local one with `-m`. Other compute processes on the GPU do not stop a launch:
it leaves them VRAM (`vram.reserve_mib` in the [settings](../docs/configuration.md)).

The source operations behind the legacy artifact formats (`index`, `facts`,
`lane_args`, `dense` and `iolog`) remain Python functions in
`inferssd.source.experts`, with explicit filesystem and output ports; they read
headers only and never run a probe. A generated alias is not a payload-integrity
certificate.

## Compatibility and failure behavior

The package was extracted from earlier adaptive scripts. Their recorded
reference revision is `8b12192920b339a4ef31c9006d1ebe20304ce45a`; verbatim
reference sources and synthetic observations are under `tests/fixtures`. Extraction preserves arithmetic, sum and mapping order, stable
ties, rounding, borrowed references, callback order and exception timing.
Toy logs are synthetic inputs, not claimed engine-run evidence. Historical
archives establish only their actual recorded input coverage.

The installation requires Python 3.9 or later. Normal-wheel regression and
exact reference transcripts are verified on Python 3.9 and 3.12. Their original
results can differ because of float summation, resident-set iteration or native
exception wording. Tests retain separate runtime transcripts and compare exactly
within each runtime, without a numeric tolerance or production version branch.
Recorded transcript checks run only on those two versions; direct comparisons
with the reference sources that do not require a recorded transcript still run.
Future versions are not assumed identical to either verified runtime.

Legacy artifact mappings are typing views, not new schemas. They are not copied,
normalized, validated or filled with defaults on ingestion. Missing, null, empty
and zero values retain distinct behavior, including failures when consumed.

### File and output ordering

Source header reads retain the original mmap lifetime and decoding: metadata
strings use replacement decoding, while textual artifacts and pack tensor names
use strict decoding. Split-model discovery uses the original sorted glob and does
not newly reject missing shards. Type 42 (`F8_E4M3_B128`) remains recognized as
historical header metadata; this does not claim native execution support.

`experts.index` retains mtime-based reuse, including the original set iteration over
source paths. Recorded filesystem transcripts use `PYTHONHASHSEED=0` as a test
condition; dynamic comparisons also check another seed. Production iteration is
unchanged. GGUF alias output is closed and replaced before the expert JSON is
written. Failure can leave a directory, temporary file or the already-published
alias, as in the reference. Pack indexing does not create a GGUF alias. Dense
placement keeps input order and validates each consumed line before writing its
alias. No new
whole-artifact validation, transaction, random seed default or filesystem probe
is inserted into these paths.

## Python interfaces

| Area | Modules and boundary |
| --- | --- |
| Source metadata | `source.header`, `source.experts`, `source.files`: header/parser and legacy artifact operations with explicit filesystem, output and random ports |
| Machine | `machine.*`: identity, probes, characteristics and their store, behind `inferssd profile` |
| Launch | `launch.*`, `strategy.*`, `configuration.settings`: typed settings, the launch strategy and the typed launch behind `serve` and `cli` |
| Owned runtime | `engine.*`, `platform.*`: the guardian, worker and owned server |
| Native observations | `telemetry.jsonl`, `telemetry.observations`, `telemetry.files`: versioned native events and local file access |

Source operations borrow their filesystem ports; import and adapter construction
perform no file or process work. These interfaces preserve existing semantics;
they are not a claim that every legacy artifact now has a versioned, validated
contract.

## How serve and cli run the server

`inferssd serve` and `cli` resolve the launch for Linux or macOS from the live
machine, then start one supervised server on a loopback port. Each launch writes
a resource policy for the whole machine, `memory_budget_bytes` set to its
physical memory, and claims a new run directory under
`~/.local/share/inferssd/state/runs`. Windows is not supported yet.

On Linux the server runs in a systemd scope with cgroup v2 whose memory limit is
`memory_budget_bytes`, with zero swap and readable ancestor OOM counters and
memory peak; see the [Linux requirements](../docs/quickstart.md#linux-requirements).
The worker checks the actual envelope and the server's inherited scope and
memlock limit. It never raises system limits.

On macOS the budget is a **sampled soft server RSS threshold**, not a cgroup or a
hard unified-memory cap. Both platforms monitor while the
server runs, and fail the execution if a threshold or the worker fails. Host
utility calls have a ten-second timeout. These checks do not replace per-host
validation or prove a model will fit.

Execution claims `.inferssd-owner` atomically before opening logs. A directory
may already exist, but an existing execution claim or execution artifact is
never overwritten, including after successful completion. A cooperating device
lock is acquired before writing the launch specification. The guardian, worker
and server receive the same locked file description; only owned groups and the
fresh exact Linux scope may be cleaned up. A disconnected controller, STOP or
signal requests cleanup. If cleanup is unknown, the guardian retains the lock
and writes quarantine evidence; a timeout never authorizes killing that guardian.
An unrelated listener on the health port is never killed.

`settings.ssd` records the file/generated settings and their receipts;
`native-events.jsonl` is exclusively created by the context-owned native writer.
`runtime.json` records server argv and scope, `memory-samples.jsonl` records
resource samples, and `ownership.json` records final status and cleanup.
Keep these artifacts private if arguments or paths contain private information.
Server loader variables are carried in the process environment and restored
only for the server, not serialized into the launch specification or applied to
the Python lock owner. Guardian and scoped worker use the same installed interpreter
with Python isolated mode; a package in the caller's directory or PYTHONPATH
cannot replace their implementation. A successful cleanup does not turn a failed server or
resource check into a successful exit.

Model, listener, parallel, thread and batch arguments are owned by the launch, so
settings cannot replace them. macOS cannot pin threads to CPUs. A positive
`cpu.decode_participants` cannot exceed the decode thread count. The launch
receipt and server argv record the thread counts and CPUs.

`inferssd.telemetry.observations.NativeObservationSession` reads a run's owned
`native-events.jsonl` stream; the benchmark CI uses it to record what each run
did. It requires the native observation contract (the `ssd.controller.begin`
event, schema 1) and versioned placement/prefill records, checks event sequence
and field types, and rejects failed layer observations. Human-readable server
logs remain diagnostic artifacts and are never parsed. Static launch does not
require debug channels. See [native observations](../docs/architecture/native-observations.md).

## Remaining integration boundaries

The native context integration must be validated with this installed package on
each target host; Python substitutes for OS effects or synthetic model headers do
not establish native execution correctness. No new performance conclusion
follows from these package tests. Importing these modules starts no process,
host probe or network call.

## Development checks

```sh
python3 -m pytest framework/tests
mypy --config-file framework/pyproject.toml framework/src
ruff check framework
ruff format --check framework
```

Use a mypy release supporting the configured Python 3.9 target (verified with
1.18.2). CI installs the normal package before running its tests and these checks.
The preserved reference sources under `tests/fixtures` are intentionally
excluded from formatting/lint rewrites.

## Installed engine resources

`serve` and `cli` use the engine the installer set up. `--resources FILE` points
them at the descriptor of a source build instead. The descriptor identifies the
executable, per-profile templates, optional native probes, and supported
configuration/telemetry contract. Paths resolve relative to the descriptor, never
the current directory. See the [deployment guide](../docs/engine-resources.md).
