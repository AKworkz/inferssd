# InferSSD framework tools

The package supplies typed settings, source metadata, an owned inference runtime,
and recorded calibration/planning tools. The production `launch` command consumes
`inferssd-settings 1` and explicit CLI overrides. It supervises one local server;
Linux also exposes a typed controller factory. Mac currently supports static
launch only; online adaptive control on Mac is not implemented yet. The runtime
uses only the Python standard library.

For download-and-run usage, `inferssd cli -hf REPO` opens terminal chat and
`inferssd serve -hf REPO` starts the API. These commands prepare model files and
a default resource policy, then use the same owned runtime. Install the optional
Hub client with `python3 -m pip install './framework[hf]'`, or use the
[one-command installer](../docs/quickstart.md#one-command-installation), which also
builds the native engine. `-m FILE` uses local weights without the Hub dependency.

## Commands and artifacts

| Command | Input | Output |
| --- | --- | --- |
| `inferssd cli` | Tested Hugging Face repository or local GGUF | Streaming terminal chat; owned server stops on exit |
| `inferssd serve` | Tested Hugging Face repository or local GGUF | OpenAI-compatible API with automatic resource policy |
| `inferssd launch --execute` | Model, typed settings, explicit resource policy | Owned server, settings/telemetry/resource/cleanup artifacts |
| `inferssd launch --dry-run` | Same configuration, recorded host facts | Prepared settings/source indexes and native argv; no server |
| `inferssd launch-recorded --dry-run` | Historical launcher inputs and recorded facts | Historical resolution for comparison; cannot execute |
| `inferssd configuration` | Typed settings and `--set` overrides | Effective values/source receipts, schema or canonical settings |
| `inferssd profile` | This machine, optional model paths and the native probes | Machine characteristics file and a summary per section; see [machine profile](../docs/machine-profile.md) |
| `inferssd source` | GGUF headers, optional v1 pack index, dense placement text | Expert JSON/alias TSV, source facts, lane arguments or fio iolog |
| `inferssd probe` | Existing probe reports and expert metadata | Device schema-2 JSON |
| `inferssd validate` | Device schema-2 JSON | Validation result |
| `inferssd fit` | Named arm directories, optional graph arm and ubatch JSON | Model schema-2 JSON, insitu schema-3 JSON and report |
| `inferssd plan` | `insitu.json`, `probe/device.json`, `model.json` | Plan JSON and ranked candidates |
| `inferssd settings` | Recorded artifacts and saved plan | Historical `settings.env` for recorded comparisons |
| `inferssd bootstrap-records` | Recorded CPU labels or ubatch logs | CPU-set candidates or historical `{first, held}` JSON |
| `inferssd report` | Saved plan, conversation events and logs | Comparison text and historical `compare-online.json` |

Install from the repository root:

```sh
python3 -m pip install ./framework
```

## Prepare a typed launch

Use `--settings FILE` for the versioned registry-defined configuration and repeat
`--set key=literal` for explicit overrides. CLI values take precedence over file
values, then the launch's own choices for this machine, then registry defaults. Comments following a file value are preserved as
source receipts. The generated native file contains file/generated values;
CLI-origin values become the matching registry-generated native `--ssd-*` option
so native effective telemetry preserves that priority. There is no environment
translation. Retired settings/launcher environment names fail by name without
printing their values.

CPU threads are server arguments, not settings. The SSD executor computes in the
server's ggml threadpools and adds only its I/O, prefetch-reader and watch
threads, which `cpu.io_cpus` places. `--threads` and `--batch-threads` select the
decode `-t` and prefill `-tb` counts; `--cpus LIST` pins both pools one thread per
CPU.

An executed launch measures the machine before its first launch and reuses the
result afterwards (`--reprofile` measures again; `--machine PATH` uses a given
characteristics file and measures nothing); see [machine profile](../docs/machine-profile.md).
From that file, the model's GGUF header and the CPUs, GPUs and memory it finds
at that moment, the launch chooses thread counts, CPU placement, `place.costs`
and, without a usable CUDA GPU, `vram.budget = 0` (on macOS also
`host.budget_mib` from the runtime policy). With one usable CUDA GPU on Linux it
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
`general.architecture`; `--profile` overrides it. The `dsv` profile also sets
DeepSeek-V4.1's own sampling defaults (`--temp 1.0 --top-p 0.95 --top-k 0
--min-p 0`), which a request's own values or your extra server arguments
replace.

The prompt cache lives on disk, one directory per model under
`inferssd/prompts` in your user cache directory, and survives restarts
(`--cache-disk`, 16 GiB per model); `--prompt-cache DIR` moves it and
`--prompt-cache off` runs without one. With `--profile dsv` the DeepSeek-V4.1
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
a separate boundary. The launcher does not download models. Use `--binary` for
an explicit server executable; otherwise `--repo` selects the checkout's build as
described in [engine resources](../docs/engine-resources.md).

Dry-run requires recorded host facts and a fresh exclusively claimed output
directory. It may create source metadata and `settings.ssd`, but does not query
the live GPU, start a server or claim a device lease. A receipt lists environment
names, never inherited environment values:

```sh
inferssd launch --dry-run --platform linux --repo . \
  --model models/model.gguf --settings recorded/settings.ssd \
  --runtime-policy recorded/policy-linux.json \
  --host-facts recorded/host-linux.json --output-dir recorded/prepared-001 \
  --threads 7 --result recorded/launch-001.json
```

Recorded host facts are consumed lazily without falling back to the current
machine. A recording with `machine_characteristics` (an `inferssd-machine/1`
object, or `--machine PATH`) also derives the launch's choices; it then needs
`runtime_facts` (`cpus`, `gpus` as `{uuid, free_bytes, processes}` or `null`,
`available_bytes`), and may give `model_drive_key` and `model_facts` (read from
the model's headers when absent). Without `machine_characteristics` the dry-run
resolves exactly as before. Every platform requires `platform_system` (`Linux`, `Darwin` or
`Windows`); Darwin also requires `platform_machine=arm64`. Linux uses
`cgroup_v2`, `euid`, and for non-root scope creation `user_manager` and
`memory_controller`. GPU-enabled Linux settings also require `gpu_available`.
These observations do not certify native source formats, GPU kernels or real
resource enforcement. Other compute processes on the GPU do not stop a launch:
it leaves them VRAM (`vram.reserve_mib` in the
[settings](../docs/configuration.md)).

The old Linux/Darwin resolvers and their recorded reference observations are
retained behind `launch-recorded --dry-run`. They still describe the historical settings/env
contract, and cannot execute a server. Their golden outputs are not observations
of the new typed production launch. Header-only source and recorded planning
commands below continue to accept their documented historical artifact formats.

Source artifacts can be generated without reading tensor payloads or running a probe:

```sh
inferssd source index models/model.gguf recorded/expert-index
inferssd source facts models/model.gguf recorded/expert-index/experts.tsv
inferssd source lane-args recorded/expert-index
inferssd source dense models/model.gguf recorded/dense-homes.tsv recorded/dense.tsv
inferssd source iolog recorded/expert-index 100 17 recorded/reads.iolog
```

`source index MODEL [PACK] DIR` optionally uses a v1 expert pack index.
`source iolog DIR N SEED OUT [FILE]` optionally selects one source file; it writes
fio input text and does not invoke fio or read the referenced payload ranges.
`dense-homes.tsv` contains `tensor<TAB>ram|drive` lines. These commands retain the
legacy artifact formats and consumed-input errors. The new CLI uses argparse
for missing/invalid command arguments rather than the old script's usage text.
A generated alias is not a payload-integrity
certificate or a substitute for the stricter header-validated alias that a launch
without `expert.source` generates.

Recorded bootstrap inputs and results can be read separately:

```sh
inferssd bootstrap-records coresets recorded/cpus.txt
inferssd bootstrap-records tier-slots recorded 32 64 128 > recorded/ubatch.json
```

The CPU-row file contains `cpu capacity max_khz [thread_siblings]`. Candidates
keep the recorded sibling filtering, 92% grouping against the first member of a
group, stable ranking and duplicate behavior. `tier-slots` reads each recorded
`ubprobe-UB.out` first line and that run's `server.log.gz`; repeated UB arguments
are read again and keep their original JSON key position. Neither command
queries the current machine. Empty/malformed rows and the legacy zero/None tier
cases retain their original errors; only consumed `OSError`s in `tier-slots`
become null entries. These commands provide recorded inputs for later steps and
do not run native bootstrap or fix the legacy parser's edge cases.

A complete recorded calibration path, assuming the destination directories already exist:

```sh
inferssd probe --from-reports records/probe --experts records/expert-index \
  > recorded/probe/device.json
inferssd validate --device recorded/probe/device.json
inferssd fit --run all=records/all --run pin=records/pin --skip 128 \
  --profile recorded/model.json -o recorded/insitu.json
inferssd plan recorded --prompts 512 --decode 256 --window 1-256 -o recorded/plan.json
inferssd settings recorded -o recorded/settings.env
inferssd report recorded
```

`fit --graph DIR` reads an existing graph arm. `--ubatch-probe FILE` accepts the
legacy numeric or `{first, held}` ubatch JSON. Neither option executes a probe.
The first named placement is the reference; repeated names are all measured and
replace the earlier value without moving its insertion position.

`plan` also accepts `--known` and `--pending`. `settings` supports `--tier`,
`--cpus`, `--prompts`, `--decode` and `--ledgers DIR ...`. Use each command's
`--help` for its arguments. For a valid split placement without truthy recorded
`default_threads`, supply `--host-threads` to `plan`, `settings` or `report`. This explicit
fallback is requested only when needed. The offline commands do not infer a
recorded machine's CPU count from the current host.

## Compatibility and failure behavior

The package was extracted from earlier adaptive scripts. Their recorded
reference revision is `8b12192920b339a4ef31c9006d1ebe20304ce45a`; verbatim
reference sources and synthetic observations are under `tests/fixtures`. Extraction preserves arithmetic, sum and mapping order, stable
ties, rounding, mutable replay state, borrowed references, callback order and
exception timing. Composed expectations come from the complete reference call
graph, including request output → calibration → plan → reload → settings →
Controller → recorded report.
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

Most legacy artifact and model mappings are typing views, not new schemas. They
are not copied, normalized, validated or filled with defaults on ingestion.
Missing, null, empty and zero values retain distinct behavior, including failures
when consumed. For example, replay keeps the reference's length-only
incremental-segment check and duplicate-hit behavior; workload iterators retain
their consumption order; singular/empty fits are not repaired. `fit_response`
returns a tuple rather than the reference's list, and calibration immediately
consumes it in the original rounded list comprehension. Complete-command tests
bind those results and failures.

Two explicit typed APIs have stricter boundaries:

- `DeviceProfile` validates schema 2 and uses immutable nested records. It rejects
  unknown schema versions and lane fields, missing required fields, numeric
  strings, booleans as
  numbers, non-finite/out-of-range numbers and non-integer expert byte sizes.
  Validation normalizes measurement fields to floats; it does not promise to
  preserve arbitrary JSON whitespace or unknown extension fields. Old artifacts
  missing required observations need an explicit adapter, not fabricated facts.
- `LayerResponse` validates finite layer/count/beta values and integer `pf_k`.
  Its `price_decode` adapter uses the single Mapping-based reference pricing
  core, `price_mapping`, and preserves successful results. Uncapped null
  readiness raises the reference's native `TypeError` at exponentiation rather
  than the adapter's former custom message; the exception class and empty-range
  behavior are unchanged.

The recorded probe loader preserves the full-path queue-depth regex, even when a
parent directory's `qd8` precedes the filename's `qd1`. Malformed/unreadable read
reports, including invalid UTF-8, are skipped; a valid primary report does not
read its fallback. Invalid encoding in required inputs or file-list/lane text
still fails. Nonempty malformed observations are not silently defaulted.

### File and output ordering

Source header reads retain the original mmap lifetime and decoding: metadata
strings use replacement decoding, while textual artifacts and pack tensor names
use strict decoding. Split-model discovery uses the original sorted glob and does
not newly reject missing shards. Type 42 (`F8_E4M3_B128`) remains recognized as
historical header metadata; this does not claim native execution support.

`source index` retains mtime-based reuse, including the original set iteration over
source paths. Recorded filesystem transcripts use `PYTHONHASHSEED=0` as a test
condition; dynamic comparisons also check another seed. Production iteration is
unchanged. GGUF alias output is closed and replaced before the expert JSON is
written. Failure can leave a directory, temporary file or the already-published
alias, as in the reference. Pack indexing does not create a GGUF alias. Dense
placement keeps input order and validates each consumed line before writing its
alias. No new
whole-artifact validation, transaction, random seed default or filesystem probe
is inserted into these paths.

Legacy arm parsing prefers a readable gzip server log; corrupt gzip is an error,
not a reason to retry the plain file. Server text uses replacement decoding,
while client JSONL and metadata use strict decoding. A supplied incremental parser
skips the server log but rereads client/metadata. The standalone ledger reader
requires a record at the start of a line; the full parser retains its different
prefix-stripping behavior. Parser construction and package imports do no file,
host or process work.

`fit` directly writes the compact model profile before drive normalization and
optional ubatch processing. It writes the indented insitu artifact before printing
the report. Later errors can leave one or both files, including partial output.
The temporary streams passed to `json.dump` retain the reference's implicit-close
behavior; there is no transaction or new explicit `with` block.

`plan` reads insitu, device and model artifacts in that order. A saved plan is
read lazily only when `pending is None`; an explicitly empty pending sequence
suppresses that read. All rejected candidates are printed before failure, and
an all-rejected plan creates no output. Successful plan JSON is written directly.
The injected clock is fixed in exact-output tests; real CLI `plan_seconds` uses
the actual elapsed time.

`settings` retains the legacy `settings.env` format and ordered values/receipts,
including its historical `framework/tests/fixtures/adaptive/settings.py` comment label. The label
identifies the retained format, not the executable used. Writing to a file uses
`OUT.tmp` followed by replacement; write, close or replacement failure can leave
the temporary file, as in the reference. With no `-o`, output goes to stdout.

`report` keeps the complete legacy comparison: static prefill predictions by route
and tier state, controller decisions, decode-placement flags and final ordered
JSON. It chooses gzip by existence, whereas the arm reader chooses by readability.
It reads the plan before constructing a Machine; only the original OSError,
KeyError and ValueError classes suppress an unavailable model's route checks.
Other errors, including late zero division, retain their point of failure and
already printed text. The report's JSON is written last and can remain partial.
Report text and field names remain legacy formats, not new measurement claims.

## Python interfaces

| Area | Modules and boundary |
| --- | --- |
| Source metadata | `source.header`, `source.experts`, `source.files`, `source.command`: header/parser and legacy artifact operations with explicit filesystem, output and random ports |
| Device facts | `contracts`, `probe.device_profile`: immutable validated device records and pure report aggregation |
| Recorded I/O | `probe.recorded`, `telemetry.files`, `telemetry.recorded`, `telemetry.legacy`: injectable file access and legacy arm observations |
| Cost and replay | `model.decode_cost`, `model.routing_replay`: fitting/pricing and mutable tier/RAM/replay state |
| Prefill | `model.prefill_lane`, `prefill_workload`, `prefill_model`, `prefill_calibration`, `prefill_memory`, `prefill_feedback`: pure costs, geometry, calibration and explicit feedback callbacks |
| Calibration | `model.observed_calibration`, `probe.calibrate`: arm measurements, working set and complete ordered `fit` orchestration |
| Planning | `model.machine`, `plan.planner`, `plan.settings`, `plan.placement`: artifact-backed model, ranked plans and ordered settings |
| Request protocols | `bootstrap.client`, `bootstrap.requests`, `bootstrap.ports`: streamed legacy client timing and cold/warm/graph bootstrap requests with required transport, file, clock and device-number ports |
| Conversation and report | `loop.conversation`, `loop.report`: complete Controller consumer and recorded comparison command |
| Controller model | `loop.records`, `loop.model`, `loop.controller`, `loop.ports`: session-owned observations, re-planning and explicit transition/effect ports |
| Command adapters | `plan.recorded`, `plan.settings_command`, `plan.settings_recorded`: lazy artifact/host/output ports |

`Machine` borrows its input mappings and owns mutable replay/sink caches. Its
artifact reader and host fallback are injected. Command functions take explicit
read, output and clock ports as applicable. Source operations borrow their filesystem ports; import and adapter construction
perform no file or process work. These interfaces preserve existing
semantics; they are not a claim that every legacy artifact now has a versioned,
validated contract.

### Controller model and observation ownership

`ObservationSession(files=...)` owns the arm cache and incremental parser/offsets.
One session reproduces one isolated instance of the reference online module;
independent sessions are isolated rather than sharing that module's hidden
globals. Reusing a session explicitly shares its cache. A gzip key retains directory, skip, size and mtime; a live stream
keeps incomplete trailing lines and the original stale-offset behavior after
truncation. This is the legacy observation format, not versioned engine telemetry.

`Controller` requires `machine_factory`, `engine_factory`, `post`, `effects` and
`session` keyword arguments. The Machine factory can provide the artifact-backed
`model.machine.Machine`; its lazy artifact and host ports still apply. Engine,
HTTP, files, environment, clock and stdout calls occur at their original sites.
No native Engine, HTTP default, process discovery or live CLI is provided.
The old Engine's port-based termination fallback is not copied into the package.

Transitions retain the reference's mutation and failure behavior: `prepare` returns borrowed
lists, `note` mutates its supplied record, and `after` appends the turn before its
event/count update. A displacement catches only `OSError`. `close()` stops the
current engine without explicitly closing the event stream, as in the reference. Applications
must not treat this model-only extraction as a complete native resource owner.
The synthetic tests use fake engines and HTTP callbacks; they establish no real
server lifecycle, online performance or device integration claim.

### Request and conversation effects

`turn` requires a transport and nanosecond clock. `run_client`, `run_bootstrap`
and `run_conversation` require every consumed effect port; no network, engine,
host-reader or process implementation is selected implicitly. Their argument
records are unvalidated views of the old parsed arguments. Request URL/body/header
bytes, history aliases, token callback order and sparse first-observed token
timestamps are retained. This legacy timing protocol is separate from
`bench/final_chat.py`, which remains authoritative for formal benchmarking.

Bootstrap preserves cold chat followed by an uncached warm prefill, graph
sentence-cut search, and optional token-count differences. Its drive counters
are whole block-device reads, including unrelated readers; supplied stat/diskstats
ports do not establish per-process I/O. The graph command retains SystemExit(0)
after its output and before drive discovery. The script client opens output
before reading input and records per-turn Exceptions. The conversation consumer
catches only OSError; other failures do not acquire a newly invented finally
cleanup. Its normal path still stops the Controller then records done. These
interfaces are not a native resource-ownership adapter or a live CLI.

Recorded reference observations end at command return. A fixture may later close its retained
event stream during object collection; that test cleanup cannot mutate a returned
transcript. Tests separately preserve the original open event stream and internal
borrowed-reference behavior.

## Explicit owned server execution

`inferssd launch --execute` resolves typed Linux or Darwin inputs with lazy live
host facts, then starts one supervised server. Windows support is in progress:
`--dry-run` prepares the server command there, and owned execution is not
implemented yet. Owned execution requires an explicit policy and writes to a new
run directory: `--output-dir`, or `runs/<UTC time>` under the current directory.
The server listens on a loopback port:

```sh
inferssd launch --execute --platform linux --repo /path/to/inferssd \
  --model /path/to/model.gguf --profile muse --settings /path/to/settings.ssd \
  --runtime-policy /path/to/runtime-policy.json --output-dir /path/to/run/instance-001 \
  --port 8080 -- --seed 17
```

A policy has these fields; values below illustrate the format, not a recommended
budget for any model or machine. `host_reserve_bytes`, `max_compressor_growth_bytes`,
`max_swap_growth_bytes` and `runtime_seconds` are optional measurement conditions:
left out, they are not checked.

```json
{
  "platform": "linux",
  "lock_path": "/path/to/device.lock",
  "stop_path": "/path/to/STOP",
  "memory_budget_bytes": 17179869184,
  "host_reserve_bytes": 3221225472,
  "minimum_memlock_bytes": 0,
  "max_compressor_growth_bytes": 0,
  "max_swap_growth_bytes": 0,
  "runtime_seconds": 3600,
  "cleanup_seconds": 30
}
```

Linux requires a functioning systemd scope with cgroup v2, the declared exact
byte `memory_budget_bytes`, zero swap, readable ancestor OOM counters and memory peak.
The worker checks the actual envelope and the server's inherited scope and
memlock limit. Set any required memlock limit before invoking the command;
`minimum_memlock_bytes: 0` is appropriate only when the chosen configuration
needs no pinned-memory minimum. This command never raises system limits.

Darwin uses `platform: "darwin"`, explicit `--profile`/`--model` and the same typed
settings snapshot. Its budget is a **sampled soft server RSS threshold**,
with host free-plus-inactive memory, compressor and swap growth checks when the
policy declares them. It is
not a cgroup or a hard unified-memory cap. Both platforms monitor while the
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
resource check into a successful exit. `--result` and `--host-facts` are dry-run
options and are rejected by execution mode.

The Python `inferssd.engine.runtime.Runtime` requires explicit `LaunchInput`,
`LaunchSpec`, `Policy`, host and output ports. `Runtime.controller(...)` connects
the measured Linux Machine decision algorithm to typed snapshots and owned
Engines. Each selection reads measured placement/CPU/tier/prefetch/cost fields
directly, validates through the registry-defined parser, and records its sources.
It also selects the threadpools: `all` runs the measured thread count less one
CPU for the I/O thread, unpinned; `pin` and `split` pin one ggml thread per
measured compute CPU and keep the I/O CPU apart. Decode runs on the threads at
the lowest CPU IDs of the mask, so `split` requires its fast CPU group there.
Explicit CLI values, including `--threads` and `--cpus`, exclude incompatible
measured candidates; they are never silently replaced by a planner. Model
prefill predictions use the selected snapshot's actual route and native ubatch.
Cross-server prompt persistence is not configured by typed launch; the production
controller predicts a full prefill on switch and refuses `keeps_history=True`.
Each Engine receives its own immutable snapshot, threadpools and fresh native
files. No production path calls `Machine.env` or accepts the
old Engine environment-control string. Model/listener/parallel/thread/batch
arguments must use the explicit launch options so extra flags cannot silently
replace those owned values. `--threads` and `--batch-threads` select decode `-t`
and prefill `-tb` (range 1..256; batch defaults to decode, and unset counts keep
the server default). `--cpus` passes the same `-C`/`-Cb` mask with
`--cpu-strict 1`/`--cpu-strict-batch 1`; the counts then default to its size and
cannot exceed it, and it must not share a CPU with `cpu.io_cpus`. macOS cannot pin
threads and refuses `--cpus`. A positive `cpu.decode_participants` cannot exceed
the decode count. This preserves measured Mac choices such as t4/tb12 or t8/tb12
without an SSD setting. The launch receipt and argv record the counts and CPUs.
Extra flags cannot override fixed placement, threadpool, template, fit, cache or
offload decisions.

The production controller uses `NativeObservationSession` to consume the owned
`native-events.jsonl` stream. It requires `ssd.controller.begin` schema 1 and
versioned placement/prefill records, checks event sequence and field types, and
rejects failed layer observations. Human-readable server logs remain diagnostic
artifacts; production decisions do not parse them. The historical observation
session and log readers remain available for recorded imports and their goldens.
Static launch does not require debug channels. Adaptive launches select the
permitted trace/profile observations through `INFERSSD_DEBUG`; these do not
control kernels or placement. This interface does not complete Mac online
adaptation or live probe/bootstrap orchestration.

## Remaining integration boundaries

Live probe/bootstrap orchestration remains separate work. The native context
integration must be validated with this installed package on each target host; Python substitutes for OS effects
or synthetic model headers do not establish native execution correctness.
No new performance conclusion follows from these package tests.

Mac automatic framework planning and a common CPU executor/cache are not
implemented yet. The static Darwin production path rejects the adaptive factory. Importing these
modules starts no process, host probe or network call.

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

Production launch can use `--resources /deployment/engine.json` instead of `--repo`.
The descriptor identifies the executable, per-profile templates, optional native
probes, and supported configuration/telemetry contract. Paths resolve relative to
the descriptor, never the current directory. `--binary` and `--template` override
individual resources. `--repo` remains an explicit development adapter for the
product checkout, not a requirement of the installed wheel. See the
[deployment guide](../docs/engine-resources.md).
