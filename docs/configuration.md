# Typed SSD settings v1

`engine/ggml/src/ggml-ssd/settings/registry.def` is the authoritative settings registry.
The C++ schema/help and packaged Python schema are generated from that table;
the cross-language contract compares the actual C++ output with the installed
Python resource. There are no environment mirrors or legacy-name aliases.

| Key | Kind / unit | Default literal | Producer | Scope | Backend |
|---|---|---|---|---|---|
| `expert.source` | paths / path | `[]` | plan | all |  |
| `cpu.io_cpus` | cpus / CPU ID | `empty` | plan | all |  |
| `cpu.decode_participants` | integer / compute participants | `0` | plan | all |  |
| `prefetch.depth` | integer / predicted experts | `2` | plan | all |  |
| `place.costs` | costs / named us and GB/s terms | `empty` | plan | all |  |
| `prefill.cpu_max_tokens` | integer / tokens | `4294967295` | plan | all | CUDA |
| `vram.budget` | budget / bytes | `auto` | plan | all |  |
| `vram.max_slots` | integer / slots | `0` | plan | all | CUDA |
| `vram.reserve_mib` | integer / MiB | `0` | launch | all | CUDA |
| `host.headroom_mib` | integer / MiB | `512` | plan | all |  |
| `host.budget_mib` | integer / MiB | `0` | plan | darwin, windows |  |
| `model.engram_prefetch` | integer / boolean | `0` | model | all |  |
| `model.hc_fused` | integer / boolean | `1` | model | all |  |

These are conservative parser defaults and retained policy priors, not claimed
per-device performance optima. The framework must provide its measured CPU,
cache and placement configuration. The platforms are `linux`, `darwin` and
`windows`; a scope is `all` or the platforms it lists. A backend names the GPU
backend whose expert tier a key configures: the key parses on every platform, and
an engine built or running without that tier refuses it when it is set. Expert compute runs on
ggml's threads (`-t`, `-tb`, `-C`, `-Cb`, `--cpu-strict`; see
[how it works](architecture.md#threads)); `cpu.io_cpus` places the executor's
own I/O, prefetch and watch threads, and macOS, which cannot pin threads, only
validates it. Zero decode participants leaves the measured/default policy; a
positive value counts the leading ggml threads of each decode phase, and a
decode call with fewer threads fails. Non-applicable platform keys cannot be
explicitly set. The `vram.*` keys size the CUDA expert tier. `vram.budget = auto`
gives the tier all the room
the GPU admits; on a GPU that shares the host's memory, such as GB10, that is
the host memory envelope below less the slots the next-layer prefetch lands in,
which are all the RAM cache keeps beside a tier there (with `vram.budget = 0`
the RAM cache takes the envelope). Auto gives no tier where no CUDA backend
runs. On Vulkan and Metal, a GPU that shares the host's memory takes the heavy
experts of a prompt pass through the [GPU lane](architecture.md#the-gpu-lane),
which keeps nothing across layers, and `vram.budget = 0` leaves them on the CPU.
On a CPU-only build, or with another GPU backend, the GPU runs the dense layers
while the CPU sink computes every routed expert, and an explicit nonzero budget
is an error. The lane holds no expert across layers, so for it a nonzero budget,
like auto, only turns it on.

`vram.reserve_mib` is VRAM that prefill and decode leave free for other processes.
On a discrete GPU it is kept against what this process holds, not against what the
card has free: another process using the reserve takes nothing from the tier, and
one going beyond it fails its own allocation. When other processes release memory
beyond the reserve, the tier grows into it. The tier also leaves 64 MiB above the
reserve for the driver's own memory. A reserve too large for the model fails at
launch. The framework chooses 1024 MiB when other compute processes are on the GPU
at launch, and 0 otherwise; a setting of yours replaces that choice. On a GPU that
shares the host's memory, the reserve adds to the host memory floor below, and the
tier gives memory back under pressure from any process.

The host memory ledger has two terms on every platform. The envelope is a finite
limit on this process's memory less a 512 MiB reserve: on Linux the tightest
cgroup v2 `memory.max`; on Windows a memory limit of the process's job, when the
job sets one, against commit; and on macOS and Windows `host.budget_mib` against
the process's own memory (Windows PrivateUsage, macOS phys_footprint). With
`host.budget_mib = 0` and no job limit, the envelope is fixed at startup as the
process's use plus the memory then available, so its room starts at the
available memory less the reserve. The system term is the memory available now
less the floor: Linux MemAvailable, Windows the smaller of available physical
memory and available commit, macOS free, file-backed, purgeable and reusable
pages. The RAM expert cache admits only under a readable envelope. On a GPU
that shares the host's memory (unified memory, such as GB10), a CUDA allocation
is not charged to the cgroup, so the Linux envelope counts the CUDA expert
tier's storage as used beside the cgroup's own usage: the tier and the RAM
cache share one envelope, and the tier also stays within MemAvailable less the
floor. An allocation that does not fit, such as a compute or work buffer, a
scratch buffer or a server checkpoint, first takes room back from the expert
caches: on unified memory from the GPU tier, whose blocks also answer any CUDA
allocation that would fail while no layer holds them, then from the RAM
cache's coldest slots.

## File and CLI contract

```text
inferssd-settings 1
cpu.io_cpus = "3"  # source=derived; input=measured CPU placement
cpu.decode_participants = 2
expert.source = ["/path/to/index # 1.tsv", "/path/to/other.tsv"]
place.costs = "cpu_fixed_us=204,link_gbps=12.3"
vram.budget = auto
```

Integers are unsigned decimal literals, not quoted strings. Source lists are
JSON arrays of nonempty UTF-8 strings; `#` inside a quoted path is preserved.
CPU lists accept ordered comma-separated IDs and inclusive ranges. Cost tables
accept the eleven schema-listed `name=value` terms separated by commas or
whitespace; values must be finite, nonnegative and at most 1e6, with positive
`link_gbps`. Omitted cost terms retain the executor priors until measured.

The precedence is command line > file > registry defaults. A launch
fills keys still at their registry default with choices for the machine (see
[machine profile](machine-profile.md#launches)) before it writes the native
file. A file must have
exactly the v1 header. Duplicate file keys, unknown keys, wrong types, bad ranges
and dependency errors fail with key/source context; later CLI occurrences win.
Trailing file comments are retained exactly after the `#` byte. CLI overrides
receive `origin=cli`; file and default values retain their own location/receipt.
Each context owns a copy of the immutable snapshot; no mutable global or ambient
environment is consulted during parsing.

The C++ `ssd::settings` API returns `[[nodiscard]] result<T>` and status/error
values. No parser, loader or telemetry error is thrown across its public boundary.
Typed accessors return const pointers, including null for an absent/wrong kind,
and never perform an unchecked `std::get`. `load` reads a file; `parse` accepts
explicit content, source name, overrides and platform. `schema_json`, `help`,
`cli_option` and `cli_override` use the same registry. The startup retired-prefix
guard accepts an explicit list of environment names and rejects the declared old
prefixes plus `TASK[0-9]+_`; it accepts `TASKRC` and `TASKDATA`.

`inferssd serve` and `cli` take a file with `--settings settings.ssd` and single
values with `--set key=value`:

```sh
inferssd serve -hf REPO --settings settings.ssd --set prefetch.depth=4
```

The launch checks both before it starts the server, and an invalid file or value
stops it with the key and its source. The table above lists every setting; the
machine-readable schema is `framework/src/inferssd/configuration/schema.json`.
Settings are never translated to environment variables. The launch
passes the settings file to the server with `--ssd-settings`, and each setting
with a command-line origin as the matching `--ssd-*` option. CPU thread counts and
placement are server threadpool arguments, not settings; the launch chooses them
for the machine (see the [framework guide](../framework/README.md)).
