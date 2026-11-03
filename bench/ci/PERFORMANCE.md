# Device performance acceptance

The reference and candidate are built once from the pinned plan on the **same
physical device**. A result from another machine, even one with the same GPU
model, cannot serve as the reference. Both sides need the same model shards,
prompt/template, launch request, machine characteristics, memory and resource
budgets, toolchain and recorded OS-cache treatment. Each side's settings, thread
pools and server arguments are derived by that side's own launch strategy, as
described in [launches](#launches). Direct GGUF and unrestricted H2D remain
mandatory. The protected broker owns source/build/observation evidence;
candidate output cannot certify itself.

## Runs

Each selected cell (device, model and prompt tier) runs the PR's merge
candidate three times and the base once, in the order candidate, base,
candidate, candidate. The base does not run first: whatever effect the first
timed run after the short preflight requests has falls on one of three candidate
runs, not on the single base run. Every run starts a fresh server
and makes one completion of the fixed 512-token prompt with 256 output tokens.
All runs are kept: nothing is retried and no best run is chosen. A failed run
makes the cell `failed`.

For every run the comparator re-reads the raw SSE and timestamps, exact
request/template IDs, the runtime receipt and cleanup, and checks that the runs
ran in the declared order on one clock, each on its own server, under equal
measured conditions. The controller binds the enclosing artifact to the current
run/attempt, reviewed source identities and reviewed device config.

## Launches

Every server is started the way `inferssd serve` starts it on a user's machine.
A user profiles a machine once and then reuses the characteristics file, and
device CI does the same: only the baseline job of a `main` commit (see
[baseline history](#baseline-history)) profiles the device. After building that
commit, it runs the commit's own `inferssd profile` once, with its probes,
which measures every section afresh and writes the characteristics file in the
framework's own format at `/scratch/machine/machine.json` in the job's Linux
sandbox (`work/machine/machine.json` under the Mac job root). The controller
keeps the file as the stage printed it and records its SHA-256 and the status
of each section; a section that failed is reported and left out, as a user's
launch leaves it out. The operator stores the file with the device's baseline
history.

A PR job does not profile. The operator takes the characteristics file of the
device's most recent baseline job from the history, seals it into the job
configuration like the history itself and names its SHA-256 in the operator
envelope (`machine_characteristics_sha256`). The root stager writes it
read-only where the sandbox reads it: `/models/machine.json` in the Linux
sandbox, `machine.json` in the Mac job's input root. Before any launch is
resolved, the controller checks that the staged file holds exactly the sealed
bytes. Nothing a PR stage writes in scratch is ever read as characteristics,
and the comparison carries no re-profiling noise. A PR job on a device that has
no baseline characteristics yet cannot be prepared: a baseline job must run on
that device first.

Then each side resolves each model's launch with its own framework: `inferssd
launch --dry-run --machine` that file, with the runtime facts (usable CPUs and
GPUs, available memory, the drive holding the model) read in the sandbox by the
side's own code, the model's GGUF header and the job's memory budget as the
runtime policy. The broker runs the server stage in its own resource scope, so
the launcher's checks for a scope of its own are stated rather than probed. The
resolution must name the characteristics hash and run the side's own server
with its generated settings file; in a PR job, every run's receipt and the
hosted replay check that hash against the sealed file. The broker then starts
exactly the resolved argv for the functional request, the short preflight and
every timed run; only the event file differs per server. A PR that changes the
strategy is therefore measured with the settings it derives.

The reviewed launch request (model profile, port, any extra server arguments
and any context or batch size it pins) and its settings file are the same for
both sides. The settings file sets only `expert.source`: everything the strategy
chooses (thread pools, CPU placement, placement cost priors, the prefill route
and pass size with a CUDA GPU, the host margin on shared memory, the VRAM budget
when no GPU is usable, the Mac memory budget) is derived, and everything it
leaves alone (the RAM cache size, the VRAM tier in automatic mode, the read
backend and numerics) stays at its default, decided by the engine. A resolution
in which the settings file sets anything else is refused. Every launch runs
without a prompt cache (`--prompt-cache off`), so no timed run starts from an
earlier run's prompt.

Each run's receipt records its side's derived launch: every setting with its
value and origin (`strategy` with the strategy's reason, `launcher` for a
registry default or the launcher's model profile, `user` for the reviewed
settings file), the thread pools, the strategy's server arguments and the argv.
The receipts of one side must agree. The comparison then allows the derived
settings and argv to differ between the sides and still requires the same model
files, machine and characteristics, template, prompt, launch request, token
budget, resources, OS-cache treatment and loaded-library set. The cell lists
what the PR's launch sets differently, and the summary shows both sides' thread
pools and every changed setting.

Five metrics are measured:

- Native prefill = server `prompt_n * 1000 / prompt_ms`.
- TTFT = completion request to first token, separate from native prefill.
- First-half decode = `127 / (t128 - t1)`.
- Second-half decode = `128 / (t256 - t128)`.
- Whole decode = `255 / (t256 - t1)`.

All metrics must be finite and positive.

## Baseline history

The base side is also measured before the PR runs. For a new commit on `main`
whose files differ from the most recently measured commit in a path the impact
rules give a performance cell (the engine, the framework, or unknown code), the
operator queues one baseline job. On every enabled device, that commit is built
once, profiles the machine once and runs each model four times, each run a
fresh server with one completion, after the same short preflight requests. A
commit that changed only tests, CI control or documentation is recorded as
`not-needed` and runs no job. The PR jobs on it compare against each cell's
most recent record, and the in-run base still has to agree with that record
(the anchor check below). Jobs run one at a time in the order they were queued,
so a PR queued after a merge normally finds its base already measured. Each clean cell of a
baseline job becomes a record in the device's baseline history: the four values
of every metric, their natural logs, the generated-token hash, the history
conditions, the SHA-256 of the characteristics file its runs launched from, the
job and run IDs, the build identity and the date. A cell whose Mac host check
flagged a run is reported and not recorded.

Generated tokens are recorded for every run but, for now, not required to
match. Where the CUDA GPU tier computes experts, the experts it holds when each
one is needed depend on timing, so runs of one build can generate different
tokens. A difference is shown in `output_workload` and does not make a cell
need review (`REQUIRE_SAME_TOKENS` in [compare.py](compare.py)).

The history lives in the `baseline-history` directory next to the reviewed
device profiles (`inputs.json`) on the preparation host, one read-only file per
device, model, tier, commit and history conditions. The characteristics file a
baseline job profiled is stored next to the records, once per device, in
`<device>/characteristics/<sha256>.json`, named by its SHA-256 and written only
from a verified baseline report; PR code cannot write there. The history
conditions are the machine's hardware identity, its OS, the model files, the
chat template and the job's machine policy: the memory budget (the cgroup
`memory_max` and the memory lock limit on Linux, the executor's memory budget on
the Mac) and the CPU set (`cpus_allowed_list` on Linux; the Mac pins none), as
the device template's `host.resources` states them. A change to the budget or
the CPU set starts a new history. Settings are not part of the conditions: each
commit derives its own launch within that policy, so the history of `main`
stays usable for a PR whose strategy derives different settings. Nor are the
characteristics: a later baseline job may profile the machine again.
[history.py](history.py) lists, verifies and prunes it:

```sh
python3 -I bench/ci/history.py --root baseline-history verify
python3 -I bench/ci/history.py --root baseline-history prune --keep 50
```

`verify` also re-hashes every characteristics file and refuses a record whose
file is missing or changed; `prune` removes a characteristics file once no kept
record names it.

Every prepared job seals its devices' records into each device's job
configuration and names their hash (`baseline_history_sha256`) in the operator
envelope; a PR job seals each device's characteristics file the same way. The
device job decides with exactly those records and records the same hash; the
hosted replay requires the envelope, the sealed configuration and the device
report to agree before it recomputes any decision.

## Decision

All values are compared as natural logs, signed so that a larger value is
worse: `x = ln(TTFT)` and `x = -ln(rate)` for the four throughput metrics. For
one metric of one cell:

- `h` is the mean of the base commit's four history runs (`n_h = 4`), `a` the
  in-run base, and `c̄` the mean of the three candidate runs.
- `σ` is the pooled within-commit standard deviation of every history record of
  that device, model and tier under the same history conditions:
  `σ² = Σ_c Σ_i (x_ci − x̄_c)² / Σ_c (n_c − 1)`, with `df = Σ_c (n_c − 1)`
  degrees of freedom. One record gives `df = 3`; each measured main commit adds 3.
- `t` is the one-sided Student-t quantile for `df` whose upper tail equals a
  normal 3σ tail (P = 0.99865). A fixed table in `compare.py` gives the same
  value everywhere: 9.219 at df 3, 4.905 at 6, 3.765 at 12, 3.545 at 16, 3.122
  at 64 and 3.060 at 128 (a df between entries uses the nearest lower entry),
  and the normal value 3.0 above 128.
- Anchor check: if `|a − h| > max(ln 1.02, t·σ·√(1 + 1/n_h))`, the device has
  changed since the baseline job, and the cell is `needs-review`. The 2% floor
  allows the drift between jobs hours apart, which the runs of one job do not
  show: on the RTX 5090, Muse TTFT varied 0.3% within a baseline job and its
  in-run base differed by 1.3% some hours later.
- The base estimate is `b`, the mean of the four history logs and `a`; the
  difference is `d = c̄ − b`, with standard error `SE = σ·√(1/3 + 1/5)`, and the
  threshold is `T = max(ln(1 + floor), t·SE)`. `floor` (1% for every metric) is
  set in `thresholds` in [policy.json](policy.json).

Only whole decode, TTFT and native prefill decide a cell. The first and second
128-token decode rates are sub-windows of whole decode; they are measured,
compared and shown for information only.

| Result | Meaning | Check behavior |
| --- | --- | --- |
| `passed` | For every deciding metric `d ≤ T`, the history matches the history conditions, the in-run base agrees with its history and the host check flagged nothing. | Passes. It means no slowdown beyond measured noise, not proof that none exists. |
| `regressed` | Some deciding metric has `d > T` and every candidate run is slower than `b`. | Blocks acceptance. Any other review reasons are listed with it. |
| `needs-review` | A deciding metric has `d > T` but the candidate runs disagree; the base has no history under the history conditions; the in-run base differs from its history; or the Mac host check flagged a run. | Blocks acceptance until a person reviews the evidence. |
| `failed` | Missing or invalid evidence, mismatched conditions, incomplete cleanup or a failed run. | Blocks acceptance; fix the evidence or implementation first. |

Without base history under the history conditions no regression can be
established, so such a cell is `needs-review` even when it is clearly slower.
The history covers the 512 tier of operator jobs only, so a 4096-tier cell, and
any cell of a job prepared without the history (for example with
`stage_selection.py`), is `needs-review`. A base run that differs from its
history also blocks `regressed`: the comparison would no longer describe the
device the history measured.

A deciding metric with `d < −T` is labelled `improved` (faster beyond noise);
within `±T` it is `within noise`. The label is part of the cell result, shown in
the summary and replayed like the decision, but it never changes the cell
status: a PR that is faster everywhere still just passes.

With no real change, a simulation of the A/A case (all runs from one normal
distribution with a 2% spread, the anchor check and the 1% floor) gives these
rates per deciding metric per cell:

| History df | False `regressed` | Statistical `needs-review` | False `improved` label |
| ---: | ---: | ---: | ---: |
| 3 | 0.081% | 0.29% | 0.13% |
| 6 | 0.113% | 0.28% | 0.13% |
| 12 | 0.126% | 0.28% | 0.13% |

The false `regressed` rate stays below the normal 3σ tail of 0.135% because a
regression also needs every candidate run to be slower; a cell has three
deciding metrics, so its false `regressed` rate is below 0.4%.

When cells are combined, the most severe result wins: `failed`, then
`regressed`, then `needs-review`, then `passed`. A device job succeeds only
when all of its cells passed. A baseline job's cells report `baseline` and make
no decision; a job whose cells are all `baseline` succeeds.

A `regressed` or `needs-review` cell is never cleared by a later run of the same
candidate. A reviewer decides whether it shows a software regression or an
identified environmental cause; a code fix is a new candidate with a new run.
When the machine, its OS, the model files, the template, the memory budget or
the CPU set changes, PR cells are `needs-review` until the next main commit's
baseline job measures the device under the new history conditions. Other
resource limits, loaded libraries and the launch request are compared between
the runs of a job instead; a difference there is invalid evidence.

## Base build cache

A Linux device whose profile sets `"base_build_cache": true` keeps the build of
each baseline job: after the broker freezes it, it copies the frozen tree into
its own state directory (`<state_root>/build-cache/<key>`), read-only, keyed by
the commit, the side-independent build configuration and the reviewed
toolchain. It keeps the three most recently stored or used entries. The history
record names the build's freeze-manifest hash and cache key, so a PR whose base
commit has a record offers that build for its base side. When the job opens, the
broker re-hashes every cached file while copying it into the job, exactly as a
freeze would leave it; any mismatch or a missing entry is refused and the base
side is built as usual. The candidate always builds. The report states whether
the base build came from the cache. The Mac does not use the cache.

## Time budget

From recorded runs, each device job has a fixed part for two builds, numerical
checks, short requests and preflights, and each timed run of both models adds a
fixed amount:

| Device | Fixed part (two builds) | One run, both models | PR job (4 runs) | Baseline job (one build, 4 runs) |
| --- | ---: | ---: | ---: | ---: |
| cloud-5090 | about 18 min | about 7.4 min | about 47 min | under 47 min |
| dgx-spark | about 15 min | about 11.0 min | about 59 min | under 59 min |
| local-mbp | about 12 min | about 11.2 min | about 57 min | under 57 min |

A PR job whose base build comes from the cache saves one build. A baseline job
runs every device at once, as one child job per device. A baseline job also
profiles the machine once (one to three minutes on a new machine, and at most
about three minutes per model drive for the SSD probe; the stage is bounded at
30 minutes); a PR job does not profile. Every job resolves each side's launches
(seconds each). All stay within the 120-minute device job limit, and every run
stays far below the 7200-second Mac stage limit.

CI samples have `purpose=ci` and are not formal benchmark samples. The formal
benchmark ([FINAL.md](../FINAL.md)) runs each prescribed 512/4K cell once on the
release source, retaining all attempts without choosing a best result.

## Legacy reference adapter

The legacy reference revision (`LEGACY_REVISION` in `reference.py`, also the
fixed baseline of the checkpoint workflow) does not implement the
typed-settings launcher. Its `bench/native_common.py` has no `typed_snapshot`
or `native_launch_policy`, and its `server_argv` does not emit `--ssd-settings`.
The CI driver requires `typed_settings` for both builds, so it cannot compare
that reference directly to a typed-settings candidate.

The controlled adapter in `reference.py` belongs to the protected controller,
with an exact reference-SHA allowlist. Integration must launch the **unchanged old binary** using its
actual reviewed argv/environment, launch the candidate through the new interface,
and record both original launch policies plus an audited mapping to the same
semantic settings and observed budgets. No reference values may be guessed, and
adding a fabricated `typed_values` field is insufficient. Values whose behavior
or resource meaning cannot be aligned must make that cell incomparable. The
adapter validates an explicit binding; it does not invent or approve a real host's
mapping. The driver/build owner must wire the reviewed binding into both builds.

Without an explicit binding, the comparator requires equal loaded-image basenames.
Directory moves and renaming/splitting libraries can legitimately change these.
Use an exact reviewed base/candidate **library-role mapping** together with each
side's actual loaded hashes, rather than globally ignoring image identity. Until
that mapping and the reference adapter are installed and verified, a comparison
against the legacy reference must remain blocked. PRs whose base already uses
typed settings take the equal-policy path without this adaptation.

### Reviewed binding contract

The protected runner config may carry `comparison_bindings` with exactly `tiny`,
`deepseek` and `muse`. Each model's binding is a JSON object with:

- `schema_version: 1`, `kind: "legacy-to-typed"`.
- `semantic_settings`: all keys of the reviewed registry generation, including inapplicable platform
  defaults. Values must have the registry's actual JSON types.
- `sides.base` and `sides.candidate`: full `revision`, `profile_sha256`, complete
  `launch_policy`, `images` and `build` objects. The base must be exactly
  `LEGACY_REVISION`; candidate must match the plan's **merge candidate**, not merely
  the PR head. Profile hashes use `native_profile_sha256`'s canonical JSON algorithm.
- `legacy_mapping`: every semantic key has its exact `value`, nonempty
  `explanation`, and nonempty `inputs` list. No mapping is inferred automatically.

Each launch policy preserves the complete original `args`, `environment`,
`template_sha256` and `alias_sha256`. Candidate also carries the actual complete
`typed_values`, exactly equal to `semantic_settings`; legacy must not claim typed
telemetry. Every observed wrapper revision, profile digest and launch-policy
value must equal the reviewed side. Canonicalization removes only the explicitly
approved launch representation and image-layout differences. Hardware, model,
template/alias hashes, all remaining effective settings, actual resource budgets,
ownership core and cache treatment remain comparable without relaxation.

Mapping input forms are explicit:

| `kind` | Additional fields | Verification |
| --- | --- | --- |
| `environment` | `name`, `value` | Exact original launch environment entry. |
| `argument` | `index`, `value` | Exact original argv element. |
| `effective-setting` | `name`, `value` | Exact wrapper effective setting; nested native metadata cannot be used here. |
| `source-default` | `revision`, `path`, `sha256` | Operator-reviewed meaning of an exact legacy source file; never presented as runtime telemetry. |

Call `verify_source_defaults(binding, source)` before launching the base. It
checks the fixed checkout revision and compares every cited file against both
the reviewed hash and that revision's actual Git blob. It runs no candidate
program. A nonempty explanation is an audit trail, not automatic proof that the
reviewer's semantic interpretation was correct; an operator must review it.

`images` is the complete actual loaded-library list, with `{path, sha256, role}`
per entry. Paths are absolute and hashes must match actual mapped-library evidence.
No unlisted or missing library is accepted. Stable role sets must match across
sides; a reviewed split can map several libraries to one role. The executable is
bound by the full profile/build identity and is not added to the library list
when the native wrapper excludes it.

`build` has `server_target`, `binary_relative`, `image_relatives`, and
`template_relatives` (a nonempty subset of the three model keys; the Muse checkpoint
needs `muse`). Paths must be safe source/build relative paths.
`build_spec(binding, side, revision)` returns a copy for the trusted
build owner; it never edits source or chooses defaults. Supported explicit server
targets are `llama-server`, `inferssd`, and `inferssd-server`.

The binding must come from the reviewed config digest, never a candidate-provided
artifact. Comparison support alone does not enable physical execution: deployment
still needs real protected observation and the driver/build integration.
