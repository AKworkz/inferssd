# Impact-selected physical CI

The current operator entry is a maintainer's exact-head `/ci test` Comment review
on a PR from a branch in the source repository; see
[review entry and results](REVIEW-ENTRY.md). It uses the preparer's existing
permissions and reports in the private controller's Actions without a check App.
It executes the explicitly enabled devices (`cloud-5090`, `dgx-spark`,
`local-mbp`) and only the 512/decode256 performance tier. Omitted devices and any
additional required tiers remain coverage gaps. Local protocol checks are
separate from physical results, and these runs add no formal benchmark samples.
The guarded per-stage executors, device lease, STOP/quarantine, read-only source
and build copies, exact PID/birth/listener checks and disconnect guardians remain
authoritative.

## Plan and scope

`impact.py` rebuilds a deterministic plan from the complete PR file API listing,
including old names on renames, and exact base/head/merge-candidate SHAs. Unknown
and shared runtime changes require all three allowlisted devices and both models.
CUDA-only changes select the two CUDA devices; Metal/Mac changes select the Mac.
Explicit model graph files select that model. Current operator/format cases are
listed in the fixed `CASES` catalogue. Model-adapter and runtime-policy changes
also require their pinned descriptor/policy equivalence executables.

A selected cell requires a real model request and its related numerical gates.
A successful ADD row cannot satisfy MXFP4/Q8_0/Q4_K/Q5_K, normalization or SWIGLU.
Unsupported/zero executed rows fail. CUDA tier cases retain their normal-provider
CPU reference comparisons and original thresholds; no synthetic replacement runs
on the device. Test/control-only changes still select functional and numerical gates;
changes affecting runtime/placement/resources require base/candidate performance.
Ordinary performance uses the fixed 512-token CI prompt
(`bench/prompts/prompt-512.txt`) with decode 256. Only the explicit DeepSeek
compressed long-context rule adds 4K. Each cell runs the candidate three times
and the base once (candidate, base, candidate, candidate) on fresh servers and
records TTFT, native prefill and first/second 128-token and whole decode rates
for every run. No automatic completion retry or best-of selection exists. The
cell is `regressed`, `needs-review` or `passed` against the device's baseline
history, as described in [performance acceptance](PERFORMANCE.md). The history
is sealed only by operator preparation and covers only the 512 tier, so
4096-tier cells and jobs generated with `stage_selection.py` have no history and
report `needs-review`.

A baseline job is the one case with an empty diff: the operator measures each
new `main` commit on its own, built once, four runs per model, selecting every
device and both models with the 512 tier. Its plan has
`purpose: "baseline-history"`, and the trusted rules rebuild it only for an
empty diff and identical base, head and candidate.

## Separate App publication mode

The following protected-branch/App path is a separate optional mode. It is not
required by the deployed review operator and does not describe its authorization.

The source repository workflow provides an unprivileged hosted preview only. Its
plan is not authorization. The private `device-performance.yml` independently
reads the real PR, complete changed-file pages, current normal CI and reviews.
The unchanged admission contract requires an open, nondraft same-repository PR
from a protected `device-ci/approved/` branch, exact merge parents `[base, head]`,
and a different writable maintainer's latest approval on that exact head with a
standalone `[device-ci-approved:FULL_SHA]` line. An outstanding maintainer request
for changes blocks admission. The controller's own main must also be protected.

The hosted coordinator dispatches only the active private
`.github/workflows/integration-smoke.yml` on main using Actions-write permission.
The request asks GitHub REST for the exact returned run ID. An ambiguous response
fails without a second POST. The child is bound by workflow path/ID, main SHA,
fresh job ID, run name and attempt 1. Required selected-plan input is checked
against the root-owned job configuration and root broker manifest before opening
the lease. A plan-less readiness run cannot satisfy a selected child.

Only the hosted publisher receives the dedicated check App token. It runs after
failed admission/dispatch as well, and publishes a completed failure for a valid
requested source PR/head when artifacts or admission are missing. It rechecks
source, exact-head approval, complete changed paths and the latest hosted CI
before publishing success. The stable check is `InferSSD device performance`.
There are no required workflow path filters. Unselected cells are explicitly
`not-required`; selected missing, failed, unclean, regressed or needs-review
cells are red.

GitHub's [dispatch REST contract](https://docs.github.com/en/rest/actions/workflows)
supports `return_run_details`; [GITHUB_TOKEN dispatch exceptions](https://docs.github.com/en/actions/how-tos/write-workflows/choose-when-workflows-run/trigger-a-workflow)
allow this fixed child dispatch. [Required check semantics](https://docs.github.com/en/pull-requests/how-tos/merge-and-close-pull-requests/troubleshooting-required-status-checks)
explain the always-run external App aggregate. No `pull_request_target` candidate
execution is used; see [GitHub's security guidance](https://docs.github.com/en/actions/reference/security/securely-using-pull_request_target).

## Operator preparation (no deployment command)

1. Pin source base/head/candidate and the reviewed control commit. Export
   control and both private workflow templates with `export_controller.py`, then
   review/install through the existing root operator process. Read the device's
   actual installed `control-source.json` revision and whole-file hash. The
   private workflow commit is a separate identity from installed device control.
2. Obtain complete changed paths as JSON (for an operator diagnostic,
   `git diff --name-only --no-renames BASE HEAD` includes both sides of renames).
   Run `python3 -I bench/ci/impact.py --paths paths.json --policy
   bench/ci/policy.json --base BASE --head HEAD --candidate CANDIDATE --output
   plan.json`. The private PR entry recomputes this plan itself.
3. Prepare a fresh root job using the existing reviewed source/model staging
   mechanism. Preserve admitted readonly model identities; do not duplicate
   hundreds of GB. The manifest's `source_revisions` must exactly match required
   base/candidate sources. No runner can install or edit this manifest.
4. Prepare a local operator template with `device`, `host` (actual hardware, OS,
   resources), `control_revision`, `control_manifest_sha256`, `manifest_path`,
   `numerical` (actual CUDA0/MTL0 and cwd), and `sides.base/candidate`. Each side
   contains `revision`, `paths`, the existing `staged_build.run_build` build spec,
   optional sandboxed `alias`, and `models` keyed by the exact selected models.
   Each model contains the existing eight-token `smoke` spec, `settings` and
   `template` (`host_path`, `runtime_path`, `sha256`), `model_identity.shards`
   (`host_path`, `runtime_path`, recorded hash, `stat` with device/inode/bytes/
   mtime_ns/ctime_ns), and `benchmarks` keyed by selected tier. Each benchmark
   uses the existing `staged_benchmark.benchmark` argv/environment/cwd/port,
   timeout_s/request_timeout_s and honest os_page_cache_treatment contract.
   Candidate targets and the read-only build's `image_relatives` must cover
   every selected numerical executable. Both arms must use the same resource cap,
   model distribution and template. A model may instead carry a `launch` request
   and no argv or port, as operator performance jobs must: each side then
   resolves its launch on the device with its own typed launch (the one behind `inferssd serve`), from the
   machine characteristics a baseline job profiled (a PR job takes those of the
   device's most recent baseline job), and its settings and thread pools may
   differ from the other side's (see
   [performance acceptance](PERFORMANCE.md#launches)). Native snapshots,
   real runtime image mappings and root-observed resources are checked.
   Before any selected 256-token request, every selected performance model/tier
   runs both sides' eight-token preflight with that performance arm's environment,
   working directory and arguments. The operator pins a separate `preflight_argv`
   (a resolved launch gets one per tier); only the single private
   `--ssd-telemetry` file may differ from the timed argv,
   and both destinations are distinct canonical files within that side's scratch
   directory. Timed run k of a side uses the timed file name with `-r<k>`
   inserted before `.jsonl`, so every fresh server owns its own event file, and
   each run has its own `<side>-<model>-r<k>-benchmark-<tier>` stage. A candidate functional smoke is reused only when its complete
   launch matches this preflight. Native typed snapshots, owned
   runtime images, resources and immutable inputs must compare equal for every
   pair first, apart from each side's derived launch. Missing declarations or
   exact root stage authorization refuse before opening the lease; a derived
   launch is checked completely once both sides have resolved it. Timed arms
   retain their existing once-only checks and must still match their successful
   short preflight; no request is retried.
5. `python3 -I bench/ci/stage_selection.py --template template.json --manifest
   manifest.json --plan plan.json --policy bench/ci/policy.json --output fresh`
   produces three reviewable local files only. The generator derives the Linux
   stage allowlist from fixed trusted operations. The manifest carries the plan
   and config payload hash; the config carries the exact final manifest hash,
   avoiding a circular hash. Install these only through the existing root path.
6. A single operator-authorized integration dispatch can use
   `dispatch-inputs.json` without the PR parent. It validates one device's selected
   stages but does not publish an App check or claim PR admission. A full parent
   run consumes the per-device dispatch-input objects as `staged_jobs` and fails
   if any required device/model is absent. Do not pass an operator diagnostic off
   as an independently approved PR attempt.

The hosted parent has a three-hour collection deadline below its four-hour job
limit. Interruption and its always-run cleanup step request cancellation of the
exact recorded children; an API cancellation response is not worker cleanup.
Each selected child checks the parent's fixed workflow/main SHA/attempt and live
`control` job before opening the lease and every 60 seconds while running. The
private Actions-read token remains only in the trusted coordinator UID. Parent
termination, identity mismatch or an unavailable observation shuts down the
coordinator socket, leaving the existing independent guardian to drain workers
and retain the original lock. Missing positive cleanup remains red. A manual
root-staged operator run has no parent watcher and is explicitly labelled; its
artifact cannot satisfy a PR parent. Publication rereads the PR after raw
aggregation before rechecking the current base/head/merge and authorization.

The Mac broker labels a requested server stop `failed` even when the server
returns zero. Only that exact reason, dedicated job scope and positive empty
worker/domain cleanup is classified as a requested stop. Deadline, resource,
STOP, quarantine, nonzero server exit or incomplete cleanup is still rejected.

macOS control must include the read-only `resource_policy` probe field, so the
compared envelope is the actual continuously enforced root executor policy. Old probes fail closed. An unresolved model admission cannot be
converted to a skip, an alternate model, a smaller budget or a green result.

## App-mode activation prerequisites

The App publication mode requires a protected controller `main` and an
exact-head approval on an open source PR. A missing prerequisite blocks
activation; it is not permission to relax the contract. Root-stage readiness
runs use a separate manual authorization path. Model admission gaps on any
device remain explicit; a file inventory alone is not model admission.

## Read-only model views

GGUF originals owned by an operator account (for example, group-writable files
on a shared development host) may be used only through root-controlled per-file
read-only views. A selected root configuration must explicitly bind the
existing `model_admission_path`/`model_admission_sha256`,
`model_mount_proof_path`/`model_mount_proof_sha256`, and `prior_model_job_id`.
Those small metadata files remain protected and immutable; `protected()` is not
relaxed for GGUFs, settings, code, or proofs. The receipt checks the exact selected
shards, original payload hashes and source/view identities, current view stats,
and exact host readonly mounts without copying, chowning, or rehashing payloads.
The proof's `source_path` is recorded provenance; live checks cover the actually
consumed readonly view and never fall back to the original source path.
Missing or previously removed views refuse admission. This path does not
provision views or turn a manual mount proof into automatic PR authorization.

The existing StageSession protocol runs `model-readonly-before` as a build stage
before freeze and `model-readonly-after` as a test stage after native work. Both
use the fixed Python probe: GGUF magic readable, model/source O_RDWR denied, and
model-parent O_TMPFILE denied. Counts derive from the selected mount set. A
mount proof that only states a check is required is not a passed worker check.
The retained admission gate binds the two real probes to this job's manifest, handles, PID and
birth identities, scopes, original lease and positive guardian/job cleanup. The
hosted aggregate reconstructs that gate and rejects missing or stale evidence.
Ordinary root-owned model inputs and the Mac protocol keep their own
contracts. Root staging and actual device readiness remain separate
operator prerequisites; local contract tests establish neither.
