# Physical-device CI

InferSSD's normal CI runs on GitHub-hosted runners. Checks that need real
hardware (CUDA and Metal builds, numerical operator checks, real model requests
and a base/candidate performance comparison) run separately on
maintainer-operated devices. This directory contains the controller code, the
per-device stage brokers and their contracts.

This repository has **no physical self-hosted runner**. Its
`.github/workflows/device-performance.yml` only previews the impact selection on
a GitHub-hosted runner. Physical execution is coordinated by a separate private
controller repository, `wici-ai/inferssd-device-ci`, which receives a reviewed
export of this directory. A different workflow could otherwise request the same
runner labels and bypass the controller's checks; runner labels and fork-workflow
approval are not access controls. GitHub documents the persistent compromise and
credential risks of self-hosted runners in its
[secure use guidance](https://docs.github.com/en/actions/reference/security/secure-use#hardening-for-self-hosted-runners).

## Authorization

Device checks are authorized by a maintainer, for PRs from branches in this
repository and from forks alike. A maintainer with current write access submits a Comment review whose
complete body is `/ci test`; GitHub binds the review to the exact current head.
The operator also measures every new `main` commit on the devices; PR results
are compared with that history. See [review entry, results and limits](REVIEW-ENTRY.md).

A fork PR's code reaches the devices only after a maintainer has reviewed its
exact head; read the diff before submitting `/ci test`. Admission requires the
latest successful hosted CI run for that head. GitHub links no PR to a run from
a fork, so that run is bound by the PR's fork and exact head instead.

An authorization is valid only for its exact head, base and GitHub test merge.
A new push, base change, dismissal, later `/ci cancel` or change request, or loss
of write permission withdraws it. Ordinary issue comments, labels and prose are
not commands, and a test authorization does not approve a merge.

## Execution paths

- **Review-authorized operator (deployed).** `review_operator.py` runs on a
  trusted preparation host. It prepares exact sources with
  [operator preparation](OPERATOR-PREPARE.md), stages them on each enabled
  device through a root-owned stager, and dispatches the private controller's
  `device-performance.yml`. The controller runs the
  [impact-selected](SELECTIVE-CI.md) functional, numerical and 512/256
  performance checks with `StageSession` on the explicitly enabled devices, and
  reports in its Actions Summary and artifacts. It does not publish a check on
  the source PR.
- **Manual readiness and checkpoint workflows.**
  `controller/integration-smoke.workflow.yml` builds one reviewed source and makes
  one short request per selected model on one device
  ([instructions](controller/INTEGRATION-SMOKE.md)).
  `controller/checkpoint.workflow.yml` compares a fixed legacy baseline with one
  reviewed candidate on one Muse 512/256 cell
  ([instructions](controller/CHECKPOINT.md)). Neither replaces the full device
  gate.
- **App-published check (inactive template).** A dedicated check App would
  publish `InferSSD device performance` on the source PR. Its admission and
  finalization contract still requires full end-to-end integration and native
  gates on all devices, so it stays inactive. Do not require or publish its check
  because a manual workflow or the review operator succeeded.

The controller requires the reviewed `INFERSSD_CI_ENABLED=true` activation
variable. That variable alone does not enable the App publication path or
establish coverage of every device. This code installs no runner and changes no
host or GitHub settings.

## App-mode source admission

The inactive App mode uses a stricter admission. A maintainer supplies both a PR
number and the full SHA, and admission requires:

- An open, nondraft same-repository PR into `main`, from a protected
  `device-ci/approved/` staging branch, still at the supplied SHA.
- An independent maintainer's latest non-comment review approving that SHA,
  with a standalone review-body line `[device-ci-approved:FULL_40_CHARACTER_SHA]`
  explicitly authorizing physical execution. An ordinary code review is
  insufficient. Dismissal, requested changes, lost write permission or a
  different commit blocks execution. Author association alone grants nothing.
- A current merge candidate whose parents are exactly the pinned base and head.
- The latest normal CI run and attempt for that head and base, with the expected
  workflow ID and path, completed successfully. An older success is not used.
- A controller dispatch on the protected workflow and ref, with current actor
  permission. Controller workflow SHA, run ID and attempt, source origin, review
  IDs, base/head/candidate and the complete changed-file listing enter the plan.

A maintainer can import a reviewed external patch into a protected staging
branch, preserving its authorship, and obtain a new exact-SHA review. This
transfers trust explicitly; reviewing code does not make arbitrary code harmless.

There is no public `workflow_run` entry: that trigger can receive secrets or
write tokens independently of the preceding run, so it is not a trust upgrade
for that code. Manual dispatch can target non-default refs, so both job
conditions and the controller verify the protected ref. See
[workflow events](https://docs.github.com/en/actions/reference/workflows-and-actions/events-that-trigger-workflows).

## Execution boundary

No candidate is checked out on a runner. Hosted preparation exports immutable
Git objects as `source.bundle`, validates both candidate parents, and binds its
hash into the plan. The runner checks out only the protected controller at its
exact workflow SHA. Source-read credentials stay on the hosted preparer. Device
jobs have only the controller's read-only Actions/content token; no publishing
App key or token reaches them.

`isolated_device.py` invokes an operator-pinned, root-owned broker client with a
fixed JSON protocol. The broker, client, proof file and all their ancestors must
be root-owned, not symlinks, and not group/world writable. Whole-file hashes
must match the reviewed per-host config. A fresh nonce binds describe/run
replies. The proof and describe response must identify separate controller and
execution UIDs, a root supervisor, sandboxed execution stages, and the result of
every required negative check:

- The candidate cannot read the controller's home directory, runner/SSH
  credentials or host service sockets, modify control or evidence, or
  signal/trace the coordinator.
- No sudo or privilege escalation, Docker socket, external network, or escape
  through user systemd/DBus. Model and source inputs are read-only; scratch is
  bounded.
- Disconnect cleanup, the original lock inode, STOP quarantine and verified reset.

These checks are host-specific: a proof covers only what it tested on that host.
The trusted coordinator stays outside each candidate stage's sandbox. Running
the entire driver and candidate as one worker UID would permit evidence forgery
and is refused by the protocol. MemoryMax, swap controls, process-group cleanup,
inherited flock descriptions and ephemeral runner registration address resources
and lifecycle; they are not malicious-code isolation. macOS Metal execution needs
per-stage OS isolation too. A disposable runner registration does not reset the
host or isolate the operator's own account.

`controller/broker-unprovisioned.py` deliberately exits 78; it is not a sandbox.
A config boolean or synthetic test result cannot replace native negative checks.
If a host cannot prove the boundary, leave its runner and overall activation
disabled. The device brokers are documented in
[linux-sandbox](linux-sandbox/README.md) and [macos-sandbox](macos-sandbox/README.md).

The full-gate `isolated_device.py` contract expects `describe` and `run`. The
staged workflows instead use `job.open`, `stage.start/poll/probe/client/output/stop`,
`job.freeze` and `job.finish` on one persistent connection; wiring the full-gate
contract to those stages is not done yet. In that contract, `run` receives the
exact plan, device/config path, source bundle, pinned control root and output
directory, and accepts no arbitrary shell command. It must validate and copy
inputs before candidate access, enforce the per-stage boundary, hold the existing
lease until all owned work is gone, and emit the complete device result plus
protected isolation/cleanup evidence. Its daemon must survive controller loss and
quarantine uncertain cleanup rather than release the lease.

## Tests and comparison

Inside that boundary the driver builds the exact base and candidate once each,
runs the reviewed small numerical/lifecycle CTest set on both, and compares short
greedy token IDs and pre-sampling top-five records. The deployed review operator
measures DeepSeek/Muse 512/256 with three candidate runs and one base run, in
the order candidate, base, candidate, candidate, each run a fresh server. The
base commit was measured four times before, by the baseline job the operator
runs for every new `main` commit. The impact plan adds
4K/256 only for DeepSeek compressed long-context changes; the review operator
runs the 512 tier and reports any required 4K cell as a coverage gap. Both
builds precede timing. Only the baseline job profiles the machine, once per
`main` commit and device with that commit's `inferssd profile`; the operator
stores the characteristics file with the device's baseline history. A PR job
takes the file of the device's most recent baseline job, sealed into the job
and staged read-only by root, and cannot run on a device that has none yet.
Each side starts its servers the way `inferssd serve` does: its own launch
strategy derives the settings and thread pools from that characteristics file,
the model's GGUF header and the job's memory budget, and the report shows what
the PR's launch sets differently. The same model shards, toolchain, templates,
launch request and resource policy are required. Raw SSE, input IDs, image hashes,
CTest XML and cleanup are revalidated for every run, including the broker
receipt. Missing, failed or partial data cannot pass.

These are `purpose=ci` samples, separate from the formal benchmark in
[FINAL.md](../FINAL.md). Whole decode, TTFT and native prefill are compared on
the log scale with the base estimate (the base commit's history and the in-run
base), against a threshold from the noise of the device's baseline history. A
cell is `regressed` when a metric is slower beyond that threshold and every
candidate run is slower; `needs-review` when the candidate runs disagree,
generated tokens differ, the base has no history for this machine, OS, model,
template, memory budget and CPU set, the in-run base differs from its history or
the host check flagged a run; otherwise `passed`. A metric faster beyond the
threshold is labelled `improved` without changing the status. Invalid evidence
is `failed`. A pass means no slowdown beyond measured noise, not proof that none
exists. See
[performance acceptance](PERFORMANCE.md) for the decision rule, the baseline
history, the base build cache and the legacy reference adapter.
Local synthetic tests do not prove OS isolation.

## Publication and deployment

In App mode, only the GitHub-hosted publisher mints the dedicated check App's
short-lived installation token. It has checks-write and source-read permissions,
not access to device hosts. It rechecks the current head/base/candidate, review
and normal-CI attempt **after** evidence processing. Old controller runs or
attempts, changed sources or incomplete broker evidence cannot be replayed. The
check is published on the exact head with a run/attempt/plan-derived external ID.

Require `InferSSD device performance` specifically from this dedicated App, not
from "any source" or the GitHub Actions App: another workflow can produce an
identically named Actions check. Require branches to be up to date, protect the
staging and controller branches, and disallow bypass. See
[required-check sources and strict status checks](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-protected-branches/about-protected-branches#require-status-checks-before-merging).

Deploy control bytes from a reviewed commit, never a dirty checkout:

```sh
python3 bench/ci/export_controller.py --source /path/to/inferssd \
  --revision FULL_REVIEWED_40_CHARACTER_SHA --output /path/to/new-controller-export
```

The source tree contains `framework/`, so export it directly. `--registry-revision`
exists only for older control trees that needed a separately pinned settings
registry; do not pass it when `framework/` is present. `--workflow` selects
`device-performance` (default), `integration-smoke` or `checkpoint`.

Review and commit that export into the controller repository. The export carries
a source-origin/file-hash manifest; the controller workflow is protected
separately. Provision read-only source credentials only for hosted preparation,
the dedicated App only for hosted publication, and immutable device configs.
Where available, environment required reviewers with self-review disabled can
supplement the exact-SHA review marker; this design does not depend on them. See
[deployment protection availability](https://docs.github.com/en/actions/how-tos/deploy/configure-and-manage-deployments/manage-environments).
Organization runner groups can restrict repositories and workflow refs, but this
design does not rely on that feature either; never grant this repository access
to the device runners. See
[runner-group access](https://docs.github.com/en/enterprise-cloud@latest/actions/how-tos/manage-runners/self-hosted-runners/manage-access).

Enable `INFERSSD_CI_ENABLED` or require the App check only after the isolation
and native cleanup checks pass on every intended device and their receipts have
been reviewed. Merging changes to this directory does not publish, install or
activate anything.
