# Operator preparation for reviewed devices

`operator_prepare.py` uses the preparer's existing `gh` login and SSH access.
It does not export a GitHub token, create a PAT, change the broker, register a
runner or contact the source API from a device. Candidate code only executes
later inside existing broker stages. The root staging helper copies source
objects as data and never executes them.

The local input directory must be trusted and outside a candidate checkout.
It contains reviewed device templates, model/settings identities and a profile:

```json
{
  "schema_version": 1,
  "controller_workflow_sha": "FULL_PRIVATE_CONTROLLER_MAIN_SHA",
  "devices": {
    "cloud-5090": {
      "template": "cloud-template.json",
      "manifest": "cloud-manifest-template.json",
      "control_revision": "FULL_REVIEWED_CONTROL_SOURCE_SHA",
      "control_manifest_sha256": "INSTALLED_CONTROL_MANIFEST_SHA256",
      "lock_identity": [0, 0],
      "gpu_uuid": "GPU-ACTUAL_UUID",
      "min_free_bytes": 8589934592,
      "rootstage_files": {
        "settings-deepseek-base.ssd": "deepseek-base.ssd",
        "settings-deepseek-candidate.ssd": "deepseek-candidate.ssd",
        "settings-muse-base.ssd": "muse-base.ssd",
        "settings-muse-candidate.ssd": "muse-candidate.ssd"
      },
      "ssh": {
        "argv": ["/usr/bin/ssh", "-T", "-a", "-o", "BatchMode=yes", "REVIEWED_ENDPOINT"],
        "sudo_password_required": false,
        "root_argv": ["/usr/bin/python3", "-I", "/opt/inferssd-ci-control/current/bench/ci/operator_stage.py"]
      }
    },
    "dgx-spark": {"EXPLICIT_REVIEWED_PROFILE": "same required fields; no defaults"}
  }
}
```

The placeholder values above are not executable configuration. Every enabled
device needs an explicit profile. The same directory holds `baseline-history/`,
the verified baseline records the review operator writes and, per device, the
characteristics files its baseline jobs profiled (see
[performance acceptance](PERFORMANCE.md#baseline-history)). Preparation
re-verifies every record and characteristics file and seals the executing
devices' records into every device's `config.json` as `baseline_history`;
`envelope.json` carries their `baseline_history_sha256`. Neither the root stager
nor the installed root policy reads or pins them. A PR job also seals, per
device, the characteristics file of that device's most recent baseline job as
`machine_characteristics` (the file's text, its SHA-256, the baseline job and
commit it comes from, and its staged paths); `envelope.json` names each
device's hash in `machine_characteristics_sha256`. The root stager writes it
read-only to `/opt/inferssd-ci-integrations/JOB-machine.json`, mounted in the
Linux sandbox as `/models/machine.json`, or to `machine.json` in the Mac job's
input root, after checking its fixed path and its hash and that the job is not a
baseline job: a baseline job seals none and profiles the machine itself. Preparing a PR job for
a device without baseline characteristics fails with "DEVICE has no baseline
machine characteristics yet; a baseline job must run on it first". A
Linux profile may set `"base_build_cache": true` (a boolean; the Mac does not
support it): a baseline job then asks the broker to keep its build, and a PR
job offers the base commit's recorded build to the broker for its base side.
`--execution-devices` can select cloud-5090,
dgx-spark and local-mbp. An optional per-device `job_id` must be fresh, distinct and 32 lowercase
hex characters; otherwise preparation generates one. Old template job/source
identities are replaced with the exact new identities. When the impact plan
requires performance, both base and candidate are exported and built. Both sides
use the same reviewed model/settings/resource values and unchanged prompt bytes;
settings payload hashes must match their declarations.

Server launches are not written into the template: each side resolves them on
the device with its own typed launch, the one behind `inferssd serve` (see
[performance acceptance](PERFORMANCE.md#launches)). Every model of both sides
carries the same `launch` request: `{"profile", "port", "extra"}` (the
launcher's model profile, the server port and extra server arguments, which the
side's launcher checks like a user's), and `context`, `batch` or `ubatch` only to
pin a size the side's strategy would otherwise choose. Its `smoke` and `benchmarks` entries keep the
environment, working directory, deadlines and OS-cache treatment, and carry no
`argv`, `preflight_argv` or `port`. The reviewed settings payload sets only
`expert.source`; the device refuses a resolution whose settings file sets any
other key, because the strategy derives those or the engine decides them. The
template's `host.resources` is the machine policy the launches take: its memory
budget is the runtime policy's budget, and its CPU set and the device lock bind
every stage as before. The memory budget and the CPU set are also part of the
baseline history conditions, so changing either starts a new history for the
device. Templates must build the probes that `inferssd profile` runs in a
baseline job: `inferssd-cpuprobe` and `inferssd-ssdprobe`, plus
`inferssd-memprobe` on the CUDA devices.

Each server gets the resolved argv with its own event file in the side's launch
directory: the functional request and the short preflight of a tier use
`native-events-preflight-<tier>.jsonl`, timed run k uses
`native-events-<tier>-r<k>.jsonl`, and a functional request of a model without
timed runs keeps the launcher's own `native-events.jsonl`. The controller refuses a timed name that
equals the preflight or functional event file. Existing events are retained;
the telemetry writer's exclusive-create contract is preserved. The functional
launch is the first tier's preflight launch, so it supplies that preflight
without another request when the two entries' other fields match.

Prepare a reviewable package without connecting to a device:

```sh
python3 -I bench/ci/operator_prepare.py prepare \
  --source /trusted/local/repository --base FULL_BASE --head FULL_HEAD \
  --candidate FULL_CANDIDATE --inputs /trusted/operator/inputs.json \
  --output /trusted/operator/fresh-output --execution-devices cloud-5090 \
  --gh /path/to/existing/gh
```

The authenticated source Git API must match each complete local commit/tree and
its parents. Candidate must equal head with base an ancestor, or merge precisely
base/head. The package preserves the complete binary diff and both names of a
rename. Each side's source map lists every regular tracked file of its commit
with its Git blob and size, including files marked `export-ignore`; no
checkout/filter/hook runs.

The sources travel as one Git bundle, `source.bundle`, holding each side's
commit. Every device keeps a root-owned bare repository of the commits staged
to it (`/var/lib/inferssd-ci-inputs/source.git` on Linux,
`/Library/InferSSD-CI/inputs/source.git` on the Mac). With `source_anchor` set
in the inputs file, the bundle leaves out everything reachable from that commit,
so a job sends only its new objects; each device's repository must hold the
anchor, which deployment seeds. Without it the bundle is complete, and a
device without a repository creates one.

`envelope.json` retains the canonical impact plan, including required Mac and
performance cells. Separate `execution_devices` and `validation_mode` limit this
run to the declared devices and the 512-tier performance comparison.
Each required executed model runs decode 256 three times on the candidate and
once on the base, in the order candidate, base, candidate, candidate; the sealed
`performance_protocol` records that order, and the Linux stage allowlist
contains one `<side>-<model>-r<k>-benchmark-<tier>` stage per run and one
`<side>-<model>-launch` stage per resolved launch, running `/usr/bin/python3`;
a baseline job also has one `machine-profile` stage, and a PR job has none. 4K
and omitted devices' requirements remain explicit coverage gaps. Envelopes
prepared with the older `--validation-mode functional-numerical` keep that
meaning. The complete workflow input must fit
60 KiB; exceeding that bound refuses without truncation. The report cannot claim
PR check publication or full impact coverage when required devices/4K work is absent.

`--baseline` prepares the baseline job of one `main` commit instead (the review
operator does this for every new `main` head). Base, head and candidate must be
that commit; the plan then selects every enabled device and both models
although the diff is empty, only the candidate side is exported and built, the
machine is profiled once and each model runs four times. The root policy stays
the same: the root stager admits the fixed PR protocol, or the baseline protocol
only when both sides name the same revision.

Mac profiles use `broker_device: "mac-m5-max"` and a pinned `machine_identity`
with the SHA256 of the uppercase hardware UUID, hardware model and CPU brand,
rather than a CUDA GPU UUID. The actual executor resources are exactly
`max_processes`, `memory_budget_bytes` and `host_reserve_bytes`. These are sampled
guards, without a hard UMA or swap limit. The memory budget is also the launch
strategy's budget, from which it derives the Mac `host.budget_mib`; the same
model files and launch request apply to both sides.

Mac staging uses the pinned Xcode Python with `-I -B` and the fixed
`/Library/InferSSD-CI/controller/current/bench/ci/operator_mac_stage.py` path.
The dedicated SSH key forces `control/operator-ssh-entry`, which admits only
that fixed argv plus one `--expected-sha256` digest. Its separately reviewed sudo
rule grants only `control/operator-root-entry`; this revalidates the digest and
executes the fixed stager. No candidate path, shell, arbitrary Python program or
interactive password is accepted. The root policy is
`/Library/InferSSD-CI/control/operator-policy.json`; the private manifest is
`/Library/InferSSD-CI/private/job-manifests/JOB.json`. Staging copies input
data read-only under `/Library/InferSSD-CI/inputs/operator-JOB`, leaving
`jobs/JOB` fresh for the existing executor's `job.open`.

The Mac settings templates pin all bytes, allowing one `{{job_id}}` only in the
single `expert.source` array item under the exact pinned source side. Preparation
retains the template digest and generated payload digest; root staging reverses
only that one namespace and checks both. No general settings rewrite is allowed.
The existing SSH-context broker recovery runs before staging takes the original
lease; see [its identity and STOP contract](macos-sandbox/RECOVERY.md). A missing
launchd service alone never authorizes replacing the broker or the lock.

Before enabling root staging, review and install each generated
`DEVICE/policy.review.json` as the local root-owned, nonwritable
`/opt/inferssd-ci-control/operator-policy.json`. Install the exact reviewed
controller export, including `operator_stage.py`, first. This is an administrator
bootstrap operation, separate from candidate preparation. The helper cannot
replace its policy. It binds the device, broker config and original lock inode,
GPU UUID, control manifest, exact resource envelope, model mounts/identities,
settings hashes and, on DGX Spark, the existing model admission and mount proof. Extra host mappings such as
runner home or `/etc/shadow` are refused. All new destinations are fixed CI
prefixes with fresh exclusive writes. Existing model bytes/mounts, broker config,
lock and STOP are not changed.

On a nonroot host, install the reviewed `inferssd-ci-stage.socket` and
`inferssd-ci-stage@.service` templates with the actual permitted peer UID/group.
The same fixed Python command above becomes a local client of
`/run/inferssd-ci-stage.sock`; use `sudo_password_required: false` and omit
`sudo_mode`. The root service accepts only that UID, a bounded SHA header and
the plain tar, then invokes the same policy-bound receiver. Its fixed service
command and owner-controlled socket do not grant arbitrary root commands.
Keep the socket outside candidate worker mounts. Preserve vendor sudo/SSH
policy; do not change its ordering to obtain passwordless execution. Review
the forced SSH command as well: it must accept this exact client command.

For an existing permitted and separately reviewed narrow NOPASSWD entry, a profile uses
`sudo_mode: "nopasswd"`, `sudo_password_required: false`, and the fixed root argv
`["/usr/bin/sudo", "-n", "/usr/bin/python3", "-I",
"/opt/inferssd-ci-control/current/bench/ci/operator_stage.py"]`. It does not permit
arbitrary Python/bash files or a caller-selected policy path. Bootstrap/debug
can instead use `sudo_password_required: true` and prefix the fixed Python
command with `"/usr/bin/sudo", "-S", "-p", ""`; `stage/run --sudo-prompt` reads
one password hidden into memory and writes its line only to sudo's SSH stdin,
before the plain tar. It is absent from argv, environment, packages and logs.

Stage and dispatch the prepared package:

```sh
python3 -I bench/ci/operator_prepare.py stage --prepared /trusted/operator/fresh-output
python3 -I bench/ci/operator_prepare.py dispatch --prepared /trusted/operator/fresh-output \
  --gh /path/to/existing/gh
```

`run` accepts the preparation arguments and performs prepare → stage → dispatch.
It is the same backend for a trusted review watcher; it is not a second
authorization mechanism. The sender uses non-PTY SSH and a plain tar stream,
matching the root receiver's `r|` contract. Root holds the original lease during
staging, rejects STOP, verifies GPU UUID and absence of compute work, checks the
entire installed control manifest and reads model stat/RO mount metadata. It
adds the bundle to its repository with Git that runs no hooks, uses no network
transport and checks every arriving object. For each side it then checks the
commit's tree and file list against the source map, and rehashes every blob it
writes before making the source read-only.
It installs only the reviewed two-sided config/settings, a PR job's sealed
characteristics file and the job manifest, then returns an identity-bound root
receipt.

Dispatch requires all chosen root receipts and an unchanged local account and
private controller main. It sends only `prepared_envelope` to the fixed
`device-performance.yml` workflow. A durable exclusive claim precedes SSH stage
or the one dispatch POST. Failed or ambiguous attempts remain recorded and
cannot be automatically retried with the same prepared directory/job. An exact
returned run ID is required; there is no heuristic run search. The CLI records
the actual parent run and prints its ID/URL. The existing hosted controller
owns child dispatch, parent cancellation, evidence collection and report-only
Actions results; a successful cancellation API call never proves cleanup.
The dispatch request uses GitHub's documented
[`return_run_details` contract](https://docs.github.com/en/rest/actions/workflows#create-a-workflow-dispatch-event).

The local functions are `prepare(args, api=None)`,
`stage(args, sender=send_stage, password=None)` and `dispatch(args, api=None)`.
`dispatch` returns the actual identity-checked run dictionary. Tests use small
temporary Git repositories and synthetic root filesystems/transports. They do
not establish real device isolation, native math, a live dispatch or deployment.
