#!/usr/bin/env python3
"""Exact reviewed PR -> root-staged selected children -> terminal dedicated App check."""

import argparse
import html
import json
import math
import os
from pathlib import Path
import re
import signal
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ci.contract import HASH, digest, load, policy_read, revision, save
from ci.impact import fingerprint, plan as selection, selected, validate
from ci.private_controller import API, admission, bind_context, check_body, context, current_ci, source_manifest, writable
from ci.selected_aggregate import aggregate, operator_aggregate
from ci.selection_contract import JOB, execution_scope, FUNCTIONAL_MODE
from ci import private_controller, selection_dispatch


def changed_paths(api, pr):
    paths, count = [], 0
    for page in range(1, 32):
        rows = api.call(f"pulls/{pr['number']}/files?per_page=100&page={page}")
        count += len(rows)
        for row in rows:
            paths.append(row["filename"])
            if row.get("previous_filename"):
                paths.append(row["previous_filename"])
        if len(rows) < 100:
            break
    else:
        raise ValueError("incomplete changed-file API listing")
    if pr.get("changed_files") != count:
        raise ValueError("complete PR changed-file count differs")
    return paths


def jobs_checked(jobs, impact, execution_devices=None, validation_mode=FUNCTIONAL_MODE):
    devices = {cell["device"] for cell in selected(impact)}
    if execution_devices is not None:
        scope = execution_scope(dict(validation_mode=validation_mode, execution_devices=execution_devices))
        if not set(scope["execution_devices"]) <= devices:
            raise ValueError("operator scope contains an unaffected device")
        devices = set(scope["execution_devices"])
    if not isinstance(jobs, dict) or set(jobs) != devices:
        raise ValueError("root-staged jobs must exactly cover selected devices")
    fields = {"device", "job_id", "source_revision", "config_sha256", "selection_plan_sha256",
              "control_revision", "control_manifest_sha256"}
    for device, job in jobs.items():
        if (not isinstance(job, dict) or set(job) != fields or job["device"] != device
                or not JOB.fullmatch(job["job_id"]) or job["source_revision"] != impact["candidate"]
                or job["selection_plan_sha256"] != fingerprint(impact)):
            raise ValueError("root-staged job/source/selection identity differs")
        revision(job["control_revision"])
        if any(not HASH.fullmatch(job[key]) for key in ("config_sha256", "control_manifest_sha256")):
            raise ValueError("root-installed config/controller hashes required")
    if len({job["job_id"] for job in jobs.values()}) != len(jobs):
        raise ValueError("each selected physical device needs a fresh distinct job")
    return jobs


def environment(args, env):
    policy = policy_read(load(args.policy))
    event = load(env["GITHUB_EVENT_PATH"])
    identity = context(policy, env, event)
    return policy, event, identity


OPERATOR_PURPOSE = "operator-selected-report-only"
MAX_ENVELOPE_BYTES = 60 * 1024


def operator_access(policy, event, identity, env):
    """Only private-repository metadata is queried at runtime; no source token."""
    if identity["run_attempt"] != 1:
        raise ValueError("operator jobs are single-use; prepare fresh jobs instead of rerunning")
    api = API(policy["controller_repository"], env["GITHUB_TOKEN"])
    metadata, branch = api.call(""), api.call("branches/main")
    triggering = env.get("GITHUB_TRIGGERING_ACTOR", "")
    sender_id = event.get("sender", {}).get("id")
    if (metadata.get("private") is not True or metadata.get("default_branch") != "main"
            or branch.get("protected") is not True
            or branch.get("commit", {}).get("sha") != identity["workflow_sha"]
            or not triggering or type(sender_id) is not int or sender_id <= 0
            or not writable(api, identity["actor"]) or not writable(api, triggering)
            or env.get("INFERSSD_CI_ENABLED") != "true"):
        raise ValueError("operator identity/permission/protected controller activation gate failed")
    return dict(login=identity["actor"], id=sender_id, triggering_actor=triggering)


def operator_envelope(value, policy, identity, actor):
    fields = {"schema_version", "purpose", "repository", "base", "head", "candidate",
              "selection", "selection_plan_sha256", "execution_devices", "validation_mode",
              "prepared_by", "controller_repository", "controller_workflow_sha", "policy_sha256",
              "baseline_history_sha256", "machine_characteristics_sha256", "jobs"}
    if not isinstance(value, dict) or set(value) != fields:
        raise ValueError("complete sealed operator envelope required")
    impact = validate(value["selection"], policy)
    scope = execution_scope(value)
    prepared = value["prepared_by"]
    machines = value["machine_characteristics_sha256"]
    if (value["schema_version"] != 1 or value["purpose"] != OPERATOR_PURPOSE or scope is None
            or value["repository"] != policy["repository"]
            or value["controller_repository"] != policy["controller_repository"]
            or value["controller_workflow_sha"] != identity["workflow_sha"]
            or value["policy_sha256"] != fingerprint(policy)
            or not isinstance(value["baseline_history_sha256"], str) or not HASH.fullmatch(value["baseline_history_sha256"])
            or not isinstance(machines, dict) or not set(machines) <= set(value["execution_devices"])
            or any(not isinstance(item, str) or not HASH.fullmatch(item) for item in machines.values())
            or value["selection_plan_sha256"] != fingerprint(impact)
            or any(value[key] != impact[key] for key in ("base", "head", "candidate"))
            or not isinstance(prepared, dict) or prepared.get("login") != actor["login"]
            or type(prepared.get("id")) is not int or prepared["id"] != actor["id"]):
        raise ValueError("operator envelope source/actor/controller/policy identity differs")
    jobs_checked(value["jobs"], impact, scope["execution_devices"], scope["validation_mode"])
    return value


def read_operator_envelope(event):
    from ci.stage_session import unique_object
    raw = event.get("inputs", {}).get("prepared_envelope")
    if not isinstance(raw, str) or len(raw.encode("utf-8")) > MAX_ENVELOPE_BYTES:
        raise ValueError("complete operator envelope exceeds the bounded dispatch input")
    def nonfinite(_):
        raise ValueError("operator envelope contains a nonfinite number")
    return json.loads(raw, object_pairs_hook=unique_object, parse_constant=nonfinite)


def operator_job_controls(args, jobs):
    revision = source_manifest(args.control)
    manifest_sha256 = digest(args.control / "control-source.json")
    if any(job["control_revision"] != revision
           or job["control_manifest_sha256"] != manifest_sha256 for job in jobs.values()):
        raise ValueError("operator jobs must use this exact reviewed controller export")


def operator_preflight(args, env=os.environ):
    policy, event, identity = environment(args, env)
    actor = operator_access(policy, event, identity, env)
    envelope = operator_envelope(read_operator_envelope(event), policy, identity, actor)
    operator_job_controls(args, envelope["jobs"])
    admitted = dict(envelope, controller=identity, operator=actor,
                    envelope_sha256=fingerprint(envelope), control_revision=source_manifest(args.control))
    save(args.plan, admitted)
    return 0


def preflight(args, env=os.environ):
    policy, event, identity = environment(args, env)
    private = API(policy["controller_repository"], env["GITHUB_TOKEN"])
    metadata = private.call("")
    branch = private.call("branches/main")
    if (metadata.get("private") is not True or metadata.get("default_branch") != "main"
            or not writable(private, identity["actor"]) or branch.get("protected") is not True
            or branch.get("commit", {}).get("sha") != identity["workflow_sha"]
            or env.get("INFERSSD_CI_ENABLED") != "true"):
        raise ValueError("private protected controller activation/permission/source gate failed")
    inputs = event["inputs"]
    number = inputs["pull_request"]
    if not re.fullmatch(r"[1-9][0-9]*", number):
        raise ValueError("positive PR required")
    api = API(policy["repository"], env["SOURCE_READ_TOKEN"])
    pr = api.call("pulls/" + number)
    admitted = admission(api, pr, policy, revision(inputs["head_sha"]))
    admitted["normal_ci"] = current_ci(api, admitted, policy)
    impact = selection(changed_paths(api, pr), policy, **{key: admitted[key] for key in ("base", "head", "candidate")})
    again = admission(api, api.call("pulls/" + number), policy, admitted["head"])
    if any(again[key] != admitted[key] for key in ("base", "head", "candidate", "authorization")):
        raise ValueError("PR or independent exact-SHA authorization changed during planning")
    from ci.stage_session import unique_object
    jobs = jobs_checked(json.loads(inputs["staged_jobs"], object_pairs_hook=unique_object), impact)
    admitted.update(selection=impact, selection_plan_sha256=fingerprint(impact), jobs=jobs,
                    controller=identity, control_revision=source_manifest(args.control),
                    policy_sha256=fingerprint(policy))
    save(args.plan, admitted)
    return 0


class ChildMetadataAPI:
    """Observe the returned child ID until its queued title is fully bound.

    Only a missing/known generic title may be pending. Every returned response
    is retained before classification; strict identity still gates all use.
    Only the canonical transport retries failed GETs, within this observer's
    deadline. This observer never retries a dispatch or cancellation.
    """

    def __init__(self, api, child, artifacts, phase, *, sleep=None, clock=None):
        self.api, self.child, self.artifacts, self.phase = api, child, artifacts, phase
        self.sleep, self.clock = sleep or time.sleep, clock or time.monotonic
        self.path = "actions/runs/" + str(child["id"])
        self.expected = dict(id=child["id"], workflow_id=child["workflow_id"],
            path=selection_dispatch.WORKFLOW, event="workflow_dispatch", head_branch="main",
            head_sha=child["controller_sha"], run_attempt=1, display_title="integration-" + child["job_id"])
        self.sequence = 0
        self.identity_confirmed = False

    def __getattr__(self, name):
        return getattr(self.api, name)

    def call(self, path, data=None, method=None):
        if path == self.path + "/cancel" and method == "POST" and not self.identity_confirmed:
            raise ValueError("child cancellation requires a fully observed exact identity")
        if path != self.path or data is not None or method not in (None, "GET"):
            return self.api.call(path, data, method)
        deadline = self.clock() + 10
        for attempt in range(6):
            if type(self.api) is private_controller.API:
                run = self.api.call(path, deadline=deadline)
            else:
                run = self.api.call(path)
            actual = {key: run.get(key) for key in self.expected} if isinstance(run, dict) else {}
            differences = {key: dict(expected=value, actual=actual.get(key))
                for key, value in self.expected.items()
                if type(actual.get(key)) is not type(value) or actual.get(key) != value}
            pending = (isinstance(run, dict) and set(differences) == {"display_title"}
                and run.get("status") in ("queued", "requested") and run.get("conclusion") is None
                and run.get("display_title") in (None, "", "InferSSD reviewed integration smoke"))
            self.sequence += 1
            save(self.artifacts / (f"child-metadata-{self.child['id']}-{self.phase}-"
                                  f"{time.time_ns()}-{self.sequence}.json"),
                 dict(schema_version=1, child_sha256=fingerprint(self.child), phase=self.phase,
                      observation=self.sequence, expected=self.expected, differences=differences,
                      pending_title=pending, run=run))
            if not differences:
                selection_dispatch.run_identity(run, self.child["workflow_id"],
                    self.child["controller_sha"], self.child["job_id"])
                self.identity_confirmed = True
                return run
            if not pending:
                raise ValueError("child workflow/ref/revision/job/attempt identity differs: "
                                 + ", ".join(differences))
            remaining = deadline - self.clock()
            if attempt == 5 or remaining <= 0:
                break
            self.sleep(min(2, remaining))
        raise TimeoutError("child title did not reach exact identity within finite observation window")


def execute(args, env=os.environ):
    policy, _, _ = environment(args, env)
    admitted = load(args.plan)
    bind_context(admitted, policy, env)
    impact = validate(admitted["selection"], policy)
    if admitted["policy_sha256"] != fingerprint(policy) or admitted["control_revision"] != source_manifest(args.control):
        raise ValueError("admitted trusted control/policy changed")
    operator = admitted.get("purpose") == OPERATOR_PURPOSE
    if operator:
        _, event, identity = environment(args, env)
        actor = operator_access(policy, event, identity, env)
        envelope = read_operator_envelope(event)
        operator_envelope(envelope, policy, identity, actor)
        if (admitted.get("envelope_sha256") != fingerprint(envelope)
                or any(admitted.get(key) != value for key, value in envelope.items())
                or admitted.get("operator") != actor):
            raise ValueError("admitted operator envelope changed")
    jobs = jobs_checked(admitted["jobs"], impact, admitted["execution_devices"] if operator else None,
                        admitted["validation_mode"] if operator else FUNCTIONAL_MODE)
    if operator:
        operator_job_controls(args, jobs)
    # Source read credentials are not used to dispatch or sent to device runners.
    api = API(policy["controller_repository"], env["GITHUB_TOKEN"], version="2026-03-10")
    args.artifacts.mkdir(parents=True, exist_ok=False)
    state = dict(schema_version=1, selection_plan_sha256=fingerprint(impact), children={}, errors={})
    dispatched = {}
    parent = f"{admitted['controller']['run_id']}-{admitted['controller']['run_attempt']}"
    deadline = time.monotonic() + 10800
    previous = {}

    def interrupted(number, frame):
        raise KeyboardInterrupt("selected controller interrupted")

    for number in (signal.SIGINT, signal.SIGTERM):
        previous[number] = signal.signal(number, interrupted)
    try:
        for device, job in jobs.items():
            try:
                dispatched[device] = selection_dispatch.start(api, job, admitted["controller"]["workflow_sha"], parent)
                save(args.artifacts / (device + "-dispatch.json"), dispatched[device])
            except Exception as error:
                state["errors"][device] = f"{type(error).__name__}: {error}"
        for device, child in dispatched.items():
            collection_attempted = False
            observed = ChildMetadataAPI(api, child, args.artifacts, "wait")
            try:
                run = selection_dispatch.wait(observed, child, timeout_s=max(1, deadline - time.monotonic()))
                state["children"][device] = run
                collection_attempted = True
                selection_dispatch.collect(api, run, jobs[device], args.artifacts / device, parent=parent)
            except Exception as error:
                state["errors"][device] = f"{type(error).__name__}: {error}"
                if not collection_attempted:
                    # A complete regressed or needs-review cell exits nonzero.
                    # Retain its raw artifact once so the always report can
                    # recompute metrics; cancellation/unknown state still
                    # cannot become a pass.
                    try:
                        run = observed.call("actions/runs/" + str(child["id"]))
                        selection_dispatch.run_identity(run, child["workflow_id"], child["controller_sha"], child["job_id"])
                        if run.get("id") != child["id"] or run.get("status") != "completed":
                            raise ValueError("failed child has no exact terminal run")
                        state["children"][device] = run
                        selection_dispatch.collect(api, run, jobs[device], args.artifacts / device, parent=parent)
                        state.setdefault("retained_failed_children", {})[device] = run["id"]
                    except Exception as retained_error:
                        state["errors"][device] += "; raw retention: " + type(retained_error).__name__ + ": " + str(retained_error)
    finally:
        save(args.artifacts / "dispatch-state.json", state)
        try:
            cancel_children(args, env)
        finally:
            for number, handler in previous.items():
                signal.signal(number, handler)
    return 1 if state["errors"] else 0


def cancel_children(args, env=os.environ):
    """Also called by the hosted always step after a killed execute process."""
    policy, _, identity = environment(args, env)
    api = API(policy["controller_repository"], env["GITHUB_TOKEN"])
    results = {}
    uncertain = False
    if args.artifacts.exists():
        original = args.artifacts / "cancellation-requests.json"
        prior = load(original) if original.exists() else None
        prior_sha = digest(original) if prior is not None else None
        for path in sorted(args.artifacts.glob("*-dispatch.json")):
            child = load(path)
            parent = f"{identity['run_id']}-{identity['run_attempt']}"
            if (child.get("controller_sha") != identity["workflow_sha"]
                    or child.get("inputs", {}).get("selection_parent") != parent):
                raise ValueError("refuse cancellation of another controller's child")
            try:
                observed = ChildMetadataAPI(api, child, args.artifacts, "cancel")
                claim = args.artifacts / ("cancellation-" + path.stem + "-attempt.json")
                expected_claim = dict(schema_version=1, parent=parent, child_sha256=fingerprint(child),
                                      run_id=child["id"], cleanup_confirmed=False)
                attempted = prior is not None or claim.exists()
                if not attempted:
                    try:
                        # Claim before the first possible POST. An interrupted
                        # or ambiguous request is never an automatic retry.
                        save(claim, expected_claim)
                    except FileExistsError:
                        attempted = True
                if attempted:
                    if claim.exists() and load(claim) != expected_claim:
                        raise ValueError("prior cancellation claim identifies another child")
                    run = observed.call("actions/runs/" + str(child["id"]))
                    selection_dispatch.run_identity(run, child["workflow_id"], child["controller_sha"], child["job_id"])
                    previous = prior.get(path.name) if isinstance(prior, dict) else None
                    unknown = (not isinstance(previous, dict) or previous.get("run_id") != child["id"] or "error" in previous
                               or previous.get("prior_outcome_unknown") is True)
                    results[path.name] = dict(run_id=child["id"], status=run.get("status"),
                        conclusion=run.get("conclusion"), prior_attempt=True, prior_outcome_unknown=unknown,
                        cancellation_requested=False, cleanup_confirmed=False)
                    uncertain |= unknown or run.get("status") != "completed"
                else:
                    results[path.name] = selection_dispatch.cancel(observed, child)
                    uncertain |= results[path.name].get("cancellation_requested") is not False
            except Exception as error:
                results[path.name] = dict(error=repr(error), cleanup_confirmed=False)
                uncertain = True
        if prior is None:
            save(original, results)
        else:
            save(args.artifacts / ("cancellation-recheck-" + str(time.time_ns()) + ".json"),
                 dict(schema_version=1, parent=f"{identity['run_id']}-{identity['run_attempt']}",
                      prior_requests_sha256=prior_sha, results=results, cleanup_confirmed=False))
    # Closed Actions runs are not a physical cleanup verdict; only retained
    # root guardian/unit/process evidence can establish that independently.
    return 1 if uncertain else 0


def summary_text(value):
    return html.escape(str(value)).replace("|", "\\|").replace("\r", " ").replace("\n", " ")


def metric_text(metric, key):
    value = metric.get(key)
    if type(value) not in (int, float) or not math.isfinite(value):
        return "unavailable"
    return f"{value:+.3f}%" if key == "change_percent" else f"{value:.4f}"


def number_text(value, signed=False):
    if type(value) not in (int, float) or not math.isfinite(value):
        return "unavailable"
    return f"{value:+.4f}" if signed else f"{value:.4f}"


def values_text(values, signed=False):
    if not isinstance(values, list) or not values:
        return "unavailable"
    return ", ".join(number_text(value, signed) for value in values)


METRIC_LABELS = (("decode_first128_tps", "Decode 0–128 tok/s"), ("decode_second128_tps", "Decode 128–256 tok/s"),
                 ("decode_tps", "Whole decode tok/s"), ("ttft_s", "Request TTFT seconds"),
                 ("server_prefill_tps", "Native prefill tok/s"))
def launch_text(launch, name):
    """One derived setting or thread pool of a side, with its origin."""
    if not isinstance(launch, dict):
        return "unavailable"
    if name in ("server_args", "argv"):
        return " ".join(map(str, launch.get(name, []))) if name == "server_args" else "see aggregate.json"
    group, key = ("threads", name.split(".", 1)[1]) if name.startswith("threads.") else ("settings", name)
    entry = launch.get(group, {}).get(key)
    if not isinstance(entry, dict):
        return "unset"
    return json.dumps(entry.get("value"), separators=(",", ":")) + " (" + str(entry.get("origin")) + ")"


RESULT_LABELS = {"within-noise": "within noise", "improved": "improved: faster beyond noise",
                 "regressed": "regressed: slower beyond noise, every PR run slower",
                 "candidates-disagree": "slower beyond noise, PR runs disagree",
                 "no-baseline-history": "no matching baseline history", "informational": "information only"}


def operator_summary(stream, report):
    """Render verified series values; never turn missing or needs-review into green."""
    status = report.get("status", "passed" if report["passed"] else "failed")
    stream.write("## Operator-authorized device report\n\n")
    stream.write("Declared device scope: **" + summary_text(status) + "**. ")
    stream.write("Full impact coverage: **" + str(report["full_impact_coverage_complete"]).lower() + "**.\n\n")
    stream.write("Linux scope: **" + summary_text(report.get("linux_scoped_status", "not-verified"))
                 + "**. Mac scope: **" + summary_text(report.get("mac_scoped_status", "not-verified")) + "**.\n\n")
    stream.write("PR publication is not enabled. These results identify sealed source; they do not attest the current PR head.\n\n")
    for key in ("base", "head", "candidate"):
        if key in report:
            stream.write(key + ": `" + summary_text(report[key]) + "`\n\n")
    if report.get("baseline_history_sha256"):
        stream.write("Sealed baseline history SHA-256: `" + summary_text(report["baseline_history_sha256"]) + "`\n\n")
    stream.write("| Device | Model | Functional | Numerical | Performance | Outcome |\n|---|---|---|---|---|---|\n")
    cells = []
    for row in report["cells"]:
        performance = row.get("performance", row["status"])
        if isinstance(performance, list):
            cells.extend((row, entry) for entry in performance)
            performance = ", ".join(entry["tier"] + ": " + entry["status"] for entry in performance) or "not-measured"
        deferred = row.get("performance_deferred_tiers", [])
        if deferred:
            performance += "; deferred tiers: " + ", ".join(deferred)
        values = [row.get("device"), row.get("model"), row.get("functional", row["status"]),
                  row.get("numerical", row["status"]), performance, row["status"]]
        stream.write("| " + " | ".join(summary_text(value) for value in values) + " |\n")
    if cells:
        baseline = [(row, entry) for row, entry in cells if entry.get("protocol", {}).get("purpose") == "baseline"]
        gate = [(row, entry) for row, entry in cells if (row, entry) not in baseline]
        if gate:
            stream.write("\n### 512 prompt / 256 decode: PR runs against the base and its baseline history\n\n")
            stream.write("Each PR cell ran the PR three times and the base once, in the order PR, base, PR, PR. "
                         "Every run is a fresh server with one completion; all runs are kept, none is retried or "
                         "selected. The base history holds the four runs of the base commit's baseline job on the "
                         "same device.\n\n")
            stream.write("| Device | Model | Tier | Metric | Base history | Base run | PR runs | Change | Threshold | Result |\n")
            stream.write("|---|---|---|---|---:|---:|---:|---:|---:|---|\n")
            for row, entry in gate:
                for key, label in METRIC_LABELS:
                    metric = entry.get("metrics", {}).get(key, {})
                    threshold = metric.get("threshold")
                    values = [row["device"], row["model"], entry["tier"], label, values_text(metric.get("history")),
                              number_text(metric.get("base_run")), values_text(metric.get("candidate")),
                              metric_text(metric, "change_percent"),
                              number_text(threshold) if threshold is not None else "none",
                              RESULT_LABELS.get(metric.get("result"), metric.get("result", "unavailable"))]
                    stream.write("| " + " | ".join(summary_text(value) for value in values) + " |\n")
            stream.write("\n")
            for row, entry in gate:
                reasons = entry.get("review_reasons", [])
                deciding = entry.get("metrics", {}).get("decode_tps", {})
                stream.write("- " + summary_text(row["device"] + "/" + row["model"] + "/" + entry["tier"]) + ": **"
                             + summary_text(entry.get("status", "unavailable")) + "**. Baseline history: "
                             + summary_text(len(entry.get("history_pool", []))) + " records, df "
                             + summary_text(deciding.get("df", "none")) + ", t "
                             + summary_text(deciding.get("k", "none")) + ". Improved: "
                             + summary_text(", ".join(entry.get("improvements", [])) or "none") + ". Review reasons: "
                             + summary_text(", ".join(reasons) if reasons else "none") + ".\n")
            stream.write("\nChange is the PR's geometric mean over the base estimate (the four history runs and the "
                         "in-run base), minus 1, in percent; lower TTFT is better. TTFT starts at the request, not model "
                         "loading. Threshold is on the log scale: max(ln(1 + floor), t·σ·√(1/3 + 1/5)), where σ is the "
                         "pooled within-commit spread of the device's baseline history for the same machine, OS, "
                         "model, template, memory budget and CPU set "
                         "and t the one-sided Student-t quantile for its degrees of freedom, matching a normal 3σ tail. "
                         "Only whole decode, TTFT and native prefill decide; the two decode half-windows are shown for "
                         "information. Regressed: a deciding metric beyond its threshold with every PR run slower than the "
                         "base estimate. Needs review: the in-run base differs from its history, PR runs disagree, generated "
                         "tokens differ, no matching history, or a flagged host check. Passed: no slowdown beyond measured "
                         "noise. Improved: a deciding metric whose PR mean is faster than the base estimate by more than "
                         "the same threshold; it is a label on that metric, and the cell status stays passed. Within "
                         "noise: inside the threshold either way.\n\n")
        if baseline:
            stream.write("\n### 512 prompt / 256 decode: baseline runs of a main commit\n\n")
            stream.write("One build measured four times; no pass or regression decision. σ and df include this record "
                         "and the device's earlier records for the same machine, OS, model, template, memory "
                         "budget and CPU set.\n\n")
            stream.write("| Device | Model | Tier | Metric | Runs | Mean ln | σ (log) | df |\n|---|---|---|---|---:|---:|---:|---:|\n")
            for row, entry in baseline:
                for key, label in METRIC_LABELS:
                    metric = entry.get("metrics", {}).get(key, {})
                    values = [row["device"], row["model"], entry["tier"], label, values_text(metric.get("values")),
                              number_text(metric.get("mean_log"), True), number_text(metric.get("sigma_log")),
                              summary_text(metric.get("df", "none"))]
                    stream.write("| " + " | ".join(summary_text(value) for value in values) + " |\n")
            stream.write("\n")
            for row, entry in baseline:
                reasons = entry.get("review_reasons", [])
                stream.write("- " + summary_text(row["device"] + "/" + row["model"] + "/" + entry["tier"]) + ": **"
                             + summary_text(entry.get("status", "unavailable")) + "**. Review reasons: "
                             + summary_text(", ".join(reasons) if reasons else "none") + ".\n")
            stream.write("\n")
        launches = [(row, entry) for row, entry in cells if isinstance(entry.get("derived_launch"), dict)]
        if launches:
            stream.write("\n### Derived launches\n\n")
            stream.write("Each side started its servers the way `inferssd serve` does: that side's own launch strategy "
                         "derived the settings, thread pools and server arguments from the device's machine "
                         "characteristics (a baseline job profiles them; a PR job takes those of the device's most "
                         "recent baseline job), the model's GGUF header and the job's memory budget. Origin is "
                         "strategy, launcher (the registry default or the launcher's model profile) or user (the "
                         "reviewed launch request). The thread pools are always shown; other rows are what the PR's "
                         "launch sets differently.\n\n")
            profiles = {}
            for row, _ in launches:
                profiles.setdefault(row["device"], row.get("machine_profile"))
            for device, machine in profiles.items():
                machine = machine if isinstance(machine, dict) else {}
                weak = [name + " " + status for name, status in machine.get("sections", {}).items() if status != "ok"]
                source = machine.get("baseline")
                origin = ("taken from the baseline job of `" + summary_text(source.get("commit", "unavailable")) + "`"
                          if isinstance(source, dict) else "profiled by this baseline job")
                stream.write("- " + summary_text(device) + ": machine characteristics SHA `"
                             + summary_text(machine.get("sha256", "unavailable")) + "`, " + origin
                             + "; sections not ok: " + summary_text(", ".join(weak) or "none") + ".\n")
            stream.write("\n| Device | Model | Tier | Setting | Base | PR or measured commit |\n|---|---|---|---|---|---|\n")
            for row, entry in launches:
                derived = entry["derived_launch"]
                sides = derived.get("sides", {})
                names = ["threads.decode", "threads.batch", "threads.cpus"]
                names += [name for name in derived.get("changes", []) if name not in names]
                for name in names:
                    values = [row["device"], row["model"], entry["tier"], name,
                              launch_text(sides.get("base"), name) if "base" in sides else "not run",
                              launch_text(sides.get("candidate"), name)]
                    stream.write("| " + " | ".join(summary_text(value) for value in values) + " |\n")
            stream.write("\n")
        mac_cells = [(row, entry) for row, entry in cells if row["device"] == "local-mbp"]
        if mac_cells:
            def scaled(value, unit):
                if type(value) not in (int, float) or not math.isfinite(value):
                    return "unavailable"
                return f"{value / unit:.4f}"

            stream.write("### Mac sampled host conditions\n\n")
            stream.write("| Model | Tier | Run | Quiet available GiB | Timed swap-out MiB | Timed compressed MiB | Non-CI CPU cores | Pressure | Timed status |\n")
            stream.write("|---|---|---|---:|---:|---:|---:|---|---|\n")
            for row, entry in mac_cells:
                runs = entry.get("environment", {}).get("runs") or {"unavailable": {}}
                for name, arm in runs.items():
                    timed = arm.get("timed", {})
                    levels = timed.get("memory_pressure_levels")
                    pressure = " → ".join(str(level) for level in levels) if isinstance(levels, list) else "unavailable"
                    values = [row["model"], entry["tier"], name,
                        scaled(arm.get("prelaunch_available_bytes"), 1 << 30),
                        scaled(timed.get("swapout_bytes"), 1 << 20),
                        scaled(timed.get("compressed_bytes"), 1 << 20),
                        metric_text(timed.get("process_activity", {}), "non_ci_cpu_cores"),
                        pressure, timed.get("status", "unavailable")]
                    stream.write("| " + " | ".join(summary_text(value) for value in values) + " |\n")
            stream.write("\n")
            for row, entry in mac_cells:
                environment = entry.get("environment", {})
                reasons = environment.get("reasons", ["missing environment evidence"])
                stream.write("- " + summary_text(row["model"] + "/" + entry["tier"]) + " environment: "
                    + summary_text(environment.get("status", "unavailable")) + "; reasons: "
                    + summary_text(", ".join(reasons) if reasons else "none") + ".\n")
            stream.write("\nQuiet memory is the final prelaunch sample; timed CPU is the sampled non-CI total. ")
            stream.write("Swap and pressure may be model-induced. These observations do not establish external interference, excuse measured slowdowns, or trigger automatic retries; they are not strict isolation.\n\n")
        stream.write("<details><summary>Cell identities and retained raw evidence</summary>\n\n")
        for row, entry in cells:
            identity = entry.get("measurement_identity", {})
            stream.write("- " + summary_text(row["device"] + "/" + row["model"] + "/" + entry["tier"]) + ": ")
            stream.write("prompt SHA `" + summary_text(identity.get("prompt_sha256", "unavailable")) + "`; ")
            stream.write("template IDs SHA `" + summary_text(identity.get("prompt_ids_sha256", "unavailable")) + "`; ")
            stream.write("baseline-history conditions SHA `" + summary_text(identity.get("conditions_sha256", "unavailable")) + "`.\n")
            for arm in entry.get("order", []):
                raw = entry.get("evidence", {}).get(arm, {}).get("result.json", {})
                side = arm.split("/")[-1]
                path = "/".join((row["device"], row.get("job_id", "unknown-job"), raw.get("path", "unavailable")))
                stream.write("  - " + summary_text(arm) + " source `" + summary_text(entry.get("sources", {}).get(side, "unavailable"))
                             + "`: `" + summary_text(path) + "`, SHA `" + summary_text(raw.get("sha256", "unavailable")) + "`.\n")
        stream.write("\n</details>\n")
    if report.get("error"):
        stream.write("\nFailure: " + summary_text(report["error"]) + "\n")
    for error in report.get("errors", []):
        stream.write("\nFailure: " + summary_text(error) + "\n")
    stream.write("\nDevices outside this declared scope and any 4K deferred coverage remain separate. ")
    if "local-mbp" in report.get("execution_devices", []):
        stream.write("Mac memory protection monitors worker physical footprint/process count and host reserve; RSS is diagnostic only. It does not disable swap or provide a Linux memory.max hard cap. ")
    stream.write("Formal benchmark samples: 0/16. Full comparison values and hashes are in aggregate.json; complete raw and cleanup evidence are in the retained selected-control artifacts.\n")


def operator_report(args, env=os.environ):
    """A terminal Actions artifact/summary, with no source status/check writes."""
    report = dict(schema_version=1, purpose=OPERATOR_PURPOSE, passed=False, status="failed", execution_scoped_status="failed",
                  linux_scoped_status="not-verified", mac_scoped_status="not-verified",
                  full_impact_coverage_complete=False, pr_publication="not-enabled",
                  source_identity="sealed-source-not-current-pr", formal_samples=0,
                  cells=[], errors=[])
    try:
        policy, event, identity = environment(args, env)
        actor = operator_access(policy, event, identity, env)
        envelope = operator_envelope(read_operator_envelope(event), policy, identity, actor)
        report.update(base=envelope["base"], head=envelope["head"], candidate=envelope["candidate"],
                      operator=actor, controller=identity, envelope_sha256=fingerprint(envelope),
                      **execution_scope(envelope))
        admitted = load(args.plan)
        bind_context(admitted, policy, env)
        if (any(admitted.get(key) != value for key, value in envelope.items())
                or admitted.get("envelope_sha256") != fingerprint(envelope)
                or admitted.get("operator") != actor
                or admitted.get("control_revision") != source_manifest(args.control)):
            raise ValueError("admitted operator envelope/control changed")
        impact = envelope["selection"]
        scope = execution_scope(envelope)
        jobs = jobs_checked(envelope["jobs"], impact, scope["execution_devices"], scope["validation_mode"])
        operator_job_controls(args, jobs)
        state = load(args.artifacts / "dispatch-state.json")
        if state["selection_plan_sha256"] != fingerprint(impact):
            raise ValueError("operator dispatch selection plan differs")
        if set(state["children"]) != set(jobs):
            raise ValueError("selected operator children are missing or unexpected")
        api = API(policy["controller_repository"], env["GITHUB_TOKEN"])
        children = {}
        for device, child in state["children"].items():
            latest = api.call("actions/runs/" + str(child["id"]))
            selection_dispatch.run_identity(latest, child["workflow_id"], identity["workflow_sha"], jobs[device]["job_id"])
            if latest.get("id") != child["id"] or latest.get("status") != "completed" or latest.get("conclusion") not in ("success", "failure"):
                raise ValueError("operator child is not an exact completed single attempt")
            children[device] = latest
        report = operator_aggregate(impact, policy, args.artifacts, jobs, children, scope,
                                    envelope["baseline_history_sha256"], envelope["machine_characteristics_sha256"])
        if state["errors"]:
            # Only a verified regressed/needs-review decision explains a child
            # that exited nonzero; any other child error fails the report.
            decided = ("regressed", "needs-review")
            decided_devices = {row["device"] for row in report["cells"] if row["status"] in decided}
            retained = state.get("retained_failed_children", {})
            complete = all(retained.get(device) == children[device]["id"] for device in state["errors"] if device in children)
            if (not set(state["errors"]) <= decided_devices or report.get("status") not in decided
                    or not complete):
                report.update(passed=False, status="failed", execution_scoped_status="failed", full_impact_coverage_complete=False)
                for platform, devices in (("linux", {"cloud-5090", "dgx-spark"}), ("mac", {"local-mbp"})):
                    if devices & set(state["errors"]):
                        report[platform + "_scoped_status"] = "failed"
            report["child_workflow_errors"] = state["errors"]
        report.update(base=envelope["base"], head=envelope["head"], candidate=envelope["candidate"],
                      operator=actor, controller=identity, envelope_sha256=fingerprint(envelope))
    except Exception as error:
        report.update(passed=False, status="failed", execution_scoped_status="failed", full_impact_coverage_complete=False,
                      error=f"{type(error).__name__}: {error}")
    save(args.output, report)
    summary = env.get("GITHUB_STEP_SUMMARY")
    if summary:
        with Path(summary).open("a", encoding="utf-8") as stream:
            operator_summary(stream, report)
    # A clean baseline job has no pass/fail decision, but it is a complete run.
    return 0 if report["passed"] or report.get("status") == "baseline" else 1


def finalize(args, env=os.environ):
    policy, event, identity = environment(args, env)
    api = API(policy["repository"], env["DEVICE_CHECK_TOKEN"])
    # Even failed admission gets a terminal failure for the explicitly requested
    # source PR/head. This fallback authorizes no device work or positive result.
    number, head = event["inputs"]["pull_request"], revision(event["inputs"]["head_sha"])
    if not re.fullmatch(r"[1-9][0-9]*", number):
        raise ValueError("positive PR required")
    pr = api.call("pulls/" + number)
    if (pr.get("head", {}).get("repo", {}).get("full_name") != policy["repository"]
            or pr.get("base", {}).get("repo", {}).get("full_name") != policy["repository"]):
        raise ValueError("terminal check cannot target an unrelated repository")
    admitted = dict(head=head, controller=identity)
    report = dict(schema_version=1, purpose="selected-ci", passed=False, formal_samples=0)
    try:
        admitted = load(args.plan)
        bind_context(admitted, policy, env)
        impact = validate(admitted["selection"], policy)
        if (admitted["head"] != head or admitted["pr"] != int(number)
                or admitted["policy_sha256"] != fingerprint(policy)
                or admitted["control_revision"] != source_manifest(args.control)):
            raise ValueError("admission/control/request binding differs")
        state = load(args.artifacts / "dispatch-state.json")
        if state["selection_plan_sha256"] != fingerprint(impact) or state["errors"]:
            raise ValueError("one or more selected dispatches failed or are missing")
        private = API(policy["controller_repository"], env["GITHUB_TOKEN"])
        for device, child in state["children"].items():
            latest = private.call("actions/runs/" + str(child["id"]))
            selection_dispatch.run_identity(latest, child["workflow_id"], identity["workflow_sha"], admitted["jobs"][device]["job_id"])
            if latest.get("status") != "completed" or latest.get("conclusion") != "success":
                raise ValueError("selected child state changed before App publication")
        report = aggregate(impact, policy, args.artifacts, admitted["jobs"], state["children"])
        current = admission(api, api.call("pulls/" + number), policy, head)
        if any(current[key] != admitted[key] for key in ("base", "head", "candidate", "authorization")):
            raise ValueError("source or independent approval changed before publication")
        if current_ci(api, admitted, policy) != admitted["normal_ci"]:
            raise ValueError("latest normal CI changed before publication")
        again = selection(changed_paths(api, pr), policy, **{key: admitted[key] for key in ("base", "head", "candidate")})
        if again != impact:
            raise ValueError("complete changed paths changed before publication")
    except Exception as error:
        report.update(passed=False, error=f"{type(error).__name__}: {error}")
    save(args.output, report)
    admitted.update(head=head, controller=identity)
    body = check_body(admitted, report, int(env["DEVICE_CHECK_APP_ID"]))
    body["output"]["title"] = "Selected device gates passed" if report["passed"] else "Selected device gate blocked"
    body["output"]["summary"] = json.dumps(report, ensure_ascii=False, allow_nan=False)[:60000]
    expected_app = body.pop("_expected_app_id")
    check = api.call("check-runs", body)
    if (check.get("app", {}).get("id") != expected_app or check.get("head_sha") != head
            or check.get("external_id") != body["external_id"]):
        raise ValueError("terminal check is not bound to the dedicated App/exact requested head")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("preflight", "operator_preflight", "execute", "cancel_children", "finalize", "operator_report"))
    for name in ("control", "policy", "plan", "artifacts", "output"):
        parser.add_argument("--" + name, type=Path, required=name in ("control", "policy", "plan"))
    args = parser.parse_args()
    raise SystemExit(globals()[args.mode](args))
