# SPDX-License-Identifier: Apache-2.0
"""Resolve each side's server launch on the device with that side's own framework.

A user starts a server with `inferssd serve`: the machine is profiled once,
and every launch derives its settings, thread pools and server arguments from
the machine characteristics, the model's GGUF header and the memory budget.
Device CI does the same inside the broker sandbox. The baseline job of a main
commit runs that commit's `inferssd profile` once per device and writes the
characteristics file, in the framework's own format, in the job's scratch; the
operator stores it with the device's baseline history. A PR job does not
profile: it takes the file of the device's most recent baseline job, sealed
into the job and staged read-only by root. Each side's own framework then
resolves its launch from that file with the typed launch's `--dry-run`, given the
runtime facts its live launch reads, and the broker starts exactly the resolved
argv as the server stage, in place of the launcher's own process scope. A
change to the launch strategy is therefore measured with the settings it
derives. A request names the context and batch sizes only to pin them; left
out, the side's strategy chooses them as it does for a user. Every timed run
starts a fresh server on the same prompt, so the launch keeps no prompt cache
(`--prompt-cache off`): a cache on disk would carry one run's prompt into the
next.

The characteristics' SHA-256 seals the file, and every resolution must name it.
Settings that the strategy leaves alone stay at their defaults, decided by the
engine; the reviewed settings file of a launch request sets only
`expert.source`.
"""

import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import time

from ci.contract import load, save
from ci.selection_contract import job_sides
from ci.staged_smoke import collect_log

PROFILE_STAGE = "machine-profile"
PROFILE_TIMEOUT_S = 1800
RESOLVE_TIMEOUT_S = 300
REQUEST = {"profile", "port", "extra"}
SIZES = ("context", "batch", "ubatch")  # optional: the strategy chooses what a request leaves out
# A PR job's sealed characteristics: the file, its hash, the baseline job it
# comes from and where root stages it.
SEALED = {"sha256", "text", "job_id", "commit", "host_path", "runtime_path"}
TELEMETRY = "--ssd-telemetry"
MARKER = "inferssd-ci-result "

# Both programs run as `python3 -I -c PROGRAM` in the sandbox, with the side's
# own framework source first on the path. The final MARKER line is the stage's
# result; everything before it is the framework's own output.
PROFILE = """\
import json, sys
source, output = sys.argv[1], sys.argv[2]
sys.path.insert(0, source)
from inferssd.cli import main
try:
    code = main(["profile", "--reprofile", "--output", output] + sys.argv[3:])
except SystemExit as stop:
    code = stop.code
# Status 1: a section failed and is recorded as failed; launches proceed without it.
if code not in (0, 1):
    raise SystemExit(code if isinstance(code, int) else 2)
with open(output, encoding="utf-8") as stream:
    text = stream.read()
sys.stdout.write("\\n" + MARKER + json.dumps(dict(returncode=code, characteristics=text)) + "\\n")
""".replace("MARKER", repr(MARKER))
RESOLVE = """\
import json, os, platform, sys
source, machine, policy, folder, shards = sys.argv[1:6]
sys.path.insert(0, source)
from inferssd.launch.typed_command import main as launch
from inferssd.strategy import inputs
# The facts an executed launch reads live, read here the same way.
runtime = inputs.read_runtime()
facts = dict(platform_system=platform.system(), platform_machine=platform.machine(),
             runtime_facts=inputs.runtime_facts_json(runtime),
             model_drive_key=inputs.model_drive_key(json.loads(shards)))
if facts["platform_system"] == "Linux":
    # The broker runs the server in its own resource scope and found the GPU idle,
    # so the launcher's checks for a scope of its own are stated, not probed.
    facts.update(systemd=True, cgroup_v2=True, euid=os.geteuid(), user_manager=True,
                 memory_controller=True, gpu_available=bool(runtime.gpus), gpu_processes="")
paths = dict(facts=folder + ".host-facts.json", policy=folder + ".policy.json", result=folder + ".result.json")
for name, value in (("facts", facts), ("policy", json.loads(policy))):
    with open(paths[name], "x", encoding="utf-8") as stream:
        json.dump(value, stream)
try:
    code = launch(["--dry-run", "--host-facts", paths["facts"], "--runtime-policy", paths["policy"],
                   "--machine", machine, "--output-dir", folder, "--result", paths["result"]] + sys.argv[6:])
except SystemExit as stop:
    code = stop.code
if code != 0:
    raise SystemExit(code if isinstance(code, int) else 2)
with open(paths["result"], encoding="utf-8") as stream:
    result = json.load(stream)
with open(os.path.join(folder, "settings.ssd"), encoding="utf-8") as stream:
    settings = stream.read()
sys.stdout.write("\\n" + MARKER + json.dumps(dict(result=result, settings=settings, host_facts=facts)) + "\\n")
""".replace("MARKER", repr(MARKER))


def derived(config):
    """Whether this job derives its server launches on the device: every model has a launch request."""
    flags = {"launch" in data for side in config["sides"].values() for data in side["models"].values()}
    if len(flags) > 1:
        raise ValueError("either every model or none carries a launch request")
    return flags == {True}


def request_checked(value):
    """The reviewed inputs of one model's launch; everything else is derived on the device."""
    if (not isinstance(value, dict) or not REQUEST <= set(value) <= REQUEST | set(SIZES)
            or not isinstance(value["profile"], str) or not value["profile"]
            or any(type(value[key]) is not int or value[key] < 1 for key in ("port", *SIZES) if key in value)
            or value["port"] > 65535 or not isinstance(value["extra"], list)
            or any(not isinstance(arg, str) or not arg or "\0" in arg for arg in value["extra"])):
        raise ValueError("a launch request has profile, port and extra server arguments, "
                         "and optionally context, batch and ubatch")
    return value


def tiers(needs, model):
    return list(needs["performance"].get(model, {}).get("tiers", []))


def sealed_paths(platform, job_id):
    """Where root stages a PR job's characteristics file, and where its sandbox reads it."""
    if platform == "darwin":
        path = "/Library/InferSSD-CI/inputs/operator-" + job_id + "/machine.json"
        return dict(host_path=path, runtime_path=path)
    return dict(host_path="/opt/inferssd-ci-integrations/" + job_id + "-machine.json",
                runtime_path="/models/machine.json")


def sealed(config):
    """A PR job's characteristics, sealed from its device's most recent baseline job; None in a baseline job.

    A baseline job profiles the machine itself. A PR job never does: every
    resolution reads the sealed file, which root staged read-only, so nothing
    a PR stage writes is read as characteristics.
    """
    value = config.get("machine_characteristics")
    if value is None:
        return None
    if (not isinstance(value, dict) or set(value) != SEALED
            or any(not isinstance(value[key], str) or not value[key] for key in SEALED)
            or hashlib.sha256(value["text"].encode()).hexdigest() != value["sha256"]):
        raise ValueError("the sealed machine characteristics are incomplete or differ from their hash")
    sections(value["text"])
    return value


def profile_side(config, needs):
    """The side whose framework profiles the machine in a baseline job: its one measured commit."""
    return job_sides(config, needs)[0]


def launch_models(config, needs, side):
    """Models whose launch a side resolves: every model for the candidate, timed ones for the base."""
    return [model for model in needs["models"] if side == "candidate" or tiers(needs, model)]


def scratch(config, side):
    paths = config["sides"][side]["paths"]
    return paths["job_root"] + "/work/" + side if paths["platform"] == "darwin" else "/scratch/" + side


def machine_path(config):
    """The characteristics file every resolution reads: the sealed one, or a baseline job's own profile in scratch."""
    value = sealed(config)
    if value is not None:
        return value["runtime_path"]
    paths = config["paths"]
    return (paths["job_root"] + "/work" if paths["platform"] == "darwin" else "/scratch") + "/machine/machine.json"


def launch_folder(config, side, model):
    return scratch(config, side) + "/launch-" + model


def binary(config, side):
    spec = config["sides"][side]
    return str(PurePosixPath(spec["paths"]["frozen"]) / spec["build"]["binary_relative"])


def shards(config, side, model):
    return sorted(row["runtime_path"] for row in config["sides"][side]["models"][model]["model_identity"]["shards"])


def runtime_policy(config, folder):
    """The launch's runtime policy: the job's memory budget. A dry-run takes no lock and starts nothing."""
    resources = config["host"]["resources"]
    budget = resources.get("memory_budget_bytes", int(resources.get("cgroup_limits", {}).get("memory_max", 0)))
    return dict(platform=config["paths"]["platform"], lock_path=folder + ".lock", stop_path=folder + ".stop",
                memory_budget_bytes=budget, host_reserve_bytes=resources.get("host_reserve_bytes", 0),
                minimum_memlock_bytes=resources.get("memlock_bytes", 0), max_compressor_growth_bytes=0,
                max_swap_growth_bytes=0, runtime_seconds=1, cleanup_seconds=1)


def run_stage(session, stage_id, argv, cwd, timeout, output):
    stage = session.start(stage_id, "test", argv, {}, cwd, timeout)
    completed = session.wait(stage, deadline=time.monotonic() + timeout)
    save(output / "stage.json", completed)
    collect_log(session, stage, output)


def final_record(folder):
    """The program's result line from a retained stage that completed cleanly."""
    folder = Path(folder)
    stage = load(folder / "stage.json")
    if stage.get("returncode") != 0 or stage.get("cleanup_confirmed") is not True:
        raise RuntimeError("sandboxed " + folder.name + " stage failed or was unclean")
    lines = [line for line in (folder / "stdout.log").read_text(encoding="utf-8").splitlines()
             if line.startswith(MARKER)]
    if not lines:
        raise ValueError("sandboxed " + folder.name + " stage reported no result")
    return json.loads(lines[-1][len(MARKER):])


def sections(text):
    """Each section's status in a characteristics file."""
    value = json.loads(text)
    value = value.get("sections") if isinstance(value, dict) else None
    if (not isinstance(value, dict) or not value
            or any(not isinstance(item, dict) or not isinstance(item.get("status"), str) for item in value.values())):
        raise ValueError("the machine characteristics have no section statuses")
    return {name: item["status"] for name, item in sorted(value.items())}


def machine_read(folder):
    """A baseline job's machine characteristics: their SHA-256 and each section's status."""
    record = final_record(folder)
    text = record.get("characteristics") if isinstance(record, dict) else None
    if not isinstance(text, str) or record.get("returncode") not in (0, 1):
        raise ValueError("the machine profile did not produce a characteristics file")
    return dict(sha256=hashlib.sha256(text.encode()).hexdigest(), returncode=record["returncode"],
                sections=sections(text))


def profiled(folder):
    """The characteristics file a baseline job's profile stage printed, as the operator stores it."""
    machine_read(folder)
    return final_record(folder)["characteristics"]


def machine_sealed(config):
    """A PR job's machine characteristics: their SHA-256, each section's status and the baseline job they come from."""
    value = sealed(config)
    return dict(sha256=value["sha256"], sections=sections(value["text"]),
                baseline=dict(job_id=value["job_id"], commit=value["commit"]))


def adopt(config):
    """Check that root staged exactly the sealed characteristics, before any resolution reads them."""
    from ci.staged_build import sealed_file
    value = sealed(config)
    _, raw = sealed_file(value["host_path"], limit=16 << 20, contents=True)
    if raw != value["text"].encode():
        raise ValueError("the staged machine characteristics differ from the sealed file")
    return machine_sealed(config)


def profile(session, config, side, models, output):
    """Profile this machine once for the job with the side's `inferssd profile` and probes."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    spec = config["sides"][side]
    argv = ["/usr/bin/python3", "-I", "-c", PROFILE, spec["paths"]["source"] + "/framework/src", machine_path(config),
            "--build", spec["paths"]["frozen"]]
    for model in models:
        argv += ["--model", shards(config, side, model)[0]]
    run_stage(session, PROFILE_STAGE, argv, spec["paths"]["cwd"], PROFILE_TIMEOUT_S, output)
    machine = dict(machine_read(output), side=side)
    save(output / "profile.json", machine)
    return machine


def resolve(session, config, side, model, machine, output):
    """Resolve one model's launch with the side's own typed launch `--dry-run`."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    spec, data = config["sides"][side], config["sides"][side]["models"][model]
    request, folder, files = request_checked(data["launch"]), launch_folder(config, side, model), shards(config, side, model)
    argv = ["/usr/bin/python3", "-I", "-c", RESOLVE, spec["paths"]["source"] + "/framework/src", machine_path(config),
            json.dumps(runtime_policy(config, folder), sort_keys=True), folder, json.dumps(files),
            "--platform", spec["paths"]["platform"], "--model", files[0], "--settings", data["settings"]["runtime_path"],
            "--binary", binary(config, side), "--template", data["template"]["runtime_path"],
            "--profile", request["profile"], "--port", str(request["port"]), "--prompt-cache", "off"]
    for key in SIZES:
        if key in request:
            argv += ["--" + key, str(request[key])]
    if request["extra"]:
        argv += ["--", *request["extra"]]
    run_stage(session, side + "-" + model + "-launch", argv, spec["paths"]["cwd"], RESOLVE_TIMEOUT_S, output)
    resolved = resolution(config, side, model, final_record(output), machine["sha256"])
    save(output / "resolved.json", resolved)
    return resolved


def argument(argv, flag):
    return [argv[index + 1] for index, value in enumerate(argv[:-1]) if value == flag]


def resolution(config, side, model, record, characteristics_sha256):
    """Check one side's resolved launch and keep what its server stages use."""
    folder = launch_folder(config, side, model)
    settings, events = folder + "/settings.ssd", folder + "/native-events.jsonl"
    result = record.get("result") if isinstance(record, dict) else None
    if (not isinstance(result, dict) or set(record) != {"result", "settings", "host_facts"}
            or not isinstance(record["settings"], str) or result.get("code") != 0 or result.get("needed") is not None
            or result.get("assigned_environment") != [] or not isinstance(result.get("argv"), list) or not result["argv"]
            or any(not isinstance(arg, str) or not arg or "\0" in arg for arg in result["argv"])
            or not isinstance(result.get("outputs"), list)):
        raise ValueError("the side's launch did not resolve completely, or it assigns server environment")
    emitted = []
    for item in result["outputs"]:
        if isinstance(item, dict) and item.get("channel") == "stdout" and isinstance(item.get("text"), str):
            try:
                value = json.loads(item["text"])
            except ValueError:
                continue
            if isinstance(value, dict) and "effective" in value:
                emitted.append(value)
    if len(emitted) != 1:
        raise ValueError("the side's launch did not report exactly one resolution")
    emitted, argv = emitted[0], result["argv"]
    if (emitted.get("settings") != settings or emitted.get("telemetry") != events or argv[0] != binary(config, side)
            or argument(argv, "--ssd-settings") != [settings] or argument(argv, TELEMETRY) != [events]):
        raise ValueError("the resolved launch does not use this job's server, settings and event files")
    strategy = emitted.get("strategy")
    if (not isinstance(strategy, dict) or not isinstance(strategy.get("inputs"), dict)
            or strategy["inputs"].get("characteristics_sha256") != characteristics_sha256):
        raise ValueError("the side's launch did not derive its strategy from this job's machine characteristics")
    effective = emitted.get("effective")
    if (not isinstance(effective, dict) or set(effective) != {"values", "sources"}
            or not isinstance(effective["values"], dict) or not isinstance(effective["sources"], dict)
            or set(effective["sources"]) != set(effective["values"])
            or any(not isinstance(source, dict) or not isinstance(source.get("origin"), str)
                   or not isinstance(source.get("location"), str) for source in effective["sources"].values())
            or not isinstance(emitted.get("launch"), dict)):
        raise ValueError("the resolved settings document is incomplete")
    resolved = dict(argv=list(argv), scope=list(result.get("scope", [])), settings=settings, telemetry=events,
                    settings_sha256=hashlib.sha256(record["settings"].encode()).hexdigest(), effective=effective,
                    launch=emitted["launch"], strategy=strategy, host_facts=record["host_facts"])
    pinned = [key for key, entry in summary(config, side, model, resolved)["settings"].items()
              if entry["origin"] == "user" and key != "expert.source"]
    if pinned:
        raise ValueError("the launch's settings file pins what the strategy or the engine decides: " + ", ".join(pinned))
    return resolved


def with_events(resolved, name):
    argv = list(resolved["argv"])
    argv[argv.index(TELEMETRY) + 1] = str(PurePosixPath(resolved["telemetry"]).parent / name)
    return argv


def fill(config, side, model, resolved, timed):
    """Give the side's server launches their resolved argv, each server its own event file.

    A tier's short preflight and timed runs get `native-events-preflight-<tier>`
    and `native-events-<tier>` (timed run k inserts `-r<k>`); the candidate's
    functional request uses its first tier's preflight launch, so the preflight
    can reuse it.
    """
    data = config["sides"][side]["models"][model]
    port = data["launch"]["port"]
    for tier in timed:
        data["benchmarks"][tier].update(argv=with_events(resolved, "native-events-" + tier + ".jsonl"),
                                        preflight_argv=with_events(resolved, "native-events-preflight-" + tier + ".jsonl"),
                                        port=port)
    if side == "candidate":
        smoke = data["smoke"]
        smoke.update(argv=list(data["benchmarks"][timed[0]]["preflight_argv"]) if timed else list(resolved["argv"]), port=port)
        if smoke.get("observations"):
            smoke["observations"]["path"] = argument(smoke["argv"], TELEMETRY)[0]
    data["derived"] = resolved


def retained(root, config, needs):
    """Replay the job's machine characteristics and every resolved launch from retained stage output."""
    root = Path(root)
    if sealed(config) is not None:
        machine = machine_sealed(config)
    else:
        machine = dict(machine_read(root / "machine"), side=profile_side(config, needs))
        if load(root / "machine" / "profile.json") != machine:
            raise ValueError("retained machine profile differs from its stage output")
    for side in job_sides(config, needs):
        for model in launch_models(config, needs, side):
            folder = root / "launch" / (side + "-" + model)
            resolved = resolution(config, side, model, final_record(folder), machine["sha256"])
            if load(folder / "resolved.json") != resolved:
                raise ValueError("retained resolved launch differs from its stage output")
            fill(config, side, model, resolved, tiers(needs, model))
    return machine


def neutral(config, side, model, value):
    """Side-specific paths as placeholders, so both sides' launches compare."""
    paths = config["sides"][side]["paths"]
    names = sorted(((paths["frozen"], "<sealed-build>"), (scratch(config, side), "<scratch>"),
                    (paths["source"], "<source>"),
                    (config["sides"][side]["models"][model]["settings"]["runtime_path"], "<launch-settings>")),
                   key=lambda item: -len(item[0]))
    pattern = re.compile("|".join(re.escape(path) + r"(?![^/:\s=\"])" for path, _ in names))
    placeholders = dict(names)

    def replace(item):
        if isinstance(item, str):
            return pattern.sub(lambda match: placeholders[match.group(0)], item)
        if isinstance(item, list):
            return [replace(entry) for entry in item]
        if isinstance(item, dict):
            return {key: replace(entry) for key, entry in item.items()}
        return item

    return replace(value)


def summary(config, side, model, resolved):
    """The side's derived launch as a reviewer reads it: each setting and thread pool with its origin.

    Origin is `strategy` (derived from the machine characteristics, with the
    strategy's reason), `user` (the reviewed launch request's settings file) or
    `launcher` (the registry default or the launcher's model profile).
    """
    from ci.selected_receipt import normalize
    user = config["sides"][side]["models"][model]["settings"]["runtime_path"]
    frozen = config["sides"][side]["paths"]["frozen"]
    chosen = resolved["strategy"].get("chosen")
    chosen = chosen if isinstance(chosen, dict) else {}
    reasons = chosen.get("settings") if isinstance(chosen.get("settings"), dict) else {}
    settings = {}
    for key in sorted(resolved["effective"]["values"]):
        source = resolved["effective"]["sources"][key]
        # A settings file entry's location is the file, then ":" and its line.
        reviewed = source["location"].rpartition(":")[0] == user
        origin = ("strategy" if source["location"] == "strategy" else
                  "user" if source["origin"] == "cli" or source["origin"] == "file" and reviewed else "launcher")
        # Values are spelled as the receipt's typed values: only the sealed build is a placeholder.
        settings[key] = dict(value=normalize(resolved["effective"]["values"][key], frozen), origin=origin)
        if origin == "strategy":
            reason = reasons.get(key)
            settings[key]["reason"] = reason.get("reason") if isinstance(reason, dict) else None
    pools = chosen.get("threads") if isinstance(chosen.get("threads"), dict) else {}
    threads = {name: dict(value=resolved["launch"].get(field), origin="strategy" if name in pools else "launcher")
               for name, field in (("decode", "decode_threads"), ("batch", "batch_threads"), ("cpus", "cpus"))}
    notes = resolved["strategy"].get("notes")
    value = neutral(config, side, model, dict(
        characteristics_sha256=resolved["strategy"]["inputs"]["characteristics_sha256"], threads=threads,
        server_args=list(chosen.get("server_args", [])), notes=list(notes) if isinstance(notes, list) else [],
        argv=["<private-events>" if arg == resolved["telemetry"] else arg for arg in resolved["argv"]]))
    return dict(value, settings=settings)


def changes(base, candidate):
    """Names of what the candidate's derived launch sets differently from the base's."""
    def values(launch):
        return {key: (entry["value"], entry["origin"]) for key, entry in launch["settings"].items()}

    before, after = values(base), values(candidate)
    changed = [key for key in sorted(set(before) | set(after)) if before.get(key) != after.get(key)]
    changed += ["threads." + name for name in ("decode", "batch", "cpus") if base["threads"][name] != candidate["threads"][name]]
    changed += [name for name in ("server_args", "argv") if base[name] != candidate[name]]
    return changed
