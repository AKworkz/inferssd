# SPDX-License-Identifier: Apache-2.0
"""Launches resolved on the device by each side's own framework; no device, broker or server.

The resolution records are real: the sandbox program runs the framework's own
the typed launch's `--dry-run` on a header-only GGUF and the x86 machine
characteristics fixture, then the paths are spelled as the Linux sandbox sees them.
"""

import copy
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "bench"))
sys.path.insert(0, str(ROOT / "framework/tests"))
from ci import compare, selected_preflight, staged_launch
from ci.contract import load, save
from ci.selected_receipt import normalize
from test_ci_operator_controller import functional_artifacts
from test_ci_performance_series import POLICY, protocol_for, record, result
from test_strategy_inputs import muse_like, write_gguf

from inferssd.machine.characteristics import dumps, load as load_machine

MACHINE = dumps(load_machine(ROOT / "framework/tests/fixtures/machine/linux-x86-rtx5090.json"))
CHARACTERISTICS = compare.hashlib.sha256(MACHINE.encode()).hexdigest()
REQUEST = dict(profile="muse", port=18080, extra=[])
BASE, CANDIDATE = "a" * 40, "c" * 40


def config_for(models=("muse",)):
    """A Linux job configuration as the sandbox sees it."""
    sides = {}
    for side, revision in (("base", BASE), ("candidate", CANDIDATE)):
        launch = dict(environment={}, cwd="/scratch/" + side, timeout_s=600, request_timeout_s=300,
                      os_page_cache_treatment="recorded, not forced")
        sides[side] = dict(revision=revision, paths=dict(platform="linux", source="/source/" + side,
                           build="/scratch/" + side, cwd="/scratch/" + side, frozen="/build/" + side),
                           build=dict(binary_relative="bin/llama-server"),
                           models={model: dict(launch=dict(REQUEST),
                                               settings=dict(runtime_path="/models/settings-" + model + "-" + side + ".ssd"),
                                               template=dict(runtime_path="/source/" + side + "/chat.jinja", sha256="7" * 64),
                                               model_identity=dict(shards=[dict(runtime_path="/models/" + model + "/model.gguf",
                                                                                sha256="8" * 64)]),
                                               smoke=dict(launch, model=model), benchmarks={"512": dict(launch)})
                                   for model in models})
    return dict(device="cloud-5090", paths=sides["candidate"]["paths"], sides=sides,
                host=dict(hardware="synthetic", os=dict(sysname="Linux", release="6.0", machine="x86_64"),
                          resources=dict(cpus_allowed_list="0-3", memlock_bytes=1,
                                         cgroup_limits=dict(memory_max=str(64 << 30), memory_swap_max="0"))))


def seal(config, text=MACHINE, host_path="/opt/inferssd-ci-integrations/" + "1" * 32 + "-machine.json"):
    """A PR job's characteristics, sealed by the operator from the device's most recent baseline job."""
    config["machine_characteristics"] = dict(sha256=compare.hashlib.sha256(text.encode()).hexdigest(), text=text,
                                             job_id="b" * 32, commit=BASE, host_path=host_path,
                                             runtime_path="/models/machine.json")
    return config


def framework(tmp_path, side, patch=None):
    """The side's framework source; PATCH replaces one line of its strategy."""
    if patch is None:
        return ROOT / "framework/src"
    target = tmp_path / ("framework-" + side)
    shutil.copytree(ROOT / "framework/src", target, ignore=shutil.ignore_patterns("__pycache__"))
    derive = target / "inferssd/strategy/derive.py"
    text = derive.read_text()
    assert text.count(patch[0]) == 1
    derive.write_text(text.replace(*patch))
    return target


def sandbox_record(tmp_path, side="candidate", model="muse", patch=None, settings_lines=(), label=None):
    """Run the sandbox resolution program for real, then spell its paths as in the sandbox."""
    root = tmp_path / ("tree-" + (label or side))
    spelled = {str(root / "build"): "/build", str(root / "scratch"): "/scratch", str(root / "source"): "/source",
               str(root / "models"): "/models"}
    server = root / "build" / side / "bin/llama-server"
    alias = root / "build" / side / "assets" / (model + "-alias.tsv")
    for path in (server, alias, root / "source" / side / "chat.jinja", root / "scratch" / side / "keep"):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("synthetic\n")
    server.chmod(server.stat().st_mode | stat.S_IXUSR)
    (root / "models" / model).mkdir(parents=True)
    shard = write_gguf(root / "models" / model / "model.gguf", *muse_like())
    settings = root / "models" / ("settings-" + model + "-" + side + ".ssd")
    settings.write_text("inferssd-settings 1\nexpert.source = [\"" + str(alias) + "\"]\n" + "".join(line + "\n" for line in settings_lines))
    machine = root / "scratch/machine/machine.json"
    machine.parent.mkdir(parents=True)
    machine.write_text(MACHINE)
    config = config_for((model,))
    folder = str(root / "scratch" / side / ("launch-" + model))
    request = config["sides"][side]["models"][model]["launch"]
    argv = [sys.executable, "-I", "-c", staged_launch.RESOLVE, str(framework(tmp_path, side, patch)), str(machine),
            json.dumps(staged_launch.runtime_policy(config, folder)), folder, json.dumps([str(shard)]),
            "--platform", "linux", "--model", str(shard), "--settings", str(settings), "--binary", str(server),
            "--template", str(root / "source" / side / "chat.jinja"), "--profile", request["profile"],
            "--port", str(request["port"]), "--prompt-cache", "off"]
    run = subprocess.run(argv, capture_output=True, text=True, timeout=120, env={"PATH": os.environ["PATH"]})
    assert run.returncode == 0, run.stderr
    text = [line for line in run.stdout.splitlines() if line.startswith(staged_launch.MARKER)][-1][len(staged_launch.MARKER):]
    for real, inside in sorted(spelled.items(), key=lambda item: -len(item[0])):
        text = text.replace(real, inside)
    return json.loads(text)


@pytest.fixture(scope="module")
def records(tmp_path_factory):
    """Real resolutions: the base's framework, and a candidate whose strategy derives one setting differently."""
    tmp_path = tmp_path_factory.mktemp("resolutions")
    patch = ('choices["cpu.decode_participants"] = ("0", "the engine measures its decode participants")',
             'choices["cpu.decode_participants"] = ("1", "one decode participant")')
    return dict(base=sandbox_record(tmp_path, "base"), candidate=sandbox_record(tmp_path, "candidate"),
                changed=sandbox_record(tmp_path, "candidate", patch=patch, label="changed"))


def resolved(records, side="candidate", name=None, config=None):
    config = config or config_for()
    return config, staged_launch.resolution(config, side, "muse", copy.deepcopy(records[name or side]), CHARACTERISTICS)


def test_the_sides_own_dry_run_resolves_the_launch_from_the_job_characteristics(records):
    config, value = resolved(records)
    assert value["argv"][0] == "/build/candidate/bin/llama-server"
    assert staged_launch.argument(value["argv"], "--ssd-settings") == ["/scratch/candidate/launch-muse/settings.ssd"]
    assert staged_launch.argument(value["argv"], "-m") == ["/models/muse/model.gguf"]
    assert value["strategy"]["inputs"]["characteristics_sha256"] == CHARACTERISTICS
    assert set(value["host_facts"]["runtime_facts"]) == {"cpus", "gpus", "available_bytes", "memory_budget_bytes"}
    launch = staged_launch.summary(config, "candidate", "muse", value)
    assert launch["settings"]["expert.source"] == dict(value=["<sealed-build>/assets/muse-alias.tsv"], origin="user")
    assert launch["settings"]["cpu.decode_participants"]["origin"] == "strategy"
    assert launch["settings"]["cpu.decode_participants"]["reason"]
    assert launch["settings"]["prefetch.depth"]["origin"] == "launcher"
    assert launch["threads"]["decode"]["origin"] == "strategy" and launch["server_args"] == ["--ctx-checkpoints", "1"]
    # No prompt cache: every timed run starts its server cold on the same prompt.
    assert staged_launch.argument(value["argv"], "--cache-ram") == ["0"] and "--cache-disk" not in value["argv"]
    assert "<private-events>" in launch["argv"] and "/scratch/candidate" not in json.dumps(launch)


def test_each_server_gets_the_resolved_argv_with_its_own_event_file(records):
    config, value = resolved(records)
    staged_launch.fill(config, "candidate", "muse", value, ["512"])
    data = config["sides"]["candidate"]["models"]["muse"]
    spec, smoke = data["benchmarks"]["512"], data["smoke"]
    events = [staged_launch.argument(argv, "--ssd-telemetry")[0] for argv in (spec["preflight_argv"], spec["argv"])]
    assert events == ["/scratch/candidate/launch-muse/native-events-preflight-512.jsonl",
                      "/scratch/candidate/launch-muse/native-events-512.jsonl"]
    assert smoke["argv"] == spec["preflight_argv"] and smoke["port"] == spec["port"] == 18080
    assert selected_preflight.same_launch(smoke, selected_preflight.launch(spec, "candidate", required=True))
    timed = selected_preflight.arm_launch(spec, "candidate", 2, required=True, functional=smoke)
    assert staged_launch.argument(timed["argv"], "--ssd-telemetry") == ["/scratch/candidate/launch-muse/native-events-512-r2.jsonl"]
    assert [arg for arg in timed["argv"] if not arg.endswith(".jsonl")] == [arg for arg in value["argv"] if not arg.endswith(".jsonl")]
    assert data["derived"] == value


@pytest.mark.parametrize("damage", ["other-characteristics", "no-strategy", "environment", "other-server", "two-event-files",
                                    "unfinished", "two-resolutions"])
def test_a_resolution_that_is_not_this_jobs_own_is_refused(records, damage):
    config, value = config_for(), copy.deepcopy(records["candidate"])
    sha = CHARACTERISTICS
    emitted = next(item for item in value["result"]["outputs"] if item["channel"] == "stdout")
    line = json.loads(emitted["text"])
    if damage == "other-characteristics": sha = "0" * 64
    elif damage == "no-strategy": line["strategy"] = {"skipped": "no machine characteristics"}
    elif damage == "environment": value["result"]["assigned_environment"] = [["CUDA_VISIBLE_DEVICES", "1"]]
    elif damage == "other-server": value["result"]["argv"][0] = "/build/base/bin/llama-server"
    elif damage == "two-event-files": value["result"]["argv"] += ["--ssd-telemetry", "/scratch/candidate/x.jsonl"]
    elif damage == "unfinished": value["result"]["code"] = None
    else: value["result"]["outputs"].append(dict(emitted))
    emitted["text"] = json.dumps(line)
    with pytest.raises(ValueError):
        staged_launch.resolution(config, "candidate", "muse", value, sha)


def test_a_reviewed_settings_file_may_not_pin_what_the_strategy_or_the_engine_decides(tmp_path):
    record = sandbox_record(tmp_path, settings_lines=["vram.budget = 0"])
    with pytest.raises(ValueError, match="pins what the strategy or the engine decides: vram.budget"):
        staged_launch.resolution(config_for(), "candidate", "muse", record, CHARACTERISTICS)


def receipt(config, side, value, *, pid, model_sha="8" * 64, template="7" * 64):
    """A comparison receipt shaped as selected_receipt.receipt writes it for a derived launch."""
    data = config["sides"][side]["models"]["muse"]
    staged_launch.fill(config, side, "muse", value, ["512"])
    policy = dict(args=["<argv>"], typed_values=value["effective"]["values"], template_sha256=template,
                  request=dict(data["launch"], settings={"expert.source": value["effective"]["values"]["expert.source"]}))
    return dict(model_identity=dict(shards=[dict(sha256=model_sha)]), build=dict(revision=config["sides"][side]["revision"],
                loaded_libraries=[dict(path="/build/" + side + "/bin/libggml.so", sha256="9" * 64)]),
                server=dict(pid=pid, start_identity=str(pid)),
                host=dict(hardware=config["host"]["hardware"], os=config["host"]["os"], memory_budget_bytes=64 << 30,
                          os_page_cache_treatment="recorded, not forced",
                          effective_settings=dict(native_launch_policy=normalize(policy, "/build/" + side),
                                                  native_resource_policy=config["host"]["resources"],
                                                  ambient_environment={}, system_images=[])),
                derived_launch=staged_launch.summary(config, side, "muse", value))


def cell_inputs(records, candidate="changed", **changes):
    config = config_for()
    _, base = resolved(records, "base", config=config)
    _, head = resolved(records, "candidate", candidate, config=config)
    order = protocol_for()["order"]
    results, receipts = {}, {}
    for position, key in enumerate(compare.schedule(order)):
        results[key] = result(start=position * 100)
        options = changes if key[1] == "candidate" else {}
        receipts[key] = receipt(config, key[1], base if key[1] == "base" else head, pid=position + 1, **options)
    return results, receipts


def history_for(receipts):
    conditions = compare.conditions_digest(compare.history_conditions(receipts[1, "base"]))
    return [record(BASE, conditions=conditions)]


def test_a_pr_whose_strategy_derives_another_setting_is_compared_and_shows_the_change(records):
    results, receipts = cell_inputs(records)
    base, head = receipts[1, "base"], receipts[1, "candidate"]
    typed = [receipt["host"]["effective_settings"]["native_launch_policy"]["typed_values"] for receipt in (base, head)]
    assert typed[0]["cpu.decode_participants"] == 0 and typed[1]["cpu.decode_participants"] == 1
    assert compare.comparable(base) == compare.comparable(head)
    assert compare.history_conditions(base) == compare.history_conditions(head)
    decision = compare.decide_cell(results, receipts, POLICY, protocol=protocol_for(), history=history_for(receipts),
                                   cell=dict(device="cloud-5090", model="muse", tier="512", job_id="f" * 32,
                                             base=BASE, candidate=CANDIDATE))
    assert decision["status"] == "passed" and decision["review_reasons"] == []
    launch = decision["derived_launch"]
    assert launch["changes"] == ["cpu.decode_participants"]
    assert launch["sides"]["base"]["settings"]["cpu.decode_participants"] == dict(
        value=0, origin="strategy", reason="the engine measures its decode participants")
    assert launch["sides"]["candidate"]["settings"]["cpu.decode_participants"] == dict(
        value=1, origin="strategy", reason="one decode participant")
    unchanged = compare.decide_cell(*cell_inputs(records, "candidate"), POLICY, protocol=protocol_for(),
                                    history=history_for(receipts), cell=decision["cell"])
    assert unchanged["derived_launch"]["changes"] == [] and unchanged["status"] == "passed"


@pytest.mark.parametrize("change", [dict(model_sha="0" * 64), dict(template="0" * 64)])
def test_derived_settings_may_differ_but_the_model_and_template_may_not(records, change):
    results, receipts = cell_inputs(records, **change)
    with pytest.raises(ValueError, match="changed between runs"):
        compare.decide_cell(results, receipts, POLICY, protocol=protocol_for(), history=history_for(receipts),
                            cell=dict(device="cloud-5090", model="muse", tier="512", job_id="f" * 32,
                                      base=BASE, candidate=CANDIDATE))


@pytest.mark.parametrize("damage", ["characteristics", "request", "budget", "typed-values", "one-run"])
def test_every_other_condition_still_must_match(records, damage):
    results, receipts = cell_inputs(records)
    head = receipts[3, "candidate"]
    policy = head["host"]["effective_settings"]["native_launch_policy"]
    if damage == "characteristics": head["derived_launch"]["characteristics_sha256"] = "0" * 64
    elif damage == "request": policy["request"]["extra"] = ["--temp", "0.5"]
    elif damage == "budget": head["host"]["memory_budget_bytes"] = 1
    elif damage == "typed-values": policy["typed_values"]["cpu.decode_participants"] = 2
    else: head["derived_launch"]["threads"]["decode"]["value"] = 99
    with pytest.raises(ValueError):
        compare.decide_cell(results, receipts, POLICY, protocol=protocol_for(), history=history_for(receipts),
                            cell=dict(device="cloud-5090", model="muse", tier="512", job_id="f" * 32,
                                      base=BASE, candidate=CANDIDATE))


def test_baseline_history_is_keyed_on_machine_os_model_template_and_machine_policy(records):
    _, receipts = cell_inputs(records)
    base = receipts[1, "base"]
    key = compare.conditions_digest(compare.history_conditions(base))
    policy = dict(memory_budget_bytes=None, memory_max=str(64 << 30), memlock_bytes=1, cpus="0-3")
    assert compare.history_conditions(base)["machine_policy"] == policy
    # Derived settings are not part of the key, and neither is what the launch derived them to.
    other = copy.deepcopy(base)
    other["host"]["effective_settings"]["native_launch_policy"]["typed_values"]["cpu.io_cpus"] = [9]
    other["derived_launch"]["threads"]["decode"]["value"] = 99
    assert compare.conditions_digest(compare.history_conditions(other)) == key
    resources = ("host", "effective_settings", "native_resource_policy")
    for path, value in ((("host", "os"), dict(sysname="Linux", release="6.1", machine="x86_64")),
                        (("host", "hardware"), "another"), (("model_identity",), dict(shards=[])),
                        (("host", "effective_settings", "native_launch_policy", "template_sha256"), "0" * 64),
                        ((*resources, "cgroup_limits", "memory_max"), str(32 << 30)),
                        ((*resources, "memlock_bytes"), 2), ((*resources, "cpus_allowed_list"), "0-7")):
        changed = copy.deepcopy(base)
        target = changed
        for name in path[:-1]:
            target = target[name]
        target[path[-1]] = value
        assert compare.conditions_digest(compare.history_conditions(changed)) != key
    # The Mac executor states a sampled budget and pins no CPUs.
    mac = dict(max_processes=64, memory_budget_bytes=24 << 30, host_reserve_bytes=8 << 30)
    assert compare.machine_policy(mac) == dict(memory_budget_bytes=24 << 30, memory_max=None, memlock_bytes=None,
                                               cpus=None)
    assert compare.machine_policy(dict(mac, memory_budget_bytes=16 << 30)) != compare.machine_policy(mac)
    assert compare.machine_policy(dict(mac, host_reserve_bytes=1)) == compare.machine_policy(mac)


def fake_framework(tmp_path, code):
    """A stand-in `inferssd.cli.main` that writes FILE and returns CODE, to exercise the program alone."""
    package = tmp_path / "fake" / "inferssd"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("")
    (package / "cli.py").write_text(
        "import os\n"
        "def main(argv):\n"
        "    output = argv[argv.index('--output') + 1]\n"
        "    os.makedirs(os.path.dirname(output), exist_ok=True)\n"
        "    open(output, 'w').write(" + repr(MACHINE) + ")\n"
        "    print('summary line from the framework')\n"
        "    return " + str(code) + "\n")
    return package.parent


@pytest.mark.parametrize("code", [0, 1, 2])
def test_the_profile_program_keeps_the_exact_file_and_accepts_recorded_failed_sections(tmp_path, code):
    output = tmp_path / "work" / "machine.json"
    run = subprocess.run([sys.executable, "-I", "-c", staged_launch.PROFILE, str(fake_framework(tmp_path, code)),
                          str(output), "--build", "/build/base", "--model", "/models/muse/model.gguf"],
                         capture_output=True, text=True, timeout=60)
    folder = tmp_path / "stage"
    folder.mkdir()
    (folder / "stdout.log").write_text(run.stdout)
    save(folder / "stage.json", dict(returncode=run.returncode, cleanup_confirmed=True))
    if code == 2:
        assert run.returncode == 2
        with pytest.raises(RuntimeError):
            staged_launch.machine_read(folder)
        return
    machine = staged_launch.machine_read(folder)
    assert machine["sha256"] == CHARACTERISTICS and machine["returncode"] == code
    assert machine["sections"] == {name: section["status"] for name, section in json.loads(MACHINE)["sections"].items()}


class Session:
    """Records the stages it is asked to run and answers them from canned stage output."""

    def __init__(self, outputs):
        self.outputs, self.started = outputs, []

    def start(self, stage_id, kind, argv, environment, cwd, timeout):
        self.started.append(dict(stage_id=stage_id, kind=kind, argv=argv, cwd=cwd, timeout=timeout))
        return stage_id

    def wait(self, stage, deadline):
        return dict(returncode=0, cleanup_confirmed=True)

    def output(self, stage, stream, offset):
        data = self.outputs[stage].encode() if stream == "stdout" and offset == 0 else b""
        return data, offset + len(data), True


NEEDS = dict(models=["muse"], performance={"muse": dict(required=True, tiers=["512"], output_tokens=256)})


def baseline_job():
    from ci.selection_contract import performance_protocol
    return dict(config_for(), performance_protocol=performance_protocol("baseline"),
                selection_plan=dict(purpose="baseline-history"))


def test_a_baseline_job_profiles_once_and_replays_from_its_stage_output(records, tmp_path):
    job = baseline_job

    config = job()
    machine_line = staged_launch.MARKER + json.dumps(dict(returncode=0, characteristics=MACHINE))
    session = Session({"machine-profile": "framework output\n" + machine_line + "\n",
                       "candidate-muse-launch": staged_launch.MARKER + json.dumps(records["candidate"]) + "\n"})
    output = tmp_path / "out"
    machine = staged_launch.profile(session, config, "candidate", ["muse"], output / "machine")
    value = staged_launch.resolve(session, config, "candidate", "muse", machine, output / "launch/candidate-muse")
    staged_launch.fill(config, "candidate", "muse", value, ["512"])
    assert [stage["stage_id"] for stage in session.started] == ["machine-profile", "candidate-muse-launch"]
    profile, launch = session.started
    assert profile["argv"][:4] == ["/usr/bin/python3", "-I", "-c", staged_launch.PROFILE]
    assert profile["argv"][4:] == ["/source/candidate/framework/src", "/scratch/machine/machine.json", "--build",
                                   "/build/candidate", "--model", "/models/muse/model.gguf"]
    assert launch["argv"][4:6] == ["/source/candidate/framework/src", "/scratch/machine/machine.json"]
    assert launch["argv"][7:9] == ["/scratch/candidate/launch-muse", '["/models/muse/model.gguf"]']
    assert launch["cwd"] == "/scratch/candidate" and launch["kind"] == "test"
    assert json.loads(launch["argv"][6])["memory_budget_bytes"] == 64 << 30
    # The request leaves the sizes to the strategy, and the launch keeps no prompt cache.
    assert launch["argv"][-6:] == ["--profile", "muse", "--port", "18080", "--prompt-cache", "off"]
    assert machine == dict(load(output / "machine/profile.json")) and machine["side"] == "candidate"
    # The hosted report carries the profiled file, so the operator can store it with the history.
    assert staged_launch.profiled(output / "machine") == MACHINE
    unfilled = job()
    replayed = staged_launch.retained(output, unfilled, NEEDS)
    assert replayed == machine and unfilled["sides"]["candidate"] == config["sides"]["candidate"]
    (output / "launch/candidate-muse/stdout.log").write_text(
        staged_launch.MARKER + json.dumps(records["changed"]) + "\n")
    with pytest.raises(ValueError, match="resolved launch differs"):
        staged_launch.retained(output, job(), NEEDS)


def test_a_pr_job_resolves_both_sides_from_the_sealed_characteristics_without_profiling(records, tmp_path):
    from ci.selection_contract import performance_protocol

    def job(text=MACHINE):
        return seal(dict(config_for(), performance_protocol=performance_protocol()), text)

    config = job()
    session = Session({"base-muse-launch": staged_launch.MARKER + json.dumps(records["base"]) + "\n",
                       "candidate-muse-launch": staged_launch.MARKER + json.dumps(records["changed"]) + "\n"})
    output = tmp_path / "out"
    machine = staged_launch.machine_sealed(config)
    assert machine == dict(sha256=CHARACTERISTICS, baseline=dict(job_id="b" * 32, commit=BASE),
                           sections={name: value["status"] for name, value in json.loads(MACHINE)["sections"].items()})
    for side in ("base", "candidate"):
        value = staged_launch.resolve(session, config, side, "muse", machine, output / "launch" / (side + "-muse"))
        staged_launch.fill(config, side, "muse", value, ["512"])
    # No profile stage: both sides read the read-only file root staged, never one in scratch.
    assert [stage["stage_id"] for stage in session.started] == ["base-muse-launch", "candidate-muse-launch"]
    assert [stage["argv"][5] for stage in session.started] == ["/models/machine.json"] * 2
    unfilled = job()
    assert staged_launch.retained(output, unfilled, NEEDS) == machine and unfilled["sides"] == config["sides"]
    assert not (output / "machine").exists()
    # Sealed characteristics of another machine: the resolutions do not name them.
    with pytest.raises(ValueError, match="machine characteristics"):
        staged_launch.retained(output, job(json.dumps(dict(json.loads(MACHINE), note="another machine"))), NEEDS)
    tampered = job()
    tampered["machine_characteristics"]["text"] += " "
    with pytest.raises(ValueError, match="differ from their hash"):
        staged_launch.retained(output, tampered, NEEDS)


def test_a_pr_job_adopts_only_the_exact_sealed_file_root_staged(tmp_path, monkeypatch):
    from ci import staged_build
    from ci.selection_contract import performance_protocol
    monkeypatch.setattr(staged_build, "protected", lambda path: Path(path))
    staged = tmp_path / "machine.json"
    config = seal(dict(config_for(), performance_protocol=performance_protocol()), host_path=str(staged))
    for raw in (MACHINE.encode(), MACHINE.encode() + b" "):
        if staged.exists():
            staged.chmod(0o600)
        staged.write_bytes(raw)
        staged.chmod(0o444)
        if raw == MACHINE.encode():
            assert staged_launch.adopt(config) == staged_launch.machine_sealed(config)
        else:
            with pytest.raises(ValueError, match="differ from the sealed file"):
                staged_launch.adopt(config)
    staged.chmod(0o600)
    staged.unlink()
    with pytest.raises(OSError):
        staged_launch.adopt(config)


def derived_job(tmp_path, monkeypatch, baseline=False):
    """The selected-integration fixture with every launch derived on the device from real resolutions.

    A PR job takes the sealed characteristics, which root staged at `staged`;
    a baseline job of one commit profiles the machine itself.
    """
    from test_ci_selected_preflight import setup, synthetic_receipt
    from test_ci_selection import POLICY as SELECTION_POLICY
    from ci import impact, selected_integration as execution, selected_preflight as preflight, staged_build
    from ci.selection_contract import performance_protocol
    patch = ('choices["cpu.decode_participants"] = ("0", "the engine measures its decode participants")',
             'choices["cpu.decode_participants"] = ("1", "one decode participant")')
    outputs = {"machine-profile": staged_launch.MARKER + json.dumps(dict(returncode=1, characteristics=MACHINE))}
    for side in ("base", "candidate"):
        for model in ("deepseek", "muse"):
            changed = patch if (side, model) == ("candidate", "muse") else None
            outputs[side + "-" + model + "-launch"] = staged_launch.MARKER + json.dumps(
                sandbox_record(tmp_path / "records", side, model, patch=changed, label=side + "-" + model))
    staged = tmp_path / "staged-machine.json"
    staged.write_text(MACHINE)
    staged.chmod(0o444)
    monkeypatch.setattr(staged_build, "protected", lambda path: Path(path))
    value = impact.plan([], SELECTION_POLICY, baseline=True, base=CANDIDATE, head=CANDIDATE,
                        candidate=CANDIDATE) if baseline else None

    def derive(config):
        config["host"] = config_for()["host"]
        if baseline:
            config.update(performance_protocol=performance_protocol("baseline"),
                          validation_mode=execution.PERFORMANCE_MODE, execution_devices=["cloud-5090"])
        else:
            seal(config, host_path=str(staged))
        for side, spec in config["sides"].items():
            spec["build"]["toolchain"] = {"CMAKE_CXX_COMPILER": "/usr/bin/c++"}
            for model, data in spec["models"].items():
                data.update(launch=dict(REQUEST), settings=dict(runtime_path="/models/settings-" + model + "-" + side + ".ssd"),
                            template=dict(runtime_path="/source/" + side + "/chat.jinja"),
                            model_identity=dict(shards=[dict(runtime_path="/models/" + model + "/model.gguf")]))
                for launch in (data["smoke"], *data["benchmarks"].values()):
                    for key in ("argv", "preflight_argv", "port"):
                        launch.pop(key, None)

    args, config, manifest, needs, events, sessions = setup(tmp_path, monkeypatch, change=derive, value=value)
    stages = []

    def run_stage(session, stage_id, argv, cwd, timeout, output):
        assert manifest["stages"][stage_id]["executables"] == [argv[0]] and manifest["stages"][stage_id]["cwd"] == cwd
        if stage_id.endswith("-launch"):
            assert argv[5] == staged_launch.machine_path(config)
        stages.append(stage_id)
        save(output / "stage.json", dict(returncode=0, cleanup_confirmed=True))
        (output / "stdout.log").write_text("framework output\n" + outputs[stage_id] + "\n")

    def receipt(config, manifest, side, model, spec, build, output, *, destination=None):
        value = synthetic_receipt(model, side, spec, output)
        launch = staged_launch.summary(config, side, model, config["sides"][side]["models"][model]["derived"])
        value["host"]["effective_settings"]["native_launch_policy"]["typed_values"] = {
            key: entry["value"] for key, entry in launch["settings"].items()}
        value["derived_launch"] = launch
        save(destination if destination is not None else output / "comparison-receipt.json", value)
        return value

    builds = []
    real_build = execution.run_build
    monkeypatch.setattr(execution, "run_build", lambda session, side, *rest, **options: (
        builds.append(len(stages)) or real_build(session, side, *rest, **options)))
    monkeypatch.setattr(staged_launch, "run_stage", run_stage)
    monkeypatch.setattr(preflight, "receipt", receipt)
    monkeypatch.setattr(execution, "receipt", receipt)
    return args, config, needs, events, stages, builds, execution, staged


SECTIONS = {name: value["status"] for name, value in json.loads(MACHINE)["sections"].items()}


def test_a_pr_job_resolves_each_side_from_the_sealed_characteristics_and_reports_what_changed(tmp_path, monkeypatch):
    args, config, needs, events, stages, builds, execution, _ = derived_job(tmp_path, monkeypatch)
    assert execution.run(args) == 0
    report = load(args.output / "integration.json")
    # No machine profile: the base resolves before the candidate is built, from the sealed file.
    assert stages == ["base-deepseek-launch", "base-muse-launch", "candidate-deepseek-launch", "candidate-muse-launch"]
    assert builds == [0, 2]
    assert report["machine_profile"] == dict(sha256=CHARACTERISTICS, sections=SECTIONS,
                                             baseline=dict(job_id="b" * 32, commit=BASE))
    assert not (args.output / "machine").exists()
    assert args.stages == [side + "-" + model + "-r" + run + "-benchmark-512" for model in ("deepseek", "muse")
                           for side, run in (("candidate", "1"), ("base", "1"), ("candidate", "2"), ("candidate", "3"))]
    cells = {cell["model"]: cell["performance"][0] for cell in report["cells"]}
    assert all(cell["status"] == "passed" for cell in cells.values())
    assert cells["muse"]["derived_launch"]["changes"] == ["cpu.decode_participants"]
    assert cells["deepseek"]["derived_launch"]["changes"] == []
    smoke = ("smoke", "candidate-smoke-muse")
    assert smoke in events and events.index(smoke) > 0
    replayed = load(args.config)
    staged_launch.retained(args.output, replayed, needs)
    assert replayed["sides"]["candidate"]["models"]["muse"]["smoke"]["argv"] == \
        config["sides"]["candidate"]["models"]["muse"]["smoke"]["argv"]


def test_a_baseline_job_profiles_the_machine_once_with_the_commit_it_measures(tmp_path, monkeypatch):
    args, config, needs, events, stages, builds, execution, _ = derived_job(tmp_path, monkeypatch, baseline=True)
    assert execution.run(args) == 0
    report = load(args.output / "integration.json")
    assert stages == ["machine-profile", "candidate-deepseek-launch", "candidate-muse-launch"]
    assert builds == [0]
    assert report["status"] == "baseline"
    assert report["machine_profile"] == dict(sha256=CHARACTERISTICS, returncode=1, side="candidate", sections=SECTIONS)
    assert staged_launch.profiled(args.output / "machine") == MACHINE
    replayed = load(args.config)
    assert staged_launch.retained(args.output, replayed, needs) == report["machine_profile"]


def test_a_pr_job_whose_staged_characteristics_differ_from_the_sealed_file_runs_no_resolution(tmp_path, monkeypatch):
    args, config, needs, events, stages, builds, execution, staged = derived_job(tmp_path, monkeypatch)
    staged.chmod(0o600)
    staged.write_text(MACHINE + " ")
    staged.chmod(0o444)
    assert execution.run(args) == 1
    report = load(args.output / "integration.json")
    assert report["status"] == "failed" and report["cleanup_confirmed"]
    assert "staged machine characteristics differ from the sealed file" in report["error"]
    assert stages == [] and not [event for event in events if event[0] in ("smoke", "timed")]


def test_a_failed_resolution_fails_the_job_before_any_server_starts(tmp_path, monkeypatch):
    args, config, needs, events, stages, builds, execution, _ = derived_job(tmp_path, monkeypatch)
    real = staged_launch.resolution

    def refused(config, side, model, record, sha):
        if (side, model) == ("candidate", "muse"):
            raise ValueError("the side's launch did not derive its strategy from this job's machine characteristics")
        return real(config, side, model, record, sha)

    monkeypatch.setattr(staged_launch, "resolution", refused)
    assert execution.run(args) == 1
    report = load(args.output / "integration.json")
    assert report["status"] == "failed" and "machine characteristics" in report["error"] and report["cleanup_confirmed"]
    assert not [event for event in events if event[0] in ("smoke", "timed")]


def test_the_report_shows_each_sides_derived_thread_pools_and_the_changed_settings(records):
    import io
    from ci import selected_controller
    results, receipts = cell_inputs(records)
    entry = compare.decide_cell(results, receipts, POLICY, protocol=protocol_for(), history=history_for(receipts),
                                cell=dict(device="cloud-5090", model="muse", tier="512", job_id="f" * 32,
                                          base=BASE, candidate=CANDIDATE))
    machine = dict(sha256=CHARACTERISTICS, baseline=dict(job_id="b" * 32, commit=BASE),
                   sections=dict(cpu="ok", gpus="failed"))
    stream = io.StringIO()
    selected_controller.operator_summary(stream, dict(passed=True, status="passed", full_impact_coverage_complete=False,
        cells=[dict(device="cloud-5090", model="muse", status="passed", machine_profile=machine,
                    performance=[dict(tier="512", **entry)]),
               dict(device="dgx-spark", model="muse", status="baseline", performance=[dict(tier="512", **entry)],
                    machine_profile=dict(sha256="9" * 64, returncode=0, side="candidate", sections=dict(cpu="ok")))]))
    text = stream.getvalue()
    assert "### Derived launches" in text and "the way `inferssd serve` does" in text
    assert ("- cloud-5090: machine characteristics SHA `" + CHARACTERISTICS + "`, taken from the baseline job of `"
            + BASE + "`; sections not ok: gpus failed.") in text
    assert "- dgx-spark: machine characteristics SHA `" + "9" * 64 + "`, profiled by this baseline job; " in text
    assert "| cloud-5090 | muse | 512 | cpu.decode_participants | 0 (strategy) | 1 (strategy) |" in text
    assert "| cloud-5090 | muse | 512 | threads.decode | " in text


def test_the_hosted_replay_binds_a_functional_receipt_to_its_resolved_launch(records, tmp_path):
    from ci import selected_aggregate
    root, folder, cfg, cell = functional_artifacts(tmp_path)
    config = config_for()
    _, value = resolved(records, config=config)
    data = cfg["sides"]["candidate"]["models"]["deepseek"]
    data.update(derived=value, settings=config["sides"]["candidate"]["models"]["muse"]["settings"])
    cfg["sides"]["candidate"]["paths"] = config["sides"]["candidate"]["paths"]
    native = normalize(value["effective"]["values"], "/build/candidate")
    (folder / "stderr.log").write_text("[ssd.settings] " + json.dumps(dict(values=native)) + "\n")
    receipt = load(folder / "runtime-receipt.json")
    launch = staged_launch.summary(cfg, "candidate", "deepseek", value)
    receipt["host"]["effective_settings"]["native_launch_policy"]["typed_values"] = native
    receipt["derived_launch"] = launch
    for tampered in (None, "origin"):
        if tampered:
            receipt["derived_launch"]["settings"]["cpu.io_cpus"]["origin"] = "launcher"
        (folder / "runtime-receipt.json").write_text(json.dumps(receipt))
        cell["runtime_receipt_sha256"] = compare.digest(folder / "runtime-receipt.json")
        if tampered:
            with pytest.raises(ValueError, match="resolved launch"):
                selected_aggregate.functional_retained(root, cfg, "deepseek", cell)
        else:
            selected_aggregate.functional_retained(root, cfg, "deepseek", cell)


@pytest.mark.parametrize("sizes", [{}, dict(context=4096), dict(context=8192, batch=512, ubatch=512)])
def test_a_request_pins_only_the_sizes_it_names(sizes):
    request = dict(REQUEST, **sizes)
    assert staged_launch.request_checked(request) == request
    for bad in (dict(request, context=0), dict(request, batch="512"), dict(request, threads=4),
                {key: value for key, value in request.items() if key != "port"}):
        with pytest.raises(ValueError, match="optionally context, batch and ubatch"):
            staged_launch.request_checked(bad)
