# SPDX-License-Identifier: Apache-2.0
"""The machine profile and the launch strategy in inferssd launch.

Recorded dry-runs use a real characteristics fixture and a synthetic GGUF
header; live paths use fake hosts and fake probes. Nothing here runs a probe,
a server or nvidia-smi.
"""

from __future__ import annotations

import json
from dataclasses import asdict, replace
from functools import partial
from pathlib import Path

import pytest
from test_engine_runtime import Effects, Owner, policy
from test_launch_platforms import golden
from test_machine_identity import linux_tree, nvme_key, write
from test_machine_probes import PATHS, FakeRunner, replies
from test_machine_profile import Clock, linux_machine
from test_strategy_inputs import SMI_APPS, SMI_GPUS, RuntimeHost, deepseek_like, write_gguf
from test_typed_launch import setup

from inferssd.configuration.settings import parse
from inferssd.contracts import ContractError
from inferssd.engine import command as engine_command
from inferssd.engine import runtime
from inferssd.launch import machine as launch_machine
from inferssd.launch.live import LiveArtifacts
from inferssd.launch.machine import MachineSource, characteristics
from inferssd.launch.prepare import prepare
from inferssd.launch.recorded import RecordedArtifacts
from inferssd.launch.resources import EngineResources, probe_paths
from inferssd.launch.typed import LaunchSpec, resolve_typed
from inferssd.launch.typed_command import main
from inferssd.launch.types import Fact, LaunchInput, Reply, Threads
from inferssd.machine.characteristics import dumps, load
from inferssd.machine.identity import Command
from inferssd.machine.store import LockBusy, default_path
from inferssd.strategy.inputs import model_facts_json, shard_facts

MACHINES = Path(__file__).parent / "fixtures" / "machine"
X86_FILE = MACHINES / "linux-x86-rtx5090.json"
X86 = json.loads(X86_FILE.read_text())
X86_GPU = "GPU-00000000-0000-4000-8000-000000000016"
X86_DRIVE = X86["storage"][0]["drive_key"]
STRATEGY_FACTS = {"machine_characteristics", "model_facts", "model_drive_key", "runtime_facts"}


def runtime_record(cpus=range(61), gpus=((X86_GPU, 30 << 30, 0),)):
    rows = [{"uuid": u, "free_bytes": free, "processes": n} for u, free, n in gpus]
    return {"cpus": list(cpus), "gpus": rows, "available_bytes": 48 << 30}


def strategic(tmp_path, **runtime_options):
    """The typed-launch fixture, recorded on the x86 machine with its characteristics."""
    inputs, spec, host, adapter, out, _ = setup(tmp_path)
    host.update(
        machine_characteristics=X86,
        runtime_facts=runtime_record(**runtime_options),
        model_drive_key=X86_DRIVE,
    )
    return inputs, replace(spec, strategy=True), host, adapter, out


def resolved(inputs, spec, adapter, out, platform="linux"):
    outputs = []
    result = prepare(
        inputs,
        resolver=partial(
            resolve_typed, spec=spec, platform=platform, directory=str(out), memory=4 << 30
        ),
        observe=adapter,
        emit=outputs.append,
    )
    stdout = [json.loads(o.text) for o in outputs if o.channel == "stdout"]
    stderr = "".join(o.text for o in outputs if o.channel == "stderr")
    return result, (stdout[-1] if stdout else None), stderr


def native(out, spec):
    text = (out / "settings.ssd").read_text()
    return parse(text, "settings.ssd", overrides=spec.overrides, platform="linux")


def test_recorded_strategy_fills_what_the_user_left_unset(tmp_path):
    inputs, spec, _, adapter, out = strategic(tmp_path)
    result, line, stderr = resolved(inputs, spec, adapter, out)
    assert result.code == 0 and stderr == ""
    assert STRATEGY_FACTS <= {fact.name for fact in result.observations}
    argv = list(result.argv)
    # 61 uniform CPUs: 60 unpinned threads, one left for the I/O thread.
    assert argv[argv.index("-t") : argv.index("-t") + 4] == ["-t", "60", "-tb", "60"]
    assert "-C" not in argv and argv[-2:] == ["--ctx-checkpoints", "1"]
    # With the CUDA GPU a prompt is prefilled in one pass as large as the context.
    assert argv[argv.index("-c") : argv.index("-c") + 6] == [
        "-c",
        "8192",
        "-b",
        "8192",
        "-ub",
        "8192",
    ]
    settings = native(out, spec)
    for key in ("cpu.io_cpus", "cpu.decode_participants", "place.costs"):
        item = settings.at(key)
        assert item.origin == "file" and item.receipt.startswith(" source=strategy; input=")
    assert sorted(dict(settings.at("place.costs").value)) == [
        "cpu_fixed_us",
        "cpu_route_us",
        "drive_gbps",
        "link_gbps",
        "pageable_gbps",
    ]
    # The GPU is usable: vram.budget keeps its default, and the CLI's prefetch.depth stays.
    assert settings.at("vram.budget").origin == "default"
    assert settings.at("prefetch.depth").origin == "cli"
    receipt = json.loads((out / "strategy.json").read_text())
    assert line["strategy"] == receipt
    assert line["launch"]["decode_threads"] == line["launch"]["batch_threads"] == 60
    assert receipt["inputs"]["model_drive"] == X86_DRIVE
    assert receipt["inputs"]["model_paths"] == [spec.model]
    assert sorted(receipt["chosen"]["settings"]) == [
        "cpu.decode_participants",
        "cpu.io_cpus",
        "place.costs",
    ]
    # The toy model's experts are so small that the CPU prefills any pass no slower.
    assert settings.at("prefill.cpu_max_tokens").origin == "default"
    assert any(
        note.startswith("prefill: the CPU route for every pass") for note in receipt["notes"]
    )
    assert receipt["chosen"]["threads"] == {"decode": 60, "batch": 60}
    assert receipt["chosen"]["server_args"] == ["--ctx-checkpoints", "1"]
    assert receipt["chosen"]["one_pass"] is True
    assert (line["launch"]["context"], line["launch"]["batch"], line["launch"]["ubatch"]) == (
        8192,
        8192,
        8192,
    )
    assert (receipt["kept"], receipt["dropped"]) == ([], [])
    assert any(note.startswith("threads: all placement") for note in receipt["notes"])


def test_cli_file_and_thread_values_win(tmp_path):
    inputs, spec, _, adapter, out = strategic(tmp_path)
    settings_file = Path(spec.settings_file)
    settings_file.write_text(settings_file.read_text() + 'place.costs = "drive_gbps=3" # mine\n')
    spec = replace(
        spec,
        overrides=spec.overrides + (("cpu.decode_participants", "2"),),
        threads=Threads(cpus=(0, 1, 2, 3)),
    )
    # The launcher refuses -ctxcp (it would read as -c); the long option is the user's.
    inputs = replace(inputs, extra=("--ctx-checkpoints", "2"))
    result, line, _ = resolved(inputs, spec, adapter, out)
    argv = list(result.argv)
    # The user's four strict CPUs size both pools; the strategy's 60 threads do not fit them.
    assert argv[argv.index("-t") : argv.index("-t") + 6] == ["-t", "4", "-tb", "4", "-C", "0xf"]
    assert argv[argv.index("--ssd-cpu-decode-participants") + 1] == "2"
    assert argv.count("--ctx-checkpoints") == 1 and argv[-2:] == ["--ctx-checkpoints", "2"]
    settings = native(out, spec)
    assert (settings.at("place.costs").value, settings.at("place.costs").receipt) == (
        (("drive_gbps", 3.0),),
        " mine",
    )
    assert settings.at("cpu.io_cpus").receipt.startswith(" source=strategy")
    receipt = line["strategy"]
    # The strategy proposed no CPU set here, so the user's replaced none.
    assert receipt["kept"] == [
        "cpu.decode_participants",
        "place.costs",
        "server --ctx-checkpoints",
    ]
    assert receipt["dropped"] == ["threads.decode", "threads.batch"]
    assert list(receipt["chosen"]["settings"]) == ["cpu.io_cpus"]
    assert receipt["chosen"]["threads"] == {} and receipt["chosen"]["server_args"] == []


def test_the_servers_own_checkpoint_variable_wins(tmp_path):
    inputs, spec, _, adapter, out = strategic(tmp_path)
    inputs = replace(inputs, environment=inputs.environment + (("LLAMA_ARG_CTX_CHECKPOINTS", "4"),))
    result, line, _ = resolved(inputs, spec, adapter, out)
    assert "--ctx-checkpoints" not in result.argv
    assert line["strategy"]["kept"] == ["server --ctx-checkpoints"]


def test_without_a_usable_gpu_the_budget_is_zero_and_no_gpu_is_checked(tmp_path):
    inputs, spec, host, adapter, out = strategic(tmp_path, gpus=())
    del host["gpu_available"], host["gpu_processes"]
    result, line, _ = resolved(inputs, spec, adapter, out)
    assert result.code == 0
    assert not {"gpu_available", "gpu_processes"} & {fact.name for fact in result.observations}
    budget = native(out, spec).at("vram.budget")
    assert (budget.value, budget.receipt) == (
        0,
        " source=strategy; input=no CUDA GPU is usable now, although the machine has 1",
    )
    assert "link_gbps" not in line["strategy"]["chosen"]["settings"]["place.costs"]["value"]


def test_user_launch_checks_still_apply_to_the_merged_result(tmp_path):
    inputs, spec, host, adapter, out = strategic(tmp_path)
    host["gpu_available"] = False
    with pytest.raises(ValueError, match="requires nvidia-smi"):
        resolved(inputs, spec, adapter, out)


def test_a_gpu_with_other_compute_processes_still_launches(tmp_path):
    inputs, spec, host, adapter, out = strategic(tmp_path)
    host["gpu_processes"] = "4242, other"
    result, _, _ = resolved(inputs, spec, adapter, out)
    assert result.code == 0
    assert "gpu_processes" not in {fact.name for fact in result.observations}


def test_without_characteristics_the_strategy_is_skipped(tmp_path):
    inputs, spec, host, adapter, out = strategic(tmp_path)
    del host["machine_characteristics"]
    result, line, stderr = resolved(inputs, spec, adapter, out)
    assert stderr == "inferssd launch: no launch strategy: no machine characteristics\n"
    assert line["strategy"] == {"skipped": "no machine characteristics"}
    # Everything else is the launch without a strategy.
    plain_out = tmp_path / "plain"
    plain_out.mkdir()
    plain, plain_line, _ = resolved(inputs, replace(spec, strategy=False), adapter, plain_out)
    assert [arg.replace(str(out), "RUN") for arg in result.argv] == [
        arg.replace(str(plain_out), "RUN") for arg in plain.argv
    ]
    assert (out / "settings.ssd").read_text() == (plain_out / "settings.ssd").read_text()
    assert "strategy" not in plain_line and not (plain_out / "strategy.json").exists()
    assert not STRATEGY_FACTS & {fact.name for fact in plain.observations}


def test_a_strategy_that_cannot_be_derived_is_skipped(tmp_path):
    inputs, spec, host, adapter, out = strategic(tmp_path, cpus=[99])
    result, line, stderr = resolved(inputs, spec, adapter, out)
    assert result.code == 0 and "none of the CPUs" in line["strategy"]["skipped"]
    assert "no launch strategy: none of the CPUs" in stderr


def render_golden(case, strategy):
    """A golden case resolved with canned host facts (see test_launch_platforms)."""
    files, written, outputs = set(case["files"]), {}, []

    def observe(fact):
        name, arguments = fact.name, fact.arguments
        if name == "resolved_path":
            return Reply(arguments[0])
        if name in ("is_file", "is_executable"):
            return Reply(arguments[0] in files)
        if name == "read_text":
            return Reply(case["settings"])
        if name == "mkdir":
            return Reply()
        if name == "gguf_alias":
            files.add(case["alias"])
            return Reply(case["alias"])
        if name == "write_exclusive":
            written[arguments[0]] = arguments[1]
            return Reply()
        if name == "engine_gpu_backends":
            return Reply(case["host"].get(name))
        return Reply(case["host"][name])

    pools = case["threads"]
    spec = LaunchSpec(
        case["model"],
        8080,
        case["settings_file"],
        tuple((key, value) for key, value in case["overrides"]),
        threads=Threads(pools.get("decode"), pools.get("batch"), tuple(pools.get("cpus", ()))),
        strategy=strategy,
    )
    result = prepare(
        LaunchInput(case["repo"], (), (), case["profile"]),
        resolver=partial(
            resolve_typed,
            spec=spec,
            platform=case["platform"],
            directory=case["directory"],
            memory=16 << 30,
        ),
        observe=observe,
        emit=outputs.append,
    )
    assert result.code == 0
    return result, written, outputs


@pytest.mark.parametrize("name", ["windows-qwen4exp", "windows-dsv"])
def test_windows_skips_the_strategy_with_a_note(name):
    expected = golden(name)
    result, written, outputs = render_golden(expected["case"], strategy=True)
    # Nothing new is asked, and the command is the golden one.
    assert list(result.argv) == expected["argv"] and list(result.scope) == expected["scope"]
    assert {path: text for path, text in written.items() if path.endswith(".ssd")} == (
        expected["settings"]
    )
    (receipt,) = [text for path, text in written.items() if path.endswith("strategy.json")]
    reason = "machine profiling is not supported on windows yet"
    assert json.loads(receipt) == {"skipped": reason}
    assert [o.text for o in outputs if o.channel == "stderr"] == [
        f"inferssd launch: no launch strategy: {reason}\n"
    ]


@pytest.mark.parametrize("name", ["linux-dsv", "linux-muse", "darwin-muse", "linux-qwen4exp"])
def test_goldens_without_a_strategy_ask_nothing_new(name):
    expected = golden(name)
    result, written, outputs = render_golden(expected["case"], strategy=False)
    assert list(result.argv) == expected["argv"] and written == expected["settings"]
    assert not STRATEGY_FACTS & {fact.name for fact in result.observations}
    assert [o.channel for o in outputs] == ["stdout"]
    assert "strategy" not in json.loads(outputs[0].text)


def recorded(host):
    return RecordedArtifacts(LaunchInput("", ()), host, run=None, replace=None, python="")


def test_recorded_engine_gpu_backends():
    fact = Fact("engine_gpu_backends", ("server",))
    # Recordings made before the fact existed: the engine's backends are unknown.
    assert recorded({})(fact).value is None
    assert recorded({"engine_gpu_backends": ["Metal"]})(fact).value == ["Metal"]
    for value in (["ROCm"], "CUDA", ["CUDA", "CUDA"]):
        with pytest.raises(ValueError, match="engine_gpu_backends"):
            recorded({"engine_gpu_backends": value})(fact)


def test_recorded_machine_characteristics(tmp_path):
    canonical = dumps(load(X86_FILE))
    assert recorded({"machine_characteristics": X86})(Fact("machine_characteristics")).value == (
        canonical
    )
    text = X86_FILE.read_text()
    assert recorded({"machine_characteristics": text})(Fact("machine_characteristics")).value == (
        canonical
    )
    assert recorded({})(Fact("machine_characteristics")).value is None
    broken = dict(X86, schema="inferssd-machine/0")
    with pytest.raises(ValueError, match="Recorded host fact machine_characteristics: schema"):
        recorded({"machine_characteristics": broken})(Fact("machine_characteristics"))


def test_recorded_model_runtime_and_drive_facts(tmp_path):
    model = str(write_gguf(tmp_path / "dsv.gguf", *deepseek_like()))
    header = model_facts_json(shard_facts([model]))
    # Without a recording the header is read from the model itself.
    assert recorded({})(Fact("model_facts", (model,))).value == header
    stored = dict(header, dense_bytes=7)
    assert recorded({"model_facts": stored})(Fact("model_facts", ("/absent.gguf",))).value == (
        stored
    )
    partial_header = {key: value for key, value in header.items() if key != "roles"}
    with pytest.raises(ValueError, match="model_facts.roles is required"):
        recorded({"model_facts": partial_header})(Fact("model_facts", (model,)))
    runtime_value = recorded({"runtime_facts": runtime_record()})(Fact("runtime_facts")).value
    assert runtime_value == dict(runtime_record(), memory_budget_bytes=None)
    with pytest.raises(ValueError, match="Missing recorded host fact: runtime_facts"):
        recorded({})(Fact("runtime_facts"))
    with pytest.raises(ValueError, match="must not repeat a CPU"):
        recorded({"runtime_facts": runtime_record(cpus=[1, 1])})(Fact("runtime_facts"))
    assert recorded({})(Fact("model_drive_key", (model,))).value is None
    assert recorded({"model_drive_key": "k"})(Fact("model_drive_key", (model,))).value == "k"
    with pytest.raises(ValueError, match="expected text or null"):
        recorded({"model_drive_key": 5})(Fact("model_drive_key", (model,)))


def test_live_strategy_facts(tmp_path):
    root = tmp_path / "root"
    commands = {
        SMI_GPUS: Command(0, "0, GPU-aaaa, 30000\n"),
        SMI_APPS: Command(0, ""),
    }
    commands.update(linux_tree(root, gpu=False))
    write(root, "/proc/meminfo", "MemTotal: 64000000 kB\nMemAvailable: 1000 kB\n")
    model = tmp_path / "models" / "m.gguf"
    model.parent.mkdir()
    model.write_bytes(b"x" * 100)
    host = RuntimeHost(
        root,
        affinity=(0, 1),
        commands=commands,
        devices={str(tmp_path / "models"): (259, 2)},
    )
    adapter = LiveArtifacts(
        LaunchInput("", (("PATH", "/bin"),)),
        host=None,
        run=None,
        replace=None,
        python="",
        machine="CHARACTERISTICS",
        strategy_host=host,
    )
    assert host.calls == []  # constructing the adapter observes nothing
    assert adapter(Fact("machine_characteristics")).value == "CHARACTERISTICS"
    assert adapter(Fact("runtime_facts")).value == {
        "cpus": [0, 1],
        "gpus": [{"uuid": "GPU-aaaa", "free_bytes": 30000 << 20, "processes": 0}],
        "available_bytes": 1000 * 1024,
        "memory_budget_bytes": None,
    }
    assert adapter(Fact("model_drive_key", (str(model),))).value == nvme_key()
    plain = LiveArtifacts(LaunchInput("", ()), host=None, run=None, replace=None, python="")
    assert plain(Fact("machine_characteristics")).value is None


def profile_launch(tmp_path, source=MachineSource(), probes=None, runner=None, **options):
    lines = []
    host, files = options.pop("machine", None) or linux_machine(tmp_path)
    text = characteristics(
        source,
        options.pop("platform", "linux"),
        files[0],
        dict(PATHS) if probes is None else probes,
        host=host,
        runner=runner or FakeRunner(replies()),
        environ={},
        home=tmp_path / "home",
        clock=Clock(),
        progress=lines.append,
    )
    return text, lines, default_path("linux", {}, tmp_path / "home")


def test_first_launch_profiles_and_later_launches_reuse_the_file(tmp_path):
    machine = linux_machine(tmp_path)
    runner = FakeRunner(replies())
    text, lines, path = profile_launch(tmp_path, runner=runner, machine=machine)
    assert text == path.read_text() and len(runner.calls) == 3
    assert lines[:2] == [
        "measuring this machine before its first launch",
        f"later launches reuse {path}",
    ]
    assert "storage:" + nvme_key() in json.loads(text)["sections"]
    unused = FakeRunner({})
    again, quiet, _ = profile_launch(tmp_path, runner=unused, machine=machine)
    assert again == text and unused.calls == [] and quiet == []
    remeasured = FakeRunner(replies())
    _, lines, _ = profile_launch(
        tmp_path, MachineSource(reprofile=True), runner=remeasured, machine=machine
    )
    assert len(remeasured.calls) == 3 and lines[0] == "measuring this machine again"


def test_missing_probes_warn_and_the_launch_continues(tmp_path):
    text, lines, path = profile_launch(tmp_path, probes={}, runner=FakeRunner({}))
    assert text == path.read_text()
    assert sum("was not found next to the server" in line for line in lines) == 3
    statuses = {name: row["status"] for name, row in json.loads(text)["sections"].items()}
    assert statuses["compute"] == "failed" and statuses["storage:" + nvme_key()] == "failed"
    assert any(line.startswith("warning: compute failed: ") for line in lines)


def test_only_the_probes_a_run_needs_are_reported_missing(tmp_path):
    # Without an NVIDIA GPU the memory probe has nothing to measure.
    machine = linux_machine(tmp_path, gpu=False)
    _, lines, _ = profile_launch(tmp_path, probes={}, runner=FakeRunner({}), machine=machine)
    missing = [line.split()[1] for line in lines if "was not found next to the server" in line]
    assert missing == ["inferssd-ssdprobe", "inferssd-cpuprobe"]
    # Retrying only what failed for want of a probe measures nothing and says so once.
    runner = FakeRunner({})
    _, again, _ = profile_launch(tmp_path, probes={}, runner=runner, machine=machine)
    assert runner.calls == [] and again == [
        "warning: inferssd-cpuprobe, inferssd-ssdprobe still not found; launching with the "
        "saved profile, which lacks what they measure"
    ]
    # Once the probes are there, the failed sections are measured again.
    found = FakeRunner(replies())
    _, lines, _ = profile_launch(tmp_path, runner=found, machine=machine)
    assert len(found.calls) == 2 and lines[0] == "measuring what changed on this machine"


def test_a_busy_profile_lock_uses_the_saved_profile(tmp_path, monkeypatch):
    machine = linux_machine(tmp_path)
    text, _, _ = profile_launch(tmp_path, machine=machine)

    def busy(path, wait_s=0.0):
        raise LockBusy("busy")

    monkeypatch.setattr(launch_machine, "exclusive", busy)
    again, lines, _ = profile_launch(tmp_path, machine=machine)
    assert again == text and "another inferssd profile is measuring" in lines[0]
    nothing, _, _ = profile_launch(tmp_path / "fresh", machine=linux_machine(tmp_path / "fresh"))
    assert nothing is None


def test_a_profiling_error_warns_and_keeps_the_saved_profile(tmp_path):
    host, _ = linux_machine(tmp_path)
    text, lines, _ = profile_launch(tmp_path, machine=(host, [str(tmp_path / "absent.gguf")]))
    assert text is None and "could not be profiled" in lines[0]


def test_an_explicit_machine_file_is_used_as_it_is(tmp_path):
    source = MachineSource(path=X86_FILE)
    text, lines, path = profile_launch(tmp_path, source, machine=(None, ["/unused.gguf"]))
    assert text == dumps(load(X86_FILE)) and lines == [] and not path.exists()
    broken = tmp_path / "broken.json"
    broken.write_text("{}")
    with pytest.raises(ContractError):
        profile_launch(tmp_path, MachineSource(path=broken), machine=(None, ["/unused.gguf"]))


def test_unprofiled_platforms_launch_without_characteristics(tmp_path):
    host, files = linux_machine(tmp_path)
    for platform in ("windows", "darwin"):
        text, lines, _ = profile_launch(tmp_path, machine=(host, files), platform=platform)
        assert text is None and "not supported" in lines[0]


def test_probe_lookup_order():
    present = {
        "/opt/server/bin/inferssd-cpuprobe",
        "/res/ssd",
        "/res/cpu",
        "/repo/build-linux-cpu/bin/inferssd-ssdprobe",
        "/repo/build/bin/inferssd-memprobe",
        "/repo/build-linux-cpu/bin/inferssd-memprobe",
    }
    resources = EngineResources("/x", (), (("ssd", "/res/ssd"), ("cpu", "/res/cpu")))
    found = probe_paths(
        "linux", "/opt/server/bin/inferssd-server", resources, "/repo", present.__contains__
    )
    # Next to the server first, then the descriptor, then the checkout's builds in their order.
    assert found == {
        "cpu": "/opt/server/bin/inferssd-cpuprobe",
        "ssd": "/res/ssd",
        "memory": "/repo/build-linux-cpu/bin/inferssd-memprobe",
    }
    assert probe_paths("linux", None, None, "", present.__contains__) == {}
    windows = {"C:\\engine\\bin\\inferssd-ssdprobe.exe"}
    assert probe_paths(
        "windows", "C:\\engine\\bin\\inferssd-server.exe", None, "", windows.__contains__
    ) == {"ssd": "C:\\engine\\bin\\inferssd-ssdprobe.exe"}


def test_runtime_measures_once_before_the_run_directory(tmp_path, monkeypatch):
    inputs, spec, _, adapter, out = strategic(tmp_path)
    effects, calls, adapters = Effects(), [], []
    owner = Owner(effects)

    def measured(source, platform, model, probes):
        calls.append((source, platform, model, probes, list(effects.events)))
        return dumps(load(X86_FILE))

    def live(*args, **options):
        adapters.append(options)
        return adapter

    monkeypatch.setattr(runtime, "machine_characteristics", measured)
    monkeypatch.setattr(runtime, "LiveArtifacts", live)
    monkeypatch.setattr(runtime, "NativeEffects", lambda: effects)
    monkeypatch.setattr(runtime, "OwnedProcess", lambda *a, **k: owner)
    monkeypatch.setattr(effects, "reservation", object(), raising=False)
    source = MachineSource(reprofile=True)
    value = runtime.Runtime(
        inputs,
        policy(tmp_path, "linux"),
        host=object(),
        emit=lambda _: None,
        configuration=spec,
        machine=source,
    )
    engine = value.engine(str(out), 8080, 0, "static", "", [])
    assert engine.ok
    # Measured once, before the port check, the run directory or the server log.
    ((given, platform, model, probes, events),) = calls
    assert (given, platform, model, probes, events) == (source, "linux", spec.model, {}, [])
    assert adapters and all(options["machine"] == dumps(load(X86_FILE)) for options in adapters)
    engine.stop()
    plain = runtime.Runtime(
        inputs, policy(tmp_path, "linux"), host=object(), emit=lambda _: None, machine=source
    )
    assert plain.characteristics(replace(spec, strategy=False)) is None and len(calls) == 1


def launch_arguments(tmp_path, *extra):
    inputs, spec, host, _, out, _ = setup(tmp_path)
    contract = tmp_path / "policy.json"
    contract.write_text(json.dumps(asdict(policy(tmp_path, "linux"))))
    host.update(runtime_facts=runtime_record(), model_drive_key=X86_DRIVE)
    facts = tmp_path / "facts.json"
    facts.write_text(json.dumps(host))
    words = ["--platform", "linux", "--repo", inputs.repo, "--model", spec.model]
    words += ["--profile", "muse"]
    words += ["--binary", spec.binary, "--settings", spec.settings_file]
    words += ["--runtime-policy", str(contract), "--output-dir", str(out)]
    return words + list(extra), facts, out


def test_dry_run_takes_characteristics_from_machine(tmp_path, monkeypatch):
    words, facts, out = launch_arguments(tmp_path)
    receipt = tmp_path / "resolution.json"
    monkeypatch.setattr("os.environ", {"PATH": "/bin"})
    command = words + ["--dry-run", "--host-facts", str(facts), "--machine", str(X86_FILE)]
    assert main(command + ["--result", str(receipt)]) == 0
    result = json.loads(receipt.read_text())
    assert result["argv"][-2:] == ["--ctx-checkpoints", "1"]
    assert json.loads((out / "strategy.json").read_text())["inputs"]["model_drive"] == X86_DRIVE


def test_dry_run_without_characteristics_keeps_todays_resolution(tmp_path, monkeypatch):
    words, facts, out = launch_arguments(tmp_path)
    receipt = tmp_path / "resolution.json"
    monkeypatch.setattr("os.environ", {"PATH": "/bin"})
    assert main(words + ["--dry-run", "--host-facts", str(facts), "--result", str(receipt)]) == 0
    result = json.loads(receipt.read_text())
    assert not STRATEGY_FACTS & {fact["name"] for fact in result["observations"]}
    assert "--ctx-checkpoints" not in result["argv"] and not (out / "strategy.json").exists()


@pytest.mark.parametrize(
    "extra, message",
    [
        (["--dry-run", "--reprofile"], "a dry-run never observes it"),
        (["--execute", "--reprofile", "--machine", "m.json"], "--machine is used as it is"),
    ],
)
def test_option_conflicts_are_refused(tmp_path, capsys, extra, message):
    words, _, out = launch_arguments(tmp_path)
    with pytest.raises(SystemExit) as error:
        main(words + extra)
    assert error.value.code == 2 and message in capsys.readouterr().err
    assert not (out / "settings.ssd").exists()


def test_execute_passes_the_machine_source_to_the_runtime(tmp_path, monkeypatch):
    words, _, _ = launch_arguments(tmp_path)
    captured = {}

    class Engine:
        ok = True

        def alive(self):
            return False

        def stop(self):
            pass

    class Runtime:
        def __init__(self, *args, **options):
            captured.update(options)

        def engine(self, *args):
            return Engine()

    monkeypatch.setattr(engine_command, "Runtime", Runtime)
    assert main(words + ["--execute", "--port", "8080", "--reprofile"]) == 0
    assert captured["machine"] == MachineSource(None, True)
    assert captured["configuration"].strategy is True


def test_help_names_the_machine_options(capsys):
    with pytest.raises(SystemExit):
        main(["--help"])
    text = capsys.readouterr().out
    assert "--reprofile" in text and "--machine PATH" in text
