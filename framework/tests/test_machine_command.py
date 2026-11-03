# SPDX-License-Identifier: Apache-2.0
"""The profile command: measure, save, summarize and show, with a fake host and probes."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pytest
from test_machine_identity import FakeHost, linux_tree, nvme_key
from test_machine_probes import FakeRunner, replies
from test_machine_profile import GIB, Clock, shards

from inferssd.app import catalog
from inferssd.cli import main
from inferssd.launch.resources import installed_resources
from inferssd.machine import command
from inferssd.machine.store import exclusive
from inferssd.strategy.inputs import RuntimeFacts


def parse(argv):
    parser = argparse.ArgumentParser(prog="inferssd")
    command.add_command(parser.add_subparsers(dest="command"))
    return parser.parse_args(["profile", *argv])


def machine(tmp_path):
    root = tmp_path / "root"
    commands = linux_tree(root)
    models = tmp_path / "models"
    files = shards(models, [2 * GIB, 2 * GIB])
    host = FakeHost(root, commands=commands, devices={os.path.realpath(models): (259, 2)})
    build = tmp_path / "build"
    (build / "bin").mkdir(parents=True)
    for name in ("inferssd-ssdprobe", "inferssd-cpuprobe", "inferssd-memprobe"):
        (build / "bin" / name).write_text("")
    return host, files, build


RUNTIME = RuntimeFacts((0, 1, 2, 3), (), 24 * GIB)


def execute(argv, host, runner, tmp_path, environ=None):
    out, progress = [], []
    code = command.run(
        parse(argv),
        host=host,
        runner=runner,
        environ={"XDG_STATE_HOME": str(tmp_path / "state"), **(environ or {})},
        home=tmp_path / "home",
        clock=Clock(),
        progress=progress.append,
        emit=out.append,
        runtime=lambda: RUNTIME,
    )
    return code, out, progress


def resource_descriptor(tmp_path, build):
    descriptor = tmp_path / "engine-resources.json"
    descriptor.write_text(
        json.dumps(
            {
                "schema": "inferssd-engine 1",
                "binary": str(build / "bin/inferssd-server"),
                "capabilities": {"settings": "inferssd-settings 1", "telemetry": 1},
                "templates": {"muse": str(build / "chat.jinja")},
                "probes": {
                    name: str(build / "bin" / executable)
                    for name, executable in (
                        ("cpu", "inferssd-cpuprobe"),
                        ("memory", "inferssd-memprobe"),
                        ("ssd", "inferssd-ssdprobe"),
                    )
                },
            }
        )
    )
    return descriptor


def test_profile_uses_the_installed_wrappers_resources(tmp_path):
    host, files, build = machine(tmp_path)
    descriptor = resource_descriptor(tmp_path, build)
    runner = FakeRunner(replies())
    code, _, _ = execute(
        ["--model", files[0]],
        host,
        runner,
        tmp_path,
        environ={"INFERSSD_RESOURCES": str(descriptor)},
    )
    assert code == 0
    assert {argv[0] for argv, _ in runner.calls} == {
        str(build / "bin" / name)
        for name in ("inferssd-cpuprobe", "inferssd-memprobe", "inferssd-ssdprobe")
    }
    saved = json.loads((tmp_path / "state/inferssd/machine.json").read_text())
    assert all(section["status"] == "ok" for section in saved["sections"].values())


@pytest.mark.parametrize("option", ["--resources", "--build"])
def test_explicit_probe_location_overrides_installed_resources(tmp_path, option):
    host, _, build = machine(tmp_path)
    location = resource_descriptor(tmp_path, build) if option == "--resources" else build
    runner = FakeRunner(replies())
    code, _, _ = execute(
        [option, str(location)],
        host,
        runner,
        tmp_path,
        environ={"INFERSSD_RESOURCES": str(tmp_path / "missing.json")},
    )
    assert code == 0 and runner.calls
    assert all(argv[0].startswith(str(build / "bin")) for argv, _ in runner.calls)


@pytest.mark.parametrize("environ", [{}, {"INFERSSD_RESOURCES": ""}])
def test_profile_uses_the_current_release_without_the_variable(tmp_path, environ):
    host, _, build = machine(tmp_path)
    current = tmp_path / "home/.local/share/inferssd/current"
    current.mkdir(parents=True)
    resource_descriptor(tmp_path, build).replace(current / "engine-resources.json")
    runner = FakeRunner(replies())
    code, _, _ = execute([], host, runner, tmp_path, environ=environ)
    assert code == 0 and runner.calls
    assert all(argv[0].startswith(str(build / "bin")) for argv, _ in runner.calls)


@pytest.mark.parametrize("environ", [{}, {"INFERSSD_RESOURCES": ""}])
def test_profile_does_not_look_for_probes_on_path(tmp_path, monkeypatch, environ):
    host, _, build = machine(tmp_path)
    for probe in (build / "bin").iterdir():
        probe.chmod(0o755)
    monkeypatch.setenv("PATH", str(build / "bin"))
    runner = FakeRunner({})
    with pytest.raises(ValueError, match="engine not installed"):
        execute([], host, runner, tmp_path, environ=environ)
    assert not runner.calls
    assert not (tmp_path / "state/inferssd/machine.json").exists()


def test_installed_resources_prefer_the_variable_over_the_current_release(tmp_path):
    current = tmp_path / ".local/share/inferssd/current/engine-resources.json"
    assert installed_resources({}, tmp_path) == current
    assert installed_resources({"INFERSSD_RESOURCES": ""}, tmp_path) == current
    assert installed_resources({"INFERSSD_RESOURCES": "/a.json"}, tmp_path) == Path("/a.json")
    assert installed_resources({"INFERSSD_HOME": "/data"}, tmp_path) == Path(
        "/data/current/engine-resources.json"
    )


def test_show_ignores_installed_resources_and_does_not_measure(tmp_path):
    host, _, build = machine(tmp_path)
    code, expected, _ = execute(["--build", str(build)], host, FakeRunner(replies()), tmp_path)
    assert code == 0
    saved = tmp_path / "state/inferssd/machine.json"
    before = saved.read_bytes()
    runner = FakeRunner({})
    code, shown, _ = execute(
        ["--show"],
        host,
        runner,
        tmp_path,
        environ={"INFERSSD_RESOURCES": str(tmp_path / "missing.json")},
    )
    assert code == 0 and shown == expected and not runner.calls
    assert saved.read_bytes() == before


def test_missing_installed_descriptor_is_not_silently_ignored(tmp_path):
    host, _, _ = machine(tmp_path)
    runner = FakeRunner({})
    with pytest.raises(ValueError, match="engine not installed"):
        execute(
            [],
            host,
            runner,
            tmp_path,
            environ={"INFERSSD_RESOURCES": str(tmp_path / "missing.json")},
        )
    assert not runner.calls
    assert not (tmp_path / "state/inferssd/machine.json").exists()


def test_profile_is_listed(capsys):
    with pytest.raises(SystemExit) as error:
        main(["--help"])
    text = capsys.readouterr().out
    assert error.value.code == 0 and "measure this machine" in text
    with pytest.raises(SystemExit) as error:
        main(["profile", "--help"])
    text = capsys.readouterr().out
    assert error.value.code == 0
    for option in ("--model PATH", "--resources", "--build", "--reprofile", "--output", "--show"):
        assert option in text


def test_profile_saves_to_the_state_directory_and_summarizes(tmp_path):
    host, files, build = machine(tmp_path)
    runner = FakeRunner(replies())
    code, out, progress = execute(
        ["--model", files[0], "--build", str(build)], host, runner, tmp_path
    )
    saved = tmp_path / "state/inferssd/machine.json"
    assert code == 0 and saved.exists()
    assert runner.calls[0][0][0] == str(build / "bin/inferssd-cpuprobe")
    assert out[0].startswith(f"machine profile {saved} (written ")
    assert out[1] == (
        "identity: linux x86_64, Example CPU 9000, 4 logical CPUs, 31.2 GiB RAM, "
        "GPUs: NVIDIA GeForce RTX 5090 (CUDA)"
    )
    names = [line.split(":", 1)[0] for line in out[2:]]
    assert names == ["cpu", "memory", "compute", "gpus", "storage"] + ["estimate"] * len(
        catalog.MODELS
    )
    # This machine measured Q4_K alone, so no supported model can be priced: each names the
    # role formats it lacks, as cpu_line does.
    assert out[7:] == [
        f"estimate: {model.repo}:{model.quantization} none: no CPU compute measurement of "
        + ", ".join(sorted({role.format.upper() for role in model.facts.roles} - {"Q4_K"}))
        for model in catalog.MODELS
    ]
    assert out[2].endswith("4 logical CPUs (4 uniform); ISA AVX2 FMA F16C")
    assert "DRAM read 11.18 GiB/s with 1 thread" in out[3]
    assert out[4].endswith("1 formats at 4 threads; skipped MXFP4 (no CPU kernel)")
    assert "NVIDIA GeForce RTX 5090 31.8 GiB, host to device 22.35 GiB/s" in out[5]
    assert out[6].startswith(f"storage:{nvme_key()}: ok (")
    assert out[6].endswith(
        "Example NVMe 2TB (nvme, io_uring), 2 files; best 4.75 GiB/s at "
        "4 MiB QD 4; knee QD 4 at 1 MiB, QD 4 at 4 MiB"
    )
    assert progress[0] == "reading machine identity"

    # A second run finds nothing to measure and leaves the file alone.
    before = saved.read_bytes()
    runner = FakeRunner({})
    code, again, _ = execute(["--model", files[0], "--build", str(build)], host, runner, tmp_path)
    assert code == 0 and runner.calls == [] and saved.read_bytes() == before and again == out

    code, shown, _ = execute(["--show", "--model", files[0]], host, FakeRunner({}), tmp_path)
    assert code == 0 and shown == out
    code, shown, _ = execute(["--show"], host, FakeRunner({}), tmp_path)
    assert code == 0 and shown == out[: -len(catalog.MODELS)]


def test_failed_sections_set_the_exit_status(tmp_path):
    host, files, _ = machine(tmp_path)
    output = tmp_path / "custom.json"
    code, out, _ = execute(
        ["--output", str(output), "--build", str(tmp_path / "none")], host, FakeRunner({}), tmp_path
    )
    assert code == 1 and output.exists()
    assert out[-2:] == ["2 section(s) partial: cpu, memory", "2 section(s) failed: compute, gpus"]
    assert any("inferssd-cpuprobe not found" in line for line in out)


def test_unreadable_saved_profile_is_replaced(tmp_path):
    host, _, build = machine(tmp_path)
    output = tmp_path / "machine.json"
    output.write_text(json.dumps({"schema": "inferssd-machine/0"}))
    code, _, progress = execute(
        ["--output", str(output), "--build", str(build)], host, FakeRunner(replies()), tmp_path
    )
    assert code == 0 and any(
        line.startswith("ignoring the unreadable saved profile") for line in progress
    )
    assert json.loads(output.read_text())["schema"] == "inferssd-machine/1"


def test_show_measures_nothing(tmp_path):
    host, files, _ = machine(tmp_path)
    with pytest.raises(ValueError, match="no machine profile at .*; run inferssd profile first"):
        execute(["--show"], host, FakeRunner({}), tmp_path)
    with pytest.raises(ValueError, match="--show measures nothing"):
        execute(["--show", "--reprofile"], host, FakeRunner({}), tmp_path)


def test_json_prints_one_object_with_the_machine_and_the_estimates(tmp_path):
    host, files, build = machine(tmp_path)
    code, out, progress = execute(
        ["--model", files[0], "--build", str(build), "--json"],
        host,
        FakeRunner(replies()),
        tmp_path,
    )
    (line,) = out
    value = json.loads(line)
    assert code == 0 and progress[0] == "reading machine identity"
    assert value["profile"] == str(tmp_path / "state/inferssd/machine.json")
    assert value["machine"] == {
        "platform": "linux",
        "cpu": "Example CPU 9000",
        "logical_cpus": 4,
        "memory_bytes": 32768000 * 1024,
        "memory_read_Bps": 3.6e10,
        "gpus": ["NVIDIA GeForce RTX 5090"],
        "models_drive": {"model": "Example NVMe 2TB", "read_Bps": 5.1e9},
    }
    assert [(row["model"], row["quantization"]) for row in value["estimates"]] == [
        (model.repo, model.quantization) for model in catalog.MODELS
    ]
    assert all(row["decode_tokens_per_second"] is None for row in value["estimates"])
    # Without --model the estimates have no drive, and the machine no models drive.
    code, out, _ = execute(["--show", "--json"], host, FakeRunner({}), tmp_path)
    value = json.loads(out[0])
    assert value["machine"]["models_drive"] is None and {
        row["detail"] for row in value["estimates"]
    } == {"pass --model with the directory that holds the models"}


def test_cli_entry_point(tmp_path, monkeypatch, capsys):
    host, files, build = machine(tmp_path)
    monkeypatch.setattr(command, "NativeHost", lambda: host)
    monkeypatch.setattr(command, "run_process", FakeRunner(replies()))
    output = tmp_path / "machine.json"
    assert (
        main(["profile", "--model", files[0], "--build", str(build), "--output", str(output)]) == 0
    )
    assert main(["profile", "--show", "--output", str(output)]) == 0
    assert capsys.readouterr().out.count("\nstorage:linux:nvme:Example NVMe 2TB:") == 2
    with pytest.raises(SystemExit) as error:
        main(["profile", "--show", "--output", str(tmp_path / "absent.json")])
    assert error.value.code == 2 and "no machine profile" in capsys.readouterr().err
    with pytest.raises(SystemExit) as error:
        main(["profile", "--build", str(build), "--resources", str(output)])
    assert error.value.code == 2 and "not allowed with argument" in capsys.readouterr().err


def test_cli_entry_uses_installer_environment(tmp_path, monkeypatch):
    host, files, build = machine(tmp_path)
    descriptor = resource_descriptor(tmp_path, build)
    runner = FakeRunner(replies())
    monkeypatch.setattr(command, "NativeHost", lambda: host)
    monkeypatch.setattr(command, "run_process", runner)
    monkeypatch.setenv("INFERSSD_RESOURCES", str(descriptor))
    output = tmp_path / "machine.json"
    assert main(["profile", "--model", files[0], "--output", str(output)]) == 0
    assert all(
        section["status"] == "ok" for section in json.loads(output.read_text())["sections"].values()
    )
    assert runner.calls[0][0][0] == str(build / "bin/inferssd-cpuprobe")


@pytest.mark.skipif(os.name == "nt", reason="flock")
def test_a_running_profile_holds_the_lock(tmp_path, monkeypatch, capsys):
    import fcntl

    host, _, build = machine(tmp_path)
    monkeypatch.setattr(command, "NativeHost", lambda: host)
    output = tmp_path / "machine.json"
    with exclusive(output):
        pass
    holder = os.open(tmp_path / "machine.json.lock", os.O_RDWR)
    try:
        fcntl.flock(holder, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(SystemExit) as error:
            main(["profile", "--build", str(build), "--output", str(output)])
    finally:
        os.close(holder)
    assert error.value.code == 2 and "another inferssd profile holds" in capsys.readouterr().err
