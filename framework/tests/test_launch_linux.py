"""Replay frozen Linux reference observations without running a shell or host probe."""

import json
import shlex
from pathlib import Path

import pytest

from inferssd.launch.linux import resolve_linux
from inferssd.launch.types import LaunchInput, Reply

GOLDENS = json.loads((Path(__file__).parent / "fixtures/launch-linux-goldens.json").read_text())


def replay(record, run):
    case, host, env = record["case"], record["host"], record["env"]
    metadata = iter(run["metadata"])
    consumed = []
    facts = {}
    inputs = LaunchInput("@repo", tuple(env.items()), tuple(case.get("extra", [])))
    while True:
        result = resolve_linux(inputs, facts)
        fact = result.needed
        if fact is None:
            return result, consumed
        name, args = fact.name, fact.arguments
        if name == "is_file":
            value = not (args[0].endswith("libggml-base.so") or args[0] == "@case/missing.gguf")
        elif name == "read_text":
            value = (
                record["conf"]
                if args[0].endswith("model.conf")
                else case["settings"]
                if args[0].endswith("settings.env")
                else case["dense"]
            )
        elif name == "model_matches":
            value = (
                ("@case/models/Synthetic-Q4.gguf", "@case/models/Synthetic-other.gguf")
                if case.get("ambiguous")
                else ("@case/models/Synthetic-Q4.gguf",)
            )
        elif name == "is_executable":
            value = args[0] not in ("@case/missing", "@case/no-server")
        elif name == "absolute_directory":
            value = args[0]
        elif name == "filesystem":
            value = host["filesystem"]
        elif name in ("index", "source_facts", "dense"):
            command = "facts" if name == "source_facts" else name
            consumed.append(command)
            if host.get("fail_metadata") == command:
                facts[fact] = Reply(stderr="fixture metadata failure: " + command + "\n", code=9)
                continue
            observed = next(metadata)
            assert observed["command"] == command
            values = {}
            if command != "index" and observed["code"] == 0:
                for word in shlex.split(observed["stdout"]):
                    key, value = word.split("=", 1)
                    values[key] = value
            facts[fact] = Reply(values, observed["stdout"], observed["stderr"], observed["code"])
            continue
        elif name in ("gpu_available", "ss_available", "user_manager", "memory_controller"):
            value = True
        elif name == "gpu_memory":
            value = host["gpu_mib"]
        elif name == "gpu_processes":
            value = host["gpu_apps"]
        elif name == "default_threads":
            value = str(host["cpu_count"])
        elif name == "port_busy":
            value = host["port_busy"]
        elif name == "has_uring":
            value = host["uring"]
        elif name == "cgroup_v2":
            value = record["cgroup_v2"]
        elif name == "memory_kib":
            value = host["mem_total_kib"], host["mem_available_kib"]
        elif name == "euid":
            value = record["uid"]
        else:
            raise AssertionError(fact)
        facts[fact] = Reply(value)


@pytest.mark.parametrize("record", GOLDENS, ids=lambda r: r["case"]["name"])
def test_linux_frozen(record):
    for run in record["runs"]:
        actual, metadata = replay(record, run)
        if record["case"]["name"] == "early-resolve":
            assert actual.argv == ("@case/models/Synthetic-Q4.gguf",)
            assert not metadata
            assert not actual.outputs
            continue
        stdout = "".join(x.text for x in actual.outputs if x.channel == "stdout")
        stderr = "".join(x.text for x in actual.outputs if x.channel == "stderr")
        assert actual.code == run["code"]
        assert stderr == run["stderr"]
        assert stdout == run["stdout"]
        if actual.code:
            continue
        assert actual.argv == tuple(x[4:] for x in stdout.splitlines() if x.startswith("ARG "))
        assert actual.scope == tuple(x[6:] for x in stdout.splitlines() if x.startswith("SCOPE "))
        assert actual.assigned_environment == tuple(
            tuple(x[4:].split("=", 1)) for x in stdout.splitlines() if x.startswith("ENV ")
        )


def external_order(calls):
    ordered = []
    for program, arguments in calls:
        if program == "stat":
            ordered.append("filesystem")
        elif program == "python3":
            ordered.append(next(x for x in arguments if x in ("index", "facts", "dense")))
        elif program == "nvidia-smi":
            ordered.append("gpu_memory" if "memory.total" in arguments[0] else "gpu_processes")
        elif program == "ss":
            ordered.append("port_busy")
        elif program == "grep" and arguments[0] == "-qs":
            ordered.append("has_uring")
        elif program == "awk" and "/proc/meminfo" in arguments:
            ordered.append("memory_kib")
        elif program == "systemctl":
            ordered.append("user_manager")
        elif program == "grep" and arguments[0] == "-qw":
            ordered.append("memory_controller")
    return ordered


@pytest.mark.parametrize("record", GOLDENS, ids=lambda r: r["case"]["name"])
def test_host_and_artifact_requests_keep_reference_order(record):
    names = {
        "filesystem",
        "index",
        "source_facts",
        "dense",
        "gpu_memory",
        "gpu_processes",
        "port_busy",
        "has_uring",
        "memory_kib",
        "user_manager",
        "memory_controller",
    }
    for run in record["runs"]:
        result, _ = replay(record, run)
        actual = [
            "facts" if f.name == "source_facts" else f.name
            for f in result.observations
            if f.name in names
        ]
        assert actual == external_order(run["calls"])
