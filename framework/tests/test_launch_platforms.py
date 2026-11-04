# SPDX-License-Identifier: Apache-2.0
"""Typed launch commands for every platform and profile, frozen as goldens.

Canned host facts stand in for each target host, so every platform's command
renders here with that platform's own paths; the resolver touches no file.
"""

import argparse
import json
import sys
from functools import partial
from pathlib import Path

import pytest

from inferssd.configuration.settings import host_platform
from inferssd.launch import typed_command
from inferssd.launch.prepare import prepare
from inferssd.launch.typed import LaunchSpec, resolve_typed
from inferssd.launch.types import LaunchInput, Reply, Threads

GOLDENS = json.loads((Path(__file__).parent / "fixtures/launch-platform-goldens.json").read_text())


def golden(name):
    return next(item for item in GOLDENS if item["case"]["name"] == name)


def render(case):
    files, written = set(case["files"]), {}

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
            assert list(arguments[1:]) == case["shards"]
            files.add(case["alias"])
            return Reply(case["alias"])
        if name == "write_exclusive":
            written[arguments[0]] = arguments[1]
            return Reply()
        if name == "engine_gpu_backends":
            # As a recorded launch: an engine whose GPU backends a case leaves out is unknown.
            return Reply(case["host"].get(name))
        return Reply(case["host"][name])

    pools = case["threads"]
    spec = LaunchSpec(
        case["model"],
        8080,
        case["settings_file"],
        tuple((key, value) for key, value in case["overrides"]),
        threads=Threads(pools.get("decode"), pools.get("batch"), tuple(pools.get("cpus", ()))),
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
        emit=lambda _: None,
    )
    assert result.code == 0 and not result.assigned_environment
    return {"argv": list(result.argv), "scope": list(result.scope), "settings": written}


@pytest.mark.parametrize("expected", GOLDENS, ids=lambda item: item["case"]["name"])
def test_typed_launch_command_matches_golden(expected):
    assert render(expected["case"]) == {key: expected[key] for key in ("argv", "scope", "settings")}


def test_windows_qwen4exp_command_uses_the_model_template_and_settings_envelope():
    expected = golden("windows-qwen4exp")
    argv, case = expected["argv"], expected["case"]
    assert argv[:3] == [
        "C:\\inferssd\\build-x64-windows-llvm-release\\bin\\inferssd-server.exe",
        "-m",
        case["model"],
    ]
    # The GGUF's own chat template; Flash Attention keeps the server's auto choice.
    assert "--chat-template-file" not in argv and "-fa" not in argv
    # No scope: host.budget_mib reaches the server through its settings file, as given.
    assert expected["scope"] == [] and not any(word.startswith("--ssd-host") for word in argv)
    settings = expected["settings"]["C:\\inferssd\\runs\\qwen4exp\\settings.ssd"]
    assert "host.budget_mib = 14336" in settings and json.dumps([case["alias"]]) in settings


def test_windows_passes_cli_settings_and_pins_threads():
    argv = golden("windows-dsv")["argv"]
    assert argv[argv.index("--ssd-host-budget-mib") + 1] == "14336"
    assert argv[argv.index("-C") + 1] == argv[argv.index("-Cb") + 1] == "0xf"
    template = argv[argv.index("--chat-template-file") + 1]
    assert template == "C:\\inferssd\\engine\\models\\templates\\deepseek-ai-DeepSeek-V4.1.jinja"


def test_linux_qwen4exp_keeps_the_linux_launch():
    expected = golden("linux-qwen4exp")
    assert "--chat-template-file" not in expected["argv"]
    assert expected["argv"][-2:] == ["-fa", "on"] and "MemoryMax=17179869184" in expected["scope"]


@pytest.mark.parametrize(
    "host,expected", [("linux", "linux"), ("darwin", "darwin"), ("win32", "windows")]
)
def test_platform_options_default_to_this_host(monkeypatch, host, expected):
    monkeypatch.setattr(sys, "platform", host)
    assert host_platform() == expected
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command")
    typed_command.add_command(commands)
    words = ["launch", "--dry-run", "--model", "model.gguf", "--runtime-policy", "policy.json"]
    assert parser.parse_args(words).platform == expected


def test_windows_owned_execution_is_refused_before_any_effect(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    command = ["--execute", "--platform", "windows", "--model", "model.gguf"]
    with pytest.raises(SystemExit) as error:
        typed_command.main(command + ["--runtime-policy", "policy.json"])
    assert error.value.code == 2 and "--dry-run" in capsys.readouterr().err
    assert not list(tmp_path.iterdir())
