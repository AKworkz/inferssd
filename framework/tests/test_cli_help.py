"""The inferssd command offers serve, cli and profile, and lists all three."""

import json
from importlib import metadata

import pytest

from inferssd.cli import main


def test_help_lists_every_command(capsys):
    with pytest.raises(SystemExit) as error:
        main(["--help"])
    text = " ".join(capsys.readouterr().out.split())
    assert error.value.code == 0 and "{cli,serve,profile}" in text
    for line in ("chat in the terminal", "start the OpenAI-compatible API", "measure this machine"):
        assert line in text


@pytest.mark.parametrize(
    "command",
    ["launch", "launch-recorded", "configuration", "source", "bootstrap-records", "report"]
    + ["probe", "validate", "plan", "settings", "fit"],
)
def test_removed_commands_are_refused(capsys, command):
    with pytest.raises(SystemExit) as error:
        main([command, "--help"])
    assert error.value.code == 2 and "invalid choice" in capsys.readouterr().err


def test_version_is_the_installed_packages(capsys, monkeypatch, tmp_path):
    # Without a descriptor, the release alone.
    monkeypatch.setenv("INFERSSD_RESOURCES", str(tmp_path / "missing.json"))
    assert main(["--version"]) == 0
    assert capsys.readouterr().out == f"inferssd {metadata.version('inferssd')}\n"


def test_version_names_the_build_the_installer_recorded(capsys, monkeypatch, tmp_path):
    descriptor = tmp_path / "engine-resources.json"
    revision = "03af9799129883f2ce84a490556351dd4ca70bbc"
    descriptor.write_text(
        json.dumps(
            {
                "schema": "inferssd-engine 1",
                "binary": "bin/server",
                "revision": revision,
                "build": 202,
                "date": "2026-11-06",
                "branch": "main",
            }
        )
    )
    monkeypatch.setenv("INFERSSD_RESOURCES", str(descriptor))
    assert main(["--version"]) == 0
    version = metadata.version("inferssd")
    assert capsys.readouterr().out == f"inferssd {version} (261106.202.03af979)\n"
    assert main(["--version", "--json"]) == 0
    assert json.loads(capsys.readouterr().out) == {
        "version": version,
        "build": 202,
        "revision": revision,
        "date": "2026-11-06",
        "branch": "main",
        "label": "261106.202.03af979",
    }


def test_version_from_a_source_archive_has_no_label(capsys, monkeypatch, tmp_path):
    descriptor = tmp_path / "engine-resources.json"
    descriptor.write_text(
        json.dumps(
            {
                "schema": "inferssd-engine 1",
                "binary": "bin/server",
                "revision": "unknown",
                "build": None,
                "date": None,
                "branch": None,
            }
        )
    )
    monkeypatch.setenv("INFERSSD_RESOURCES", str(descriptor))
    assert main(["--version", "--json"]) == 0
    value = json.loads(capsys.readouterr().out)
    assert value["label"] is None and value["revision"] is None
    assert main(["--version"]) == 0
    assert capsys.readouterr().out == f"inferssd {metadata.version('inferssd')}\n"


def test_version_rejects_other_arguments(capsys):
    with pytest.raises(SystemExit) as error:
        main(["--version", "serve"])
    assert error.value.code == 2
