"""The inferssd command offers serve, cli and profile, and lists all three."""

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


def test_version_is_the_installed_packages(capsys):
    with pytest.raises(SystemExit) as error:
        main(["--version"])
    assert error.value.code == 0
    assert capsys.readouterr().out == f"inferssd {metadata.version('inferssd')}\n"
