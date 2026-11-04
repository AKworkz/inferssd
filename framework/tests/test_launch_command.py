"""The recorded-host adapter behind the typed launch dry-run: explicit facts, no host queries."""

import subprocess

import pytest

from inferssd.launch.recorded import RecordedArtifacts
from inferssd.launch.types import Fact, LaunchInput


def test_recorded_adapter_never_queries_host(monkeypatch, tmp_path):
    def forbidden(*a, **kw):
        raise AssertionError("unexpected process")

    monkeypatch.setattr(subprocess, "run", forbidden)
    monkeypatch.setattr(subprocess, "check_output", forbidden)
    adapter = RecordedArtifacts(
        LaunchInput("/repo", ()),
        {"gpu_available": False},
        run=forbidden,
        replace=forbidden,
        python="unused",
    )
    assert adapter(Fact("gpu_available")).value is False
    with pytest.raises(ValueError, match="Missing recorded host fact: physical_memory"):
        adapter(Fact("physical_memory"))
    assert list(tmp_path.iterdir()) == []


def test_a_settings_file_is_read_as_utf8_whatever_the_locale(tmp_path):
    settings = tmp_path / "input.ssd"
    text = "inferssd-settings 1\nprefetch.depth=4 # 来源=测量, Kovač\n"
    settings.write_bytes(text.encode())
    adapter = RecordedArtifacts(
        LaunchInput("/repo", ()), {}, run=None, replace=None, python="unused"
    )
    assert adapter(Fact("read_text", (str(settings),))).value == text


@pytest.mark.parametrize(
    "name,value",
    [
        ("gpu_available", "false"),
        ("has_uring", "false"),
        ("ss_available", 0),
        ("euid", "0"),
        ("euid", False),
        ("memory_kib", [1]),
        ("memory_kib", [True, 2]),
        ("memory_kib", ["1", 2]),
        ("memory_kib", {}),
        ("gpu_memory", 16384),
        ("filesystem", None),
        ("gpu_processes", []),
        ("physical_memory", True),
        ("physical_memory", "not an integer"),
        ("disk_free", 1.5),
        ("default_threads", False),
    ],
)
def test_requested_host_fact_type_is_explicit(name, value):
    def forbidden(*a, **kw):
        raise AssertionError("no effect")

    adapter = RecordedArtifacts(
        LaunchInput("/repo", ()), {name: value}, run=forbidden, replace=forbidden, python="unused"
    )
    with pytest.raises(ValueError, match="Recorded host fact " + name + ":"):
        adapter(Fact(name))


def test_unused_malformed_host_fact_is_not_read(tmp_path):
    def forbidden(*a, **kw):
        raise AssertionError("no effect")

    adapter = RecordedArtifacts(
        LaunchInput("/repo", ()),
        {"gpu_available": "false", "euid": "0", "platform_system": "Darwin"},
        run=forbidden,
        replace=forbidden,
        python="unused",
    )
    assert adapter(Fact("is_file", (str(tmp_path / "missing"),))).value is False
    assert adapter(Fact("platform_system")).value == "Darwin"


@pytest.mark.parametrize(
    "name,value,expected",
    [
        ("gpu_memory", "[N/A]", "[N/A]"),
        ("euid", 0, 0),
        ("memory_kib", [16, 14], [16, 14]),
        ("default_threads", 8, "8"),
        ("default_threads", "8", "8"),
        ("physical_memory", "34359738368\n", "34359738368\n"),
        ("disk_free", 64, "64"),
    ],
)
def test_recorded_host_valid_representations(name, value, expected):
    def forbidden(*a, **kw):
        raise AssertionError("no effect")

    adapter = RecordedArtifacts(
        LaunchInput("/repo", ()), {name: value}, run=forbidden, replace=forbidden, python="unused"
    )
    assert adapter(Fact(name)).value == expected
