# SPDX-License-Identifier: Apache-2.0
"""File location, atomic replacement under the lock, and the refresh plan."""

from __future__ import annotations

import copy
import json
import os
import sys
from pathlib import Path

import pytest
from test_machine_characteristics import DRIVE, sample

from inferssd.contracts import ContractError
from inferssd.machine import store
from inferssd.machine.store import (
    LockBusy,
    default_path,
    exclusive,
    load_current,
    plan_refresh,
    save,
)

OTHER = "linux:sata:Example SSD:" + "cd" * 32


def test_default_locations(tmp_path):
    home = tmp_path / "home"
    assert default_path("linux", {}, home) == home / ".local/state/inferssd/machine.json"
    state = str(tmp_path / "state")
    assert default_path("linux", {"XDG_STATE_HOME": state}, home) == Path(
        state, "inferssd/machine.json"
    )
    assert default_path("linux", {"XDG_STATE_HOME": "relative"}, home) == (
        home / ".local/state/inferssd/machine.json"
    )
    assert default_path("darwin", {}, home) == (
        home / "Library/Application Support/inferssd/machine.json"
    )
    local = str(tmp_path / "Local")
    assert default_path("windows", {"LOCALAPPDATA": local}, home) == Path(
        local, "inferssd", "machine.json"
    )
    assert default_path("windows", {}, home) == home / "AppData/Local/inferssd/machine.json"


def test_default_location_follows_this_host(tmp_path):
    system = {"darwin": "darwin", "win32": "windows"}.get(sys.platform, "linux")
    assert default_path(environ={}, home=tmp_path) == default_path(system, {}, tmp_path)


def test_save_is_atomic_and_load_current_reads_it(tmp_path):
    path = tmp_path / "state" / "machine.json"
    assert load_current(path) is None
    save(sample(), path)
    assert load_current(path) == sample()
    assert sorted(item.name for item in path.parent.iterdir()) == [
        "machine.json",
        "machine.json.lock",
    ]
    if sys.platform != "win32":
        assert path.stat().st_mode & 0o777 == 0o600


def test_failed_replace_keeps_the_old_file(tmp_path, monkeypatch):
    path = tmp_path / "machine.json"
    save(sample(), path)
    before = path.read_bytes()
    changed = sample()
    changed["created_utc"] = "2026-10-30T00:00:00Z"

    def broken(source, target):
        raise OSError("disk full")

    monkeypatch.setattr(store.os, "replace", broken)
    with pytest.raises(OSError, match="disk full"):
        save(changed, path)
    assert path.read_bytes() == before
    assert sorted(item.name for item in tmp_path.iterdir()) == ["machine.json", "machine.json.lock"]


def test_invalid_values_are_never_written(tmp_path):
    path = tmp_path / "machine.json"
    broken = sample()
    del broken["memory"]
    with pytest.raises(ContractError, match="memory is required"):
        save(broken, path)
    assert not path.exists()
    path.write_text(json.dumps(broken))
    with pytest.raises(ContractError, match="memory is required"):
        load_current(path)


@pytest.mark.skipif(sys.platform == "win32", reason="flock")
def test_lock_excludes_other_holders_and_is_reentrant(tmp_path):
    import fcntl

    path = tmp_path / "machine.json"
    with exclusive(path):
        save(sample(), path)  # reentrant inside this process
    holder = os.open(tmp_path / "machine.json.lock", os.O_RDWR)
    try:
        fcntl.flock(holder, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(LockBusy, match="another inferssd profile holds"):
            save(sample(), path, wait_s=0)
        ticks = iter([0.0, 0.05, 0.2])
        naps = []
        with pytest.raises(LockBusy):
            with exclusive(path, 0.1, clock=lambda: next(ticks), sleep=naps.append):
                pass
        assert naps == [0.1]
    finally:
        os.close(holder)
    with exclusive(path):
        pass


def identity():
    return copy.deepcopy(sample()["identity"])


def test_plan_without_saved_profile_or_on_request_measures_everything():
    full = plan_refresh(None, identity(), [DRIVE], False)
    assert full.full and full.stale_storage
    assert full.sections == tuple(
        (name, "no saved profile")
        for name in ("cpu", "memory", "compute", "gpus", "storage:" + DRIVE)
    )
    again = plan_refresh(sample(), identity(), [], True)
    assert again.full and [name for name, _ in again.sections] == [
        "cpu",
        "memory",
        "compute",
        "gpus",
    ]
    assert again.reason("cpu") == "re-profile requested"


def test_plan_reuses_everything_when_nothing_changed():
    plan = plan_refresh(sample(), identity(), [DRIVE], False)
    assert plan.sections == () and not plan.full and not plan.stale_storage


def test_hard_identity_change_measures_everything():
    live = identity()
    live["hard"]["cpu_brand"] = "Another CPU"
    plan = plan_refresh(sample(), live, [DRIVE], False)
    assert plan.full and plan.reason("storage:" + DRIVE) == "hardware changed: cpu_brand"
    live = identity()
    live["hard"]["ram_bytes"] += live["hard"]["ram_bytes"] // 200  # 0.5%: a kernel reservation
    assert plan_refresh(sample(), live, [DRIVE], False).sections == ()


@pytest.mark.parametrize("field,value", [("kernel", "6.9.0"), ("os_release", "Ubuntu 26.04 LTS")])
def test_kernel_or_os_change_measures_storage_only(field, value):
    live = identity()
    live["soft"][field] = value
    plan = plan_refresh(sample(), live, [DRIVE, OTHER], False)
    assert [name for name, _ in plan.sections] == ["storage:" + DRIVE, "storage:" + OTHER]
    assert plan.stale_storage and not plan.full


def test_driver_change_measures_gpus():
    live = identity()
    live["soft"]["gpu_driver"] = "590.1"
    plan = plan_refresh(sample(), live, [DRIVE], False)
    assert plan.sections == (("gpus", "GPU driver changed"),) and not plan.stale_storage


@pytest.mark.parametrize(
    "probe,sections,stale",
    [
        ("cpu", ["cpu", "memory", "compute"], False),
        ("memory", ["gpus"], False),
        ("ssd", ["storage:" + DRIVE], True),
    ],
)
def test_probe_schema_change_measures_its_sections(probe, sections, stale):
    live = identity()
    live["soft"]["probe_versions"][probe] = f"inferssd-{probe}/2"
    plan = plan_refresh(sample(), live, [DRIVE], False)
    assert [name for name, _ in plan.sections] == sections
    assert (
        plan.reason(sections[0]).startswith("probe schema inferssd-")
        and plan.stale_storage == stale
    )


def test_new_drive_measures_that_drive_only():
    plan = plan_refresh(sample(), identity(), [DRIVE, OTHER], False)
    assert plan.sections == (("storage:" + OTHER, "new drive"),) and not plan.stale_storage


def test_partial_sections_are_measured_again():
    saved = sample()
    saved["sections"]["gpus"]["status"] = "partial"
    plan = plan_refresh(saved, identity(), [DRIVE], False)
    assert plan.sections == (("gpus", "last attempt was partial"),)
    assert plan.including("cpu", "x").sections == (
        ("cpu", "x"),
        ("gpus", "last attempt was partial"),
    )
    assert plan.including("gpus", "x") is plan


def test_failed_or_missing_sections_are_measured_again():
    saved = sample()
    saved["sections"]["compute"]["status"] = "failed"
    saved["sections"]["storage:" + OTHER] = dict(saved["sections"]["cpu"], status="failed")
    plan = plan_refresh(saved, identity(), [DRIVE, OTHER], False)
    assert plan.sections == (
        ("cpu", "last attempt failed"),
        ("memory", "last attempt failed"),
        ("compute", "last attempt failed"),
        ("storage:" + OTHER, "last attempt failed"),
    )
