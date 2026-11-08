# SPDX-License-Identifier: Apache-2.0
"""The machine characteristics an executed launch derives its strategy from.

Before it resolves, an executed launch makes sure the per-user characteristics
file describes this machine and the drive holding the model. What is missing
or out of date is measured first and saved, so the first launch on a machine
takes a few minutes and later launches reuse the file. A file named with
``--machine`` is used as it is and never measured.

Profiling never stops a launch: a probe that is missing or fails is reported
and its section recorded as failed, and the strategy leaves out what it cannot
ground.
"""

from __future__ import annotations

import os
import sys
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from inferssd.contracts import ContractError
from inferssd.machine.characteristics import Characteristics, dumps, load
from inferssd.machine.identity import Host, NativeHost
from inferssd.machine.probes import EXECUTABLES, Probes, Runner, run_process
from inferssd.machine.profile import Clock, Progress, Survey, profile, survey
from inferssd.machine.store import (
    CPU_GROUP,
    LockBusy,
    default_path,
    exclusive,
    load_current,
    save,
)

PROFILED = ("linux", "darwin", "windows")


@dataclass(frozen=True)
class MachineSource:
    """Where an executed launch takes its characteristics from."""

    path: Optional[Path] = None  # --machine: this file, as it is
    reprofile: bool = False  # --reprofile: measure every section again first


def launch_progress(line: str) -> None:
    print("inferssd launch: " + line, file=sys.stderr, flush=True)


def _saved(path: Path, progress: Progress) -> Optional[Characteristics]:
    try:
        return load_current(path)
    except (ContractError, UnicodeDecodeError, OSError) as error:
        progress(f"ignoring the unreadable machine profile {path}: {error}")
        return None


RETRIES = ("last attempt failed", "last attempt was partial")


def _probe(section: str) -> str:
    return "cpu" if section in CPU_GROUP else "memory" if section == "gpus" else "ssd"


def needed(found: Survey) -> set[str]:
    """The probes the run FOUND plans will start."""
    names = {_probe(name) for name, _ in found.refresh.sections}
    if "memory" not in found.identity["soft"]["probe_versions"] or found.gpu_failure:
        names.discard("memory")  # no NVIDIA GPU to measure, or nvidia-smi failed
    return names


def futile(found: Survey, probes: Mapping[str, str]) -> bool:
    """FOUND only retries sections that failed, each with a probe that is still missing."""
    sections = found.refresh.sections
    return (
        bool(sections)
        and not found.refresh.stale_storage
        and all(reason in RETRIES and _probe(name) not in probes for name, reason in sections)
    )


def characteristics(
    source: MachineSource,
    platform: str,
    model: str,
    probes: Mapping[str, str],
    *,
    host: Optional[Host] = None,
    runner: Optional[Runner] = None,
    environ: Mapping[str, str] = os.environ,
    home: Optional[Path] = None,
    clock: Clock = time.time,
    progress: Progress = launch_progress,
) -> Optional[str]:
    """The characteristics text for a launch of MODEL on PLATFORM, or None without any.

    PROBES are the probe executables by name ("ssd", "cpu", "memory"). An
    explicit file must be valid; everything else only warns.
    """
    if source.path is not None:
        return dumps(load(source.path))
    live = host or NativeHost()
    if platform not in PROFILED or live.platform() != platform:
        progress(f"machine profiling is not supported on {platform} here; launching without it")
        return None
    path = default_path(platform, environ, home)
    try:
        with exclusive(path):
            previous = _saved(path, progress)
            found = survey(live, [model], previous, source.reprofile, progress)
            if previous is not None and found.current:
                return dumps(previous)
            if previous is not None and futile(found, probes):
                missing = sorted({EXECUTABLES[_probe(name)] for name, _ in found.refresh.sections})
                progress(
                    f"warning: {', '.join(missing)} still not found; launching with the saved "
                    "profile, which lacks what they measure"
                )
                return dumps(previous)
            if source.reprofile:
                progress("measuring this machine again")
            elif previous is None:
                progress("measuring this machine before its first launch")
            else:
                progress("measuring what changed on this machine")
            progress(f"later launches reuse {path}")
            for name, program in EXECUTABLES.items():
                if name in needed(found) and name not in probes:
                    progress(
                        f"warning: {program} was not found next to the server, in the "
                        "deployment descriptor or in the checkout's builds; its sections "
                        "are recorded as failed"
                    )
            value = profile(
                live,
                Probes(tuple(probes.items()), runner or run_process, environ),
                [model],
                previous,
                source.reprofile,
                clock=clock,
                progress=progress,
                surveyed=found,
            )
            if value is not previous:
                save(value, path)
    except LockBusy:
        progress("another inferssd profile is measuring this machine; using the saved profile")
        saved = _saved(path, progress)
        return dumps(saved) if saved is not None else None
    except (OSError, ValueError, RuntimeError) as error:
        progress(f"warning: the machine could not be profiled ({error}); using the saved profile")
        saved = _saved(path, progress)
        return dumps(saved) if saved is not None else None
    old = previous["sections"] if previous is not None else {}
    for name, section in sorted(value["sections"].items()):
        if old.get(name) is not section and section["status"] != "ok":
            progress(f"warning: {name} {section['status']}: {section['detail']}")
    return dumps(value)
