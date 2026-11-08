# SPDX-License-Identifier: Apache-2.0
"""Where the characteristics file lives, how it is replaced, and what to re-measure."""

from __future__ import annotations

import contextlib
import os
import sys
import tempfile
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Optional

from inferssd.machine.characteristics import (
    BASE_SECTIONS,
    STORAGE_PREFIX,
    Characteristics,
    Identity,
    dumps,
    load,
)
from inferssd.machine.identity import NativeHost, same_hardware

FILENAME = "machine.json"
# One probe run measures all three; they are refreshed together.
CPU_GROUP = ("cpu", "memory", "compute")
_HELD: set[str] = set()


class LockBusy(RuntimeError):
    """Another process holds the characteristics lock."""


def default_path(
    system: Optional[str] = None,
    environ: Mapping[str, str] = os.environ,
    home: Optional[Path] = None,
) -> Path:
    """The per-user location of the characteristics file on SYSTEM (default: this host)."""
    system = system or NativeHost().platform()
    home = home if home is not None else Path.home()
    if system == "darwin":
        return home / "Library" / "Application Support" / "inferssd" / FILENAME
    if system == "windows":
        local = environ.get("LOCALAPPDATA")
        return (Path(local) if local else home / "AppData" / "Local") / "inferssd" / FILENAME
    # XDG ignores a relative XDG_STATE_HOME.
    state = environ.get("XDG_STATE_HOME", "")
    base = Path(state) if os.path.isabs(state) else home / ".local" / "state"
    return base / "inferssd" / FILENAME


def load_current(path: Optional[Path] = None) -> Optional[Characteristics]:
    """The saved characteristics, or None when there are none yet."""
    path = path or default_path()
    if not path.exists():
        return None
    return load(path)


def _try_lock(fd: int) -> bool:
    if sys.platform == "win32":
        import msvcrt

        try:
            msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
        except OSError:
            return False
        return True
    import fcntl

    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return False
    return True


def _unlock(fd: int) -> None:
    if sys.platform == "win32":
        import msvcrt

        os.lseek(fd, 0, os.SEEK_SET)
        msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
    else:
        import fcntl

        fcntl.flock(fd, fcntl.LOCK_UN)


@contextlib.contextmanager
def exclusive(
    path: Path,
    wait_s: float = 0.0,
    *,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> Iterator[None]:
    """Hold the lock file next to PATH; reentrant within this process."""
    lock = path.with_name(path.name + ".lock")
    lock.parent.mkdir(parents=True, exist_ok=True)
    key = str(lock.resolve())
    if key in _HELD:
        yield
        return
    fd = os.open(lock, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        deadline = clock() + wait_s
        while not _try_lock(fd):
            if clock() >= deadline:
                raise LockBusy(f"another inferssd profile holds {lock}")
            sleep(0.1)
        _HELD.add(key)
        try:
            yield
        finally:
            _HELD.discard(key)
            _unlock(fd)
    finally:
        os.close(fd)


def save(value: Characteristics, path: Optional[Path] = None, *, wait_s: float = 10.0) -> None:
    """Validate, then replace PATH (default: the per-user file) atomically under the lock."""
    path = path or default_path()
    content = dumps(value)
    with exclusive(path, wait_s):
        fd, temporary = tempfile.mkstemp(dir=path.parent, prefix="." + path.name + ".")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        except BaseException:
            with contextlib.suppress(OSError):
                os.unlink(temporary)
            raise


@dataclass(frozen=True)
class Refresh:
    """Sections to measure, each with its reason, in measurement order.

    ``full``: nothing saved is reused. ``stale_storage``: saved drives that are
    not measured now are dropped, because their measurements no longer apply.
    """

    sections: tuple[tuple[str, str], ...]
    full: bool = False
    stale_storage: bool = False

    def __contains__(self, section: object) -> bool:
        return any(name == section for name, _ in self.sections)

    def reason(self, section: str) -> Optional[str]:
        return dict(self.sections).get(section)

    def including(self, section: str, reason: str) -> Refresh:
        """This refresh with SECTION measured too, kept in measurement order."""
        if section in self:
            return self
        rank = {name: index for index, name in enumerate(BASE_SECTIONS)}
        sections = sorted(
            (*self.sections, (section, reason)), key=lambda item: rank.get(item[0], len(rank))
        )
        return replace(self, sections=tuple(sections))


def plan_refresh(
    saved: Optional[Characteristics],
    live: Identity,
    drives: Sequence[str],
    reprofile: bool,
) -> Refresh:
    """What to (re)measure given the saved file, the live identity and the model drives."""
    order = (*BASE_SECTIONS, *(STORAGE_PREFIX + key for key in drives))

    def everything(reason: str) -> Refresh:
        return Refresh(tuple((name, reason) for name in order), full=True, stale_storage=True)

    if saved is None:
        return everything("no saved profile")
    if reprofile:
        return everything("re-profile requested")
    changed = same_hardware(saved["identity"], live)
    if changed:
        return everything("hardware changed: " + ", ".join(changed))
    reasons: dict[str, str] = {}
    stale = False

    def mark(names: Sequence[str], reason: str) -> None:
        for name in names:
            reasons.setdefault(name, reason)

    storage = [STORAGE_PREFIX + key for key in drives]
    old, new = saved["identity"]["soft"], live["soft"]
    if old["kernel"] != new["kernel"]:
        mark(storage, "kernel changed")
        stale = True
    if old["os_release"] != new["os_release"]:
        mark(storage, "OS release changed")
        stale = True
    if old["gpu_driver"] != new["gpu_driver"]:
        mark(["gpus"], "GPU driver changed")
    for probe, names in (("cpu", CPU_GROUP), ("memory", ("gpus",)), ("ssd", storage)):
        before = old["probe_versions"].get(probe)
        after = new["probe_versions"].get(probe)
        if before != after:
            mark(names, f"probe schema {before or 'none'} -> {after or 'none'}")
            stale = stale or probe == "ssd"
    # Outside Linux no probe measures GPUs, so a partial gpus section stays partial.
    unmeasured = () if "memory" in new["probe_versions"] else ("gpus",)
    for name in BASE_SECTIONS:
        section = saved["sections"].get(name)
        if section is None:
            mark([name], "not measured yet")
        elif section["status"] == "failed":
            mark([name], "last attempt failed")
        elif section["status"] == "partial" and name not in unmeasured:
            mark([name], "last attempt was partial")
    known = {entry["drive_key"] for entry in saved["storage"]}
    for key in drives:
        if key not in known:
            section = saved["sections"].get(STORAGE_PREFIX + key)
            failed = section is not None and section["status"] == "failed"
            mark([STORAGE_PREFIX + key], "last attempt failed" if failed else "new drive")
    group = [reasons[name] for name in CPU_GROUP if name in reasons]
    if group:
        mark(CPU_GROUP, group[0])
    return Refresh(
        tuple((name, reasons[name]) for name in order if name in reasons), stale_storage=stale
    )
