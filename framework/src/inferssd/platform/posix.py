# SPDX-License-Identifier: Apache-2.0
"""POSIX mechanisms behind the shared ownership algorithms: descriptors and sessions.

The device lock is flock on one open file description, which every owner
inherits; it is released only when the last of them closes it. Children run in
sessions of their own and receive descriptors at their own numbers.
"""

from __future__ import annotations

import os
import signal
import socket
import stat
import subprocess
import sys
from collections.abc import Sequence
from typing import Any

from inferssd.platform.policy import lock_held

if sys.platform != "win32":
    import fcntl

# Termination requests that an owner turns into an orderly stop.
STOP_SIGNALS: tuple[int, ...] = tuple(
    getattr(signal, name) for name in ("SIGINT", "SIGTERM", "SIGHUP") if hasattr(signal, name)
)


def _lock(fd: int, path: str) -> None:
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        raise lock_held(path) from error


def acquire_lock(path: str) -> int:
    """Open and exclusively lock the device lock; a held lock is never waited for."""
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        _lock(fd, path)
    except BaseException:
        os.close(fd)
        raise
    return fd


def borrow_lock(fd: int, path: str) -> int:
    """A copy of a caller's held lease on the existing device lock inode."""
    actual, expected = os.fstat(fd), os.stat(path)
    if not stat.S_ISREG(actual.st_mode) or (actual.st_dev, actual.st_ino) != (
        expected.st_dev,
        expected.st_ino,
    ):
        raise ValueError("inherited lease differs from the existing device lock inode")
    # dup retains the same flock open-file-description. Closing this
    # owner's copy does not release the caller's batch lease.
    copy = os.dup(fd)
    try:
        _lock(copy, path)
    except BaseException:
        os.close(copy)
        raise
    return copy


def descriptor(fd: int) -> int:
    """The value that names fd in a child started with pass_fds."""
    return fd


def adopt(value: int) -> int:
    """The descriptor a parent passed as value."""
    return value


def link() -> tuple[socket.socket, socket.socket]:
    """A controller connection and the end its guardian inherits."""
    return socket.socketpair()


def watch(value: int) -> socket.socket:
    """The guardian's nonblocking end of its controller connection: closed means gone."""
    connection = socket.socket(fileno=value)
    connection.setblocking(False)
    return connection


def spawn_detached(argv: Sequence[str], **options: Any) -> subprocess.Popen[bytes]:
    """An owner that outlives its starter: a guardian, in a session of its own."""
    return subprocess.Popen(argv, **options)


def spawn_owned(argv: Sequence[str], **options: Any) -> subprocess.Popen[bytes]:
    """An owned child; start_new_session=True gives it the process group stop_owned ends."""
    return subprocess.Popen(argv, **options)


def replace(source: str, destination: str) -> None:
    os.replace(source, destination)
