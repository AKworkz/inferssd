# SPDX-License-Identifier: Apache-2.0
"""Windows device lock, descriptor passing and leases, over inherited handles.

The device lock is its lock file opened with no sharing. No other open of the
file can succeed while any handle to that open exists, in any process: every
owner that inherits a copy holds the lock until the last of them closes it, as
with flock on one open file description. LockFileEx and msvcrt.locking are not
equivalent: Windows releases their locks when the locking process exits, even
while its children still hold the handle.

A child receives a handle by inheritance, restricted to the handles named for
it and under the handle's own value. A liveness channel is an anonymous pipe
whose write end only the watched owner holds; it breaks when that owner exits.
"""

from __future__ import annotations

import os
import stat
import time
from pathlib import Path
from typing import Optional, TypedDict

from inferssd.platform.policy import lock_held
from inferssd.platform.windows import win32


class Peer(TypedDict):
    pid: int


def acquire_lock(path: str) -> int:
    """Open the device lock exclusively; a held lock is never waited for."""
    handle = win32.open_exclusive(path)
    if handle is None:
        raise lock_held(path)
    return win32.handle_fd(handle)


def borrow_lock(fd: int, path: str) -> int:
    """A copy of a caller's exclusive open of the existing device lock file."""
    actual, expected = os.fstat(fd), os.stat(path)
    if not stat.S_ISREG(actual.st_mode) or (actual.st_dev, actual.st_ino) != (
        expected.st_dev,
        expected.st_ino,
    ):
        raise ValueError("inherited lease differs from the existing device lock inode")
    # fd opens the lock file with data access, so an open that shares nothing
    # can coexist with it only as fd's own open: the caller holds the lock.
    if not win32.opened_exclusively(path):
        raise ValueError("inherited lease is not an exclusive open of the device lock")
    # A duplicate handle refers to the same open, which is the lock.
    return os.dup(fd)


def descriptor(fd: int) -> int:
    """The value that names fd in a child started with pass_fds: its handle."""
    return win32.fd_handle(fd)


def adopt(value: int) -> int:
    """A descriptor for the handle a parent passed as value."""
    return win32.handle_fd(value)


class End:
    """One end of an anonymous pipe, as a descriptor."""

    def __init__(self, fd: int) -> None:
        self.fd: Optional[int] = fd

    def fileno(self) -> int:
        assert self.fd is not None
        return self.fd

    def close(self) -> None:
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None


def link() -> tuple[End, End]:
    """A controller connection and the end its guardian inherits: an anonymous pipe."""
    read, write = os.pipe()
    return End(write), End(read)


class PipeWatch(End):
    """A pipe's read end, peeked as a nonblocking socket is with MSG_PEEK.

    recv returns waiting bytes without consuming them, b"" once the writer is
    gone, and raises BlockingIOError while the pipe is open and empty.
    """

    def setblocking(self, flag: bool) -> None:
        if flag:
            raise ValueError("a watched pipe is never read blocking")

    def recv(self, size: int, flags: int = 0) -> bytes:
        data = win32.pipe_peek(win32.fd_handle(self.fileno()), size)
        if data is None:
            return b""
        if not data:
            raise BlockingIOError("nothing waiting in the pipe")
        return data


def watch(value: int) -> PipeWatch:
    """The guardian's end of its controller connection: closed means gone."""
    return PipeWatch(adopt(value))


class LeaseServer:
    """The guardian's lease of its device lock to the one worker it starts.

    The worker inherits the lock's handle and the read end of a pipe whose
    write end stays with this guardian. The lease's address names both handles,
    whose values the worker inherits unchanged; only the started worker has them.
    """

    def __init__(self, lock_fd: int) -> None:
        self.lock_fd = lock_fd
        read, self._write = os.pipe()
        self._read: Optional[int] = read
        self.path = f"{win32.fd_handle(lock_fd)}:{win32.fd_handle(read)}"
        self.peer: Optional[Peer] = None
        self.expected_pid: Optional[int] = None

    @property
    def inherit(self) -> tuple[int, ...]:
        """The descriptors the worker inherits."""
        assert self._read is not None
        return (self.lock_fd, self._read)

    def handed(self) -> None:
        """The worker exists and holds its copies; the guardian keeps only the writer."""
        if self._read is not None:
            os.close(self._read)
            self._read = None

    def poll(self) -> None:
        # The lease was delivered when the worker was created with its handles.
        if self.peer is None and self.expected_pid is not None:
            self.peer = {"pid": self.expected_pid}

    def close(self) -> None:
        self.handed()
        os.close(self._write)


class Lease:
    """The worker's copy of its guardian's lock, and the guardian's liveness."""

    def __init__(
        self,
        path: str,
        lock_path: Optional[str] = None,
        expected_identity: Optional[tuple[int, int]] = None,
    ) -> None:
        lock, channel = (int(value) for value in path.split(":"))
        self.fd: Optional[int] = None
        self.connection = PipeWatch(win32.handle_fd(channel))
        try:
            self.fd = win32.handle_fd(lock)
            got = os.fstat(self.fd)
            if expected_identity is None:
                if lock_path is None:
                    raise ValueError("lock_path or expected_identity is required")
                expected = Path(lock_path).stat()
                expected_identity = (expected.st_dev, expected.st_ino)
            if (got.st_dev, got.st_ino) != expected_identity:
                raise RuntimeError("lock lease inode differs")
        except BaseException:
            self.close()
            raise

    def check(self) -> None:
        try:
            data = self.connection.recv(1, 0)
        except BlockingIOError:
            return
        if data == b"":
            raise InterruptedError("controller connection lost; cleanup while retaining lock lease")
        raise RuntimeError("unexpected controller lease message")

    def close(self) -> None:
        self.connection.close()
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None


def replace(source: str, destination: str) -> None:
    """os.replace, waiting out a reader that has destination open.

    Python opens files without FILE_SHARE_DELETE, and Windows refuses to
    replace a file while such an open exists; readers hold it briefly.
    """
    until = time.monotonic() + 5
    while True:
        try:
            os.replace(source, destination)
            return
        except PermissionError:
            if time.monotonic() > until:
                raise
            time.sleep(0.01)
