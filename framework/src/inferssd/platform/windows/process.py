# SPDX-License-Identifier: Apache-2.0
"""Windows process ownership: job objects for groups and scopes, console Ctrl+C for stops.

Windows has no process groups to signal. Each owned server runs in a
kill-on-close job of its own, its group: its descendants stay in that job, and
closing the job's last handle, which only its worker holds, ends them all. A
graceful stop is Ctrl+C on the owned tree's console. The guardian starts in a
new console without a window, which its worker and that worker's server share
and nothing else attaches to; Ctrl+C on that console reaches exactly the owned
tree. The guardian, started in a new process group, ignores Ctrl+C itself.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import IO, Any, Optional, Union

from inferssd.platform.policy import Policy
from inferssd.platform.process import Cleanup, Process
from inferssd.platform.windows import win32
from inferssd.platform.windows.lease import LeaseServer
from inferssd.platform.windows.resources import commit_limit

# Ctrl+Break arrives as signal.SIGBREAK, which only Windows builds of Python define.
SIGBREAK = 21
# Termination requests that an owner turns into an orderly stop: SIGINT, SIGTERM, SIGBREAK.
STOP_SIGNALS: tuple[int, ...] = (2, 15, SIGBREAK)

_Stream = Union[int, IO[Any], None]


class Group:
    """An owned process and the kill-on-close job that is its group."""

    def __init__(self, process: subprocess.Popen[bytes], job: int) -> None:
        self.process, self.job = process, job

    @property
    def pid(self) -> int:
        return self.process.pid

    @property
    def returncode(self) -> Optional[int]:
        return self.process.returncode

    def poll(self) -> Optional[int]:
        return self.process.poll()

    def wait(self, timeout: Optional[float] = None) -> int:
        return self.process.wait(timeout)


def interpreter(argv: Sequence[str], env: Mapping[str, str]) -> tuple[list[str], dict[str, str]]:
    """Run this Python as itself rather than through a virtual environment's redirector.

    A venv's python.exe starts the base interpreter as a child, so the PID of
    the process it starts would not be the PID of the owner. As multiprocessing
    does, start the base interpreter and name the venv's launcher to it.
    """
    command, environment = list(argv), dict(env)
    base = getattr(sys, "_base_executable", sys.executable)
    if command and os.path.normcase(command[0]) == os.path.normcase(sys.executable):
        if os.path.normcase(base) != os.path.normcase(sys.executable):
            command[0] = base
            environment["__PYVENV_LAUNCHER__"] = sys.executable
    return command, environment


def create(
    argv: Sequence[str],
    *,
    env: Mapping[str, str],
    stdout: _Stream = None,
    stderr: _Stream = None,
    stdin: _Stream = None,
    pass_fds: Sequence[int] = (),
    flags: int = 0,
    job: Optional[int] = None,
) -> subprocess.Popen[bytes]:
    """Start argv with exactly pass_fds inherited, inside job from its first instruction."""
    command, environment = interpreter(argv, env)
    process = win32.create_process(
        command,
        env=environment,
        stdout=stdout,
        stderr=stderr,
        stdin=stdin,
        inherit=[win32.fd_handle(fd) for fd in pass_fds],
        flags=flags | (win32.CREATE_SUSPENDED if job is not None else 0),
    )
    if job is not None:
        try:
            # Created suspended: nothing has run, so nothing escaped the job.
            win32.assign(job, process.pid)
            win32.resume(process.pid)
        except BaseException:
            process.kill()
            process.wait()
            raise
    return process


def spawn_detached(
    argv: Sequence[str],
    *,
    env: Mapping[str, str],
    stdout: _Stream = None,
    stderr: _Stream = None,
    stdin: _Stream = None,
    start_new_session: bool = False,
    pass_fds: Sequence[int] = (),
) -> subprocess.Popen[bytes]:
    """A guardian: a new console without a window, in a new process group, in no new job.

    Its console is the owned tree's; the user's console and its Ctrl+C stay the
    controller's. No job is created, so the guardian outlives its controller.
    """
    flags = win32.CREATE_NEW_PROCESS_GROUP | win32.CREATE_NO_WINDOW if start_new_session else 0
    return create(
        argv, env=env, stdout=stdout, stderr=stderr, stdin=stdin, pass_fds=pass_fds, flags=flags
    )


def spawn_owned(
    argv: Sequence[str],
    *,
    env: Mapping[str, str],
    stdout: _Stream = None,
    stderr: _Stream = None,
    stdin: _Stream = None,
    start_new_session: bool = False,
    pass_fds: Sequence[int] = (),
) -> Process:
    """An owned child; start_new_session=True gives it the kill-on-close job stop_owned ends.

    The job repeats the commit limit of the job this process runs in, which
    the kernel applies to the child anyway, so the child's innermost job, the
    one a process can read, states the envelope it runs in.
    """
    options: dict[str, Any] = dict(env=env, stdout=stdout, stderr=stderr, stdin=stdin)
    if not start_new_session:
        return create(argv, pass_fds=pass_fds, **options)
    limit = win32.job_memory(None)["job_memory_limit"] if win32.in_job() else None
    job = win32.create_job(kill_on_close=True, memory_limit=limit)
    try:
        return Group(create(argv, pass_fds=pass_fds, job=job, **options), job)
    except BaseException:
        win32.close(job)
        raise


def group_exists(process: Process) -> bool:
    """A process in the job that is the process's group; without one, the process itself."""
    process.poll()
    if isinstance(process, Group):
        return win32.job_active(process.job) > 0
    return process.returncode is None


def stop_owned(process: Process, timeout: float) -> Cleanup:
    """Ctrl+C, a bounded wait, then the job's termination: the POSIX stop's steps."""
    if not isinstance(process, Group):
        raise TypeError("only a process started in a group of its own can be stopped as one")
    was_running = process.poll() is None
    orphan = not was_running and group_exists(process)
    if group_exists(process):
        try:
            win32.interrupt_console()
        except OSError:
            pass  # without a console only the termination below remains
    until = time.monotonic() + timeout
    while group_exists(process) and time.monotonic() < until:
        time.sleep(0.05)
    forced = group_exists(process)
    if forced:
        win32.terminate_job(process.job, 1)
    process.wait(timeout=5)
    until = time.monotonic() + 2
    while group_exists(process) and time.monotonic() < until:
        time.sleep(0.05)
    return {
        "returncode": process.returncode,
        "group_gone": not group_exists(process),
        "forced_kill": forced,
        "orphaned_descendants": orphan,
        "was_running": was_running,
    }


class JobScope:
    """The guardian's job: the worker and everything it starts, which the guardian ends.

    Unlike a server's group it does not end when the guardian's handle closes,
    so a worker whose guardian died still stops its server in order. Its commit
    limit is the policy's memory budget, as a Linux scope's MemoryMax is, plus
    the dedicated memory of the discrete GPUs, which WDDM charges to commit
    (resources.commit_limit). It is also the guardian's native port: the
    worker's lease, start and stop signals.
    """

    unit: Optional[str] = None
    stop_signals = STOP_SIGNALS

    def __init__(self, policy: Policy, directory: Path) -> None:
        self.directory = directory
        self.job = win32.create_job(kill_on_close=False, memory_limit=commit_limit(policy))
        self.native = self

    def lease(self, lock_fd: int) -> LeaseServer:
        return LeaseServer(lock_fd)

    def spawn(
        self, argv: list[str], environment: Mapping[str, str], log: IO[bytes], lease: Any
    ) -> subprocess.Popen[bytes]:
        """The worker, in this job, inheriting its lease; it shares this console."""
        # The guardian ignores Ctrl+C; its worker must not, so that Ctrl+C on
        # this console stops the worker and, through it, its server.
        win32.ignore_ctrl_c(False)
        try:
            worker = create(
                argv,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                pass_fds=lease.inherit,
                job=self.job,
            )
        finally:
            win32.ignore_ctrl_c(True)
        lease.handed()
        return worker

    def stop(self) -> dict[str, Any]:
        """End every process left in the job and wait, bounded, until none is left."""
        active = win32.job_active(self.job)
        if active:
            win32.terminate_job(self.job, 1)
        until = time.monotonic() + 10
        while win32.job_active(self.job) and time.monotonic() < until:
            time.sleep(0.05)
        left = win32.job_active(self.job)
        return {
            "job": {"terminated_processes": active, "memory": win32.job_memory(self.job)},
            "state": {
                "gone": left == 0,
                "active_processes": left,
                "pids": win32.job_processes(self.job) if left else [],
            },
        }

    def request_stop(self, pid: int) -> None:
        # The worker's own stop request: it stops its server with one Ctrl+C.
        (self.directory / "STOP").touch()

    def close(self) -> None:
        win32.close(self.job)
