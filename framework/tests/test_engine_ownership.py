"""Owned-entry fault tests using file locks and substituted process/socket ports."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from types import SimpleNamespace

import pytest
from test_engine_runtime import policy, resolution

from inferssd.engine import owned
from inferssd.platform import worker
from inferssd.platform.run_directory import reserve

if sys.platform != "win32":
    import fcntl

pytestmark = pytest.mark.skipif(
    sys.platform == "win32",
    reason="POSIX owned entry: the flock device lock, systemd scopes and process groups",
)


class Connection:
    def __init__(self, fd=123):
        self.fd, self.closed = fd, False

    def fileno(self):
        return self.fd

    def setblocking(self, value):
        pass

    def recv(self, *args):
        return b""

    def close(self):
        self.closed = True


class Guardian:
    def __init__(self):
        self.returncode, self.timeout = None, False

    def poll(self):
        return self.returncode

    def wait(self, *, timeout):
        if self.timeout:
            raise subprocess.TimeoutExpired("fixture guardian", timeout)
        self.returncode = 0
        return 0


def launch_mock(tmp_path, monkeypatch):
    claim = reserve(str(tmp_path / "run"))
    p, guardian, calls = policy(tmp_path), Guardian(), []
    ends = Connection(101), Connection(102)
    monkeypatch.setattr(owned.socket, "socketpair", lambda: ends)

    def spawn(argv, **kwargs):
        calls.append((argv, kwargs))
        return guardian

    monkeypatch.setattr(owned.subprocess, "Popen", spawn)
    value = owned.OwnedProcess(
        resolution(environment=(("GGML_TEST", "private-value"),)),
        str(claim.directory),
        p,
        reservation=claim,
    )
    return value, guardian, calls, ends


def test_owned_metadata_after_lock_and_stable_module(tmp_path, monkeypatch):
    value, guardian, calls, ends = launch_mock(tmp_path, monkeypatch)
    try:
        args, kw = calls[0]
        assert args[1:4] == ["-I", "-m", "inferssd.platform.worker"]
        assert kw["pass_fds"] == (102, value._lock)
        assert "GGML_TEST" not in kw["env"]
        assert "private-value" not in (value.directory / "runtime.json").read_text()
        assert ends[1].closed and not ends[0].closed
        other = os.open(value.policy.lock_path, os.O_RDWR)
        try:
            with pytest.raises(BlockingIOError):
                fcntl.flock(other, fcntl.LOCK_EX | fcntl.LOCK_NB)
        finally:
            os.close(other)
        (value.directory / "ownership.json").write_text(
            json.dumps(
                {
                    "status": "passed",
                    "server": {"owned_processes_gone": True, "resource_scope_gone": True},
                }
            )
        )
        value.stop()
        assert ends[0].closed and value._lock is None
        other = os.open(value.policy.lock_path, os.O_RDWR)
        try:
            fcntl.flock(other, fcntl.LOCK_EX | fcntl.LOCK_NB)
        finally:
            os.close(other)
    finally:
        if value._lock is not None:
            os.close(value._lock)


def test_busy_device_never_writes_runtime_or_owner_log(tmp_path):
    claim = reserve(str(tmp_path / "run"))
    p = policy(tmp_path)
    fd = os.open(p.lock_path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        before = {x.name: x.read_bytes() for x in claim.directory.iterdir()}
        with pytest.raises(BlockingIOError):
            owned.OwnedProcess(resolution(), str(claim.directory), p, reservation=claim)
        assert before == {x.name: x.read_bytes() for x in claim.directory.iterdir()}
    finally:
        os.close(fd)


def test_unknown_cleanup_retains_lock_and_does_not_kill_guardian(tmp_path, monkeypatch):
    value, guardian, _, ends = launch_mock(tmp_path, monkeypatch)
    guardian.timeout = True
    try:
        with pytest.raises(RuntimeError, match="retains the device lock"):
            value.stop()
        assert value._lock is not None and not ends[0].closed
        assert (value.directory / "STOP").exists()
    finally:
        os.close(value._lock)


def test_failed_but_confirmed_cleanup_releases_then_reports_failure(tmp_path, monkeypatch):
    value, _, _, ends = launch_mock(tmp_path, monkeypatch)
    (value.directory / "ownership.json").write_text(
        json.dumps(
            {
                "status": "failed",
                "server": {"owned_processes_gone": True, "resource_scope_gone": True},
            }
        )
    )
    with pytest.raises(RuntimeError, match="execution failed"):
        value.stop()
    assert value._lock is None and ends[0].closed


def test_failure_reports_the_end_of_the_server_log(tmp_path, monkeypatch):
    value, _, _, _ = launch_mock(tmp_path, monkeypatch)
    (value.directory / "ownership.json").write_text(
        json.dumps(
            {
                "status": "failed",
                "error": "server exited before a requested stop",
                "server": {"owned_processes_gone": True, "resource_scope_gone": True},
            }
        )
    )
    cause = "failed to initialize the context: layer 0: this quantization is not supported yet"
    (value.directory / "server.log").write_text(
        "".join(f"early line {n}\n" for n in range(50)) + cause + "\n" + "cleanup\n" * 5
    )
    with pytest.raises(RuntimeError) as failure:
        value.stop()
    message = str(failure.value)
    assert "server exited before a requested stop" in message
    assert cause in message
    assert "early line 0" not in message
    assert len(message.splitlines()) == 1 + 1 + owned.SERVER_LOG_TAIL_LINES


def test_failure_without_a_server_log_still_reports_the_reason(tmp_path, monkeypatch):
    value, _, _, _ = launch_mock(tmp_path, monkeypatch)
    (value.directory / "ownership.json").write_text(
        json.dumps(
            {
                "status": "failed",
                "error": "RuntimeError: before the server started",
                "server": {"owned_processes_gone": True, "resource_scope_gone": True},
            }
        )
    )
    with pytest.raises(RuntimeError) as failure:
        value.stop()
    assert "before the server started" in str(failure.value)
    assert "server.log" not in str(failure.value)


def test_guardian_entry_literal_under_main_and_disconnect_port(tmp_path, monkeypatch):
    p = policy(tmp_path)
    out = tmp_path / "run"
    out.mkdir()
    connection, captured = Connection(), {}
    monkeypatch.setattr(worker, "__name__", "__main__")
    monkeypatch.setattr(worker.socket, "socket", lambda **_: connection)
    monkeypatch.setattr(worker.os, "close", lambda fd: captured.update(closed=fd))

    def guard(argv, environment, lock, deadline, **kwargs):
        captured.update(argv=argv, lock=lock, deadline=deadline)
        assert kwargs["interrupted"]() is True
        worker.save(
            out / "ownership.json",
            {"server": {"owned_processes_gone": True, "resource_scope_gone": True}},
        )
        assert kwargs["cleanup_confirmed"](out / "ownership.json")
        return 0

    monkeypatch.setattr(worker, "guard_worker", guard)
    args = argparse.Namespace(controller_fd=6, lock_fd=7, spec=str(out / "runtime.json"))
    assert worker.run_guardian(args, {"directory": str(out), "scope": [], "unit": None}, p) == 0
    assert captured["argv"][1:4] == ["-I", "-m", "inferssd.platform.worker"]
    assert captured["closed"] == 7 and connection.closed


@pytest.mark.parametrize(
    "reason", ["stop", "resource", "disconnect", "exited", "spawn", "environment"]
)
def test_worker_cleanup_receipt_precedes_lease_release(tmp_path, monkeypatch, reason):
    p = policy(tmp_path)
    out = tmp_path / "run"
    out.mkdir()
    events = []

    class Lease:
        fd = 88

        def __init__(self, *a, **kw):
            events.append("lease")

        def check(self):
            if reason == "disconnect":
                raise InterruptedError("controller lost")

        def close(self):
            assert worker.cleanup_confirmed(out / "ownership.json")
            events.append("release")

    class Resources:
        baseline, scope = None, None

        def __init__(self, _):
            pass

        def prepare(self):
            if reason == "resource":
                raise RuntimeError("below reserve")
            self.baseline = {}
            return {}

        def sample(self, pid):
            if pid is not None:
                (out / "STOP").touch()
                return {"soft": True}
            return {"after": True}

    child = SimpleNamespace(pid=123, returncode=None)
    child.poll = lambda: 0 if reason == "exited" else child.returncode

    def spawn(*a, **kw):
        events.append("spawn")
        assert kw["pass_fds"] == (88,)
        assert kw["env"]["GGML_TEST"] == "yes"
        assert "INFERSSD_INTERNAL_SERVER_ENV" not in kw["env"]
        if reason == "spawn":
            raise OSError("spawn failed")
        return child

    def stop(c, seconds):
        events.append("clean")
        c.returncode = -15
        return {"group_gone": True}

    monkeypatch.setattr(worker, "Lease", Lease)
    monkeypatch.setattr(worker, "Resources", Resources)
    monkeypatch.setattr(worker.subprocess, "Popen", spawn)
    monkeypatch.setattr(worker, "stop_owned", stop)
    monkeypatch.setattr(worker.signal, "signal", lambda *a: None)
    monkeypatch.setattr(worker.time, "sleep", lambda _: None)
    monkeypatch.setenv(
        "INFERSSD_INTERNAL_SERVER_ENV", "[]" if reason == "environment" else '{"GGML_TEST":"yes"}'
    )
    args = argparse.Namespace(socket="unused", lock_dev=1, lock_ino=2)
    result = worker.run_worker(args, {"directory": str(out), "argv": ["/fixture/server"]}, p)
    assert result == (0 if reason == "stop" else 1)
    assert events[-1] == "release"
    if reason not in ("resource", "spawn", "environment"):
        assert events.index("clean") < events.index("release")
    receipt = json.loads((out / "ownership.json").read_text())
    assert receipt["server"] == {"owned_processes_gone": True, "resource_scope_gone": True}


def test_worker_stop_before_spawn_proves_empty_without_signalling(tmp_path, monkeypatch):
    out = tmp_path / "run"
    out.mkdir()
    (out / "STOP").touch()
    events = []

    class Lease:
        fd = 88

        def __init__(self, *a, **kw):
            pass

        def close(self):
            events.append("release")

    class Resources:
        scope, baseline = None, None

        def __init__(self, _):
            pass

    # A pre-spawn STOP proves there was no child: no group signalling is allowed.
    monkeypatch.setattr(worker, "Lease", Lease)
    monkeypatch.setattr(worker, "Resources", Resources)
    monkeypatch.setattr(worker, "stop_owned", lambda *_: pytest.fail("no owned child"))
    monkeypatch.setattr(worker.signal, "signal", lambda *a: None)
    monkeypatch.setenv("INFERSSD_INTERNAL_SERVER_ENV", "{}")
    assert (
        worker.run_worker(
            argparse.Namespace(socket="x", lock_dev=1, lock_ino=2),
            {"directory": str(out), "argv": ["unused"]},
            policy(tmp_path),
        )
        == 1
    )
    assert events == ["release"]


def test_guardian_nonzero_is_failure_even_if_cleanup_and_worker_receipt_pass(tmp_path, monkeypatch):
    value, guardian, _, _ = launch_mock(tmp_path, monkeypatch)
    guardian.wait = lambda **_: 7
    (value.directory / "ownership.json").write_text(
        json.dumps(
            {
                "status": "passed",
                "server": {"owned_processes_gone": True, "resource_scope_gone": True},
            }
        )
    )
    with pytest.raises(RuntimeError, match="execution failed"):
        value.stop()
    assert value._lock is None


def test_worker_waits_for_group_proof_before_releasing(tmp_path, monkeypatch):
    out = tmp_path / "run"
    out.mkdir()
    events = []

    class Lease:
        fd = 88

        def __init__(self, *a, **kw):
            pass

        def close(self):
            events.append("release")

    class Resources:
        scope, baseline = None, None

        def __init__(self, _):
            pass

        def prepare(self):
            return {}

    child = SimpleNamespace(pid=123, returncode=1, poll=lambda: 1)

    def stop(*_):
        events.append("check-group")
        return {"group_gone": events.count("check-group") == 2}

    def sleep(_):
        assert "release" not in events
        assert json.loads((out / "worker-quarantine.json").read_text())["retaining_lock"] is True

    monkeypatch.setattr(worker, "Lease", Lease)
    monkeypatch.setattr(worker, "Resources", Resources)
    monkeypatch.setattr(worker.subprocess, "Popen", lambda *a, **kw: child)
    monkeypatch.setattr(worker, "stop_owned", stop)
    monkeypatch.setattr(worker.signal, "signal", lambda *a: None)
    monkeypatch.setattr(worker.time, "sleep", sleep)
    monkeypatch.setenv("INFERSSD_INTERNAL_SERVER_ENV", "{}")
    assert (
        worker.run_worker(
            argparse.Namespace(socket="x", lock_dev=1, lock_ino=2),
            {"directory": str(out), "argv": ["fixture"]},
            policy(tmp_path),
        )
        == 1
    )
    assert events == ["check-group", "check-group", "release"]


def test_preexisting_linux_unit_never_stopped_or_spawned(tmp_path, monkeypatch):
    connection = Connection()
    monkeypatch.setattr(worker.socket, "socket", lambda **_: connection)
    monkeypatch.setattr(worker, "scope_state", lambda *_: {"fields": {"LoadState": "loaded"}})
    monkeypatch.setattr(
        worker, "systemctl", lambda *_: pytest.fail("foreign unit must not be stopped")
    )
    monkeypatch.setattr(worker, "guard_worker", lambda *_a, **_k: pytest.fail("no spawn"))
    with pytest.raises(RuntimeError, match="already present or unknown"):
        worker.run_guardian(
            argparse.Namespace(controller_fd=6, lock_fd=7, spec="fixture"),
            {
                "directory": str(tmp_path),
                "scope": ["systemd-run", "--scope"],
                "unit": "already-present.scope",
            },
            policy(tmp_path, "linux"),
        )


def test_command_owner_borrows_the_same_batch_lock(tmp_path, monkeypatch):
    claim = reserve(str(tmp_path / "run"))
    p = policy(tmp_path)
    fd = os.open(p.lock_path, os.O_CREAT | os.O_RDWR, 0o600)
    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    monkeypatch.setattr(owned.socket, "socketpair", lambda: (Connection(101), Connection(102)))
    monkeypatch.setattr(owned.subprocess, "Popen", lambda *a, **kw: Guardian())
    value = owned.OwnedProcess(
        resolution(),
        str(claim.directory),
        p,
        reservation=claim,
        inherited_lock_fd=fd,
        command_mode=True,
    )
    try:
        assert json.loads((claim.directory / "runtime.json").read_text())["command_mode"] is True
        (claim.directory / "ownership.json").write_text(
            json.dumps(
                {
                    "status": "passed",
                    "server": {"owned_processes_gone": True, "resource_scope_gone": True},
                }
            )
        )
        value.stop()
        with open(p.lock_path, "r+") as other:
            with pytest.raises(BlockingIOError):
                fcntl.flock(other, fcntl.LOCK_EX | fcntl.LOCK_NB)
    finally:
        os.close(fd)


@pytest.mark.parametrize(
    "returncode,orphan,force,expected",
    [(0, False, False, 0), (7, False, False, 1), (0, True, False, 1), (0, False, True, 1)],
)
def test_command_mode_checks_natural_exit_and_descendants(
    tmp_path, monkeypatch, returncode, orphan, force, expected
):
    out = tmp_path / "run"
    out.mkdir()

    class Lease:
        fd = 88

        def __init__(self, *a, **kw):
            pass

        def close(self):
            pass

    class Resources:
        baseline, scope = None, None

        def __init__(self, *_):
            pass

        def prepare(self):
            return {}

    child = SimpleNamespace(pid=123, returncode=returncode, poll=lambda: returncode)
    monkeypatch.setattr(worker, "Lease", Lease)
    monkeypatch.setattr(worker, "Resources", Resources)
    monkeypatch.setattr(worker.subprocess, "Popen", lambda *a, **kw: child)
    monkeypatch.setattr(
        worker,
        "stop_owned",
        lambda *a: {"group_gone": True, "forced_kill": force, "orphaned_descendants": orphan},
    )
    monkeypatch.setenv("INFERSSD_INTERNAL_SERVER_ENV", "{}")
    result = worker.run_worker(
        argparse.Namespace(socket="unused", lock_dev=1, lock_ino=2),
        {"directory": str(out), "argv": ["fixture"], "command_mode": True},
        policy(tmp_path),
    )
    assert result == expected


def test_inherited_batch_fd_must_match_existing_lock_inode(tmp_path):
    claim = reserve(str(tmp_path / "run"))
    p = policy(tmp_path)
    with open(p.lock_path, "w"), open(tmp_path / "wrong", "w+") as wrong:
        with pytest.raises(ValueError, match="inode"):
            owned.OwnedProcess(
                resolution(),
                str(claim.directory),
                p,
                reservation=claim,
                inherited_lock_fd=wrong.fileno(),
                command_mode=True,
            )
    assert not (claim.directory / "runtime.json").exists()
