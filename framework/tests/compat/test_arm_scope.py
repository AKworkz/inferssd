"""Process ownership checks must precede any drive or model work."""
import os
from pathlib import Path
import subprocess
import sys

import pytest

ARM = Path(__file__).resolve().parents[1] / "fixtures/adaptive/arm.sh"
pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="arm.sh: bash, ss and systemd scopes")


def test_busy_port_never_launches_or_signals(tmp_path):
    stub = tmp_path / "bin"
    stub.mkdir()
    (stub / "ss").write_text("#!/bin/sh\necho 'LISTEN 0 128 *:18250 *:*'\n")
    (stub / "systemctl").write_text("#!/bin/sh\necho unexpected >> \"$CALL_LOG\"\n")
    for path in stub.iterdir():
        path.chmod(0o755)
    log = tmp_path / "calls"
    result = subprocess.run(["bash", str(ARM)], env={**os.environ,
        "PATH": f"{stub}:{os.environ['PATH']}", "PORT": "18250", "CALL_LOG": str(log)},
        capture_output=True, text=True)
    assert result.returncode == 2
    assert "already in use" in result.stderr
    assert not log.exists()


def test_invalid_port_never_probes_or_signals(tmp_path):
    stub = tmp_path / "bin"
    stub.mkdir()
    for name in ("ss", "systemctl"):
        path = stub / name
        path.write_text("#!/bin/sh\necho unexpected >> \"$CALL_LOG\"\n")
        path.chmod(0o755)
    log = tmp_path / "calls"
    result = subprocess.run(["bash", str(ARM)], env={**os.environ,
        "PATH": f"{stub}:{os.environ['PATH']}", "PORT": "18250;bad", "CALL_LOG": str(log)},
        capture_output=True, text=True)
    assert result.returncode == 2
    assert "invalid PORT" in result.stderr
    assert not log.exists()


def test_cleanup_signals_only_the_generated_scope(tmp_path):
    stub = tmp_path / "bin"
    stub.mkdir()
    (stub / "ss").write_text("#!/bin/sh\nexit 0\n")
    (stub / "systemctl").write_text(
        '#!/bin/sh\nprintf "%s\\n" "$*" >> "$CALL_LOG"\n'
        'case "$*" in *is-active*) exit 3;; esac\n')
    for path in stub.iterdir():
        path.chmod(0o755)
    log = tmp_path / "calls"
    # Load the actual lifecycle code, stopping before any model or drive operation.
    lifecycle = ARM.read_text().split("label=$1;", 1)[0]
    result = subprocess.run(["bash", "-c", lifecycle +
        '\nscope_started=1\ncleanup\ntrap - EXIT\nprintf "%s" "$scope"\n'],
        env={**os.environ, "PATH": f"{stub}:{os.environ['PATH']}",
             "PORT": "18250", "CALL_LOG": str(log)}, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    scope = result.stdout
    assert scope.startswith("inferssd-arm-") and scope.endswith(".scope")
    calls = log.read_text().splitlines()
    assert len(calls) == 3
    assert all(call.endswith(scope) for call in calls)
    assert "kill --kill-whom=all --signal=TERM" in calls[0]
    assert "is-active --quiet" in calls[1]
    assert "kill --kill-whom=all --signal=KILL" in calls[2]
