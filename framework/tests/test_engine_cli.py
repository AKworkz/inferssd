"""Typed launch entry checks; substitute Runtime before any native effect."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import asdict

import pytest
from test_engine_runtime import policy


@pytest.mark.parametrize("failure", [False, True])
def test_command_reports_final_owner_failure_in_subprocess(tmp_path, failure):
    contract = tmp_path / "policy.json"
    contract.write_text(json.dumps(asdict(policy(tmp_path))))
    # This child imports the package selected by this interpreter. Final release
    # validation runs it in a normal wheel environment without PYTHONPATH.
    script = r"""
import sys
from inferssd.engine import command
from inferssd.launch import typed_command
class Engine:
    ok = True
    def alive(self): return False
    def stop(self):
        if sys.argv[1] == "failure":
            raise RuntimeError("owned resource threshold failed after confirmed cleanup")
class Runtime:
    def __init__(self, *args, **kwargs): pass
    def engine(self, *args): return Engine()
command.Runtime = Runtime
arguments = ["--execute", "--platform", "darwin", "--repo", sys.argv[3],
             "--runtime-policy", sys.argv[2], "--output-dir", sys.argv[3] + "/run",
             "--port", "8080", "--model", sys.argv[3] + "/fixture.gguf"]
raise SystemExit(typed_command.main(arguments))
"""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            "failure" if failure else "normal",
            str(contract),
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
        timeout=10,
        env=dict(os.environ),
    )
    assert result.returncode == (2 if failure else 0), result.stderr
    assert ("owned resource threshold failed" in result.stderr) is failure
    assert not (tmp_path / "run").exists()


def test_installed_worker_module_help_is_stable_and_effect_free():
    result = subprocess.run(
        [sys.executable, "-m", "inferssd.platform.worker", "--help"],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr
    assert "--guardian" in result.stdout and "--worker" in result.stdout


def test_execute_missing_policy_fails_before_host_or_files(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from inferssd.launch import typed_command; raise SystemExit(typed_command.main())",
            "--execute",
            "--platform",
            "linux",
            "--repo",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 2, result.stderr
    assert "--runtime-policy" in result.stderr and "required" in result.stderr
    assert list(tmp_path.iterdir()) == []


def test_installed_worker_ignores_shadow_package_in_cwd(tmp_path):
    shadow = tmp_path / "inferssd"
    shadow.mkdir()
    (shadow / "__init__.py").write_text("raise RuntimeError('shadow-package-was-imported')\n")
    unisolated = subprocess.run(
        [sys.executable, "-m", "inferssd.platform.worker", "--help"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert unisolated.returncode != 0
    assert "shadow-package-was-imported" in unisolated.stderr
    isolated = subprocess.run(
        [sys.executable, "-I", "-m", "inferssd.platform.worker", "--help"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert isolated.returncode == 0, isolated.stderr
    assert "--guardian" in isolated.stdout and "--worker" in isolated.stdout
