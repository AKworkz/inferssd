"""Install synthetic Windows packages without Python, Git or compilers on PATH."""

import hashlib
import http.server
import json
import os
import shutil
import subprocess
import sys
import threading
import zipfile
from pathlib import Path

import pytest

INSTALLER = Path(__file__).resolve().parents[2] / "scripts/install.ps1"
pytestmark = pytest.mark.skipif(sys.platform != "win32", reason="Windows package installer")


@pytest.fixture
def windows_install(tmp_path):
    powershell = shutil.which("powershell.exe") or shutil.which("pwsh")
    compiler = Path(os.environ["WINDIR"]) / "Microsoft.NET/Framework64/v4.0.30319/csc.exe"
    if not powershell or not compiler.is_file():
        pytest.skip("PowerShell and the Windows .NET Framework test compiler are required")
    source, command = tmp_path / "command.cs", tmp_path / "command.exe"
    source.write_text(
        "using System; using System.IO; class Command { static int Main(string[] args) {"
        'if (Environment.GetEnvironmentVariable("FAIL_PACKAGE") == "1") return 9;'
        'if (args.Length > 0 && args[0] == "--ssd-schema" && '
        'Environment.GetEnvironmentVariable("FAIL_NATIVE") == "1") return 8;'
        'if (args.Length > 0 && args[0] == "--version") '
        "Console.WriteLine(File.ReadAllText(Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "
        '"..", "engine-resources.json"))); return 0; } }'
    )
    subprocess.run([str(compiler), "/nologo", "/out:" + str(command), str(source)], check=True)
    home, bindir = tmp_path / "install with spaces", tmp_path / "bin with spaces"
    env = dict(os.environ, INFERSSD_HOME=str(home), INFERSSD_BIN_DIR=str(bindir))
    for key in ("INFERSSD_REF", "INFERSSD_FROM_SOURCE", "INFERSSD_RESOURCES"):
        env.pop(key, None)
    # The installation itself cannot find developer tools or an external Python.
    env["PATH"] = str(Path(os.environ["WINDIR"]) / "System32")

    def make_package(build, *, member=None, checksum=None, backend=None, driver="576.57"):
        suffix = "windows-x64-cuda" if backend == "cuda" else "windows-x64"
        name = f"inferssd-261109.{build}.aaaaaaa-{suffix}"
        archive = tmp_path / (name + ".zip")
        metadata = {
            "version": "0.1.3",
            "build": build,
            "revision": "a" * 40,
            "date": "2026-11-09",
            "branch": "main",
            "label": f"261109.{build}.aaaaaaa",
        }
        with zipfile.ZipFile(archive, "w") as zipped:
            zipped.writestr(name + "/engine-resources.json", json.dumps(metadata))
            zipped.write(command, name + "/bin/inferssd.exe")
            zipped.write(command, name + "/bin/inferssd-server.exe")
            zipped.write(command, name + "/python/python.exe")
            if member:
                zipped.writestr(member, "escape")
        manifest = {
            "schema": "inferssd-package 1",
            "platform": "windows-x64",
            "name": name,
            "file": archive.name,
            "url": archive.as_uri(),
            "minimum_windows": "10.0.22000",
            "sha256": checksum or hashlib.sha256(archive.read_bytes()).hexdigest(),
            "size": archive.stat().st_size,
            **metadata,
        }
        if backend:
            manifest["backend"] = backend
        if backend == "cuda" and driver:
            manifest["minimum_nvidia_driver"] = driver
        listing = tmp_path / (name + ".json")
        listing.write_text(json.dumps(manifest))
        return listing, name

    def run(listing, **extra):
        return subprocess.run(
            [
                powershell,
                "-NoProfile",
                "-NonInteractive",
                "-ExecutionPolicy",
                "Bypass",
                "-File",
                str(INSTALLER),
            ],
            env=dict(
                env,
                INFERSSD_PACKAGES=listing.as_uri() if isinstance(listing, Path) else listing,
                **extra,
            ),
            text=True,
            capture_output=True,
            timeout=60,
        )

    return home, bindir, env, make_package, run


def test_install_and_upgrade_prebuilt_without_developer_tools(windows_install):
    home, bindir, env, make, run = windows_install
    first, name = make(401)
    result = run(first)
    assert result.returncode == 0, result.stderr
    assert (home / "current/bin/inferssd.exe").is_file()
    assert (home / "current").resolve().parent == home / "releases"
    assert not (home / "current" / name).exists()
    assert "Downloading InferSSD" in result.stdout and "Unpacking InferSSD" in result.stdout
    assert "Building InferSSD" not in result.stdout
    assert (bindir / "inferssd.cmd").is_file()
    old = (home / "current").resolve()
    models = home / "models/user-model.gguf"
    models.parent.mkdir()
    models.write_bytes(b"model")
    second, name = make(402)
    result = run(second)
    assert result.returncode == 0, result.stderr
    assert (home / "current").resolve() != old and old.is_dir()
    assert models.read_bytes() == b"model"
    output = subprocess.check_output(
        [str(home / "current/bin/inferssd.exe"), "--version", "--json"], env=env, text=True
    )
    assert json.loads(output)["build"] == 402


@pytest.mark.parametrize(
    "failure", ["checksum", "traversal", "inner-traversal", "command", "native"]
)
def test_failed_package_keeps_the_working_installation(windows_install, failure):
    home, bindir, _, make, run = windows_install
    listing, _ = make(401)
    assert run(listing).returncode == 0
    old, wrapper = (home / "current").resolve(), (bindir / "inferssd.cmd").read_bytes()
    member = {
        "traversal": "../outside.txt",
        "inner-traversal": "inferssd-261109.402.aaaaaaa-windows-x64/../outside.txt",
    }.get(failure)
    bad, _ = make(
        402,
        member=member,
        checksum="0" * 64 if failure == "checksum" else None,
    )
    flags = {"command": {"FAIL_PACKAGE": "1"}, "native": {"FAIL_NATIVE": "1"}}
    result = run(bad, **flags.get(failure, {}))
    assert result.returncode != 0
    assert (home / "current").resolve() == old
    assert (bindir / "inferssd.cmd").read_bytes() == wrapper
    assert not (home / "releases/outside.txt").exists()
    assert len(list((home / "releases").iterdir())) == 1


def test_default_prebuilt_path_has_no_implicit_source_fallback(windows_install):
    home, _, _, _, run = windows_install
    missing = home.parent / "missing.json"
    result = run(missing)
    assert result.returncode != 0
    assert "No source build was started" in result.stderr
    assert not (home / "current").exists()


def test_downloads_manifest_and_archive_over_http(windows_install):
    home, _, _, make, run = windows_install
    listing, _ = make(401)

    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(listing.parent), **kwargs)

        def log_message(self, *_):
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        base = f"http://127.0.0.1:{server.server_port}/"
        manifest = json.loads(listing.read_text())
        manifest["url"] = base + manifest["file"]
        listing.write_text(json.dumps(manifest))
        result = run(base + listing.name)
        assert result.returncode == 0, result.stderr
        assert (home / "current/bin/inferssd.exe").is_file()
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_ps5_installs_deep_python_paths_without_duplicate_archive_root(
    windows_install, tmp_path_factory
):
    _, _, env, make, run = windows_install
    home = tmp_path_factory.mktemp("deep") / "user data"
    env["INFERSSD_HOME"] = str(home)
    listing, name = make(401)
    relative = (
        "python/Lib/site-packages/huggingface_hub/inference/_providers/__pycache__/"
        "featherless_ai.cpython-312.pyc"
    )
    # Production standalone packages have entries this long. The old installer
    # duplicated the named archive folder below a 32-digit staging GUID.
    padding = max(0, 166 - len(name + "/" + relative))
    relative = relative.replace("featherless_ai", "featherless_ai" + "x" * padding)
    archive = Path(listing.with_suffix(".zip"))
    with zipfile.ZipFile(archive, "a") as zipped:
        zipped.writestr(name + "/" + relative, b"python bytecode fixture")
    manifest = json.loads(listing.read_text())
    manifest.update(
        size=archive.stat().st_size, sha256=hashlib.sha256(archive.read_bytes()).hexdigest()
    )
    listing.write_text(json.dumps(manifest))
    result = run(listing)
    assert result.returncode == 0, result.stderr
    installed = home / "current" / relative
    assert installed.read_bytes() == b"python bytecode fixture"
    assert not (home / "current" / name).exists()
    assert len(str((home / "current").resolve() / relative)) < 260


def test_installs_the_cuda_package_a_package_list_names(windows_install):
    home, _, _, make_package, run = windows_install
    listing, name = make_package(403, backend="cuda")
    result = run(listing)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "261109.403.aaaaaaa, cuda" in result.stdout
    assert (home / "current" / "bin" / "inferssd-server.exe").is_file()


@pytest.mark.parametrize("driver", [None, "newest"])
def test_a_cuda_package_without_its_driver_minimum_is_refused(windows_install, driver):
    home, _, _, make_package, run = windows_install
    listing, _ = make_package(404, backend="cuda", driver=driver)
    result = run(listing)
    assert result.returncode != 0
    assert "invalid Windows package metadata" in result.stdout + result.stderr
    assert not (home / "current").exists()
