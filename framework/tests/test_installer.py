# SPDX-License-Identifier: Apache-2.0
"""Exercise the piped installer with a synthetic toolchain, never compiling a model."""

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

INSTALLER = Path(__file__).resolve().parents[2] / "scripts/install.sh"


@pytest.fixture
def installation(tmp_path):
    tools = tmp_path / "tools"
    tools.mkdir()
    helper = tools / "helper"
    helper.write_text(
        f"#!{sys.executable}\n"
        + r"""
import json, os, pathlib, subprocess, sys
name = pathlib.Path(sys.argv[0]).name
args = sys.argv[1:]
with open(os.environ["CALL_LOG"], "a") as log:
    log.write(name + "\n")
if name == "uname":
    print(os.environ.get("TEST_SYSTEM", "Darwin") if args == ["-s"] else "arm64")
elif name == "git":
    root = pathlib.Path(args[-1])
    (root / "framework").mkdir(parents=True)
    (root / "scripts").mkdir()
elif name == "plutil":
    # plutil -extract KEY raw -o - FILE, for the package list
    value = json.loads(pathlib.Path(args[-1]).read_text())
    for key in args[1].split("."):
        if key not in value:
            sys.exit(1)
        value = value[key]
    print(value)
elif name == "sw_vers":
    print(os.environ.get("TEST_MACOS", "15.1"))
elif name == "shasum":
    import hashlib
    print(hashlib.sha256(pathlib.Path(args[-1]).read_bytes()).hexdigest() + "  " + args[-1])
elif name == "xcode-select":
    sys.exit(2 if os.environ.get("NO_COMMAND_LINE_TOOLS") else 0)
elif name == "cmake":
    with open(os.environ["CMAKE_LOG"], "a") as log:
        log.write(" ".join(args) + "\n")
    if os.environ.get("FAIL_BUILD"):
        sys.exit(1)
elif name in ("python3", "python"):
    if args[:2] == ["-m", "venv"]:
        bin = pathlib.Path(args[2]) / "bin"
        bin.mkdir(parents=True)
        (bin / "python").symlink_to(os.path.realpath(sys.argv[0]))
        app = bin / "inferssd"
        app.write_text("#!/bin/sh\nprintf '%s\\n' \"$INFERSSD_RESOURCES\"\n")
        app.chmod(0o755)
    elif args[:2] == ["-m", "pip"]:
        pass
    elif args[:2] == ["-m", "inferssd.machine.backend"]:
        assert os.environ["PYTHONPATH"].endswith("/source/framework/src")
        print(os.environ.get("TEST_GPU_BACKEND", "CUDA"))
    elif args and args[0] == "scripts/engine-resources.py":
        pathlib.Path(args[args.index("--output") + 1]).write_text("{}")
    else:
        sys.exit(subprocess.call([sys.executable, *args]))
"""
    )
    helper.chmod(0o755)
    tools_used = ("uname", "git", "cc", "c++", "xcode-select", "python3", "cmake", "nvcc")
    for tool in (*tools_used, "plutil", "sw_vers", "shasum"):
        (tools / tool).symlink_to(helper)
    home = tmp_path / "install with spaces"
    bindir = tmp_path / "bin with spaces"
    env = dict(
        os.environ,
        PATH=str(tools) + os.pathsep + os.environ["PATH"],
        INFERSSD_HOME=str(home),
        INFERSSD_BIN_DIR=str(bindir),
        CMAKE_LOG=str(tmp_path / "cmake.log"),
        CALL_LOG=str(tmp_path / "calls.log"),
    )
    return home, bindir, env


@pytest.mark.parametrize("system", ["Darwin", "Linux"])
def test_piped_install_and_upgrade(installation, system):
    home, bindir, env = installation
    env["TEST_SYSTEM"] = system
    # From main a Mac installs the prebuilt package unless asked to build.
    env["INFERSSD_FROM_SOURCE"] = "1"
    result = subprocess.run(
        ["sh"], input=INSTALLER.read_text(), text=True, env=env, capture_output=True
    )
    assert result.returncode == 0, result.stderr
    app = bindir / "inferssd"
    previous = app.read_text()
    descriptor = subprocess.check_output([str(app), "serve", "--help"], env=env, text=True).strip()
    assert Path(descriptor).is_file()
    assert home.joinpath("current/engine-resources.json").is_file()
    assert "export PATH=" in result.stdout
    log = Path(env["CMAKE_LOG"]).read_text()
    assert ("inferssd-memprobe" in log) == (system == "Linux")
    result = subprocess.run(
        ["sh"],
        input=INSTALLER.read_text(),
        text=True,
        env=dict(env, FAIL_BUILD="1"),
        capture_output=True,
    )
    assert result.returncode != 0
    assert app.read_text() == previous
    assert Path(descriptor).is_file()
    assert len(list((home / "releases").iterdir())) == 1


@pytest.mark.parametrize(
    "backend, glslc, preset, warned",
    [
        ("CPU", False, "--preset linux-cpu\n", False),
        ("Vulkan", False, "--preset linux-cpu\n", True),
        ("Vulkan", True, "--preset linux-cpu -DGGML_VULKAN=ON\n", False),
    ],
)
def test_linux_builds_for_the_machines_gpus(installation, backend, glslc, preset, warned):
    home, bindir, env = installation
    tools = Path(env["PATH"].split(os.pathsep)[0])
    # Without an NVIDIA GPU the CUDA toolkit is not needed.
    (tools / "nvcc").unlink()
    if glslc:
        (tools / "glslc").symlink_to(tools / "helper")
    env.update(TEST_SYSTEM="Linux", TEST_GPU_BACKEND=backend)
    result = subprocess.run(
        ["sh"], input=INSTALLER.read_text(), text=True, env=env, capture_output=True
    )
    assert result.returncode == 0, result.stderr
    log = Path(env["CMAKE_LOG"]).read_text()
    assert preset in log and "inferssd-memprobe" not in log
    assert ("install the Vulkan SDK" in result.stderr) == warned


def test_mac_asks_for_the_command_line_tools(installation):
    _, _, env = installation
    result = subprocess.run(
        ["sh"],
        input=INSTALLER.read_text(),
        text=True,
        env=dict(env, NO_COMMAND_LINE_TOOLS="1", INFERSSD_FROM_SOURCE="1"),
        capture_output=True,
    )
    assert result.returncode != 0
    assert "xcode-select --install" in result.stderr


def test_release_workflow_can_pin_the_installer():
    # .github/workflows/release.yml rewrites exactly this line to the tag.
    assert re.search(r"^DEFAULT_REF=main$", INSTALLER.read_text(), re.M)


def test_install_help_needs_no_toolchain():
    result = subprocess.run(["sh", str(INSTALLER), "--help"], text=True, capture_output=True)
    assert result.returncode == 0
    assert "INFERSSD_REF" in result.stdout


def package_name(label):
    """The archive's name as scripts/package-macos.sh writes it."""
    return f"inferssd-{label}-macos-arm64"


def make_package(tmp_path, label, sha256=None):
    """A package as scripts/package-macos.sh lays it out, with its list, for file:// URLs."""
    import hashlib
    import json
    import tarfile

    name = package_name(label)
    root = tmp_path / "packages" / label / name
    (root / "bin").mkdir(parents=True)
    (root / "python/bin").mkdir(parents=True)
    (root / "python/bin/python3").symlink_to(sys.executable)
    (root / "engine-resources.json").write_text("{}")
    command = root / "bin/inferssd"
    command.write_text("#!/bin/sh\nprintf '%s\\n' \"$INFERSSD_RESOURCES\"\n")
    activate = root / "activate"
    activate.write_text(
        "#!/bin/sh\nset -eu\n"
        'root=$(cd -P "$(dirname "$0")" && pwd)\n'
        'ln -sfn "$root" "$INFERSSD_HOME/current"\n'
        'printf \'#!/bin/sh\\nexport INFERSSD_RESOURCES="%s/engine-resources.json"\\n'
        'exec "%s/bin/inferssd" "$@"\\n\' "$root" "$root" >"$INFERSSD_BIN_DIR/inferssd"\n'
        'chmod 755 "$INFERSSD_BIN_DIR/inferssd"\n'
    )
    command.chmod(0o755)
    activate.chmod(0o755)
    archive = root.parent / (name + ".tar.gz")
    with tarfile.open(archive, "w:gz") as stream:
        stream.add(root, arcname=name)
    listing = root.parent / "latest.json"
    listing.write_text(
        json.dumps(
            {
                "schema": "inferssd-package 1",
                "name": name,
                "label": label,
                "platform": "darwin-arm64",
                "minimum_macos": "15.0",
                "url": archive.as_uri(),
                "size": archive.stat().st_size,
                "sha256": sha256 or hashlib.sha256(archive.read_bytes()).hexdigest(),
            }
        )
    )
    return listing.as_uri()


def install(env, **extra):
    return subprocess.run(
        ["sh"], input=INSTALLER.read_text(), text=True, env=dict(env, **extra), capture_output=True
    )


def test_mac_installs_the_package_without_developer_tools(installation, tmp_path):
    home, bindir, env = installation
    result = install(env, INFERSSD_PACKAGES=make_package(tmp_path, "261107.207.84bcc56"))
    assert result.returncode == 0, result.stderr
    assert "Downloading InferSSD 261107.207.84bcc56" in result.stdout
    assert "Building" not in result.stdout
    assert "export PATH=" in result.stdout
    current = home.joinpath("current").resolve()
    assert current.name == "inferssd-261107.207.84bcc56-macos-arm64"
    assert current.parent.parent == home.resolve() / "releases"
    descriptor = subprocess.check_output([str(bindir / "inferssd"), "serve", "--help"], text=True)
    assert Path(descriptor.strip()) == current / "engine-resources.json"
    called = set(Path(env["CALL_LOG"]).read_text().split())
    assert not called & {"git", "cc", "c++", "xcode-select", "python3", "cmake"}


def test_mac_refuses_a_package_that_does_not_match_its_list(installation, tmp_path):
    home, bindir, env = installation
    first = make_package(tmp_path, "261106.202.03af979")
    assert install(env, INFERSSD_PACKAGES=first).returncode == 0
    previous = (bindir / "inferssd").read_text()
    result = install(env, INFERSSD_PACKAGES=make_package(tmp_path, "261107.207.84bcc56", "0" * 64))
    assert result.returncode != 0
    assert "does not match the package list" in result.stderr
    assert (bindir / "inferssd").read_text() == previous
    assert len(list((home / "releases").iterdir())) == 1


def test_mac_package_needs_the_macos_it_was_built_for(installation, tmp_path):
    _, _, env = installation
    listing = make_package(tmp_path, "261107.207.84bcc56")
    result = install(env, INFERSSD_PACKAGES=listing, TEST_MACOS="14.7.1")
    assert result.returncode != 0
    assert "needs macOS 15.0 or later" in result.stderr
    assert "INFERSSD_FROM_SOURCE=1" in result.stderr


@pytest.mark.parametrize("asked", [{"INFERSSD_FROM_SOURCE": "1"}, {"INFERSSD_REF": "main"}])
def test_mac_builds_from_source_when_asked(installation, asked):
    _, _, env = installation
    result = install(env, **asked)
    assert result.returncode == 0, result.stderr
    assert "--preset macos-metal" in Path(env["CMAKE_LOG"]).read_text()


def test_an_install_keeps_the_one_it_replaced_and_removes_older(installation, tmp_path):
    _, _, env = installation
    # The installation reached through a link: the package's activate names its real path.
    (tmp_path / "real").mkdir()
    (tmp_path / "linked").symlink_to(tmp_path / "real")
    home = tmp_path / "linked" / "inferssd"
    env["INFERSSD_HOME"] = str(home)
    labels = ["261105.190.0000001", "261106.202.03af979", "261107.207.84bcc56"]
    for label in labels:
        assert install(env, INFERSSD_PACKAGES=make_package(tmp_path, label)).returncode == 0
    releases = (home / "releases").iterdir()
    kept = sorted(path.name for release in releases for path in release.iterdir())
    assert kept == [package_name(label) for label in labels[1:]]
    assert home.joinpath("current").resolve().name == package_name(labels[-1])


def test_mac_builds_from_source_until_a_package_is_published(installation, tmp_path):
    import functools
    import http.server
    import threading

    _, _, env = installation
    (tmp_path / "served").mkdir()
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=tmp_path / "served")
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        listing = f"http://127.0.0.1:{server.server_port}/latest-macos.json"
        result = install(env, INFERSSD_PACKAGES=listing)
    finally:
        server.shutdown()
    assert result.returncode == 0, result.stderr
    assert "no package is published yet" in result.stderr
    assert "--preset macos-metal" in Path(env["CMAKE_LOG"]).read_text()


def test_mac_stops_when_the_package_list_cannot_be_read(installation, tmp_path):
    home, _, env = installation
    result = install(env, INFERSSD_PACKAGES=(tmp_path / "missing.json").as_uri())
    assert result.returncode != 0
    assert "Cannot read the package list" in result.stderr
    assert not list((home / "releases").iterdir())
