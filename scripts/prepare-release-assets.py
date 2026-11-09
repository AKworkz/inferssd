#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Prepare one numbered release from checked macOS and Windows packages of one commit."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from pathlib import Path
from typing import Any

PLATFORMS = {
    "darwin-arm64": ("macos-arm64", ".tar.gz", "latest-macos.json"),
    "windows-x64": ("windows-x64", ".zip", "latest-windows.json"),
}


def checked_package(
    directory: Path, platform: str, version: str, revision: str
) -> dict[str, Any]:
    suffix, extension, _ = PLATFORMS[platform]
    listings = sorted(directory.glob(f"inferssd-*-{suffix}.json"))
    if len(listings) != 1:
        raise ValueError(
            f"expected one checked {platform} package, found {len(listings)}"
        )
    package = json.loads(listings[0].read_text(encoding="utf-8"))
    if (
        package.get("schema") != "inferssd-package 1"
        or package.get("platform") != platform
    ):
        raise ValueError(f"wrong package schema/platform: {listings[0]}")
    if package.get("version") != version or package.get("revision") != revision:
        raise ValueError(
            f"{platform} package does not match the release version and tag commit"
        )
    name = package.get("name", "")
    if not re.fullmatch(rf"inferssd-[A-Za-z0-9_.-]+-{suffix}", name):
        raise ValueError("invalid package name")
    if package.get("file") != name + extension:
        raise ValueError("archive filename does not match package name")
    if (
        type(package.get("build")) is not int
        or package["build"] <= 0
        or not package.get("label")
    ):
        raise ValueError("package has no build identity")
    archive = directory / package["file"]
    with archive.open("rb") as stream:
        digest = (
            hashlib.file_digest(stream, "sha256").hexdigest()
            if hasattr(hashlib, "file_digest")
            else hashlib.sha256(stream.read()).hexdigest()
        )
    if (
        type(package.get("size")) is not int
        or archive.stat().st_size != package["size"]
        or digest != package.get("sha256")
    ):
        raise ValueError(
            f"{platform} archive size/SHA-256 does not match its checked manifest"
        )
    return package


def replace_once(text: str, old: str, new: str) -> str:
    if text.count(old) != 1:
        raise ValueError(f"installer pin marker is missing or ambiguous: {old}")
    return text.replace(old, new)


def prepare(
    root: Path,
    macos: Path,
    windows: Path,
    output: Path,
    tag: str,
    revision: str,
    repository: str,
) -> list[str]:
    if not re.fullmatch(r"v[0-9]+\.[0-9]+\.[0-9]+", tag) or not re.fullmatch(
        r"[a-f0-9]{40}", revision
    ):
        raise ValueError("a numbered version tag and its exact commit SHA are required")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repository):
        raise ValueError("invalid GitHub repository")
    version = re.search(
        r'^version = "([^"]+)"$',
        (root / "framework/pyproject.toml").read_text(encoding="utf-8"),
        re.MULTILINE,
    )
    if version is None or version[1] != tag[1:]:
        raise ValueError("tag does not match the framework version")
    packages = [
        checked_package(macos, "darwin-arm64", tag[1:], revision),
        checked_package(windows, "windows-x64", tag[1:], revision),
    ]
    if any(
        packages[0].get(key) != packages[1].get(key)
        for key in ("build", "label", "date")
    ):
        raise ValueError("macOS and Windows packages do not share one build identity")
    if output.exists() and any(output.iterdir()):
        raise ValueError("release output is not empty; existing assets were preserved")
    base = f"https://github.com/{repository}/releases/download/{tag}"
    shell = replace_once(
        (root / "scripts/install.sh").read_text(encoding="utf-8"),
        "DEFAULT_REF=main",
        "DEFAULT_REF=" + tag,
    )
    shell = replace_once(
        shell,
        "PACKAGES=https://appapi.wici.ai/versioning/inferssd/config/latest-macos.json",
        "PACKAGES=" + base + "/latest-macos.json",
    )
    powershell = replace_once(
        (root / "scripts/install.ps1").read_text(encoding="utf-8"),
        "$DefaultRef = 'main'",
        "$DefaultRef = '" + tag + "'",
    )
    powershell = replace_once(
        powershell,
        "$Packages = 'https://appapi.wici.ai/versioning/inferssd/config/latest-windows.json'",
        "$Packages = '" + base + "/latest-windows.json'",
    )
    output.mkdir(parents=True, exist_ok=True)
    names = []
    for directory, package in zip((macos, windows), packages):
        shutil.copy2(directory / package["file"], output / package["file"])
        latest = PLATFORMS[package["platform"]][2]
        (output / latest).write_text(
            json.dumps({**package, "url": base + "/" + package["file"]}, indent=2)
            + "\n",
            encoding="utf-8",
        )
        names.extend((package["file"], latest))
    (output / "install.sh").write_text(shell, encoding="utf-8")
    (output / "install.sh").chmod(0o755)
    (output / "install.ps1").write_text(powershell, encoding="utf-8")
    return names + ["install.sh", "install.ps1"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--macos", required=True, type=Path)
    parser.add_argument("--windows", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--repository", required=True)
    args = parser.parse_args()
    files = prepare(
        Path(__file__).resolve().parents[1],
        args.macos,
        args.windows,
        args.output,
        args.tag,
        args.revision,
        args.repository,
    )
    print(json.dumps({"tag": args.tag, "revision": args.revision, "assets": files}))


if __name__ == "__main__":
    main()
