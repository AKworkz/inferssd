#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Publish a prebuilt platform package and point its package list at it.

Usage: APP_VERSIONING_TOKEN=... scripts/publish-package.py DIST_DIR

DIST_DIR holds one versioned macOS tar.gz or Windows ZIP (Vulkan or CUDA) and its .json. The archive goes to
cdn.wici.ai/apps/inferssd/ through WiCi AI's release service, which hands out an upload URL for it,
as the desktop app's releases do. Then the package list, the .json with the archive's URL added,
replaces latest-macos.json, latest-windows-vulkan.json or latest-windows-cuda.json, which the
installers read. The list never moves to an older build.

The CDN keeps a file for a year, so a name never gets other bytes: a package built again from a
commit already uploaded is refused. Re-run the failed publish job of the run that built it
instead; it uploads the same bytes, which the service skips. See docs/macos-package.md.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

SERVICE = os.environ.get("INFERSSD_PUBLISH_SERVICE", "https://appapi.wici.ai/versioning/inferssd")
# Each package by the suffix of its name: its platform, package list and archive extension.
LISTS = {"macos-arm64": ("darwin-arm64", "latest-macos.json", ".tar.gz"),
         "windows-x64-vulkan": ("windows-x64", "latest-windows-vulkan.json", ".zip"),
         "windows-x64-cuda": ("windows-x64", "latest-windows-cuda.json", ".zip")}
# The CDN in front of the service refuses Python's default User-Agent.
AGENT = "inferssd-publish"


def call(
    method: str,
    url: str,
    body: bytes | None = None,
    headers: dict[str, str] | None = None,
    token: str | None = None,
) -> tuple[int, bytes]:
    request = urllib.request.Request(
        url, data=body, method=method, headers={"User-Agent": AGENT, **(headers or {})}
    )
    if token:
        request.add_header("Authorization", "Bearer " + token)
    try:
        with urllib.request.urlopen(request, timeout=600) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as error:
        return error.code, error.read()


def fail(message: str) -> None:
    sys.exit("publish-package: " + message)


def main() -> None:
    if len(sys.argv) != 2:
        fail("usage: publish-package.py DIST_DIR")
    token = os.environ.get("APP_VERSIONING_TOKEN")
    if not token:
        fail("APP_VERSIONING_TOKEN is not set")
    folder = Path(sys.argv[1])
    found = [(suffix, path) for suffix in LISTS for path in sorted(folder.glob(f"inferssd-*-{suffix}.json"))]
    if len(found) != 1:
        fail(f"expected one package in {sys.argv[1]}, found {len(found)}")
    suffix, manifest = found[0]
    package = json.loads(manifest.read_text(encoding="utf-8"))
    platform, listing_name, extension = LISTS[suffix]
    if package.get("platform") != platform:
        fail("the package platform does not match its name")
    name = package.get("name")
    if (not isinstance(name, str) or not name or "/" in name or "\\" in name
            or package.get("file") != name + extension):
        fail("the package filename does not match its platform and name")
    archive = manifest.with_name(package["file"])
    data = archive.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    matches = digest == package["sha256"] and len(data) == package["size"]
    if package["schema"] != "inferssd-package 1" or not matches:
        fail(f"{archive.name} does not match {manifest.name}")

    status, body = call("GET", f"{SERVICE}/config/{listing_name}")
    if status == 200:
        current = json.loads(body)
        if current.get("build", 0) > package["build"]:
            fail(
                f"{listing_name} already names build {current['build']} ({current.get('label')}), "
                f"later than {package['build']}"
            )
    elif status != 404:
        fail(f"reading {listing_name} answered {status}")

    name = archive.name
    status, body = call(
        "POST",
        f"{SERVICE}/upload",
        json.dumps({"name": name, "length": len(data), "sha256": digest}).encode(),
        {"Content-Type": "application/json"},
        token,
    )
    if status != 200:
        fail(f"the upload request answered {status}: {body[:300]!r}")
    upload = json.loads(body)
    if upload["status"] == "ready":
        if upload.get("replacing"):
            fail(
                f"{name} is already published with other bytes, and the CDN may keep that copy "
                "for a year: re-run the publish job of the run that built it, "
                "or package a newer commit"
            )
        status, body = call("PUT", upload["upload_url"], data, upload["headers"])
        if status != 200:
            fail(f"the upload answered {status}: {body[:300]!r}")
        print(f"Uploaded {name} ({len(data)} bytes)")
    elif upload["status"] == "exists":
        print(f"{name} is already published")
    else:
        fail(f"unexpected upload answer {upload!r}")

    # The CDN must serve exactly these bytes before the list names them.
    status, served = call("GET", upload["url"])
    if status != 200 or hashlib.sha256(served).hexdigest() != digest:
        fail(f"{upload['url']} does not serve the package (status {status})")
    listing = json.dumps({**package, "url": upload["url"]}, indent=2).encode() + b"\n"
    status, body = call(
        "PUT", f"{SERVICE}/config/{listing_name}", listing, {"Content-Type": "application/json"}, token
    )
    if status != 200:
        fail(f"writing {listing_name} answered {status}: {body[:300]!r}")
    stored = json.loads(body)
    print(f"{stored['url']} now names {upload['url']}")
    print(f"{package['label']}, sha256 {digest}")


if __name__ == "__main__":
    main()
