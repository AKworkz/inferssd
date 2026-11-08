#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Publish a package from scripts/package-macos.sh and point the package list at it.

Usage: APP_VERSIONING_TOKEN=... scripts/publish-package.py DIST_DIR

DIST_DIR holds one inferssd-<label>-macos-arm64.tar.gz and its .json. The archive goes to
cdn.wici.ai/apps/inferssd/ through WiCi AI's release service, which hands out an upload URL for it,
as the desktop app's releases do. Then the package list, the .json with the archive's URL added,
replaces latest-macos.json, which scripts/install.sh reads. The list never moves to an older
build.

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
LIST = "latest-macos.json"
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
    listings = sorted(Path(sys.argv[1]).glob("inferssd-*-macos-arm64.json"))
    if len(listings) != 1:
        fail(f"expected one package in {sys.argv[1]}, found {len(listings)}")
    package = json.loads(listings[0].read_text(encoding="utf-8"))
    archive = listings[0].with_name(package["file"])
    data = archive.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    matches = digest == package["sha256"] and len(data) == package["size"]
    if package["schema"] != "inferssd-package 1" or not matches:
        fail(f"{archive.name} does not match {listings[0].name}")

    status, body = call("GET", f"{SERVICE}/config/{LIST}")
    if status == 200:
        current = json.loads(body)
        if current.get("build", 0) > package["build"]:
            fail(
                f"{LIST} already names build {current['build']} ({current.get('label')}), "
                f"later than {package['build']}"
            )
    elif status != 404:
        fail(f"reading {LIST} answered {status}")

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
        "PUT", f"{SERVICE}/config/{LIST}", listing, {"Content-Type": "application/json"}, token
    )
    if status != 200:
        fail(f"writing {LIST} answered {status}: {body[:300]!r}")
    stored = json.loads(body)
    print(f"{stored['url']} now names {upload['url']}")
    print(f"{package['label']}, sha256 {digest}")


if __name__ == "__main__":
    main()
