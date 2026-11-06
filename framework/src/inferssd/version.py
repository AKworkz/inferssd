"""What `inferssd --version` says: the release, and the build this engine was made from.

The build number is the branch's commit count and the day the commit's, both read from the clone
by scripts/engine-resources.py at install time; together with the commit they name the engine as
the app does, v<build>.<mmdd>.<commit>. Whether an engine is one the app works with is the app's
own call, by this build number.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from typing import Any, Optional

from inferssd import __version__
from inferssd.launch.resources import installed_resources


def label(build: Optional[int], date: Optional[str], revision: Optional[str]) -> Optional[str]:
    """v<build>.<mmdd>.<commit>, or None without a build number, a day and a commit."""
    if not build or not date or not revision or revision == "unknown" or len(date) < 10:
        return None
    return f"v{build}.{date[5:7]}{date[8:10]}.{revision[:7]}"


def report(environ: Mapping[str, str] = os.environ) -> dict[str, Any]:
    """The version and, when the installer recorded them, the build, commit, day and branch."""
    value: dict[str, Any] = {
        "version": __version__,
        "build": None,
        "revision": None,
        "date": None,
        "branch": None,
        "label": None,
    }
    try:
        data = json.loads(installed_resources(environ).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return value
    if not isinstance(data, dict):
        return value
    build = data.get("build")
    revision = data.get("revision")
    for key, kind in (("build", int), ("revision", str), ("date", str), ("branch", str)):
        item = data.get(key)
        if isinstance(item, kind) and not isinstance(item, bool) and item != "unknown":
            value[key] = item
    value["label"] = label(
        build if isinstance(build, int) else None,
        value["date"],
        revision if isinstance(revision, str) else None,
    )
    return value


def line(value: Mapping[str, Any]) -> str:
    """The line for people: `inferssd 0.1.3 (v202.1106.03af979)`, the label when there is one."""
    text = f"inferssd {value['version']}"
    return f"{text} ({value['label']})" if value.get("label") else text
