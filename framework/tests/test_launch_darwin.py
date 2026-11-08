"""The new Mac path replays the original effects, including partial publication."""

import copy
import json
import re
import sys
from pathlib import Path

import pytest
from launch_test_paths import canonical_roots
from observe_launch_candidate import mac_candidate
from observe_launch_reference import mac_case

GOLDENS = json.loads((Path(__file__).parent / "fixtures/launch-mac-goldens.json").read_text())
pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="the macOS launch path")


def original_cache_namespace(value):
    """Project only the renamed alias-cache path components onto the frozen reference."""
    if isinstance(value, str):
        return re.sub(
            r"(^|/)cache/inferssd/mac-alias(?=/|['\"]|$)",
            r"\1cache/ssdash/mac-alias",
            value,
        )
    if isinstance(value, list):
        return [original_cache_namespace(item) for item in value]
    if isinstance(value, dict):
        projected = {}
        for key, item in value.items():
            key = original_cache_namespace(key)
            assert key not in projected, "cache namespace projection would hide duplicate artifacts"
            projected[key] = original_cache_namespace(item)
        return projected
    return value


def original_helper_spelling(observation):
    """Normalize the explicit entrypoint and cache-namespace renames only.

    Keep reference files and observations immutable. The actual new argv and
    isolated installed-wheel execution are checked independently by alias tests.
    Every argument following the helper entrypoint, all effects, errors and
    partial-publication artifacts still compare against the original records.
    """
    result = copy.deepcopy(observation)
    runs = [event for event in result["events"] if event[0] == "run"]
    assert len(runs) <= 1
    for event in runs:
        command = event[1]
        assert command[:5] == ["@python", "-I", "-m", "inferssd.source.gguf_alias", "--repo"]
        assert canonical_roots(command[5]) == "@repo"
        original = [
            "@python",
            command[5] + "/scripts/kimi-build-gguf-expert-alias-tsv.py",
            *command[6:],
        ]
        # CalledProcessError displays the actual command on generation failure;
        # its status and surrounding error text retain the original contract.
        result["stderr"] = result["stderr"].replace(repr(command), repr(original))
        event[1] = original
    return original_cache_namespace(result)


@pytest.mark.parametrize("expected", GOLDENS, ids=lambda x: x["case"]["name"])
def test_mac_frozen(expected):
    actual = mac_candidate(expected["case"])
    normalized = original_helper_spelling(actual)
    assert canonical_roots(normalized) == canonical_roots(expected)
    assert normalized == mac_case(expected["case"])
