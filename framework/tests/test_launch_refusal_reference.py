"""Reference-only supplement for the native refusal branch omitted by dry-run extraction."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest
from observe_launch_refusal_reference import CASES, ORACLE, definitions, observe

FROZEN = json.loads(
    (Path(__file__).parent / "fixtures/launch-native-refusal-reference.json").read_text(
        encoding="utf-8"
    )
)


@pytest.mark.skipif(sys.platform == "win32", reason="the Linux launcher's bash refusal")
@pytest.mark.parametrize("case", CASES, ids=lambda c: c["name"])
def test_original_refusal(case):
    assert observe(case) == next(x for x in FROZEN["cases"] if x["name"] == case["name"])


def test_original_refusal_source():
    assert hashlib.sha256(ORACLE.read_bytes()).hexdigest() == FROZEN["source_sha256"]
    assert definitions() == FROZEN["definitions"]
