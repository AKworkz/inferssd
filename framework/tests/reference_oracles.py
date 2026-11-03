# SPDX-License-Identifier: Apache-2.0
"""Loading the frozen reference scripts in fixtures/oracle, and exact value encoding."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"
ORACLES = FIXTURES / "oracle"


def load(name):
    spec = importlib.util.spec_from_file_location(
        f"planner_{name}_reference", ORACLES / f"{name}_reference.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def exact(value):
    if isinstance(value, float):
        return {"float_hex": value.hex()}
    if isinstance(value, dict):
        return {"items": [[exact(k), exact(v)] for k, v in value.items()]}
    if isinstance(value, (list, tuple, set)):
        return {type(value).__name__: [exact(v) for v in value]}
    return value


def runtime_key():
    key = f"{sys.version_info.major}.{sys.version_info.minor}"
    if key not in ("3.9", "3.12"):
        pytest.skip("recorded reference transcripts cover Python 3.9 and 3.12 only")
    return key
