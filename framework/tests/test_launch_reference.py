"""Independent frozen source/golden contracts; no new resolver is imported."""

import hashlib
import json
import sys
from pathlib import Path

import pytest
from launch_cases import LINUX_CASES, MAC_CASES
from launch_test_paths import canonical_roots
from observe_launch_reference import mac_case

FIXTURES = Path(__file__).parent / "fixtures"


def test_launch_oracle_source_bytes():
    manifest = json.loads((FIXTURES / "oracle/launch-manifest.json").read_text())
    for name, expected in manifest["files"].items():
        assert (
            hashlib.sha256((FIXTURES / "oracle/launch" / name).read_bytes()).hexdigest() == expected
        )


def test_linux_expected_cases_and_partial_outputs():
    observations = json.loads((FIXTURES / "launch-linux-goldens.json").read_text())
    assert [x["case"] for x in observations] == LINUX_CASES
    by_name = {x["case"]["name"]: x for x in observations}
    assert by_name["early-resolve"]["runs"][0]["calls"] == []
    assert by_name["bad-threads"]["runs"][0]["code"] == 2
    assert by_name["bad-threads"]["runs"][0]["stdout"].startswith("[ssd_setting]")
    fresh = by_name["cache-freshness"]["runs"]
    assert ["indexing the experts" in r["stderr"] for r in fresh] == [True, False, True]
    assert fresh[0]["artifacts"] == fresh[1]["artifacts"] == fresh[2]["artifacts"]


@pytest.mark.skipif(sys.platform == "win32", reason="the macOS launcher")
@pytest.mark.parametrize("case", MAC_CASES, ids=lambda case: case["name"])
def test_original_mac_launcher_observation(case):
    golden = json.loads((FIXTURES / "launch-mac-goldens.json").read_text())
    expected = next(x for x in golden if x["case"]["name"] == case["name"])
    assert canonical_roots(mac_case(case)) == canonical_roots(expected)
