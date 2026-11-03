"""What a run's native-events.jsonl yields when read with the native observation reader."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from inferssd.telemetry.observations import LocalObservations, NativeObservationSession


def events(turns):
    values = [{"ev": "ssd.controller.begin", "schema": 1}]
    for prompt, tokens in turns:
        values += [
            {"ev": "ssd.prefill.graph", "requested_tokens": prompt, "configured_max": 128},
            {"ev": "ssd.place.layer", "decode": False, "success": True, "compute_us": 12000},
            {
                "ev": "ssd.prefill.copy",
                "bytes": 1048576,
                "read_us": 500,
                "passes": 1,
                "tokens": prompt,
            },
        ]
        for i in range(tokens):
            for layer in range(2):
                values.append(
                    {
                        "ev": "ssd.place.layer",
                        "epoch": 1,
                        "token": prompt + i,
                        "layer": layer,
                        "decode": True,
                        "success": True,
                        "has_token": True,
                        "selected_experts": [i % 7, i % 7 + 1],
                        "vram_hit": [True, False],
                        "cache_hit": [False, True],
                        "wall_us": 1500 + i,
                        "compute_us": 500,
                        "wait_io_us": 500,
                    }
                )
    return [dict(v=1, seq=i, **value) for i, value in enumerate(values)]


def write_events(directory: Path, values):
    data = "".join(json.dumps(v) + "\n" for v in values)
    (directory / "native-events.jsonl").write_text(data)
    return data


def test_routes_profiles_and_prefill_graphs(tmp_path):
    # The second request starts below the first one's last position in the same epoch,
    # so its tokens are a new run.
    data = write_events(tmp_path, events([(128, 2), (64, 2)]))
    reader = NativeObservationSession(files=LocalObservations())
    arm = reader.read_arm_cached(str(tmp_path), 0)
    first, second = ((1, 0), 128), ((1, 1), 64)
    following = [(first[0], 129), (second[0], 65)]
    assert arm["groups"] == [[first, following[0]], [second, following[1]]]
    for key, i in ((first, 0), (following[0], 1), (second, 0), (following[1], 1)):
        route = ([i % 7, i % 7 + 1], [True, False], [False, True])
        assert arm["trace"][key] == {0: route, 1: route}
        wall = 2 * (1500 + i)
        assert arm["prof"][key] == [wall, 1000, 1000, wall - 2000, 2]
    assert arm["pf_graphs"] == [
        {
            "n_tokens": prompt,
            "ubatch": 128,
            "cpu_s": 0.012,
            "sink_bytes": 0.0,
            "graph_bytes": 1048576,
            "graph_read_us": 500,
            "graph_passes": 1,
            "graph_tokens": prompt,
        }
        for prompt in (128, 64)
    ]
    # An incomplete final line waits for the next read; the rest is kept.
    with (tmp_path / "native-events.jsonl").open("a") as stream:
        stream.write('{"v":1,"seq":')
    assert reader.read_arm_cached(str(tmp_path), 0) == arm
    (tmp_path / "native-events.jsonl").write_text(data[:20])
    with pytest.raises(ValueError, match="truncated"):
        reader.read_arm_cached(str(tmp_path), 0)


@pytest.mark.parametrize(
    "mutation", ["version", "gap", "missing-route", "failed-layer", "contract"]
)
def test_invalid_native_observations_fail_closed(tmp_path, mutation):
    values = copy.deepcopy(events([(128, 1)]))
    if mutation == "version":
        values[0]["v"] = 2
    elif mutation == "gap":
        values[-1]["seq"] += 1
    elif mutation == "missing-route":
        del values[-1]["selected_experts"]
    elif mutation == "failed-layer":
        values[-1]["success"] = False
    else:
        values[0]["schema"] = 2
    write_events(tmp_path, values)
    with pytest.raises(ValueError):
        NativeObservationSession(files=LocalObservations()).read_arm_cached(str(tmp_path), 0)
