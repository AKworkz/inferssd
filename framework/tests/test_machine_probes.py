# SPDX-License-Identifier: Apache-2.0
"""Probe discovery, invocation and JSON parsing with recorded outputs; nothing is executed."""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

from inferssd.machine import probes as module
from inferssd.machine.identity import Drive
from inferssd.machine.probes import (
    Completed,
    ProbeError,
    Probes,
    compute_rows,
    dram_rates,
    gpu_rows,
    knee,
    locate,
    parse,
    run_process,
    skipped,
    skipped_devices,
    storage_entry,
)


def latency(base):
    return {"mean": base, "p50": base * 0.9, "p90": base * 1.3, "p99": base * 2, "max": base * 3}


def run(rate, base=1000.0):
    return {
        "bytes_read": 268435456,
        "seconds": 268435456 / rate,
        "Bps": rate,
        "lat_us": latency(base),
    }


# The contracts of the native probes, as their outputs are specified.
SSD_REPORT = {
    "schema": "inferssd-ssdprobe/1",
    "reader": "io_uring",
    "alignment_bytes": 4096,
    "qd_cap": None,
    "files": [
        {"path": "/models/x-00001-of-00003.gguf", "size_bytes": 50000000000},
        {"path": "/models/x-00002-of-00003.gguf", "size_bytes": 50000000000},
    ],
    "points": [
        {"bytes": 1048576, "qd": 1, "runs": [run(1.0e9, 900.0), run(1.1e9, 880.0)]},
        {"bytes": 1048576, "qd": 4, "runs": [run(2.9e9, 1400.0), run(3.3e9, 1300.0)]},
        {"bytes": 1048576, "qd": 8, "runs": [run(3.2e9, 2600.0), run(3.4e9, 2500.0)]},
        {"bytes": 4194304, "qd": 1, "runs": [run(2.0e9, 2100.0), run(2.2e9, 2000.0)]},
        {"bytes": 4194304, "qd": 4, "runs": [run(5.0e9, 3300.0), run(5.2e9, 3200.0)]},
    ],
}
CPU_REPORT = {
    "schema": "inferssd-cpuprobe/2",
    "ggml_features": ["AVX2", "FMA", "F16C"],
    "dram": [
        {"threads": 1, "bytes": 1073741824, "Bps": 1.2e10},
        {"threads": 4, "bytes": 1073741824, "Bps": 3.6e10},
    ],
    "compute": [
        {
            "format": "Q4_K",
            "threads": 4,
            "rows": 4096,
            "cols": 4096,
            "weight_bytes": 9437184,
            "points": [
                {"tokens": 1, "us": 210.0, "us_q1": 205.0, "us_q3": 220.0, "weight_Bps": 4.5e10},
                {"tokens": 32, "us": 900.0, "us_q1": 890.0, "us_q3": 915.0, "flops": 1.2e12},
            ],
        }
    ],
    "skipped": [{"format": "MXFP4", "reason": "no CPU kernel"}],
}
MEMORY_REPORT = {
    "schema": "inferssd-memprobe/1",
    "devices": [
        {
            "index": 0,
            "name": "NVIDIA GeForce RTX 5090",
            "uuid": "GPU-1111",
            "pci_bus_id": "0000:01:00.0",
            "integrated": False,
            "vram_total_bytes": 34190917632,
            "vram_free_bytes": 33000000000,
            "h2d": {
                "pinned": [{"bytes": 1048576, "Bps": 2.4e10}],
                "pageable": [{"bytes": 1048576, "Bps": 1.1e10}],
            },
            "d2h": {"pinned": [{"bytes": 1048576, "Bps": 2.5e10}]},
            "d2d_Bps": 1.5e12,
            "read_Bps": 1.6e12,
        }
    ],
}
REPORTS = {"ssd": SSD_REPORT, "cpu": CPU_REPORT, "memory": MEMORY_REPORT}


class FakeRunner:
    """Replies by executable name; records every call and its environment."""

    def __init__(self, replies):
        self.replies, self.calls, self.environments = dict(replies), [], []

    def __call__(self, argv, timeout_s, env):
        self.calls.append((list(argv), timeout_s))
        self.environments.append(dict(env))
        name = argv[0].rsplit("/", 1)[-1]
        reply = self.replies[name]
        if isinstance(reply, Exception):
            raise reply
        return reply


def replies(**overrides):
    value = {
        "inferssd-" + name + "probe": Completed(0, json.dumps(report))
        for name, report in {"ssd": SSD_REPORT, "cpu": CPU_REPORT, "mem": MEMORY_REPORT}.items()
    }
    value.update(overrides)
    return value


PATHS = (
    ("ssd", "/opt/inferssd/bin/inferssd-ssdprobe"),
    ("cpu", "/opt/inferssd/bin/inferssd-cpuprobe"),
    ("memory", "/opt/inferssd/bin/inferssd-memprobe"),
)


def test_locate_from_resources_build_and_path(tmp_path):
    bin_dir = tmp_path / "build" / "bin"
    bin_dir.mkdir(parents=True)
    for name in ("inferssd-server", "inferssd-ssdprobe", "inferssd-cpuprobe", "inferssd-ioreplay"):
        (bin_dir / name).write_text("")
    descriptor = tmp_path / "engine.json"
    descriptor.write_text(
        json.dumps(
            {
                "schema": "inferssd-engine 1",
                "binary": "build/bin/inferssd-server",
                "capabilities": {"settings": "inferssd-settings 1", "telemetry": 1},
                "templates": {"muse": "build/bin/inferssd-server"},
                "probes": {
                    "io": "build/bin/inferssd-ioreplay",
                    "ssd": "build/bin/inferssd-ssdprobe",
                    "cpu": "build/bin/inferssd-cpuprobe",
                },
            }
        )
    )
    expected = {"ssd": bin_dir / "inferssd-ssdprobe", "cpu": bin_dir / "inferssd-cpuprobe"}
    assert locate(descriptor) == {name: str(path.resolve()) for name, path in expected.items()}
    assert locate(build=tmp_path / "build") == {name: str(path) for name, path in expected.items()}
    assert locate() == {}


@pytest.mark.parametrize("name", ["ssd", "cpu", "memory"])
def test_specified_reports_parse(name):
    assert parse(name, json.dumps(REPORTS[name])) == REPORTS[name]


def test_reports_may_carry_more_fields():
    report = dict(SSD_REPORT, seed=1, repeats=2)
    assert parse("ssd", json.dumps(report))["seed"] == 1


@pytest.mark.parametrize(
    "name,change,message",
    [
        ("ssd", {"schema": "inferssd-ssdprobe/2"}, "schema must be 'inferssd-ssdprobe/1'"),
        ("ssd", {"reader": "mmap"}, "reader must be one of"),
        (
            "ssd",
            {"points": [{"bytes": 4096, "qd": 1, "runs": []}]},
            r"points\[0\].runs must be an array",
        ),
        ("cpu", {"dram": [{"threads": 0, "bytes": 1, "Bps": 1.0}]}, r"dram\[0\].threads must be"),
        (
            "cpu",
            {
                "compute": [
                    dict(
                        CPU_REPORT["compute"][0],
                        points=[{"tokens": 32, "us": 1.0, "us_q1": 1.0, "us_q3": 1.0}],
                    )
                ]
            },
            "Q4_K 32 tokens lack flops",
        ),
        ("memory", {"devices": [{"index": 0}]}, r"devices\[0\].name is required"),
    ],
)
def test_report_errors_name_the_field(name, change, message):
    with pytest.raises(ProbeError, match=message):
        parse(name, json.dumps(dict(REPORTS[name], **change)))


def test_report_must_be_one_json_object():
    with pytest.raises(ProbeError, match="inferssd-cpuprobe output"):
        parse("cpu", "AVX2 yes\n")
    with pytest.raises(ProbeError, match="one JSON object"):
        parse("cpu", "[]")


def test_run_passes_arguments_and_timeout():
    runner = FakeRunner(replies())
    probes = Probes(PATHS, runner)
    assert probes.run("cpu", ["--threads", "1,4"]) == CPU_REPORT
    assert runner.calls == [(["/opt/inferssd/bin/inferssd-cpuprobe", "--threads", "1,4"], 1800.0)]


def test_memory_probe_sees_every_gpu():
    runner = FakeRunner(replies())
    probes = Probes(PATHS, runner, {"PATH": "/usr/bin", "CUDA_VISIBLE_DEVICES": "1"})
    probes.run("memory", [])
    probes.run("cpu", [])
    assert runner.environments == [
        {"PATH": "/usr/bin"},
        {"PATH": "/usr/bin", "CUDA_VISIBLE_DEVICES": "1"},
    ]


def test_memory_report_lists_skipped_devices():
    report = dict(MEMORY_REPORT, skipped=[{"index": 1, "reason": "only 300 MiB of VRAM free"}])
    assert skipped_devices(parse("memory", json.dumps(report))) == {1: "only 300 MiB of VRAM free"}
    assert skipped_devices(MEMORY_REPORT) == {}
    with pytest.raises(ProbeError, match=r"skipped\[0\].reason is required"):
        parse("memory", json.dumps(dict(MEMORY_REPORT, skipped=[{"index": 1}])))


def test_run_failures_are_probe_errors():
    with pytest.raises(ProbeError, match="inferssd-ssdprobe not found; pass --build"):
        Probes((), FakeRunner({})).run("ssd", [])
    failing = FakeRunner(
        replies(
            **{"inferssd-ssdprobe": Completed(2, "", "warming up\ncannot open /x for direct I/O\n")}
        )
    )
    with pytest.raises(ProbeError, match="exited with status 2: cannot open /x for direct I/O$"):
        Probes(PATHS, failing).run("ssd", ["--file", "/x"])
    timeout = FakeRunner(
        replies(**{"inferssd-memprobe": ProbeError("inferssd-memprobe timed out")})
    )
    with pytest.raises(ProbeError, match="timed out"):
        Probes(PATHS, timeout).run("memory", [])


def test_run_process_captures_output_and_times_out():
    env = dict(os.environ, INFERSSD_TEST_VALUE="x")
    script = "import os, sys; print('{}'); sys.stderr.write(os.environ['INFERSSD_TEST_VALUE'])"
    result = run_process([sys.executable, "-c", script], 30, env)
    assert result == Completed(0, "{}\n", "x")
    with pytest.raises(ProbeError, match="timed out after 0.5 s"):
        run_process([sys.executable, "-c", "import time; time.sleep(5)"], 0.5, os.environ)
    with pytest.raises(ProbeError, match="cannot run"):
        run_process(["/nonexistent/inferssd-cpuprobe"], 1, os.environ)


def test_run_process_replaces_bytes_that_are_not_utf8():
    script = "import sys; sys.stdout.buffer.write(b'caf\\xe9 \\xc3\\xa9')"
    result = run_process([sys.executable, "-c", script], 30, os.environ)
    assert result.stdout == "caf\ufffd \u00e9"


def test_run_process_uses_no_shell(monkeypatch):
    seen = {}

    def fake(argv, **options):
        seen.update(options, argv=argv)
        return subprocess.CompletedProcess(argv, 0, "{}", "")

    monkeypatch.setattr(module.subprocess, "run", fake)
    run_process(["probe", "--file", "a b"], 3, {"PATH": "/bin"})
    assert seen["argv"] == ["probe", "--file", "a b"] and "shell" not in seen
    assert seen["capture_output"] and seen["text"] and seen["timeout"] == 3
    assert (seen["encoding"], seen["errors"]) == ("utf-8", "replace")
    assert seen["env"] == {"PATH": "/bin", "LC_ALL": "C"}


def point(size, qd, *rates):
    return {
        "bytes": size,
        "qd": qd,
        "Bps": sorted(rates)[len(rates) // 2] if len(rates) % 2 else sum(rates) / len(rates),
        "Bps_runs": list(rates),
        "lat_us": {"mean": 1.0, "p50": 1.0, "p90": 1.0, "p99": 1.0},
    }


def test_knee_is_the_smallest_qd_whose_best_run_reaches_the_fastest_worst_run():
    points = [
        point(1 << 20, 1, 1.0, 1.1),
        point(1 << 20, 2, 2.0, 2.9),
        point(1 << 20, 4, 3.0, 3.2),
        point(1 << 20, 8, 3.1, 3.3),
        point(1 << 22, 8, 5.0, 5.0),
        point(1 << 22, 2, 4.0, 5.0),
        point(1 << 22, 1, 3.0, 3.0),
    ]
    # 1 MiB: the fastest is QD 8 (median 3.2, worst 3.1); QD 4's best run 3.2 reaches it.
    # 4 MiB: the fastest is QD 8 (worst 5.0); QD 2's best run 5.0 reaches it.
    assert knee(points) == [{"bytes": 1 << 20, "qd": 4}, {"bytes": 1 << 22, "qd": 2}]


def test_knee_tie_prefers_the_smaller_qd():
    assert knee([point(4096, 16, 2.0, 2.0), point(4096, 4, 2.0, 2.0)]) == [{"bytes": 4096, "qd": 4}]


def test_storage_entry_takes_medians_and_the_median_run_latency():
    drive = Drive("linux:nvme:Example NVMe 2TB:" + "ab" * 32, "Example NVMe 2TB", "nvme")
    entry = storage_entry(SSD_REPORT, drive)
    assert entry["drive_key"] == drive.key and entry["model"] == "Example NVMe 2TB"
    assert entry["reader"] == "io_uring" and entry["qd_cap"] is None
    assert entry["alignment_bytes"] == 4096 and entry["measured_files"] == 2
    first = entry["random_read"][1]
    assert (first["bytes"], first["qd"], first["Bps_runs"]) == (1048576, 4, [2.9e9, 3.3e9])
    assert first["Bps"] == pytest.approx(3.1e9)
    # Two runs: latency of the lower median run, without the probe's max.
    assert first["lat_us"] == pytest.approx(
        {"mean": 1400.0, "p50": 1260.0, "p90": 1820.0, "p99": 2800.0}
    )
    assert entry["knee"] == [{"bytes": 1048576, "qd": 4}, {"bytes": 4194304, "qd": 4}]


def test_cpu_and_memory_conversions():
    assert dram_rates(CPU_REPORT) == [{"threads": 1, "Bps": 1.2e10}, {"threads": 4, "Bps": 3.6e10}]
    assert compute_rows(CPU_REPORT) == CPU_REPORT["compute"]
    # Pinned, thread i ran on the i-th CPU: each row names the first ones.
    row = CPU_REPORT["compute"][0]
    pinned = dict(
        CPU_REPORT, cpus=[16, 5, 15, 6], compute=[dict(row, threads=n) for n in (1, 2, 4)]
    )
    assert [r["cpus"] for r in compute_rows(pinned)] == [[5], [5, 6], [5, 6, 15, 16]]
    assert skipped(CPU_REPORT) == ["MXFP4 (no CPU kernel)"]
    rows = gpu_rows(MEMORY_REPORT)
    expected = dict(MEMORY_REPORT["devices"][0])
    del expected["vram_free_bytes"]
    assert rows == [expected]
