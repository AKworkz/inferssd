# SPDX-License-Identifier: Apache-2.0
"""Locate and run the native probes, and turn their JSON into characteristics sections.

Each probe prints one JSON object on stdout and a one-line reason on stderr when
it fails. Fields this version does not use are ignored, so a probe may add some.
"""

from __future__ import annotations

import json
import os
import statistics
import subprocess
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from inferssd.contracts import ContractError
from inferssd.launch.resources import PROBES, read_resources
from inferssd.machine.characteristics import (
    COUNT,
    GPU,
    NUMBER,
    POSITIVE,
    PROBE_SCHEMAS,
    READERS,
    TEXT,
    Check,
    ComputePoint,
    ComputeRow,
    Knee,
    RandomRead,
    Storage,
    ThreadRate,
    Transfer,
    array,
    constant,
    flag,
    nullable,
    one_of,
    record,
)
from inferssd.machine.identity import Drive
from inferssd.platform.captured_command import run_captured

EXECUTABLES = PROBES
TIMEOUT_S = {"ssd": 1800.0, "cpu": 1800.0, "memory": 900.0}
# The profile describes the whole machine, so a probe sees every device.
HIDDEN = {"memory": ("CUDA_VISIBLE_DEVICES",)}


class ProbeError(RuntimeError):
    """A probe is missing, failed, timed out or printed output outside its schema."""


@dataclass(frozen=True)
class Completed:
    returncode: int
    stdout: str
    stderr: str = ""


Runner = Callable[[Sequence[str], float, Mapping[str, str]], Completed]


def run_process(argv: Sequence[str], timeout_s: float, env: Mapping[str, str]) -> Completed:
    """Run without a shell in ENV, capturing both streams as UTF-8 (bad bytes replaced)."""
    name = os.path.basename(argv[0])
    try:
        result = run_captured(argv, timeout=timeout_s, env=env)
    except subprocess.TimeoutExpired as error:
        raise ProbeError(f"{name} timed out after {timeout_s:g} s") from error
    except OSError as error:
        raise ProbeError(f"cannot run {name}: {error}") from error
    return Completed(result.returncode, result.stdout, result.stderr)


def locate(
    resources: Optional[Path] = None,
    build: Optional[Path] = None,
    *,
    suffix: str = "",
    is_file: Callable[[str], bool] = os.path.isfile,
) -> dict[str, str]:
    """Probe executables by name, from a resource descriptor or a build directory."""
    if resources is not None:
        named = dict(read_resources(resources).probes)
        return {name: named[name] for name in EXECUTABLES if name in named}
    if build is None:
        return {}
    paths = {name: str(build / "bin" / (exe + suffix)) for name, exe in EXECUTABLES.items()}
    return {name: path for name, path in paths.items() if is_file(path)}


LATENCY = record(
    {"mean": NUMBER, "p50": NUMBER, "p90": NUMBER, "p99": NUMBER, "max": NUMBER}, extra=True
)
TRANSFERS = array(record({"bytes": POSITIVE, "Bps": NUMBER}, extra=True))
REPORTS: Mapping[str, Check] = {
    "ssd": record(
        {
            "schema": constant(PROBE_SCHEMAS["ssd"]),
            "reader": one_of(*READERS),
            "alignment_bytes": POSITIVE,
            "qd_cap": nullable(POSITIVE),
            "files": array(record({"path": TEXT, "size_bytes": COUNT}, extra=True)),
            "points": array(
                record(
                    {
                        "bytes": POSITIVE,
                        "qd": POSITIVE,
                        "runs": array(
                            record(
                                {
                                    "bytes_read": COUNT,
                                    "seconds": NUMBER,
                                    "Bps": NUMBER,
                                    "lat_us": LATENCY,
                                },
                                extra=True,
                            ),
                            minimum=1,
                        ),
                    },
                    extra=True,
                )
            ),
        },
        extra=True,
    ),
    "cpu": record(
        {
            "schema": constant(PROBE_SCHEMAS["cpu"]),
            "ggml_features": array(TEXT),
            "dram": array(
                record({"threads": POSITIVE, "bytes": POSITIVE, "Bps": NUMBER}, extra=True)
            ),
            "compute": array(
                record(
                    {
                        "format": TEXT,
                        "threads": POSITIVE,
                        "rows": POSITIVE,
                        "cols": POSITIVE,
                        "weight_bytes": POSITIVE,
                        "points": array(
                            record(
                                {
                                    "tokens": POSITIVE,
                                    "us": NUMBER,
                                    "us_q1": NUMBER,
                                    "us_q3": NUMBER,
                                },
                                {"weight_Bps": NUMBER, "flops": NUMBER},
                                extra=True,
                            )
                        ),
                    },
                    extra=True,
                )
            ),
        },
        {
            "skipped": array(record({"format": TEXT, "reason": TEXT}, extra=True)),
            "cpus": array(COUNT),
        },
        extra=True,
    ),
    "memory": record(
        {
            "schema": constant(PROBE_SCHEMAS["memory"]),
            "devices": array(
                record(
                    {
                        "index": COUNT,
                        "name": TEXT,
                        "uuid": TEXT,
                        "pci_bus_id": TEXT,
                        "integrated": flag,
                        "vram_total_bytes": COUNT,
                        "vram_free_bytes": COUNT,
                        "h2d": record({"pinned": TRANSFERS, "pageable": TRANSFERS}, extra=True),
                        "d2h": record({"pinned": TRANSFERS}, extra=True),
                        "d2d_Bps": NUMBER,
                        "read_Bps": NUMBER,
                    },
                    extra=True,
                )
            ),
        },
        {"skipped": array(record({"index": COUNT, "reason": TEXT}, extra=True))},
        extra=True,
    ),
}


def parse(name: str, stdout: str) -> Any:
    """The validated report of probe NAME; ProbeError names the offending field."""
    try:
        value: object = json.loads(stdout)
        if not isinstance(value, dict):
            raise ContractError("output must be one JSON object")
        REPORTS[name](value, "")
        if name == "cpu":
            for row in value["compute"]:
                for point in row["points"]:
                    rate = "weight_Bps" if point["tokens"] == 1 else "flops"
                    if rate not in point:
                        raise ContractError(f"{row['format']} {point['tokens']} tokens lack {rate}")
    except ValueError as error:
        raise ProbeError(f"{EXECUTABLES[name]} output: {error}") from error
    return value


@dataclass(frozen=True)
class Probes:
    """Probe executables by name ("ssd", "cpu", "memory"), their runner and environment."""

    paths: tuple[tuple[str, str], ...]
    runner: Runner = run_process
    environ: Mapping[str, str] = field(
        default_factory=lambda: os.environ, compare=False, repr=False
    )

    def run(self, name: str, arguments: Sequence[str]) -> Any:
        path = dict(self.paths).get(name)
        if path is None:
            raise ProbeError(f"{EXECUTABLES[name]} not found; pass --build DIR or --resources FILE")
        hidden = HIDDEN.get(name, ())
        env = {key: value for key, value in self.environ.items() if key not in hidden}
        result = self.runner([path, *arguments], TIMEOUT_S[name], env)
        if result.returncode:
            lines = [line.strip() for line in result.stderr.splitlines() if line.strip()]
            raise ProbeError(
                f"{EXECUTABLES[name]} exited with status {result.returncode}: "
                + (lines[-1] if lines else "no reason given")
            )
        return parse(name, result.stdout)


def knee(points: Sequence[RandomRead]) -> list[Knee]:
    """Per request size, the smallest QD whose best run reaches the worst run of the fastest QD."""
    sizes: dict[int, list[RandomRead]] = {}
    for point in sorted(points, key=lambda point: (point["bytes"], point["qd"])):
        sizes.setdefault(point["bytes"], []).append(point)
    result: list[Knee] = []
    for size, rows in sorted(sizes.items()):
        fastest = max(rows, key=lambda row: row["Bps"])  # the smallest QD on a tie
        floor = min(fastest["Bps_runs"])
        qd = min(row["qd"] for row in rows if max(row["Bps_runs"]) >= floor)
        result.append({"bytes": size, "qd": qd})
    return result


def storage_entry(report: Any, drive: Drive) -> Storage:
    """Bps is the median of the runs; latency comes from the median run (the lower one if even)."""
    runs: dict[tuple[int, int], list[Any]] = {}
    for point in report["points"]:
        runs.setdefault((point["bytes"], point["qd"]), []).extend(point["runs"])
    points: list[RandomRead] = []
    for (size, qd), group in sorted(runs.items()):
        rates = [float(run["Bps"]) for run in group]
        middle = sorted(group, key=lambda run: float(run["Bps"]))[(len(group) - 1) // 2]
        latency = middle["lat_us"]
        points.append(
            {
                "bytes": size,
                "qd": qd,
                "Bps": float(statistics.median(rates)),
                "Bps_runs": rates,
                "lat_us": {
                    "mean": float(latency["mean"]),
                    "p50": float(latency["p50"]),
                    "p90": float(latency["p90"]),
                    "p99": float(latency["p99"]),
                },
            }
        )
    return {
        "drive_key": drive.key,
        "model": drive.model,
        "bus": drive.bus,
        "reader": report["reader"],
        "alignment_bytes": report["alignment_bytes"],
        "qd_cap": report["qd_cap"],
        "measured_files": len(report["files"]),
        "random_read": points,
        "knee": knee(points),
    }


def dram_rates(report: Any) -> list[ThreadRate]:
    return [{"threads": row["threads"], "Bps": float(row["Bps"])} for row in report["dram"]]


def compute_rows(report: Any) -> list[ComputeRow]:
    """The report's rows; pinned, thread i ran on the i-th CPU, so a row names the first ones."""
    rows: list[ComputeRow] = []
    pinned = sorted(report.get("cpus", []))
    for row in report["compute"]:
        points: list[ComputePoint] = []
        for point in row["points"]:
            value: ComputePoint = {
                "tokens": point["tokens"],
                "us": float(point["us"]),
                "us_q1": float(point["us_q1"]),
                "us_q3": float(point["us_q3"]),
            }
            if point["tokens"] == 1:
                value["weight_Bps"] = float(point["weight_Bps"])
            else:
                value["flops"] = float(point["flops"])
            points.append(value)
        value_row: ComputeRow = {
            "format": row["format"],
            "threads": row["threads"],
            "rows": row["rows"],
            "cols": row["cols"],
            "weight_bytes": row["weight_bytes"],
            "points": points,
        }
        if pinned:
            value_row["cpus"] = pinned[: row["threads"]]
        rows.append(value_row)
    return rows


def skipped(report: Any) -> list[str]:
    return [f"{row['format']} ({row['reason']})" for row in report.get("skipped", [])]


def skipped_devices(report: Any) -> dict[int, str]:
    """Memory probe devices it could not measure, by CUDA index, with the reason."""
    return {row["index"]: row["reason"] for row in report.get("skipped", [])}


def gpu_rows(report: Any) -> list[GPU]:
    """Memory probe devices without free VRAM, which is a runtime fact."""

    def transfers(rows: Any) -> list[Transfer]:
        return [{"bytes": row["bytes"], "Bps": float(row["Bps"])} for row in rows]

    return [
        {
            "index": device["index"],
            "name": device["name"],
            "uuid": device["uuid"],
            "pci_bus_id": device["pci_bus_id"],
            "integrated": device["integrated"],
            "vram_total_bytes": device["vram_total_bytes"],
            "h2d": {
                "pinned": transfers(device["h2d"]["pinned"]),
                "pageable": transfers(device["h2d"]["pageable"]),
            },
            "d2h": {"pinned": transfers(device["d2h"]["pinned"])},
            "d2d_Bps": float(device["d2d_Bps"]),
            "read_Bps": float(device["read_Bps"]),
        }
        for device in report["devices"]
    ]
