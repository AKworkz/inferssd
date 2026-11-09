# SPDX-License-Identifier: Apache-2.0
"""The machine characteristics file, schema inferssd-machine/1.

The file describes the machine only: no model facts, placement or cache sizes,
and no runtime facts such as available memory or free VRAM. Every number names
its unit: ``_bytes``, ``_Bps`` (bytes per second), ``_us``, ``_s``, ``_khz``.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Optional, TypedDict, cast

from inferssd.contracts import ContractError

SCHEMA = "inferssd-machine/1"
# Probe output schemas this version reads, by the section group each one measures.
PROBE_SCHEMAS = {
    "ssd": "inferssd-ssdprobe/1",
    "cpu": "inferssd-cpuprobe/2",
    "memory": "inferssd-memprobe/1",
}
BASE_SECTIONS = ("cpu", "memory", "compute", "gpus")
STORAGE_PREFIX = "storage:"
STATUSES = ("ok", "partial", "failed")
CLUSTER_KINDS = ("performance", "efficiency", "uniform")
BUSES = ("nvme", "sata", "usb", "apple", "unknown")
READERS = ("io_uring", "ioring", "overlapped", "pool")
UTC = re.compile(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ")


class GPUIdentity(TypedDict):
    backend: str
    uuid: str
    name: str
    vram_bytes: int


class HardIdentity(TypedDict):
    platform: str
    arch: str
    machine_id_sha256: Optional[str]
    model: Optional[str]
    cpu_brand: str
    logical_cpus: int
    ram_bytes: int
    gpus: list[GPUIdentity]


class SoftIdentity(TypedDict):
    os_release: Optional[str]
    kernel: str
    gpu_driver: Optional[str]
    probe_versions: dict[str, str]


class Identity(TypedDict):
    hard: HardIdentity
    soft: SoftIdentity


class Section(TypedDict):
    measured_utc: str
    duration_s: float
    method: str
    status: str
    detail: str


class LogicalCPU(TypedDict):
    id: int
    core: Optional[int]
    package: Optional[int]
    smt_siblings: Optional[list[int]]
    capacity: Optional[int]
    max_khz: Optional[int]
    cluster: int


class Cluster(TypedDict):
    id: int
    cpus: list[int]
    kind: str


class CPU(TypedDict):
    logical: list[LogicalCPU]
    clusters: list[Cluster]
    affinity_supported: bool
    isa: list[str]


class ThreadRate(TypedDict):
    threads: int
    Bps: float


class Memory(TypedDict):
    total_bytes: int
    page_bytes: int
    unified: bool
    dram_read_Bps: list[ThreadRate]


class Transfer(TypedDict):
    bytes: int
    Bps: float


class HostToDevice(TypedDict):
    pinned: list[Transfer]
    pageable: list[Transfer]


class DeviceToHost(TypedDict):
    pinned: list[Transfer]


class GPU(TypedDict):
    index: int
    name: str
    uuid: str
    pci_bus_id: str
    integrated: bool
    vram_total_bytes: int
    h2d: HostToDevice
    d2h: DeviceToHost
    d2d_Bps: float
    read_Bps: float


class Latency(TypedDict):
    mean: float
    p50: float
    p90: float
    p99: float


class RandomRead(TypedDict):
    bytes: int
    qd: int
    Bps: float
    Bps_runs: list[float]
    lat_us: Latency


class Knee(TypedDict):
    bytes: int
    qd: int


class Storage(TypedDict):
    drive_key: str
    model: str
    bus: str
    reader: str
    alignment_bytes: int
    qd_cap: Optional[int]
    measured_files: int
    random_read: list[RandomRead]
    knee: list[Knee]


class _ComputePoint(TypedDict):
    tokens: int
    us: float


class ComputePoint(_ComputePoint, total=False):
    weight_Bps: float  # tokens == 1
    flops: float  # tokens > 1, FLOP/s
    # One call's time at the point's first and third quartile, as us is its median.
    us_q1: float
    us_q3: float


class _ComputeRow(TypedDict):
    format: str
    threads: int
    rows: int
    cols: int
    weight_bytes: int
    points: list[ComputePoint]


class ComputeRow(_ComputeRow, total=False):
    cpus: list[int]  # one thread pinned to each, ascending; absent where the threads are unpinned


class Compute(TypedDict):
    cpu: list[ComputeRow]


class Characteristics(TypedDict):
    schema: str
    created_utc: str
    identity: Identity
    sections: dict[str, Section]
    cpu: CPU
    memory: Memory
    gpus: list[GPU]
    storage: list[Storage]
    compute: Compute


Check = Callable[[object, str], None]


def _fail(field: str, expected: str) -> ContractError:
    return ContractError(f"{field} must be {expected}")


def integer(minimum: int = 0) -> Check:
    def check(value: object, field: str) -> None:
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise _fail(field, f"an integer >= {minimum}")

    return check


def number(value: object, field: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _fail(field, "a number")
    if not math.isfinite(value) or value < 0:
        raise _fail(field, "a finite number >= 0")


def text(value: object, field: str) -> None:
    if not isinstance(value, str):
        raise _fail(field, "a string")


def flag(value: object, field: str) -> None:
    if not isinstance(value, bool):
        raise _fail(field, "true or false")


def utc(value: object, field: str) -> None:
    if not isinstance(value, str) or not UTC.fullmatch(value):
        raise _fail(field, "a UTC time like 2026-10-29T12:00:00Z")


def one_of(*values: str) -> Check:
    def check(value: object, field: str) -> None:
        if value not in values:
            raise _fail(field, "one of " + ", ".join(values))

    return check


def constant(expected: str) -> Check:
    def check(value: object, field: str) -> None:
        if value != expected:
            raise _fail(field, repr(expected))

    return check


def nullable(inner: Check) -> Check:
    def check(value: object, field: str) -> None:
        if value is not None:
            inner(value, field)

    return check


def array(inner: Check, *, minimum: int = 0) -> Check:
    def check(value: object, field: str) -> None:
        if not isinstance(value, list):
            raise _fail(field, "an array")
        if len(value) < minimum:
            raise _fail(field, f"an array of at least {minimum} items")
        for index, item in enumerate(value):
            inner(item, f"{field}[{index}]")

    return check


def mapping(inner: Check) -> Check:
    def check(value: object, field: str) -> None:
        if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
            raise _fail(field, "an object with string keys")
        for key, item in value.items():
            inner(item, f"{field}.{key}")

    return check


def record(
    fields: Mapping[str, Check],
    optional: Optional[Mapping[str, Check]] = None,
    *,
    extra: bool = False,
) -> Check:
    """An object with exactly FIELDS (plus any of OPTIONAL); EXTRA admits unknown fields."""
    known: Mapping[str, Check] = optional or {}

    def check(value: object, field: str) -> None:
        if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
            raise _fail(field, "an object with string keys")
        prefix = field + "." if field else ""
        for key, inner in fields.items():
            if key not in value:
                raise ContractError(f"{prefix}{key} is required")
            inner(value[key], prefix + key)
        for key, inner in known.items():
            if key in value:
                inner(value[key], prefix + key)
        unknown = sorted(set(value) - set(fields) - set(known))
        if unknown and not extra:
            raise ContractError(f"{prefix}{unknown[0]} is not a known field")

    return check


NUMBER: Check = number
TEXT: Check = text
COUNT = integer(0)
POSITIVE = integer(1)
TRANSFERS = array(record({"bytes": POSITIVE, "Bps": NUMBER}))
LATENCY = record({"mean": NUMBER, "p50": NUMBER, "p90": NUMBER, "p99": NUMBER})
COMPUTE_POINT = record(
    {"tokens": POSITIVE, "us": NUMBER},
    {"weight_Bps": NUMBER, "flops": NUMBER, "us_q1": NUMBER, "us_q3": NUMBER},
)
COMPUTE_ROW = record(
    {
        "format": TEXT,
        "threads": POSITIVE,
        "rows": POSITIVE,
        "cols": POSITIVE,
        "weight_bytes": POSITIVE,
        "points": array(COMPUTE_POINT),
    },
    {"cpus": array(COUNT, minimum=1)},
)
GPU_ROW = record(
    {
        "index": COUNT,
        "name": TEXT,
        "uuid": TEXT,
        "pci_bus_id": TEXT,
        "integrated": flag,
        "vram_total_bytes": COUNT,
        "h2d": record({"pinned": TRANSFERS, "pageable": TRANSFERS}),
        "d2h": record({"pinned": TRANSFERS}),
        "d2d_Bps": NUMBER,
        "read_Bps": NUMBER,
    }
)
SECTION = record(
    {
        "measured_utc": utc,
        "duration_s": NUMBER,
        "method": TEXT,
        "status": one_of(*STATUSES),
        "detail": TEXT,
    }
)
FILE = record(
    {
        "schema": constant(SCHEMA),
        "created_utc": utc,
        "identity": record(
            {
                "hard": record(
                    {
                        "platform": one_of("linux", "darwin", "windows"),
                        "arch": TEXT,
                        "machine_id_sha256": nullable(TEXT),
                        "model": nullable(TEXT),
                        "cpu_brand": TEXT,
                        "logical_cpus": POSITIVE,
                        "ram_bytes": POSITIVE,
                        "gpus": array(
                            record(
                                {
                                    "backend": TEXT,
                                    "uuid": TEXT,
                                    "name": TEXT,
                                    "vram_bytes": COUNT,
                                }
                            )
                        ),
                    }
                ),
                "soft": record(
                    {
                        "os_release": nullable(TEXT),
                        "kernel": TEXT,
                        "gpu_driver": nullable(TEXT),
                        "probe_versions": mapping(TEXT),
                    }
                ),
            }
        ),
        "sections": mapping(SECTION),
        "cpu": record(
            {
                "logical": array(
                    record(
                        {
                            "id": COUNT,
                            "core": nullable(COUNT),
                            "package": nullable(COUNT),
                            "smt_siblings": nullable(array(COUNT)),
                            "capacity": nullable(COUNT),
                            "max_khz": nullable(COUNT),
                            "cluster": COUNT,
                        }
                    )
                ),
                "clusters": array(
                    record(
                        {
                            "id": COUNT,
                            "cpus": array(COUNT, minimum=1),
                            "kind": one_of(*CLUSTER_KINDS),
                        }
                    )
                ),
                "affinity_supported": flag,
                "isa": array(TEXT),
            }
        ),
        "memory": record(
            {
                "total_bytes": COUNT,
                "page_bytes": COUNT,
                "unified": flag,
                "dram_read_Bps": array(record({"threads": POSITIVE, "Bps": NUMBER})),
            }
        ),
        "gpus": array(GPU_ROW),
        "storage": array(
            record(
                {
                    "drive_key": TEXT,
                    "model": TEXT,
                    "bus": one_of(*BUSES),
                    "reader": one_of(*READERS),
                    "alignment_bytes": POSITIVE,
                    "qd_cap": nullable(POSITIVE),
                    "measured_files": POSITIVE,
                    "random_read": array(
                        record(
                            {
                                "bytes": POSITIVE,
                                "qd": POSITIVE,
                                "Bps": NUMBER,
                                "Bps_runs": array(NUMBER, minimum=1),
                                "lat_us": LATENCY,
                            }
                        ),
                        minimum=1,
                    ),
                    "knee": array(record({"bytes": POSITIVE, "qd": POSITIVE}), minimum=1),
                }
            )
        ),
        "compute": record({"cpu": array(COMPUTE_ROW)}),
    }
)


def validate(value: object) -> Characteristics:
    """Check VALUE against inferssd-machine/1; errors name the offending field."""
    if not isinstance(value, dict):
        raise ContractError("machine characteristics must be a JSON object")
    FILE(value, "")
    data = cast(Characteristics, value)
    keys = [entry["drive_key"] for entry in data["storage"]]
    if len(set(keys)) != len(keys):
        raise ContractError("storage has a repeated drive_key")
    for name in data["sections"]:
        if name not in BASE_SECTIONS and not name.startswith(STORAGE_PREFIX):
            raise ContractError(f"sections.{name} is not a known section")
    for name in BASE_SECTIONS:
        if name not in data["sections"]:
            raise ContractError(f"sections.{name} is required")
    for index, key in enumerate(keys):
        if STORAGE_PREFIX + key not in data["sections"]:
            raise ContractError(f"storage[{index}] has no section {STORAGE_PREFIX}{key}")
    clusters = {cluster["id"] for cluster in data["cpu"]["clusters"]}
    for index, cpu in enumerate(data["cpu"]["logical"]):
        if cpu["cluster"] not in clusters:
            raise ContractError(f"cpu.logical[{index}].cluster names no cluster")
    return data


def dumps(value: Characteristics) -> str:
    """Stable text: validated, sorted keys, two-space indent, trailing newline."""
    return json.dumps(validate(value), sort_keys=True, indent=2) + "\n"


def loads(content: str) -> Characteristics:
    try:
        data: object = json.loads(content)
    except ValueError as error:
        raise ContractError(f"machine characteristics are not JSON: {error}") from error
    return validate(data)


def load(path: Path) -> Characteristics:
    return loads(path.read_text(encoding="utf-8"))
