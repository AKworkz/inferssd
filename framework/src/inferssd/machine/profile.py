# SPDX-License-Identifier: Apache-2.0
"""Measure the machine and build its characteristics.

CPU topology and memory size are read directly. The CPU probe measures ggml
compute and DRAM reads; the SSD probe reads random aligned extents inside
existing large files on each drive, and on a drive where a given directory holds
none, inside a test file of random data written there and removed after; on
Linux with an NVIDIA GPU the memory probe measures transfers. Sections the
saved file still describes correctly are reused (see ``plan_refresh``).

CPU clusters follow the engine's split (``ssd::platform::topology``): Linux
takes each CPU's ``cpu_capacity``, else ``cpuinfo_max_freq``, sorts them in
descending order and cuts at the largest relative drop above 10%; one CPU
without either value makes one cluster. macOS takes ``hw.perflevel0.logicalcpu``
as the performance cluster and the remaining CPUs as efficiency.
"""

from __future__ import annotations

import glob
import posixpath
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Optional

from inferssd.configuration.settings import cpus
from inferssd.machine.characteristics import (
    CPU,
    GPU,
    PROBE_SCHEMAS,
    SCHEMA,
    STORAGE_PREFIX,
    Characteristics,
    Cluster,
    Compute,
    Identity,
    LogicalCPU,
    Memory,
    Section,
    Storage,
)
from inferssd.machine.identity import (
    Drive,
    GPUQueryFailed,
    Host,
    drive_info,
    is_test_file,
    live_identity,
    online_cpus,
    read,
    same_hardware,
    sysctl,
    sysctl_int,
    with_saved_gpus,
)
from inferssd.machine.probes import (
    ProbeError,
    Probes,
    compute_rows,
    dram_rates,
    gpu_rows,
    skipped,
    skipped_devices,
    storage_entry,
)
from inferssd.machine.store import CPU_GROUP, Refresh, plan_refresh
from inferssd.source.header import SHARD

MIN_FILE_BYTES = 1 << 30
MAX_FILES = 8
TEST_FILE_BYTES = 4 << 30  # spread over 4 GiB, reads do not stay in a drive's own cache
# ggml's CPU feature list also names build options; they are not instruction sets.
BUILD_OPTIONS = frozenset(("ACCELERATE", "CPU_HBM", "KLEIDIAI", "LLAMAFILE", "OPENMP", "REPACK"))
Clock = Callable[[], float]
Progress = Callable[[str], None]


def utc(seconds: float) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(seconds))


def stderr_progress(line: str) -> None:
    print("inferssd profile: " + line, file=sys.stderr, flush=True)


def section(start: float, end: float, method: str, status: str, detail: str = "") -> Section:
    return {
        "measured_utc": utc(start),
        "duration_s": round(max(0.0, end - start), 3),
        "method": method,
        "status": status,
        "detail": detail,
    }


def split_fastest(values: Sequence[tuple[int, float]]) -> tuple[list[int], list[int]]:
    """The engine's split: cut the descending values at the largest relative drop above 10%."""
    ordered = sorted(values, key=lambda item: -item[1])
    cut, step = len(ordered), 0.10
    for index in range(1, len(ordered)):
        drop = 1.0 - ordered[index][1] / ordered[index - 1][1]
        if drop > step:
            step, cut = drop, index
    return sorted(cpu for cpu, _ in ordered[:cut]), sorted(cpu for cpu, _ in ordered[cut:])


def clusters(fast: Sequence[int], slow: Sequence[int]) -> list[Cluster]:
    if not slow:
        return [{"id": 0, "cpus": list(fast), "kind": "uniform"}]
    return [
        {"id": 0, "cpus": list(fast), "kind": "performance"},
        {"id": 1, "cpus": list(slow), "kind": "efficiency"},
    ]


def _number(value: Optional[str]) -> Optional[int]:
    return int(value) if value is not None and value.isdigit() else None


def linux_topology(host: Host) -> CPU:
    logical: list[LogicalCPU] = []
    values: list[tuple[int, float]] = []
    uniform = False
    for cpu in online_cpus(host):
        base = f"/sys/devices/system/cpu/cpu{cpu}"
        capacity = _number(read(host, base + "/cpu_capacity"))
        max_khz = _number(read(host, base + "/cpufreq/cpuinfo_max_freq"))
        siblings = read(host, base + "/topology/thread_siblings_list")
        logical.append(
            {
                "id": cpu,
                "core": _number(read(host, base + "/topology/core_id")),
                "package": _number(read(host, base + "/topology/physical_package_id")),
                "smt_siblings": list(cpus(siblings)) if siblings else None,
                "capacity": capacity,
                "max_khz": max_khz,
                "cluster": 0,
            }
        )
        value = capacity or max_khz
        uniform = uniform or not value
        values.append((cpu, float(value or 0)))
    if not logical:
        raise ValueError("no online CPU found")
    ids = [row["id"] for row in logical]
    fast, slow = (ids, []) if uniform else split_fastest(values)
    for row in logical:
        row["cluster"] = 0 if row["id"] in fast else 1
    return {
        "logical": logical,
        "clusters": clusters(fast, slow),
        "affinity_supported": True,
        "isa": [],
    }


def darwin_topology(host: Host) -> CPU:
    """macOS has no affinity, so CPU ids only count CPUs; the engine treats them the same."""
    count = sysctl_int(host, "hw.logicalcpu")
    physical = _number(sysctl(host, "hw.physicalcpu"))
    level0 = _number(sysctl(host, "hw.perflevel0.logicalcpu")) or 0
    cut = level0 if 0 < level0 < count else count
    single = physical == count
    logical: list[LogicalCPU] = [
        {
            "id": cpu,
            "core": cpu if single else None,
            "package": 0,
            "smt_siblings": [cpu] if single else None,
            "capacity": None,
            "max_khz": None,
            "cluster": 0 if cpu < cut else 1,
        }
        for cpu in range(count)
    ]
    return {
        "logical": logical,
        "clusters": clusters(range(cut), range(cut, count)),
        "affinity_supported": False,
        "isa": [],
    }


def probe_threads(cpu: CPU) -> list[int]:
    """1, the performance-cluster size and every logical CPU, without repeats."""
    total = len(cpu["logical"])
    fast = next(
        (len(row["cpus"]) for row in cpu["clusters"] if row["kind"] == "performance"), total
    )
    return sorted({1, fast, total})


def pinned_sets(cpu: CPU) -> list[list[int]]:
    """Each cluster's CPUs and every CPU, where threads can be pinned and there are clusters.

    One token streams a format's weights at a rate the strategy compares
    between them, a thread pinned to each CPU, to place decode.
    """
    if not cpu["affinity_supported"] or len(cpu["clusters"]) < 2:
        return []
    sets = [sorted(row["cpus"]) for row in cpu["clusters"]]
    return sets + [sorted(row["id"] for row in cpu["logical"])]


def model_files(host: Host, paths: Sequence[str]) -> list[tuple[str, int]]:
    """Regular files under PATHS (directories recursively); a split GGUF adds all its shards."""
    found: dict[str, int] = {}
    for path in paths:
        match = SHARD.match(posixpath.basename(path))
        names = [path]
        if match:
            directory = posixpath.dirname(path)
            pattern = f"{glob.escape(match.group(1))}-*-of-{match.group(2)}.gguf"
            if directory:
                pattern = posixpath.join(glob.escape(directory), pattern)
            names = host.glob(pattern) or names
        # An empty directory lists nothing: its drive is measured with a test file.
        listed = [row for name in names for row in host.regular_files(name)]
        for real, size in listed:
            if is_test_file(real):  # a drive test file, never a model
                continue
            found.setdefault(real, size)
    return list(found.items())


@dataclass(frozen=True)
class ModelDrive:
    """A drive the given paths are on: their files there and the given directories on it."""

    drive: Drive
    files: list[tuple[str, int]]
    directories: list[str]


def model_drives(host: Host, paths: Sequence[str]) -> dict[str, ModelDrive]:
    devices: dict[tuple[int, int], Drive] = {}
    drives: dict[str, ModelDrive] = {}

    def holding(path: str) -> ModelDrive:
        device = host.device(path)
        if device not in devices:
            devices[device] = drive_info(host, path)
        drive = devices[device]
        return drives.setdefault(drive.key, ModelDrive(drive, [], []))

    seen: set[str] = set()
    for path in paths:
        listed = model_files(host, [path])
        for real, size in listed:
            if real not in seen:
                seen.add(real)
                holding(real).files.append((real, size))
        # Only a directory with no file to read gets a test file: one whose files sit on
        # another drive (a mount, links) has that drive measured from them instead.
        if host.is_directory(path) and all(size < MIN_FILE_BYTES for _, size in listed):
            holding(path).directories.append(path)
    return drives


def models_drive(drives: dict[str, ModelDrive]) -> Optional[str]:
    """The drive holding most of the given files' bytes, else the first given directory's."""
    held = {key: sum(size for _, size in place.files) for key, place in drives.items()}
    if any(held.values()):
        return max(held, key=lambda key: held[key])
    return next((key for key, place in drives.items() if place.directories), None)


def measure_cpu(
    host: Host, identity: Identity, probes: Probes, clock: Clock, progress: Progress
) -> tuple[CPU, Memory, Compute, dict[str, Section]]:
    system = identity["hard"]["platform"]
    start = clock()
    progress("reading CPU topology and memory size")
    topology_error = ""
    try:
        cpu = linux_topology(host) if system == "linux" else darwin_topology(host)
    except (OSError, ValueError, RuntimeError) as error:
        topology_error = f"topology unavailable: {error}"
        count = identity["hard"]["logical_cpus"]
        cpu = {
            "logical": [
                {
                    "id": index,
                    "core": None,
                    "package": None,
                    "smt_siblings": None,
                    "capacity": None,
                    "max_khz": None,
                    "cluster": 0,
                }
                for index in range(count)
            ],
            "clusters": clusters(range(count), []),
            "affinity_supported": system == "linux",
            "isa": [],
        }
    memory: Memory = {
        "total_bytes": identity["hard"]["ram_bytes"],
        "page_bytes": host.page_bytes(),
        "unified": False,
        "dram_read_Bps": [],
    }
    read_end = clock()
    threads = probe_threads(cpu)
    method = f"{PROBE_SCHEMAS['cpu']} ggml_mul_mat, threads {','.join(map(str, threads))}"
    progress(f"CPU probe: ggml compute and DRAM read with {','.join(map(str, threads))} threads")
    probe_error = ""
    report: Any = None
    try:
        report = probes.run("cpu", ["--threads", ",".join(map(str, threads))])
    except ProbeError as error:
        probe_error = str(error)
    compute: Compute = {"cpu": []}
    unpinned: list[str] = []
    if report is not None:
        cpu["isa"] = [name for name in report["ggml_features"] if name not in BUILD_OPTIONS]
        memory["dram_read_Bps"] = dram_rates(report)
        compute["cpu"] = compute_rows(report)
        for group in pinned_sets(cpu):
            listed = ",".join(map(str, group))
            progress(f"CPU probe: one token with a thread pinned to each of CPUs {listed}")
            arguments = ["--cpus", listed, "--threads", str(len(group)), "--tokens", "1"]
            try:
                pinned = probes.run("cpu", arguments)
            except ProbeError as error:
                unpinned.append(f"pinned to CPUs {listed}: {error}")
                continue
            compute["cpu"] += compute_rows(pinned)
        if pinned_sets(cpu):
            method += "; one token pinned to each cluster and to every CPU"
    end = clock()
    source = "sysfs" if system == "linux" else "sysctl"
    failed = "the CPU probe failed (see compute)" if probe_error else ""
    isa_missing = f"ISA unknown: {failed}" if failed else ""
    errors = "; ".join(filter(None, (topology_error, isa_missing)))
    cpu_status = "failed" if topology_error and probe_error else "partial" if errors else "ok"
    dram_missing = failed or ("" if memory["dram_read_Bps"] else "the probe reported none")
    skip = "skipped " + ", ".join(skipped(report)) if report is not None and skipped(report) else ""
    skip = "; ".join(filter(None, [skip] + unpinned))
    compute_error = probe_error or ("" if compute["cpu"] else "the probe measured no format")
    return (
        cpu,
        memory,
        compute,
        {
            "cpu": section(
                start,
                read_end,
                f"{source} topology; ISA from {PROBE_SCHEMAS['cpu']}",
                cpu_status,
                errors,
            ),
            "memory": section(
                start,
                end,
                f"{source} memory size; DRAM read from {PROBE_SCHEMAS['cpu']}",
                "partial" if dram_missing else "ok",
                f"DRAM read unknown: {dram_missing}" if dram_missing else "",
            ),
            "compute": section(
                read_end,
                end,
                method,
                "failed" if compute_error else "ok",
                compute_error or skip,
            ),
        },
    )


def measure_gpus(
    identity: Identity, probes: Probes, clock: Clock, progress: Progress, failure: str = ""
) -> tuple[list[GPU], Section]:
    """GPU rows; a GPU of the identity that the probe did not measure makes the section partial."""
    start = clock()
    if failure:
        return [], section(start, clock(), "nvidia-smi", "failed", failure)
    if "memory" not in identity["soft"]["probe_versions"]:
        why = (
            "no NVIDIA GPU found by nvidia-smi"
            if identity["hard"]["platform"] == "linux"
            else "the GPU transfer probe runs on Linux with CUDA only"
        )
        return [], section(start, clock(), "none", "ok", why)
    progress("GPU probe: VRAM size and transfer rates")
    method = PROBE_SCHEMAS["memory"]
    try:
        report = probes.run("memory", [])
    except ProbeError as error:
        return [], section(start, clock(), method, "failed", str(error))
    gpus = gpu_rows(report)
    if not gpus:
        return [], section(start, clock(), method, "failed", "the probe found no CUDA device")
    measured = {gpu["uuid"].lower() for gpu in gpus}
    missing = [
        gpu["uuid"]
        for gpu in identity["hard"]["gpus"]
        if gpu["backend"] == "CUDA" and gpu["uuid"].lower() not in measured
    ]
    reasons = [f"device {index}: {why}" for index, why in sorted(skipped_devices(report).items())]
    notes = []
    if missing:
        notes.append("not measured: " + ", ".join(missing))
    if reasons:
        notes.append("the probe skipped " + "; ".join(reasons))
    status = "partial" if missing else "ok"
    return gpus, section(start, clock(), method, status, "; ".join(notes))


def measure_drive(
    host: Host, place: ModelDrive, probes: Probes, clock: Clock, progress: Progress
) -> tuple[Optional[Storage], Section]:
    """Read rates of PLACE's drive: in its largest files, else in a test file in its directory."""
    start = clock()
    drive = place.drive
    for directory in place.directories:
        host.clear_test_files(directory)
    chosen = [
        path
        for path, size in sorted(place.files, key=lambda item: -item[1])
        if size >= MIN_FILE_BYTES
    ][:MAX_FILES]
    if chosen:
        progress(
            f"SSD probe: random reads in {len(chosen)} existing files on {drive.model} "
            f"({drive.bus})"
        )
        method = f"{PROBE_SCHEMAS['ssd']} read-only random extents in {len(chosen)} existing files"
        return _read_rates(drive, chosen, method, probes, clock, start)
    largest = max((size for _, size in place.files), default=0)
    reason = f"no file of at least {MIN_FILE_BYTES} bytes on this drive among the given paths"
    if not place.directories:
        detail = f"{reason} (largest {largest} bytes)"
        return None, section(start, clock(), "none", "failed", detail)
    directory = place.directories[0]
    free = host.free_bytes(directory)
    if free < TEST_FILE_BYTES + MIN_FILE_BYTES:
        return None, section(
            start,
            clock(),
            "none",
            "failed",
            f"{reason}, and the {free} bytes free in {directory} leave no room for a "
            f"{TEST_FILE_BYTES}-byte test file",
        )
    progress(
        f"SSD probe: random reads in a {TEST_FILE_BYTES >> 30} GiB test file written to "
        f"{directory} on {drive.model} ({drive.bus})"
    )
    method = (
        f"{PROBE_SCHEMAS['ssd']} random extents in a {TEST_FILE_BYTES}-byte test file of "
        "random data, removed after"
    )
    try:
        with host.test_file(directory, TEST_FILE_BYTES) as path:
            return _read_rates(drive, [path], method, probes, clock, start)
    except OSError as error:
        return None, section(start, clock(), method, "failed", f"the test file failed: {error}")


def _read_rates(
    drive: Drive,
    paths: Sequence[str],
    method: str,
    probes: Probes,
    clock: Clock,
    start: float,
) -> tuple[Optional[Storage], Section]:
    try:
        entry = storage_entry(
            probes.run("ssd", [argument for path in paths for argument in ("--file", path)]), drive
        )
    except ProbeError as error:
        return None, section(start, clock(), method, "failed", str(error))
    if not entry["random_read"]:
        return None, section(start, clock(), method, "failed", "the probe measured no point")
    return entry, section(start, clock(), f"{method}, {entry['reader']} reader", "ok")


@dataclass(frozen=True)
class Survey:
    """What a profile run would measure: the live identity, the model drives and the plan."""

    identity: Identity
    gpu_failure: str  # why nvidia-smi failed, or ""
    saved_unified: bool  # the saved file's unified memory, kept while the GPUs are unknown
    drives: dict[str, ModelDrive]
    refresh: Refresh

    @property
    def current(self) -> bool:
        """Nothing needs measuring: a saved profile can be used as it is."""
        return not self.refresh.sections and not self.refresh.stale_storage


def survey(
    host: Host,
    model_paths: Sequence[str],
    previous: Optional[Characteristics] = None,
    reprofile: bool = False,
    progress: Progress = stderr_progress,
) -> Survey:
    """Read the live identity and the model drives, and plan what to measure. Runs no probe."""
    gpu_failure, saved_unified = "", False
    try:
        identity = live_identity(host)
    except GPUQueryFailed as error:
        # Keep the saved GPUs, so a passing driver problem does not look like new hardware.
        gpu_failure, identity = str(error), error.identity
        if previous is not None:
            kept = with_saved_gpus(identity, previous["identity"])
            if not same_hardware(previous["identity"], kept):
                identity, saved_unified = kept, previous["memory"]["unified"]
        progress(f"GPUs unknown: {gpu_failure}")
    drives = model_drives(host, model_paths) if model_paths else {}
    refresh = plan_refresh(previous, identity, list(drives), reprofile)
    if gpu_failure:
        refresh = refresh.including("gpus", "nvidia-smi failed")
    return Survey(identity, gpu_failure, saved_unified, drives, refresh)


def profile(
    host: Host,
    probes: Probes,
    model_paths: Sequence[str],
    previous: Optional[Characteristics] = None,
    reprofile: bool = False,
    clock: Clock = time.time,
    progress: Progress = stderr_progress,
    surveyed: Optional[Survey] = None,
) -> Characteristics:
    """New characteristics; PREVIOUS itself when nothing needs measuring.

    SURVEYED, when given, is the survey of these same arguments made just before.
    """
    if surveyed is None:
        progress("reading machine identity")
        surveyed = survey(host, model_paths, previous, reprofile, progress)
    identity, gpu_failure, saved_unified = (
        surveyed.identity,
        surveyed.gpu_failure,
        surveyed.saved_unified,
    )
    drives, refresh = surveyed.drives, surveyed.refresh
    if previous is not None and surveyed.current:
        progress("the saved profile is up to date")
        return previous
    for name, reason in refresh.sections:
        progress(f"will measure {name}: {reason}")
    base = None if refresh.full else previous
    sections = dict(base["sections"]) if base is not None else {}
    if base is None or any(name in refresh for name in CPU_GROUP):
        cpu, memory, compute, measured = measure_cpu(host, identity, probes, clock, progress)
        sections.update(measured)
    else:
        cpu, memory, compute = base["cpu"], base["memory"].copy(), base["compute"]
    if base is None or "gpus" in refresh:
        gpus, sections["gpus"] = measure_gpus(identity, probes, clock, progress, gpu_failure)
    else:
        gpus = base["gpus"]
    storage: list[Storage] = []
    if base is not None:
        keep = not refresh.stale_storage
        for name in list(sections):
            if name.startswith(STORAGE_PREFIX) and (not keep or name in refresh):
                del sections[name]
        if keep:
            storage = [
                entry
                for entry in base["storage"]
                if STORAGE_PREFIX + entry["drive_key"] not in refresh
            ]
    for key, place in drives.items():
        if STORAGE_PREFIX + key in refresh:
            entry, sections[STORAGE_PREFIX + key] = measure_drive(
                host, place, probes, clock, progress
            )
            if entry is not None:
                storage.append(entry)
    memory["unified"] = (
        (identity["hard"]["platform"] == "darwin" and identity["hard"]["arch"] == "arm64")
        or any(gpu["integrated"] for gpu in gpus)
        or saved_unified
    )
    value: Characteristics = {
        "schema": SCHEMA,
        "created_utc": utc(clock()),
        "identity": identity,
        "sections": {name: sections[name] for name in sorted(sections)},
        "cpu": cpu,
        "memory": memory,
        "gpus": gpus,
        "storage": sorted(storage, key=lambda entry: entry["drive_key"]),
        "compute": compute,
    }
    return value
