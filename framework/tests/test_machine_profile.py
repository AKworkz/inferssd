# SPDX-License-Identifier: Apache-2.0
"""End-to-end profiling with a fake host and recorded probe outputs."""

from __future__ import annotations

import copy
import json
import os
import plistlib
import subprocess
from pathlib import Path

import pytest
from test_machine_identity import SMI, FakeHost, darwin_commands, linux_tree, nvme_key, ok
from test_machine_probes import CPU_REPORT, MEMORY_REPORT, PATHS, FakeRunner, replies

from inferssd.machine.characteristics import validate
from inferssd.machine.identity import Command, Drive
from inferssd.machine.probes import Completed, Probes
from inferssd.machine.profile import (
    ModelDrive,
    linux_topology,
    model_files,
    models_drive,
    profile,
    split_fastest,
)

GIB = 1 << 30


class Clock:
    def __init__(self, start=1_800_000_000.0):
        self.now = start

    def __call__(self):
        self.now += 0.5
        return self.now


def shards(directory, sizes):
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for index, size in enumerate(sizes, 1):
        path = directory / f"model-{index:05d}-of-{len(sizes):05d}.gguf"
        with open(path, "wb") as stream:
            stream.truncate(size)  # sparse: nothing is read
        paths.append(str(Path(os.path.realpath(path))))
    return paths


def linux_machine(tmp_path, **tree):
    root = tmp_path / "root"
    commands = linux_tree(root, **tree)
    models = tmp_path / "models"
    files = shards(models, [2 * GIB, 3 * GIB, GIB // 2])
    host = FakeHost(root, commands=commands, devices={os.path.realpath(models): (259, 2)})
    return host, files


def run(host, files, runner, previous=None, reprofile=False):
    lines = []
    value = profile(
        host,
        Probes(PATHS, runner),
        files,
        previous,
        reprofile,
        clock=Clock(),
        progress=lines.append,
    )
    return value, lines


def executables(runner):
    return [argv[0].rsplit("/", 1)[-1] for argv, _ in runner.calls]


def test_linux_profile_end_to_end(tmp_path):
    host, files = linux_machine(tmp_path)
    runner = FakeRunner(replies())
    value, lines = run(host, [files[0]], runner)
    assert validate(value) is value
    assert {name: section["status"] for name, section in value["sections"].items()} == {
        "cpu": "ok",
        "memory": "ok",
        "compute": "ok",
        "gpus": "ok",
        "storage:" + nvme_key(): "ok",
    }
    # The first shard brings its set; the two shards of at least 1 GiB are read, largest first.
    assert runner.calls == [
        (["/opt/inferssd/bin/inferssd-cpuprobe", "--threads", "1,4"], 1800.0),
        (["/opt/inferssd/bin/inferssd-memprobe"], 900.0),
        (["/opt/inferssd/bin/inferssd-ssdprobe", "--file", files[1], "--file", files[0]], 1800.0),
    ]
    assert value["identity"]["hard"]["cpu_brand"] == "Example CPU 9000"
    assert value["cpu"]["clusters"] == [{"id": 0, "cpus": [0, 1, 2, 3], "kind": "uniform"}]
    assert value["cpu"]["isa"] == ["AVX2", "FMA", "F16C"] and value["cpu"]["affinity_supported"]
    assert value["memory"] == {
        "total_bytes": 32768000 * 1024,
        "page_bytes": 4096,
        "unified": False,
        "dram_read_Bps": [{"threads": 1, "Bps": 1.2e10}, {"threads": 4, "Bps": 3.6e10}],
    }
    assert value["compute"]["cpu"] == CPU_REPORT["compute"]
    assert value["sections"]["compute"]["detail"] == "skipped MXFP4 (no CPU kernel)"
    assert [gpu["uuid"] for gpu in value["gpus"]] == ["GPU-1111"]
    assert "vram_free_bytes" not in value["gpus"][0]
    (drive,) = value["storage"]
    assert (drive["model"], drive["bus"], drive["reader"]) == (
        "Example NVMe 2TB",
        "nvme",
        "io_uring",
    )
    assert value["sections"]["storage:" + nvme_key()]["method"] == (
        "inferssd-ssdprobe/1 read-only random extents in 2 existing files, io_uring reader"
    )
    text = json.dumps(value)
    assert "vram_free" not in text and "available" not in text and "model-0000" not in text
    assert lines[0] == "reading machine identity" and len(lines) == 10
    assert any(line.startswith("SSD probe: random reads in 2 existing files") for line in lines)


def test_unchanged_machine_reuses_the_saved_profile(tmp_path):
    host, files = linux_machine(tmp_path)
    first, _ = run(host, files, FakeRunner(replies()))
    saved = json.loads(json.dumps(first))
    runner = FakeRunner({})
    again, lines = run(host, files, runner, previous=saved)
    assert again is saved and runner.calls == []
    assert lines[-1] == "the saved profile is up to date"


def test_kernel_change_measures_storage_only(tmp_path):
    host, files = linux_machine(tmp_path)
    saved = json.loads(json.dumps(run(host, files, FakeRunner(replies()))[0]))
    saved["identity"]["soft"]["kernel"] = "6.1.0-old"
    stale = copy.deepcopy(saved["storage"][0])
    stale["drive_key"] = "linux:sata:Elsewhere:" + "ee" * 32
    saved["storage"].append(stale)
    saved["sections"]["storage:" + stale["drive_key"]] = saved["sections"]["cpu"]
    runner = FakeRunner(replies())
    value, _ = run(host, files, runner, previous=saved)
    assert executables(runner) == ["inferssd-ssdprobe"]
    for name in ("cpu", "memory", "compute", "gpus"):
        assert value["sections"][name] is saved["sections"][name]
    # A drive not measured now is dropped: its numbers predate the kernel change.
    assert [entry["drive_key"] for entry in value["storage"]] == [nvme_key()]
    assert sorted(value["sections"]) == [
        "compute",
        "cpu",
        "gpus",
        "memory",
        "storage:" + nvme_key(),
    ]
    assert value["identity"]["soft"]["kernel"] == "6.8.0-50-generic"
    validate(value)


def test_new_drive_keeps_the_other_drives(tmp_path):
    host, files = linux_machine(tmp_path)
    saved = json.loads(json.dumps(run(host, files, FakeRunner(replies()))[0]))
    other = copy.deepcopy(saved["storage"][0])
    other["drive_key"] = "linux:sata:Elsewhere:" + "ee" * 32
    saved["storage"] = [other]
    saved["sections"] = {
        name: section
        for name, section in saved["sections"].items()
        if not name.startswith("storage:")
    }
    saved["sections"]["storage:" + other["drive_key"]] = saved["sections"]["cpu"]
    runner = FakeRunner(replies())
    value, _ = run(host, files, runner, previous=saved)
    assert executables(runner) == ["inferssd-ssdprobe"]
    assert [entry["drive_key"] for entry in value["storage"]] == sorted(
        [nvme_key(), other["drive_key"]]
    )
    validate(value)


def test_missing_probes_record_failed_sections(tmp_path):
    root = tmp_path / "root"
    commands = linux_tree(root, gpu=False)
    small = shards(tmp_path / "models", [GIB // 4])
    host = FakeHost(
        root, commands=commands, devices={os.path.realpath(tmp_path / "models"): (259, 2)}
    )
    value = profile(host, Probes((), FakeRunner({})), small, clock=Clock(), progress=print)
    validate(value)
    sections = value["sections"]
    assert sections["compute"]["status"] == "failed"
    assert "inferssd-cpuprobe not found" in sections["compute"]["detail"]
    assert sections["cpu"]["status"] == "partial" and sections["memory"]["status"] == "partial"
    assert sections["gpus"] == dict(
        sections["gpus"], status="ok", detail="no NVIDIA GPU found by nvidia-smi"
    )
    storage = sections["storage:" + nvme_key()]
    assert (
        storage["status"] == "failed"
        and "no file of at least 1073741824 bytes" in storage["detail"]
    )
    assert value["storage"] == [] and value["compute"]["cpu"] == [] and value["cpu"]["isa"] == []


def test_failing_probe_reports_its_reason(tmp_path):
    host, files = linux_machine(tmp_path)
    runner = FakeRunner(
        replies(
            **{
                "inferssd-ssdprobe": Completed(2, "", "cannot open for direct I/O: /models/x\n"),
                "inferssd-memprobe": Completed(0, json.dumps(dict(MEMORY_REPORT, devices=[]))),
            }
        )
    )
    value, _ = run(host, files, runner)
    storage = value["sections"]["storage:" + nvme_key()]
    assert storage["status"] == "failed"
    assert (
        storage["detail"]
        == "inferssd-ssdprobe exited with status 2: cannot open for direct I/O: /models/x"
    )
    assert value["sections"]["gpus"]["detail"] == "the probe found no CUDA device"


def test_missing_model_path_is_an_error(tmp_path):
    host, _ = linux_machine(tmp_path)
    with pytest.raises(FileNotFoundError):
        run(host, [str(tmp_path / "absent.gguf")], FakeRunner(replies()))


def test_a_directory_without_large_files_is_measured_with_a_test_file(tmp_path):
    host, files = linux_machine(tmp_path)
    empty = tmp_path / "models" / "empty"
    (empty / "sub").mkdir(parents=True)
    runner = FakeRunner(replies())
    value, lines = run(host, [str(empty)], runner)
    ((test_file, size, existed),) = host.test_files
    assert (os.path.dirname(test_file), size, existed) == (str(empty), 4 * GIB, True)
    assert not os.path.exists(test_file) and sorted(os.listdir(empty)) == ["sub"]
    assert runner.calls[-1] == (
        ["/opt/inferssd/bin/inferssd-ssdprobe", "--file", test_file],
        1800.0,
    )
    section = value["sections"]["storage:" + nvme_key()]
    assert section["status"] == "ok" and section["method"] == (
        "inferssd-ssdprobe/1 random extents in a 4294967296-byte test file of random data, "
        "removed after, io_uring reader"
    )
    assert any(line.startswith("SSD probe: random reads in a 4 GiB test file") for line in lines)
    # Existing files of 1 GiB on the drive are read instead; no test file is written.
    run(host, [str(tmp_path / "models")], FakeRunner(replies()), reprofile=True)
    assert len(host.test_files) == 1


def test_a_killed_runs_test_file_is_neither_read_nor_kept(tmp_path):
    host, _ = linux_machine(tmp_path)
    empty = tmp_path / "models" / "empty"
    empty.mkdir()
    child = subprocess.Popen(["true"])
    child.wait()
    left = empty / f".inferssd-drive-test-{child.pid}-x"
    with open(left, "wb") as stream:
        stream.truncate(4 * GIB)
    runner = FakeRunner(replies())
    run(host, [str(empty)], runner)
    ((test_file, _, _),) = host.test_files
    assert runner.calls[-1][0][-1] == test_file != str(left)
    assert os.listdir(empty) == []


def test_the_models_drive_holds_most_of_the_files_else_the_directory():
    big, small = (Drive(key, "x", "nvme") for key in ("big", "small"))
    assert (
        models_drive(
            {"small": ModelDrive(small, [("/a", 5)], []), "big": ModelDrive(big, [("/b", 9)], [])}
        )
        == "big"
    )
    assert (
        models_drive({"small": ModelDrive(small, [], []), "big": ModelDrive(big, [], ["/m"])})
        == "big"
    )
    assert models_drive({}) is None


def test_a_directory_without_room_for_the_test_file_fails_its_drive(tmp_path):
    host, _ = linux_machine(tmp_path)
    host.free = 5 * GIB - 1
    empty = tmp_path / "models" / "empty"
    empty.mkdir()
    value, _ = run(host, [str(empty)], FakeRunner(replies()))
    section = value["sections"]["storage:" + nvme_key()]
    assert (
        section["status"] == "failed"
        and "leave no room for a 4294967296-byte test file" in (section["detail"])
    )
    assert host.test_files == [] and value["storage"] == []


def test_split_gguf_in_a_directory_with_glob_characters(tmp_path):
    files = shards(tmp_path / "[q8]", [GIB, GIB, GIB])
    assert sorted(path for path, _ in model_files(FakeHost(tmp_path), [files[1]])) == files


def test_isa_leaves_out_build_options(tmp_path):
    host, _ = linux_machine(tmp_path)
    features = ["AVX2", "FMA", "LLAMAFILE", "OPENMP", "F16C", "REPACK"]
    report = dict(CPU_REPORT, ggml_features=features)
    runner = FakeRunner(replies(**{"inferssd-cpuprobe": Completed(0, json.dumps(report))}))
    assert run(host, [], runner)[0]["cpu"]["isa"] == ["AVX2", "FMA", "F16C"]


SECOND_GPU = "GPU-2222, NVIDIA GeForce RTX 5090, 32607, 580.65.06\n"


def test_gpu_the_probe_skipped_is_named_and_measured_again(tmp_path):
    host, _ = linux_machine(tmp_path)
    host.commands[SMI] = Command(0, host.commands[SMI].stdout + SECOND_GPU)
    busy = dict(MEMORY_REPORT, skipped=[{"index": 1, "reason": "only 300 MiB of VRAM free"}])
    runner = FakeRunner(replies(**{"inferssd-memprobe": Completed(0, json.dumps(busy))}))
    first, _ = run(host, [], runner)
    assert [gpu["uuid"] for gpu in first["gpus"]] == ["GPU-1111"]
    assert first["sections"]["gpus"]["status"] == "partial"
    assert first["sections"]["gpus"]["detail"] == (
        "not measured: GPU-2222; the probe skipped device 1: only 300 MiB of VRAM free"
    )
    both = dict(
        MEMORY_REPORT,
        devices=[
            *MEMORY_REPORT["devices"],
            dict(MEMORY_REPORT["devices"][0], index=1, uuid="GPU-2222"),
        ],
    )
    runner = FakeRunner(replies(**{"inferssd-memprobe": Completed(0, json.dumps(both))}))
    value, lines = run(host, [], runner, previous=json.loads(json.dumps(first)))
    assert executables(runner) == ["inferssd-memprobe"]
    assert "will measure gpus: last attempt was partial" in lines
    assert [gpu["uuid"] for gpu in value["gpus"]] == ["GPU-1111", "GPU-2222"]
    assert value["sections"]["gpus"]["status"] == "ok"


def test_failing_nvidia_smi_keeps_the_saved_gpus(tmp_path):
    host, files = linux_machine(tmp_path)
    working = host.commands[SMI]
    saved = json.loads(json.dumps(run(host, files, FakeRunner(replies()))[0]))
    host.commands[SMI] = Command(18, "Failed to initialize NVML: Driver/library version mismatch\n")
    runner = FakeRunner(replies())
    value, lines = run(host, files, runner, previous=saved)
    # Nothing is measured: the GPUs are unknown, not gone, so this is not new hardware.
    assert runner.calls == [] and value["identity"] == saved["identity"]
    assert value["gpus"] == [] and value["sections"]["gpus"]["status"] == "failed"
    assert value["sections"]["gpus"]["detail"] == (
        "nvidia-smi failed with status 18: "
        "Failed to initialize NVML: Driver/library version mismatch"
    )
    for name in ("cpu", "memory", "compute", "storage:" + nvme_key()):
        assert value["sections"][name] is saved["sections"][name]
    assert "will measure gpus: nvidia-smi failed" in lines
    validate(value)
    host.commands[SMI] = working
    runner = FakeRunner(replies())
    again, _ = run(host, files, runner, previous=json.loads(json.dumps(value)))
    assert executables(runner) == ["inferssd-memprobe"]
    assert again["sections"]["gpus"]["status"] == "ok" and len(again["gpus"]) == 1


def test_failing_nvidia_smi_keeps_saved_unified_memory(tmp_path):
    host, _ = linux_machine(tmp_path)
    device = dict(MEMORY_REPORT["devices"][0], integrated=True)
    report = dict(MEMORY_REPORT, devices=[device])
    runner = FakeRunner(replies(**{"inferssd-memprobe": Completed(0, json.dumps(report))}))
    saved = json.loads(json.dumps(run(host, [], runner)[0]))
    host.commands[SMI] = Command(18, "Failed to initialize NVML: Driver/library version mismatch\n")
    value, _ = run(host, [], FakeRunner(replies()), previous=saved, reprofile=True)
    assert value["gpus"] == [] and value["memory"]["unified"] is True


def test_failing_nvidia_smi_without_a_saved_profile(tmp_path):
    host, _ = linux_machine(tmp_path)
    host.commands[SMI] = Command(9, "NVIDIA-SMI has failed because it couldn't communicate\n")
    value, _ = run(host, [], FakeRunner(replies()))
    assert value["identity"]["hard"]["gpus"] == [] and value["gpus"] == []
    assert value["sections"]["gpus"]["status"] == "failed"
    assert "status 9: NVIDIA-SMI has failed" in value["sections"]["gpus"]["detail"]


def test_integrated_gpu_means_unified_memory(tmp_path):
    host, files = linux_machine(tmp_path)
    device = dict(MEMORY_REPORT["devices"][0], integrated=True)
    runner = FakeRunner(
        replies(
            **{"inferssd-memprobe": Completed(0, json.dumps(dict(MEMORY_REPORT, devices=[device])))}
        )
    )
    assert run(host, [], runner)[0]["memory"]["unified"] is True


@pytest.mark.parametrize(
    "capacities,max_khz,fast,slow",
    [
        ((1024, 1024, 1024, 1024, 600, 600, 600, 600), None, [0, 1, 2, 3], [4, 5, 6, 7]),
        ((None,) * 4, (5400000, 5400000, 4100000, 4000000), [0, 1], [2, 3]),
        ((1024, 980, 960, 1024), None, [0, 1, 2, 3], []),
        # Three levels: the engine cuts once, at the largest drop.
        ((1024, 1024, 800, 800, 400, 400), None, [0, 1, 2, 3], [4, 5]),
        ((1024, None, 512, 512), None, [0, 1, 2, 3], []),
    ],
)
def test_linux_clusters_follow_the_engine_split(tmp_path, capacities, max_khz, fast, slow):
    linux_tree(tmp_path, capacities=capacities, max_khz=max_khz)
    cpu = linux_topology(FakeHost(tmp_path))
    kinds = ["performance", "efficiency"] if slow else ["uniform"]
    assert [(row["cpus"], row["kind"]) for row in cpu["clusters"]] == list(
        zip([fast, slow] if slow else [fast], kinds)
    )
    assert [row["cluster"] for row in cpu["logical"]] == [
        0 if index in fast else 1 for index in range(len(capacities))
    ]


def test_split_fastest_matches_the_engine_rule():
    assert split_fastest([(0, 100.0), (1, 95.0), (2, 90.0)]) == ([0, 1, 2], [])
    assert split_fastest([(0, 100.0), (1, 95.0), (2, 85.0)]) == ([0, 1], [2])
    assert split_fastest([(3, 50.0), (0, 100.0), (1, 80.0)]) == ([0, 1], [3])


def test_smt_siblings_and_cores(tmp_path):
    linux_tree(tmp_path, capacities=(1024,) * 4, smt=True)
    logical = linux_topology(FakeHost(tmp_path))["logical"]
    assert [(row["core"], row["smt_siblings"]) for row in logical] == [
        (0, [0, 2]),
        (1, [1, 3]),
        (0, [0, 2]),
        (1, [1, 3]),
    ]


def test_darwin_profile_end_to_end(tmp_path):
    models = tmp_path / "models"
    files = shards(models, [2 * GIB])
    volume = {
        "DeviceIdentifier": "disk3s5",
        "BusProtocol": "Apple Fabric",
        "VolumeUUID": "D8CB630F-7471-44CB-92B8-31C7A0372333",
        "APFSPhysicalStores": [{"APFSPhysicalStore": "disk0s2"}],
    }
    disk = {
        "DeviceIdentifier": "disk0",
        "MediaName": "APPLE SSD AP0512Z",
        "BusProtocol": "Apple Fabric",
    }
    commands = darwin_commands()
    commands[("df", "-P", files[0])] = ok(
        "Filesystem 512-blocks Used Available Capacity Mounted on\n"
        "/dev/disk3s5 965492904 812117032 95648840 90% /System/Volumes/Data\n"
    )
    commands[("diskutil", "info", "-plist", "/System/Volumes/Data")] = ok(
        plistlib.dumps(volume).decode()
    )
    commands[("diskutil", "info", "-plist", "disk0")] = ok(plistlib.dumps(disk).decode())
    host = FakeHost(
        tmp_path / "root", "darwin", "arm64", commands, devices={os.path.realpath(models): (1, 5)}
    )
    ssd = json.loads(replies()["inferssd-ssdprobe"].stdout)
    ssd.update(reader="pool", alignment_bytes=16384, qd_cap=8)
    runner = FakeRunner(replies(**{"inferssd-ssdprobe": Completed(0, json.dumps(ssd))}))
    value, _ = run(host, [str(models)], runner)
    validate(value)
    assert executables(runner) == ["inferssd-cpuprobe", "inferssd-ssdprobe"]
    assert runner.calls[0][0][1:] == ["--threads", "1,4,10"]
    assert value["cpu"]["affinity_supported"] is False
    assert [(row["kind"], len(row["cpus"])) for row in value["cpu"]["clusters"]] == [
        ("performance", 4),
        ("efficiency", 6),
    ]
    assert value["memory"]["unified"] is True and value["memory"]["page_bytes"] == 16384
    assert [gpu["name"] for gpu in value["identity"]["hard"]["gpus"]] == ["Apple M5, 10-core GPU"]
    assert value["gpus"] == [] and value["sections"]["gpus"]["status"] == "ok"
    (drive,) = value["storage"]
    assert drive["drive_key"].startswith("darwin:apple:APPLE SSD AP0512Z:")
    assert (drive["reader"], drive["qd_cap"], drive["alignment_bytes"]) == ("pool", 8, 16384)


class PinnedRunner(FakeRunner):
    """A CPU probe that reports pinned runs as pinned, at a rate set by their CPUs."""

    def __init__(self, rates, failing=()):
        super().__init__(replies())
        self.rates, self.failing = rates, failing

    def __call__(self, argv, timeout_s, env):
        if "--cpus" not in argv:
            return super().__call__(argv, timeout_s, env)
        self.calls.append((list(argv), timeout_s))
        listed = argv[argv.index("--cpus") + 1]
        if listed in self.failing:
            return Completed(1, "", "CPU 3 is not available to this process")
        cpus = [int(cpu) for cpu in listed.split(",")]
        row = {
            "format": "Q4_K",
            "threads": len(cpus),
            "rows": 4096,
            "cols": 4096,
            "weight_bytes": 9437184,
            "points": [{"tokens": 1, "us": 100.0, "weight_Bps": self.rates[listed]}],
        }
        return Completed(0, json.dumps(dict(CPU_REPORT, compute=[row], cpus=cpus, skipped=[])))


def test_two_clusters_measure_one_token_pinned_to_each_and_to_every_cpu(tmp_path):
    host, files = linux_machine(tmp_path, capacities=(512, 1024, 1024, 512))
    runner = PinnedRunner({"1,2": 6e10, "0,3": 2e10, "0,1,2,3": 5e10})
    value, lines = run(host, [files[0]], runner)
    assert validate(value) is value
    assert value["cpu"]["clusters"] == [
        {"id": 0, "cpus": [1, 2], "kind": "performance"},
        {"id": 1, "cpus": [0, 3], "kind": "efficiency"},
    ]
    pinned = [argv[1:] for argv, _ in runner.calls if "--cpus" in argv]
    assert pinned == [
        ["--cpus", "1,2", "--threads", "2", "--tokens", "1"],
        ["--cpus", "0,3", "--threads", "2", "--tokens", "1"],
        ["--cpus", "0,1,2,3", "--threads", "4", "--tokens", "1"],
    ]
    rows = value["compute"]["cpu"]
    assert rows[0] == CPU_REPORT["compute"][0] and "cpus" not in rows[0]
    assert [(row["cpus"], row["points"][0]["weight_Bps"]) for row in rows[1:]] == [
        ([1, 2], 6e10),
        ([0, 3], 2e10),
        ([0, 1, 2, 3], 5e10),
    ]
    assert value["sections"]["compute"]["status"] == "ok"
    assert value["sections"]["compute"]["method"].endswith(
        "; one token pinned to each cluster and to every CPU"
    )
    assert any("pinned to each of CPUs 1,2" in line for line in lines)


def test_a_failed_pinned_run_is_noted_and_left_out(tmp_path):
    host, files = linux_machine(tmp_path, capacities=(512, 1024, 1024, 512))
    runner = PinnedRunner({"1,2": 6e10, "0,3": 2e10}, failing=("0,1,2,3",))
    value, _ = run(host, [files[0]], runner)
    assert [row.get("cpus") for row in value["compute"]["cpu"]] == [None, [1, 2], [0, 3]]
    detail = value["sections"]["compute"]["detail"]
    assert detail.startswith("skipped MXFP4 (no CPU kernel); pinned to CPUs 0,1,2,3: ")
    assert value["sections"]["compute"]["status"] == "ok"
