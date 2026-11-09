# SPDX-License-Identifier: Apache-2.0
"""The inferssd-machine/1 file: validation names the field, serialization is stable."""

from __future__ import annotations

import copy
import json

import pytest

from inferssd.contracts import ContractError
from inferssd.machine.characteristics import dumps, load, loads, validate

DRIVE = "linux:nvme:Example NVMe 2TB:" + "ab" * 32


def section(status="ok", detail=""):
    return {
        "measured_utc": "2026-10-29T12:00:00Z",
        "duration_s": 1.5,
        "method": "test",
        "status": status,
        "detail": detail,
    }


def sample():
    """A valid file for a Linux machine with one CUDA GPU and one measured NVMe drive."""
    return {
        "schema": "inferssd-machine/1",
        "created_utc": "2026-10-29T12:00:00Z",
        "identity": {
            "hard": {
                "platform": "linux",
                "arch": "x86_64",
                "machine_id_sha256": "00" * 32,
                "model": "Workstation X",
                "cpu_brand": "Example CPU 9000",
                "logical_cpus": 4,
                "ram_bytes": 34359738368,
                "gpus": [
                    {
                        "backend": "CUDA",
                        "uuid": "GPU-1111",
                        "name": "NVIDIA GeForce RTX 5090",
                        "vram_bytes": 34190917632,
                    }
                ],
            },
            "soft": {
                "os_release": "Ubuntu 24.04.1 LTS",
                "kernel": "6.8.0-50-generic",
                "gpu_driver": "580.65.06",
                "probe_versions": {
                    "ssd": "inferssd-ssdprobe/1",
                    "cpu": "inferssd-cpuprobe/2",
                    "memory": "inferssd-memprobe/1",
                },
            },
        },
        "sections": {
            "cpu": section(),
            "memory": section(),
            "compute": section(),
            "gpus": section(),
            "storage:" + DRIVE: section(),
        },
        "cpu": {
            "logical": [
                {
                    "id": cpu,
                    "core": cpu,
                    "package": 0,
                    "smt_siblings": [cpu],
                    "capacity": 1024 if cpu < 2 else 512,
                    "max_khz": None,
                    "cluster": 0 if cpu < 2 else 1,
                }
                for cpu in range(4)
            ],
            "clusters": [
                {"id": 0, "cpus": [0, 1], "kind": "performance"},
                {"id": 1, "cpus": [2, 3], "kind": "efficiency"},
            ],
            "affinity_supported": True,
            "isa": ["AVX2", "FMA"],
        },
        "memory": {
            "total_bytes": 34359738368,
            "page_bytes": 4096,
            "unified": False,
            "dram_read_Bps": [{"threads": 1, "Bps": 1.2e10}, {"threads": 4, "Bps": 4.0e10}],
        },
        "gpus": [
            {
                "index": 0,
                "name": "NVIDIA GeForce RTX 5090",
                "uuid": "GPU-1111",
                "pci_bus_id": "0000:01:00.0",
                "integrated": False,
                "vram_total_bytes": 34190917632,
                "h2d": {
                    "pinned": [{"bytes": 1048576, "Bps": 2.4e10}],
                    "pageable": [{"bytes": 1048576, "Bps": 1.1e10}],
                },
                "d2h": {"pinned": [{"bytes": 1048576, "Bps": 2.5e10}]},
                "d2d_Bps": 1.5e12,
                "read_Bps": 1.6e12,
            }
        ],
        "storage": [
            {
                "drive_key": DRIVE,
                "model": "Example NVMe 2TB",
                "bus": "nvme",
                "reader": "io_uring",
                "alignment_bytes": 4096,
                "qd_cap": None,
                "measured_files": 3,
                "random_read": [
                    {
                        "bytes": 1048576,
                        "qd": 4,
                        "Bps": 1.27e9,
                        "Bps_runs": [1.25e9, 1.29e9],
                        "lat_us": {"mean": 3100.0, "p50": 2900.0, "p90": 4100.0, "p99": 6000.0},
                    }
                ],
                "knee": [{"bytes": 1048576, "qd": 4}],
            }
        ],
        "compute": {
            "cpu": [
                {
                    "format": "Q4_K",
                    "threads": 4,
                    "rows": 4096,
                    "cols": 4096,
                    "weight_bytes": 9437184,
                    "points": [
                        {"tokens": 1, "us": 210.0, "weight_Bps": 4.5e10},
                        {"tokens": 32, "us": 900.0, "flops": 1.2e12},
                    ],
                }
            ]
        },
    }


def test_sample_is_valid_and_round_trips(tmp_path):
    value = sample()
    text = dumps(value)
    assert text.endswith("}\n") and json.loads(text) == value
    assert dumps(json.loads(text)) == text
    assert list(json.loads(text)) == sorted(value)
    path = tmp_path / "machine.json"
    path.write_text(text)
    assert load(path) == value


def mutate(path, value):
    data = sample()
    *parents, last = path
    target = data
    for key in parents:
        target = target[key]
    if value is KeyError:
        del target[last]
    else:
        target[last] = value
    return data


@pytest.mark.parametrize(
    "path,value,message",
    [
        (("schema",), "inferssd-machine/2", "schema must be 'inferssd-machine/1'"),
        (("created_utc",), "yesterday", "created_utc must be a UTC time"),
        (("identity", "hard", "cpu_brand"), KeyError, "identity.hard.cpu_brand is required"),
        (
            ("identity", "hard", "logical_cpus"),
            0,
            "identity.hard.logical_cpus must be an integer >= 1",
        ),
        (("identity", "hard", "platform"), "plan9", "identity.hard.platform must be one of"),
        (("identity", "soft", "extra"), 1, "identity.soft.extra is not a known field"),
        (("sections", "cpu", "status"), "great", "sections.cpu.status must be one of ok"),
        (("sections", "gpus"), KeyError, "sections.gpus is required"),
        (("sections", "network"), section(), "sections.network is not a known section"),
        (("memory", "page_bytes"), True, "memory.page_bytes must be an integer"),
        (
            ("memory", "dram_read_Bps", 0, "Bps"),
            float("nan"),
            r"memory.dram_read_Bps\[0\].Bps must be a finite",
        ),
        (
            ("memory", "dram_read_Bps", 1, "Bps"),
            -1,
            r"memory.dram_read_Bps\[1\].Bps must be a finite number >= 0",
        ),
        (("cpu", "logical", 2, "cluster"), 7, r"cpu.logical\[2\].cluster names no cluster"),
        (("cpu", "clusters", 0, "kind"), "big", r"cpu.clusters\[0\].kind must be one of"),
        (
            ("gpus", 0, "h2d", "pinned", 0, "bytes"),
            "1M",
            r"gpus\[0\].h2d.pinned\[0\].bytes must be an integer",
        ),
        (("gpus", 0, "vram_free_bytes"), 1, r"gpus\[0\].vram_free_bytes is not a known field"),
        (("storage", 0, "bus"), "scsi", r"storage\[0\].bus must be one of"),
        (
            ("storage", 0, "random_read", 0, "Bps_runs"),
            [],
            r"Bps_runs must be an array of at least 1",
        ),
        (("storage", 0, "random_read", 0, "lat_us", "p99"), KeyError, r"lat_us.p99 is required"),
        (
            ("storage", 0, "random_read"),
            [],
            r"storage\[0\].random_read must be an array of at least 1",
        ),
        (("storage", 0, "knee"), [], r"storage\[0\].knee must be an array of at least 1"),
        (
            ("compute", "cpu", 0, "points", 1, "gflops"),
            1.0,
            r"points\[1\].gflops is not a known field",
        ),
    ],
)
def test_validation_names_the_field(path, value, message):
    with pytest.raises(ContractError, match=message):
        validate(mutate(path, value))


def test_storage_needs_its_section_and_unique_drives():
    data = sample()
    del data["sections"]["storage:" + DRIVE]
    with pytest.raises(ContractError, match=r"storage\[0\] has no section storage:"):
        validate(data)
    data = sample()
    data["storage"].append(copy.deepcopy(data["storage"][0]))
    with pytest.raises(ContractError, match="repeated drive_key"):
        validate(data)


def test_not_json_or_not_an_object():
    with pytest.raises(ContractError, match="not JSON"):
        loads("{")
    with pytest.raises(ContractError, match="must be a JSON object"):
        loads("[]")
