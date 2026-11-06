# SPDX-License-Identifier: Apache-2.0
"""Strategy inputs: model facts from synthetic GGUF headers, runtime facts from fake hosts."""

from __future__ import annotations

import struct
from pathlib import Path

import pytest
from test_machine_identity import FakeHost, linux_tree, nvme_key, ok, write

from inferssd.machine.identity import Command
from inferssd.strategy.inputs import (
    GPUState,
    RuntimeFacts,
    model_drive_key,
    model_facts,
    read_runtime,
    visible_gpus,
)

MXFP4, Q3_K, Q4_K, Q5_K, Q6_K, Q8_0, F32 = 39, 11, 12, 13, 14, 8, 0
# ggml type -> (elements per block, bytes per block), as the GGUF format defines them
BLOCKS = {MXFP4: (32, 17), Q3_K: (256, 110), Q4_K: (256, 144), Q5_K: (256, 176)}
BLOCKS.update({Q6_K: (256, 210), Q8_0: (32, 34), F32: (1, 4)})
PAGE = 4096


def _string(value):
    raw = value.encode()
    return struct.pack("<Q", len(raw)) + raw


def write_gguf(path, tensors, kv, *, alignment=PAGE):
    """A header-only GGUF file: KV maps keys to str or int (u32); TENSORS are (name, shape, type).

    Tensors are laid out back to back, each at a multiple of ALIGNMENT.
    """
    items = dict(kv)
    items.setdefault("general.alignment", alignment)
    raw = b"GGUF" + struct.pack("<IQQ", 3, len(tensors), len(items))
    for key, value in items.items():
        if isinstance(value, str):
            raw += _string(key) + struct.pack("<I", 8) + _string(value)
        else:
            raw += _string(key) + struct.pack("<II", 4, value)
    offset = 0
    for name, shape, kind in tensors:
        raw += _string(name) + struct.pack(f"<I{len(shape)}QIQ", len(shape), *shape, kind, offset)
        block, size = BLOCKS[kind]
        count = 1
        for dimension in shape:
            count *= dimension
        offset += -(-(count // block * size) // alignment) * alignment
    Path(path).write_bytes(raw)
    return path


def moe(arch, *, layers, experts, used, latent, hidden, gate, up, down, first=0):
    """Tensors and KV of a mixture-of-experts model; GATE, UP, DOWN give the type per layer."""
    tensors = [("token_embd.weight", (latent, 1024), Q8_0), ("output_norm.weight", (latent,), F32)]
    for index, layer in enumerate(range(first, first + layers)):
        for role, kinds in (("gate", gate), ("up", up), ("down", down)):
            kind = kinds[index] if isinstance(kinds, list) else kinds
            shape = (hidden, latent, experts) if role == "down" else (latent, hidden, experts)
            tensors.append((f"blk.{layer}.ffn_{role}_exps.weight", shape, kind))
        tensors.append((f"blk.{layer}.attn_norm.weight", (latent,), F32))
    kv = {"general.architecture": arch, f"{arch}.expert_used_count": used}
    return tensors, kv


def deepseek_like():
    """DeepSeek-V4.1-Flash's routed layout: 40 MoE layers after 3 dense ones, 384 MXFP4 experts."""
    return moe(
        "deepseek41",
        layers=40,
        experts=384,
        used=6,
        latent=5120,
        hidden=2304,
        gate=MXFP4,
        up=MXFP4,
        down=MXFP4,
        first=3,
    )


def muse_like():
    """Muse Spark's routed layout: 62 MoE layers, 256 experts, Q4_K gate/up and Q5_K down."""
    return moe(
        "muse-spark",
        layers=62,
        experts=256,
        used=8,
        latent=4096,
        hidden=4096,
        gate=Q4_K,
        up=Q4_K,
        down=Q5_K,
    )


def test_deepseek_like_header(tmp_path):
    facts = model_facts(str(write_gguf(tmp_path / "dsv.gguf", *deepseek_like())))
    assert facts.paths == (str(tmp_path / "dsv.gguf"),)
    assert (facts.architecture, facts.layers, facts.experts, facts.experts_used) == (
        "deepseek41",
        40,
        384,
        6,
    )
    assert [(r.name, r.format, r.rows, r.cols) for r in facts.roles] == [
        ("gate", "MXFP4", 2304, 5120),
        ("up", "MXFP4", 2304, 5120),
        ("down", "MXFP4", 5120, 2304),
    ]
    role = 5120 * 2304 // 32 * 17
    assert [r.slice_bytes() for r in facts.roles] == [role] * 3 == [6266880] * 3
    # Page-aligned tensors of whole-page slices read exactly their bytes.
    assert facts.expert_bytes == facts.expert_read_bytes == 3 * role
    assert facts.dense_bytes == 5120 * 1024 // 32 * 34 + 5120 * 4 + 40 * 5120 * 4
    assert facts.format_bytes == (("MXFP4", 40 * 3 * 384 * role),)
    assert facts.dominant_format == "MXFP4"


def test_muse_like_header_and_dominant_format(tmp_path):
    facts = model_facts(str(write_gguf(tmp_path / "muse.gguf", *muse_like())))
    assert (facts.architecture, facts.layers, facts.experts, facts.experts_used) == (
        "muse-spark",
        62,
        256,
        8,
    )
    assert [(r.format, r.slice_bytes()) for r in facts.roles] == [
        ("Q4_K", 9437184),
        ("Q4_K", 9437184),
        ("Q5_K", 11534336),
    ]
    assert facts.expert_read_bytes == 2 * 9437184 + 11534336
    # Gate and up together hold more bytes than down.
    assert facts.format_bytes[0] == ("Q4_K", 62 * 2 * 256 * 9437184)
    assert facts.dominant_format == "Q4_K"


def test_split_model_reads_every_shard(tmp_path):
    tensors, kv = moe(
        "toy", layers=4, experts=8, used=2, latent=256, hidden=512, gate=Q8_0, up=Q8_0, down=Q8_0
    )
    first, second = tensors[:8], tensors[8:]
    write_gguf(tmp_path / "toy-00001-of-00002.gguf", first, kv)
    write_gguf(tmp_path / "toy-00002-of-00002.gguf", second, {})
    facts = model_facts(str(tmp_path / "toy-00001-of-00002.gguf"))
    assert facts.paths == (
        str(tmp_path / "toy-00001-of-00002.gguf"),
        str(tmp_path / "toy-00002-of-00002.gguf"),
    )
    assert (facts.architecture, facts.layers, facts.experts_used) == ("toy", 4, 2)


def test_mixed_formats_rank_by_layers_and_widen_unaligned_reads(tmp_path):
    downs = [Q5_K, Q6_K, Q6_K]
    tensors, kv = moe(
        "toy", layers=3, experts=2, used=1, latent=256, hidden=256, gate=Q4_K, up=Q4_K, down=downs
    )
    facts = model_facts(str(write_gguf(tmp_path / "mixed.gguf", tensors, kv, alignment=32)))
    assert facts.down.formats == (("Q6_K", 2), ("Q5_K", 1))
    assert facts.down.format == "Q6_K"
    assert facts.down.slice_bytes() == 256 * 210
    assert facts.down.slice_bytes("Q5_K") == 256 * 176
    assert facts.experts_used == 1
    # Slices of 36-54 KiB at 32-byte offsets each read whole pages around them.
    assert facts.expert_bytes == 2 * 256 * 144 + 256 * 210
    assert facts.expert_read_bytes > facts.expert_bytes
    assert facts.expert_read_bytes % PAGE == 0


def test_headers_without_routed_roles_are_refused(tmp_path):
    with pytest.raises(ValueError, match="no routed expert tensors"):
        model_facts(str(write_gguf(tmp_path / "dense.gguf", [("w", (32, 32), Q8_0)], {})))
    tensors, kv = moe(
        "toy", layers=1, experts=2, used=1, latent=256, hidden=256, gate=Q8_0, up=Q8_0, down=Q8_0
    )
    tensors = [tensor for tensor in tensors if "gate" not in tensor[0]]
    with pytest.raises(ValueError, match="no routed gate expert tensors"):
        model_facts(str(write_gguf(tmp_path / "fused.gguf", tensors, kv)))


def test_model_drive_key_follows_the_files(tmp_path):
    linux_tree(tmp_path / "root")
    model = tmp_path / "models" / "m.gguf"
    model.parent.mkdir()
    model.write_bytes(b"x" * 100)
    host = FakeHost(tmp_path / "root", devices={str(tmp_path / "models"): (259, 2)})
    assert model_drive_key([str(model)], host) == nvme_key()
    assert model_drive_key([str(tmp_path / "missing.gguf")], host) is None


class RuntimeHost(FakeHost):
    def __init__(self, root, *, affinity=None, environment=None, **options):
        super().__init__(root, **options)
        self.allowed, self.env = affinity, dict(environment or {})

    def affinity(self):
        return self.allowed

    def environment(self):
        return self.env


SMI_GPUS = ("nvidia-smi", "--query-gpu=index,uuid,memory.free", "--format=csv,noheader,nounits")
SMI_APPS = ("nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader")


def linux_runtime(root, *, gpus="0, GPU-aaaa, 30000\n1, GPU-bbbb, [N/A]\n", apps=""):
    write(root, "/proc/meminfo", "MemTotal: 64000000 kB\nMemAvailable: 48000000 kB\n")
    write(root, "/proc/self/cgroup", "0::/user.slice/run.scope\n")
    write(root, "/sys/fs/cgroup/user.slice/memory.max", f"{40 << 30}\n")
    write(root, "/sys/fs/cgroup/user.slice/memory.current", f"{14 << 30}\n")
    stat = f"anon {9 << 30}\nfile {5 << 30}\nactive_file {3 << 30}\ninactive_file {1 << 30}\n"
    write(root, "/sys/fs/cgroup/user.slice/memory.stat", stat)
    write(root, "/sys/fs/cgroup/user.slice/run.scope/memory.max", "max\n")
    write(root, "/sys/fs/cgroup/user.slice/run.scope/memory.current", "4096\n")
    return {SMI_GPUS: ok(gpus), SMI_APPS: ok(apps)}


def test_linux_runtime_facts(tmp_path):
    commands = linux_runtime(tmp_path, apps="GPU-bbbb, 4242\n")
    host = RuntimeHost(tmp_path, affinity=(0, 2, 3), commands=commands)
    facts = read_runtime(memory_budget_bytes=1 << 34, host=host)
    assert facts == RuntimeFacts(
        (0, 2, 3),
        (GPUState("GPU-aaaa", 30000 << 20, 0), GPUState("GPU-bbbb", None, 1)),
        30 << 30,  # 40 GiB less 14 in use, 4 of them file cache: below MemAvailable
        1 << 34,
    )


def test_cgroup_room_counts_file_cache_and_needs_memory_stat(tmp_path):
    linux_runtime(tmp_path)
    # A deeper, tighter limit whose memory is mostly file cache.
    write(tmp_path, "/sys/fs/cgroup/user.slice/run.scope/memory.max", f"{8 << 30}\n")
    write(tmp_path, "/sys/fs/cgroup/user.slice/run.scope/memory.current", f"{9 << 30}\n")
    stat = f"active_file {6 << 30}\ninactive_file {2 << 30}\n"
    write(tmp_path, "/sys/fs/cgroup/user.slice/run.scope/memory.stat", stat)
    assert read_runtime(host=RuntimeHost(tmp_path, affinity=(0,))).available_bytes == 7 << 30
    # More file cache than memory.current: nothing counts as in use.
    write(tmp_path, "/sys/fs/cgroup/user.slice/run.scope/memory.current", f"{1 << 30}\n")
    assert read_runtime(host=RuntimeHost(tmp_path, affinity=(0,))).available_bytes == 8 << 30
    # Over the limit in anonymous memory: no room, never negative.
    write(tmp_path, "/sys/fs/cgroup/user.slice/run.scope/memory.current", f"{20 << 30}\n")
    assert read_runtime(host=RuntimeHost(tmp_path, affinity=(0,))).available_bytes == 0
    # A bounded level without memory.stat gives no reading.
    (tmp_path / "sys/fs/cgroup/user.slice/run.scope/memory.stat").unlink()
    assert read_runtime(host=RuntimeHost(tmp_path, affinity=(0,))).available_bytes is None


def test_linux_runtime_respects_cuda_visible_devices(tmp_path):
    commands = linux_runtime(tmp_path)
    for value, expected in (("GPU-bb", ("GPU-bbbb",)), ("1,0", ("GPU-bbbb", "GPU-aaaa")), ("", ())):
        host = RuntimeHost(
            tmp_path,
            affinity=(0,),
            environment={"CUDA_VISIBLE_DEVICES": value},
            commands=commands,
        )
        gpus = read_runtime(host=host).gpus
        assert gpus is not None and tuple(gpu.uuid for gpu in gpus) == expected


def test_linux_runtime_without_or_with_a_failing_nvidia_smi(tmp_path):
    linux_runtime(tmp_path)
    assert read_runtime(host=RuntimeHost(tmp_path, affinity=(0,))).gpus == ()
    failing = {SMI_GPUS: Command(9, "", "Failed to initialize NVML: Driver/library mismatch")}
    assert read_runtime(host=RuntimeHost(tmp_path, affinity=(0,), commands=failing)).gpus is None
    absent = {SMI_GPUS: Command(6, "No devices were found\n")}
    assert read_runtime(host=RuntimeHost(tmp_path, affinity=(0,), commands=absent)).gpus == ()


def test_linux_runtime_without_affinity_uses_the_online_cpus(tmp_path):
    linux_runtime(tmp_path)
    write(tmp_path, "/sys/devices/system/cpu/online", "0-3\n")
    assert read_runtime(host=RuntimeHost(tmp_path)).cpus == (0, 1, 2, 3)


def test_darwin_runtime_facts(tmp_path):
    counters = {
        "hw.logicalcpu": 10,
        "vm.pagesize": 16384,
        "vm.page_free_count": 1000,
        "vm.vm_page_free_target": 400,
        "vm.page_pageable_external_count": 9000,
        "vm.vm_page_filecache_min": 4000,
        "vm.page_purgeable_count": 70,
        "vm.page_reusable_count": 30,
    }
    commands = {("sysctl", "-n", name): ok(f"{value}\n") for name, value in counters.items()}
    host = RuntimeHost(tmp_path, system="darwin", commands=commands)
    facts = read_runtime(host=host)
    # free above the free target, file pages above the file cache minimum, purgeable, reusable
    assert facts == RuntimeFacts(tuple(range(10)), (), (600 + 5000 + 70 + 30) * 16384, None)
    below = dict(commands)
    below[("sysctl", "-n", "vm.vm_page_free_target")] = ok("5000\n")
    below[("sysctl", "-n", "vm.vm_page_filecache_min")] = ok("12000\n")
    facts = read_runtime(host=RuntimeHost(tmp_path, system="darwin", commands=below))
    assert facts.available_bytes == (70 + 30) * 16384


@pytest.mark.parametrize(
    "value, expected",
    [
        (None, [(0, "GPU-a1"), (1, "GPU-b2")]),
        ("", []),
        ("-1", []),
        ("1", [(1, "GPU-b2")]),
        ("GPU-b", [(1, "GPU-b2")]),
        ("GPU-", []),  # ambiguous: CUDA stops at it
        ("0,7,1", [(0, "GPU-a1")]),  # an unknown device ends the list
        ("1,1", [(1, "GPU-b2")]),
    ],
)
def test_visible_gpus(value, expected):
    assert visible_gpus([(0, "GPU-a1"), (1, "GPU-b2")], value) == expected
