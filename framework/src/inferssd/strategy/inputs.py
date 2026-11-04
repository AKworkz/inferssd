# SPDX-License-Identifier: Apache-2.0
"""What a strategy is derived from besides the machine characteristics.

``ModelFacts`` come from the GGUF header of the model being launched.
``RuntimeFacts`` are read when the launch starts, because they change while the
machine runs: the CPUs this process may use, the CUDA GPUs it can use and
their free memory, the memory available now, and the user's memory budget.
Readers take an injectable host and never write anything.
"""

from __future__ import annotations

import os
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Optional, Protocol, cast

from inferssd.configuration.settings import cpus as cpu_list
from inferssd.contracts import ContractError
from inferssd.machine import identity
from inferssd.machine.characteristics import (
    COUNT,
    POSITIVE,
    TEXT,
    Check,
    array,
    nullable,
    record,
)
from inferssd.source import header as gguf_header
from inferssd.source.experts import ROLE, is_role, widened
from inferssd.source.files import LOCAL_FILES, SourceFiles

ROLES = ("gate", "up", "down")
# ggml type name -> (elements per block, bytes per block)
BLOCKS = {name: (block, size) for name, block, size in gguf_header.TYPES.values()}


@dataclass(frozen=True)
class Role:
    """One routed-expert role as the header describes it.

    A role is a matrix of ``rows`` output features by ``cols`` reduction
    columns per expert; ``formats`` counts the layers per ggml type, the most
    common first.
    """

    name: str
    formats: tuple[tuple[str, int], ...]
    rows: int
    cols: int

    @property
    def format(self) -> str:
        return self.formats[0][0]

    def slice_bytes(self, format: Optional[str] = None) -> int:
        """One expert's bytes of this role in FORMAT (default: the most common one)."""
        block, size = BLOCKS[format or self.format]
        return self.rows * (self.cols // block) * size


@dataclass(frozen=True)
class ModelFacts:
    """Model facts from the GGUF header; nothing here depends on the machine."""

    paths: tuple[str, ...]
    architecture: Optional[str]
    layers: int  # layers with routed experts
    experts: int  # routed experts per layer
    experts_used: Optional[int]  # routed experts per token
    gate: Role
    up: Role
    down: Role
    expert_bytes: int  # the largest expert, all three roles
    expert_read_bytes: int  # its three role reads, each widened to whole 4 KiB pages
    dense_bytes: int  # every tensor that is not a routed expert
    format_bytes: tuple[tuple[str, int], ...]  # routed bytes per ggml type, the most first

    @property
    def roles(self) -> tuple[Role, Role, Role]:
        return self.gate, self.up, self.down

    @property
    def routed_bytes(self) -> int:
        """Every routed expert's bytes, of every type."""
        return sum(size for _, size in self.format_bytes)

    @property
    def mean_expert_read_bytes(self) -> float:
        """What the engine reads for one routed expert on average.

        Layers may quantize their experts to different types, so the largest expert overstates
        the rest: this is the mean expert's bytes, widened to whole pages as the largest one's
        three role reads are. It equals expert_read_bytes when every layer has one type.
        """
        mean = self.routed_bytes / (self.layers * self.experts)
        return mean + self.expert_read_bytes - self.expert_bytes

    @property
    def dominant_format(self) -> str:
        """The ggml type that holds the most routed-expert bytes."""
        return self.format_bytes[0][0]


def _ranked(counts: Mapping[str, int]) -> tuple[tuple[str, int], ...]:
    return tuple(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


def facts_from_headers(headers: Sequence[gguf_header.Header]) -> ModelFacts:
    """ModelFacts of a model whose shards have HEADERS, in shard order."""
    if not headers:
        raise ValueError("a model needs at least one GGUF header")
    kv: dict[str, object] = {}
    for item in headers:
        for key, value in item.kv.items():
            kv.setdefault(key, value)
    tensors = [tensor for item in headers for tensor in item.tensors]
    routed = [tensor for tensor in tensors if is_role(tensor)]
    if not routed:
        raise ValueError(f"{headers[0].path}: the model has no routed expert tensors")
    counts = {tensor.shape[2] for tensor in routed}
    if len(counts) != 1:
        raise ValueError("routed expert tensors disagree on the expert count: " + str(counts))
    experts = counts.pop()
    formats: dict[str, Counter[str]] = {role: Counter() for role in ROLES}
    shapes: dict[str, Counter[tuple[int, int]]] = {role: Counter() for role in ROLES}
    per_format: Counter[str] = Counter()
    expert: Counter[tuple[int, int]] = Counter()
    read: Counter[tuple[int, int]] = Counter()
    for tensor in routed:
        match = cast(re.Match[str], ROLE.match(tensor.name))
        layer, role = int(match.group(1)), match.group(2)
        formats[role][tensor.type_name] += 1
        shapes[role][(tensor.shape[1], tensor.shape[0])] += 1
        per_format[tensor.type_name] += tensor.nbytes
        size = tensor.nbytes // experts
        for index in range(experts):
            expert[(layer, index)] += size
            read[(layer, index)] += widened(tensor.offset + index * size, size)
    missing = [role for role in ROLES if not formats[role]]
    if missing:
        raise ValueError("the model has no routed " + " or ".join(missing) + " expert tensors")
    roles = {}
    for role in ROLES:
        (rows, cols), _ = max(sorted(shapes[role].items()), key=lambda item: item[1])
        ranked = _ranked(formats[role])
        for name, _ in ranked:
            if name not in BLOCKS or cols % BLOCKS[name][0]:
                raise ValueError(f"routed {role} tensors of type {name} have {cols} columns")
        roles[role] = Role(role, ranked, rows, cols)
    architecture = kv.get("general.architecture")
    used = kv.get(f"{architecture}.expert_used_count") if isinstance(architecture, str) else None
    return ModelFacts(
        paths=tuple(item.path for item in headers),
        architecture=architecture if isinstance(architecture, str) else None,
        layers=len({layer for layer, _ in expert}),
        experts=experts,
        experts_used=used if isinstance(used, int) and not isinstance(used, bool) else None,
        gate=roles["gate"],
        up=roles["up"],
        down=roles["down"],
        expert_bytes=max(expert.values()),
        expert_read_bytes=max(read.values()),
        dense_bytes=sum(tensor.nbytes for tensor in tensors if not is_role(tensor)),
        format_bytes=_ranked(per_format),
    )


def model_facts(path: str, *, filesystem: SourceFiles = LOCAL_FILES) -> ModelFacts:
    """ModelFacts of the model PATH names; a split model's first shard brings all of them."""
    return facts_from_headers(gguf_header.read_model(path, filesystem=filesystem))


def shard_facts(paths: Sequence[str], *, filesystem: SourceFiles = LOCAL_FILES) -> ModelFacts:
    """ModelFacts of exactly the shards PATHS, in their order."""
    return facts_from_headers([gguf_header.read(path, filesystem=filesystem) for path in paths])


def _type_counts(value: object, field: str) -> None:
    """A nonempty list of [ggml type name, positive count] pairs."""
    if not isinstance(value, list) or not value:
        raise ContractError(f"{field} must be a nonempty array")
    for index, item in enumerate(value):
        if not isinstance(item, list) or len(item) != 2 or item[0] not in BLOCKS:
            raise ContractError(f"{field}[{index}] must be [a ggml type name, a count]")
        POSITIVE(item[1], f"{field}[{index}][1]")


PAIRS: Check = _type_counts
ROLE_RECORD = record({"formats": PAIRS, "rows": POSITIVE, "cols": POSITIVE})
MODEL_RECORD = record(
    {
        "paths": array(TEXT, minimum=1),
        "architecture": nullable(TEXT),
        "layers": POSITIVE,
        "experts": POSITIVE,
        "experts_used": nullable(POSITIVE),
        "roles": record({role: ROLE_RECORD for role in ROLES}),
        "expert_bytes": POSITIVE,
        "expert_read_bytes": POSITIVE,
        "dense_bytes": COUNT,
        "format_bytes": PAIRS,
    }
)


def _pairs(value: Sequence[tuple[str, int]]) -> list[list[object]]:
    return [[name, count] for name, count in value]


def model_facts_json(facts: ModelFacts) -> dict[str, object]:
    """FACTS as a JSON object; model_facts_from reads it back."""
    return {
        "paths": list(facts.paths),
        "architecture": facts.architecture,
        "layers": facts.layers,
        "experts": facts.experts,
        "experts_used": facts.experts_used,
        "roles": {
            role.name: {"formats": _pairs(role.formats), "rows": role.rows, "cols": role.cols}
            for role in facts.roles
        },
        "expert_bytes": facts.expert_bytes,
        "expert_read_bytes": facts.expert_read_bytes,
        "dense_bytes": facts.dense_bytes,
        "format_bytes": _pairs(facts.format_bytes),
    }


def _ranked_pairs(value: object) -> tuple[tuple[str, int], ...]:
    return tuple((name, count) for name, count in cast(list[tuple[str, int]], value))


def model_facts_from(value: object) -> ModelFacts:
    """ModelFacts from their JSON object; errors name the offending field."""
    MODEL_RECORD(value, "model_facts")
    item = cast(dict[str, object], value)
    roles = cast(dict[str, dict[str, object]], item["roles"])
    built = {
        name: Role(
            name,
            _ranked_pairs(roles[name]["formats"]),
            cast(int, roles[name]["rows"]),
            cast(int, roles[name]["cols"]),
        )
        for name in ROLES
    }
    for role in built.values():
        if any(role.cols % BLOCKS[name][0] for name, _ in role.formats):
            raise ContractError(f"model_facts.roles.{role.name}: cols fit no block of its types")
    return ModelFacts(
        paths=tuple(cast(list[str], item["paths"])),
        architecture=cast(Optional[str], item["architecture"]),
        layers=cast(int, item["layers"]),
        experts=cast(int, item["experts"]),
        experts_used=cast(Optional[int], item["experts_used"]),
        gate=built["gate"],
        up=built["up"],
        down=built["down"],
        expert_bytes=cast(int, item["expert_bytes"]),
        expert_read_bytes=cast(int, item["expert_read_bytes"]),
        dense_bytes=cast(int, item["dense_bytes"]),
        format_bytes=_ranked_pairs(item["format_bytes"]),
    )


def model_drive_key(paths: Sequence[str], host: Optional[identity.Host] = None) -> Optional[str]:
    """The storage key of the drive holding most of the model's bytes, or None when unknown."""
    live = host or identity.NativeHost()
    sizes: Counter[str] = Counter()
    keys: dict[tuple[int, int], str] = {}
    try:
        for path in paths:
            for name, size in live.regular_files(path):
                device = live.device(name)
                if device not in keys:
                    keys[device] = identity.drive_key(live, name)
                sizes[keys[device]] += size
    except (OSError, RuntimeError, ValueError):
        return None
    if not sizes:
        return None
    return max(sizes.items(), key=lambda item: item[1])[0]


@dataclass(frozen=True)
class GPUState:
    """A CUDA GPU the launch can use, as it is now."""

    uuid: str
    free_bytes: Optional[int]  # None where the driver reports no dedicated memory
    processes: int  # compute processes running on it


@dataclass(frozen=True)
class RuntimeFacts:
    """Facts read at launch; never taken from the characteristics file."""

    cpus: tuple[int, ...]  # CPUs this process may run on
    gpus: Optional[tuple[GPUState, ...]]  # usable CUDA GPUs; None when they cannot be read
    # The launcher's view of the memory available now: MemAvailable within the
    # room of every bounded cgroup (file cache counted as reclaimable), or on
    # macOS the free, inactive and speculative pages. None when it cannot be
    # read. The engine sizes its caches from its own reading at run time.
    available_bytes: Optional[int]
    memory_budget_bytes: Optional[int] = None  # the user's memory budget, when given
    # The GPU backends of the engine the launch starts, as its build names them; None when
    # unknown, and then taken to be the backends of the machine's GPUs (engine_gpu_backends).
    # Set by the launch, not recorded with the facts above.
    gpu_backends: Optional[tuple[str, ...]] = None


class Host(identity.Host, Protocol):
    def affinity(self) -> Optional[tuple[int, ...]]: ...
    def environment(self) -> Mapping[str, str]: ...


GPU_RECORD = record({"uuid": TEXT, "free_bytes": nullable(COUNT), "processes": COUNT})
RUNTIME_RECORD = record(
    {
        "cpus": array(COUNT, minimum=1),
        "gpus": nullable(array(GPU_RECORD)),
        "available_bytes": nullable(COUNT),
    },
    {"memory_budget_bytes": nullable(POSITIVE)},
)


def runtime_facts_json(facts: RuntimeFacts) -> dict[str, object]:
    """FACTS as a JSON object; runtime_facts_from reads it back."""
    return {
        "cpus": list(facts.cpus),
        "gpus": None
        if facts.gpus is None
        else [
            {"uuid": gpu.uuid, "free_bytes": gpu.free_bytes, "processes": gpu.processes}
            for gpu in facts.gpus
        ],
        "available_bytes": facts.available_bytes,
        "memory_budget_bytes": facts.memory_budget_bytes,
    }


def runtime_facts_from(value: object) -> RuntimeFacts:
    """RuntimeFacts from their JSON object; errors name the offending field."""
    RUNTIME_RECORD(value, "runtime_facts")
    item = cast(dict[str, object], value)
    cpus = cast(list[int], item["cpus"])
    if len(set(cpus)) != len(cpus):
        raise ContractError("runtime_facts.cpus must not repeat a CPU")
    gpus = cast(Optional[list[dict[str, object]]], item["gpus"])
    return RuntimeFacts(
        tuple(cpus),
        None
        if gpus is None
        else tuple(
            GPUState(
                cast(str, gpu["uuid"]),
                cast(Optional[int], gpu["free_bytes"]),
                cast(int, gpu["processes"]),
            )
            for gpu in gpus
        ),
        cast(Optional[int], item["available_bytes"]),
        cast(Optional[int], item.get("memory_budget_bytes")),
    )


class NativeHost(identity.NativeHost):
    """This host; ENVIRONMENT, the launch's (default: this process's), selects the CUDA GPUs."""

    def __init__(self, environment: Optional[Mapping[str, str]] = None) -> None:
        self.env = environment

    def affinity(self) -> Optional[tuple[int, ...]]:
        if not hasattr(os, "sched_getaffinity"):
            return None
        return tuple(sorted(os.sched_getaffinity(0)))

    def environment(self) -> Mapping[str, str]:
        return os.environ if self.env is None else self.env


def visible_gpus(listed: Sequence[tuple[int, str]], value: Optional[str]) -> list[tuple[int, str]]:
    """The (index, UUID) pairs CUDA_VISIBLE_DEVICES=VALUE leaves, in its order.

    Like CUDA, the list ends at its first entry that names no GPU or one again.
    Numbers are taken as nvidia-smi indices, which follow the PCI bus; CUDA
    numbers devices fastest first unless CUDA_DEVICE_ORDER=PCI_BUS_ID, so on a
    machine with different GPUs a UUID is the unambiguous choice.
    """
    if value is None:
        return list(listed)
    chosen: list[tuple[int, str]] = []
    for token in (item.strip() for item in value.split(",")):
        if token.isdigit():
            found = [gpu for gpu in listed if gpu[0] == int(token)]
        elif token.startswith("GPU-"):
            found = [gpu for gpu in listed if gpu[1].startswith(token)]
        else:
            found = []
        if len(found) != 1 or found[0] in chosen:
            break
        chosen.append(found[0])
    return chosen


def _gpus(host: Host) -> Optional[tuple[GPUState, ...]]:
    if not host.which("nvidia-smi"):
        return ()
    listed = host.command(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,memory.free",
            "--format=csv,noheader,nounits",
        ]
    )
    if listed.returncode:
        if "No devices were found" in listed.stdout + listed.stderr:
            return ()
        return None
    rows: list[tuple[int, str, Optional[int]]] = []
    for line in listed.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 3 or not fields[0].isdigit():
            continue
        free = int(fields[2]) * 1048576 if fields[2].isdigit() else None
        rows.append((int(fields[0]), fields[1], free))
    apps = host.command(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"]
    )
    if apps.returncode:
        return None
    busy = Counter(line.split(",")[0].strip() for line in apps.stdout.splitlines() if line.strip())
    memory = {uuid: value for _, uuid, value in rows}
    chosen = visible_gpus(
        [(index, uuid) for index, uuid, _ in rows],
        host.environment().get("CUDA_VISIBLE_DEVICES"),
    )
    return tuple(GPUState(uuid, memory[uuid], busy[uuid]) for _, uuid in chosen)


def _cgroup_room(host: Host) -> tuple[bool, Optional[int]]:
    """Whether the cgroup v2 limits could be read, and the room the tightest one leaves.

    At each ancestor with a finite memory.max the room is that limit less
    memory.current, where the file cache (active_file and inactive_file in
    memory.stat) counts as reclaimable, as the engine's host ledger counts it.
    A bounded level without memory.current or memory.stat gives no reading.
    """
    try:
        line = next(
            (row for row in host.text("/proc/self/cgroup").splitlines() if row.startswith("0::")),
            None,
        )
    except OSError:
        return True, None
    if line is None:
        return True, None
    path = line[3:].strip().rstrip("/")
    room: Optional[int] = None
    while True:
        base = "/sys/fs/cgroup" + path
        limit = identity.read(host, base + "/memory.max")
        if limit and limit.isdigit():
            used = identity.read(host, base + "/memory.current")
            stat = identity.read(host, base + "/memory.stat")
            fields = dict(row.split(None, 1) for row in (stat or "").splitlines() if " " in row)
            file = [fields.get(name, "").strip() for name in ("active_file", "inactive_file")]
            if not used or not used.isdigit() or not all(value.isdigit() for value in file):
                return False, None
            anonymous = max(0, int(used) - sum(int(value) for value in file))
            value = max(0, int(limit) - anonymous)
            room = value if room is None else min(room, value)
        if not path:
            return True, room
        path = path.rsplit("/", 1)[0]


def _available(host: Host, system: str) -> Optional[int]:
    if system == "linux":
        try:
            value: Optional[int] = identity.meminfo_bytes(host, "MemAvailable")
        except (OSError, ValueError):
            value = None
        readable, room = _cgroup_room(host)
        if not readable:
            return None
        if room is not None:
            value = room if value is None else min(value, room)
        return value
    if system == "darwin":
        text = identity.output(host, ["vm_stat"]) or ""
        page = re.search(r"page size of (\d+) bytes", text)
        pages = dict(re.findall(r"^Pages (free|inactive|speculative):\s+(\d+)\.", text, re.M))
        if not page or len(pages) != 3:
            return None
        return sum(int(count) for count in pages.values()) * int(page.group(1))
    return None


def read_runtime(
    *, memory_budget_bytes: Optional[int] = None, host: Optional[Host] = None
) -> RuntimeFacts:
    """Read this machine's runtime facts now (Linux and macOS)."""
    live: Host = host or NativeHost()
    system = live.platform()
    if system == "linux":
        allowed = live.affinity()
        cpus = allowed if allowed is not None else identity.online_cpus(live)
        gpus = _gpus(live)
    elif system == "darwin":
        # macOS has no affinity: every logical CPU is usable.
        cpus = cpu_list(f"0-{identity.sysctl_int(live, 'hw.logicalcpu') - 1}")
        gpus = ()
    else:
        raise identity.UnsupportedPlatform(f"runtime facts are not read on {system} yet")
    return RuntimeFacts(tuple(cpus), gpus, _available(live, system), memory_budget_bytes)
