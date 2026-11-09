"""Concrete bounded-resource checks used only by an explicitly started worker.

Linux observes a hard cgroup envelope and OOM counters. Darwin observes sampled
process RSS and system memory/compression/swap; it cannot promise a hard cap.
"""

from __future__ import annotations

import re
import socket
import struct
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Optional

from inferssd.platform.captured_command import run_captured
from inferssd.platform.policy import Policy

if sys.platform != "win32":
    import resource


def command(argv: Sequence[str]) -> str:
    value = run_captured(argv, timeout=10)
    if value.returncode:
        raise RuntimeError(
            f"resource command failed ({value.returncode}): {argv[0]}: {value.stderr}"
        )
    return value.stdout


def scope_pids(root: Path) -> set[int]:
    found = {int(x) for x in (root / "cgroup.procs").read_text().split()}
    for path in root.rglob("cgroup.procs"):
        found.update(int(x) for x in path.read_text().split())
    return found


class Resources:
    def __init__(self, policy: Policy) -> None:
        self.policy = policy
        self.scope: Optional[Path] = None
        self.baseline: Optional[dict[str, Any]] = None
        self.memlock: Optional[tuple[int, int]] = None

    def prepare(self) -> dict[str, Any]:
        if sys.platform != self.policy.platform:
            raise ValueError("execution platform differs from runtime policy")
        soft, hard = resource.getrlimit(resource.RLIMIT_MEMLOCK)
        if soft != resource.RLIM_INFINITY and soft < self.policy.minimum_memlock_bytes:
            raise ValueError("RLIMIT_MEMLOCK is below minimum_memlock_bytes; set it before launch")
        self.memlock = soft, hard
        first = self.sample(None)
        self.baseline = first
        return dict(
            first,
            memlock={
                "soft": soft,
                "hard": hard,
                "minimum_bytes": self.policy.minimum_memlock_bytes,
            },
        )

    def sample(self, pid: Optional[int]) -> dict[str, Any]:
        if self.policy.platform == "linux":
            value = self.linux(pid)
        else:
            value = self.darwin(pid)
        reserve = self.policy.host_reserve_bytes
        if reserve is not None and value["available_bytes"] < reserve:
            raise RuntimeError("host available-memory estimate is below reserve")
        return value

    def linux(self, pid: Optional[int]) -> dict[str, Any]:
        mount = Path("/sys/fs/cgroup")
        raw = Path("/proc/self/cgroup").read_text()
        paths = [x[3:] for x in raw.splitlines() if x.startswith("0::")]
        if len(paths) != 1 or ".." in Path(paths[0]).parts:
            raise ValueError("one visible cgroup v2 scope is required")
        scope = mount / paths[0].lstrip("/")
        if self.scope is not None and scope != self.scope:
            raise RuntimeError("worker resource scope changed")
        self.scope = scope
        current, ancestors = scope, []
        while True:
            row: dict[str, Any] = {"path": str(current)}
            for name in (
                "memory.max",
                "memory.swap.max",
                "memory.current",
                "memory.peak",
                "memory.events",
            ):
                path = current / name
                if path.exists():
                    row[name] = path.read_text().strip()
            ancestors.append(row)
            if current == mount:
                break
            current = current.parent
        limits = [int(r["memory.max"]) for r in ancestors if r.get("memory.max", "max") != "max"]
        swap = [
            int(r["memory.swap.max"]) for r in ancestors if r.get("memory.swap.max", "max") != "max"
        ]
        if (
            not limits
            or min(limits) != self.policy.memory_budget_bytes
            or not swap
            or min(swap) != 0
        ):
            raise RuntimeError("actual ancestor memory budget or zero-swap contract differs")
        if (
            ancestors[0].get("memory.max") != str(self.policy.memory_budget_bytes)
            or ancestors[0].get("memory.swap.max") != "0"
            or "memory.peak" not in ancestors[0]
        ):
            raise RuntimeError(
                "actual worker scope must carry the hard memory budget and zero swap"
            )
        for row in ancestors:
            if "memory.max" not in row:
                continue
            counters = dict(line.split() for line in row.get("memory.events", "").splitlines())
            if not {"oom", "oom_kill"} <= counters.keys():
                raise RuntimeError("ancestor OOM counters unavailable")
        if self.baseline is not None:
            before = self.baseline["ancestors"]
            if len(before) != len(ancestors):
                raise RuntimeError("resource ancestors changed")
            for left, right in zip(before, ancestors):
                a = dict(line.split() for line in left.get("memory.events", "").splitlines())
                b = dict(line.split() for line in right.get("memory.events", "").splitlines())
                if left["path"] != right["path"] or any(
                    a.get(k, "0") != b.get(k, "0") for k in ("oom", "oom_kill", "oom_group_kill")
                ):
                    raise RuntimeError("resource OOM counters changed")
        if pid is not None:
            if Path(f"/proc/{pid}/cgroup").read_text() != raw:
                raise RuntimeError("server escaped its owned resource scope")
            text = Path(f"/proc/{pid}/limits").read_text()
            match = re.search(r"^Max locked memory\s+(\S+)\s+(\S+)\s+bytes\s*$", text, re.M)
            if (
                match is None
                or tuple(
                    resource.RLIM_INFINITY if x == "unlimited" else int(x) for x in match.groups()
                )
                != self.memlock
            ):
                raise RuntimeError("server memlock inheritance differs")
        memory = dict(
            (line.split()[0], int(line.split()[1]))
            for line in Path("/proc/meminfo").read_text().splitlines()
            if len(line.split()) >= 2
        )
        return {
            "enforcement": "cgroup-v2 hard limit; zero swap",
            "scope": str(scope),
            "ancestors": ancestors,
            "available_bytes": memory["MemAvailable:"] * 1024,
        }

    def darwin(self, pid: Optional[int]) -> dict[str, Any]:
        vm, swap = command(["vm_stat"]), command(["sysctl", "-n", "vm.swapusage"])
        size = re.search(r"page size of (\d+) bytes", vm)
        used = re.search(r"used\s*=\s*([0-9.]+)([KMG])", swap)
        if size is None or used is None:
            raise RuntimeError("Darwin memory counters unavailable")
        pages = {name.strip(): int(n) for name, n in re.findall(r"^([^:\n]+):\s*(\d+)\.", vm, re.M)}
        rss = int(command(["ps", "-o", "rss=", "-p", str(pid)]).strip()) * 1024 if pid else 0
        row: dict[str, Any] = {
            "enforcement": "sampled soft RSS/system thresholds; no hard memory cap",
            "available_bytes": (pages["Pages free"] + pages["Pages inactive"]) * int(size[1]),
            "compressor_bytes": pages["Pages occupied by compressor"] * int(size[1]),
            "swap_bytes": int(float(used[1]) * {"K": 1024, "M": 1024**2, "G": 1024**3}[used[2]]),
            "rss_bytes": rss,
            "raw_vm_stat": vm,
            "raw_swap": swap,
        }
        if rss > self.policy.memory_budget_bytes:
            raise RuntimeError("sampled server RSS exceeded soft memory budget")
        if self.baseline is not None:
            compressor_limit = self.policy.max_compressor_growth_bytes
            swap_limit = self.policy.max_swap_growth_bytes
            if (
                compressor_limit is not None
                and int(row["compressor_bytes"]) - self.baseline["compressor_bytes"]
                > compressor_limit
            ):
                raise RuntimeError("sampled compressor growth exceeded policy")
            if (
                swap_limit is not None
                and int(row["swap_bytes"]) - self.baseline["swap_bytes"] > swap_limit
            ):
                raise RuntimeError("sampled swap growth exceeded policy")
        return row

    def confirm_listener(self, pid: int, host: str, port: int) -> None:
        """PID listens on IPv4 HOST at PORT; 0.0.0.0 is every interface."""
        if self.policy.platform == "darwin":
            raw = command(
                [
                    "/usr/sbin/lsof",
                    "-nP",
                    "-a",
                    "-p",
                    str(pid),
                    "-iTCP:" + str(port),
                    "-sTCP:LISTEN",
                    "-F0pn",
                ]
            )
            fields = [s.lstrip("\n") for s in raw.split("\0") if s.lstrip("\n")]
            # lsof names a listener on every interface *.
            name = "n" + ("*" if host == "0.0.0.0" else host) + ":" + str(port)
            if f"p{pid}" not in fields or name not in fields:
                raise RuntimeError("health listener is not owned by the exact server PID")
            return
        # /proc/net/tcp shows the address as its 32-bit word in hex, in memory order.
        word = struct.unpack("=I", socket.inet_aton(host))[0]
        address = f"{word:08X}:{port:04X}"
        sockets = {
            p.readlink().as_posix()[8:-1]
            for p in Path(f"/proc/{pid}/fd").iterdir()
            if p.is_symlink() and str(p.readlink()).startswith("socket:[")
        }
        for line in Path(f"/proc/{pid}/net/tcp").read_text().splitlines()[1:]:
            fields = line.split()
            if fields[1] == address and fields[3] == "0A" and fields[9] in sockets:
                return
        raise RuntimeError("health listener is not owned by the exact server PID")
