# SPDX-License-Identifier: Apache-2.0
"""Windows bounded-resource checks, used only by an explicitly started worker.

The worker runs in its guardian's job, whose commit limit is the policy's
memory budget plus the dedicated memory of the discrete GPUs (commit_limit):
the kernel refuses commit beyond it, as a cgroup's memory.max does. WDDM
charges a device allocation to commit without taking host memory, so the
budget alone would count the server's GPU memory as host memory; the engine
measures its host memory without it. Paging-file use and the memory
compression store are system-wide, so their growth since launch is sampled
against the policy, as on Darwin.
"""

from __future__ import annotations

import os
from typing import Any, Optional

from inferssd.configuration.settings import host_platform
from inferssd.platform.policy import Policy
from inferssd.platform.windows import win32

# The system process whose working set is the memory compression store.
COMPRESSION = "Memory Compression"


def commit_limit(policy: Policy) -> int:
    """The job's commit limit: the memory budget and room for the discrete GPUs' memory."""
    return policy.memory_budget_bytes + win32.discrete_video_memory()


class Resources:
    def __init__(self, policy: Policy) -> None:
        self.policy = policy
        self.limit: Optional[int] = None
        # Owned processes cannot leave their jobs, so the worker's group proves
        # its server gone and the guardian proves this scope empty.
        self.scope: Optional[Any] = None
        self.baseline: Optional[dict[str, Any]] = None

    def prepare(self) -> dict[str, Any]:
        if host_platform() != self.policy.platform:
            raise ValueError("execution platform differs from runtime policy")
        if self.policy.minimum_memlock_bytes:
            raise ValueError(
                "Windows sets no locked-memory limit to check; set minimum_memlock_bytes to 0"
            )
        self.limit = commit_limit(self.policy)
        first = self.sample(None)
        self.baseline = first
        return first

    def compressor_bytes(self) -> int:
        pid = win32.find_process(COMPRESSION)
        return win32.process_working_set(COMPRESSION, pid) if pid is not None else 0

    def sample(self, pid: Optional[int]) -> dict[str, Any]:
        job = win32.job_memory(None)
        if job["job_memory_limit"] != self.limit:
            raise RuntimeError("actual worker job must carry the hard memory budget")
        members = win32.job_processes(None)
        if os.getpid() not in members:
            raise RuntimeError("worker resource scope changed")
        system = win32.system_memory()
        row: dict[str, Any] = {
            "enforcement": "job object hard commit limit; sampled paging file and compression",
            "job": dict(job, processes=len(members)),
            "available_bytes": system["physical_available_bytes"],
            "system": system,
            "compressor_bytes": self.compressor_bytes(),
            "swap_bytes": win32.pagefile_in_use(),
        }
        if pid is not None:
            if pid not in members:
                raise RuntimeError("server escaped its owned resource scope")
            row["server"] = win32.process_memory(pid)
        reserve = self.policy.host_reserve_bytes
        if reserve is not None and row["available_bytes"] < reserve:
            raise RuntimeError("host available-memory estimate is below reserve")
        if self.baseline is not None:
            compressor_limit = self.policy.max_compressor_growth_bytes
            swap_limit = self.policy.max_swap_growth_bytes
            if (
                compressor_limit is not None
                and row["compressor_bytes"] - self.baseline["compressor_bytes"] > compressor_limit
            ):
                raise RuntimeError("sampled compressor growth exceeded policy")
            if (
                swap_limit is not None
                and row["swap_bytes"] - self.baseline["swap_bytes"] > swap_limit
            ):
                raise RuntimeError("sampled swap growth exceeded policy")
        return row

    def confirm_listener(self, pid: int, port: int) -> None:
        if ("127.0.0.1", port, pid) not in win32.tcp_listeners():
            raise RuntimeError("health listener is not owned by the exact server PID")
