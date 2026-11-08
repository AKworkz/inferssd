"""Explicit execution policy, separate from model or planner calculations."""

from __future__ import annotations

import errno
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Optional, cast

from inferssd.configuration.settings import PLATFORMS

# Conditions a measurement declares. Left out, they are not checked: the engine
# alone decides how much memory to use.
OPTIONAL = (
    "host_reserve_bytes",
    "max_compressor_growth_bytes",
    "max_swap_growth_bytes",
    "runtime_seconds",
)


@dataclass(frozen=True)
class Policy:
    platform: str
    lock_path: str
    stop_path: str
    memory_budget_bytes: int
    host_reserve_bytes: Optional[int]
    minimum_memlock_bytes: int
    max_compressor_growth_bytes: Optional[int]
    max_swap_growth_bytes: Optional[int]
    runtime_seconds: Optional[int]
    cleanup_seconds: int

    @classmethod
    def read(cls, data: Mapping[str, object]) -> Policy:
        expected = set(cls.__dataclass_fields__)
        differ = (set(data) - expected) | (expected - set(OPTIONAL) - set(data))
        if differ:
            raise ValueError("runtime policy fields differ: " + str(sorted(differ)))
        for name in ("platform", "lock_path", "stop_path"):
            if not isinstance(data[name], str) or not data[name]:
                raise ValueError("runtime policy " + name + " must be nonempty text")
        if data["platform"] not in PLATFORMS:
            raise ValueError("runtime policy platform must be one of " + ", ".join(PLATFORMS))
        numeric = expected - {"platform", "lock_path", "stop_path"}
        for name in numeric:
            value = data.get(name)
            if value is None and name in OPTIONAL:
                continue
            minimum = (
                1 if name in ("memory_budget_bytes", "runtime_seconds", "cleanup_seconds") else 0
            )
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"runtime policy {name} must be an integer >= {minimum}")
        return cls(
            platform=str(data["platform"]),
            lock_path=str(data["lock_path"]),
            stop_path=str(data["stop_path"]),
            memory_budget_bytes=cast(int, data["memory_budget_bytes"]),
            host_reserve_bytes=cast(Optional[int], data.get("host_reserve_bytes")),
            minimum_memlock_bytes=cast(int, data["minimum_memlock_bytes"]),
            max_compressor_growth_bytes=cast(
                Optional[int], data.get("max_compressor_growth_bytes")
            ),
            max_swap_growth_bytes=cast(Optional[int], data.get("max_swap_growth_bytes")),
            runtime_seconds=cast(Optional[int], data.get("runtime_seconds")),
            cleanup_seconds=cast(int, data["cleanup_seconds"]),
        )


def lock_held(path: str) -> BlockingIOError:
    """The refusal of a launch whose device lock another owner holds."""
    return BlockingIOError(
        errno.EWOULDBLOCK,
        f"the device lock {path} is held by another owned launch, or by one whose cleanup is "
        "not confirmed yet; its run directory's guardian-quarantine.json or "
        "worker-quarantine.json says why",
    )
