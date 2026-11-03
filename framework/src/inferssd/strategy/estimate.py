# SPDX-License-Identifier: Apache-2.0
"""A rough decode speed of a model on this machine, before the model has run.

A token's time is the sum of three terms, each from this machine's profile and
the model's GGUF header:

- the dense layers read their weights once, at the fastest measured memory read
  rate, and the part that does not fit in memory at the drive's rate; the
  lookup tables a token reads one row of do not count;
- every routed expert costs the CPU what the launch strategy prices one route
  at (``cpu_line``);
- the experts the RAM cache misses are read from the model's drive at the
  strategy's rate for the engine's reads (``drive_rate``).

Memory available now, less the host margin a launch keeps, holds the dense
weights first; the RAM cache gets the rest, and a token finds an expert there as
often as the share of all experts that fits, as if routing were uniform.
Nothing the profile does not measure enters: the GPU's own time, prefetch and
the skew of real routing. So this is a guide, not a prediction; the speed a
model reaches on the machine replaces it once it has run.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from inferssd.machine.characteristics import Characteristics
from inferssd.strategy.derive import cpu_line, drive_rate, host_headroom_mib, placement
from inferssd.strategy.inputs import ModelFacts, RuntimeFacts

GB = 1e9
ENGINE_RESERVE_BYTES = 512 << 20  # the host margin the engine keeps without host.headroom_mib


@dataclass(frozen=True)
class Estimate:
    tokens_per_second: Optional[float]  # None when a term cannot be priced
    detail: str  # the terms, or why there is no estimate


def headroom_bytes(characteristics: Characteristics, runtime: RuntimeFacts, platform: str) -> int:
    """The host margin a launch keeps free: the one it sets, or the engine's own."""
    mib = host_headroom_mib(characteristics, runtime, platform)
    return ENGINE_RESERVE_BYTES if mib is None else mib << 20


def decode_estimate(
    characteristics: Characteristics,
    model: ModelFacts,
    dense_read_bytes: int,
    runtime: RuntimeFacts,
    platform: str,
    drive_key: Optional[str],
) -> Estimate:
    """MODEL's estimated decode tokens per second when its shards are on DRIVE_KEY."""
    if not model.experts_used:
        return Estimate(None, "the GGUF header names no routed experts per token")
    memory_Bps = max((row["Bps"] for row in characteristics["memory"]["dram_read_Bps"]), default=0)
    if memory_Bps <= 0:
        return Estimate(None, "no memory read rate in the profile")
    try:
        threads, _, _ = placement(characteristics, model, runtime, [])
    except ValueError as error:  # the CPUs this process may use are not the profiled ones
        return Estimate(None, str(error))
    # The line a launch prices a route at, on the batch pool (derive's place.costs).
    line, why = cpu_line(characteristics, model, threads.batch or threads.decode or 1)
    if line is None:
        return Estimate(None, why)
    drive_gbps, why = drive_rate(characteristics, model, drive_key)
    if drive_gbps is None:
        return Estimate(None, why)
    available = runtime.available_bytes
    if available is None:
        available = characteristics["memory"]["total_bytes"]
    room = max(0, available - headroom_bytes(characteristics, runtime, platform))
    resident = min(dense_read_bytes, room)
    cache = room - resident
    experts = model.routed_bytes
    hit = min(1.0, cache / experts)
    routes = model.layers * model.experts_used
    dense_s = resident / memory_Bps + (dense_read_bytes - resident) / (drive_gbps * GB)
    cpu_s = routes * (line[0] + line[1]) / 1e6
    drive_s = routes * (1.0 - hit) * model.mean_expert_read_bytes / (drive_gbps * GB)
    seconds = dense_s + cpu_s + drive_s
    spill = dense_read_bytes - resident
    return Estimate(
        1.0 / seconds,
        f"{seconds * 1e3:.0f} ms a token: dense {dense_s * 1e3:.0f} ms"
        + (f" ({spill / GB:.1f} GB of it from the drive)" if spill else "")
        + f", {routes} experts on the CPU {cpu_s * 1e3:.0f} ms, "
        f"{1 - hit:.0%} of them from the drive at {drive_gbps:.2f} GB/s {drive_s * 1e3:.0f} ms "
        f"(a {cache / GB:.1f} GB cache of {experts / GB:.0f} GB of experts)",
    )
