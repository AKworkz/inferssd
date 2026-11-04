# SPDX-License-Identifier: Apache-2.0
"""Derive one launch's native settings, thread pools and server arguments.

``derive`` is a pure function of the machine characteristics, the model's
header facts, the runtime facts and the platform. It chooses thread counts and
CPU placement, placement cost priors, the host memory budget and margin, and,
with a CUDA GPU on Linux, which prompt passes the GPU graph prefills and that a
prompt is prefilled in one pass. The last two change the computation (GPU
against CPU kernels, the pass shape) for speed; everything else keeps it
bit-exact. The VRAM budget's behaviour stays as the engine has it.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Optional, cast

from inferssd.configuration.settings import PLATFORMS, Setting, Snapshot, parse, render, uncomment
from inferssd.launch.types import Threads
from inferssd.machine.characteristics import GPU, Characteristics, ComputeRow, dumps
from inferssd.strategy.inputs import ModelFacts, RuntimeFacts

# The engine's GB/s cost terms are 1e9 bytes per second: it prices a copy or a
# read at 1e3 / gbps microseconds per 1e6 bytes.
GB = 1e9
COST_MAX = 1e6  # the registry's bound on every place.costs term
MAX_THREADS = 256  # the launcher's bound on a thread pool
MAX_CPU = 511  # the highest CPU a server CPU mask holds
# Role reads the engine keeps in flight: 8 ring experts of 3 roles each.
IN_FLIGHT = 24
# One context checkpoint: a conversation's next turn most often returns to the
# end of the previous prompt. The engine makes room for it from the RAM expert
# cache, and the server's default of 32 would hold far more memory.
SERVER_ARGS = ("--ctx-checkpoints", "1")
# The host margin on memory a GPU shares with the host: the smallest margin that
# held under stress with the automatic VRAM tier growing into host memory.
UNIFIED_HEADROOM_MIB = 4096
# VRAM left free when other processes already use the GPU at launch, for them to
# grow into. A GPU this launch has to itself keeps nothing back: that memory would
# only cost expert slots.
SHARED_GPU_RESERVE_MIB = 1024
# On macOS the share of memory kept out of the RAM expert cache: macOS compresses
# a cache's cold pages long before free memory runs out, and every hit on a
# compressed slot then decompresses it.
DARWIN_HEADROOM_SHARE = 4
# The longest prompt pass the prefill route is priced for.
MAX_PASS = 1 << 20


@dataclass(frozen=True)
class Choice:
    """A proposed setting: its literal as a settings file takes it, and why."""

    value_text: str
    reason: str


@dataclass(frozen=True)
class Inputs:
    characteristics_sha256: str
    model_paths: tuple[str, ...]
    platform: str


@dataclass(frozen=True)
class Strategy:
    settings: Mapping[str, Choice]
    threads: Threads
    server_args: tuple[str, ...]
    inputs: Inputs
    notes: tuple[str, ...]
    # Prefill each prompt in one pass: -b and -ub as large as the context.
    one_pass: bool = False


def span(cpus: Sequence[int]) -> str:
    """CPU IDs as ascending ranges, such as 0-4,10; "none" for no CPU."""
    runs: list[list[int]] = []
    for cpu in sorted(cpus):
        if runs and runs[-1][1] + 1 == cpu:
            runs[-1][1] = cpu
        else:
            runs.append([cpu, cpu])
    return ",".join(str(a) if a == b else f"{a}-{b}" for a, b in runs) or "none"


def literal(item: Setting, platform: str) -> str:
    """ITEM's value as render() writes it, without key or receipt."""
    body, _ = uncomment(render(Snapshot((item,), platform)).split("\n")[1])
    return body.partition("=")[2].strip(" \t\r\n")


def engine_gpu_backends(
    known: Optional[Sequence[str]], characteristics: Optional[Characteristics]
) -> frozenset[str]:
    """The launch engine's GPU backends: KNOWN, else those of the machine's GPUs, else none."""
    if known is not None:
        return frozenset(known)
    if characteristics is None:
        return frozenset()
    return frozenset(gpu["backend"] for gpu in characteristics["identity"]["hard"]["gpus"])


def _validated(choices: Mapping[str, tuple[str, str]], platform: str) -> dict[str, Choice]:
    """Each choice parsed by the configuration layer and written back in its canonical form."""
    for _, reason in choices.values():
        if "\n" in reason or "\r" in reason:
            raise ValueError("a strategy reason must be one line")
    checked = parse(
        None,
        "strategy",
        overrides=[(key, value) for key, (value, _) in choices.items()],
        platform=platform,
    )
    return {
        key: Choice(literal(checked.at(key), platform), why) for key, (_, why) in choices.items()
    }


def _performance(characteristics: Characteristics) -> set[int]:
    """The fastest cluster's CPUs; every CPU on a uniform machine."""
    clusters = characteristics["cpu"]["clusters"]
    fast = [row for row in clusters if row["kind"] == "performance"]
    return set((fast or clusters)[0]["cpus"]) if clusters else set()


def _unpinned(characteristics: Characteristics) -> list[ComputeRow]:
    """The compute rows measured with threads the scheduler places."""
    return [row for row in characteristics["compute"]["cpu"] if not row.get("cpus")]


def _measured_counts(
    characteristics: Characteristics, model: ModelFacts, allowed: Sequence[int], notes: list[str]
) -> Threads:
    """Without affinity: the measured thread counts that run the dominant format fastest."""
    name = model.dominant_format.upper()
    rows = [
        row
        for row in _unpinned(characteristics)
        if row["format"].upper() == name and row["threads"] <= len(allowed)
    ]
    one = [
        (point["weight_Bps"], row["threads"])
        for row in rows
        for point in row["points"]
        if point["tokens"] == 1 and point.get("weight_Bps", 0) > 0
    ]
    tokens = max(
        (p["tokens"] for row in rows for p in row["points"] if p["tokens"] > 1 and p.get("flops")),
        default=0,
    )
    many = [
        (point["flops"], row["threads"])
        for row in rows
        for point in row["points"]
        if point["tokens"] == tokens and point.get("flops", 0) > 0
    ]
    fallback = len(_performance(characteristics) & set(allowed)) or len(allowed)
    fallback = min(fallback, MAX_THREADS)
    # The fastest rate wins; between equal rates, fewer threads.
    decode = max(one, key=lambda item: (item[0], -item[1]))[1] if one else fallback
    batch = max(many, key=lambda item: (item[0], -item[1]))[1] if many else fallback
    if one:
        notes.append(
            f"threads: decode {decode}, the measured count with the most {name} weight bytes/s "
            "at one token"
        )
    else:
        notes.append(f"threads: decode {decode}, the performance cluster ({name} not measured)")
    if many:
        notes.append(
            f"threads: batch {batch}, the measured count with the most {name} FLOP/s "
            f"at {tokens} tokens"
        )
    else:
        notes.append(f"threads: batch {batch}, the performance cluster ({name} not measured)")
    return Threads(min(decode, MAX_THREADS), min(batch, MAX_THREADS), ())


def pinned_rate(
    characteristics: Characteristics, model: ModelFacts, cpus: Sequence[int]
) -> Optional[float]:
    """The dominant format's weight bytes/s at one token, one thread pinned to each of CPUS."""
    name, want = model.dominant_format.upper(), sorted(cpus)
    for row in characteristics["compute"]["cpu"]:
        if row["format"].upper() == name and sorted(row.get("cpus", ())) == want:
            for point in row["points"]:
                if point["tokens"] == 1 and point.get("weight_Bps", 0) > 0:
                    return point["weight_Bps"]
    return None


def placement(
    characteristics: Characteristics, model: ModelFacts, runtime: RuntimeFacts, notes: list[str]
) -> tuple[Threads, tuple[int, ...], str]:
    """The thread pools, the I/O CPUs and the reason for both.

    Decode streams each expert's weights once a token, so it runs where one
    token streams them fastest: on the performance cluster alone, pinned one
    thread a CPU, when the profile measured that faster than every CPU;
    otherwise on every allowed CPU but one, unpinned. A prompt pass computes,
    so the batch pool always takes every allowed CPU but one, unpinned.
    """
    logical = {row["id"] for row in characteristics["cpu"]["logical"]}
    allowed = sorted(set(runtime.cpus) & logical)
    if not allowed:
        raise ValueError("none of the CPUs this process may use is in the machine characteristics")
    if not characteristics["cpu"]["affinity_supported"]:
        threads = _measured_counts(characteristics, model, allowed, notes)
        return threads, (), "no I/O CPU: threads cannot be pinned on this platform"
    performance = _performance(characteristics)
    fast = [cpu for cpu in allowed if cpu in performance]
    slow = [cpu for cpu in allowed if cpu not in performance]
    count = min(MAX_THREADS, max(1, len(allowed) - 1))
    shape = f"{len(fast)} performance and {len(slow)} other" if fast and slow else "uniform"
    why = "one CPU cluster"
    if fast and slow:
        name = model.dominant_format.upper()
        on_fast = pinned_rate(characteristics, model, fast) if fast == sorted(performance) else None
        on_all = (
            pinned_rate(characteristics, model, allowed) if allowed == sorted(logical) else None
        )
        if on_fast is None or on_all is None:
            why = f"no pinned one-token {name} measurement of the performance cluster and every CPU"
        elif on_fast <= on_all:
            why = (
                f"every CPU streams {name} at one token as fast or faster "
                f"({on_all / GB:.1f} against {on_fast / GB:.1f} GB/s pinned)"
            )
        elif fast[-1] > MAX_CPU:
            why = f"the performance CPUs up to CPU {fast[-1]} exceed a CPU mask"
        else:
            reason = (
                f"decode on the performance cluster {span(fast)}: pinned, one token streams "
                f"{name} at {on_fast / GB:.1f} GB/s there against {on_all / GB:.1f} GB/s on every "
                f"CPU; prefill on {count} unpinned threads; no I/O CPU"
            )
            notes.append("threads: " + reason)
            return Threads(len(fast), count, (), tuple(fast)), (), reason
    notes.append(
        f"threads: all placement, {count} unpinned threads for the {len(allowed)} allowed CPUs "
        f"{span(allowed)} ({shape}); decode not on a cluster alone: {why}"
    )
    return (
        Threads(count, count, ()),
        (),
        f"all placement: no I/O CPU, {count} unpinned threads for the {len(allowed)} allowed "
        f"CPUs {span(allowed)}; decode not on a cluster alone: {why}",
    )


def launch_conflicts(
    threads: Threads, io: Sequence[int], participants: int, platform: str
) -> list[str]:
    """What the launcher would refuse in these thread pools and CPU settings."""
    found = []
    counts = [count for count in (threads.decode, threads.batch) if count is not None]
    if any(type(count) is not int or not 1 <= count <= MAX_THREADS for count in counts):
        found.append(f"thread counts must be integers in 1..{MAX_THREADS}")
    cpus, decode_cpus = threads.cpus, threads.decode_cpus
    for ids in (cpus, decode_cpus):
        if len(set(ids)) != len(ids) or any(not 0 <= cpu <= MAX_CPU for cpu in ids):
            found.append(f"strict CPUs must be unique CPU IDs in 0..{MAX_CPU}")
    if cpus and decode_cpus:
        found.append("a decode CPU set excludes a CPU set for both thread pools")
    if (cpus or decode_cpus) and platform == "darwin":
        found.append("macOS cannot pin threads to CPUs")
    if cpus and any(count > len(cpus) for count in counts):
        found.append("thread counts exceed the strict CPUs")
    if decode_cpus and (threads.decode or 0) > len(decode_cpus):
        found.append("the decode thread count exceeds the decode CPUs")
    if set(io) & set(cpus + decode_cpus):
        found.append("cpu.io_cpus overlaps the strict compute CPUs")
    decode = threads.decode or (len(cpus) if cpus else None)
    if decode is not None and participants > decode:
        found.append("cpu.decode_participants exceeds the decode thread count")
    return found


def size(nbytes: float) -> str:
    """A byte count for a receipt: whole MiB where it is one."""
    if nbytes >= 1 << 20 and nbytes % (1 << 20) == 0:
        return f"{int(nbytes) >> 20} MiB"
    return f"{nbytes:.0f} bytes"


# Per role: weight bytes/s at one token, and FLOP/s by token count.
Rates = list[tuple[float, dict[int, float]]]


def _rates(row: ComputeRow) -> tuple[Optional[float], dict[int, float]]:
    """A row's weight bytes/s at one token and its FLOP/s by token count, positive values only."""
    one = next(
        (p["weight_Bps"] for p in row["points"] if p["tokens"] == 1 and p.get("weight_Bps", 0) > 0),
        None,
    )
    many = {p["tokens"]: p.get("flops", 0) for p in row["points"] if p["tokens"] > 1}
    return one, {tokens: flops for tokens, flops in many.items() if flops > 0}


def cpu_line(
    characteristics: Characteristics, model: ModelFacts, threads: int
) -> tuple[Optional[tuple[float, float]], str]:
    """The CPU's cost line of one expert in a grouped prefill pass, and how it was built.

    The engine prices an expert routed to ``r`` tokens of a pass at
    ``cpu_fixed_us + cpu_route_us * r`` microseconds of wall time: its gate,
    up and down products over those r tokens on the pass's THREADS (the batch
    pool); before decode has measured its own experts, the line at one route
    also prices a decode expert. At one token each role streams its weights,
    its slice bytes over the measured weight bytes/s; at n tokens it computes
    2 * rows * cols * n FLOP at the measured FLOP/s, n the largest token count
    measured. The line runs through both points and keeps both terms
    nonnegative as the engine's own fit does. Rates are interpolated linearly
    between the measured thread counts around THREADS, or taken from the
    nearest one outside them.
    """
    names = [role.format.upper() for role in model.roles]
    rows = {(row["format"].upper(), row["threads"]): row for row in _unpinned(characteristics)}
    missing = sorted(set(names) - {name for name, _ in rows})
    if missing:
        return None, "no CPU compute measurement of " + ", ".join(missing)
    measured: dict[int, Rates] = {}
    for count in sorted({count for _, count in rows}):
        roles = [_rates(rows[name, count]) for name in names if (name, count) in rows]
        if len(roles) == len(names) and all(one is not None for one, _ in roles):
            measured[count] = [(one or 0.0, many) for one, many in roles]

    def shared(counts: Sequence[int]) -> set[int]:
        """Token counts above one measured for every role at every count of COUNTS."""
        return set.intersection(*(set(many) for count in counts for _, many in measured[count]))

    measured = {count: rates for count, rates in measured.items() if shared([count])}
    if not measured:
        return None, "no thread count measured every expert format at one and more tokens"
    lower = max((count for count in measured if count <= threads), default=None)
    upper = min((count for count in measured if count >= threads), default=None)
    pair = sorted({count for count in (lower, upper) if count is not None})
    if not shared(pair):
        pair = [min(pair, key=lambda count: (abs(count - threads), count))]
    tokens = max(shared(pair))
    share = (threads - pair[0]) / (pair[1] - pair[0]) if len(pair) == 2 else 0.0

    def rate(role: int, at: int) -> float:
        values = [
            measured[count][role][0] if at == 1 else measured[count][role][1][at] for count in pair
        ]
        return values[0] + share * (values[-1] - values[0])

    one_us = sum(role.slice_bytes() / rate(i, 1) for i, role in enumerate(model.roles)) * 1e6
    many_us = 1e6 * sum(
        2.0 * role.rows * role.cols * tokens / rate(i, tokens) for i, role in enumerate(model.roles)
    )
    route = (many_us - one_us) / (tokens - 1)
    fixed = one_us - route
    if fixed < 0:
        fixed, route = 0.0, (one_us + tokens * many_us) / (1 + tokens * tokens)
    elif route < 0:
        fixed, route = (one_us + many_us) / 2, 0.0
    source = (
        f"the {pair[0]}-thread measurement"
        if len(pair) == 1
        else f"the {pair[0]}- and {pair[1]}-thread measurements"
    )
    why = (
        f"one {'/'.join(names)} expert, latent {model.up.cols} hidden {model.up.rows}, "
        f"on {threads} threads ({source}): {one_us:.1f} us at 1 token, "
        f"{many_us:.1f} us at {tokens} tokens"
    )
    if max(fixed, route) > COST_MAX:
        return None, why + ", beyond the cost term range"
    return (fixed, route), why


def drive_rate(
    characteristics: Characteristics, model: ModelFacts, key: Optional[str]
) -> tuple[Optional[float], str]:
    """The model drive's read rate in GB/s for the requests the engine issues.

    The engine reads each role of an expert as its own request, so the size
    looked up is a third of the mean expert read; the nearest measured size is
    taken on a log scale. The engine keeps up to IN_FLIGHT of those reads
    queued, so the rate is the best median between the size's knee and that
    depth (or the reader's own cap).
    """
    if key is None:
        return None, "the drive holding the model is not known"
    entry = next((row for row in characteristics["storage"] if row["drive_key"] == key), None)
    if entry is None:
        return None, f"no measurement of the drive {key}; run inferssd profile --model <model>"
    name = key if entry["model"] == "unknown" else entry["model"]
    request = model.mean_expert_read_bytes / 3
    knee = min(entry["knee"], key=lambda row: (abs(math.log(row["bytes"] / request)), row["bytes"]))
    deepest = min(IN_FLIGHT, entry["qd_cap"] or IN_FLIGHT)
    points = [
        row
        for row in entry["random_read"]
        if row["bytes"] == knee["bytes"] and knee["qd"] <= row["qd"] <= deepest and row["Bps"] > 0
    ]
    if not points:
        return None, f"no read rate of {size(knee['bytes'])} reads on {name} up to QD {deepest}"
    # The fastest median wins; between equal rates, the shallower queue.
    best = max(points, key=lambda row: (row["Bps"], -row["qd"]))
    rate = best["Bps"] / GB
    if rate > COST_MAX:
        return None, f"a read rate of {rate} GB/s is beyond the cost term range"
    return rate, (
        f"the drive {name} reads {rate:.2f} GB/s with {size(knee['bytes'])} reads at QD "
        f"{best['qd']}, the best from its knee QD {knee['qd']} up to QD {deepest} "
        f"(a role read is {size(request)} on average)"
    )


def link_rates(
    characteristics: Characteristics, runtime: RuntimeFacts
) -> tuple[dict[str, float], str]:
    """Pinned and pageable host-to-device rates in GB/s at the largest measured copy."""
    if runtime.gpus is None:
        return {}, "the CUDA GPUs could not be read now"
    present = runtime.gpus
    if len(present) != 1:
        if not present:
            return {}, "no CUDA GPU is usable now"
        return {}, f"{len(present)} CUDA GPUs are usable and the engine drives one of them"
    gpu = next((row for row in characteristics["gpus"] if row["uuid"] == present[0].uuid), None)
    if gpu is None:
        return {}, f"the GPU {present[0].uuid} was not measured"
    if gpu["integrated"]:
        return {}, f"{gpu['name']} shares host memory: no copy over a link"
    terms, parts = {}, []
    copies = (
        ("link_gbps", "pinned", gpu["h2d"]["pinned"]),
        ("pageable_gbps", "pageable", gpu["h2d"]["pageable"]),
    )
    for term, kind, rows in copies:
        if not rows:
            continue
        largest = max(rows, key=lambda row: row["bytes"])
        rate = largest["Bps"] / GB
        if 0 < rate <= COST_MAX:
            terms[term] = rate
            parts.append(f"{kind} {rate:.2f} GB/s at {size(largest['bytes'])}")
    return terms, f"{gpu['name']} host-to-device " + (", ".join(parts) or "not measured")


def prefill_split(
    model: ModelFacts, line: tuple[float, float], drive_gbps: float, io_gbps: float
) -> tuple[Optional[int], str]:
    """The longest prompt pass the CPU prefills faster than the GPU graph, and why.

    A pass of n tokens touches U(n) = E (1 - (1 - k/E)^n) of a layer's E
    experts when each token routes to k of them uniformly. The CPU route reads
    those experts from the drive while it computes them, n k routes a layer on
    the CPU line, so it takes the longer of the two; the graph reads the same
    experts into the GPU at the slower of the drive and the link, and its
    compute is taken as free. Only a link slower than the drive leaves the CPU
    faster, for short passes while U(n) still grows with n, and slower past the
    crossing. A tie goes to the graph: it leaves those experts in the VRAM tier,
    which decode reads next. None when the GPU never wins within MAX_PASS tokens.
    """
    k = model.experts_used
    E, L, B = model.experts, model.layers, model.mean_expert_read_bytes
    fixed, route = line
    if not k:
        return None, "the GGUF header gives no experts per token"

    def cpu_faster(n: int) -> bool:
        touched = L * E * (1 - (1 - k / E) ** n)
        read = touched * B / (drive_gbps * GB)
        cpu = (touched * fixed + n * k * L * route) * 1e-6
        return max(read, cpu) < touched * B / (io_gbps * GB)

    why = (
        f"{L} layers of {E} experts, {k} a token, {size(B)} read an expert; CPU {fixed:.1f} + "
        f"{route:.2f} us a route reading {drive_gbps:.2f} GB/s, "
        f"GPU graph reading {io_gbps:.2f} GB/s"
    )
    if not cpu_faster(1):
        return 0, why
    lo, hi = 1, 2
    while cpu_faster(hi):
        lo, hi = hi, hi * 2
        if hi > MAX_PASS:
            return None, why + f": the CPU is faster up to {MAX_PASS} tokens"
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if cpu_faster(mid):
            lo = mid
        else:
            hi = mid
    return lo, why


def _costs(terms: Mapping[str, float], reasons: Sequence[str]) -> Optional[tuple[str, str]]:
    if not terms:
        return None
    value = ",".join(f"{term}={number:.17g}" for term, number in sorted(terms.items()))
    return value, "; ".join(reasons)


def single_gpu(characteristics: Characteristics, runtime: RuntimeFacts) -> Optional[GPU]:
    """The profiled GPU when it is the one CUDA GPU the launch can use."""
    return next(
        (
            row
            for row in characteristics["gpus"]
            if runtime.gpus and len(runtime.gpus) == 1 and row["uuid"] == runtime.gpus[0].uuid
        ),
        None,
    )


def host_headroom_mib(
    characteristics: Characteristics, runtime: RuntimeFacts, platform: str
) -> Optional[int]:
    """The host margin a launch sets as host.headroom_mib, or None to keep the engine's own.

    On macOS it is a quarter of memory; on Linux with one usable GPU that shares host memory,
    UNIFIED_HEADROOM_MIB.
    """
    if platform == "darwin":
        return characteristics["memory"]["total_bytes"] // DARWIN_HEADROOM_SHARE >> 20
    if platform == "linux":
        device = single_gpu(characteristics, runtime)
        if device is not None and device["integrated"]:
            return UNIFIED_HEADROOM_MIB
    return None


def derive(
    characteristics: Characteristics,
    model: ModelFacts,
    runtime: RuntimeFacts,
    platform: str,
    *,
    model_drive_key: Optional[str],
) -> Strategy:
    """The strategy for one launch of MODEL on this machine; see the module docstring."""
    if platform not in PLATFORMS:
        raise ValueError("unknown platform " + platform)
    text = dumps(characteristics)
    if characteristics["identity"]["hard"]["platform"] != platform:
        raise ValueError("the machine characteristics describe another platform")
    notes: list[str] = []
    choices: dict[str, tuple[str, str]] = {}
    threads, io, why = placement(characteristics, model, runtime, notes)
    choices["cpu.io_cpus"] = (span(io) if io else "", why)
    choices["cpu.decode_participants"] = ("0", "the engine measures its decode participants")
    conflicts = launch_conflicts(threads, io, 0, platform)
    if conflicts or threads.batch is None:
        raise AssertionError("derived thread pools the launcher refuses: " + "; ".join(conflicts))

    terms: dict[str, float] = {}
    reasons: list[str] = []
    line, why = cpu_line(characteristics, model, threads.batch)
    if line is not None:
        terms["cpu_fixed_us"], terms["cpu_route_us"] = line
        reasons.append(f"cpu {line[0]:.1f} + {line[1]:.2f} us a route for {why}")
    notes.append(
        "place.costs: cpu "
        + (f"{line[0]:.1f} + {line[1]:.2f} us a route for " if line else "omitted, ")
        + why
    )
    link, why = link_rates(characteristics, runtime)
    terms.update(link)
    if link:
        reasons.append(why)
    notes.append("place.costs: link " + ("" if link else "omitted, ") + why)
    drive, why = drive_rate(characteristics, model, model_drive_key)
    if drive is not None:
        terms["drive_gbps"] = drive
        reasons.append(why)
    notes.append("place.costs: drive " + ("" if drive is not None else "omitted, ") + why)
    costs = _costs(terms, reasons)
    if costs is not None:
        choices["place.costs"] = costs

    one_pass = False
    mixed = [role for role in model.roles if len(role.formats) > 1]
    # The CUDA tier's budget, where the engine has the tier. A GPU lane (Vulkan, Metal) takes
    # the default.
    if "CUDA" in engine_gpu_backends(runtime.gpu_backends, characteristics):
        listed = {
            gpu["uuid"]
            for gpu in characteristics["identity"]["hard"]["gpus"]
            if gpu["backend"] == "CUDA"
        }
        if runtime.gpus is None:
            notes.append("vram.budget: default; the CUDA GPUs could not be read now")
        elif not runtime.gpus:
            extra = f", although the machine has {len(listed)}" if listed else ""
            choices["vram.budget"] = ("0", "no CUDA GPU is usable now" + extra)
            notes.append("vram.budget: 0, no CUDA GPU is usable now" + extra)
        elif mixed:
            # The VRAM tier and its prefill lane compute one expert format a role.
            why = "the GPU tier computes one expert format a role, and this model's " + "; ".join(
                f"{role.name} experts are "
                + ", ".join(f"{name} in {n} layers" for name, n in role.formats)
                for role in mixed
            )
            choices["vram.budget"] = ("0", why)
            notes.append("vram.budget: 0, " + why)
        else:
            known = [gpu for gpu in runtime.gpus if gpu.uuid in listed]
            if known:
                notes.append("vram.budget: default auto for " + ", ".join(g.uuid for g in known))
            else:
                notes.append(
                    "vram.budget: default auto; the usable GPU is not in the characteristics, "
                    "run inferssd profile again"
                )
            busy = [
                f"{gpu.uuid} runs {gpu.processes} compute process(es)"
                for gpu in runtime.gpus
                if gpu.processes
            ]
            if busy:
                why = ", ".join(busy) + f"; {SHARED_GPU_RESERVE_MIB} MiB stays free for them"
                choices["vram.reserve_mib"] = (str(SHARED_GPU_RESERVE_MIB), why)
                notes.append("vram.reserve_mib: " + why)
        device = single_gpu(characteristics, runtime)
        if host_headroom_mib(characteristics, runtime, platform) == UNIFIED_HEADROOM_MIB:
            assert device is not None
            choices["host.headroom_mib"] = (
                str(UNIFIED_HEADROOM_MIB),
                f"{device['name']} shares host memory: the smallest host margin that held under "
                "stress with the automatic VRAM tier",
            )
            notes.append(f"host.headroom_mib: {UNIFIED_HEADROOM_MIB}, the GPU shares host memory")
        rates = (
            None
            if device is None or line is None or drive is None
            else (
                drive,
                drive if device["integrated"] else min(drive, link.get("link_gbps", drive)),
            )
        )
        if rates is None:
            notes.append(
                "prefill: the CPU route for every pass and passes of the launch's batch; "
                "no single profiled CUDA GPU with a CPU line and a drive rate"
            )
        else:
            split, why = prefill_split(model, cast(tuple[float, float], line), *rates)
            if split is None:
                notes.append("prefill: the CPU route for every pass; " + why)
            else:
                choices["prefill.cpu_max_tokens"] = (
                    str(split),
                    f"the longest pass the CPU prefills faster than the GPU graph: {why}",
                )
                notes.append(f"prefill.cpu_max_tokens: {split}; {why}")
            one_pass = True
            notes.append(
                "prefill: one pass a prompt, -b and -ub as large as the context; a pass reads "
                "every expert it routes to once"
            )
    if platform == "darwin":
        mib = cast(int, host_headroom_mib(characteristics, runtime, platform))
        choices["host.headroom_mib"] = (
            str(mib),
            f"a quarter of the {size(characteristics['memory']['total_bytes'])} of memory: macOS "
            "compresses a RAM cache's cold pages long before free memory runs out",
        )
        notes.append(f"host.headroom_mib: {mib}, a quarter of the machine's memory")
    if platform in ("darwin", "windows") and runtime.memory_budget_bytes:
        mib = runtime.memory_budget_bytes >> 20
        if mib:
            choices["host.budget_mib"] = (
                str(mib),
                f"the memory budget of {runtime.memory_budget_bytes} bytes",
            )
            notes.append(f"host.budget_mib: {mib} from the memory budget")
    notes.append("server: " + " ".join(SERVER_ARGS))
    return Strategy(
        settings=_validated(choices, platform),
        threads=threads,
        server_args=SERVER_ARGS,
        inputs=Inputs(hashlib.sha256(text.encode()).hexdigest(), model.paths, platform),
        notes=tuple(notes),
        one_pass=one_pass,
    )
