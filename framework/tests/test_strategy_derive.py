# SPDX-License-Identifier: Apache-2.0
"""Strategy derivation on recorded machine characteristics and synthetic GGUF headers."""

from __future__ import annotations

import copy
import hashlib
import math
from pathlib import Path

import pytest
from test_strategy_inputs import (
    MXFP4,
    Q3_K,
    Q4_K,
    Q5_K,
    Q8_0,
    deepseek_like,
    moe,
    muse_like,
    write_gguf,
)

from inferssd.configuration.settings import parse, render, schema
from inferssd.launch.typed import threadpools
from inferssd.launch.types import Threads
from inferssd.machine.characteristics import dumps, load
from inferssd.strategy.derive import (
    SERVER_ARGS,
    cpu_line,
    decode_threads,
    derive,
    drive_rate,
    io_cpu,
    launch_conflicts,
    literal,
    offered,
    prefill_threads,
)
from inferssd.strategy.inputs import GPUState, RuntimeFacts, model_facts

MACHINES = Path(__file__).parent / "fixtures" / "machine"
X86 = "linux-x86-rtx5090.json"  # 61 uniform vCPUs, RTX 5090, virtual disk
ARM = "linux-arm-gb10.json"  # 10 + 10 Arm cores, GB10 sharing host memory, NVMe
MAC = "macos-m5.json"  # 4 + 6 Apple cores, Apple SSD
X86_GPU = "GPU-00000000-0000-4000-8000-000000000016"
ARM_GPU = "GPU-00000000-0000-4000-8000-000000000032"


def machine(name):
    return load(MACHINES / name)


def drive(characteristics):
    return characteristics["storage"][0]["drive_key"]


@pytest.fixture(scope="module")
def models(tmp_path_factory):
    directory = tmp_path_factory.mktemp("models")
    return {
        "dsv": model_facts(str(write_gguf(directory / "dsv.gguf", *deepseek_like()))),
        "muse": model_facts(str(write_gguf(directory / "muse.gguf", *muse_like()))),
    }


def runtime(cpus, gpus=(), budget=None, backends=None):
    return RuntimeFacts(
        tuple(cpus), tuple(gpus) if gpus is not None else None, 1 << 36, budget, backends
    )


def costs(strategy):
    value = parse(None, overrides=[("place.costs", strategy.settings["place.costs"].value_text)])
    return dict(value.at("place.costs").value)


def point(characteristics, fmt, threads, tokens, field):
    row = next(
        row
        for row in characteristics["compute"]["cpu"]
        if row["format"] == fmt and row["threads"] == threads
    )
    return next(p[field] for p in row["points"] if p["tokens"] == tokens)


def test_x86_uniform_machine_with_a_discrete_gpu(models):
    c = machine(X86)
    gpu = GPUState(X86_GPU, 30 << 30, 0)
    s = derive(c, models["dsv"], runtime(range(61), [gpu]), "linux", model_drive_key=drive(c))
    # The file measured one token on 1 and 61 threads only: 61 stream MXFP4 fastest.
    assert s.threads == Threads(61, 60, ())
    # The batch pool leaves the last CPU to the sink's I/O threads.
    assert s.settings["cpu.io_cpus"].value_text == '"60"'
    assert s.settings["cpu.decode_participants"].value_text == "0"
    assert "vram.budget" not in s.settings  # auto stays the registry default
    assert "host.budget_mib" not in s.settings
    terms = costs(s)
    assert sorted(terms) == [
        "cpu_fixed_us",
        "cpu_route_us",
        "drive_gbps",
        "link_gbps",
        "pageable_gbps",
    ]
    # The largest copies: 28.2812e9 B/s pinned and 16.563e9 pageable, in 1e9 bytes per second.
    assert terms["link_gbps"] == pytest.approx(28.2812)
    assert terms["pageable_gbps"] == pytest.approx(16.563)
    # A 6266880-byte role read takes the 4 MiB reads: the best median from the
    # knee (QD 4) up to the engine's 24 reads in flight is at QD 16; QD 32 is
    # faster still but deeper than the engine queues.
    assert terms["drive_gbps"] == pytest.approx(9.623315)
    assert "4 MiB reads at QD 16" in s.settings["place.costs"].reason
    assert s.server_args == SERVER_ARGS == ("--ctx-checkpoints", "1")
    # Every pass takes the engine's SSD sink, each prompt in one pass.
    assert any(note.startswith("prefill: the SSD sink for every pass") for note in s.notes)
    assert s.one_pass
    assert "host.headroom_mib" not in s.settings  # a discrete GPU
    assert s.inputs.platform == "linux"
    assert s.inputs.model_paths == models["dsv"].paths
    assert s.inputs.characteristics_sha256 == hashlib.sha256(dumps(c).encode()).hexdigest()


def test_cpu_line_units_at_an_exact_thread_count(models):
    """Mac, 10 threads measured: three 6266880-byte MXFP4 roles and 2 x 5120 x 2304 FLOP a token."""
    c = machine(MAC)
    line, why = cpu_line(c, models["dsv"], 10)
    assert line is not None
    one = 3 * 6266880 / point(c, "MXFP4", 10, 1, "weight_Bps") * 1e6
    many = 3 * 2 * 5120 * 2304 * 512 / point(c, "MXFP4", 10, 512, "flops") * 1e6
    route = (many - one) / 511
    assert line == pytest.approx((one - route, route))
    assert one == pytest.approx(162.7, abs=0.1) and many == pytest.approx(76370, rel=1e-4)
    assert "10-thread measurement" in why


def test_cpu_line_mixes_role_formats(models):
    c = machine(MAC)
    line, _ = cpu_line(c, models["muse"], 10)
    assert line is not None
    one = (2 * 9437184 / point(c, "Q4_K", 10, 1, "weight_Bps")) + (
        11534336 / point(c, "Q5_K", 10, 1, "weight_Bps")
    )
    flop = 2 * 4096 * 4096 * 512
    many = 2 * flop / point(c, "Q4_K", 10, 512, "flops") + flop / point(c, "Q5_K", 10, 512, "flops")
    route = (many - one) * 1e6 / 511
    assert line == pytest.approx((one * 1e6 - route, route))


def test_drive_rate_stays_within_the_readers_cap(models):
    c = machine(MAC)
    # The macOS reader pool runs at most 8 reads at once: 16 MiB reads from the knee, QD 8.
    s = derive(c, models["muse"], runtime(range(10)), "darwin", model_drive_key=drive(c))
    assert costs(s)["drive_gbps"] == pytest.approx(6.84916)
    assert "16 MiB reads at QD 8, the best from its knee QD 8 up to QD 8" in (
        s.settings["place.costs"].reason
    )


@pytest.mark.parametrize("cap, expected_qd, expected_rate", [(None, 16, 7.0), (8, 4, 6.0)])
def test_drive_rate_uses_measured_points_when_the_knee_exceeds_the_cap(
    models, cap, expected_qd, expected_rate
):
    c = machine(X86)
    storage = c["storage"][0]
    storage["qd_cap"] = cap
    storage["knee"] = [{"bytes": 4 << 20, "qd": 32}]
    template = next(row for row in storage["random_read"] if row["bytes"] == 4 << 20)
    storage["random_read"] = [
        dict(template, qd=qd, Bps=rate)
        for qd, rate in [(1, 3e9), (4, 6e9), (8, 5e9), (16, 7e9), (32, 9e9)]
    ]
    rate, why = drive_rate(c, models["dsv"], drive(c))
    assert rate == expected_rate
    assert f"4 MiB reads at QD {expected_qd}" in why
    assert f"up to QD {cap or 24}, capped below its knee QD 32" in why
    s = derive(c, models["dsv"], runtime(range(61)), "linux", model_drive_key=drive(c))
    assert costs(s)["drive_gbps"] == expected_rate


def test_an_unreachable_knee_still_needs_a_positive_admissible_measurement(models):
    c = machine(X86)
    storage = c["storage"][0]
    storage["knee"] = [{"bytes": 4 << 20, "qd": 32}]
    template = next(row for row in storage["random_read"] if row["bytes"] == 4 << 20)
    storage["random_read"] = [dict(template, qd=16, Bps=0), dict(template, qd=32, Bps=9e9)]
    rate, why = drive_rate(c, models["dsv"], drive(c))
    assert rate is None and "no read rate of 4 MiB reads" in why


def test_a_point_without_flops_is_dropped(models):
    c = machine(MAC)
    for row in c["compute"]["cpu"]:
        if row["format"] == "MXFP4" and row["threads"] == 10:
            del next(p for p in row["points"] if p["tokens"] == 512)["flops"]
    line, why = cpu_line(c, models["dsv"], 10)
    # The 10-thread row keeps its 32-token point; 512 tokens come from no row at 10 threads.
    assert line is not None and "at 32 tokens" in why


def test_cpu_line_interpolates_between_measured_thread_counts(models):
    c = machine(X86)
    line, why = cpu_line(c, models["dsv"], 31)
    assert line is not None and "1- and 61-thread measurements" in why
    share = 30 / 60
    one_rate = point(c, "MXFP4", 1, 1, "weight_Bps")
    one_rate += share * (point(c, "MXFP4", 61, 1, "weight_Bps") - one_rate)
    assert line[0] + line[1] == pytest.approx(3 * 6266880 / one_rate * 1e6)
    # Beyond the measured counts the nearest one stands.
    assert cpu_line(c, models["dsv"], 200)[0] == cpu_line(c, models["dsv"], 61)[0]


@pytest.mark.parametrize("scale", [1000, 1 / 1000])
def test_cpu_line_keeps_both_terms_nonnegative(models, scale):
    """Like the engine's fit: a negative term is 0 and the other fits the two points alone."""
    c = machine(MAC)
    for row in c["compute"]["cpu"]:
        for p in row["points"]:
            if p["tokens"] == 1:
                p["weight_Bps"] *= scale
    one = 3 * 6266880 / point(c, "MXFP4", 10, 1, "weight_Bps") * 1e6
    many = 3 * 2 * 5120 * 2304 * 512 / point(c, "MXFP4", 10, 512, "flops") * 1e6
    line, _ = cpu_line(c, models["dsv"], 10)
    if scale > 1:  # one token nearly free: the line runs through the origin
        assert line == pytest.approx((0.0, (one + 512 * many) / (1 + 512 * 512)))
    else:  # one token slower than 512: a flat line at the mean
        assert line == pytest.approx(((one + many) / 2, 0.0))


def test_arm_unified_memory(models):
    c = machine(ARM)
    gpu = GPUState(ARM_GPU, None, 0)
    s = derive(c, models["muse"], runtime(range(20), [gpu]), "linux", model_drive_key=drive(c))
    # Without pinned measurements decode stays unpinned, on the 10 threads that stream Q4_K at
    # 113.0 GB/s against 80.5 on 20.
    assert s.threads == Threads(10, 19, ())
    # The highest CPU of the efficiency cluster keeps the I/O threads.
    assert s.settings["cpu.io_cpus"].value_text == '"14"'
    assert "no one-token Q4_K measurement of the performance cluster pinned" in (
        s.settings["cpu.io_cpus"].reason
    )
    terms = costs(s)
    # GB10 shares host memory: no copy over a link is priced.
    assert sorted(terms) == ["cpu_fixed_us", "cpu_route_us", "drive_gbps"]
    # A 10136235-byte mean role read takes the 16 MiB reads, best at the knee, QD 2.
    assert terms["drive_gbps"] == pytest.approx(12.6298)
    # The automatic VRAM tier shares the host memory envelope with the RAM cache there.
    assert "vram.budget" not in s.settings
    assert any(note.startswith("vram.budget: default auto for ") for note in s.notes)
    assert s.settings["host.headroom_mib"].value_text == "4096"
    assert s.one_pass


def two_clusters(fast, slow):
    """The x86 file as a machine whose FAST CPUs are the performance cluster."""
    c = copy.deepcopy(machine(X86))
    ids = sorted(fast + slow)
    # The probe measures this machine's counts: the file's rows below its CPUs, and the file's
    # every-CPU rows as this machine's.
    total = max(row["threads"] for row in c["compute"]["cpu"] if not row.get("cpus"))
    c["compute"]["cpu"] = [
        dict(row, threads=len(ids)) if row["threads"] == total else row
        for row in c["compute"]["cpu"]
        if not row.get("cpus") and (row["threads"] < len(ids) or row["threads"] == total)
    ]
    c["cpu"]["logical"] = [
        dict(c["cpu"]["logical"][0], id=cpu, core=cpu, smt_siblings=[cpu], cluster=int(cpu in slow))
        for cpu in ids
    ]
    c["cpu"]["clusters"] = [
        {"id": 0, "cpus": sorted(fast), "kind": "performance"},
        {"id": 1, "cpus": sorted(slow), "kind": "efficiency"},
    ]
    c["identity"]["hard"]["logical_cpus"] = len(ids)
    return c


def every_cpu_rate(c, rate, fmt="MXFP4"):
    """C's unpinned one-token FMT measurement on every CPU, set to RATE weight bytes/s."""
    total = len(c["cpu"]["logical"])
    for row in c["compute"]["cpu"]:
        if row["format"] == fmt and row["threads"] == total and not row.get("cpus"):
            for p in row["points"]:
                if p["tokens"] == 1:
                    p["us"] = p["us_q1"] = p["us_q3"] = row["weight_bytes"] / rate * 1e6
                    p["weight_Bps"] = rate


def pin(c, cpus, rate, fmt="MXFP4"):
    """A pinned one-token measurement of FMT on CPUS, at RATE weight bytes/s."""
    row = {
        "format": fmt,
        "threads": len(cpus),
        "rows": 4096,
        "cols": 4096,
        "weight_bytes": 8912896,
        "points": [{"tokens": 1, "us": 8912896 / rate * 1e6, "weight_Bps": rate}],
        "cpus": sorted(cpus),
    }
    c["compute"]["cpu"].append(row)


@pytest.mark.parametrize("fast, slow", [(list(range(6, 12)), list(range(6))), ([0, 1], [2, 3, 4])])
def test_decode_on_the_performance_cluster_measured_faster(models, fast, slow):
    c = two_clusters(fast, slow)
    pin(c, fast, 110e9)
    every_cpu_rate(c, 80e9)
    allowed = sorted(fast + slow)
    s = derive(c, models["dsv"], runtime(allowed), "linux", model_drive_key=drive(c))
    count = len(allowed) - 1
    assert s.threads == Threads(len(fast), count, (), tuple(fast))
    # The I/O threads keep a CPU outside the pinned decode cluster: the highest of the other one.
    assert s.settings["cpu.io_cpus"].value_text == f'"{max(slow)}"'
    assert "110.0 GB/s there against 80.0 GB/s" in s.settings["cpu.io_cpus"].reason
    assert f"I/O on CPU {max(slow)}" in s.settings["cpu.io_cpus"].reason
    assert threadpools(s.threads, "linux") == s.threads
    assert launch_conflicts(s.threads, (max(slow),), 0, "linux") == []
    # The pinned rows do not enter the CPU cost line, which prices the batch pool.
    assert f"on {count} threads (the 1- and {len(allowed)}-thread measurements)" in (
        s.settings["place.costs"].reason
    )


@pytest.mark.parametrize(
    "allowed, rates, reason",
    [
        (range(12), None, "no one-token MXFP4 measurement of the performance cluster pinned"),
        (range(12), (80e9, 80e9), "12 unpinned threads stream MXFP4 at one token as fast or"),
        (range(12), (80e9, 90e9), "(90.0 against 80.0 GB/s pinned)"),
        # Two of the cluster's CPUs allowed: no pinned measurement of those.
        ([0, 1, 6, 7, 8], (110e9, 80e9), "no one-token MXFP4 measurement of the performance"),
        (range(6), (110e9, 80e9), "one CPU cluster"),
    ],
)
def test_decode_on_every_cpu(models, allowed, rates, reason):
    fast, slow = list(range(6)), list(range(6, 12))
    c = two_clusters(fast, slow)
    if rates:
        pin(c, fast, rates[0])
        every_cpu_rate(c, rates[1])
    s = derive(c, models["dsv"], runtime(allowed), "linux", model_drive_key=drive(c))
    # One token was measured on 1 and 61 threads: within the allowed CPUs the rate may still
    # rise past one thread, so decode takes them all.
    count = len(list(allowed)) - 1
    assert s.threads == Threads(count + 1, count, ())
    # The highest allowed CPU outside the performance cluster, or the highest of all without one.
    io = max([cpu for cpu in allowed if cpu not in fast] or allowed)
    assert s.settings["cpu.io_cpus"].value_text == f'"{io}"'
    assert reason in s.settings["cpu.io_cpus"].reason


def flops_row(threads, flops, fmt="MXFP4"):
    return {
        "format": fmt,
        "threads": threads,
        "rows": 4096,
        "cols": 4096,
        "weight_bytes": 8912896,
        "points": [{"tokens": 512, "us": 1.0, "flops": flops}],
    }


def test_prefill_threads_takes_the_measured_fastest_count_short_of_every_cpu():
    rows = [
        flops_row(n, f) for n, f in [(1, 50e9), (8, 400e9), (16, 600e9), (19, 650e9), (20, 660e9)]
    ]
    # Every CPU but one is measured: the fastest of the counts up to it.
    count, why = prefill_threads(rows, "MXFP4", 19)
    assert count == 19 and "the measured count with the most MXFP4 FLOP/s at 512 tokens" in why
    # A slower 19 leaves 16.
    rows[3] = flops_row(19, 550e9)
    assert prefill_threads(rows, "MXFP4", 19)[0] == 16


def test_prefill_threads_on_a_profile_without_every_cpu_but_one():
    # The fastest up to 19 is the most threads measured there, and 20 computes faster still.
    rows = [flops_row(n, f) for n, f in [(1, 50e9), (8, 400e9), (16, 600e9), (20, 660e9)]]
    count, why = prefill_threads(rows, "MXFP4", 19)
    assert count == 19 and "faster on 20" in why
    # 8 slower than 4 (a curve that turned down): 4, measured.
    rows = [flops_row(n, f) for n, f in [(1, 80e9), (4, 300e9), (8, 250e9)]]
    assert prefill_threads(rows, "MXFP4", 7)[0] == 4
    assert prefill_threads([], "MXFP4", 7) is None


def test_io_cpu_is_the_highest_spare_cpu_off_the_faster_cluster():
    # Outside the strict compute CPUs, on the slower cluster where there is one, the highest.
    gb10 = [5, 6, 7, 8, 9, 15, 16, 17, 18, 19]
    assert io_cpu(range(20), gb10, set(gb10)) == (14,)
    assert io_cpu([0, 1, 6, 7, 8], (), set(range(6))) == (8,)
    # A uniform machine names every CPU its performance cluster: the highest allowed CPU.
    assert io_cpu(range(61), (), set(range(61))) == (60,)
    # Only strict CPUs off the faster cluster: the highest CPU left.
    assert io_cpu([0, 1, 2, 3], [2, 3], {0, 1}) == (1,)
    # Fewer than two CPUs, or none outside the strict ones: no I/O CPU.
    assert io_cpu([3], (), {3}) == ()
    assert io_cpu([0, 1], [0, 1], {0, 1}) == ()


# One token on the 61 vCPUs of the x86 file's machine, in GB/s: the median call and the middle
# half of calls, by thread count.
X86_CURVE = {
    "MXFP4": {
        1: (10.1, 10.0, 10.1),
        2: (19.6, 19.4, 19.7),
        4: (38.3, 37.6, 39.0),
        8: (74.0, 69.5, 74.9),
        16: (117.5, 110.4, 120.7),
        32: (111.0, 107.1, 114.1),
        61: (116.5, 111.1, 119.7),
    },
    "Q4_K": {
        8: (90.0, 86.2, 92.4),
        16: (110.0, 106.4, 112.5),
        32: (116.5, 111.7, 119.3),
        61: (114.9, 111.2, 117.1),
    },
}


def curve_rows(fmt, rates, cpus=None):
    """One-token rows of FMT at RATES (GB/s by thread count), pinned to the first of CPUS."""
    rows = []
    for threads, (median, low, high) in rates.items():
        size = 8912896
        point = {"tokens": 1, "weight_Bps": median * 1e9, "us": size / median / 1e3}
        point.update(us_q1=size / high / 1e3, us_q3=size / low / 1e3)
        row = {"format": fmt, "threads": threads, "rows": 4096, "cols": 4096}
        row.update(weight_bytes=size, points=[point])
        if cpus is not None:
            row["cpus"] = sorted(cpus)[:threads]
        rows.append(row)
    return rows


def test_decode_takes_the_fewest_threads_at_the_best_rate():
    # MXFP4 is fastest on 16 threads; Q4_K on 32, whose slowest quarter of calls 16 reach.
    for fmt in X86_CURVE:
        assert decode_threads(curve_rows(fmt, X86_CURVE[fmt]), fmt, 61)[0] == 16
    count, why = decode_threads(curve_rows("Q4_K", X86_CURVE["Q4_K"]), "Q4_K", 61)
    assert why == (
        "the fewest threads that stream Q4_K at one token as fast as the best: "
        "16 (110.0 GB/s) against 32 (116.5 GB/s)"
    )
    # 8 threads do not reach 16: their fastest quarter is slower than its slowest.
    rates = {n: X86_CURVE["MXFP4"][n] for n in (1, 8, 16)}
    assert decode_threads(curve_rows("MXFP4", rates), "MXFP4", 16)[0] == 16
    # Still rising at the most threads measured, short of the CPUs: all of them.
    rates = {n: X86_CURVE["MXFP4"][n] for n in (1, 2, 4, 8)}
    assert decode_threads(curve_rows("MXFP4", rates), "MXFP4", 12)[0] == 12
    assert decode_threads(curve_rows("MXFP4", rates), "Q8_0", 12) is None


def test_decode_on_the_threads_the_curve_saturates_at(models):
    c = machine(X86)
    for fmt, rates in X86_CURVE.items():
        c["compute"]["cpu"] += curve_rows(fmt, {n: r for n, r in rates.items() if n not in (1, 61)})
    s = derive(c, models["dsv"], runtime(range(61)), "linux", model_drive_key=drive(c))
    assert s.threads == Threads(16, 60, ())
    assert launch_conflicts(s.threads, (), 0, "linux") == []


def test_decode_on_the_first_cpus_of_the_performance_cluster(models):
    fast, slow = list(range(6, 12)), list(range(6))
    c = two_clusters(fast, slow)
    every_cpu_rate(c, 80e9)
    c["compute"]["cpu"] += curve_rows(
        "MXFP4",
        {
            1: (20.0, 19.0, 21.0),
            2: (40.0, 38.0, 42.0),
            4: (108.0, 104.0, 112.0),
            6: (110.0, 106.0, 114.0),
        },
        fast,
    )
    s = derive(c, models["dsv"], runtime(range(12)), "linux", model_drive_key=drive(c))
    assert s.threads == Threads(4, 11, (), (6, 7, 8, 9))
    assert "110.0 GB/s there against 80.0 GB/s" in s.settings["cpu.io_cpus"].reason
    assert threadpools(s.threads, "linux") == s.threads
    assert launch_conflicts(s.threads, (), 0, "linux") == []


def test_affinity_limits_the_threads_and_the_cpu_line(models):
    c = machine(X86)
    s = derive(c, models["dsv"], runtime([3, 4, 5, 6, 7, 99]), "linux", model_drive_key=None)
    assert s.threads == Threads(5, 4, ())
    assert s.settings["cpu.io_cpus"].value_text == '"7"'
    assert "on 4 threads (the 1- and 61-thread measurements)" in s.settings["place.costs"].reason
    one = derive(c, models["dsv"], runtime([3]), "linux", model_drive_key=None)
    assert one.threads == Threads(1, 1, ())
    # One CPU has none to spare for the I/O threads.
    assert one.settings["cpu.io_cpus"].value_text == '""'
    assert "no I/O CPU: no CPU to spare" in one.settings["cpu.io_cpus"].reason
    with pytest.raises(ValueError, match="none of the CPUs"):
        derive(c, models["dsv"], runtime([99]), "linux", model_drive_key=None)


def test_macos_measured_thread_counts_and_budget(models):
    c = machine(MAC)
    s = derive(
        c, models["dsv"], runtime(range(10), budget=12 << 30), "darwin", model_drive_key=drive(c)
    )
    # 10 threads stream MXFP4 weights fastest and compute it fastest at 512 tokens; the batch
    # pool leaves a CPU to the I/O thread.
    assert s.threads == Threads(10, 9, ())
    assert "threads: decode 10, MXFP4 streams fastest at one token on 10 threads (115.6 GB/s)" in (
        s.notes
    )
    assert s.settings["cpu.io_cpus"].value_text == '""'
    assert s.settings["host.budget_mib"].value_text == "12288"
    # macOS keeps the engine's host margin: no share of memory is kept out of the RAM cache.
    assert "host.headroom_mib" not in s.settings
    assert "vram.budget" not in s.settings
    assert sorted(costs(s)) == ["cpu_fixed_us", "cpu_route_us", "drive_gbps"]
    assert threadpools(s.threads, "darwin") == s.threads
    without = derive(c, models["dsv"], runtime(range(10)), "darwin", model_drive_key=drive(c))
    assert "host.budget_mib" not in without.settings


def test_macos_picks_the_measured_best_counts(models):
    c = machine(MAC)
    for row in c["compute"]["cpu"]:
        if row["format"] == "MXFP4" and row["threads"] == 4:
            for p in row["points"]:
                p["weight_Bps" if p["tokens"] == 1 else "flops"] *= 100
    s = derive(c, models["dsv"], runtime(range(10)), "darwin", model_drive_key=drive(c))
    assert s.threads == Threads(4, 4, ())


def test_unmeasured_format_falls_back(models, tmp_path):
    tensors, kv = moe(
        "toy", layers=2, experts=4, used=2, latent=4096, hidden=1024, gate=Q3_K, up=Q3_K, down=Q3_K
    )
    q3 = model_facts(str(write_gguf(tmp_path / "q3.gguf", tensors, kv)))
    c = machine(MAC)
    s = derive(c, q3, runtime(range(10)), "darwin", model_drive_key=drive(c))
    # Q3_K was not measured: every allowed CPU but one for both pools, and no CPU line.
    assert s.threads == Threads(9, 9, ())
    assert sorted(costs(s)) == ["drive_gbps"]
    assert any("no CPU compute measurement of Q3_K" in note for note in s.notes)


def test_model_drive_without_a_measurement(models):
    c = machine(X86)
    gpu = GPUState(X86_GPU, 30 << 30, 0)
    for key, expected in ((None, "not known"), ("linux:nvme:other:" + "cd" * 32, "run inferssd")):
        s = derive(c, models["dsv"], runtime(range(61), [gpu]), "linux", model_drive_key=key)
        assert "drive_gbps" not in costs(s)
        assert any(expected in note for note in s.notes)


def test_no_usable_gpu_sets_a_zero_vram_budget(models):
    c = machine(X86)
    s = derive(c, models["dsv"], runtime(range(61)), "linux", model_drive_key=drive(c))
    assert s.settings["vram.budget"].value_text == "0"
    assert "although the machine has 1" in s.settings["vram.budget"].reason
    assert sorted(costs(s)) == ["cpu_fixed_us", "cpu_route_us", "drive_gbps"]
    c["identity"]["hard"]["gpus"], c["gpus"] = [], []
    c["identity"]["soft"]["probe_versions"].pop("memory")
    # A CUDA engine on a machine without a GPU.
    facts = runtime(range(61), backends=("CUDA",))
    s = derive(c, models["dsv"], facts, "linux", model_drive_key=drive(c))
    assert s.settings["vram.budget"].value_text == "0"
    assert s.settings["vram.budget"].reason == "no CUDA GPU is usable now"


def test_the_cuda_budget_follows_the_engine_not_the_platform(models):
    # An engine built without CUDA leaves the budget alone beside a usable NVIDIA GPU.
    c = machine(X86)
    gpu = GPUState(X86_GPU, 30 << 30, 2)
    for backends in ((), ("Vulkan",)):
        facts = runtime(range(61), [gpu], backends=backends)
        s = derive(c, models["dsv"], facts, "linux", model_drive_key=drive(c))
        assert "vram.budget" not in s.settings and "vram.reserve_mib" not in s.settings
    # A Windows engine with CUDA sizes its tier as a Linux one does.
    c = machine(WINDOWS)
    facts = runtime(range(8), backends=("CUDA",))
    s = derive(c, models["dsv"], facts, "windows", model_drive_key=drive(c))
    assert s.settings["vram.budget"].value_text == "0"
    assert s.settings["vram.budget"].reason == "no CUDA GPU is usable now"


def test_gpu_states_that_leave_the_vram_budget_alone(models):
    c = machine(X86)
    cases = {
        "unreadable": None,
        "unprofiled": [GPUState("GPU-new", 1 << 30, 0)],
        "two": [GPUState(X86_GPU, 1 << 30, 0), GPUState("GPU-new", 1 << 30, 0)],
        "busy": [GPUState(X86_GPU, 1 << 30, 2)],
    }
    notes = {
        "unreadable": "could not be read",
        "unprofiled": "run inferssd profile again",
        "two": "2 CUDA GPUs are usable",
        "busy": "runs 2 compute process(es)",
    }
    for name, gpus in cases.items():
        s = derive(c, models["dsv"], runtime(range(61), gpus), "linux", model_drive_key=drive(c))
        assert "vram.budget" not in s.settings, name
        assert any(notes[name] in note for note in s.notes), name
        link = "link_gbps" in costs(s)
        assert link == (name == "busy"), name


def test_other_processes_on_the_gpu_get_a_reserve_and_a_gpu_alone_keeps_none(models):
    c = machine(X86)
    busy = derive(
        c,
        models["dsv"],
        runtime(range(61), [GPUState(X86_GPU, 1 << 30, 2)]),
        "linux",
        model_drive_key=drive(c),
    )
    assert busy.settings["vram.reserve_mib"].value_text == "1024"
    assert "runs 2 compute process(es)" in busy.settings["vram.reserve_mib"].reason
    alone = derive(
        c,
        models["dsv"],
        runtime(range(61), [GPUState(X86_GPU, 1 << 30, 0)]),
        "linux",
        model_drive_key=drive(c),
    )
    assert "vram.reserve_mib" not in alone.settings


def test_characteristics_of_another_platform_are_refused(models):
    with pytest.raises(ValueError, match="another platform"):
        derive(machine(MAC), models["dsv"], runtime(range(10)), "linux", model_drive_key=None)
    with pytest.raises(ValueError, match="unknown platform"):
        derive(machine(MAC), models["dsv"], runtime(range(10)), "beos", model_drive_key=None)


def schema_holds(prop, value):
    """The subset of the settings JSON schema the strategy's keys use."""
    if "oneOf" in prop:
        return any(schema_holds(option, value) for option in prop["oneOf"])
    if "const" in prop:
        return value == prop["const"]
    if prop["type"] == "integer":
        return isinstance(value, int) and prop["minimum"] <= value <= prop["maximum"]
    if prop["type"] == "array":
        return len(set(value)) == len(value) and all(schema_holds(prop["items"], x) for x in value)
    terms = prop["properties"]
    return all(
        name in terms
        and terms[name]["maximum"] >= number
        and (
            number > terms[name]["exclusiveMinimum"] if "exclusiveMinimum" in terms[name] else True
        )
        and (number >= terms[name].get("minimum", -math.inf))
        for name, number in value.items()
    )


@pytest.mark.parametrize("name, platform", [(X86, "linux"), (ARM, "linux"), (MAC, "darwin")])
@pytest.mark.parametrize("model", ["dsv", "muse"])
@pytest.mark.parametrize("gpus", ["present", "none"])
def test_values_round_trip_through_the_configuration_layer(models, name, platform, model, gpus):
    c = machine(name)
    present = [GPUState(g["uuid"], None, 0) for g in c["gpus"]] if gpus == "present" else []
    facts = runtime([row["id"] for row in c["cpu"]["logical"]], present, budget=8 << 30)
    s = derive(c, models[model], facts, platform, model_drive_key=drive(c))
    assert launch_conflicts(s.threads, (), 0, platform) == []
    assert threadpools(s.threads, platform) == s.threads
    properties = schema()["properties"]
    overrides = [(key, choice.value_text) for key, choice in s.settings.items()]
    snapshot = parse(None, overrides=overrides, platform=platform)
    document = snapshot.document()["values"]
    for key, choice in s.settings.items():
        assert literal(snapshot.at(key), platform) == choice.value_text
        assert schema_holds(properties[key], document[key]), key
        assert "\n" not in choice.reason and choice.reason
    again = parse(render(snapshot), platform=platform)
    assert [item.value for item in again.entries] == [item.value for item in snapshot.entries]


def test_dense_only_shapes_do_not_reach_the_cpu_line(models, tmp_path):
    tensors, kv = moe(
        "toy", layers=1, experts=4, used=2, latent=4096, hidden=1024, gate=Q8_0, up=Q8_0, down=Q8_0
    )
    facts = model_facts(str(write_gguf(tmp_path / "q8.gguf", tensors, kv)))
    line, why = cpu_line(machine(X86), facts, 60)
    assert line is not None and "latent 4096 hidden 1024" in why


@pytest.mark.parametrize(
    "experts, down",
    [
        # like a Q4_K_S Muse: Q5_K down experts in the first layers, Q4_K after them
        (Q4_K, [Q5_K] * 2 + [Q4_K] * 6),
        # MXFP4 is one more format: mixed with Q8_0 it keeps the tier too
        (MXFP4, [MXFP4] * 6 + [Q8_0] * 2),
    ],
)
def test_experts_mixing_formats_across_layers_use_the_gpu_tier(tmp_path, experts, down):
    tensors, kv = moe(
        "muse-spark",
        layers=8,
        experts=16,
        used=4,
        latent=4096,
        hidden=1024,
        gate=experts,
        up=experts,
        down=down,
    )
    mixed = model_facts(str(write_gguf(tmp_path / "mixed.gguf", tensors, kv)))
    c = machine(X86)
    gpu = GPUState(X86_GPU, 30 << 30, 0)
    s = derive(c, mixed, runtime(range(61), [gpu]), "linux", model_drive_key=drive(c))
    assert "vram.budget" not in s.settings
    assert s.one_pass


WINDOWS = "windows-x64-igpu.json"  # 4 + 4 Intel cores with an integrated GPU, one NVMe drive


def test_windows_runs_the_one_derivation(models):
    c = machine(WINDOWS)
    s = derive(
        c,
        models["dsv"],
        runtime(range(8), budget=32 << 30),
        "windows",
        model_drive_key=drive(c),
    )
    # The registry's scopes decide: Windows has no GPU tier setting and has a host budget.
    assert not any(key.startswith(("vram.", "prefill.")) for key in s.settings)
    assert s.settings["host.budget_mib"].value_text == str(32 << 10)
    assert "host.headroom_mib" not in s.settings
    # 8 threads compute faster than 4 and 7 was not measured: every CPU but one.
    assert s.threads.batch == 7 and s.threads.decode_cpus in ((), (0, 1, 2, 3))
    terms = costs(s)
    assert "link_gbps" not in terms and terms["drive_gbps"] > 0
    assert not s.one_pass


@pytest.mark.parametrize(
    "key,platform,expected",
    [
        # The CUDA tier's keys are on every platform; the engine's backends decide if they apply.
        ("vram.max_slots", "linux", True),
        ("vram.max_slots", "darwin", True),
        ("vram.max_slots", "windows", True),
        ("host.budget_mib", "linux", False),
        ("host.budget_mib", "darwin", True),
        ("host.budget_mib", "windows", True),
    ],
)
def test_the_registry_decides_what_a_platform_derives(key, platform, expected):
    assert offered(key, platform) is expected


def test_windows_decode_goes_on_the_performance_cluster_when_it_streams_faster(models):
    c = copy.deepcopy(machine(WINDOWS))
    name = models["dsv"].dominant_format.upper()
    for row in c["compute"]["cpu"]:
        if row.get("cpus") == [0, 1, 2, 3] and row["format"].upper() == name:
            row["points"][0]["weight_Bps"] = 200e9
    s = derive(c, models["dsv"], runtime(range(8)), "windows", model_drive_key=drive(c))
    assert s.threads.decode_cpus == (0, 1, 2, 3) and s.threads.decode == 4 and s.threads.batch == 7


@pytest.mark.parametrize("full_rate", [0.3e9, 100e9])
@pytest.mark.parametrize("fast_rate, pinned", [(55e9, True), (45e9, False)])
def test_windows_chooses_measured_eligible_pools_and_keeps_their_placement(
    models, full_rate, fast_rate, pinned
):
    c = machine(WINDOWS)
    fast = [0, 1, 6, 7]
    c["cpu"]["clusters"][0]["cpus"] = fast
    c["cpu"]["clusters"][1]["cpus"] = [2, 3, 4, 5]
    for row in c["cpu"]["logical"]:
        row["cluster"] = int(row["id"] not in fast)
    name = models["dsv"].dominant_format.upper()
    template = next(row for row in c["compute"]["cpu"] if row["format"] == name)
    c["compute"]["cpu"] = [
        dict(
            template,
            threads=threads,
            points=[
                {"tokens": 1, "us": 1, "weight_Bps": rate},
                {"tokens": 512, "us": 1, "flops": flops},
            ],
        )
        for threads, rate, flops in [(1, 10e9, 100e9), (4, 50e9, 500e9), (8, full_rate, 1e12)]
    ]
    pin(c, fast, fast_rate, name)
    s = derive(c, models["dsv"], runtime(range(8)), "windows", model_drive_key=drive(c))
    # Decode takes 8 threads where they stream fastest, as the engine then leaves the other work
    # its CPUs; 4 when 8 are slow, and pinned to the cluster where it streams faster than those
    # 4 unpinned threads. Prefill: 8 threads compute faster than 4, so every CPU but one.
    decode = 8 if full_rate > 50e9 else 4
    on_cluster = pinned and decode == 4
    assert s.threads == Threads(decode, 7, (), tuple(fast) if on_cluster else ())
    assert f"{decode} unpinned threads" in s.settings["cpu.io_cpus"].reason
    # The I/O threads keep the highest CPU outside the performance cluster.
    assert s.settings["cpu.io_cpus"].value_text == '"5"'


def test_windows_selects_decode_and_batch_from_their_own_measurements(models):
    c = machine(WINDOWS)
    name = models["dsv"].dominant_format.upper()
    for row in c["compute"]["cpu"]:
        if row["format"] == name and row["threads"] == 1 and not row.get("cpus"):
            row["points"][0]["weight_Bps"] = 1e12
    s = derive(c, models["dsv"], runtime(range(8)), "windows", model_drive_key=drive(c))
    assert s.threads == Threads(1, 7, ())


@pytest.mark.parametrize("missing", ["one", "many"])
def test_windows_without_complete_eligible_measurements_keeps_its_fallback(models, missing):
    c = machine(WINDOWS)
    name = models["dsv"].dominant_format.upper()
    for row in c["compute"]["cpu"]:
        if row["format"] == name and not row.get("cpus"):
            row["points"] = [p for p in row["points"] if (p["tokens"] == 1) == (missing == "many")]
    for row in c["compute"]["cpu"]:
        if row["format"] == name and row.get("cpus"):
            row["points"][0]["weight_Bps"] = 80e9 if len(row["cpus"]) == 8 else 60e9
    s = derive(c, models["dsv"], runtime(range(8)), "windows", model_drive_key=drive(c))
    # Without one-token rows decode keeps every CPU but one; with them the 8 threads that stream
    # fastest, and as fast as the cluster pinned.
    decode = 7 if missing == "one" else 8
    why = (
        f"no one-token {name} measurement of 7 unpinned threads"
        if missing == "one"
        else f"8 unpinned threads stream {name} at one token as fast or faster (69.1 against 60.0 "
        "GB/s pinned)"
    )
    assert s.threads == Threads(decode, 7, ())
    assert s.settings["cpu.io_cpus"].reason == (
        f"all placement: I/O on CPU 7, decode {decode} and prefill 7 unpinned threads for the 8 "
        f"allowed CPUs 0-7; decode not on a cluster alone: {why}"
    )
