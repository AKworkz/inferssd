# SPDX-License-Identifier: Apache-2.0
"""Strategy derivation on recorded machine characteristics and synthetic GGUF headers."""

from __future__ import annotations

import copy
import hashlib
import math
from pathlib import Path

import pytest
from test_strategy_inputs import Q3_K, Q4_K, Q5_K, Q8_0, deepseek_like, moe, muse_like, write_gguf

from inferssd.configuration.settings import parse, render, schema
from inferssd.launch.typed import threadpools
from inferssd.launch.types import Threads
from inferssd.machine.characteristics import dumps, load
from inferssd.strategy.derive import (
    SERVER_ARGS,
    cpu_line,
    derive,
    launch_conflicts,
    literal,
    prefill_split,
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
    assert s.threads == Threads(60, 60, ())
    assert s.settings["cpu.io_cpus"].value_text == '""'
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
    # The 5090 graph reads at the drive's rate, slower than its link, so the CPU
    # is never faster: every pass goes to the graph, each prompt in one pass.
    assert s.settings["prefill.cpu_max_tokens"].value_text == "0"
    assert "GPU graph reading 9.62 GB/s" in s.settings["prefill.cpu_max_tokens"].reason
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
    # Without pinned measurements decode stays on every CPU but one, unpinned.
    assert s.threads == Threads(19, 19, ())
    assert s.settings["cpu.io_cpus"].value_text == '""'
    assert "no pinned one-token Q4_K measurement" in s.settings["cpu.io_cpus"].reason
    terms = costs(s)
    # GB10 shares host memory: no copy over a link is priced.
    assert sorted(terms) == ["cpu_fixed_us", "cpu_route_us", "drive_gbps"]
    # A 10136235-byte mean role read takes the 16 MiB reads, best at the knee, QD 2.
    assert terms["drive_gbps"] == pytest.approx(12.6298)
    # The automatic VRAM tier shares the host memory envelope with the RAM cache there.
    assert "vram.budget" not in s.settings
    assert any(note.startswith("vram.budget: default auto for ") for note in s.notes)
    assert s.settings["host.headroom_mib"].value_text == "4096"
    # The graph reads at the drive's rate too: every pass goes to the GPU.
    assert s.settings["prefill.cpu_max_tokens"].value_text == "0"
    assert s.one_pass


def two_clusters(fast, slow):
    """The x86 file as a machine whose FAST CPUs are the performance cluster."""
    c = copy.deepcopy(machine(X86))
    ids = sorted(fast + slow)
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
    pin(c, fast + slow, 80e9)
    allowed = sorted(fast + slow)
    s = derive(c, models["dsv"], runtime(allowed), "linux", model_drive_key=drive(c))
    count = len(allowed) - 1
    assert s.threads == Threads(len(fast), count, (), tuple(fast))
    assert s.settings["cpu.io_cpus"].value_text == '""'
    assert "110.0 GB/s there against 80.0 GB/s" in s.settings["cpu.io_cpus"].reason
    assert threadpools(s.threads, "linux") == s.threads
    assert launch_conflicts(s.threads, (), 0, "linux") == []
    # The pinned rows do not enter the CPU cost line, which prices the batch pool.
    assert f"on {count} threads (the 1- and 61-thread measurements)" in (
        s.settings["place.costs"].reason
    )


@pytest.mark.parametrize(
    "allowed, rates, reason",
    [
        (range(12), None, "no pinned one-token MXFP4 measurement"),
        (range(12), (80e9, 80e9), "every CPU streams MXFP4 at one token as fast or faster"),
        (range(12), (80e9, 90e9), "(90.0 against 80.0 GB/s pinned)"),
        ([0, 1, 6, 7, 8], (110e9, 80e9), "no pinned one-token MXFP4 measurement"),
        (range(6), (110e9, 80e9), "one CPU cluster"),
    ],
)
def test_decode_on_every_cpu(models, allowed, rates, reason):
    fast, slow = list(range(6)), list(range(6, 12))
    c = two_clusters(fast, slow)
    if rates:
        pin(c, fast, rates[0])
        pin(c, fast + slow, rates[1])
    s = derive(c, models["dsv"], runtime(allowed), "linux", model_drive_key=drive(c))
    count = len(list(allowed)) - 1
    assert s.threads == Threads(count, count, ())
    assert s.settings["cpu.io_cpus"].value_text == '""'
    assert reason in s.settings["cpu.io_cpus"].reason


def test_prefill_split_crossing(models):
    dsv = models["dsv"]
    line = (176.7, 114.28)
    # A tie goes to the graph, which leaves the experts in the VRAM tier for decode.
    split, why = prefill_split(dsv, line, 11.85, 11.85)
    assert split == 0 and "40 layers of 384 experts, 6 a token" in why

    # A link slower than the drive leaves the CPU the short passes, up to the crossing.
    split, _ = prefill_split(dsv, line, 11.85, 5.0)

    def seconds(n, graph):
        touched = dsv.layers * dsv.experts * (1 - (1 - 6 / dsv.experts) ** n)
        if graph:
            return touched * dsv.expert_read_bytes / 5.0e9
        read = touched * dsv.expert_read_bytes / 11.85e9
        return max(read, (touched * line[0] + n * 6 * dsv.layers * line[1]) * 1e-6)

    assert split > 0
    assert seconds(split, False) < seconds(split, True)
    assert seconds(split + 1, False) >= seconds(split + 1, True)
    # A slower link makes the graph slower: the CPU keeps longer passes.
    assert prefill_split(dsv, line, 11.85, 3.0)[0] > split
    # A CPU that is slower even for one token leaves every pass to the graph.
    assert prefill_split(dsv, (1e6, 1e6), 11.85, 5.0)[0] == 0
    # A CPU that is never slower within the bound leaves the engine's default.
    assert prefill_split(dsv, (0.0, 0.0), 11.85, 5.0)[0] is None


def test_affinity_limits_the_threads_and_the_cpu_line(models):
    c = machine(X86)
    s = derive(c, models["dsv"], runtime([3, 4, 5, 6, 7, 99]), "linux", model_drive_key=None)
    assert s.threads == Threads(4, 4, ())
    assert "on 4 threads (the 1- and 61-thread measurements)" in s.settings["place.costs"].reason
    one = derive(c, models["dsv"], runtime([3]), "linux", model_drive_key=None)
    assert one.threads == Threads(1, 1, ())
    with pytest.raises(ValueError, match="none of the CPUs"):
        derive(c, models["dsv"], runtime([99]), "linux", model_drive_key=None)


def test_macos_measured_thread_counts_and_budget(models):
    c = machine(MAC)
    s = derive(
        c, models["dsv"], runtime(range(10), budget=12 << 30), "darwin", model_drive_key=drive(c)
    )
    # 10 threads stream MXFP4 weights fastest and compute it fastest at 512 tokens.
    assert s.threads == Threads(10, 10, ())
    assert s.settings["cpu.io_cpus"].value_text == '""'
    assert s.settings["host.budget_mib"].value_text == "12288"
    # A quarter of the machine's memory stays out of the RAM cache.
    quarter = c["memory"]["total_bytes"] // 4 >> 20
    assert s.settings["host.headroom_mib"].value_text == str(quarter)
    assert "macOS compresses" in s.settings["host.headroom_mib"].reason
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
    # Q3_K was not measured: the performance cluster's four CPUs, and no CPU line.
    assert s.threads == Threads(4, 4, ())
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


def test_experts_mixing_formats_across_layers_keep_the_gpu_tier_off(tmp_path):
    """Like a Q4_K_S Muse: Q5_K down experts in the first layers, Q4_K after them."""
    down = [Q5_K] * 2 + [Q4_K] * 6
    tensors, kv = moe(
        "muse-spark",
        layers=8,
        experts=16,
        used=4,
        latent=4096,
        hidden=1024,
        gate=Q4_K,
        up=Q4_K,
        down=down,
    )
    mixed = model_facts(str(write_gguf(tmp_path / "mixed.gguf", tensors, kv)))
    c = machine(X86)
    gpu = GPUState(X86_GPU, 30 << 30, 0)
    s = derive(c, mixed, runtime(range(61), [gpu]), "linux", model_drive_key=drive(c))
    assert s.settings["vram.budget"].value_text == "0"
    assert s.settings["vram.budget"].reason == (
        "the GPU tier computes one expert format a role, and this model's down experts are "
        "Q4_K in 6 layers, Q5_K in 2 layers"
    )
    # The graph still prefills long passes.
    assert s.one_pass and "prefill.cpu_max_tokens" in s.settings
