# SPDX-License-Identifier: Apache-2.0
"""Decode speed estimates of the supported models from a machine profile."""

from __future__ import annotations

import copy
from collections import Counter
from pathlib import Path

from inferssd.app import catalog, hub
from inferssd.machine.store import load
from inferssd.strategy.estimate import decode_estimate
from inferssd.strategy.inputs import RuntimeFacts

MAC = load(Path(__file__).parent / "fixtures/machine/macos-m5.json")  # M5, 16 GiB, Apple SSD
DRIVE = MAC["storage"][0]["drive_key"]
GIB = 1 << 30


def runtime(available):
    return RuntimeFacts(tuple(row["id"] for row in MAC["cpu"]["logical"]), (), available)


def estimate(model, characteristics=MAC, available=9 * GIB, drive=DRIVE):
    return decode_estimate(
        characteristics, model.facts, model.dense_read_bytes, runtime(available), "darwin", drive
    )


def ssd_models():
    """The supported models with an expert store; a RAM-only model has no expert facts."""
    return [model for model in catalog.MODELS if model.facts is not None]


def unmeasured(model):
    """MODEL's expert formats that the Mac profile has no CPU compute measurement of."""
    measured = {row["format"].upper() for row in MAC["compute"]["cpu"] if not row.get("cpus")}
    return sorted({role.format.upper() for role in model.facts.roles} - measured)


def test_the_catalog_is_the_download_list_and_its_facts_add_up():
    assert hub.MODELS == {model.repo: model.quantization for model in catalog.MODELS}
    for model in catalog.MODELS:
        facts = model.facts
        if facts is None:  # RAM-only: a token reads nothing from the drive
            assert model.lookup_bytes == 0 and model.dense_read_bytes == 0
            continue
        assert 0 < model.dense_read_bytes < facts.dense_bytes
        # Each type's total is exactly its roles' layers of that type, over every expert.
        per_type: Counter[str] = Counter()
        for role in facts.roles:
            assert sum(layers for _, layers in role.formats) == facts.layers
            for name, layers in role.formats:
                per_type[name] += layers * facts.experts * role.slice_bytes(name)
        assert dict(facts.format_bytes) == dict(per_type)
        assert facts.expert_bytes <= facts.expert_read_bytes
        assert facts.mean_expert_read_bytes <= facts.expert_read_bytes


def test_every_ssd_model_is_priced_on_a_mac_or_names_the_formats_it_lacks():
    # The CPU probe measures a fixed set of ggml types. A model whose experts use another,
    # such as Flash Next's IQ4_XS and IQ4_NL, has no estimate and launches on default threads.
    for model in ssd_models():
        value = estimate(model)
        missing = unmeasured(model)
        if missing:
            assert value.tokens_per_second is None
            assert value.detail == "no CPU compute measurement of " + ", ".join(missing)
        else:
            assert value.tokens_per_second is not None and 0.05 < value.tokens_per_second < 100
            assert " ms a token: dense " in value.detail


def test_more_memory_and_a_faster_drive_are_faster():
    faster = copy.deepcopy(MAC)
    for row in faster["storage"][0]["random_read"]:
        row["Bps"] *= 2
    for model in ssd_models():
        if unmeasured(model):
            continue
        base = estimate(model).tokens_per_second
        assert estimate(model, available=40 * GIB).tokens_per_second > base
        assert estimate(model, characteristics=faster).tokens_per_second > base


def test_no_estimate_without_the_models_drive():
    value = estimate(catalog.MODELS[0], drive=None)
    assert value.tokens_per_second is None
    assert value.detail == "the drive holding the model is not known"


def test_qwen_estimate_uses_a_measured_rate_below_an_unreachable_knee():
    c = load(Path(__file__).parent / "fixtures/machine/windows-x64-igpu.json")
    storage = c["storage"][0]
    storage["qd_cap"] = None
    storage["knee"] = [{"bytes": 1 << 20, "qd": 32}]
    template = next(row for row in storage["random_read"] if row["bytes"] == 1 << 20)
    storage["random_read"] = [
        dict(template, qd=qd, Bps=rate) for qd, rate in [(8, 6e9), (16, 7e9), (32, 9e9)]
    ]
    model = catalog.MODELS[2]
    facts = RuntimeFacts(tuple(row["id"] for row in c["cpu"]["logical"]), (), 9 * GIB)
    value = decode_estimate(
        c, model.facts, model.dense_read_bytes, facts, "windows", storage["drive_key"]
    )
    assert value.tokens_per_second is not None and value.tokens_per_second > 0
    assert "from the drive at 7.00 GB/s" in value.detail
    storage["knee"][0]["qd"] = 8
    assert value == decode_estimate(
        c, model.facts, model.dense_read_bytes, facts, "windows", storage["drive_key"]
    )


def test_dense_weights_that_do_not_fit_are_read_from_the_drive():
    muse = catalog.MODELS[1]  # 21 GB of dense weights, more than the 16 GiB Mac has
    spilled = estimate(muse, available=9 * GIB)
    fits = estimate(muse, available=64 * GIB)
    assert "GB of it from the drive" in spilled.detail and "of it from the drive" not in fits.detail
    assert spilled.tokens_per_second < fits.tokens_per_second


def test_no_available_memory_is_none_rather_than_unknown():
    model = catalog.MODELS[0]
    assert estimate(model, available=0).tokens_per_second < estimate(model).tokens_per_second
    assert "(a 0.0 GB cache of" in estimate(model, available=0).detail


def test_a_profile_of_other_cpus_gives_a_reason_not_an_error():
    value = decode_estimate(
        MAC, catalog.MODELS[0].facts, 1, RuntimeFacts((99,), (), 9 * GIB), "darwin", DRIVE
    )
    assert value.tokens_per_second is None and "none of the CPUs" in value.detail
