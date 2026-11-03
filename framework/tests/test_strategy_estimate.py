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


def test_the_catalog_is_the_download_list_and_its_facts_add_up():
    assert hub.MODELS == {model.repo: model.quantization for model in catalog.MODELS}
    for model in catalog.MODELS:
        facts = model.facts
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


def test_every_supported_model_is_priced_on_a_mac():
    for model in catalog.MODELS:
        value = estimate(model)
        assert value.tokens_per_second is not None and 0.05 < value.tokens_per_second < 100
        assert " ms a token: dense " in value.detail


def test_more_memory_and_a_faster_drive_are_faster():
    faster = copy.deepcopy(MAC)
    for row in faster["storage"][0]["random_read"]:
        row["Bps"] *= 2
    for model in catalog.MODELS:
        base = estimate(model).tokens_per_second
        assert estimate(model, available=40 * GIB).tokens_per_second > base
        assert estimate(model, characteristics=faster).tokens_per_second > base


def test_no_estimate_without_the_models_drive():
    value = estimate(catalog.MODELS[0], drive=None)
    assert value.tokens_per_second is None
    assert value.detail == "the drive holding the model is not known"


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
