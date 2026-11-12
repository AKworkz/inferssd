"""One ordinary RAM launch policy shared by Mac and Windows, without a model download."""

import json
from dataclasses import replace
from functools import partial

import pytest
from test_engine_runtime import policy
from test_strategy_inputs import F32, write_gguf
from test_typed_launch import setup

from inferssd.engine import runtime
from inferssd.launch import ram
from inferssd.launch.machine import MachineSource
from inferssd.launch.prepare import prepare
from inferssd.launch.typed import resolve_typed
from inferssd.launch.types import Threads
from inferssd.source.header import Header, Tensor


def gemma(tmp_path, platform, backends):
    inputs, spec, host, adapter, out, indexed = setup(tmp_path, platform)
    write_gguf(
        spec.model,
        [("token_embd.weight", (32, 32), F32)],
        {
            "general.architecture": "gemma4",
            "gemma4.block_count": 4,
            "gemma4.attention.shared_kv_layers": 1,
            "gemma4.attention.head_count_kv": 1,
            "gemma4.attention.key_length": 64,
            "gemma4.attention.value_length": 64,
        },
    )
    # The application's existing global SSD settings are valid inputs, but must
    # not activate an expert store or backend-specific tier for this RAM model.
    with open(spec.settings_file, "w", encoding="utf-8") as stream:
        stream.write('inferssd-settings 1\nexpert.source=["missing.tsv"]\nvram.budget=4096\n')
    host["engine_gpu_backends"] = backends
    host["runtime_facts"] = {"cpus": [0, 1], "gpus": [], "available_bytes": 16 << 30}
    host["cpu_topology"] = {
        "affinity_supported": platform != "darwin",
        "clusters": [{"kind": "performance", "cpus": [0]}, {"kind": "efficiency", "cpus": [1]}],
    }
    inputs = replace(inputs, profile="")
    return inputs, replace(spec, strategy=True), host, adapter, out, indexed


def resolved(inputs, spec, adapter, out, platform, memory=8 << 30, reserve=0):
    return prepare(
        inputs,
        resolver=partial(
            resolve_typed,
            spec=spec,
            platform=platform,
            directory=str(out),
            memory=memory,
            reserve=reserve,
        ),
        observe=adapter,
        emit=lambda _: None,
    )


@pytest.mark.parametrize(
    "platform,backends,mapping,layers",
    [
        ("darwin", ["Metal"], "--no-mmap", "99"),
        ("darwin", [], "--mmap", "0"),
        ("windows", ["Vulkan"], "--no-mmap", "99"),
        ("windows", [], "--mmap", "0"),
    ],
)
def test_same_ram_profile_uses_existing_native_backends(
    tmp_path, platform, backends, mapping, layers
):
    inputs, spec, _, adapter, out, indexed = gemma(tmp_path, platform, backends)
    result = resolved(inputs, spec, adapter, out, platform)
    assert result.code == 0 and indexed == []
    argv = list(result.argv)
    assert mapping in argv and "--mlock" in argv and "--no-repack" in argv
    assert argv[argv.index("-ngl") + 1] == layers
    assert "--defer-experts" not in argv
    assert not any(x.startswith("--ssd-") for x in argv)
    assert not (out / "settings.ssd").exists() and not (out / "artifacts").exists()
    assert not {"machine_characteristics", "gguf_alias", "model_drive_key"} & {
        fact.name for fact in result.observations
    }
    receipt = json.loads(result.outputs[-1].text)
    assert receipt["profile"] == "gemma4" and receipt["mode"] == "ram"
    assert receipt["memory"]["required_bytes"] + receipt["memory"]["headroom_bytes"] < 16 << 30


@pytest.mark.parametrize("available", [None, 2 << 30])
@pytest.mark.parametrize("platform", ["darwin", "windows"])
def test_ram_launch_refuses_unknown_or_insufficient_available_memory(tmp_path, platform, available):
    inputs, spec, host, adapter, out, indexed = gemma(tmp_path, platform, [])
    host["runtime_facts"]["available_bytes"] = available
    with pytest.raises(ValueError, match="RAM-only model"):
        resolved(inputs, spec, adapter, out, platform)
    assert not indexed and list(out.iterdir()) == []


def test_ram_launch_respects_the_policy_reserve_and_budget(tmp_path):
    inputs, spec, host, adapter, out, _ = gemma(tmp_path, "windows", [])
    host["runtime_facts"]["available_bytes"] = 4 << 30
    with pytest.raises(ValueError, match="3.00 GiB headroom"):
        resolved(inputs, spec, adapter, out, "windows", reserve=3 << 30)
    with pytest.raises(ValueError, match="budget 1.00"):
        resolved(inputs, spec, adapter, out, "windows", memory=1 << 30)


def test_ram_launch_steps_an_unset_context_down_to_what_fits(tmp_path):
    inputs, spec, _, adapter, out, _ = gemma(tmp_path, "darwin", [])
    # 768 bytes of KV a token (3 layers, 256 bytes each): 24576 tokens fit beside the workspace.
    memory = ram.WORKSPACE_BYTES + (20 << 20)
    result = resolved(inputs, spec, adapter, out, "darwin", memory=memory)
    assert result.code == 0
    argv = list(result.argv)
    assert argv[argv.index("-c") + 1] == "24576"
    assert json.loads(result.outputs[-1].text)["launch"]["context"] == 24576


def test_ram_launch_refuses_a_given_context_that_does_not_fit(tmp_path):
    inputs, spec, _, adapter, out, _ = gemma(tmp_path, "darwin", [])
    with pytest.raises(ValueError, match="RAM-only model"):
        resolved(
            inputs,
            replace(spec, context=32768),
            adapter,
            out,
            "darwin",
            memory=ram.WORKSPACE_BYTES + (20 << 20),
        )


@pytest.mark.parametrize(
    "extra",
    [
        ("--cache_type_k=f32",),
        ("-ctkf32",),
        ("--no-mlock",),
        ("--ctx-checkpoints", "32"),
        ("--swa-checkpoints", "32"),
        ("--swa_full",),
        ("--override-kv", "gemma4.attention.shared_kv_layers=int:0"),
        ("--model-draft", "draft.gguf"),
        ("-md", "draft.gguf"),
        ("--spec-draft-model", "draft.gguf"),
        ("--mmproj", "projector.gguf"),
        ("--lora", "adapter.gguf"),
    ],
)
def test_ram_capacity_options_cannot_be_overridden_behind_the_fit_check(tmp_path, extra):
    inputs, spec, _, adapter, out, _ = gemma(tmp_path, "windows", [])
    with pytest.raises(ValueError, match="RAM profile owns|explicit typed launch"):
        resolved(replace(inputs, extra=extra), spec, adapter, out, "windows")


@pytest.mark.parametrize(
    "variable", ["LLAMA_ARG_SWA_FULL", "LLAMA_ARG_SPEC_DRAFT_MODEL", "LLAMA_ARG_MMPROJ"]
)
def test_ram_fit_cannot_be_bypassed_by_native_environment_defaults(tmp_path, variable):
    inputs, spec, _, adapter, out, _ = gemma(tmp_path, "windows", [])
    inputs = replace(inputs, environment=inputs.environment + ((variable, "1"),))
    with pytest.raises(ValueError, match="unbudgeted model or KV"):
        resolved(inputs, spec, adapter, out, "windows")


@pytest.mark.parametrize("platform", ["windows", "darwin"])
def test_ram_cpu_defaults_to_fast_cluster_without_ssd_measurements(tmp_path, platform):
    inputs, spec, host, adapter, out, _ = gemma(tmp_path, platform, [])
    fast = [0, 1, 10, 11, 12, 13]
    host["runtime_facts"]["cpus"] = list(range(16))
    host["cpu_topology"]["clusters"] = [{"kind": "performance", "cpus": fast}]
    result = resolved(inputs, spec, adapter, out, platform)
    argv = list(result.argv)
    assert argv[argv.index("-t") + 1] == "6" and argv[argv.index("-tb") + 1] == "6"
    if platform == "windows":
        assert argv[argv.index("-C") + 1] == "0x3c03"
        assert argv[argv.index("-Cb") + 1] == "0x3c03"
    else:
        assert "-C" not in argv and "-Cb" not in argv
    assert "machine_characteristics" not in {fact.name for fact in result.observations}


def test_ram_cpu_uses_only_allowed_fast_cpus_and_preserves_explicit_threads(tmp_path):
    inputs, spec, host, adapter, out, _ = gemma(tmp_path, "windows", [])
    host["runtime_facts"]["cpus"] = [1, 2, 3]
    host["cpu_topology"]["clusters"] = [{"kind": "performance", "cpus": [0, 1]}]
    result = resolved(inputs, spec, adapter, out, "windows")
    argv = list(result.argv)
    assert argv[argv.index("-t") + 1] == "1" and argv[argv.index("-C") + 1] == "0x2"
    given = replace(spec, threads=Threads(3, 3))
    del host["cpu_topology"]  # Explicit pools do not require a topology observation.
    result = resolved(inputs, given, adapter, out, "windows")
    argv = list(result.argv)
    assert argv[argv.index("-t") + 1] == "3" and "-C" not in argv


def test_ram_model_skips_ssd_machine_calibration(tmp_path, monkeypatch):
    inputs, spec, _, _, out, _ = gemma(tmp_path, "windows", [])
    monkeypatch.setattr(
        runtime, "machine_characteristics", lambda *_: pytest.fail("SSD calibration")
    )
    instance = runtime.Runtime(
        inputs,
        policy(out, "windows"),
        host=object(),
        emit=lambda _: None,
        configuration=spec,
        machine=MachineSource(),
    )
    assert instance.characteristics(spec) is None


def test_gemma_header_accounts_shared_kv_and_sliding_windows():
    # E4B: the last18 layers share KV; of the first24,20 slide and4 keep full context.
    kv = {
        "general.architecture": "gemma4",
        "gemma4.block_count": 42,
        "gemma4.attention.shared_kv_layers": 18,
        "gemma4.attention.head_count_kv": 2,
        "gemma4.attention.key_length": 512,
        "gemma4.attention.value_length": 512,
        "gemma4.attention.sliding_window": 512,
        "gemma4.attention.key_length_swa": 256,
        "gemma4.attention.value_length_swa": 256,
        "gemma4.attention.sliding_window_pattern": [True] * 5 + [False],
    }
    kv["gemma4.attention.sliding_window_pattern"] *= 7
    header = Header(
        "gemma.gguf",
        3,
        kv,
        [Tensor("token_embd.weight", F32, (32, 32), 4096, 4096, "gemma.gguf")],
        32,
        4096,
    )
    metadata = ram.facts([header])
    fit = ram.admission(
        metadata, context=8192, batch=512, gpu=False, available=16 << 30, budget=8 << 30, headroom=0
    )
    assert fit["kv_bytes"] == 168 << 20  # The native ordinary engine's F16 KV allocation.
