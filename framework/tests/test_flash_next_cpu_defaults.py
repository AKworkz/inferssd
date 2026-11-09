"""Unpriced Flash Next CPU launches share the fast-cluster default on both desktops."""

from dataclasses import replace
from pathlib import Path

import pytest
from test_ram_launch import resolved
from test_strategy_inputs import Q8_0, moe, write_gguf
from test_typed_launch import setup

from inferssd.launch.types import Threads


def flash(tmp_path, platform, backends=()):
    inputs, spec, host, adapter, out, _ = setup(tmp_path, platform)
    write_gguf(
        spec.model,
        *moe(
            "qwen4exp",
            layers=1,
            experts=512,
            used=10,
            latent=32,
            hidden=32,
            gate=Q8_0,
            up=Q8_0,
            down=Q8_0,
        ),
    )
    host["engine_gpu_backends"] = list(backends)
    host["runtime_facts"] = {"cpus": list(range(16)), "gpus": [], "available_bytes": 16 << 30}
    host["cpu_topology"] = {
        "affinity_supported": platform != "darwin",
        "clusters": [
            {"kind": "performance", "cpus": [0, 1, 10, 11, 12, 13]},
            {"kind": "efficiency", "cpus": [2, 3, 4, 5, 6, 7, 8, 9, 14, 15]},
        ],
    }
    return replace(inputs, profile=""), replace(spec, strategy=True), host, adapter, out


@pytest.mark.parametrize("platform", ["windows", "darwin"])
def test_unpriced_cpu_uses_fast_cluster_without_changing_ssd_mode(tmp_path, platform):
    inputs, spec, _, adapter, out = flash(tmp_path, platform)
    result = resolved(inputs, spec, adapter, out, platform)
    argv = list(result.argv)
    assert argv[argv.index("-t") + 1] == "6" and argv[argv.index("-tb") + 1] == "6"
    assert "--ssd-settings" in argv and "--defer-experts" in argv
    assert (argv[argv.index("-C") + 1] if "-C" in argv else None) == (
        "0x3c03" if platform == "windows" else None
    )


@pytest.mark.parametrize("kind", ["explicit_threads", "cpu_priors", "gpu"])
def test_flash_preserves_explicit_or_priced_or_gpu_pool_choices(tmp_path, kind):
    inputs, spec, host, adapter, out = flash(
        tmp_path, "windows", ("Vulkan",) if kind == "gpu" else ()
    )
    del host["cpu_topology"]  # These paths must not apply an unmeasured topology override.
    if kind == "explicit_threads":
        spec = replace(spec, threads=Threads(3, 3, (0, 1, 2)))
    elif kind == "cpu_priors":
        with Path(spec.settings_file).open("a") as stream:
            stream.write('place.costs="cpu_fixed_us=1,cpu_route_us=2"\n')
    result = resolved(inputs, spec, adapter, out, "windows")
    argv = list(result.argv)
    if kind == "explicit_threads":
        assert argv[argv.index("-t") + 1] == "3" and argv[argv.index("-C") + 1] == "0x7"
    else:
        assert "-t" not in argv and "-C" not in argv
    assert "cpu_topology" not in {fact.name for fact in result.observations}
