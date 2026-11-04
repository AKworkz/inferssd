"""Installed deployments resolve resources without a checkout or cwd discovery."""

import json
from dataclasses import replace

import pytest
from test_typed_launch import resolve, setup

from inferssd.launch.resources import read_resources


def descriptor(tmp_path, spec, profile="muse"):
    path = tmp_path / "engine.json"
    path.write_text(
        json.dumps(
            {
                "schema": "inferssd-engine 1",
                "binary": "repo/bin/server",
                "templates": {profile: "repo/engine/models/templates/muse-spark.jinja"},
                "capabilities": {"settings": "inferssd-settings 1", "telemetry": 1},
                "probes": {"io": "bin/inferssd-ioreplay"},
                "revision": "fixture",
            }
        )
    )
    return path


@pytest.mark.parametrize("platform", ["linux", "darwin"])
def test_deployed_resources_do_not_need_checkout(tmp_path, monkeypatch, platform):
    inputs, spec, _, adapter, out, _ = setup(tmp_path, platform)
    resources = read_resources(descriptor(tmp_path, spec))
    foreign = tmp_path / "unrelated-cwd"
    foreign.mkdir()
    monkeypatch.chdir(foreign)
    inputs = replace(inputs, repo="")
    adapter.inputs = inputs
    result = resolve(
        inputs, replace(spec, binary=None, resources=resources), adapter, out, platform
    )
    assert result.code == 0
    assert result.argv[0] == spec.binary
    assert result.argv[result.argv.index("--chat-template-file") + 1] == resources.template("muse")
    assert all("unrelated-cwd" not in argument for argument in result.argv)


def test_explicit_paths_override_descriptor(tmp_path):
    inputs, spec, _, adapter, out, _ = setup(tmp_path)
    resources = replace(read_resources(descriptor(tmp_path, spec)), binary="/missing")
    explicit = tmp_path / "custom.jinja"
    explicit.write_text("custom")
    result = resolve(
        replace(inputs, repo=""),
        replace(spec, template=str(explicit), resources=resources),
        adapter,
        out,
    )
    assert result.code == 0 and result.argv[0] == spec.binary
    assert str(explicit) in result.argv


@pytest.mark.parametrize(
    "capabilities",
    [{}, {"settings": "old", "telemetry": 1}, {"settings": "inferssd-settings 1", "telemetry": 2}],
)
def test_incompatible_capabilities_refuse_before_preparation(tmp_path, capabilities):
    inputs, spec, _, _, out, _ = setup(tmp_path)
    path = descriptor(tmp_path, spec)
    value = json.loads(path.read_text())
    value["capabilities"] = capabilities
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        read_resources(path)
    assert not list(out.iterdir())


def test_descriptor_without_a_profile_template_launches_with_the_gguf_template(tmp_path):
    inputs, spec, _, adapter, out, _ = setup(tmp_path)
    resources = read_resources(descriptor(tmp_path, spec))
    result = resolve(
        replace(inputs, repo="", profile="qwen4exp"),
        replace(spec, binary=None, resources=resources),
        adapter,
        out,
    )
    assert result.code == 0 and result.argv[0] == spec.binary
    assert "--chat-template-file" not in result.argv


def test_windows_descriptor_binds_only_a_windows_launch(tmp_path):
    inputs, spec, _, adapter, out, _ = setup(tmp_path)
    path = descriptor(tmp_path, spec)
    value = json.loads(path.read_text())
    value["capabilities"]["platform"] = "windows"
    path.write_text(json.dumps(value))
    resources = read_resources(path)
    assert resources.platform == "windows"
    with pytest.raises(ValueError, match="resource platform differs"):
        resolve(inputs, replace(spec, binary=None, resources=resources), adapter, out)


def test_a_descriptor_names_the_gpu_backends_its_engine_was_built_with(tmp_path):
    _, spec, _, _, _, _ = setup(tmp_path)
    path = descriptor(tmp_path, spec)
    assert read_resources(path).gpu_backends is None
    value = json.loads(path.read_text())
    cases = (([], ()), (["CUDA"], ("CUDA",)), (["Metal", "Vulkan"], ("Metal", "Vulkan")))
    for listed, read in cases:
        value["capabilities"]["gpu_backends"] = listed
        path.write_text(json.dumps(value))
        assert read_resources(path).gpu_backends == read
    for listed in (["ROCm"], ["CUDA", "CUDA"], "CUDA"):
        value["capabilities"]["gpu_backends"] = listed
        path.write_text(json.dumps(value))
        with pytest.raises(ValueError, match="gpu_backends"):
            read_resources(path)


def test_the_descriptors_gpu_backends_decide_whether_the_cuda_tier_needs_nvidia_smi(tmp_path):
    # The recorded build has CUDA; the descriptor of the engine launched speaks for it.
    inputs, spec, host, adapter, _, _ = setup(tmp_path)
    host["gpu_available"] = False
    path = descriptor(tmp_path, spec)
    value = json.loads(path.read_text())
    for listed, refused in (([], False), (["Vulkan"], False), (["CUDA"], True)):
        value["capabilities"]["gpu_backends"] = listed
        path.write_text(json.dumps(value))
        launch = replace(spec, binary=None, resources=read_resources(path))
        out = tmp_path / ("run-" + "-".join(listed))
        out.mkdir()
        if refused:
            with pytest.raises(ValueError, match="requires nvidia-smi"):
                resolve(replace(inputs, repo=""), launch, adapter, out)
        else:
            assert resolve(replace(inputs, repo=""), launch, adapter, out).code == 0


@pytest.mark.parametrize("platform", ["linux", "darwin"])
def test_a_cuda_tier_setting_needs_an_engine_with_cuda(tmp_path, platform):
    inputs, spec, host, adapter, out, _ = setup(tmp_path, platform)
    launch = replace(spec, overrides=spec.overrides + (("vram.max_slots", "4"),))
    if platform == "linux":
        assert resolve(inputs, launch, adapter, out, platform).code == 0
    else:
        with pytest.raises(ValueError, match="vram.max_slots .* configures the CUDA expert tier"):
            resolve(inputs, launch, adapter, out, platform)
