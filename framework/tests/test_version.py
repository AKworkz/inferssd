import json

from inferssd.launch.typed import PROFILES
from inferssd.version import report


def test_supported_launch_profiles_do_not_depend_on_descriptor_build(tmp_path):
    path = tmp_path / "resources.json"
    path.write_text(json.dumps({"build": 224, "branch": "windows"}))
    value = report({"INFERSSD_RESOURCES": str(path)})
    assert value["build"] == 224
    assert value["capabilities"]["model_profiles"] == list(PROFILES)
    assert "gemma4" in value["capabilities"]["model_profiles"]
    assert "qwen4exp" in value["capabilities"]["model_profiles"]


def test_supported_profiles_are_reported_without_an_installed_descriptor(tmp_path):
    value = report({"INFERSSD_RESOURCES": str(tmp_path / "missing.json")})
    assert value["build"] is None
    assert value["capabilities"]["model_profiles"] == list(PROFILES)


def test_gpu_backends_come_from_the_descriptor(tmp_path):
    path = tmp_path / "resources.json"
    path.write_text(json.dumps({"build": 225, "capabilities": {"gpu_backends": ["CUDA"]}}))
    assert report({"INFERSSD_RESOURCES": str(path)})["capabilities"]["gpu_backends"] == ["CUDA"]
    path.write_text(json.dumps({"build": 225, "capabilities": {"gpu_backends": []}}))
    assert report({"INFERSSD_RESOURCES": str(path)})["capabilities"]["gpu_backends"] == []


def test_gpu_backends_are_left_out_when_not_recorded_or_not_known(tmp_path):
    path = tmp_path / "resources.json"
    for capabilities in (
        {},
        {"gpu_backends": ["OpenCL"]},
        {"gpu_backends": [["CUDA"]]},
        {"gpu_backends": "CUDA"},
    ):
        path.write_text(json.dumps({"build": 225, "capabilities": capabilities}))
        assert "gpu_backends" not in report({"INFERSSD_RESOURCES": str(path)})["capabilities"]
    assert (
        "gpu_backends"
        not in report({"INFERSSD_RESOURCES": str(tmp_path / "missing.json")})["capabilities"]
    )
