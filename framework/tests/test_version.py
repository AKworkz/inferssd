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
