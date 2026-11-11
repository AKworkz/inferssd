"""An explicit reviewed bridge from the one accepted legacy main to typed settings.

Bindings are trusted controller inputs, pinned by the runner-config digest. This
module checks their observations; it does not infer environment translations or
turn operator-reviewed source defaults into claimed runtime telemetry.
"""

import copy
import hashlib
import json
import subprocess
from pathlib import Path, PurePosixPath

from ci.contract import HASH, MODELS, revision

LEGACY_REVISION = "f5e6e26ff26d2014cccef0118048039b88163f54"
SETTING_KEYS = frozenset((
    "expert.source", "cpu.io_cpus", "cpu.decode_participants", "prefetch.depth",
    "place.costs", "prefill.lookahead", "vram.budget", "vram.max_slots",
    "vram.reserve_mib", "host.headroom_mib", "host.budget_mib",
    "model.engram_prefetch", "model.hc_fused",
))
SETTING_GENERATIONS = (SETTING_KEYS,)


def identity(value):
    # Unlike Python equality, JSON identity distinguishes true from 1.
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def text(value):
    return isinstance(value, str) and bool(value.strip()) and "\0" not in value


def hash_value(value):
    return isinstance(value, str) and HASH.fullmatch(value) is not None


def relative(value):
    return (text(value) and not PurePosixPath(value).is_absolute()
            and ".." not in PurePosixPath(value).parts and "\\" not in value
            and value not in (".", "") and str(PurePosixPath(value)) == value)


def validate_semantics(values):
    from inferssd.configuration.settings import schema, typed

    properties = schema()["properties"]
    if frozenset(properties) not in SETTING_GENERATIONS or frozenset(values) not in SETTING_GENERATIONS:
        raise ValueError("reference adapter requires the explicit reviewed settings schema")
    if not set(values) <= set(properties):
        raise ValueError("installed settings registry cannot validate this binding generation")
    for key, value in values.items():
        prop = properties[key]
        kind = prop["kind"]
        if kind == "cpus":
            if not isinstance(value, list):
                raise ValueError("canonical CPU lists must remain arrays")
            literal = ",".join(str(item) for item in value)
        elif kind == "costs":
            if not isinstance(value, dict):
                raise ValueError("canonical costs must remain an object")
            literal = ",".join(str(name) + "=" + str(cost) for name, cost in value.items())
        else:
            literal = json.dumps(value, allow_nan=False)
        parsed = typed(prop, literal)
        canonical = dict(parsed) if kind == "costs" else list(parsed) if kind in ("cpus", "paths") else parsed
        if identity(canonical) != identity(value):
            raise ValueError("canonical setting type/value differs from registry: " + key)


def image_map(images):
    if not isinstance(images, list) or not images:
        raise ValueError("reference binding needs the complete actual image map")
    paths = set()
    for item in images:
        if (not isinstance(item, dict) or set(item) != {"path", "sha256", "role"}
                or not text(item["path"]) or not PurePosixPath(item["path"]).is_absolute()
                or ".." in PurePosixPath(item["path"]).parts
                or not hash_value(item["sha256"]) or not text(item["role"])
                or item["path"] in paths):
            raise ValueError("invalid or duplicate full-path image role binding")
        paths.add(item["path"])
    return images


def validate_build(spec):
    if (not isinstance(spec, dict)
            or set(spec) != {"server_target", "binary_relative", "image_relatives", "template_relatives"}
            or spec["server_target"] not in ("llama-server", "inferssd", "inferssd-server")
            or not relative(spec["binary_relative"])
            or not isinstance(spec["image_relatives"], list) or not spec["image_relatives"]
            or any(not relative(path) for path in spec["image_relatives"])
            or len(set(spec["image_relatives"])) != len(spec["image_relatives"])
            or not isinstance(spec["template_relatives"], dict)
            or not spec["template_relatives"]
            or not set(spec["template_relatives"]) <= {*MODELS, "tiny"}
            or any(not relative(path) for path in spec["template_relatives"].values())):
        raise ValueError("reference build needs reviewed targets and safe relative images/templates")


def launch_policy(value, *, typed):
    required = {"args", "environment", "template_sha256", "alias_sha256"}
    if typed:
        required.add("typed_values")
    if (not isinstance(value, dict) or set(value) != required
            or not isinstance(value["args"], list) or any(not text(x) for x in value["args"])
            or not isinstance(value["environment"], dict)
            or any(not text(k) or not isinstance(v, str) or "\0" in v for k, v in value["environment"].items())
            or not hash_value(value["template_sha256"])
            or not isinstance(value["alias_sha256"], list) or not value["alias_sha256"]
            or any(not hash_value(x) for x in value["alias_sha256"])):
        raise ValueError("reference requires the original complete launch policy")


def validate_mapping(binding, settings=None):
    mapping = binding.get("legacy_mapping")
    if not isinstance(mapping, dict) or set(mapping) != set(binding["semantic_settings"]):
        raise ValueError("legacy mapping must explicitly cover every bound setting")
    policy = binding["sides"]["base"]["launch_policy"]
    for key, item in mapping.items():
        if (not isinstance(item, dict) or set(item) != {"value", "inputs", "explanation"}
                or identity(item["value"]) != identity(binding["semantic_settings"][key])
                or not text(item["explanation"])
                or not isinstance(item["inputs"], list) or not item["inputs"]):
            raise ValueError("missing explicit reviewed legacy mapping for " + key)
        for source in item["inputs"]:
            if not isinstance(source, dict):
                raise ValueError("legacy mapping input must be an object")
            kind = source.get("kind")
            if kind == "environment":
                if (set(source) != {"kind", "name", "value"}
                        or source.get("name") not in policy["environment"]
                        or identity(source["value"]) != identity(policy["environment"][source["name"]])):
                    raise ValueError("legacy environment mapping differs from actual launch policy")
            elif kind == "argument":
                index = source.get("index")
                if (set(source) != {"kind", "index", "value"} or type(index) is not int
                        or not 0 <= index < len(policy["args"])
                        or identity(source["value"]) != identity(policy["args"][index])):
                    raise ValueError("legacy argv mapping differs from actual launch policy")
            elif kind == "effective-setting":
                if (set(source) != {"kind", "name", "value"} or not text(source.get("name"))
                        or source["name"].startswith("native_")
                        or settings is not None and (
                            source["name"] not in settings
                            or identity(settings[source["name"]]) != identity(source["value"]))):
                    raise ValueError("legacy observed setting differs from its mapping")
            elif kind == "source-default":
                if (set(source) != {"kind", "revision", "path", "sha256"}
                        or source["revision"] != LEGACY_REVISION
                        or not relative(source["path"]) or not hash_value(source["sha256"])):
                    raise ValueError("reviewed defaults require exact legacy source provenance")
            else:
                raise ValueError("unknown legacy mapping input kind")


def validate_binding(binding):
    if (not isinstance(binding, dict) or set(binding) != {
            "schema_version", "kind", "semantic_settings", "legacy_mapping", "sides"}
            or binding["schema_version"] != 1 or binding["kind"] != "legacy-to-typed"
            or not isinstance(binding["semantic_settings"], dict)
            or frozenset(binding["semantic_settings"]) not in SETTING_GENERATIONS
            or not isinstance(binding["sides"], dict) or set(binding["sides"]) != {"base", "candidate"}):
        raise ValueError("explicit legacy-to-typed binding with every reviewed setting required")
    identity(binding)
    validate_semantics(binding["semantic_settings"])
    for side, spec in binding["sides"].items():
        if (not isinstance(spec, dict) or set(spec) != {
                "revision", "profile_sha256", "launch_policy", "images", "build"}
                or not hash_value(spec["profile_sha256"])):
            raise ValueError("each reference side requires exact source/profile/image/build identities")
        revision(spec["revision"])
        if (side == "base" and spec["revision"] != LEGACY_REVISION
                or side == "candidate" and spec["revision"] == LEGACY_REVISION):
            raise ValueError("legacy adaptation is only allowed for the pinned accepted base")
        launch_policy(spec["launch_policy"], typed=side == "candidate")
        image_map(spec["images"])
        validate_build(spec["build"])
    if identity(binding["sides"]["candidate"]["launch_policy"]["typed_values"]) != identity(binding["semantic_settings"]):
        raise ValueError("candidate typed snapshot differs from the complete reviewed semantic values")
    if ({item["role"] for item in binding["sides"]["base"]["images"]}
            != {item["role"] for item in binding["sides"]["candidate"]["images"]}):
        raise ValueError("reviewed library roles differ across the comparison")
    validate_mapping(binding)
    return binding


def build_spec(binding, side, source_revision):
    """Return only the reviewed per-side layout, without modifying a build config."""
    validate_binding(binding)
    if side not in ("base", "candidate") or binding["sides"][side]["revision"] != source_revision:
        raise ValueError("reference build layout belongs to another source revision")
    return copy.deepcopy(binding["sides"][side]["build"])


def verify_source_defaults(binding, source):
    """Verify cited default source bytes before launching the untouched old build.

    The meaning of those bytes remains an operator review, not native telemetry.
    This read-only helper runs no candidate program and changes no source files.
    """
    validate_binding(binding)
    source = Path(source).resolve(strict=True)
    if subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip() != LEGACY_REVISION:
        raise ValueError("default evidence checkout is not the pinned legacy revision")
    files = {}
    for item in binding["legacy_mapping"].values():
        for entry in item["inputs"]:
            if entry["kind"] != "source-default":
                continue
            if entry["path"] in files:
                if files[entry["path"]] != entry["sha256"]:
                    raise ValueError("conflicting reviewed hashes for a legacy source path")
                continue
            path = (source / entry["path"]).resolve(strict=True)
            if not path.is_relative_to(source) or not path.is_file():
                raise ValueError("default evidence escapes the pinned source checkout")
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
            original = subprocess.check_output(["git", "-C", str(source), "show", LEGACY_REVISION + ":" + entry["path"]])
            if actual != entry["sha256"] or hashlib.sha256(original).hexdigest() != actual:
                raise ValueError("reviewed legacy default source bytes changed")
            files[entry["path"]] = actual
    return {"revision": LEGACY_REVISION, "files": files}


def comparable(receipt, binding, side):
    validate_binding(binding)
    if side not in ("base", "candidate"):
        raise ValueError("explicit reference side required")
    spec, host = binding["sides"][side], receipt["host"]
    settings = dict(host["effective_settings"])
    if (receipt.get("revision") != spec["revision"] or receipt["build"].get("revision") != spec["revision"]
            or settings.get("native_profile_sha256") != spec["profile_sha256"]
            or identity(settings.get("native_launch_policy")) != identity(spec["launch_policy"])):
        raise ValueError("observed reference source/profile/launch policy differs from reviewed binding")
    if side == "base":
        validate_mapping(binding, settings)
    actual = receipt["build"]["loaded_libraries"]
    expected = [{"path": item["path"], "sha256": item["sha256"]} for item in spec["images"]]
    if (not isinstance(actual, list)
            or identity(sorted(actual, key=lambda item: item["path"]))
            != identity(sorted(expected, key=lambda item: item["path"]))):
        raise ValueError("actual loaded image paths/hashes differ from the reviewed role map")
    settings.pop("native_profile_sha256")
    settings.pop("native_launch_policy")
    return {
        "model": receipt["model_identity"],
        "host": {k: host[k] for k in ("hardware", "os", "memory_budget_bytes", "os_page_cache_treatment")},
        "settings": settings,
        "semantic_settings": binding["semantic_settings"],
        "template_sha256": spec["launch_policy"]["template_sha256"],
        "alias_sha256": spec["launch_policy"]["alias_sha256"],
        "image_roles": sorted({item["role"] for item in spec["images"]}),
    }
