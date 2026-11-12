"""Small trusted impact rules. Paths are data; a plan never supplies commands."""

import hashlib
import json
from pathlib import PurePosixPath
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ci.contract import DEVICES, MODELS, revision


RULES_VERSION = 2
SELECTION = "impact-selection"
BASELINE = "baseline-history"
CUDA = DEVICES[:2]
MAC = (DEVICES[2],)
# Each ID resolves to a fixed trusted invocation in staged_numerical.py.
CASES = {
    "backend-add": ("test-backend-ops", "ADD", "type=f32"),
    "moe-mxfp4": ("test-backend-ops", "MUL_MAT_ID", "type_a=mxfp4"),
    "moe-q8": ("test-backend-ops", "MUL_MAT_ID", "type_a=q8_0"),
    "moe-q4": ("test-backend-ops", "MUL_MAT_ID", "type_a=q4_K"),
    "moe-q5": ("test-backend-ops", "MUL_MAT_ID", "type_a=q5_K"),
    "normalization": ("test-backend-ops", "RMS_NORM", ""),
    "activation": ("test-backend-ops", "SWIGLU", ""),
    "dsv-hc": ("test-backend-ops", "DSV4_HC_COMB", ""),
    "cuda-generic-tier": ("test-ssd-vram-generic", None, None),
    "model-adapter": ("test-ssd-model-adapter", None, None),
    "runtime-policy": ("test-ssd-runtime-policy", None, None),
}
MARKERS = {"cuda-generic-tier": b"REAL_GPU_GENERIC_TIER_PASS ",
           "model-adapter": b"model adapter: ",
           "runtime-policy": b"RUNTIME_POLICY_PASS "}


def fingerprint(value):
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()).hexdigest()


def paths_checked(paths):
    if not isinstance(paths, (list, tuple)):
        raise ValueError("complete changed path list required")
    for path in paths:
        if (not isinstance(path, str) or not path or "\\" in path or "\0" in path
                or path.startswith("/") or any(p in ("", ".", "..") for p in path.split("/"))):
            raise ValueError("canonical repository-relative changed paths required")
    return sorted(set(paths))


def documentation(path, policy):
    item = PurePosixPath(path)
    # Documentation in an executable/build directory is still only docs; a
    # disguised workflow/build file is never excluded by its parent directory.
    return (item.suffix in policy["documentation_suffixes"]
            or item.name in policy["documentation_names"])


def requirements(path):
    """Return devices, models, performance, long-context, reason.

    Specific intersections precede shared rules. Unknown code expands to all
    six cells. No substring match such as `muse` narrows an unknown shared file.
    """
    if path.startswith("bench/ci/macos-sandbox/") or path == "bench/ci/operator_mac_stage.py":
        return MAC, MODELS, True, False, "Mac isolated execution/resource/lease path"
    if path.startswith("engine/ggml/src/ggml-cuda/"):
        models = ("deepseek",) if ("mxfp4" in PurePosixPath(path).name
                                  or PurePosixPath(path).name.startswith("dsv4-")) else MODELS
        return CUDA, models, True, False, "CUDA executor/kernel/transport"
    if path.startswith(("engine/ggml/src/ggml-metal/", "bench/macos/")):
        return MAC, MODELS, path.startswith("engine/"), False, "Mac Metal executor"
    if path.startswith("engine/ggml/src/ggml-cpu/ssd/mac-"):
        return MAC, MODELS, True, False, "Mac shared source/cache"
    if path in ("engine/src/models/deepseek41.cpp", "engine/src/models/deepseek41.h"):
        return DEVICES, ("deepseek",), True, False, "DeepSeek model graph"
    if path in ("engine/src/models/muse.cpp", "engine/src/models/muse.h",
                "engine/models/templates/muse-spark.jinja"):
        return DEVICES, ("muse",), True, False, "Muse model graph/template"
    if path.startswith("engine/src/llama-kv-cache-dsv4"):
        return DEVICES, ("deepseek",), True, True, "DeepSeek compressed long-context state"
    if path.startswith(("bench/", ".github/", "tests/", "scripts/")):
        return DEVICES, MODELS, False, False, "shared test/control contract"
    if path.startswith(("engine/", "framework/")):
        return DEVICES, MODELS, True, False, "shared runtime/policy/public interface"
    return DEVICES, MODELS, True, False, "unknown code: conservative full coverage"


def numerical_cases(device, models):
    cases = {"backend-add", "normalization", "activation"}
    if "deepseek" in models:
        cases.update(("moe-mxfp4", "moe-q8"))
        if device in CUDA:
            # the tier's one engine family serves every format, DeepSeek's MXFP4 too
            cases.update(("dsv-hc", "cuda-generic-tier"))
    if "muse" in models:
        cases.update(("moe-q4", "moe-q5"))
        if device in CUDA:
            cases.add("cuda-generic-tier")
    return sorted(cases)


def plan(paths, policy, *, base, head, candidate, baseline=False):
    """Select cells from changed paths, or every cell for a baseline job.

    A baseline job measures one main commit on its own, so its diff is empty.
    It selects every device and model with the 512 performance tier.
    """
    paths = paths_checked(paths)
    identities = {k: revision(v) for k, v in (
        ("base", base), ("head", head), ("candidate", candidate))}
    if type(baseline) is not bool:
        raise ValueError("explicit baseline flag required")
    cells = {(d, m): dict(device=d, model=m, required=False, reasons=[],
                          numerical=[], functional=False,
                          performance=dict(required=False, tiers=[], output_tokens=256))
             for d in DEVICES for m in MODELS}
    if baseline:
        if paths or len(set(identities.values())) != 1:
            raise ValueError("a baseline job measures one exact main commit with an empty diff")
        for cell in cells.values():
            cell.update(required=True, functional=True)
            cell["reasons"].append(dict(rule="baseline history of one main commit"))
            cell["performance"].update(required=True, tiers=["512"])
    for path in paths:
        if documentation(path, policy):
            continue
        devices, models, perf, long, reason = requirements(path)
        for device in devices:
            for model in models:
                cell = cells[device, model]
                cell.update(required=True, functional=True)
                cell["reasons"].append(dict(path=path, rule=reason))
                if perf:
                    cell["performance"]["required"] = True
                    cell["performance"]["tiers"] = ["512", "4096"] if (
                        long or "4096" in cell["performance"]["tiers"]) else ["512"]
    for (device, model), cell in cells.items():
        if cell["required"]:
            cell["numerical"] = numerical_cases(device, [model])
            if any("model-adapter" in path for path in paths):
                cell["numerical"].append("model-adapter")
            if any("runtime-policy" in path or "runtime/policy" in path for path in paths):
                cell["numerical"].append("runtime-policy")
            cell["numerical"].sort()
        else:
            cell["reason"] = "not-required: no affected path"
    return dict(schema_version=1, purpose=BASELINE if baseline else SELECTION, rules_version=RULES_VERSION,
                repository=policy["repository"], **identities, changed_paths=paths,
                changed_paths_sha256=fingerprint(paths), cells=list(cells.values()),
                formal_samples=0)


def validate(value, policy):
    if not isinstance(value, dict):
        raise ValueError("impact selection plan required")
    expected = plan(value["changed_paths"], policy, baseline=value.get("purpose") == BASELINE, **{
        key: value[key] for key in ("base", "head", "candidate")})
    if value != expected:
        raise ValueError("selection differs from the trusted rules/source identities")
    return value


def selected(value, device=None):
    return [cell for cell in value["cells"] if cell["required"]
            and (device is None or cell["device"] == device)]


def device_requirements(value, device):
    if device not in DEVICES:
        raise ValueError("unadmitted CI device")
    cells = selected(value, device)
    return dict(models=[m for m in MODELS if any(c["model"] == m for c in cells)],
                numerical=sorted({case for c in cells for case in c["numerical"]}),
                performance={c["model"]: c["performance"] for c in cells
                             if c["performance"]["required"]})


if __name__ == "__main__":
    import argparse
    from ci.contract import load, policy_read, save
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("paths", "policy", "output", "base", "head", "candidate"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    save(args.output, plan(load(args.paths), policy_read(load(args.policy)),
                           base=args.base, head=args.head, candidate=args.candidate))
