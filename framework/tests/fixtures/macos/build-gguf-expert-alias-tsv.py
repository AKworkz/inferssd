#!/usr/bin/env python3
import argparse
import csv
import glob
import json
import os
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path


ALIGNMENT = 4096
EXPERT_RE = re.compile(r"^blk\.(?P<layer>\d+)\.ffn_(?P<kind>up|gate|down)_exps\.weight$")


def install_gguf_import(repo_root: Path) -> None:
    gguf_py = repo_root / "gguf-py"
    if str(gguf_py) not in sys.path:
        sys.path.insert(0, str(gguf_py))


def git_head(repo_root: Path) -> str:
    override = os.environ.get("SSD_LLAMA_REPO_HEAD")
    if override:
        return override
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def tensor_type_name(tensor) -> str:
    tt = getattr(tensor, "tensor_type", "")
    return getattr(tt, "name", str(tt))


def parse_model_paths(patterns: list[str]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        matched = [Path(p) for p in sorted(glob.glob(pattern))]
        if not matched:
            raise RuntimeError(f"model glob matched no files: {pattern}")
        paths.extend(matched)
    unique = []
    seen = set()
    for path in paths:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(path)
    return unique


def build_rows(model_paths: list[Path], include_kinds: set[str], n_experts: int, require_aligned: bool) -> tuple[list[dict[str, object]], dict[str, object]]:
    from gguf.gguf_reader import GGUFReader

    rows: list[dict[str, object]] = []
    seen_tensors = set()
    by_kind = defaultdict(lambda: {"tensors": 0, "entries": 0, "bytes": 0})
    by_type = defaultdict(lambda: {"tensors": 0, "entries": 0, "bytes": 0})
    unaligned_rows = 0
    total_bytes = 0

    for shard_index, model_path in enumerate(model_paths):
        resolved_path = model_path.resolve()
        if any(char in str(resolved_path) for char in "\t\r\n"):
            raise RuntimeError("expert source path cannot contain tabs or newlines")
        reader = GGUFReader(str(model_path))
        for tensor in reader.tensors:
            match = EXPERT_RE.match(tensor.name)
            if not match:
                continue
            kind = match.group("kind")
            if kind not in include_kinds:
                continue
            if tensor.name in seen_tensors:
                raise RuntimeError(f"duplicate expert tensor across shards: {tensor.name}")
            seen_tensors.add(tensor.name)
            if len(tensor.name.encode("utf-8")) >= 128:
                raise RuntimeError(f"tensor name too long for runtime key: {tensor.name}")
            shape = tuple(int(value) for value in tensor.shape)
            if len(shape) != 3 or shape[2] != n_experts:
                raise RuntimeError(f"{tensor.name}: expected 3D tensor with {n_experts} experts, got {shape}")
            tensor_bytes = int(tensor.n_bytes)
            if tensor_bytes <= 0 or tensor_bytes % n_experts != 0:
                raise RuntimeError(f"{tensor.name}: {tensor_bytes} bytes not divisible by n_experts={n_experts}")
            expert_bytes = tensor_bytes // n_experts
            tensor_type = tensor_type_name(tensor)
            base = int(tensor.data_offset)
            for expert_idx in range(n_experts):
                offset = base + expert_idx * expert_bytes
                aligned = (offset % ALIGNMENT == 0) and (expert_bytes % ALIGNMENT == 0)
                if not aligned:
                    unaligned_rows += 1
                    if require_aligned:
                        raise RuntimeError(
                            f"{tensor.name}:{expert_idx}: unaligned offset={offset} nbytes={expert_bytes}"
                        )
                rows.append({
                    "tensor": tensor.name,
                    "expert_idx": expert_idx,
                    "nbytes": expert_bytes,
                    "source_path": str(resolved_path),
                    "offset": offset,
                    "kind": kind,
                    "layer": int(match.group("layer")),
                    "type": tensor_type,
                    "aligned": 1 if aligned else 0,
                    "shard_index": shard_index,
                })
            for bucket in (by_kind[kind], by_type[tensor_type]):
                bucket["tensors"] += 1
                bucket["entries"] += n_experts
                bucket["bytes"] += tensor_bytes
            total_bytes += tensor_bytes

    if not rows:
        raise RuntimeError("no expert tensors matched the requested kinds")
    rows.sort(key=lambda r: (str(r["tensor"]), int(r["expert_idx"]), int(r["nbytes"])))
    summary = {
        "model_paths": [str(p) for p in model_paths],
        "model_shards": len(model_paths),
        "expert_tensors": len(seen_tensors),
        "entries": len(rows),
        "n_experts": n_experts,
        "include_kinds": sorted(include_kinds),
        "total_payload_bytes": total_bytes,
        "total_payload_gib": total_bytes / 1024**3,
        "unaligned_rows": unaligned_rows,
        "by_kind": dict(sorted(by_kind.items())),
        "by_type": dict(sorted(by_type.items())),
    }
    return rows, summary


def write_tsv(path: Path, rows: list[dict[str, object]]) -> None:
    fields = ["tensor", "expert_idx", "nbytes", "source_path", "offset", "kind", "layer", "type", "aligned", "shard_index"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a prompt-independent GGUF-offset expert alias TSV.")
    parser.add_argument("--model-glob", action="append", required=True, help="GGUF shard glob; may be repeated.")
    parser.add_argument("--n-experts", type=int, default=384)
    parser.add_argument("--include-kind", default="up,gate,down", help="Comma-separated subset of up,gate,down.")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--require-aligned", action="store_true", help="Fail if any row is not 4KiB aligned.")
    args = parser.parse_args()

    include_kinds = {k.strip() for k in args.include_kind.split(",") if k.strip()}
    invalid = include_kinds - {"up", "gate", "down"}
    if invalid:
        raise RuntimeError(f"invalid --include-kind values: {sorted(invalid)}")
    if not include_kinds:
        raise RuntimeError("--include-kind selected no tensor kinds")
    if args.n_experts <= 0:
        raise RuntimeError("--n-experts must be positive")

    repo_root = Path(__file__).resolve().parents[1]
    install_gguf_import(repo_root)
    model_paths = parse_model_paths(args.model_glob)
    rows, summary = build_rows(model_paths, include_kinds, args.n_experts, args.require_aligned)
    summary["repo_head"] = git_head(repo_root)
    summary["output_tsv"] = str(args.out)

    write_tsv(args.out, rows)
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"entries={summary['entries']}")
    print(f"expert_tensors={summary['expert_tensors']}")
    print(f"payload_gib={summary['total_payload_gib']:.3f}")
    print(f"unaligned_rows={summary['unaligned_rows']}")
    print(f"out={args.out}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1)
