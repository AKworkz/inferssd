#!/usr/bin/env python3
"""Launch DeepSeek or Muse on Apple Silicon using a regenerated GGUF expert index."""
import argparse
import glob
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import tempfile

REPO = Path(__file__).resolve().parents[1]
PROFILES = {
    "dsv": (384, "deepseek-ai-DeepSeek-V4.1.jinja"),
    "muse": (256, "muse-spark.jinja"),
}
MAX_CACHE_MIB = (sys.maxsize * 2 + 1) >> 20
MODEL_OPTIONS = {"-m", "--model", "-mu", "--model-url", "-dr", "--docker-repo",
                 "-hf", "-hfr", "--hf-repo", "-hff", "--hf-file"}
REMOTE_MODEL_ENV = ("LLAMA_ARG_MODEL_URL", "LLAMA_ARG_DOCKER_REPO", "LLAMA_ARG_HF_REPO", "LLAMA_ARG_HF_FILE")


def model_shards(first):
    first = first.expanduser().resolve()
    match = re.fullmatch(r"(.+)-(\d{5})-of-(\d{5})\.gguf", first.name)
    if match:
        prefix, number, count = match.groups()
        if number != "00001" or int(count) < 1:
            raise ValueError("SSD_LLAMA_MODEL must name the first GGUF shard")
        paths = [first.with_name(f"{prefix}-{index:05d}-of-{count}.gguf")
                 for index in range(1, int(count) + 1)]
    else:
        paths = [first]
    for path in paths:
        if not path.is_file():
            raise ValueError(f"Missing GGUF shard: {path}")
    return paths


def integer_env(name, default, minimum, maximum):
    text = os.environ.get(name, str(default))
    if not re.fullmatch(r"[0-9]+", text) or not minimum <= int(text) <= maximum:
        raise ValueError(f"{name} must be {minimum}..{maximum}")
    return str(int(text))


def generate_alias(paths, profile, cache_dir):
    cache_dir.mkdir(parents=True, exist_ok=True)
    name = hashlib.sha256((profile + "\0" + str(paths[0])).encode()).hexdigest()[:20]
    alias = cache_dir / f"{name}.tsv"
    summary = cache_dir / f"{name}.json"
    with tempfile.TemporaryDirectory(prefix=".index-", dir=cache_dir) as temporary:
        pending_alias, pending_summary = Path(temporary) / "index.tsv", Path(temporary) / "summary.json"
        command = [sys.executable, str(REPO / "scripts/build-gguf-expert-alias-tsv.py"),
                   "--n-experts", str(PROFILES[profile][0]), "--out", str(pending_alias),
                   "--summary-json", str(pending_summary)]
        for path in paths:
            command.extend(("--model-glob", glob.escape(str(path))))
        subprocess.run(command, check=True)
        metadata = json.loads(pending_summary.read_text())
        if metadata.get("model_shards") != len(paths) or metadata.get("entries", 0) <= 0:
            raise ValueError("GGUF metadata did not produce a complete nonempty expert index")
        # Each launch rebuilds from current metadata. Publish only a complete
        # index, so a failed or concurrent generation cannot expose partial rows.
        metadata["output_tsv"] = str(alias)
        pending_summary.write_text(json.dumps(metadata, indent=2) + "\n")
        os.replace(pending_alias, alias)
        os.replace(pending_summary, summary)
    return alias


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, epilog="Extra llama arguments follow MODEL MODE.")
    parser.add_argument("model", choices=PROFILES)
    parser.add_argument("mode", choices=("chat", "server"))
    parser.add_argument("extra", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    try:
        if platform.system() != "Darwin" or platform.machine() != "arm64":
            raise ValueError("This launcher requires an Apple Silicon Mac")
        model = os.environ.get("SSD_LLAMA_MODEL")
        if not model:
            raise ValueError("Set SSD_LLAMA_MODEL to the first GGUF shard")
        if os.environ.get("SSD_LLAMA_MAC_EXPERT_PACK"):
            raise ValueError("Unset SSD_LLAMA_MAC_EXPERT_PACK for GGUF mode, or use the legacy pack launcher")
        extra = args.extra[1:] if args.extra[:1] == ["--"] else args.extra
        if any(word.split("=", 1)[0] in MODEL_OPTIONS for word in extra) or any(os.environ.get(name) for name in REMOTE_MODEL_ENV):
            raise ValueError("Set the model with SSD_LLAMA_MODEL so the expert index matches it")
        paths = model_shards(Path(model))
        build = Path(os.environ.get("SSD_LLAMA_BUILD_DIR", str(REPO / "build-mac"))).expanduser().resolve()
        binary = build / "bin" / ("llama-cli" if args.mode == "chat" else "llama-server")
        if not binary.is_file() or not os.access(binary, os.X_OK):
            raise ValueError(f"Missing executable: {binary}; run scripts/build-macos-metal.sh first")
        template = REPO / "models/templates" / PROFILES[args.model][1]
        if not template.is_file():
            raise ValueError(f"Missing model chat template: {template}")
        # Bound cache capacity conservatively. The read policy overlaps only
        # selected experts and keeps prefill misses out of the decode cache.
        settings = {
            "SSD_LLAMA_MAC_EXPERT_CACHE_MIB": integer_env("SSD_LLAMA_MAC_EXPERT_CACHE_MIB", 4096, 0, MAX_CACHE_MIB),
            "SSD_LLAMA_MAC_EXPERT_READ_AHEAD": integer_env("SSD_LLAMA_MAC_EXPERT_READ_AHEAD", 2, 0, 4),
            "SSD_LLAMA_MAC_EXPERT_READ_ALIGNMENT": integer_env("SSD_LLAMA_MAC_EXPERT_READ_ALIGNMENT", 4096, 1, 16384),
            "SSD_LLAMA_MAC_EXPERT_PREFILL": integer_env("SSD_LLAMA_MAC_EXPERT_PREFILL", 1, 0, 1),
        }
        if settings["SSD_LLAMA_MAC_EXPERT_READ_ALIGNMENT"] not in ("1", "4096", "16384"):
            raise ValueError("SSD_LLAMA_MAC_EXPERT_READ_ALIGNMENT must be 1, 4096 or 16384")
        policy = os.environ.get("SSD_LLAMA_MAC_EXPERT_CACHE_POLICY", "lfu")
        if policy not in ("lfu", "lru"):
            raise ValueError("SSD_LLAMA_MAC_EXPERT_CACHE_POLICY must be lfu or lru")
        ctx = integer_env("SSD_LLAMA_CTX", 8192, 1, 1048576)
        batch = integer_env("SSD_LLAMA_BATCH", 512, 1, 65536)
        ubatch = integer_env("SSD_LLAMA_UBATCH", 512, 1, 65536)
        threads = integer_env("SSD_LLAMA_THREADS", 8, 1, 1024)
        prefill_threads = integer_env("SSD_LLAMA_PREFILL_THREADS", 12, 1, 1024)
        port = integer_env("SSD_LLAMA_PORT", 8080, 1, 65535)
        cache_dir = Path(os.environ.get("XDG_CACHE_HOME", str(Path.home() / "Library/Caches"))) / "inferssd/mac-alias"
        cache_dir.mkdir(parents=True, exist_ok=True)
        memory = int(subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True))
        free = shutil.disk_usage(cache_dir).free
        print(f"Physical memory: {memory / 1024**3:.1f} GiB; expert cache limit: "
              f"{settings['SSD_LLAMA_MAC_EXPERT_CACHE_MIB']} MiB. Model buffers and the OS need additional RAM.", flush=True)
        print(f"Reading {len(paths)} GGUF file(s) directly; no expert-pack copy. "
              f"Index storage has {free / 1024**3:.1f} GiB free.", flush=True)
        alias = generate_alias(paths, args.model, cache_dir)
        env = dict(os.environ, **settings)
        env["GGML_MOE_EXPERT_GGUF_ALIAS_TSV"] = str(alias)
        env["SSD_LLAMA_MAC_EXPERT_CACHE_POLICY"] = policy
        for name, value in {
            "LLAMA_MMAP_PREFETCH": "0", "LLAMA_DSV41_TOPK_REUSE": "0",
            "LLAMA_MAC_COMPACT_METAL": "1", "LLAMA_DROP_MMAP_TENSOR_AFTER_UPLOAD": "1",
            "LLAMA_MAC_ADAPTIVE_PREFILL": "0",
        }.items():
            env.setdefault(name, value)
        command = [str(binary), "-m", str(paths[0]), "--mmap", "--no-repack", "--fit", "off",
                   "--no-warmup", "--no-host", "-ngl", "99", "--cpu-moe", "--no-op-offload",
                   "-c", ctx, "-b", batch, "-ub", ubatch, "-t", threads, "-tb", prefill_threads,
                   "--chat-template-file", str(template), "--reasoning", "off", "--cache-ram", "0"]
        if args.model == "dsv":
            command.extend(("-ot", r"blk\.[0-9]+\.engram_embd\.weight=CPU"))
        if args.mode == "server":
            command.extend(("--host", "127.0.0.1", "--port", port, "-np", "1"))
        else:
            command.extend(("-n", "128"))
        command.extend(extra)
        print(f"Starting {args.model} {args.mode} with a fresh GGUF alias index: {alias}", flush=True)
        os.execvpe(str(binary), command, env)
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        parser.exit(2, f"{error}\n")


if __name__ == "__main__":
    main()
