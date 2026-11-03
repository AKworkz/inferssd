# SPDX-License-Identifier: Apache-2.0
"""Select one complete tested quantization and download it at a pinned revision."""

from __future__ import annotations

import re
import shutil
import sys
from collections.abc import Sequence
from pathlib import Path, PurePosixPath

from inferssd.app import catalog

MODELS = {model.repo: model.quantization for model in catalog.MODELS}
SPLIT = re.compile(r"(.+)-(\d{5})-of-(\d{5})\.gguf$")


def selection(reference: str) -> tuple[str, str]:
    repo, _, quant = reference.partition(":")
    if repo not in MODELS:
        raise ValueError("-hf supports the README models: " + ", ".join(MODELS))
    if quant and quant != MODELS[repo]:
        raise ValueError(f"the tested quantization for {repo} is {MODELS[repo]}")
    return repo, MODELS[repo]


def select_files(names: Sequence[str], quant: str) -> list[str]:
    # Match a full quantization token, never Q4_K in Q4_K_M or a vision projector.
    token = re.compile(r"(?:^|[-./])" + re.escape(quant) + r"(?:[-./]|$)", re.I)
    files = sorted(
        name
        for name in names
        if name.endswith(".gguf")
        and token.search(name)
        and "mmproj" not in PurePosixPath(name).name.lower()
    )
    if not files:
        raise ValueError(f"no {quant} GGUF files found in the Hugging Face repository")
    for name in files:
        path = PurePosixPath(name)
        if path.is_absolute() or ".." in path.parts or "\\" in name:
            raise ValueError("unsafe model filename in repository")
    matches = [SPLIT.fullmatch(name) for name in files]
    if len(files) == 1 and matches[0] is None:
        return files
    if any(match is None for match in matches):
        raise ValueError("ambiguous GGUF selection: expected one complete shard set")
    parts = [match.groups() for match in matches if match is not None]
    prefix, _, total = parts[0]
    count = int(total)
    if (
        count < 1
        or len(files) != count
        or any(p != prefix or t != total for p, _, t in parts)
        or [int(number) for _, number, _ in parts] != list(range(1, count + 1))
    ):
        raise ValueError("incomplete or ambiguous GGUF shard set; refusing to download")
    return files


def download(reference: str, revision: str, root: Path) -> Path:
    repo, quant = selection(reference)
    try:
        from huggingface_hub import HfApi, snapshot_download
    except ImportError as error:
        raise ValueError('Hugging Face support requires: pip install "inferssd[hf]"') from error
    try:
        info = HfApi().model_info(repo, revision=revision, files_metadata=True)
        commit: object = info.sha
        if not isinstance(commit, str) or not re.fullmatch(r"[0-9a-f]{40,64}", commit):
            raise ValueError("Hugging Face did not return a pinned commit")
        siblings = info.siblings or []
        files = select_files([item.rfilename for item in siblings], quant)
        sizes = {
            item.rfilename: item.size
            for item in siblings
            if isinstance(item.size, int) and item.size > 0
        }
        if any(name not in sizes for name in files):
            raise ValueError("Hugging Face did not return the size of every GGUF shard")
        # One directory per repository: the Hub client compares each file with its
        # metadata, so a commit that leaves a shard unchanged does not fetch it again.
        directory = root / repo.replace("/", "--")
        directory.mkdir(parents=True, exist_ok=True)
        missing = sum(
            sizes[name]
            for name in files
            if not (directory / name).is_file() or (directory / name).stat().st_size != sizes[name]
        )
        # Conservatively require space for whole unfinished shards, plus metadata.
        if missing and shutil.disk_usage(directory).free < missing + 1024**3:
            raise ValueError(f"not enough SSD space: need {missing / 1024**3:.1f} GiB + 1 GiB")
        print(
            f"{repo}:{quant}: {len(files)} shards, "
            f"{sum(sizes[name] for name in files) / 1024**3:.1f} GiB; {directory}",
            file=sys.stderr,
        )
        # local_dir preserves shard filenames for native sibling discovery. The Hub
        # client resumes interrupted transfers and verifies its download metadata.
        snapshot_download(
            repo_id=repo,
            revision=commit,
            allow_patterns=files,
            local_dir=directory,
            max_workers=2,
        )
        for name in files:
            if not (directory / name).is_file() or (directory / name).stat().st_size != sizes[name]:
                raise ValueError(f"download incomplete: {name}; run the command again to resume")
        return directory / files[0]
    except (OSError, ValueError):
        raise
    except Exception as error:
        # Hub v0.x and v1.x use different HTTP exception hierarchies. Do not echo
        # requests or headers, which may contain HF_TOKEN.
        raise RuntimeError(
            f"Hugging Face download failed ({type(error).__name__}); check connectivity, "
            "the revision and model access (HF_TOKEN), then rerun to resume"
        ) from error
