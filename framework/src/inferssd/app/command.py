# SPDX-License-Identifier: Apache-2.0
"""Friendly commands that prepare inputs for the existing owned launcher."""

from __future__ import annotations

import argparse
import json
import os
import sys
import uuid
from pathlib import Path

from inferssd.app.hub import download, selection
from inferssd.launch import typed_command
from inferssd.launch.resources import data_directory, installed_resources, read_resources
from inferssd.platform.captured_command import run_captured


def add_commands(commands: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    for mode in ("cli", "serve"):
        parser = commands.add_parser(
            mode,
            help="download a tested GGUF and "
            + ("chat in the terminal" if mode == "cli" else "start the OpenAI-compatible API"),
        )
        source = parser.add_mutually_exclusive_group(required=True)
        source.add_argument(
            "-hf", "--hf-repo", help="README Hugging Face repository[:quantization]"
        )
        source.add_argument("-m", "--model", type=Path, help="local first GGUF shard (no download)")
        parser.add_argument("--revision", default="main", help="Hugging Face revision")
        parser.add_argument("--model-dir", type=Path, default=data_directory() / "models")
        parser.add_argument(
            "--resources",
            type=Path,
            help="engine descriptor of a source build (the installer sets this)",
        )
        parser.add_argument("--port", type=int, default=8080)
        parser.add_argument("--context", type=int, default=8192)
        parser.add_argument(
            "--settings", type=Path, help="typed settings file (see docs/configuration.md)"
        )
        parser.add_argument(
            "--set",
            dest="overrides",
            action="append",
            default=[],
            metavar="KEY=VALUE",
            help="override one setting; repeat for more",
        )
        parser.add_argument("--machine", type=Path, help=argparse.SUPPRESS)
        parser.add_argument("--reprofile", action="store_true")
        if mode == "cli":
            parser.add_argument("-p", "--prompt", help="answer one prompt and exit")
            parser.add_argument("-n", "--max-tokens", type=int, default=512)


def physical_memory() -> int:
    if sys.platform == "darwin":
        result = run_captured(["sysctl", "-n", "hw.memsize"], timeout=10)
        result.check_returncode()
        return int(result.stdout)
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemTotal:"):
            return int(line.split()[1]) * 1024
    raise ValueError("cannot read system RAM; provide --runtime-policy")


def default_policy(state: Path) -> dict[str, object]:
    """The whole machine: the engine uses what is available and keeps its own margin."""
    return {
        "platform": sys.platform,
        "lock_path": str(state / "runtime.lock"),
        "stop_path": str(state / "runtime.stop"),
        "memory_budget_bytes": physical_memory(),
        "minimum_memlock_bytes": 0,
        "cleanup_seconds": 30,
    }


def run(args: argparse.Namespace) -> int:
    from inferssd.app.chat import chat
    from inferssd.engine.command import run_native

    if sys.platform not in ("linux", "darwin"):
        raise ValueError("cli and serve currently support Linux CUDA and Apple Silicon macOS")
    if not 1 <= args.port <= 65535 or args.context < 1:
        raise ValueError("port must be 1..65535; context must be positive")
    if args.command == "cli" and args.max_tokens < 1:
        raise ValueError("max-tokens must be positive")
    if args.machine and args.reprofile:
        raise ValueError("choose --machine or --reprofile")
    resources = args.resources or installed_resources()
    if not resources.is_file():
        raise ValueError("engine not installed; run scripts/install.sh or pass --resources FILE")
    deployment = read_resources(resources)
    if deployment.platform is not None and deployment.platform != sys.platform:
        raise ValueError("engine resources target a different platform")
    if not Path(deployment.binary).is_file() or not os.access(deployment.binary, os.X_OK):
        raise ValueError("engine server is missing or not executable; rerun the installer")
    for _, template in deployment.templates:
        if not Path(template).is_file():
            raise ValueError("engine chat template is missing; rerun the installer")
    state = (data_directory() / "state").resolve()
    state.mkdir(parents=True, exist_ok=True)
    policy_path = state / ("policy-" + uuid.uuid4().hex + ".json")
    # Keep the policy alongside launch receipts for reproducibility.
    policy_path.write_text(json.dumps(default_policy(state), indent=2) + "\n")
    if args.hf_repo:
        selection(args.hf_repo)
        model = download(args.hf_repo, args.revision, args.model_dir.expanduser().resolve())
    else:
        model = args.model.expanduser().resolve(strict=True)
    parser = argparse.ArgumentParser()
    typed_command.add_command(parser.add_subparsers(dest="command", required=True))
    command = [
        "launch",
        "--execute",
        "--resources",
        str(resources.resolve()),
        "--runtime-policy",
        str(policy_path),
        "--model",
        str(model),
        "--port",
        str(args.port),
        "--context",
        str(args.context),
        "--output-dir",
        str(state / "runs" / uuid.uuid4().hex),
    ]
    for name in ("settings", "machine"):
        if getattr(args, name):
            command.extend(["--" + name, str(getattr(args, name))])
    for value in args.overrides:
        command.extend(["--set", value])
    if args.reprofile:
        command.append("--reprofile")
    launch = parser.parse_args(command)
    if args.command == "serve":
        print(f"Starting API at http://127.0.0.1:{args.port}/v1", file=sys.stderr)
        return typed_command.run(launch)
    return run_native(
        launch,
        typed_command.inputs(launch),
        session=lambda: chat(args.port, args.prompt, args.max_tokens),
    )
