"""Local artifact preparation plus required recorded host facts; never starts an engine."""

from __future__ import annotations

import glob
import hashlib
import io
import json
import os
import shlex
import subprocess
import sys
import tempfile
import traceback
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, Optional

from inferssd.launch.darwin import PROFILES
from inferssd.launch.resources import GPU_BACKENDS
from inferssd.launch.types import Fact, LaunchInput, Reply
from inferssd.machine import characteristics
from inferssd.source import experts
from inferssd.strategy import inputs as strategy_inputs


def metadata(name: str, arguments: tuple[str, ...]) -> Reply:
    """Keep the subprocess-style reply boundary using the typed artifact functions.

    Tracebacks identify the actual Python source. They are not rewritten to look
    like the former shell's helper process. SystemExit and partial writes remain.
    """
    stdout, stderr = io.StringIO(), io.StringIO()

    def emit(*values: object, **keywords: Any) -> None:
        stream = stderr if keywords.pop("file", None) is sys.stderr else stdout
        print(*values, file=stream, **keywords)

    try:
        if name == "index":
            experts.index(arguments[0], None, arguments[1], emit=emit)
        elif name == "source_facts":
            experts.facts(arguments[0], arguments[1] or None, emit=emit)
        elif name == "dense":
            experts.dense(*arguments, emit=emit)
        else:
            raise ValueError("Unknown artifact operation: " + name)
    except SystemExit as error:
        if isinstance(error.code, int):
            code = error.code
        elif error.code is None:
            code = 0
        else:
            stderr.write(str(error.code) + "\n")
            code = 1
        return Reply(None, stdout.getvalue(), stderr.getvalue(), code)
    except Exception:
        stderr.write(traceback.format_exc())
        return Reply(None, stdout.getvalue(), stderr.getvalue(), 1)
    values = (
        dict(word.split("=", 1) for word in shlex.split(stdout.getvalue()))
        if name != "index"
        else {}
    )
    return Reply(values, stdout.getvalue(), stderr.getvalue())


def generate_alias(
    paths: tuple[str, ...],
    profile: Optional[str],
    cache_dir: Path,
    *,
    repo: str,
    python: str,
    run: Callable[[list[str]], None],
    replace: Callable[[str, str], None],
) -> str:
    """Run the strict GGUF alias indexer, then publish the index and its summary in order.

    The historical launcher's profile also fixes the expert count it checks; without one
    the indexer reads it from the model.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    name = hashlib.sha256(((profile or "") + "\0" + paths[0]).encode()).hexdigest()[:20]
    alias, summary = cache_dir / f"{name}.tsv", cache_dir / f"{name}.json"
    with tempfile.TemporaryDirectory(prefix=".index-", dir=cache_dir) as temporary:
        pending_alias, pending_summary = (
            Path(temporary) / "index.tsv",
            Path(temporary) / "summary.json",
        )
        command = [
            python,
            "-I",
            "-m",
            "inferssd.source.gguf_alias",
            "--out",
            str(pending_alias),
            "--summary-json",
            str(pending_summary),
        ]
        if profile:
            command[4:4] = ["--n-experts", str(PROFILES[profile][0])]
        if repo:
            command[4:4] = ["--repo", repo]
        for path in paths:
            command.extend(("--model-glob", glob.escape(path)))
        run(command)
        data = json.loads(pending_summary.read_text())
        if data.get("model_shards") != len(paths) or data.get("entries", 0) <= 0:
            raise ValueError("GGUF metadata did not produce a complete nonempty expert index")
        data["output_tsv"] = str(alias)
        pending_summary.write_text(json.dumps(data, indent=2) + "\n")
        replace(str(pending_alias), str(alias))
        replace(str(pending_summary), str(summary))
    return str(alias)


def run_indexer(command: list[str]) -> None:
    # The only subprocess allowed by this adapter is the installed strict header
    # indexer, assembled by generate_alias. Engine argv never reaches this call.
    subprocess.run(command, check=True)


class RecordedArtifacts:
    """Filesystem effects are local; host observations must be supplied explicitly.

    A missing host field fails only when requested. This object does not call
    nvidia-smi, sysctl, systemd, sockets, /proc or a server executable.
    """

    def __init__(
        self,
        inputs: LaunchInput,
        host: Mapping[str, object],
        *,
        run: Callable[[list[str]], None],
        replace: Callable[[str, str], None],
        python: str,
    ):
        self.inputs, self.host = inputs, host
        self.run, self.replace, self.python = run, replace, python

    def __call__(self, fact: Fact) -> Reply:
        name, args = fact.name, fact.arguments
        if name == "is_file":
            return Reply(Path(args[0]).is_file())
        if name == "is_executable":
            return Reply(os.access(args[0], os.X_OK))
        if name == "read_text":
            with open(args[0], encoding="utf-8", newline="") as stream:
                return Reply(stream.read())
        if name == "model_matches":
            directory, pattern = args
            paths = [
                path
                for level in ("", "*/", "*/*/")
                for path in sorted(glob.glob(directory + "/" + level + pattern))
            ]
            return Reply(tuple(paths))
        if name == "absolute_directory":
            return Reply(str(Path(args[0]).resolve(strict=True)))
        if name == "resolved_path":
            path = args[0]
            if path == "~" or path.startswith("~/"):
                path = dict(self.inputs.environment).get("HOME", str(Path.home())) + path[1:]
            return Reply(str(Path(path).expanduser().resolve()))
        if name == "older":
            return Reply(os.stat(args[0]).st_mtime < os.stat(args[1]).st_mtime)
        if name == "mkdir":
            Path(args[0]).mkdir(parents=True, exist_ok=True)
            return Reply()
        if name == "write_exclusive":
            with Path(args[0]).open("x", encoding="utf-8", newline="\n") as stream:
                stream.write(args[1])
            return Reply()
        if name in ("index", "source_facts", "dense"):
            return metadata(name, args)
        if name in ("mac_alias", "gguf_alias"):
            # The historical Mac launcher names its profile first; the typed launch does not.
            historical = name == "mac_alias"
            profile = args[0] if historical else None
            directory, *paths = args[1:] if historical else args
            return Reply(
                generate_alias(
                    tuple(paths),
                    profile,
                    Path(directory),
                    repo=self.inputs.repo,
                    python=self.python,
                    run=self.run,
                    replace=self.replace,
                )
            )
        if name == "model_facts" and name not in self.host:
            # Read from the model's own headers, as the alias index is.
            header = strategy_inputs.shard_facts(args)
            return Reply(strategy_inputs.model_facts_json(header))
        if (
            name in ("machine_characteristics", "model_drive_key", "engine_gpu_backends")
            and name not in self.host
        ):
            # Absent: no strategy, no drive term in it, or an engine whose GPU backends are unknown.
            return Reply(None)
        if name == "systemd" and name not in self.host:
            # Recordings made before this fact existed came from systemd hosts.
            return Reply(True)
        if name not in self.host:
            raise ValueError(f"Missing recorded host fact: {name}")
        value = self.host[name]
        boolean = {
            "gpu_available",
            "ss_available",
            "has_uring",
            "systemd",
            "cgroup_v2",
            "user_manager",
            "memory_controller",
        }
        textual = {
            "filesystem",
            "gpu_memory",
            "gpu_processes",
            "port_busy",
            "scope_suffix",
            "platform_system",
            "platform_machine",
        }
        converted = {"default_threads", "physical_memory", "disk_free"}
        if name == "machine_characteristics":
            # An inferssd-machine/1 object, or its text; the resolver gets canonical text.
            try:
                if isinstance(value, str):
                    return Reply(characteristics.dumps(characteristics.loads(value)))
                return Reply(characteristics.dumps(characteristics.validate(value)))
            except ValueError as error:
                raise ValueError(f"Recorded host fact {name}: {error}") from error
        if name in ("model_facts", "runtime_facts"):
            # Serialised strategy inputs, read back strictly and written out again.
            try:
                if name == "model_facts":
                    facts = strategy_inputs.model_facts_from(value)
                    return Reply(strategy_inputs.model_facts_json(facts))
                runtime = strategy_inputs.runtime_facts_from(value)
                return Reply(strategy_inputs.runtime_facts_json(runtime))
            except ValueError as error:
                raise ValueError(f"Recorded host fact {name}: {error}") from error
        if name == "model_drive_key":
            if value is not None and not isinstance(value, str):
                raise ValueError(f"Recorded host fact {name}: expected text or null")
            return Reply(value)
        if name == "engine_gpu_backends":
            if value is not None and (
                not isinstance(value, list)
                or any(backend not in GPU_BACKENDS for backend in value)
                or len(set(value)) != len(value)
            ):
                raise ValueError(
                    f"Recorded host fact {name}: expected null or distinct names of "
                    + ", ".join(GPU_BACKENDS)
                )
            return Reply(value)
        if name in boolean:
            if not isinstance(value, bool):
                raise ValueError(f"Recorded host fact {name}: expected a JSON boolean")
        elif name == "euid":
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"Recorded host fact {name}: expected an integer")
        elif name == "memory_kib":
            if (
                not isinstance(value, (list, tuple))
                or len(value) != 2
                or any(isinstance(item, bool) or not isinstance(item, int) for item in value)
            ):
                raise ValueError(
                    f"Recorded host fact {name}: expected two integers [total, available]"
                )
        elif name in textual:
            if not isinstance(value, str):
                raise ValueError(f"Recorded host fact {name}: expected text")
        elif name in converted:
            if isinstance(value, bool) or not isinstance(value, (str, int)):
                raise ValueError(f"Recorded host fact {name}: expected integer text or an integer")
            if name in ("physical_memory", "disk_free"):
                try:
                    int(value)
                except ValueError as error:
                    raise ValueError(f"Recorded host fact {name}: expected integer text") from error
            value = str(value)
        else:
            raise ValueError(f"Unknown recorded host fact: {name}")
        return Reply(value)
