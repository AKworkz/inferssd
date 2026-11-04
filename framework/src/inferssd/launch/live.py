"""Lazy live host facts for explicitly requested native launch preparation.

Constructing this adapter performs no observations. Filesystem artifact effects
remain the existing implementation; each host fact is queried at its original
resolver call point. These observations are not execution admission or a memory
budget guarantee: the worker verifies its actual resource envelope separately.
"""

from __future__ import annotations

import datetime
import os
import platform
import shutil
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Protocol

from inferssd.launch.linux import cpu_count
from inferssd.launch.recorded import RecordedArtifacts
from inferssd.launch.resources import cmake_gpu_backends
from inferssd.launch.types import Fact, LaunchInput, Reply
from inferssd.platform.captured_command import run_captured
from inferssd.strategy import inputs as strategy_inputs


@dataclass(frozen=True)
class Command:
    returncode: int
    stdout: str
    stderr: str = ""


class Host(Protocol):
    def command(self, argv: Sequence[str]) -> Command: ...
    def exists(self, path: str) -> bool: ...
    def readable(self, path: str) -> bool: ...
    def which(self, name: str) -> bool: ...
    def text(self, path: str) -> str: ...
    def contains(self, path: str, needle: bytes) -> bool: ...
    def uid(self) -> int: ...
    def suffix(self) -> str: ...
    def system(self) -> str: ...
    def machine(self) -> str: ...
    def disk_free(self, path: str) -> int: ...


class NativeHost:
    """Concrete standard-library host operations, with bounded command waits."""

    def command(self, argv: Sequence[str]) -> Command:
        result = run_captured(argv, timeout=10)
        return Command(result.returncode, result.stdout, result.stderr)

    def exists(self, path: str) -> bool:
        return Path(path).exists()

    def readable(self, path: str) -> bool:
        return os.access(path, os.R_OK)

    def which(self, name: str) -> bool:
        return shutil.which(name) is not None

    def text(self, path: str) -> str:
        return Path(path).read_text(encoding="utf-8", errors="replace")

    def contains(self, path: str, needle: bytes) -> bool:
        try:
            with open(path, "rb") as stream:
                tail = b""
                while chunk := stream.read(1 << 20):
                    data = tail + chunk
                    if needle in data:
                        return True
                    tail = data[-len(needle) :]
        except OSError:
            return False
        return False

    def uid(self) -> int:
        return os.geteuid()

    def suffix(self) -> str:
        return datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + str(os.getpid())

    def system(self) -> str:
        return platform.system()

    def machine(self) -> str:
        return platform.machine()

    def disk_free(self, path: str) -> int:
        return shutil.disk_usage(path).free


HOST_FACTS = frozenset(
    "gpu_available ss_available has_uring systemd cgroup_v2 user_manager memory_controller euid "
    "memory_kib filesystem gpu_memory gpu_processes port_busy scope_suffix "
    "platform_system platform_machine default_threads physical_memory disk_free "
    "runtime_facts model_drive_key engine_gpu_backends".split()
)


class LiveArtifacts:
    def __init__(
        self,
        inputs: LaunchInput,
        *,
        host: Host,
        run: Callable[[list[str]], None],
        replace: Callable[[str, str], None],
        python: str,
        machine: Optional[str] = None,
        strategy_host: Optional[strategy_inputs.Host] = None,
    ) -> None:
        """MACHINE: the characteristics text the launch strategy derives from, if any.

        STRATEGY_HOST reads the strategy's runtime facts; by default this host,
        with the launch's environment.
        """
        self.inputs, self.host = inputs, host
        self.run, self.replace, self.python = run, replace, python
        self.machine = machine
        self.strategy_host = strategy_host or strategy_inputs.NativeHost(dict(inputs.environment))
        self.artifacts = RecordedArtifacts(inputs, {}, run=run, replace=replace, python=python)

    def __call__(self, fact: Fact) -> Reply:
        if fact.name == "machine_characteristics":
            return Reply(self.machine)
        if fact.name not in HOST_FACTS:
            return self.artifacts(fact)
        value = self.value(fact)
        # Reuse the exact lazy fact-shape contract used by recorded launches.
        return RecordedArtifacts(
            self.inputs, {fact.name: value}, run=self.run, replace=self.replace, python=self.python
        )(fact)

    def output(self, argv: Sequence[str], *, check: bool = True) -> str:
        result = self.host.command(argv)
        if check and result.returncode:
            raise RuntimeError(
                f"host command failed ({result.returncode}): {argv[0]}: {result.stderr}"
            )
        return result.stdout.rstrip("\n")

    def value(self, fact: Fact) -> object:
        name, args, host = fact.name, fact.arguments, self.host
        if name in ("gpu_available", "ss_available"):
            return host.which("nvidia-smi" if name == "gpu_available" else "ss")
        if name == "gpu_memory":
            result = host.command(
                ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"]
            )
            return result.stdout.splitlines()[0] if not result.returncode and result.stdout else ""
        if name == "gpu_processes":
            return self.output(
                ["nvidia-smi", "--query-compute-apps=pid,process_name", "--format=csv,noheader"]
            )
        if name == "port_busy":
            return self.output(["ss", "-Hltn", "sport = :" + args[0]], check=False)
        if name == "has_uring":
            return any(host.contains(path, b"io_uring_queue_init") for path in args)
        if name == "systemd":
            # The same test as sd_booted(): systemd is running as the init system.
            return host.exists("/run/systemd/system")
        if name == "cgroup_v2":
            return host.exists("/sys/fs/cgroup/cgroup.controllers")
        if name == "memory_kib":
            rows = {
                line.split()[0]: line.split()[1]
                for line in host.text("/proc/meminfo").splitlines()
                if len(line.split()) >= 2
            }
            return int(rows.get("MemTotal:", "0")), int(rows.get("MemAvailable:", "0"))
        if name == "euid":
            return host.uid()
        if name == "user_manager":
            return host.command(["systemctl", "--user", "show-environment"]).returncode == 0
        if name == "memory_controller":
            path = (
                f"/sys/fs/cgroup/user.slice/user-{args[0]}.slice/"
                f"user@{args[0]}.service/cgroup.controllers"
            )
            try:
                return "memory" in host.text(path).split()
            except OSError:
                return False
        if name == "default_threads":
            paths = ("/sys/devices/cpu_core/cpus", "/sys/devices/cpu_atom/cpus")
            if all(host.readable(path) for path in paths):
                value = cpu_count(",".join(host.text(path).strip() for path in paths))
                if value:
                    return value
            return self.output(["nproc"])
        if name == "filesystem":
            ancestor = Path(args[0])
            while not host.exists(str(ancestor)):
                parent = ancestor.parent
                if parent == ancestor:
                    raise FileNotFoundError("no existing ancestor for prompt cache: " + args[0])
                ancestor = parent
            return self.output(["stat", "-f", "-c", "%T", str(ancestor)])
        if name == "scope_suffix":
            return host.suffix()
        if name == "platform_system":
            return host.system()
        if name == "platform_machine":
            return host.machine()
        if name == "physical_memory":
            return self.output(["sysctl", "-n", "hw.memsize"])
        if name == "disk_free":
            return host.disk_free(args[0])
        if name == "runtime_facts":
            facts = strategy_inputs.read_runtime(host=self.strategy_host)
            return strategy_inputs.runtime_facts_json(facts)
        if name == "model_drive_key":
            return strategy_inputs.model_drive_key(args, self.strategy_host)
        if name == "engine_gpu_backends":
            # The build the server executable came from: build/bin/inferssd-server lies beside
            # build/CMakeCache.txt.
            build = os.path.dirname(os.path.dirname(args[0]))
            try:
                return list(cmake_gpu_backends(host.text(os.path.join(build, "CMakeCache.txt"))))
            except OSError:
                return None
        raise ValueError("Unknown live host fact: " + name)
