"""Explicit deployment resources; paths are relative to the descriptor, never cwd."""

from __future__ import annotations

import json
import ntpath
import os
import posixpath
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from inferssd.configuration.settings import PLATFORMS

# A checkout's build directories, searched in this order: the presets first,
# then the conventional ad hoc ones.
BUILDS = {
    "linux": ("build-linux-cuda", "build-linux-cpu", "build"),
    "darwin": ("build-macos-metal", "build-mac", "build"),
    "windows": ("build-x64-windows-llvm-release", "build"),
}
# The machine probes by the name a resource descriptor's probes map gives them.
PROBES = {"ssd": "inferssd-ssdprobe", "cpu": "inferssd-cpuprobe", "memory": "inferssd-memprobe"}
# The GPU backends an engine can be built with, by the name the machine identity
# gives a GPU's backend, and the CMake option that builds each.
GPU_BACKENDS = {"CUDA": "GGML_CUDA", "Metal": "GGML_METAL", "Vulkan": "GGML_VULKAN"}


def cmake_gpu_backends(cache: str) -> tuple[str, ...]:
    """The GPU backends a build has, read from the text of its CMakeCache.txt."""
    values = {}
    for line in cache.splitlines():
        name, _, value = line.partition("=")
        values[name.partition(":")[0].strip()] = value.strip().upper()
    return tuple(
        backend
        for backend, option in GPU_BACKENDS.items()
        if values.get(option) in ("ON", "TRUE", "YES", "Y", "1")
    )


def data_directory(environ: Mapping[str, str] = os.environ, home: Optional[Path] = None) -> Path:
    """Where the installer keeps the releases, the models and the launch state."""
    return Path(environ.get("INFERSSD_HOME", (home or Path.home()) / ".local/share/inferssd"))


def installed_resources(
    environ: Mapping[str, str] = os.environ, home: Optional[Path] = None
) -> Path:
    """The installed engine's descriptor: the wrapper's variable, else the current release's."""
    named = environ.get("INFERSSD_RESOURCES")
    return Path(named) if named else data_directory(environ, home) / "current/engine-resources.json"


@dataclass(frozen=True)
class EngineResources:
    binary: str
    templates: tuple[tuple[str, str], ...]
    probes: tuple[tuple[str, str], ...] = ()
    platform: Optional[str] = None
    revision: str = "unknown"
    # The GPU backends the engine was built with; None in a descriptor written before they
    # were recorded.
    gpu_backends: Optional[tuple[str, ...]] = None

    def template(self, profile: str) -> str:
        try:
            return dict(self.templates)[profile]
        except KeyError as error:
            raise ValueError("engine resources have no template for " + profile) from error


def read_resources(path: Path) -> EngineResources:
    path = path.resolve(strict=True)
    data: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("schema") != "inferssd-engine 1":
        raise ValueError("expected inferssd-engine 1 resource descriptor")
    if set(data) - {"schema", "binary", "templates", "probes", "capabilities", "revision"}:
        raise ValueError("unknown engine resource field")
    capabilities = data.get("capabilities")
    if not isinstance(capabilities, dict) or capabilities.get("settings") != "inferssd-settings 1":
        raise ValueError("engine resources must declare inferssd-settings 1")
    if capabilities.get("telemetry") != 1:
        raise ValueError("engine resources must declare telemetry version 1")
    platform = capabilities.get("platform")
    if platform is not None and platform not in PLATFORMS:
        raise ValueError("unsupported engine resource platform")
    revision = data.get("revision", "unknown")
    if not isinstance(revision, str) or not revision:
        raise ValueError("engine revision must be nonempty text")
    backends = capabilities.get("gpu_backends")
    if backends is not None and (
        not isinstance(backends, list)
        or any(backend not in GPU_BACKENDS for backend in backends)
        or len(set(backends)) != len(backends)
    ):
        raise ValueError("engine gpu_backends must list distinct " + ", ".join(GPU_BACKENDS))

    def filename(value: object) -> str:
        if not isinstance(value, str) or not value or "\0" in value:
            raise ValueError("engine resource paths must be nonempty strings")
        return str((path.parent / value).resolve())

    def paths(name: str, required: bool) -> tuple[tuple[str, str], ...]:
        values = data.get(name, {})
        if not isinstance(values, dict) or required and not values:
            raise ValueError(name + " must be a nonempty path map")
        if any(not isinstance(key, str) or not key for key in values):
            raise ValueError(name + " must have nonempty string keys")
        return tuple((key, filename(value)) for key, value in values.items())

    return EngineResources(
        filename(data.get("binary")),
        paths("templates", True),
        paths("probes", False),
        platform,
        revision,
        None if backends is None else tuple(backends),
    )


def executable(platform: str, name: str) -> str:
    return name + ".exe" if platform == "windows" else name


def server_binary(
    platform: str,
    binary: Optional[str],
    resources: Optional[EngineResources],
    repo: str,
    is_file: Callable[[str], bool],
) -> Optional[str]:
    """The server executable: BINARY, else the descriptor's, else the first REPO build with one.

    Without one in any build, the first build's path is named, so the error
    names where it was expected. None when there is nowhere to look.
    """
    path = binary or (resources.binary if resources else None)
    if path or not repo:
        return path
    join = ntpath.join if platform == "windows" else posixpath.join
    server = executable(platform, "inferssd-server")
    candidates = [join(repo, name, "bin", server) for name in BUILDS[platform]]
    return next((candidate for candidate in candidates if is_file(candidate)), candidates[0])


def probe_paths(
    platform: str,
    binary: Optional[str],
    resources: Optional[EngineResources],
    repo: str,
    is_file: Callable[[str], bool],
) -> dict[str, str]:
    """Each probe found: next to the server BINARY, else in the descriptor, else in a REPO build.

    The repo's builds are searched in the order the server is. A probe found
    nowhere is left out.
    """
    join = ntpath.join if platform == "windows" else posixpath.join
    dirname = ntpath.dirname if platform == "windows" else posixpath.dirname
    named = dict(resources.probes) if resources else {}
    found = {}
    for name, program in PROBES.items():
        filename = executable(platform, program)
        candidates = [join(dirname(binary), filename)] if binary else []
        candidates += [named[name]] if name in named else []
        if repo:
            candidates += [join(repo, build, "bin", filename) for build in BUILDS[platform]]
        path = next((candidate for candidate in candidates if is_file(candidate)), None)
        if path is not None:
            found[name] = path
    return found
