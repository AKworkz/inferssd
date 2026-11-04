"""Resource checks consume synthetic native files/command replies, never this host."""

from __future__ import annotations

import resource
from dataclasses import replace
from pathlib import Path

import pytest
from test_engine_runtime import policy

from inferssd.launch.live import Command, LiveArtifacts, NativeHost
from inferssd.launch.resources import cmake_gpu_backends
from inferssd.launch.types import Fact, LaunchInput
from inferssd.platform import resources


def linux_files(monkeypatch, budget):
    files = {
        "/proc/self/cgroup": "0::/fixture.scope\n",
        "/proc/123/cgroup": "0::/fixture.scope\n",
        "/proc/123/limits": "Max locked memory 4096 8192 bytes\n",
        "/proc/meminfo": "MemTotal: 2000000 kB\nMemAvailable: 1000000 kB\n",
    }
    for path, maximum in (("/sys/fs/cgroup", "max"), ("/sys/fs/cgroup/fixture.scope", str(budget))):
        for name, value in {
            "memory.max": maximum,
            "memory.swap.max": "0",
            "memory.current": "100",
            "memory.peak": "200",
            "memory.events": "oom 0\noom_kill 0\n",
        }.items():
            files[path + "/" + name] = value
    monkeypatch.setattr(Path, "read_text", lambda self, *a, **kw: files[str(self)])
    monkeypatch.setattr(Path, "exists", lambda self: str(self) in files)
    monkeypatch.setattr(resources.sys, "platform", "linux")
    monkeypatch.setattr(resources.resource, "getrlimit", lambda _: (4096, 8192))
    return files


def test_linux_actual_budget_memlock_and_oom(tmp_path, monkeypatch):
    p = replace(policy(tmp_path, "linux"), minimum_memlock_bytes=4096)
    files = linux_files(monkeypatch, p.memory_budget_bytes)
    runtime = resources.Resources(p)
    first = runtime.prepare()
    assert first["memlock"] == {"soft": 4096, "hard": 8192, "minimum_bytes": 4096}
    assert runtime.sample(123)["available_bytes"] == 1000000 * 1024
    files["/sys/fs/cgroup/fixture.scope/memory.events"] = "oom 1\noom_kill 0\n"
    with pytest.raises(RuntimeError, match="OOM counters changed"):
        runtime.sample(123)


@pytest.mark.parametrize(
    "file,value,error",
    [
        ("/sys/fs/cgroup/memory.max", "1024", "ancestor memory"),
        ("/sys/fs/cgroup/fixture.scope/memory.max", "max", "ancestor memory"),
        ("/sys/fs/cgroup/fixture.scope/memory.events", "low 0", "OOM counters unavailable"),
        ("/proc/123/cgroup", "0::/foreign.scope\n", "escaped"),
        ("/proc/123/limits", "Max locked memory 0 8192 bytes\n", "memlock inheritance"),
        ("/proc/meminfo", "MemAvailable: 1 kB\n", "below reserve"),
    ],
)
def test_linux_negative_envelope(tmp_path, monkeypatch, file, value, error):
    p = policy(tmp_path, "linux")
    files = linux_files(monkeypatch, p.memory_budget_bytes)
    runtime = resources.Resources(p)
    runtime.prepare()
    files[file] = value
    with pytest.raises(RuntimeError, match=error):
        runtime.sample(123)


@pytest.mark.parametrize(
    "soft,minimum,accepted",
    [(0, 4096, False), (4096, 4096, True), (resource.RLIM_INFINITY, 1 << 40, True)],
)
def test_memlock_finite_unlimited_insufficient(tmp_path, monkeypatch, soft, minimum, accepted):
    p = replace(policy(tmp_path, "linux"), minimum_memlock_bytes=minimum)
    linux_files(monkeypatch, p.memory_budget_bytes)
    monkeypatch.setattr(resources.resource, "getrlimit", lambda _: (soft, resource.RLIM_INFINITY))
    runtime = resources.Resources(p)
    if accepted:
        assert runtime.prepare()["memlock"]["soft"] == soft
    else:
        with pytest.raises(ValueError, match="RLIMIT_MEMLOCK"):
            runtime.prepare()


def test_darwin_explicit_soft_budget_and_growth(tmp_path, monkeypatch):
    p = replace(policy(tmp_path), memory_budget_bytes=1 << 20, host_reserve_bytes=0)
    replies = {
        "vm_stat": (
            "Mach Virtual Memory Statistics: (page size of 4096 bytes)\n"
            "Pages free: 10.\nPages inactive: 20.\nPages occupied by compressor: 2.\n"
        ),
        "sysctl": "total = 0.00M used = 0.00M free = 0.00M",
        "ps": "32\n",
    }
    monkeypatch.setattr(resources, "command", lambda argv: replies[argv[0]])
    monkeypatch.setattr(resources.sys, "platform", "darwin")
    monkeypatch.setattr(resources.resource, "getrlimit", lambda _: (0, 0))
    runtime = resources.Resources(p)
    first = runtime.prepare()
    assert "soft" in first["enforcement"] and "no hard" in first["enforcement"]
    assert runtime.sample(123)["rss_bytes"] == 32 * 1024
    replies["ps"] = "2048"
    with pytest.raises(RuntimeError, match="RSS"):
        runtime.sample(123)
    replies["ps"] = "32"
    replies["sysctl"] = "used = 1.00M"
    with pytest.raises(RuntimeError, match="swap growth"):
        runtime.sample(123)
    replies["sysctl"] = "used = 0.00M"
    replies["vm_stat"] = replies["vm_stat"].replace("compressor: 2", "compressor: 3")
    with pytest.raises(RuntimeError, match="compressor growth"):
        runtime.sample(123)


def test_exact_listener_pid_is_required(tmp_path, monkeypatch):
    runtime = resources.Resources(policy(tmp_path))
    monkeypatch.setattr(resources, "command", lambda _: "p999\0n127.0.0.1:8080\0")
    with pytest.raises(RuntimeError, match="exact server PID"):
        runtime.confirm_listener(123, 8080)
    monkeypatch.setattr(resources, "command", lambda _: "p123\0n127.0.0.1:8080\0")
    runtime.confirm_listener(123, 8080)


class Host:
    def __init__(self):
        self.events, self.replies = [], {}

    def command(self, args):
        self.events.append(tuple(args))
        return self.replies[args[0]]

    def which(self, name):
        self.events.append(("which", name))
        return True

    def contains(self, path, needle):
        self.events.append(("contains", path))
        return path == "first"

    def exists(self, path):
        self.events.append(("exists", path))
        return path == "/fixture"

    def readable(self, path):
        self.events.append(("readable", path))
        return False

    def text(self, path):
        self.events.append(("text", path))
        return "MemTotal: 4000 kB\nMemAvailable: 2000 kB\n"

    def uid(self):
        self.events.append(("uid",))
        return 501

    def suffix(self):
        self.events.append(("suffix",))
        return "fixture"

    def system(self):
        self.events.append(("system",))
        return "Darwin"

    def machine(self):
        self.events.append(("machine",))
        return "arm64"

    def disk_free(self, path):
        self.events.append(("disk", path))
        return 1024


def test_host_is_lazy_and_short_circuits_original_facts():
    host = Host()
    adapter = LiveArtifacts(
        LaunchInput("/fixture/repo", ()),
        host=host,
        run=lambda _: pytest.fail("unexpected indexer"),
        replace=lambda *_: None,
        python="fixture-python",
    )
    assert host.events == []
    assert adapter(Fact("has_uring", ("first", "never"))).value is True
    assert host.events == [("contains", "first")]
    assert adapter(Fact("memory_kib")).value == (4000, 2000)
    host.replies["nproc"] = Command(0, "8\n")
    assert adapter(Fact("default_threads")).value == "8"
    host.replies["stat"] = Command(0, "ext2/ext3\n")
    assert adapter(Fact("filesystem", ("/fixture/missing/cache",))).value == "ext2/ext3"
    assert host.events[-1] == ("stat", "-f", "-c", "%T", "/fixture")


def test_gpu_failed_or_na_and_checked_command_error():
    host = Host()
    host.replies["nvidia-smi"] = Command(1, "invalid")
    adapter = LiveArtifacts(
        LaunchInput("/fixture", ()),
        host=host,
        run=lambda _: None,
        replace=lambda *_: None,
        python="fixture",
    )
    assert adapter(Fact("gpu_memory")).value == ""
    host.replies["nvidia-smi"] = Command(0, "N/A\n999\n")
    assert adapter(Fact("gpu_memory")).value == "N/A"
    host.replies["nvidia-smi"] = Command(1, "", "failure")
    with pytest.raises(RuntimeError, match="host command failed"):
        adapter(Fact("gpu_processes"))


def test_a_builds_gpu_backends_are_read_from_its_cmake_cache():
    cache = "GGML_CUDA:BOOL=ON\nGGML_METAL:BOOL=OFF\n// comment\nGGML_VULKAN:BOOL=TRUE\n"
    assert cmake_gpu_backends(cache) == ("CUDA", "Vulkan")
    metal = "GGML_METAL:BOOL=ON\nGGML_CUDA_KIMI_VRAM_TIER:BOOL=ON\n"
    assert cmake_gpu_backends(metal) == ("Metal",)
    assert cmake_gpu_backends("") == ()


def test_the_engine_gpu_backends_fact_reads_the_servers_build(tmp_path):
    build = tmp_path / "build-linux-cuda"
    (build / "bin").mkdir(parents=True)
    binary = str(build / "bin" / "inferssd-server")
    adapter = LiveArtifacts(
        LaunchInput(str(tmp_path), ()),
        host=NativeHost(),
        run=lambda _: pytest.fail("unexpected indexer"),
        replace=lambda *_: None,
        python="fixture-python",
    )
    # A server outside a CMake build: its backends are unknown.
    assert adapter(Fact("engine_gpu_backends", (binary,))).value is None
    # A UTF-8 cache, whatever this host's locale: it names the build's paths.
    cache = (
        "CMAKE_HOME_DIRECTORY:INTERNAL=/home/测/Kovač\nGGML_CUDA:BOOL=ON\nGGML_VULKAN:BOOL=OFF\n"
    )
    (build / "CMakeCache.txt").write_bytes(cache.encode())
    assert adapter(Fact("engine_gpu_backends", (binary,))).value == ["CUDA"]
