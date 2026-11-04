# SPDX-License-Identifier: Apache-2.0
"""Live machine identity and drive keys, read through an injectable host.

Hard identity names the hardware: a mismatch makes a saved profile describe a
different machine. Soft identity names software that can change a measurement
(kernel, OS release, GPU driver, probe schemas). Serial numbers and machine IDs
are stored only as SHA-256 digests.
"""

from __future__ import annotations

import contextlib
import copy
import glob
import hashlib
import os
import platform
import plistlib
import posixpath
import re
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Iterator, Mapping, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Protocol
from xml.parsers.expat import ExpatError

from inferssd.configuration.settings import cpus
from inferssd.machine.characteristics import PROBE_SCHEMAS, GPUIdentity, Identity
from inferssd.platform.captured_command import run_captured

MEMORY_TOLERANCE = 0.01  # MemTotal moves with kernel reservations
TEST_FILE_PREFIX = ".inferssd-drive-test-"


class UnsupportedPlatform(RuntimeError):
    """This platform cannot be profiled yet."""


class GPUQueryFailed(RuntimeError):
    """nvidia-smi is installed but failed, so the GPUs are unknown.

    ``identity`` is the live identity without GPUs; ``with_saved_gpus`` can
    complete it from a saved profile.
    """

    def __init__(self, reason: str, identity: Identity) -> None:
        super().__init__(reason)
        self.identity = identity


@dataclass(frozen=True)
class Command:
    returncode: int
    stdout: str
    stderr: str = ""


class Host(Protocol):
    def platform(self) -> str: ...
    def machine(self) -> str: ...
    def text(self, path: str) -> str: ...
    def exists(self, path: str) -> bool: ...
    def listdir(self, path: str) -> list[str]: ...
    def realpath(self, path: str) -> str: ...
    def command(self, argv: Sequence[str]) -> Command: ...
    def which(self, name: str) -> bool: ...
    def device(self, path: str) -> tuple[int, int]: ...
    def regular_files(self, path: str) -> list[tuple[str, int]]: ...
    def is_directory(self, path: str) -> bool: ...
    def free_bytes(self, path: str) -> int: ...
    def test_file(self, directory: str, size: int) -> AbstractContextManager[str]: ...
    def clear_test_files(self, directory: str) -> None: ...
    def glob(self, pattern: str) -> list[str]: ...
    def page_bytes(self) -> int: ...


class NativeHost:
    """Standard-library host operations, with bounded command waits."""

    def platform(self) -> str:
        if sys.platform.startswith("linux"):
            return "linux"
        return {"darwin": "darwin", "win32": "windows"}.get(sys.platform, sys.platform)

    def machine(self) -> str:
        return platform.machine()

    def text(self, path: str) -> str:
        return Path(path).read_text(encoding="utf-8", errors="replace")

    def exists(self, path: str) -> bool:
        return os.path.exists(path)

    def listdir(self, path: str) -> list[str]:
        return sorted(os.listdir(path))

    def realpath(self, path: str) -> str:
        return os.path.realpath(path)

    def command(self, argv: Sequence[str]) -> Command:
        try:
            result = run_captured(argv, timeout=30)
        except FileNotFoundError:
            return Command(127, "", argv[0] + ": not found")
        except subprocess.TimeoutExpired:
            return Command(124, "", argv[0] + ": timed out")
        return Command(result.returncode, result.stdout, result.stderr)

    def which(self, name: str) -> bool:
        return shutil.which(name) is not None

    def device(self, path: str) -> tuple[int, int]:
        value = os.stat(path).st_dev
        return os.major(value), os.minor(value)

    def regular_files(self, path: str) -> list[tuple[str, int]]:
        if os.path.isdir(path):
            found = []
            for directory, _, names in os.walk(path):
                for name in sorted(names):
                    full = os.path.join(directory, name)
                    if os.path.isfile(full):
                        found.append((os.path.realpath(full), os.stat(full).st_size))
            return found
        if not os.path.isfile(path):
            raise FileNotFoundError("not a file or directory: " + path)
        return [(os.path.realpath(path), os.stat(path).st_size)]

    def is_directory(self, path: str) -> bool:
        return os.path.isdir(path)

    def free_bytes(self, path: str) -> int:
        return shutil.disk_usage(path).free

    @contextlib.contextmanager
    def test_file(self, directory: str, size: int) -> Iterator[str]:
        """A file of SIZE random bytes in DIRECTORY, removed on exit, to measure its drive.

        The bytes are random, so no layer below can compress them, and are written past the
        page cache where the platform allows (F_NOCACHE on macOS; written back and dropped
        on Linux), so reading them reaches the drive. Its name carries this process's ID,
        so another run tells it from one a killed run left behind (see clear_test_files).
        """
        handle, path = tempfile.mkstemp(prefix=f"{TEST_FILE_PREFIX}{os.getpid()}-", dir=directory)
        try:
            try:
                if sys.platform == "darwin":
                    import fcntl

                    fcntl.fcntl(handle, getattr(fcntl, "F_NOCACHE", 48), 1)
                written = 0
                while written < size:
                    block = memoryview(os.urandom(min(64 << 20, size - written)))
                    while block:
                        count = os.write(handle, block)
                        block, written = block[count:], written + count
                os.fsync(handle)
                if hasattr(os, "posix_fadvise"):
                    os.posix_fadvise(handle, 0, 0, os.POSIX_FADV_DONTNEED)
            finally:
                os.close(handle)
            yield path
        finally:
            with contextlib.suppress(FileNotFoundError):
                os.unlink(path)

    def clear_test_files(self, directory: str) -> None:
        """Removes the test files in DIRECTORY whose run has ended; a live run's stay."""
        for path in glob.glob(os.path.join(glob.escape(directory), TEST_FILE_PREFIX + "*")):
            owner = owner_of_test_file(path)
            if owner is not None and _running(owner):
                continue
            with contextlib.suppress(OSError):
                os.unlink(path)

    def glob(self, pattern: str) -> list[str]:
        return sorted(glob.glob(pattern))

    def page_bytes(self) -> int:
        return os.sysconf("SC_PAGE_SIZE")


def is_test_file(path: str) -> bool:
    return posixpath.basename(path).startswith(TEST_FILE_PREFIX)


def owner_of_test_file(path: str) -> Optional[int]:
    """The process ID a test file's name carries, or None for a name without one."""
    head = posixpath.basename(path)[len(TEST_FILE_PREFIX) :].split("-", 1)[0]
    return int(head) if head.isdigit() else None


def _running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:  # another user's process
        return True
    return True


def sha256(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def read(host: Host, path: str) -> Optional[str]:
    """The stripped content of PATH, or None when it is absent, unreadable or empty."""
    try:
        value = host.text(path).replace("\0", "").strip()
    except OSError:
        return None
    return value or None


def output(host: Host, argv: Sequence[str]) -> Optional[str]:
    result = host.command(argv)
    return result.stdout.strip() or None if result.returncode == 0 else None


def same_hardware(saved: Identity, live: Identity) -> list[str]:
    """Names of hard identity fields that differ; RAM compares within MEMORY_TOLERANCE."""
    old: Mapping[str, object] = saved["hard"]
    new: Mapping[str, object] = live["hard"]
    exact = ("platform", "arch", "machine_id_sha256", "model", "cpu_brand", "logical_cpus")
    changed = [key for key in exact if old[key] != new[key]]
    before, after = saved["hard"]["ram_bytes"], live["hard"]["ram_bytes"]
    if abs(before - after) > MEMORY_TOLERANCE * after:
        changed.append("ram_bytes")
    if sorted(saved["hard"]["gpus"], key=lambda gpu: gpu["uuid"]) != sorted(
        live["hard"]["gpus"], key=lambda gpu: gpu["uuid"]
    ):
        changed.append("gpus")
    return changed


def live_identity(host: Host) -> Identity:
    system = host.platform()
    if system == "linux":
        return _linux(host)
    if system == "darwin":
        return _darwin(host)
    raise UnsupportedPlatform(f"machine profiling is not supported on {system} yet")


def online_cpus(host: Host) -> tuple[int, ...]:
    listed = read(host, "/sys/devices/system/cpu/online")
    if listed:
        return cpus(listed)
    count = sum(line.startswith("processor") for line in host.text("/proc/cpuinfo").splitlines())
    return tuple(range(count))


def meminfo_bytes(host: Host, key: str) -> int:
    for line in host.text("/proc/meminfo").splitlines():
        fields = line.split()
        if len(fields) >= 2 and fields[0] == key + ":":
            return int(fields[1]) * (1024 if fields[2:3] == ["kB"] else 1)
    raise ValueError(f"/proc/meminfo has no {key}")


def cpu_brand(cpuinfo: str) -> str:
    """The x86 model name; Arm kernels list implementer and part codes instead."""
    fields: dict[str, set[str]] = {}
    for line in cpuinfo.splitlines():
        key, colon, value = line.partition(":")
        if colon and value.strip():
            fields.setdefault(key.strip(), set()).add(value.strip())
    for key in ("model name", "Hardware", "cpu model"):
        if key in fields:
            return " | ".join(sorted(fields[key]))
    if "CPU implementer" in fields and "CPU part" in fields:
        return (
            "Arm implementer "
            + ",".join(sorted(fields["CPU implementer"]))
            + " parts "
            + ",".join(sorted(fields["CPU part"]))
        )
    return "unknown"


def nvidia(host: Host) -> tuple[list[GPUIdentity], Optional[str], Optional[str]]:
    """GPUs, driver version, and why nvidia-smi failed when it is installed but fails."""
    if not host.which("nvidia-smi"):
        return [], None, None
    result = host.command(
        [
            "nvidia-smi",
            "--query-gpu=uuid,name,memory.total,driver_version",
            "--format=csv,noheader,nounits",
        ]
    )
    if result.returncode:
        lines = [
            line.strip()
            for line in (result.stdout + "\n" + result.stderr).splitlines()
            if line.strip()
        ]
        if any("No devices were found" in line for line in lines):
            return [], None, None
        reason = lines[0] if lines else "no reason given"
        return [], None, f"nvidia-smi failed with status {result.returncode}: {reason}"
    gpus: list[GPUIdentity] = []
    drivers = set()
    for line in result.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 4:
            continue
        uuid, name, memory, driver = fields
        # Unified-memory parts report no dedicated memory.
        vram = int(memory) * 1048576 if memory.isdigit() else 0
        gpus.append({"backend": "CUDA", "uuid": uuid, "name": name, "vram_bytes": vram})
        drivers.add(driver)
    return gpus, ",".join(sorted(drivers)) or None, None


# The GPU backend for each PCI vendor whose GPUs the engine runs: NVIDIA's on CUDA, as nvidia-smi
# lists them; AMD's and Intel's on Vulkan. Apple's GPU runs on Metal and is found through ioreg.
PCI_BACKENDS = {"10de": "CUDA", "1002": "Vulkan", "8086": "Vulkan"}
# The order an engine's build prefers the backends of a machine's GPUs in.
BUILD_ORDER = ("CUDA", "Metal", "Vulkan")


def build_backend(gpus: Sequence[GPUIdentity]) -> Optional[str]:
    """The GPU backend to build the engine with for GPUS, or None to build it for the CPU alone."""
    present = {gpu["backend"] for gpu in gpus}
    return next((backend for backend in BUILD_ORDER if backend in present), None)


def linux_adapters(host: Host) -> list[GPUIdentity]:
    """The PCI display controllers a Vulkan build can use (PCI_BACKENDS), by their bus address."""
    root = "/sys/bus/pci/devices"
    try:
        addresses = host.listdir(root)
    except OSError:
        return []
    found: list[GPUIdentity] = []
    for address in addresses:
        base = f"{root}/{address}/"
        kind, vendor, device = (read(host, base + name) for name in ("class", "vendor", "device"))
        vendor = (vendor or "").lower().removeprefix("0x")
        if not (kind or "").startswith("0x03") or PCI_BACKENDS.get(vendor) != "Vulkan":
            continue
        # Only a GPU with its own memory reports it (amdgpu); an integrated one shares the host's.
        vram = read(host, base + "mem_info_vram_total")
        found.append(
            {
                "backend": "Vulkan",
                "uuid": "pci-" + address,
                "name": f"PCI {vendor}:{(device or '').lower().removeprefix('0x')}",
                "vram_bytes": int(vram) if vram and vram.isdigit() else 0,
            }
        )
    return found


def probe_versions(system: str, gpus: Sequence[GPUIdentity]) -> dict[str, str]:
    versions = {"ssd": PROBE_SCHEMAS["ssd"], "cpu": PROBE_SCHEMAS["cpu"]}
    if system == "linux" and any(gpu["backend"] == "CUDA" for gpu in gpus):
        versions["memory"] = PROBE_SCHEMAS["memory"]
    return versions


def os_release(host: Host) -> Optional[str]:
    try:
        content = host.text("/etc/os-release")
    except OSError:
        return None
    values = {}
    for line in content.splitlines():
        key, equal, value = line.partition("=")
        if equal:
            values[key.strip()] = value.strip().strip("\"'")
    return values.get("PRETTY_NAME") or (
        " ".join(filter(None, (values.get("NAME"), values.get("VERSION_ID")))) or None
    )


def with_saved_gpus(live: Identity, saved: Identity) -> Identity:
    """LIVE with the GPUs and driver of SAVED, for when the GPUs cannot be queried now."""
    value = copy.deepcopy(live)
    value["hard"]["gpus"] = copy.deepcopy(saved["hard"]["gpus"])
    value["soft"]["gpu_driver"] = saved["soft"]["gpu_driver"]
    value["soft"]["probe_versions"] = probe_versions(
        live["hard"]["platform"], value["hard"]["gpus"]
    )
    return value


def _linux(host: Host) -> Identity:
    machine_id = read(host, "/etc/machine-id") or read(host, "/var/lib/dbus/machine-id")
    gpus, driver, failure = nvidia(host)
    gpus = gpus + linux_adapters(host)
    identity: Identity = {
        "hard": {
            "platform": "linux",
            "arch": host.machine(),
            "machine_id_sha256": sha256(machine_id) if machine_id else None,
            "model": read(host, "/sys/devices/virtual/dmi/id/product_name")
            or read(host, "/proc/device-tree/model"),
            "cpu_brand": cpu_brand(host.text("/proc/cpuinfo")),
            "logical_cpus": len(online_cpus(host)),
            "ram_bytes": meminfo_bytes(host, "MemTotal"),
            "gpus": gpus,
        },
        "soft": {
            "os_release": os_release(host),
            "kernel": output(host, ["uname", "-r"]) or "unknown",
            "gpu_driver": driver,
            "probe_versions": probe_versions("linux", gpus),
        },
    }
    if failure:
        raise GPUQueryFailed(failure, identity)
    return identity


def sysctl(host: Host, name: str) -> Optional[str]:
    return output(host, ["sysctl", "-n", name])


def sysctl_int(host: Host, name: str) -> int:
    value = sysctl(host, name)
    if value is None or not value.isdigit():
        raise RuntimeError("sysctl cannot read " + name)
    return int(value)


def metal(host: Host) -> list[GPUIdentity]:
    """The Apple GPU as ioreg names it, or nothing when ioreg or a key is missing."""
    listing = output(host, ["ioreg", "-rd1", "-c", "AGXAccelerator"]) or ""
    model = re.search(r'^\s*"model"\s*=\s*"([^"]+)"\s*$', listing, re.MULTILINE)
    cores = re.search(r'^\s*"gpu-core-count"\s*=\s*(\d+)\s*$', listing, re.MULTILINE)
    if model is None or cores is None:
        return []
    # The GPU has no hardware UUID; the model and core count tell the variants apart.
    # Its memory is the machine's, recorded as ram_bytes.
    return [
        {
            "backend": "Metal",
            "uuid": f"{model.group(1)}:{cores.group(1)}",
            "name": f"{model.group(1)}, {cores.group(1)}-core GPU",
            "vram_bytes": 0,
        }
    ]


def _darwin(host: Host) -> Identity:
    gpus = metal(host)
    platform_uuid = re.findall(
        r'"IOPlatformUUID"\s*=\s*"([A-Fa-f0-9-]{36})"',
        output(host, ["ioreg", "-rd1", "-c", "IOPlatformExpertDevice"]) or "",
    )
    versions = {}
    for line in (output(host, ["sw_vers"]) or "").splitlines():
        key, colon, value = line.partition(":")
        if colon:
            versions[key.strip()] = value.strip()
    release = " ".join(filter(None, (versions.get("ProductName"), versions.get("ProductVersion"))))
    if release and versions.get("BuildVersion"):
        release += f" ({versions['BuildVersion']})"
    return {
        "hard": {
            "platform": "darwin",
            "arch": host.machine(),
            "machine_id_sha256": sha256(platform_uuid[0].upper())
            if len(platform_uuid) == 1
            else None,
            "model": sysctl(host, "hw.model"),
            "cpu_brand": sysctl(host, "machdep.cpu.brand_string") or "unknown",
            "logical_cpus": sysctl_int(host, "hw.logicalcpu"),
            "ram_bytes": sysctl_int(host, "hw.memsize"),
            "gpus": gpus,
        },
        "soft": {
            "os_release": release or None,
            "kernel": output(host, ["uname", "-r"]) or "unknown",
            "gpu_driver": None,
            "probe_versions": probe_versions("darwin", gpus),
        },
    }


@dataclass(frozen=True)
class Drive:
    """The physical drive (or, without one, the filesystem) that holds a file."""

    key: str
    model: str
    bus: str


def drive_key(host: Host, path: str) -> str:
    return drive_info(host, path).key


def drive_info(host: Host, path: str) -> Drive:
    system = host.platform()
    if system == "linux":
        return _linux_drive(host, path)
    if system == "darwin":
        return _darwin_drive(host, path)
    raise UnsupportedPlatform(f"drive identity is not supported on {system} yet")


def _unescape(value: str) -> str:
    return re.sub(r"\\([0-7]{3})", lambda match: chr(int(match.group(1), 8)), value)


def _mount(host: Host, path: str, device: str) -> Optional[tuple[str, str, str]]:
    """(filesystem type, mount point, source) of the deepest mount holding PATH."""
    target = host.realpath(path)
    best: Optional[tuple[int, bool, tuple[str, str, str]]] = None
    for line in host.text("/proc/self/mountinfo").splitlines():
        fields = line.split()
        if "-" not in fields[6:]:
            continue
        tail = fields[fields.index("-", 6) + 1 :]
        point = _unescape(fields[4])
        inside = target == point or target.startswith(point.rstrip("/") + "/")
        if not inside or len(tail) < 2:
            continue
        rank = (len(point), fields[2] == device, (tail[0], point, _unescape(tail[1])))
        if best is None or rank[:2] > best[:2]:
            best = rank
    return best[2] if best else None


def _bus(name: str, directory: str) -> str:
    if "/usb" in directory:
        return "usb"
    if name.startswith("nvme"):
        return "nvme"
    if "/ata" in directory:
        return "sata"
    return "unknown"


def _block(host: Host, directory: str, depth: int = 0) -> Drive:
    if host.exists(directory + "/partition"):
        directory = posixpath.dirname(directory)
    slaves = directory + "/slaves"
    members = host.listdir(slaves) if host.exists(slaves) else []
    if members and depth < 8:
        parts = sorted(
            {_block(host, host.realpath(slaves + "/" + name), depth + 1) for name in members},
            key=lambda drive: drive.key,
        )
        if len(parts) == 1:
            return parts[0]
        buses = {drive.bus for drive in parts}
        return Drive(
            "linux:array:" + "+".join(drive.key for drive in parts),
            " + ".join(drive.model for drive in parts),
            buses.pop() if len(buses) == 1 else "unknown",
        )
    name = posixpath.basename(directory)
    model = " ".join((read(host, directory + "/device/model") or "unknown").split())
    serial = (
        read(host, directory + "/device/serial")
        or read(host, directory + "/serial")
        or read(host, directory + "/wwid")
        or read(host, directory + "/device/wwid")
    )
    bus = _bus(name, directory)
    # Without a serial (virtual disks) the kernel name keeps two disks apart.
    unique = sha256(serial) if serial else "noserial-" + name
    return Drive(f"linux:{bus}:{model}:{unique}", model, bus)


def _linux_drive(host: Host, path: str) -> Drive:
    major, minor = host.device(path)
    block = f"/sys/dev/block/{major}:{minor}"
    if host.exists(block):
        return _block(host, host.realpath(block))
    mount = _mount(host, path, f"{major}:{minor}")
    if mount is None:
        return Drive("linux:fs:unknown:/", "unknown", "unknown")
    kind, point, source = mount
    # Some filesystems (btrfs) report an anonymous device; their source names the disk.
    if source.startswith("/dev/"):
        named = "/sys/class/block/" + posixpath.basename(host.realpath(source))
        if host.exists(named):
            return _block(host, host.realpath(named))
    return Drive(f"linux:fs:{kind}:{point}", kind, "unknown")


def _plist(host: Host, argv: Sequence[str]) -> Optional[dict[str, object]]:
    """What diskutil reports, or None when it knows no disk there (network, devfs, FUSE)."""
    result = host.command(argv)
    if result.returncode:
        return None
    try:
        value = plistlib.loads(result.stdout.encode())
    except (ValueError, ExpatError) as error:
        raise RuntimeError(
            f"{' '.join(argv)} printed no property list ({error}): {result.stdout.strip()[:200]}"
        ) from error
    if not isinstance(value, dict):
        raise RuntimeError(
            f"{' '.join(argv)} returned no dictionary: {result.stdout.strip()[:200]}"
        )
    return value


def _darwin_filesystem(host: Host, point: str) -> str:
    """The filesystem type ``mount`` lists for mount point POINT."""
    for line in (output(host, ["mount"]) or "").splitlines():
        _, marker, rest = line.partition(f" on {point} (")
        if marker:
            return rest.split(",")[0].rstrip(")").strip() or "unknown"
    return "unknown"


def _darwin_drive(host: Host, path: str) -> Drive:
    df = host.command(["df", "-P", path])
    lines = df.stdout.strip().splitlines()
    match = re.search(r"\s\d+%\s+(.+)$", lines[-1]) if df.returncode == 0 and lines else None
    if match is None:
        reason = df.stderr.strip() or df.stdout.strip() or f"status {df.returncode}"
        raise RuntimeError(f"df cannot find the mount point of {path}: {reason}")
    point = match.group(1)
    volume = _plist(host, ["diskutil", "info", "-plist", point])
    if volume is None:
        kind = _darwin_filesystem(host, point)
        return Drive(f"darwin:fs:{kind}:{point}", kind, "unknown")
    stores = volume.get("APFSPhysicalStores")
    store = stores[0].get("APFSPhysicalStore") if isinstance(stores, list) and stores else None
    physical = str(store or volume.get("ParentWholeDisk") or volume.get("DeviceIdentifier", ""))
    whole = re.match(r"disk\d+", physical)
    disk = (_plist(host, ["diskutil", "info", "-plist", whole.group()]) if whole else None) or {}
    model = " ".join(str(disk.get("MediaName") or volume.get("MediaName") or "unknown").split())
    protocol = str(disk.get("BusProtocol") or volume.get("BusProtocol") or "")
    bus = (
        "apple"
        if protocol == "Apple Fabric"
        else "usb"
        if protocol == "USB"
        else "sata"
        if protocol == "SATA"
        else "nvme"
        if protocol in ("PCI-Express", "NVMe")
        else "unknown"
    )
    uuid = volume.get("VolumeUUID") or volume.get("DiskUUID")
    digest = sha256(str(uuid).upper()) if uuid else "nouuid"
    return Drive(f"darwin:{bus}:{model}:{digest}", model, bus)
