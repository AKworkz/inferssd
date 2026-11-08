# SPDX-License-Identifier: Apache-2.0
"""What machine profiling reads from Windows, through Win32 and the registry.

``System`` is what the profile asks; ``Win32`` answers it with ctypes calls
into kernel32 and registry reads. Both are injectable, so tests run every
Windows path on any OS. Win32 structures are read from raw bytes with their
fixed layouts, never through ctypes.wintypes, whose sizes follow the OS Python
runs on.

Logical CPUs are logical processor indices of processor group 0, as in the
engine's platform layer, which refuses machines with more than one group.
"""

from __future__ import annotations

import ctypes
import struct
import sys
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Optional, Protocol, Union

Value = Union[str, int, None]
Registry = Callable[[str, str], Value]

ALL_PROCESSOR_GROUPS = 0xFFFF
CPU_SET_INFORMATION = 0  # CPU_SET_INFORMATION_TYPE CpuSetInformation
DRIVE_REMOTE = 4
FILE_SHARE_READ_WRITE = 0x1 | 0x2
OPEN_EXISTING = 3
IOCTL_VOLUME_GET_VOLUME_DISK_EXTENTS = 0x00560000
IOCTL_STORAGE_QUERY_PROPERTY = 0x002D1400
MAX_EXTENTS = 64
INVALID_HANDLE = ctypes.c_void_p(-1).value
MAX_PATH_CHARS = 32768
# The display adapters' device class: one numbered subkey per driver instance.
DISPLAY_CLASS = (
    r"SYSTEM\CurrentControlSet\Control\Class\{4d36e968-e325-11ce-bfc1-08002be10318}" + "\\"
)
MAX_ADAPTERS = 16


@dataclass(frozen=True)
class CPUSet:
    """A CPU Set of processor group 0, as GetSystemCpuSetInformation reports it."""

    index: int  # logical processor index in group 0
    core: int  # CoreIndex: equal for the CPU Sets of one core
    efficiency: int  # EfficiencyClass: a higher class is a faster core


@dataclass(frozen=True)
class Processors:
    groups: int  # active processor groups
    logical: int  # active logical processors in all groups
    group0: int  # active logical processors in group 0
    sets: tuple[CPUSet, ...]  # group 0's CPU Sets; empty where Windows has none


@dataclass(frozen=True)
class Memory:
    total_bytes: int
    available_bytes: int


@dataclass(frozen=True)
class Volume:
    """The volume that holds a path."""

    point: str  # its mount point, such as C:\ or \\server\share\
    filesystem: str  # such as NTFS; "unknown" when Windows does not say
    disks: tuple[int, ...]  # its physical disk numbers; none for a network or diskless volume


@dataclass(frozen=True)
class Disk:
    """A physical disk, as IOCTL_STORAGE_QUERY_PROPERTY describes it."""

    number: int
    vendor: str
    product: str
    serial: str
    bus: int  # STORAGE_BUS_TYPE


@dataclass(frozen=True)
class Adapter:
    """A display adapter, as its driver's registry key describes it."""

    name: str  # DriverDesc
    device: str  # MatchingDeviceId, such as PCI\\VEN_8086&DEV_B090&SUBSYS_389117AA
    memory_bytes: int  # HardwareInformation.qwMemorySize: its own memory; 0 when it has none


class System(Protocol):
    def registry(self, key: str, name: str) -> Value: ...
    def adapters(self) -> tuple[Adapter, ...]: ...
    def processors(self) -> Processors: ...
    def process_cpus(self) -> tuple[int, ...]: ...
    def memory(self) -> Memory: ...
    def volume(self, path: str) -> Volume: ...
    def disk(self, number: int) -> Disk: ...


def cpu_sets(data: bytes) -> list[CPUSet]:
    """Group 0's CPU Sets in a SYSTEM_CPU_SET_INFORMATION array."""
    found = []
    at = 0
    while at + 8 <= len(data):
        size, kind = struct.unpack_from("<II", data, at)
        if size < 8 or at + size > len(data):
            break
        if kind == CPU_SET_INFORMATION and size >= 19:
            # Id, Group, LogicalProcessorIndex, CoreIndex, LastLevelCacheIndex,
            # NumaNodeIndex, EfficiencyClass
            _, group, index, core, _, _, efficiency = struct.unpack_from("<IHBBBBB", data, at + 8)
            if group == 0:
                found.append(CPUSet(index, core, efficiency))
        at += size
    return found


def disk_extents(data: bytes) -> tuple[int, ...]:
    """The disk numbers in a VOLUME_DISK_EXTENTS structure, ascending."""
    (count,) = struct.unpack_from("<I", data)
    count = min(count, (len(data) - 8) // 24)
    return tuple(sorted({struct.unpack_from("<I", data, 8 + 24 * i)[0] for i in range(count)}))


def storage_descriptor(number: int, data: bytes) -> Disk:
    """Disk NUMBER as a STORAGE_DEVICE_DESCRIPTOR describes it."""
    if len(data) < 32:
        raise OSError(f"PhysicalDrive{number}: the storage descriptor has {len(data)} bytes")
    vendor, product, _, serial, bus = struct.unpack_from("<IIIII", data, 12)

    def text(offset: int) -> str:
        if not 0 < offset < len(data):
            return ""
        end = data.find(b"\0", offset)
        raw = data[offset : end if end >= 0 else len(data)]
        return raw.decode("ascii", "replace").strip()

    return Disk(number, text(vendor), text(product), text(serial), bus)


def native_registry(key: str, name: str) -> Value:
    """Value NAME of HKEY_LOCAL_MACHINE\\KEY in the 64-bit view, or None when absent."""
    if sys.platform != "win32":
        raise OSError("the Windows registry exists on Windows only")
    import winreg

    access = winreg.KEY_READ | winreg.KEY_WOW64_64KEY
    try:
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, key, 0, access) as handle:
            value, _ = winreg.QueryValueEx(handle, name)
    except OSError:
        return None
    return value if isinstance(value, (str, int)) else None


def native_kernel32() -> Any:
    if sys.platform != "win32":
        raise OSError("kernel32 exists on Windows only")
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    # Handles are pointer sized; the default int result would truncate them.
    kernel32.CreateFileW.restype = ctypes.c_void_p
    kernel32.GetCurrentProcess.restype = ctypes.c_void_p
    return kernel32


def native_last_error() -> int:
    if sys.platform != "win32":
        return 0
    return ctypes.get_last_error()


class Win32:
    """System on this Windows host; KERNEL32, REGISTRY and LAST_ERROR replace the native ones."""

    def __init__(
        self,
        kernel32: Any = None,
        registry: Optional[Registry] = None,
        last_error: Optional[Callable[[], int]] = None,
    ) -> None:
        self.kernel32 = kernel32 if kernel32 is not None else native_kernel32()
        self.read = registry or native_registry
        self.last_error = last_error or native_last_error

    def error(self, call: str, subject: str = "") -> OSError:
        about = f" ({subject})" if subject else ""
        return OSError(f"{call}{about} failed with Windows error {self.last_error()}")

    def registry(self, key: str, name: str) -> Value:
        return self.read(key, name)

    def adapters(self) -> tuple[Adapter, ...]:
        found = []
        for index in range(MAX_ADAPTERS):
            key = f"{DISPLAY_CLASS}{index:04d}"
            name, device = self.read(key, "DriverDesc"), self.read(key, "MatchingDeviceId")
            if isinstance(name, str) and name and isinstance(device, str) and device:
                memory = self.read(key, "HardwareInformation.qwMemorySize")
                found.append(Adapter(name, device, memory if isinstance(memory, int) else 0))
        return tuple(found)

    def processors(self) -> Processors:
        kernel32 = self.kernel32
        groups = int(kernel32.GetActiveProcessorGroupCount())
        logical = int(kernel32.GetActiveProcessorCount(ALL_PROCESSOR_GROUPS))
        group0 = int(kernel32.GetActiveProcessorCount(0))
        if not groups or not logical or not group0:
            raise self.error("GetActiveProcessorCount")
        length = ctypes.c_uint32(0)
        kernel32.GetSystemCpuSetInformation(None, 0, ctypes.byref(length), None, 0)
        if not length.value:
            return Processors(groups, logical, group0, ())
        buffer = ctypes.create_string_buffer(length.value)
        if not kernel32.GetSystemCpuSetInformation(
            buffer, length.value, ctypes.byref(length), None, 0
        ):
            raise self.error("GetSystemCpuSetInformation")
        sets = cpu_sets(buffer.raw[: length.value])
        return Processors(groups, logical, group0, tuple(s for s in sets if s.index < group0))

    def process_cpus(self) -> tuple[int, ...]:
        kernel32 = self.kernel32
        if int(kernel32.GetActiveProcessorGroupCount()) != 1:
            raise OSError("more than one processor group (over 64 logical CPUs) is not supported")
        process, system = ctypes.c_uint64(0), ctypes.c_uint64(0)
        handle = ctypes.c_void_p(kernel32.GetCurrentProcess())
        if not kernel32.GetProcessAffinityMask(handle, ctypes.byref(process), ctypes.byref(system)):
            raise self.error("GetProcessAffinityMask")
        return tuple(cpu for cpu in range(64) if process.value >> cpu & 1)

    def memory(self) -> Memory:
        # MEMORYSTATUSEX: dwLength must hold its size, 64 bytes.
        buffer = ctypes.create_string_buffer(struct.pack("<I", 64), 64)
        if not self.kernel32.GlobalMemoryStatusEx(buffer):
            raise self.error("GlobalMemoryStatusEx")
        total, available = struct.unpack_from("<QQ", buffer.raw, 8)
        return Memory(total, available)

    def volume(self, path: str) -> Volume:
        kernel32 = self.kernel32
        point = ctypes.create_unicode_buffer(MAX_PATH_CHARS)
        if not kernel32.GetVolumePathNameW(path, point, len(point)):
            raise self.error("GetVolumePathNameW", path)
        root = point.value
        name = ctypes.create_unicode_buffer(261)
        known = kernel32.GetVolumeInformationW(root, None, 0, None, None, None, name, len(name))
        filesystem = (name.value if known else "") or "unknown"
        if kernel32.GetDriveTypeW(root) == DRIVE_REMOTE:
            return Volume(root, filesystem, ())
        guid = ctypes.create_unicode_buffer(261)
        if not kernel32.GetVolumeNameForVolumeMountPointW(root, guid, len(guid)):
            return Volume(root, filesystem, ())  # no volume device, such as a substituted drive
        # The volume device is its GUID path without the trailing backslash.
        handle = self.open(guid.value.rstrip("\\"))
        if handle is None:
            return Volume(root, filesystem, ())
        try:
            extents = ctypes.create_string_buffer(8 + 24 * MAX_EXTENTS)
            returned = ctypes.c_uint32(0)
            if not kernel32.DeviceIoControl(
                ctypes.c_void_p(handle),
                IOCTL_VOLUME_GET_VOLUME_DISK_EXTENTS,
                None,
                0,
                extents,
                len(extents),
                ctypes.byref(returned),
                None,
            ):
                return Volume(root, filesystem, ())  # no disk behind it, such as a RAM disk
            return Volume(root, filesystem, disk_extents(extents.raw[: returned.value]))
        finally:
            kernel32.CloseHandle(ctypes.c_void_p(handle))

    def disk(self, number: int) -> Disk:
        device = f"\\\\.\\PhysicalDrive{number}"
        handle = self.open(device)
        if handle is None:
            raise self.error("CreateFileW", device)
        try:
            # STORAGE_PROPERTY_QUERY: StorageDeviceProperty, PropertyStandardQuery.
            query = struct.pack("<II4x", 0, 0)
            descriptor = ctypes.create_string_buffer(4096)
            returned = ctypes.c_uint32(0)
            if not self.kernel32.DeviceIoControl(
                ctypes.c_void_p(handle),
                IOCTL_STORAGE_QUERY_PROPERTY,
                query,
                len(query),
                descriptor,
                len(descriptor),
                ctypes.byref(returned),
                None,
            ):
                raise self.error("IOCTL_STORAGE_QUERY_PROPERTY", device)
            return storage_descriptor(number, descriptor.raw[: returned.value])
        finally:
            self.kernel32.CloseHandle(ctypes.c_void_p(handle))

    def open(self, device: str) -> Optional[int]:
        """A handle to DEVICE opened for queries only (no read or write access), or None."""
        handle = self.kernel32.CreateFileW(
            device, 0, FILE_SHARE_READ_WRITE, None, OPEN_EXISTING, 0, None
        )
        return None if handle in (None, 0, INVALID_HANDLE) else int(handle)
