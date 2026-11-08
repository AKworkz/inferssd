# SPDX-License-Identifier: Apache-2.0
"""The Win32 calls of the Windows ownership mechanisms, through ctypes.

This is the only module that reaches the Win32 API. Each function makes one
documented call or a short fixed sequence of them and raises OSError when one
fails. Nothing is loaded until a function is called, so the modules built on
these functions import on any platform; their tests replace these functions.
"""

from __future__ import annotations

import ctypes
import mmap
import os
import socket
import struct
import subprocess
import sys
from collections.abc import Mapping, Sequence
from ctypes import (
    POINTER,
    Structure,
    byref,
    c_int,
    c_int64,
    c_size_t,
    c_uint32,
    c_uint64,
    c_void_p,
    c_wchar,
    c_wchar_p,
    sizeof,
)
from functools import cache
from typing import IO, Any, Optional, Union

# Process creation flags.
CREATE_SUSPENDED = 0x00000004
CREATE_NEW_PROCESS_GROUP = 0x00000200
CREATE_NO_WINDOW = 0x08000000

_INVALID_HANDLE = c_void_p(-1).value
_ERROR_SHARING_VIOLATION = 32
_ERROR_BROKEN_PIPE = 109
_ERROR_INSUFFICIENT_BUFFER = 122
_ERROR_NO_DATA = 232
_ERROR_MORE_DATA = 234

# Job object information classes and limit flags.
_BASIC_ACCOUNTING, _PROCESS_LIST, _EXTENDED_LIMITS, _LIMIT_VIOLATION = 1, 3, 9, 13
_LIMIT_PROCESS_MEMORY = 0x0100
_LIMIT_JOB_MEMORY = 0x0200
_LIMIT_DIE_ON_UNHANDLED_EXCEPTION = 0x0400
_LIMIT_KILL_ON_JOB_CLOSE = 0x2000


class _BasicLimits(Structure):
    _fields_ = [
        ("PerProcessUserTimeLimit", c_int64),
        ("PerJobUserTimeLimit", c_int64),
        ("LimitFlags", c_uint32),
        ("MinimumWorkingSetSize", c_size_t),
        ("MaximumWorkingSetSize", c_size_t),
        ("ActiveProcessLimit", c_uint32),
        ("Affinity", c_size_t),
        ("PriorityClass", c_uint32),
        ("SchedulingClass", c_uint32),
    ]


class _ExtendedLimits(Structure):
    _fields_ = [
        ("BasicLimitInformation", _BasicLimits),
        ("IoInfo", c_uint64 * 6),
        ("ProcessMemoryLimit", c_size_t),
        ("JobMemoryLimit", c_size_t),
        ("PeakProcessMemoryUsed", c_size_t),
        ("PeakJobMemoryUsed", c_size_t),
    ]


class _Accounting(Structure):
    _fields_ = [
        ("TotalUserTime", c_int64),
        ("TotalKernelTime", c_int64),
        ("ThisPeriodTotalUserTime", c_int64),
        ("ThisPeriodTotalKernelTime", c_int64),
        ("TotalPageFaultCount", c_uint32),
        ("TotalProcesses", c_uint32),
        ("ActiveProcesses", c_uint32),
        ("TotalTerminatedProcesses", c_uint32),
    ]


class _Violation(Structure):
    _fields_ = [
        ("LimitFlags", c_uint32),
        ("ViolationLimitFlags", c_uint32),
        ("IoReadBytes", c_uint64),
        ("IoReadBytesLimit", c_uint64),
        ("IoWriteBytes", c_uint64),
        ("IoWriteBytesLimit", c_uint64),
        ("PerJobUserTime", c_int64),
        ("PerJobUserTimeLimit", c_int64),
        ("JobMemory", c_uint64),
        ("JobMemoryLimit", c_uint64),
        ("RateControlTolerance", c_int),
        ("RateControlToleranceLimit", c_int),
    ]


class _ProcessMemory(Structure):
    _fields_ = [("cb", c_uint32), ("PageFaultCount", c_uint32)] + [
        (name, c_size_t)
        for name in (
            "PeakWorkingSetSize",
            "WorkingSetSize",
            "QuotaPeakPagedPoolUsage",
            "QuotaPagedPoolUsage",
            "QuotaPeakNonPagedPoolUsage",
            "QuotaNonPagedPoolUsage",
            "PagefileUsage",
            "PeakPagefileUsage",
            "PrivateUsage",
        )
    ]


class _MemoryStatus(Structure):
    _fields_ = [("dwLength", c_uint32), ("dwMemoryLoad", c_uint32)] + [
        (name, c_uint64)
        for name in (
            "ullTotalPhys",
            "ullAvailPhys",
            "ullTotalPageFile",
            "ullAvailPageFile",
            "ullTotalVirtual",
            "ullAvailVirtual",
            "ullAvailExtendedVirtual",
        )
    ]


class _ThreadEntry(Structure):
    _fields_ = [
        ("dwSize", c_uint32),
        ("cntUsage", c_uint32),
        ("th32ThreadID", c_uint32),
        ("th32OwnerProcessID", c_uint32),
        ("tpBasePri", c_int),
        ("tpDeltaPri", c_int),
        ("dwFlags", c_uint32),
    ]


class _ProcessEntry(Structure):
    _fields_ = [
        ("dwSize", c_uint32),
        ("cntUsage", c_uint32),
        ("th32ProcessID", c_uint32),
        ("th32DefaultHeapID", c_size_t),
        ("th32ModuleID", c_uint32),
        ("cntThreads", c_uint32),
        ("th32ParentProcessID", c_uint32),
        ("pcPriClassBase", c_int),
        ("dwFlags", c_uint32),
        ("szExeFile", c_wchar * 260),
    ]


class _RawCounter(Structure):
    _fields_ = [
        ("CStatus", c_uint32),
        ("TimeStamp", c_uint32 * 2),
        ("FirstValue", c_int64),
        ("SecondValue", c_int64),
        ("MultiCount", c_uint32),
    ]


class _KmtAdapterInfo(Structure):
    _fields_ = [
        ("hAdapter", c_uint32),
        ("AdapterLuidLowPart", c_uint32),
        ("AdapterLuidHighPart", c_int),
        ("NumOfSources", c_uint32),
        ("bPrecisePresentRegionsPreferred", c_int),
    ]


class _KmtEnumAdapters2(Structure):
    _fields_ = [("NumAdapters", c_uint32), ("pAdapters", POINTER(_KmtAdapterInfo))]


class _KmtVideoMemoryInfo(Structure):
    _fields_ = [
        ("hProcess", c_void_p),
        ("hAdapter", c_uint32),
        ("MemorySegmentGroup", c_int),
        ("Budget", c_uint64),
        ("CurrentUsage", c_uint64),
        ("CurrentReservation", c_uint64),
        ("AvailableForReservation", c_uint64),
        ("PhysicalAdapterIndex", c_uint32),
    ]


class _KmtSegmentSize(Structure):
    _fields_ = [
        ("DedicatedVideoMemorySize", c_uint64),
        ("DedicatedSystemMemorySize", c_uint64),
        ("SharedSystemMemorySize", c_uint64),
    ]


class _KmtQueryAdapterInfo(Structure):
    _fields_ = [
        ("hAdapter", c_uint32),
        ("Type", c_int),
        ("pPrivateDriverData", c_void_p),
        ("PrivateDriverDataSize", c_uint32),
    ]


_H, _D, _B, _P = c_void_p, c_uint32, c_int, c_void_p
_PROTOTYPES: Mapping[str, Mapping[str, tuple[Any, list[Any]]]] = {
    "kernel32": {
        "AssignProcessToJobObject": (_B, [_H, _H]),
        "CloseHandle": (_B, [_H]),
        "CreateFileW": (_H, [c_wchar_p, _D, _D, _P, _D, _D, _H]),
        "CreateJobObjectW": (_H, [_P, c_wchar_p]),
        "CreateToolhelp32Snapshot": (_H, [_D, _D]),
        "GenerateConsoleCtrlEvent": (_B, [_D, _D]),
        "GetCurrentProcess": (_H, []),
        "GlobalMemoryStatusEx": (_B, [POINTER(_MemoryStatus)]),
        "IsProcessInJob": (_B, [_H, _H, POINTER(_B)]),
        "K32GetProcessMemoryInfo": (_B, [_H, POINTER(_ProcessMemory), _D]),
        "OpenProcess": (_H, [_D, _B, _D]),
        "OpenThread": (_H, [_D, _B, _D]),
        "PeekNamedPipe": (_B, [_H, _P, _D, POINTER(_D), _P, _P]),
        "Process32FirstW": (_B, [_H, POINTER(_ProcessEntry)]),
        "Process32NextW": (_B, [_H, POINTER(_ProcessEntry)]),
        "QueryInformationJobObject": (_B, [_H, c_int, _P, _D, POINTER(_D)]),
        "ResumeThread": (_D, [_H]),
        "SetConsoleCtrlHandler": (_B, [_P, _B]),
        "SetInformationJobObject": (_B, [_H, c_int, _P, _D]),
        "TerminateJobObject": (_B, [_H, c_uint32]),
        "Thread32First": (_B, [_H, POINTER(_ThreadEntry)]),
        "Thread32Next": (_B, [_H, POINTER(_ThreadEntry)]),
    },
    "gdi32": {
        "D3DKMTCloseAdapter": (c_int, [POINTER(c_uint32)]),
        "D3DKMTEnumAdapters2": (c_int, [POINTER(_KmtEnumAdapters2)]),
        "D3DKMTQueryAdapterInfo": (c_int, [POINTER(_KmtQueryAdapterInfo)]),
        "D3DKMTQueryVideoMemoryInfo": (c_int, [POINTER(_KmtVideoMemoryInfo)]),
    },
    "iphlpapi": {"GetExtendedTcpTable": (_D, [_P, POINTER(_D), _B, _D, c_int, _D])},
    "pdh": {
        "PdhAddEnglishCounterW": (_D, [_H, c_wchar_p, c_size_t, POINTER(_H)]),
        "PdhCloseQuery": (_D, [_H]),
        "PdhCollectQueryData": (_D, [_H]),
        "PdhGetRawCounterValue": (_D, [_H, POINTER(_D), POINTER(_RawCounter)]),
        "PdhOpenQueryW": (_D, [c_wchar_p, c_size_t, POINTER(_H)]),
    },
}


@cache
def _dll(name: str) -> Any:
    if sys.platform != "win32":
        raise OSError("the Win32 API is available only on Windows")
    library = ctypes.WinDLL(name, use_last_error=True)
    for function, (result, arguments) in _PROTOTYPES[name].items():
        call = getattr(library, function)
        call.restype, call.argtypes = result, arguments
    return library


def _last_error() -> int:
    if sys.platform != "win32":
        return 0
    return ctypes.get_last_error()


def _failed(what: str, code: Optional[int] = None) -> OSError:
    if sys.platform != "win32":
        return OSError(what)
    return ctypes.WinError(ctypes.get_last_error() if code is None else code, what)


def _check(ok: Any, what: str) -> None:
    if not ok:
        raise _failed(what)


def _handle(value: Any, what: str) -> int:
    if value in (None, _INVALID_HANDLE):
        raise _failed(what)
    return int(value)


def close(handle: int) -> None:
    _check(_dll("kernel32").CloseHandle(handle), "CloseHandle")


def fd_handle(fd: int) -> int:
    """The OS handle behind a C runtime descriptor."""
    if sys.platform != "win32":
        raise OSError("C runtime handles exist only on Windows")
    import msvcrt

    return int(msvcrt.get_osfhandle(fd))


def handle_fd(handle: int) -> int:
    """A C runtime descriptor that owns handle; closing it closes the handle."""
    if sys.platform != "win32":
        raise OSError("C runtime handles exist only on Windows")
    import msvcrt

    return int(msvcrt.open_osfhandle(handle, 0))


# Files


def open_exclusive(path: str) -> Optional[int]:
    """Open or create path with no sharing; None while another open of it exists."""
    kernel = _dll("kernel32")
    # GENERIC_READ | GENERIC_WRITE, no sharing, OPEN_ALWAYS, FILE_ATTRIBUTE_NORMAL.
    value = kernel.CreateFileW(path, 0xC0000000, 0, None, 4, 0x80, None)
    if value in (None, _INVALID_HANDLE):
        code = _last_error()
        if code == _ERROR_SHARING_VIOLATION:
            return None
        raise _failed("CreateFileW " + path, code)
    return int(value)


def opened_exclusively(path: str) -> bool:
    """Whether an open of the existing file path that shares nothing exists now."""
    kernel = _dll("kernel32")
    # GENERIC_READ with full sharing fails only against an open that shares nothing.
    value = kernel.CreateFileW(path, 0x80000000, 0x7, None, 3, 0x80, None)
    if value in (None, _INVALID_HANDLE):
        code = _last_error()
        if code == _ERROR_SHARING_VIOLATION:
            return True
        raise _failed("CreateFileW " + path, code)
    close(int(value))
    return False


# Processes and consoles


def create_process(
    argv: Sequence[str],
    *,
    env: Mapping[str, str],
    stdout: Union[int, IO[Any], None],
    stderr: Union[int, IO[Any], None],
    stdin: Union[int, IO[Any], None],
    inherit: Sequence[int],
    flags: int,
) -> subprocess.Popen[bytes]:
    """Start argv inheriting exactly inherit (PROC_THREAD_ATTRIBUTE_HANDLE_LIST)."""
    if sys.platform != "win32":
        raise OSError("Win32 processes exist only on Windows")
    # The handle list names handles the child inherits; they must be inheritable
    # while the process is created, and are not otherwise.
    for handle in inherit:
        os.set_handle_inheritable(handle, True)
    try:
        return subprocess.Popen(
            list(argv),
            env=dict(env),
            stdout=stdout,
            stderr=stderr,
            stdin=stdin,
            close_fds=True,
            creationflags=flags,
            startupinfo=subprocess.STARTUPINFO(lpAttributeList={"handle_list": list(inherit)}),
        )
    finally:
        for handle in inherit:
            os.set_handle_inheritable(handle, False)


def resume(pid: int) -> None:
    """Resume the threads of a process created suspended."""
    kernel = _dll("kernel32")
    snapshot = _handle(kernel.CreateToolhelp32Snapshot(0x4, 0), "CreateToolhelp32Snapshot")
    threads = []
    try:
        entry = _ThreadEntry(dwSize=sizeof(_ThreadEntry))
        found = kernel.Thread32First(snapshot, byref(entry))
        while found:
            if entry.th32OwnerProcessID == pid:
                threads.append(int(entry.th32ThreadID))
            found = kernel.Thread32Next(snapshot, byref(entry))
    finally:
        close(snapshot)
    if not threads:
        raise OSError(f"process {pid} has no thread to resume")
    for thread in threads:
        # THREAD_SUSPEND_RESUME
        handle = _handle(kernel.OpenThread(0x0002, False, thread), "OpenThread")
        try:
            if kernel.ResumeThread(handle) == 0xFFFFFFFF:
                raise _failed("ResumeThread")
        finally:
            close(handle)


def ignore_ctrl_c(ignore: bool) -> None:
    """Set this process's Ctrl+C attribute, which the processes it starts inherit."""
    _check(_dll("kernel32").SetConsoleCtrlHandler(None, ignore), "SetConsoleCtrlHandler")


def interrupt_console() -> None:
    """Ctrl+C to every process attached to this process's console."""
    _check(_dll("kernel32").GenerateConsoleCtrlEvent(0, 0), "GenerateConsoleCtrlEvent")


def find_process(name: str) -> Optional[int]:
    """The first process whose image name is name."""
    kernel = _dll("kernel32")
    snapshot = _handle(kernel.CreateToolhelp32Snapshot(0x2, 0), "CreateToolhelp32Snapshot")
    try:
        entry = _ProcessEntry(dwSize=sizeof(_ProcessEntry))
        found = kernel.Process32FirstW(snapshot, byref(entry))
        while found:
            if entry.szExeFile == name:
                return int(entry.th32ProcessID)
            found = kernel.Process32NextW(snapshot, byref(entry))
        return None
    finally:
        close(snapshot)


def process_memory(pid: int) -> dict[str, int]:
    """GetProcessMemoryInfo of pid, in bytes."""
    kernel = _dll("kernel32")
    # PROCESS_QUERY_LIMITED_INFORMATION
    handle = _handle(kernel.OpenProcess(0x1000, False, pid), f"OpenProcess {pid}")
    try:
        counters = _ProcessMemory(cb=sizeof(_ProcessMemory))
        _check(
            kernel.K32GetProcessMemoryInfo(handle, byref(counters), counters.cb),
            "GetProcessMemoryInfo",
        )
        return {
            "private_bytes": int(counters.PrivateUsage),
            "working_set_bytes": int(counters.WorkingSetSize),
            "peak_working_set_bytes": int(counters.PeakWorkingSetSize),
            "peak_pagefile_bytes": int(counters.PeakPagefileUsage),
        }
    finally:
        close(handle)


def process_running(pid: int) -> bool:
    """Whether pid exists, without sending it a signal or requiring access to it."""
    kernel = _dll("kernel32")
    snapshot = _handle(kernel.CreateToolhelp32Snapshot(0x2, 0), "CreateToolhelp32Snapshot")
    try:
        entry = _ProcessEntry(dwSize=sizeof(_ProcessEntry))
        found = kernel.Process32FirstW(snapshot, byref(entry))
        while found:
            if entry.th32ProcessID == pid:
                return True
            found = kernel.Process32NextW(snapshot, byref(entry))
        return False
    finally:
        close(snapshot)


def process_working_set(name: str, pid: int) -> int:
    """Working-set bytes from PDH, including protected processes, with a PID check."""
    pdh = _dll("pdh")
    query = c_void_p()
    if pdh.PdhOpenQueryW(None, 0, byref(query)):
        raise OSError("PdhOpenQueryW failed")
    try:
        counters = []
        for field in ("ID Process", "Working Set"):
            counter = c_void_p()
            if pdh.PdhAddEnglishCounterW(query, f"\\Process({name})\\{field}", 0, byref(counter)):
                raise OSError("PdhAddEnglishCounterW failed")
            counters.append(counter)
        if pdh.PdhCollectQueryData(query):
            raise OSError("PdhCollectQueryData failed")
        values = []
        for counter in counters:
            raw, kind = _RawCounter(), c_uint32()
            if pdh.PdhGetRawCounterValue(counter, byref(kind), byref(raw)) or raw.CStatus > 1:
                raise OSError("process working-set counter unavailable")
            values.append(int(raw.FirstValue))
        if values[0] != pid:
            raise OSError("process working-set counter belongs to another PID")
        return values[1]
    finally:
        pdh.PdhCloseQuery(query)


# Job objects


def create_job(*, kill_on_close: bool, memory_limit: Optional[int]) -> int:
    """A new unnamed job; its processes die on an unhandled exception, without a dialog."""
    kernel = _dll("kernel32")
    job = _handle(kernel.CreateJobObjectW(None, None), "CreateJobObjectW")
    try:
        limits = _ExtendedLimits()
        flags = _LIMIT_DIE_ON_UNHANDLED_EXCEPTION
        if kill_on_close:
            flags |= _LIMIT_KILL_ON_JOB_CLOSE
        if memory_limit is not None:
            flags |= _LIMIT_JOB_MEMORY
            limits.JobMemoryLimit = memory_limit
        limits.BasicLimitInformation.LimitFlags = flags
        _check(
            kernel.SetInformationJobObject(job, _EXTENDED_LIMITS, byref(limits), sizeof(limits)),
            "SetInformationJobObject",
        )
    except BaseException:
        close(job)
        raise
    return job


def assign(job: int, pid: int) -> None:
    kernel = _dll("kernel32")
    # PROCESS_SET_QUOTA | PROCESS_TERMINATE
    process = _handle(kernel.OpenProcess(0x0100 | 0x0001, False, pid), f"OpenProcess {pid}")
    try:
        _check(kernel.AssignProcessToJobObject(job, process), "AssignProcessToJobObject")
    finally:
        close(process)


def terminate_job(job: int, code: int) -> None:
    _check(_dll("kernel32").TerminateJobObject(job, code), "TerminateJobObject")


def job_active(job: int) -> int:
    """Processes in job and in the jobs nested in it that have not exited."""
    kernel = _dll("kernel32")
    accounting = _Accounting()
    _check(
        kernel.QueryInformationJobObject(
            job, _BASIC_ACCOUNTING, byref(accounting), sizeof(accounting), None
        ),
        "QueryInformationJobObject",
    )
    return int(accounting.ActiveProcesses)


def in_job() -> bool:
    """Whether this process belongs to a job."""
    kernel = _dll("kernel32")
    result = c_int()
    _check(kernel.IsProcessInJob(kernel.GetCurrentProcess(), None, byref(result)), "IsProcessInJob")
    return bool(result.value)


def job_processes(job: Optional[int]) -> list[int]:
    """Process IDs in job, or in this process's own innermost job when job is None."""
    kernel = _dll("kernel32")
    count = 256
    while True:
        # Two DWORD counts, assigned and listed, then one ULONG_PTR per process.
        buffer = (c_size_t * (1 + count))()
        if kernel.QueryInformationJobObject(job, _PROCESS_LIST, buffer, sizeof(buffer), None):
            return [int(buffer[1 + i]) for i in range(int(buffer[0]) >> 32)]
        if _last_error() != _ERROR_MORE_DATA:
            raise _failed("QueryInformationJobObject")
        count *= 4


def job_memory(job: Optional[int]) -> dict[str, Optional[int]]:
    """Commit limits and use of job, or of this process's innermost job when job is None."""
    kernel = _dll("kernel32")
    limits, violation = _ExtendedLimits(), _Violation()
    _check(
        kernel.QueryInformationJobObject(
            job, _EXTENDED_LIMITS, byref(limits), sizeof(limits), None
        ),
        "QueryInformationJobObject",
    )
    _check(
        kernel.QueryInformationJobObject(
            job, _LIMIT_VIOLATION, byref(violation), sizeof(violation), None
        ),
        "QueryInformationJobObject",
    )
    flags = int(limits.BasicLimitInformation.LimitFlags)
    return {
        "job_memory_limit": int(limits.JobMemoryLimit) if flags & _LIMIT_JOB_MEMORY else None,
        "process_memory_limit": (
            int(limits.ProcessMemoryLimit) if flags & _LIMIT_PROCESS_MEMORY else None
        ),
        "kill_on_close": bool(flags & _LIMIT_KILL_ON_JOB_CLOSE),
        "job_memory": int(violation.JobMemory),
        "peak_job_memory": int(limits.PeakJobMemoryUsed),
    }


# Pipes


def pipe_peek(pipe: int, size: int) -> Optional[bytes]:
    """Up to size bytes waiting in pipe, left unread, or None once its writer is gone."""
    buffer, done = ctypes.create_string_buffer(size), c_uint32()
    if _dll("kernel32").PeekNamedPipe(pipe, buffer, size, byref(done), None, None):
        return buffer.raw[: done.value]
    code = _last_error()
    if code in (_ERROR_BROKEN_PIPE, _ERROR_NO_DATA):
        return None
    raise _failed("PeekNamedPipe", code)


# System memory and listeners


def system_memory() -> dict[str, int]:
    """GlobalMemoryStatusEx, in bytes."""
    status = _MemoryStatus(dwLength=sizeof(_MemoryStatus))
    _check(_dll("kernel32").GlobalMemoryStatusEx(byref(status)), "GlobalMemoryStatusEx")
    return {
        "physical_total_bytes": int(status.ullTotalPhys),
        "physical_available_bytes": int(status.ullAvailPhys),
        "commit_limit_bytes": int(status.ullTotalPageFile),
        "commit_available_bytes": int(status.ullAvailPageFile),
    }


# The non-local segment group and the adapter query for segment sizes (d3dkmthk.h).
_KMT_NON_LOCAL, _KMT_GETSEGMENTSIZE = 1, 3


def discrete_video_memory() -> int:
    """Dedicated video memory of the discrete adapters, in bytes.

    WDDM charges a device allocation there to the allocating process's commit
    without taking host memory, so a job's commit limit leaves room for it
    beside the host budget. An adapter is discrete when it also has a
    non-local segment group, the system memory it maps; an integrated
    adapter's segments are host memory.
    """
    gdi = _dll("gdi32")
    listing = _KmtEnumAdapters2()
    if gdi.D3DKMTEnumAdapters2(byref(listing)):
        raise OSError("D3DKMTEnumAdapters2 failed")
    adapters = (_KmtAdapterInfo * listing.NumAdapters)()
    listing.pAdapters = adapters
    if gdi.D3DKMTEnumAdapters2(byref(listing)):
        raise OSError("D3DKMTEnumAdapters2 failed")
    total = 0
    try:
        for adapter in adapters[: listing.NumAdapters]:
            non_local = _KmtVideoMemoryInfo(
                hAdapter=adapter.hAdapter, MemorySegmentGroup=_KMT_NON_LOCAL
            )
            if gdi.D3DKMTQueryVideoMemoryInfo(byref(non_local)) or not non_local.Budget:
                continue
            sizes = _KmtSegmentSize()
            query = _KmtQueryAdapterInfo(
                hAdapter=adapter.hAdapter,
                Type=_KMT_GETSEGMENTSIZE,
                pPrivateDriverData=ctypes.cast(byref(sizes), c_void_p),
                PrivateDriverDataSize=sizeof(sizes),
            )
            if gdi.D3DKMTQueryAdapterInfo(byref(query)):
                raise OSError("D3DKMTQueryAdapterInfo failed")
            total += int(sizes.DedicatedVideoMemorySize)
    finally:
        for adapter in adapters[: listing.NumAdapters]:
            gdi.D3DKMTCloseAdapter(byref(c_uint32(adapter.hAdapter)))
    return total


def pagefile_in_use() -> int:
    """Bytes in use across the paging files, from their % Usage performance counter."""
    pdh = _dll("pdh")
    query, counter = c_void_p(), c_void_p()
    if pdh.PdhOpenQueryW(None, 0, byref(query)):
        raise OSError("PdhOpenQueryW failed")
    try:
        if pdh.PdhAddEnglishCounterW(query, "\\Paging File(_Total)\\% Usage", 0, byref(counter)):
            raise OSError("PdhAddEnglishCounterW failed")
        if pdh.PdhCollectQueryData(query):
            raise OSError("PdhCollectQueryData failed")
        raw, kind = _RawCounter(), c_uint32()
        # PDH_CSTATUS_VALID_DATA or PDH_CSTATUS_NEW_DATA
        if pdh.PdhGetRawCounterValue(counter, byref(kind), byref(raw)) or raw.CStatus > 1:
            raise OSError("paging file usage counter unavailable")
        # A raw fraction: pages in use over the pages of all paging files.
        return int(raw.FirstValue) * mmap.PAGESIZE
    finally:
        pdh.PdhCloseQuery(query)


def tcp_listeners() -> list[tuple[str, int, int]]:
    """(address, port, owning process ID) of every IPv4 TCP listener."""
    iphlpapi = _dll("iphlpapi")
    size = c_uint32(0)
    while True:
        buffer = ctypes.create_string_buffer(size.value or 4)
        # AF_INET, TCP_TABLE_OWNER_PID_LISTENER
        code = iphlpapi.GetExtendedTcpTable(buffer, byref(size), False, 2, 3, 0)
        if code == 0:
            break
        if code != _ERROR_INSUFFICIENT_BUFFER:
            raise _failed("GetExtendedTcpTable", code)
    (count,) = struct.unpack_from("<I", buffer.raw)
    rows = []
    for index in range(count):
        # MIB_TCPROW_OWNER_PID: state, local address, local port, remote address
        # and port, owning PID; the ports are in network order.
        _state, address, port, _remote, _remote_port, pid = struct.unpack_from(
            "<6I", buffer.raw, 4 + 24 * index
        )
        rows.append(
            (socket.inet_ntoa(struct.pack("<I", address)), socket.ntohs(port & 0xFFFF), pid)
        )
    return rows
