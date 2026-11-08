# SPDX-License-Identifier: Apache-2.0
"""The Win32 calls themselves, on a real Windows host; other platforms skip this file."""

from __future__ import annotations

import os
import socket
import subprocess
import sys
from pathlib import Path

import pytest

from inferssd.machine.identity import NativeHost, drive_info, live_identity
from inferssd.platform.policy import lock_held

pytestmark = pytest.mark.skipif(sys.platform != "win32", reason="needs Windows")

if sys.platform == "win32":
    from inferssd.platform.windows import lease, process, system, win32


def test_profile_reads_this_host():
    layer = system.Win32()
    found = layer.processors()
    assert found.logical >= found.group0 >= 1 and found.groups >= 1
    assert all(row.index < found.group0 for row in found.sets)
    assert set(layer.process_cpus()) <= set(range(found.group0))
    memory = layer.memory()
    assert 0 < memory.available_bytes <= memory.total_bytes
    assert Path(layer.volume(__file__).point).exists()
    identity = live_identity(NativeHost())
    assert identity["hard"]["platform"] == "windows"
    assert identity["hard"]["ram_bytes"] == memory.total_bytes
    assert drive_info(NativeHost(), __file__).key.startswith("windows:")


def test_process_lookup_does_not_signal_a_live_process():
    child = subprocess.Popen([sys.executable, "-I", "-c", "import time; time.sleep(120)"])
    try:
        assert win32.process_running(child.pid)
        assert child.poll() is None
    finally:
        child.terminate()
        child.wait(timeout=30)
    assert not win32.process_running(child.pid)


def test_the_device_lock_is_exclusive_until_its_last_handle_closes(tmp_path):
    path = str(tmp_path / "device.lock")
    first = lease.acquire_lock(path)
    with pytest.raises(BlockingIOError, match="held by another"):
        lease.acquire_lock(path)
    # A duplicate is the same open: closing the original leaves the lock held.
    copy = lease.borrow_lock(first, path)
    os.close(first)
    with pytest.raises(BlockingIOError):
        lease.acquire_lock(path)
    os.close(copy)
    os.close(lease.acquire_lock(path))
    assert lock_held(path).errno is not None


def test_a_closed_writer_breaks_the_watched_pipe():
    writer, reader = lease.link()
    watched = lease.PipeWatch(os.dup(reader.fileno()))
    reader.close()
    with pytest.raises(BlockingIOError):
        watched.recv(1, 0)
    writer.close()
    assert watched.recv(1, 0) == b""
    watched.close()


def test_an_owned_child_dies_with_its_job_and_states_its_memory():
    argv = [sys.executable, "-I", "-c", "import time; time.sleep(120)"]
    child = process.spawn_owned(
        argv,
        env=dict(os.environ),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        stdin=subprocess.DEVNULL,
        start_new_session=True,
    )
    assert isinstance(child, process.Group)
    assert process.group_exists(child)
    assert child.pid in win32.job_processes(child.job)
    assert win32.job_memory(child.job)["kill_on_close"] is True
    assert win32.process_memory(child.pid)["working_set_bytes"] > 0
    win32.close(child.job)
    # Ended by the job's last handle closing, long before its own two minutes.
    child.wait(timeout=30)
    assert child.poll() is not None


def test_discrete_video_memory_reads():
    # Zero without a discrete GPU; the VRAM of each one that has a non-local segment group.
    assert win32.discrete_video_memory() >= 0


def test_a_job_memory_limit_reads_back():
    job = win32.create_job(kill_on_close=False, memory_limit=1 << 30)
    try:
        assert win32.job_memory(job)["job_memory_limit"] == 1 << 30
        assert win32.job_active(job) == 0
    finally:
        win32.close(job)


def test_this_process_owns_its_listener():
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        port = listener.getsockname()[1]
        assert ("127.0.0.1", port, os.getpid()) in win32.tcp_listeners()


def test_system_memory_and_paging_file_counters():
    memory = win32.system_memory()
    assert 0 < memory["physical_available_bytes"] <= memory["physical_total_bytes"]
    assert win32.pagefile_in_use() >= 0
    # The compression store can be protected from OpenProcess; PDH still samples it.
    pid = win32.find_process("Memory Compression")
    if pid is not None:
        assert win32.process_working_set("Memory Compression", pid) >= 0
