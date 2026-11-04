# SPDX-License-Identifier: Apache-2.0
"""Machine identity and drive keys from fake sysfs/procfs trees and command replies."""

from __future__ import annotations

import contextlib
import hashlib
import os
import plistlib
import subprocess
from pathlib import Path

import pytest

from inferssd.machine.backend import machine_backend
from inferssd.machine.identity import (
    Command,
    Drive,
    GPUQueryFailed,
    NativeHost,
    UnsupportedPlatform,
    build_backend,
    cpu_brand,
    drive_info,
    drive_key,
    is_test_file,
    live_identity,
    owner_of_test_file,
    same_hardware,
    with_saved_gpus,
)

SYSTEM = ("/sys/", "/proc/", "/etc/", "/var/lib/dbus/", "/dev/")
NVME = "pci0000:00/0000:00:01.1/0000:01:00.0/nvme/nvme0"
SMI = (
    "nvidia-smi",
    "--query-gpu=uuid,name,memory.total,driver_version",
    "--format=csv,noheader,nounits",
)
AGX = ("ioreg", "-rd1", "-c", "AGXAccelerator")
AGX_LISTING = """\
+-o AGXAccelerator  <class AGXAccelerator, id 0x100000a2f, registered, matched, active>
    {
      "model" = "Apple M5"
      "gpu-core-count" = 10
      "vendor-id" = <6b100000>
    }
"""


class FakeHost(NativeHost):
    """System paths resolve under ROOT; commands and devices come from tables."""

    def __init__(
        self, root, system="linux", arch="x86_64", commands=None, devices=None, free=1 << 40
    ):
        self.root, self.system, self.arch = Path(root), system, arch
        self.commands = dict(commands or {})
        self.devices = dict(devices or {})
        self.free = free
        self.calls = []
        self.test_files = []  # (path, size, existed while in use)

    def inside(self, path):
        return path.startswith(SYSTEM) or path in ("/sys", "/proc")

    def mapped(self, path):
        return str(self.root / path.lstrip("/")) if self.inside(path) else path

    def platform(self):
        return self.system

    def machine(self):
        return self.arch

    def text(self, path):
        return super().text(self.mapped(path))

    def exists(self, path):
        return super().exists(self.mapped(path))

    def listdir(self, path):
        return super().listdir(self.mapped(path))

    def realpath(self, path):
        if not self.inside(path):
            return super().realpath(path)
        real = os.path.realpath(self.mapped(path))
        root = os.path.realpath(self.root)
        return "/" + os.path.relpath(real, root)

    def command(self, argv):
        self.calls.append(tuple(argv))
        return self.commands.get(tuple(argv), Command(127, "", argv[0] + ": not found"))

    def which(self, name):
        return any(argv[0] == name for argv in self.commands)

    def device(self, path):
        for prefix, value in sorted(self.devices.items(), key=lambda item: -len(item[0])):
            if path == prefix or path.startswith(prefix.rstrip("/") + "/"):
                return value
        raise FileNotFoundError(path)

    def page_bytes(self):
        return 16384 if self.system == "darwin" else 4096

    def free_bytes(self, path):
        return self.free

    @contextlib.contextmanager
    def test_file(self, directory, size):
        # Sparse, so a test writes nothing; the real host fills it with random bytes.
        path = os.path.join(directory, f".inferssd-drive-test-{os.getpid()}-fake")
        with open(path, "wb") as stream:
            stream.truncate(size)
        try:
            yield path
        finally:
            self.test_files.append((path, size, os.path.exists(path)))
            os.unlink(path)


def write(root, path, content):
    target = Path(root) / path.lstrip("/")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content)


def link(root, path, target):
    source = Path(root) / path.lstrip("/")
    source.parent.mkdir(parents=True, exist_ok=True)
    source.symlink_to(target)


def ok(stdout):
    return Command(0, stdout)


def linux_tree(root, *, capacities=(1024,) * 4, max_khz=None, smt=False, gpu=True):
    """A Linux machine: identity files, CPUs, one NVMe drive and the uname/nvidia-smi replies."""
    count = len(capacities)
    write(root, "/etc/machine-id", "0123456789abcdef0123456789abcdef\n")
    write(root, "/etc/os-release", 'NAME="Ubuntu"\nPRETTY_NAME="Ubuntu 24.04.1 LTS"\n')
    write(root, "/sys/devices/virtual/dmi/id/product_name", "Workstation X\n")
    write(
        root,
        "/proc/cpuinfo",
        "".join(f"processor\t: {cpu}\nmodel name\t: Example CPU 9000\n\n" for cpu in range(count)),
    )
    write(root, "/proc/meminfo", "MemTotal:       32768000 kB\nMemAvailable:   1000 kB\n")
    write(root, "/sys/devices/system/cpu/online", f"0-{count - 1}\n")
    for cpu, capacity in enumerate(capacities):
        base = f"/sys/devices/system/cpu/cpu{cpu}"
        if capacity is not None:
            write(root, base + "/cpu_capacity", f"{capacity}\n")
        if max_khz is not None and max_khz[cpu] is not None:
            write(root, base + "/cpufreq/cpuinfo_max_freq", f"{max_khz[cpu]}\n")
        core = cpu % (count // 2) if smt else cpu
        siblings = f"{core},{core + count // 2}" if smt else f"{cpu}"
        write(root, base + "/topology/core_id", f"{core}\n")
        write(root, base + "/topology/physical_package_id", "0\n")
        write(root, base + "/topology/thread_siblings_list", siblings + "\n")
    write(root, f"/sys/devices/{NVME}/model", "Example NVMe 2TB          \n")
    write(root, f"/sys/devices/{NVME}/serial", "  S123SERIAL  \n")
    write(root, f"/sys/devices/{NVME}/nvme0n1/wwid", "eui.0001\n")
    link(root, f"/sys/devices/{NVME}/nvme0n1/device", "../../nvme0")
    for index in (1, 2, 3):
        write(root, f"/sys/devices/{NVME}/nvme0n1/nvme0n1p{index}/partition", f"{index}\n")
        link(root, f"/sys/dev/block/259:{index}", f"../../devices/{NVME}/nvme0n1/nvme0n1p{index}")
        link(
            root,
            f"/sys/class/block/nvme0n1p{index}",
            f"../../devices/{NVME}/nvme0n1/nvme0n1p{index}",
        )
        write(root, f"/dev/nvme0n1p{index}", "")
    write(
        root,
        "/proc/self/mountinfo",
        "22 1 259:2 / / rw,relatime shared:1 - ext4 /dev/nvme0n1p2 rw\n",
    )
    commands = {("uname", "-r"): ok("6.8.0-50-generic\n")}
    if gpu:
        commands[SMI] = ok("GPU-1111, NVIDIA GeForce RTX 5090, 32607, 580.65.06\n")
    return commands


def nvme_key():
    return "linux:nvme:Example NVMe 2TB:" + hashlib.sha256(b"S123SERIAL").hexdigest()


def test_linux_identity(tmp_path):
    host = FakeHost(tmp_path, commands=linux_tree(tmp_path))
    identity = live_identity(host)
    assert identity == {
        "hard": {
            "platform": "linux",
            "arch": "x86_64",
            "machine_id_sha256": hashlib.sha256(b"0123456789abcdef0123456789abcdef").hexdigest(),
            "model": "Workstation X",
            "cpu_brand": "Example CPU 9000",
            "logical_cpus": 4,
            "ram_bytes": 32768000 * 1024,
            "gpus": [
                {
                    "backend": "CUDA",
                    "uuid": "GPU-1111",
                    "name": "NVIDIA GeForce RTX 5090",
                    "vram_bytes": 32607 * 1048576,
                }
            ],
        },
        "soft": {
            "os_release": "Ubuntu 24.04.1 LTS",
            "kernel": "6.8.0-50-generic",
            "gpu_driver": "580.65.06",
            "probe_versions": {
                "ssd": "inferssd-ssdprobe/1",
                "cpu": "inferssd-cpuprobe/1",
                "memory": "inferssd-memprobe/1",
            },
        },
    }


def test_linux_identity_without_gpu_or_machine_id(tmp_path):
    host = FakeHost(tmp_path, commands=linux_tree(tmp_path, gpu=False))
    (tmp_path / "etc/machine-id").unlink()
    identity = live_identity(host)
    assert identity["hard"]["machine_id_sha256"] is None and identity["hard"]["gpus"] == []
    assert identity["soft"]["gpu_driver"] is None
    assert "memory" not in identity["soft"]["probe_versions"]


def test_unified_gpu_reports_no_dedicated_memory(tmp_path):
    commands = linux_tree(tmp_path)
    commands[SMI] = ok("GPU-2222, NVIDIA GB10, [N/A], 580.95.05\n")
    gpus = live_identity(FakeHost(tmp_path, commands=commands))["hard"]["gpus"]
    assert gpus == [{"backend": "CUDA", "uuid": "GPU-2222", "name": "NVIDIA GB10", "vram_bytes": 0}]


def test_failing_nvidia_smi_is_not_an_absent_gpu(tmp_path):
    commands = linux_tree(tmp_path)
    saved = live_identity(FakeHost(tmp_path, commands=commands))
    commands[SMI] = Command(
        18,
        "Failed to initialize NVML: Driver/library version mismatch\nNVML library version: 590\n",
    )
    with pytest.raises(GPUQueryFailed, match="status 18: Failed to initialize NVML") as error:
        live_identity(FakeHost(tmp_path, commands=commands))
    partial = error.value.identity
    assert partial["hard"]["gpus"] == [] and "memory" not in partial["soft"]["probe_versions"]
    assert with_saved_gpus(partial, saved) == saved
    # A driver without devices is a machine without GPUs, not a failure.
    commands[SMI] = Command(6, "No devices were found\n")
    assert live_identity(FakeHost(tmp_path, commands=commands))["hard"]["gpus"] == []


def pci(root, address, kind, vendor, device, vram=None):
    base = f"/sys/bus/pci/devices/{address}/"
    write(root, base + "class", kind + "\n")
    write(root, base + "vendor", vendor + "\n")
    write(root, base + "device", device + "\n")
    if vram is not None:
        write(root, base + "mem_info_vram_total", f"{vram}\n")


def test_linux_lists_the_gpus_a_vulkan_build_can_use(tmp_path):
    commands = linux_tree(tmp_path)
    pci(tmp_path, "0000:00:02.0", "0x030000", "0x8086", "0x7d55")  # an Intel iGPU
    pci(tmp_path, "0000:03:00.0", "0x030000", "0x1002", "0x744c", vram=24 << 30)  # an AMD card
    pci(tmp_path, "0000:01:00.0", "0x030000", "0x10de", "0x2b85")  # nvidia-smi lists it
    pci(tmp_path, "0000:05:00.0", "0x030000", "0x1a03", "0x2000")  # a server's BMC display
    pci(tmp_path, "0000:00:1f.3", "0x040300", "0x8086", "0x51c8")  # Intel audio, not a display
    gpus = live_identity(FakeHost(tmp_path, commands=commands))["hard"]["gpus"]
    assert [(gpu["backend"], gpu["uuid"], gpu["name"], gpu["vram_bytes"]) for gpu in gpus] == [
        ("CUDA", "GPU-1111", "NVIDIA GeForce RTX 5090", 32607 * 1048576),
        ("Vulkan", "pci-0000:00:02.0", "PCI 8086:7d55", 0),
        ("Vulkan", "pci-0000:03:00.0", "PCI 1002:744c", 24 << 30),
    ]


def test_the_engine_builds_for_the_machines_gpus(tmp_path):
    assert build_backend([]) is None
    vulkan = {"backend": "Vulkan", "uuid": "pci-0", "name": "PCI 8086:7d55", "vram_bytes": 0}
    cuda = {"backend": "CUDA", "uuid": "GPU-1", "name": "NVIDIA", "vram_bytes": 1}
    metal = {"backend": "Metal", "uuid": "M5:10", "name": "Apple M5", "vram_bytes": 0}
    assert build_backend([vulkan]) == "Vulkan"
    assert build_backend([vulkan, cuda]) == "CUDA" and build_backend([metal]) == "Metal"
    commands = linux_tree(tmp_path)
    assert machine_backend(FakeHost(tmp_path, commands=commands)) == "CUDA"
    del commands[SMI]
    assert machine_backend(FakeHost(tmp_path, commands=commands)) == "CPU"
    pci(tmp_path, "0000:00:02.0", "0x030000", "0x8086", "0x7d55")
    assert machine_backend(FakeHost(tmp_path, commands=commands)) == "Vulkan"
    # A driver that cannot list its GPUs still has them.
    commands[SMI] = Command(18, "Failed to initialize NVML: Driver/library version mismatch\n")
    assert machine_backend(FakeHost(tmp_path, commands=commands)) == "CUDA"


def test_arm_cpu_brand_lists_implementer_and_parts():
    cpuinfo = "processor\t: 0\nCPU implementer\t: 0x41\nCPU part\t: 0xd87\n\n"
    cpuinfo += "processor\t: 1\nCPU implementer\t: 0x41\nCPU part\t: 0xd81\n"
    assert cpu_brand(cpuinfo) == "Arm implementer 0x41 parts 0xd81,0xd87"


def test_memory_tolerance_and_hard_changes(tmp_path):
    host = FakeHost(tmp_path, commands=linux_tree(tmp_path))
    saved = live_identity(host)
    live = live_identity(host)
    live["hard"]["ram_bytes"] = int(saved["hard"]["ram_bytes"] * 1.009)
    live["soft"]["kernel"] = "7.0"
    assert same_hardware(saved, live) == []
    live["hard"]["ram_bytes"] = int(saved["hard"]["ram_bytes"] * 1.02)
    live["hard"]["cpu_brand"] = "Other"
    live["hard"]["gpus"] = []
    assert same_hardware(saved, live) == ["cpu_brand", "ram_bytes", "gpus"]


def darwin_commands():
    return {
        ("ioreg", "-rd1", "-c", "IOPlatformExpertDevice"): ok(
            '  "IOPlatformUUID" = "0000AAAA-1111-2222-3333-444455556666"\n'
        ),
        ("sysctl", "-n", "hw.model"): ok("Mac17,3\n"),
        ("sysctl", "-n", "machdep.cpu.brand_string"): ok("Apple M5\n"),
        ("sysctl", "-n", "hw.memsize"): ok("17179869184\n"),
        ("sysctl", "-n", "hw.logicalcpu"): ok("10\n"),
        ("sysctl", "-n", "hw.physicalcpu"): ok("10\n"),
        ("sysctl", "-n", "hw.perflevel0.logicalcpu"): ok("4\n"),
        ("sysctl", "-n", "hw.perflevel1.logicalcpu"): ok("6\n"),
        ("sw_vers",): ok(
            "ProductName:\t\tmacOS\nProductVersion:\t\t26.4\nBuildVersion:\t\t25E246\n"
        ),
        ("uname", "-r"): ok("25.4.0\n"),
        AGX: ok(AGX_LISTING),
    }


def test_darwin_identity(tmp_path):
    identity = live_identity(FakeHost(tmp_path, "darwin", "arm64", darwin_commands()))
    assert identity == {
        "hard": {
            "platform": "darwin",
            "arch": "arm64",
            "machine_id_sha256": hashlib.sha256(
                b"0000AAAA-1111-2222-3333-444455556666"
            ).hexdigest(),
            "model": "Mac17,3",
            "cpu_brand": "Apple M5",
            "logical_cpus": 10,
            "ram_bytes": 17179869184,
            "gpus": [
                {
                    "backend": "Metal",
                    "uuid": "Apple M5:10",
                    "name": "Apple M5, 10-core GPU",
                    "vram_bytes": 0,
                }
            ],
        },
        "soft": {
            "os_release": "macOS 26.4 (25E246)",
            "kernel": "25.4.0",
            "gpu_driver": None,
            "probe_versions": {"ssd": "inferssd-ssdprobe/1", "cpu": "inferssd-cpuprobe/1"},
        },
    }


@pytest.mark.parametrize(
    "reply",
    [
        Command(127, "", "ioreg: not found"),
        Command(1, "", "ioreg: AGXAccelerator: no such class"),
        ok(AGX_LISTING.replace('      "gpu-core-count" = 10\n', "")),
        ok(""),
    ],
    ids=["absent", "failed", "no core count", "empty"],
)
def test_darwin_identity_without_a_readable_gpu(tmp_path, reply):
    commands = darwin_commands()
    commands[AGX] = reply
    identity = live_identity(FakeHost(tmp_path, "darwin", "arm64", commands))
    assert identity["hard"]["gpus"] == [] and identity["soft"]["gpu_driver"] is None
    assert "memory" not in identity["soft"]["probe_versions"]


def test_gpu_core_count_tells_apple_variants_apart(tmp_path):
    commands = darwin_commands()
    saved = live_identity(FakeHost(tmp_path, "darwin", "arm64", commands))
    commands[AGX] = ok(AGX_LISTING.replace('"gpu-core-count" = 10', '"gpu-core-count" = 8'))
    live = live_identity(FakeHost(tmp_path, "darwin", "arm64", commands))
    assert same_hardware(saved, live) == ["gpus"]


def test_windows_is_not_supported_yet(tmp_path):
    with pytest.raises(UnsupportedPlatform, match="not supported on windows yet"):
        live_identity(FakeHost(tmp_path, "windows"))


def test_drive_key_nvme_partition(tmp_path):
    linux_tree(tmp_path)
    host = FakeHost(tmp_path, devices={"/models": (259, 2)})
    assert drive_info(host, "/models/a.gguf") == Drive(nvme_key(), "Example NVMe 2TB", "nvme")
    assert drive_key(host, "/models/a.gguf") == nvme_key()


def test_drive_key_device_mapper_over_nvme(tmp_path):
    linux_tree(tmp_path)
    link(tmp_path, "/sys/dev/block/253:0", "../../devices/virtual/block/dm-0")
    link(
        tmp_path,
        "/sys/devices/virtual/block/dm-0/slaves/nvme0n1p3",
        f"../../../../{NVME}/nvme0n1/nvme0n1p3",
    )
    host = FakeHost(tmp_path, devices={"/data": (253, 0)})
    assert drive_key(host, "/data/model.gguf") == nvme_key()


def test_drive_key_md_array_names_every_member(tmp_path):
    linux_tree(tmp_path)
    second = NVME.replace("0000:01:00.0", "0000:02:00.0").replace("nvme0", "nvme1")
    write(tmp_path, f"/sys/devices/{second}/model", "Example NVMe 2TB\n")
    write(tmp_path, f"/sys/devices/{second}/serial", "S999\n")
    link(tmp_path, f"/sys/devices/{second}/nvme1n1/device", "../../nvme1")
    link(tmp_path, "/sys/dev/block/9:0", "../../devices/virtual/block/md0")
    link(tmp_path, "/sys/devices/virtual/block/md0/slaves/nvme0n1", f"../../../../{NVME}/nvme0n1")
    link(tmp_path, "/sys/devices/virtual/block/md0/slaves/nvme1n1", f"../../../../{second}/nvme1n1")
    drive = drive_info(FakeHost(tmp_path, devices={"/raid": (9, 0)}), "/raid/x")
    members = sorted(
        [nvme_key(), "linux:nvme:Example NVMe 2TB:" + hashlib.sha256(b"S999").hexdigest()]
    )
    assert drive == Drive(
        "linux:array:" + "+".join(members), "Example NVMe 2TB + Example NVMe 2TB", "nvme"
    )


def test_disks_without_serial_keep_their_kernel_names(tmp_path):
    for index, name in enumerate(("vda", "vdb")):
        device = f"/sys/devices/pci0000:00/0000:00:0{index + 5}.0/virtio{index + 2}"
        write(tmp_path, f"{device}/block/{name}/{name}1/partition", "1\n")
        link(
            tmp_path,
            f"/sys/dev/block/252:{16 * index + 1}",
            f"../../{device[5:]}/block/{name}/{name}1",
        )
        link(tmp_path, f"{device}/block/{name}/device", "../..")
    host = FakeHost(tmp_path, devices={"/a": (252, 1), "/b": (252, 17)})
    assert drive_info(host, "/a/x") == Drive(
        "linux:unknown:unknown:noserial-vda", "unknown", "unknown"
    )
    assert drive_key(host, "/b/x") == "linux:unknown:unknown:noserial-vdb"


def test_drive_key_overlay_uses_the_filesystem(tmp_path):
    linux_tree(tmp_path)
    data = tmp_path / "data"
    (data / "deep").mkdir(parents=True)
    write(
        tmp_path,
        "/proc/self/mountinfo",
        "22 1 259:2 / / rw - ext4 /dev/nvme0n1p2 rw\n"
        f"300 22 0:45 / {data} rw,relatime - overlay overlay rw,lowerdir=/l,upperdir=/u\n",
    )
    host = FakeHost(tmp_path, devices={str(data): (0, 45)})
    assert drive_info(host, str(data / "deep")) == Drive(
        f"linux:fs:overlay:{data}", "overlay", "unknown"
    )


def test_drive_key_anonymous_device_follows_the_mount_source(tmp_path):
    linux_tree(tmp_path)
    data = tmp_path / "btrfs"
    data.mkdir()
    write(
        tmp_path,
        "/proc/self/mountinfo",
        f"40 1 0:50 /@ {data} rw - btrfs /dev/nvme0n1p3 rw,ssd\n",
    )
    host = FakeHost(tmp_path, devices={str(data): (0, 50)})
    assert drive_key(host, str(data)) == nvme_key()


def test_darwin_drive_key(tmp_path):
    volume = {
        "DeviceIdentifier": "disk3s5",
        "MediaName": "",
        "BusProtocol": "Apple Fabric",
        "Internal": True,
        "VolumeUUID": "d8cb630f-7471-44cb-92b8-31c7a0372333",
        "APFSPhysicalStores": [{"APFSPhysicalStore": "disk0s2"}],
        "ParentWholeDisk": "disk3",
    }
    disk = {
        "DeviceIdentifier": "disk0",
        "MediaName": "APPLE SSD AP0512Z",
        "BusProtocol": "Apple Fabric",
    }
    commands = {
        ("df", "-P", "/opt/models/a.gguf"): ok(
            "Filesystem 512-blocks Used Available Capacity Mounted on\n"
            "/dev/disk3s5  965492904 812117032  95648840    90%    /System/Volumes/Data\n"
        ),
        ("diskutil", "info", "-plist", "/System/Volumes/Data"): ok(plistlib.dumps(volume).decode()),
        ("diskutil", "info", "-plist", "disk0"): ok(plistlib.dumps(disk).decode()),
    }
    drive = drive_info(FakeHost(tmp_path, "darwin", "arm64", commands), "/opt/models/a.gguf")
    digest = hashlib.sha256(b"D8CB630F-7471-44CB-92B8-31C7A0372333").hexdigest()
    assert drive == Drive(f"darwin:apple:APPLE SSD AP0512Z:{digest}", "APPLE SSD AP0512Z", "apple")


def test_darwin_drive_needs_a_mount_point(tmp_path):
    with pytest.raises(RuntimeError, match="mount point of /missing: df: not found"):
        drive_info(FakeHost(tmp_path, "darwin", "arm64", {}), "/missing")


@pytest.mark.parametrize(
    "source,point,mount,kind",
    [
        ("devfs", "/dev", "devfs on /dev (devfs, local, nobrowse)", "devfs"),
        (
            "//user@nas/models",
            "/Volumes/models share",
            "//user@nas/models on /Volumes/models share (smbfs, nodev, nosuid, mounted by user)",
            "smbfs",
        ),
    ],
)
def test_darwin_drive_without_a_disk_uses_the_filesystem(tmp_path, source, point, mount, kind):
    commands = {
        ("df", "-P", point + "/a.gguf"): ok(
            "Filesystem 512-blocks Used Available Capacity Mounted on\n"
            f"{source} 100 100 0 100% {point}\n"
        ),
        ("diskutil", "info", "-plist", point): Command(1, "Could not find disk: " + point + "\n"),
        ("mount",): ok("/dev/disk3s1s1 on / (apfs, sealed, local, read-only, journaled)\n" + mount),
    }
    drive = drive_info(FakeHost(tmp_path, "darwin", "arm64", commands), point + "/a.gguf")
    assert drive == Drive(f"darwin:fs:{kind}:{point}", kind, "unknown")


def test_darwin_drive_reports_unreadable_diskutil_output(tmp_path):
    commands = {
        ("df", "-P", "/x"): ok(
            "Filesystem 512-blocks Used Available Capacity Mounted on\n"
            "/dev/disk3s5 1 1 0 100% /System/Volumes/Data\n"
        ),
        ("diskutil", "info", "-plist", "/System/Volumes/Data"): ok("Volume busy\n"),
    }
    with pytest.raises(RuntimeError, match="printed no property list .*: Volume busy"):
        drive_info(FakeHost(tmp_path, "darwin", "arm64", commands), "/x")


def ended_process():
    child = subprocess.Popen(["true"])
    child.wait()
    return child.pid


def test_native_test_file_is_random_uncompressible_and_removed(tmp_path):
    (tmp_path / "model.gguf").write_bytes(b"kept")
    with NativeHost().test_file(str(tmp_path), (3 << 20) + 5) as path:
        data = Path(path).read_bytes()
        assert len(data) == (3 << 20) + 5 and len(set(data[:65536])) == 256
        assert os.path.basename(path).startswith(f".inferssd-drive-test-{os.getpid()}-")
        assert owner_of_test_file(path) == os.getpid() and is_test_file(path)
        assert sorted(os.listdir(tmp_path)) == sorted([os.path.basename(path), "model.gguf"])
    assert os.listdir(tmp_path) == ["model.gguf"]


def test_only_test_files_of_ended_runs_are_cleared(tmp_path):
    names = [
        f".inferssd-drive-test-{ended_process()}-a",  # a killed run's
        ".inferssd-drive-test-left",  # from before names carried a process
        f".inferssd-drive-test-{os.getpid()}-b",  # a live run's
        "model.gguf",
    ]
    for name in names:
        (tmp_path / name).write_bytes(b"x")
    NativeHost().clear_test_files(str(tmp_path))
    assert sorted(os.listdir(tmp_path)) == sorted(names[2:])
