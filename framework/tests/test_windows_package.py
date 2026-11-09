import importlib.util
import struct
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "check_windows_package", ROOT / "scripts/check-package-windows.py"
)
check = importlib.util.module_from_spec(spec)
spec.loader.exec_module(check)


def binary(dependency=None):
    data = bytearray(1024)
    data[:2] = b"MZ"
    struct.pack_into("<I", data, 0x3C, 0x80)
    data[0x80:0x84] = b"PE\0\0"
    struct.pack_into("<H", data, 0x86, 1)
    struct.pack_into("<H", data, 0x94, 240)
    struct.pack_into("<H", data, 0x98, 0x20B)
    struct.pack_into("<IIII", data, 0x188 + 8, 512, 0x1000, 512, 512)
    if dependency:
        struct.pack_into("<II", data, 0x98 + 112 + 8, 0x1000, 40)
        struct.pack_into("<I", data, 512 + 12, 0x1050)
        name = dependency.encode("ascii") + b"\0"
        data[592 : 592 + len(name)] = name
    return data


def test_pe_import_reader_does_not_need_dumpbin(tmp_path):
    path = tmp_path / "engine.exe"
    path.write_bytes(binary("KERNEL32.dll"))
    assert check.imports(path) == ["kernel32.dll"]
    path.write_bytes(b"not a binary")
    with pytest.raises(ValueError, match="not a PE"):
        check.imports(path)


def test_missing_non_system_dependency_fails_package_audit(tmp_path, monkeypatch):
    system = tmp_path / "Windows/System32"
    system.mkdir(parents=True)
    package = tmp_path / "package"
    package.mkdir()
    monkeypatch.setenv("SystemRoot", str(system.parent))
    (package / "engine.exe").write_bytes(binary("missing.dll"))
    with pytest.raises(ValueError, match="unbundled dependency missing.dll"):
        check.audit(package)


def test_visual_cpp_runtime_must_be_bundled_even_on_developer_machine(tmp_path, monkeypatch):
    system = tmp_path / "Windows/System32"
    system.mkdir(parents=True)
    package = tmp_path / "package"
    package.mkdir()
    monkeypatch.setenv("SystemRoot", str(system.parent))
    (system / "vcruntime140.dll").write_bytes(binary())
    (package / "engine.exe").write_bytes(binary("vcruntime140.dll"))
    with pytest.raises(ValueError, match="unbundled dependency vcruntime140.dll"):
        check.audit(package)
    (package / "vcruntime140.dll").write_bytes(binary())
    assert check.audit(package)["pe_files"] == 2
