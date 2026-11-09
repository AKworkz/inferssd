#!/usr/bin/env python3
"""Checks run on the package's Python after developer tools have been removed from PATH."""
from __future__ import annotations

import argparse
import json
import os
import socket
import struct
import subprocess
import time
import urllib.request
from pathlib import Path


def imports(path: Path) -> list[str]:
    """Read normal PE import names without dumpbin, LLVM, or another installed dependency."""
    data = path.read_bytes()
    if data[:2] != b"MZ":
        raise ValueError(f"not a PE binary: {path}")
    pe = struct.unpack_from("<I", data, 0x3C)[0]
    if data[pe:pe + 4] != b"PE\0\0":
        raise ValueError(f"invalid PE header: {path}")
    sections, optional_size = struct.unpack_from("<H12xH", data, pe + 6)
    optional = pe + 24
    magic = struct.unpack_from("<H", data, optional)[0]
    directories = optional + (112 if magic == 0x20B else 96 if magic == 0x10B else 0)
    if directories == optional:
        raise ValueError(f"unknown PE optional header: {path}")
    table = optional + optional_size

    def offset(rva: int) -> int:
        for index in range(sections):
            virtual_size, virtual, raw_size, raw = struct.unpack_from("<IIII", data, table + index * 40 + 8)
            if virtual <= rva < virtual + max(virtual_size, raw_size):
                return raw + rva - virtual
        raise ValueError(f"unmapped PE RVA in {path}")

    rva, _ = struct.unpack_from("<II", data, directories + 8)
    if not rva:
        return []
    cursor, names = offset(rva), []
    while any(data[cursor:cursor + 20]):
        name = struct.unpack_from("<I", data, cursor + 12)[0]
        start = offset(name)
        names.append(data[start:data.index(b"\0", start)].decode("ascii").casefold())
        cursor += 20
    return names


def audit(package: Path) -> dict[str, int]:
    files = [p for p in package.rglob("*") if p.suffix.casefold() in (".exe", ".dll", ".pyd")]
    bundled = {p.name.casefold() for p in files}
    system = Path(os.environ["SystemRoot"]) / "System32"
    count = 0
    for path in files:
        for dependency in imports(path):
            crt = dependency.startswith(("vcruntime", "msvcp", "concrt"))
            api_set = dependency.startswith(("api-ms-win-", "ext-ms-win-"))
            if dependency not in bundled and (crt or not api_set and not (system / dependency).is_file()):
                raise ValueError(f"unbundled dependency {dependency}: {path}")
            count += 1
    return {"pe_files": len(files), "import_dependencies_checked": count}


def owned_runtime(package: Path, directory: Path) -> dict[str, object]:
    from inferssd.engine.owned import OwnedProcess
    from inferssd.launch.types import Resolution
    from inferssd.platform.policy import Policy
    from inferssd.platform.run_directory import reserve

    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    code = (
        "from http.server import BaseHTTPRequestHandler,HTTPServer\n"
        "class H(BaseHTTPRequestHandler):\n"
        " def do_GET(self):\n"
        "  self.send_response(200);self.end_headers();self.wfile.write(b'{}')\n"
        " def log_message(self,*args): pass\n"
        f"HTTPServer(('127.0.0.1',{port}),H).serve_forever()\n"
    )
    env = tuple(os.environ.items())
    value = Resolution((str(package / "python/python.exe"), "-I", "-c", code), env, env, (), (), (), None, 0)
    policy = Policy.read({"platform": "windows", "lock_path": str(directory.parent / "runtime.lock"),
                          "stop_path": str(directory / "STOP"), "memory_budget_bytes": 512 << 20,
                          "minimum_memlock_bytes": 0, "cleanup_seconds": 15})
    owner = OwnedProcess(value, str(directory), policy, reservation=reserve(str(directory)))
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        until = time.monotonic() + 30
        while True:
            if owner.poll() is not None:
                raise RuntimeError("packaged ownership worker exited before readiness")
            try:
                with opener.open(f"http://127.0.0.1:{port}/health", timeout=1) as response:
                    if response.status == 200:
                        owner.confirm_listener(port)
                        break
            except OSError:
                if time.monotonic() >= until:
                    raise TimeoutError("packaged owned runtime did not become ready")
                time.sleep(.1)
    finally:
        owner.stop()
    receipt = json.loads((directory / "ownership.json").read_text(encoding="utf-8"))
    if receipt.get("status") != "passed":
        raise RuntimeError("packaged runtime cleanup failed")
    return {"ownership": receipt["status"], "server": receipt["server"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package", type=Path)
    parser.add_argument("work", type=Path)
    args = parser.parse_args()
    package = args.package.resolve()
    import hf_xet
    import httpx
    import inferssd
    import yaml
    from inferssd.launch.ram import admission
    from inferssd.machine.probes import parse as parse_probe

    for module in (inferssd, hf_xet, httpx, yaml):
        if not Path(module.__file__).resolve().is_relative_to(package):
            raise ValueError(f"module escaped package: {module.__name__}")
    resources = json.loads((package / "engine-resources.json").read_text(encoding="utf-8"))
    for relative in [resources["binary"], *resources["templates"].values(), *resources["probes"].values()]:
        path = package / relative
        if Path(relative).is_absolute() or not path.resolve().is_relative_to(package) or not path.is_file():
            raise ValueError("nonrelocatable engine resource: " + relative)
    command = str(package / "bin/inferssd.exe")
    version = json.loads(subprocess.check_output([command, "--version", "--json"]))
    assert {"gemma4", "qwen4exp"} <= set(version["capabilities"]["model_profiles"])
    subprocess.run([command, "cli", "--help"], check=True, stdout=subprocess.DEVNULL)
    subprocess.run([command, "serve", "--help"], check=True, stdout=subprocess.DEVNULL)
    argument = 'space \u4f60\u597d "quoted" C:\\trailing\\'
    rejected = subprocess.run([command, "serve", "-m", "unused.gguf", "--port", argument],
                              check=False, capture_output=True, encoding="utf-8")
    if rejected.returncode != 2 or repr(argument) not in rejected.stderr:
        raise ValueError("the native launcher did not preserve Unicode/quoted arguments or exit status")
    schema = json.loads(subprocess.check_output([str(package / resources["binary"]), "--ssd-schema"]))
    assert schema
    subprocess.run([str(package / resources["binary"]), "--list-devices"], check=True)
    cpu = parse_probe("cpu", subprocess.check_output([str(package / resources["probes"]["cpu"]),
        "--rows", "64", "--cols", "256", "--tokens", "1,2", "--min-seconds", "0", "--threads", "1"]).decode("utf-8"))
    assert cpu["compute"] and not cpu["skipped"]
    # Exercise the packaged Gemma RAM admission contract without downloading model weights.
    fit = admission({"architecture": "gemma4", "model_bytes": 1024, "tensor_bytes": 1024,
                     "embedding_bytes": 512, "kv_layers": [[0, 1024]]},
                    context=512, batch=512, gpu=False, available=8 << 30, budget=4 << 30, headroom=0)
    receipt = {"version": version, "binary_audit": audit(package), "gemma_ram_admission": fit,
               "owned_runtime": owned_runtime(package, args.work / "owned run")}
    (args.work / "check-receipt.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(json.dumps({"package_check": "passed", "label": version["label"], **receipt["binary_audit"]}))


if __name__ == "__main__":
    main()
