#!/usr/bin/env python3
"""Describe one built engine for an installed framework without source discovery."""
import argparse
import json
import subprocess
import sys
from pathlib import Path

# The framework beside this script reads a build's GPU backends, whether or not it is installed.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "framework" / "src"))
from inferssd.launch.resources import cmake_gpu_backends  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--build", required=True, type=Path)
parser.add_argument("--output", required=True, type=Path)
parser.add_argument("--engine", type=Path, default=Path(__file__).resolve().parents[1] / "engine")
args = parser.parse_args()
platform = {"darwin": "darwin", "win32": "windows"}.get(sys.platform, "linux")
executable = ".exe" if platform == "windows" else ""
binary = (args.build / ("bin/inferssd-server" + executable)).resolve(strict=True)
engine = args.engine.resolve(strict=True)
templates = {
    "dsv": (engine / "models/templates/deepseek-ai-DeepSeek-V4.1.jinja").resolve(strict=True),
    "muse": (engine / "models/templates/muse-spark.jinja").resolve(strict=True),
}
# Only inspect the schema; never start inference or probe a device.
subprocess.run([str(binary), "--ssd-schema"], check=True, stdout=subprocess.DEVNULL)
# A source archive has no Git metadata to name the revision.
try:
    revision = subprocess.run(["git", "-C", str(engine), "rev-parse", "HEAD"], check=True,
                              capture_output=True, text=True).stdout.strip() or "unknown"
except (OSError, subprocess.CalledProcessError):
    revision = "unknown"
value = {
    "schema": "inferssd-engine 1", "binary": str(binary), "revision": revision,
    "capabilities": {
        "settings": "inferssd-settings 1", "telemetry": 1, "platform": platform,
        "gpu_backends": list(cmake_gpu_backends(
            (args.build / "CMakeCache.txt").read_text(encoding="utf-8", errors="replace"))),
    },
    "templates": {name: str(path) for name, path in templates.items()},
    "probes": {name: str(path.resolve()) for name, path in {
        "io": args.build / ("bin/inferssd-ioreplay" + executable),
        "memory": args.build / ("bin/inferssd-memprobe" + executable),
        "ssd": args.build / ("bin/inferssd-ssdprobe" + executable),
        "cpu": args.build / ("bin/inferssd-cpuprobe" + executable),
    }.items() if path.is_file()},
}
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(value, indent=2) + "\n")
