# SPDX-License-Identifier: Apache-2.0
"""The GPU backend to build the engine with on this machine, for the installers.

``python -m inferssd.machine.backend`` prints CUDA, Metal, Vulkan or CPU. It
follows the machine's GPUs through the table the launch reads
(identity.PCI_BACKENDS, identity.build_backend). An installed nvidia-smi that
fails still means an NVIDIA GPU: its driver is there.
"""

from __future__ import annotations

import sys
from typing import Optional

from inferssd.machine.identity import GPUQueryFailed, Host, NativeHost, build_backend, live_identity


def machine_backend(host: Host) -> str:
    """The GPU backend for HOST's GPUs, or CPU when none of them has one."""
    try:
        gpus = live_identity(host)["hard"]["gpus"]
    except GPUQueryFailed as failure:
        print(f"inferssd: {failure}; building for CUDA", file=sys.stderr)
        return "CUDA"
    return build_backend(gpus) or "CPU"


def main(host: Optional[Host] = None) -> int:
    print(machine_backend(host or NativeHost()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
