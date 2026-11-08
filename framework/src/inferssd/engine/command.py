"""Opt-in native launch command; dry-run never imports or invokes this path."""

from __future__ import annotations

import argparse
import json
import signal
import sys
import time
from collections.abc import Callable
from typing import Optional

from inferssd.engine.core import Engine
from inferssd.engine.runtime import Runtime
from inferssd.launch.live import NativeHost
from inferssd.launch.machine import MachineSource
from inferssd.launch.typed_command import specification
from inferssd.launch.types import LaunchInput, Output
from inferssd.platform.host import STOP_SIGNALS
from inferssd.platform.policy import Policy


def run_native(
    args: argparse.Namespace, inputs: LaunchInput, *, session: Optional[Callable[[], int]] = None
) -> int:
    if args.result is not None:
        raise ValueError("--result is dry-run only; execution writes receipts in --output-dir")
    if args.host_facts is not None:
        raise ValueError("--execute observes the live host; recorded --host-facts are dry-run only")
    if args.runtime_policy is None or args.output_dir is None or args.port is None:
        raise ValueError("--execute requires a runtime policy, a run directory and a port")
    if inputs.mode != "server":
        raise ValueError("owned execution currently supports server mode only")
    data = json.loads(args.runtime_policy.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("runtime policy must be a JSON object")
    policy = Policy.read(data)
    if policy.platform != args.platform:
        raise ValueError("--platform differs from runtime policy")

    def emit(event: Output) -> None:
        stream = sys.stdout if event.channel == "stdout" and session is None else sys.stderr
        stream.write(event.text)
        stream.flush()

    def interrupted(number: int, _frame: object) -> None:
        raise KeyboardInterrupt(f"signal {number}")

    runtime = Runtime(
        inputs,
        policy,
        host=NativeHost(),
        emit=emit,
        configuration=specification(args, strategy=True),
        machine=MachineSource(args.machine, args.reprofile),
    )
    engine: Optional[Engine] = None
    handlers = {number: signal.signal(number, interrupted) for number in STOP_SIGNALS}
    try:
        engine = runtime.engine(
            str(args.output_dir), args.port, 0, "static", "", list(inputs.extra)
        )
        if not engine.ok:
            raise RuntimeError("owned server did not become healthy")
        if session is not None:
            return session()
        while engine.alive():
            time.sleep(0.5)
        return 0
    except KeyboardInterrupt:
        return 130
    finally:
        try:
            if engine is not None:
                engine.stop()
        finally:
            for number, previous in handlers.items():
                signal.signal(number, previous)
