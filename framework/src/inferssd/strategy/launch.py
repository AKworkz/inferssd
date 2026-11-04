# SPDX-License-Identifier: Apache-2.0
"""The strategy step of a typed launch's resolution.

The resolver asks four facts: the machine characteristics, the model's header
facts, the key of the drive holding the model and the runtime facts. A live
launch reads them on this host; a recorded dry-run takes them from its host
facts. The strategy fills only settings and thread pool fields the user left
unset, and its receipt records what it chose, what the user's values replaced
and what it left out to agree with them.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Optional, cast

from inferssd.configuration.settings import Snapshot
from inferssd.launch.types import State, Threads
from inferssd.machine.characteristics import loads
from inferssd.strategy.derive import derive
from inferssd.strategy.inputs import model_facts_from, runtime_facts_from
from inferssd.strategy.merge import apply, server_arguments


@dataclass(frozen=True)
class Planned:
    """The settings and thread pools to launch with, extra server arguments and the receipt."""

    settings: Snapshot
    threads: Threads
    server_args: tuple[str, ...]
    receipt: dict[str, object]
    # Prefill each prompt in one pass: -b and -ub as large as the context.
    one_pass: bool = False


def plan(
    s: State,
    platform: str,
    memory: int,
    paths: Sequence[str],
    settings: Snapshot,
    threads: Threads,
    gpu_backends: Optional[tuple[str, ...]] = None,
) -> Planned:
    """SETTINGS and the user's THREADS completed by the strategy for this launch.

    MEMORY is the launch's memory budget: it is the strategy's budget and
    bounds the memory available to it. GPU_BACKENDS are the engine's, None
    when unknown. Without characteristics, or on a platform that cannot be
    profiled yet, the launch keeps its inputs and the receipt names why.
    """

    def skipped(reason: str) -> Planned:
        s.emit("inferssd launch: no launch strategy: " + reason + "\n", "stderr")
        return Planned(settings, threads, (), {"skipped": reason})

    if platform == "windows":
        return skipped("machine profiling is not supported on windows yet")
    text = s.value("machine_characteristics")
    if text is None:
        return skipped("no machine characteristics")
    characteristics = loads(cast(str, text))
    model = model_facts_from(s.value("model_facts", *paths))
    key = cast(Optional[str], s.value("model_drive_key", *paths))
    runtime = runtime_facts_from(s.value("runtime_facts"))
    available = runtime.available_bytes
    runtime = replace(
        runtime,
        available_bytes=None if available is None else min(available, memory),
        memory_budget_bytes=memory,
        gpu_backends=gpu_backends,
    )
    try:
        strategy = derive(characteristics, model, runtime, platform, model_drive_key=key)
    except ValueError as error:
        return skipped(str(error))
    applied = apply(strategy, settings, threads)
    arguments = server_arguments(strategy, s.inputs.extra, s.env)
    pairs = strategy.server_args
    yielded = [option for option in pairs[::2] if option not in arguments[::2]]
    kept = list(applied.kept) + ["server " + option for option in yielded]
    left = set(kept) | set(applied.dropped)
    pools = {
        name: getattr(applied.threads, name)
        for name in ("decode", "batch", "cpus", "decode_cpus")
        if getattr(strategy.threads, name) and "threads." + name not in left
    }
    receipt: dict[str, object] = {
        "inputs": {
            "characteristics_sha256": strategy.inputs.characteristics_sha256,
            "model_paths": list(strategy.inputs.model_paths),
            "model_drive": key,
            "platform": strategy.inputs.platform,
        },
        "chosen": {
            "settings": {
                name: {"value": choice.value_text, "reason": choice.reason}
                for name, choice in strategy.settings.items()
                if name not in left
            },
            "threads": {
                name: list(value) if name.endswith("cpus") else value
                for name, value in pools.items()
            },
            "one_pass": strategy.one_pass,
            "server_args": list(arguments),
        },
        "kept": kept,
        "dropped": list(applied.dropped),
        "notes": list(strategy.notes),
    }
    return Planned(applied.settings, applied.threads, arguments, receipt, strategy.one_pass)
