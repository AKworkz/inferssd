"""Versioned native observations read from a run's native-events.jsonl.

The benchmark CI reads them to record what each run did. Only native-events.jsonl
is read; nothing is inferred from human-readable logging.
"""

from __future__ import annotations

import os
from collections import defaultdict
from typing import Any, BinaryIO, Optional, Protocol

from inferssd.telemetry.files import LocalArmFiles
from inferssd.telemetry.jsonl import Event, read_events
from inferssd.telemetry.types import ArmFiles, ObservedArm, Record, TokenKey, TraceLayer


class FileStatus(Protocol):
    @property
    def st_size(self) -> int: ...

    @property
    def st_mtime_ns(self) -> int: ...


class ObservationFiles(ArmFiles, Protocol):
    def exists(self, path: str) -> bool: ...

    def stat(self, path: str) -> FileStatus: ...

    def open_binary(self, path: str) -> BinaryIO: ...


class LocalObservations(LocalArmFiles):
    """ObservationFiles on this machine's file system."""

    def exists(self, path: str) -> bool:
        return os.path.exists(path)

    def stat(self, path: str) -> os.stat_result:
        return os.stat(path)

    def open_binary(self, path: str) -> BinaryIO:
        return open(path, "rb")


def number(payload: Record, key: str, *, integer: bool = False) -> Any:
    value = payload.get(key)
    kinds = (int,) if integer else (int, float)
    if value is None or type(value) not in kinds or value < 0:
        raise ValueError("invalid native observation field " + key)
    return value


class NativeObservations:
    def __init__(self) -> None:
        self.started = False
        self.sequence = 0
        self.trace: defaultdict[TokenKey, dict[int, TraceLayer]] = defaultdict(dict)
        self.prof: defaultdict[TokenKey, list[float]] = defaultdict(
            lambda: [0.0, 0.0, 0.0, 0.0, 0.0]
        )
        self.graphs: list[Record] = []
        self.last: Optional[tuple[int, int, int]] = None
        self.run = 0

    def feed(self, event: Event) -> None:
        if event.sequence != self.sequence:
            raise ValueError("native event sequence mismatch")
        self.sequence += 1
        p = dict(event.payload)
        if event.name == "ssd.controller.begin":
            if self.started or type(p.get("schema")) is not int or p["schema"] != 1:
                raise ValueError("unsupported or duplicate native observation contract")
            self.started = True
            return
        if event.name not in (
            "ssd.place.layer",
            "ssd.prefill.graph",
            "ssd.prefill.layer",
            "ssd.prefill.copy",
        ):
            return
        if not self.started:
            raise ValueError("native engine lacks the observation contract")
        if event.name == "ssd.prefill.graph":
            self.graphs.append(
                {
                    "n_tokens": number(p, "requested_tokens", integer=True),
                    "ubatch": number(p, "configured_max", integer=True),
                    "cpu_s": 0.0,
                    "sink_bytes": 0.0,
                    "graph_bytes": 0.0,
                    "graph_read_us": 0.0,
                    "graph_passes": 0,
                    "graph_tokens": 0,
                }
            )
        elif event.name == "ssd.prefill.layer":
            read_bytes = number(p, "read_bytes", integer=True)
            if self.graphs:
                self.graphs[-1]["sink_bytes"] += read_bytes
        elif event.name == "ssd.prefill.copy":
            values = {
                key: number(p, key, integer=True)
                for key in ("bytes", "read_us", "passes", "tokens")
            }
            if self.graphs:
                for target, source in (
                    ("graph_bytes", "bytes"),
                    ("graph_read_us", "read_us"),
                    ("graph_passes", "passes"),
                    ("graph_tokens", "tokens"),
                ):
                    self.graphs[-1][target] += values[source]
        else:
            if p.get("success") is not True or type(p.get("decode")) is not bool:
                raise ValueError("failed or invalid native layer observation")
            compute = number(p, "compute_us")
            if not p["decode"]:
                if self.graphs:
                    self.graphs[-1]["cpu_s"] += compute / 1_000_000.0
                return
            if p.get("has_token") is not True:
                raise ValueError("decode observation has no token identity")
            epoch, token, layer = (
                number(p, key, integer=True) for key in ("epoch", "token", "layer")
            )
            selected, vram, cache = (
                p.get(key) for key in ("selected_experts", "vram_hit", "cache_hit")
            )
            if (
                not isinstance(selected, list)
                or not selected
                or any(type(x) is not int or x < 0 for x in selected)
                or not isinstance(vram, list)
                or not isinstance(cache, list)
                or len(selected) != len(vram)
                or len(selected) != len(cache)
                or any(type(x) is not bool for x in vram + cache)
            ):
                raise ValueError("invalid native route observation")
            if (
                self.last
                and self.last[0] == epoch
                and (token < self.last[1] or (token == self.last[1] and layer <= self.last[2]))
            ):
                self.run += 1
            self.last = epoch, token, layer
            key = ((epoch, self.run), token)
            self.trace[key][layer] = (selected, vram, cache)
            total, wait = number(p, "wall_us"), number(p, "wait_io_us")
            profile = self.prof[key]
            for index, value in enumerate(
                (total, wait, compute, max(0.0, total - compute - wait), 1)
            ):
                profile[index] += value

    def result(self) -> ObservedArm:
        if not self.started:
            raise ValueError("native engine did not publish the observation contract")
        groups: list[list[TokenKey]] = []
        for key in sorted(self.trace):
            if groups and groups[-1][-1][0] == key[0] and key[1] == groups[-1][-1][1] + 1:
                groups[-1].append(key)
            else:
                groups.append([key])
        return {
            "groups": groups,
            "trace": self.trace,
            "prof": self.prof,
            "pf_graphs": [g for g in self.graphs if g["n_tokens"] >= 64],
        }


class NativeObservationSession:
    """Incremental, caller-owned JSONL tails; an incomplete final line waits."""

    def __init__(self, *, files: ObservationFiles) -> None:
        self.files = files
        self.live: dict[str, tuple[NativeObservations, int]] = {}

    def read_arm_cached(self, directory: str, skip: int) -> ObservedArm:
        # skip is applied by the caller when selecting its window.
        path = directory + "/native-events.jsonl"
        parser, offset = self.live.get(path, (NativeObservations(), 0))
        if self.files.stat(path).st_size < offset:
            raise ValueError("native observation file was truncated")
        with self.files.open_binary(path) as stream:
            stream.seek(offset)
            data = stream.read()
        cut = data.rfind(b"\n") + 1
        for event in read_events(data[:cut].decode("utf-8").splitlines(), start=parser.sequence):
            parser.feed(event)
        self.live[path] = parser, offset + cut
        return parser.result()
