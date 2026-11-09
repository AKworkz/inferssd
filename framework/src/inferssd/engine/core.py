"""One explicit owned server handle, preserving the controller-facing contract."""

from __future__ import annotations

import gzip
import json
import os
import shutil
import socket
import time
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from typing import Optional, Protocol, TextIO

from inferssd.launch.types import Resolution
from inferssd.platform.run_directory import Reservation, reserve
from inferssd.telemetry.types import Record

TRACE_ENV = "GGML_KIMI_CPU_SINK_TRACE=1 GGML_KIMI_CPU_SINK_PROFILE=1"


class Owner(Protocol):
    def poll(self) -> Optional[int]: ...
    def confirm_listener(self, port: int) -> None: ...
    def stop(self) -> None: ...


class Effects(Protocol):
    def makedirs(self, path: str) -> None: ...
    def open_log(self, path: str) -> TextIO: ...
    def clock(self) -> float: ...
    def sleep(self, seconds: float) -> None: ...
    def health(self, port: int) -> bytes: ...
    def archive(self, directory: str) -> None: ...
    def port_busy(self, port: int) -> bool: ...


class NativeEffects:
    def __init__(self, address: str) -> None:
        """ADDRESS is where this machine reaches the owned server."""
        self.address = address
        self.reservation: Optional[Reservation] = None

    def makedirs(self, path: str) -> None:
        self.reservation = reserve(path)

    def open_log(self, path: str) -> TextIO:
        return open(path, "x", encoding="utf-8")

    def clock(self) -> float:
        return time.time()

    def sleep(self, seconds: float) -> None:
        time.sleep(seconds)

    def health(self, port: int) -> bytes:
        with urllib.request.urlopen(f"http://{self.address}:{port}/health", timeout=2) as response:
            value: bytes = response.read()
            return value

    def archive(self, directory: str) -> None:
        with (
            open(f"{directory}/server.log", "rb") as source,
            gzip.open(f"{directory}/server.log.gz", "wb", 1) as target,
        ):
            shutil.copyfileobj(source, target)
        os.remove(f"{directory}/server.log")

    def port_busy(self, port: int) -> bool:
        with socket.socket() as connection:
            return connection.connect_ex((self.address, port)) == 0


def post(port: int, path: str, body: Record, timeout: float = 7200) -> object:
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}",
        json.dumps(body, ensure_ascii=False).encode(),
        {"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        value: object = json.loads(response.read())
        return value


def environment_for(environment: Mapping[str, str], port: int, envs: str) -> dict[str, str]:
    value = dict(environment, LLAMA_SERVER_PORT=str(port), PORT=str(port))
    knobs = (environment.get("TRACE_ENV", TRACE_ENV) + " " + envs).split()
    for pair in knobs:
        key, item = pair.split("=", 1)
        value[key] = item
    if environment.get("KNOBS_VIA", "env") == "extra_env":
        value["EXTRA_ENV"] = " ".join(knobs)
    return value


class Engine:
    """Start/health/stop used by Controller, with an explicit ownership implementation.

    Resource monitoring lives in the separately supervised worker. Startup
    exceptions clean only the owned server. A foreign listener is never killed.
    """

    def __init__(
        self,
        directory: str,
        port: int,
        tier: int,
        cpus: str,
        envs: str,
        args: Sequence[str],
        *,
        environment: Mapping[str, str],
        effects: Effects,
        resolve: Callable[[Mapping[str, str], Sequence[str]], Resolution],
        own: Callable[[Resolution, str], Owner],
        prepare_environment: Callable[
            [Mapping[str, str], int, str], dict[str, str]
        ] = environment_for,
        check_port: bool = False,
    ) -> None:
        self.dir, self.port, self.tier, self.cpus, self.envs = directory, port, tier, cpus, envs
        self._effects = effects
        self.stopping, self.ok = False, False
        if check_port and effects.port_busy(port):
            raise ValueError("owned listener port is already in use")
        effects.makedirs(directory)
        env = prepare_environment(environment, port, envs)
        beginning = effects.clock()
        self.log = effects.open_log(f"{directory}/server.log")
        self._owner: Optional[Owner] = None
        try:
            resolution = resolve(env, args)
            self._owner = own(resolution, directory)
            for _ in range(900):
                try:
                    if b'"ok"' in effects.health(port):
                        self._owner.confirm_listener(port)
                        self.ok = True
                        break
                except OSError:
                    pass
                if self._owner.poll() is not None:
                    break
                effects.sleep(1)
            self.start_s = effects.clock() - beginning
        except BaseException:
            # The original implementation left an owned process/log behind on these
            # failures. Cleanup is an intentional safety boundary, not event equivalence.
            try:
                if self._owner is not None:
                    self._owner.stop()
            finally:
                self.log.close()
            raise

    def alive(self) -> bool:
        return self._owner is not None and self._owner.poll() is None

    def stop(self) -> None:
        self.stopping = True
        try:
            if self._owner is not None:
                self._owner.stop()
        finally:
            self.log.close()
        self._effects.archive(self.dir)
        for _ in range(60):
            if not self._effects.port_busy(self.port):
                return
            self._effects.sleep(1)
        raise RuntimeError(
            "port remains occupied after owned cleanup; no unrelated process was killed"
        )
