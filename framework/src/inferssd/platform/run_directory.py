"""Exclusive execution artifact directory; a completed receipt is never overwritten."""

from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass
from pathlib import Path

RESERVED = (
    "settings.ssd",
    "strategy.json",
    "native-events.jsonl",
    "server.log",
    "server.log.gz",
    "runtime.json",
    "owner.log",
    "ownership.json",
    "server-state.json",
    "guardian.log",
    "guardian-launch.json",
    "STOP",
    "resources-before.json",
    "resources-after.json",
    "memory-samples.jsonl",
    "scope-cleanup.json",
    "guardian-cleanup.json",
    "guardian-quarantine.json",
    "worker-quarantine.json",
)


@dataclass(frozen=True)
class Reservation:
    directory: Path
    token: str
    device: int
    inode: int

    def verify(self) -> None:
        marker = self.directory / ".inferssd-owner"
        info = marker.stat()
        if (info.st_dev, info.st_ino) != (self.device, self.inode) or json.loads(
            marker.read_text(encoding="utf-8")
        ).get("token") != self.token:
            raise RuntimeError("execution directory ownership changed")


def reserve(directory: str) -> Reservation:
    path = Path(directory).resolve()
    path.mkdir(parents=True, exist_ok=True)
    for name in RESERVED:
        if (path / name).exists():
            raise FileExistsError(
                "execution artifact already exists; use a fresh run directory: " + name
            )
    marker = path / ".inferssd-owner"
    token = uuid.uuid4().hex
    fd = os.open(marker, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        info = os.fstat(fd)
        payload = (json.dumps({"token": token, "controller_pid": os.getpid()}) + "\n").encode()
        with os.fdopen(fd, "wb", closefd=False) as stream:
            stream.write(payload)
        # Recheck after winning the atomic claim; no log/spec writes precede it.
        for name in RESERVED:
            if (path / name).exists():
                raise FileExistsError(
                    "execution artifact appeared while claiming directory: " + name
                )
    finally:
        os.close(fd)
    return Reservation(path, token, info.st_dev, info.st_ino)
