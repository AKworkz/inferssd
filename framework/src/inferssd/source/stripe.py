# SPDX-License-Identifier: Apache-2.0
"""Spread the routed experts of an alias index over several drives, to read them in parallel.

The SSD executor opens each distinct ``source_path`` of an alias index as its own direct-I/O
source, and its read ring keeps reads of different files in flight together, each on its own
drive. This tool keeps the experts placed on the first drive where the index has them (the GGUF
shards) and copies the others into one pack file per further drive: each row's payload at a
4 KiB-aligned offset, padded with zeros to the next 4 KiB, and read back through the operating
system and compared by SHA-256 after the pack is written. The output is an alias index with the
same rows, each naming where its payload now is; the engine takes it as ``expert.source``.

The unit of placement is one role of one expert, ``3 * expert + role`` (gate, up, down), the same
in every layer. In order, the k-th unit goes to drive ``k mod n``: consecutive roles, an expert's
and then the next one's, lie on consecutive drives, so the reads of the experts a layer misses run
on several drives at once. Given weights (for example each drive's read rate), each drive instead
gets a share of the units in proportion to its weight, interleaved over them. The placement
balances the number of units, not their bytes or how often the router selects them.

Each run creates new packs, named with a fresh generation, and publishes the index last by
renaming it into place: an index written earlier keeps naming its own packs, which stay until they
are removed. The packs and the index are flushed before the index is published; on POSIX the
packs' directories too, and the index's directory once it is renamed into place.
"""

from __future__ import annotations

import argparse
import csv
import errno
import hashlib
import io
import json
import math
import os
import secrets
import shutil
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from inferssd.source.gguf_alias import ALIGNMENT, EXPERT_RE

ROLES = {"gate": 0, "up": 1, "down": 2}
REQUIRED = ("tensor", "expert_idx", "nbytes", "source_path", "offset")
ZEROS = bytes(ALIGNMENT)
# The native alias reader keeps a source path in 512 bytes (store/catalog.h) and reads a row,
# newline included, into a 4096-byte line buffer (store/parser.cpp).
MAX_PATH_BYTES = 511
MAX_LINE_BYTES = 4095


def padded(nbytes: int) -> int:
    return (nbytes + ALIGNMENT - 1) // ALIGNMENT * ALIGNMENT


def unit(row: dict[str, str]) -> int:
    """A row's unit of placement: one role of one expert, the same in every layer."""
    match = EXPERT_RE.fullmatch(row["tensor"])
    if match is None:
        raise ValueError(f"not a routed expert tensor: {row['tensor']}")
    return 3 * int(row["expert_idx"]) + ROLES[match.group("kind")]


def assign(units: Sequence[int], weights: Sequence[float]) -> dict[int, int]:
    """The drive of each distinct unit. With equal weights the k-th unit in order goes to drive
    k mod n. Otherwise each drive gets as many units as its weight's share (largest remainder),
    interleaved over the units in order."""
    shares = [float(weight) for weight in weights]
    if not shares or not all(math.isfinite(share) and share >= 0 for share in shares):
        raise ValueError("weights must be finite and non-negative")
    top = max(shares)
    if not top > 0:
        raise ValueError("at least one weight must be positive")
    shares = [share / top for share in shares]
    ordered = sorted(set(units))
    if len(set(shares)) == 1:
        return {u: k % len(shares) for k, u in enumerate(ordered)}
    total = sum(shares)
    quota = [len(ordered) * share / total for share in shares]
    counts = [int(part) for part in quota]
    by_remainder = sorted(range(len(shares)), key=lambda i: (counts[i] - quota[i], i))
    for i in by_remainder[: len(ordered) - sum(counts)]:
        counts[i] += 1
    taken = [0] * len(shares)
    owner = {}
    for u in ordered:
        drive = min(
            (i for i in range(len(shares)) if taken[i] < counts[i]),
            key=lambda i: ((taken[i] + 1) / counts[i], i),
        )
        owner[u] = drive
        taken[drive] += 1
    return owner


def read_index(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        fields = list(reader.fieldnames or [])
        missing = [name for name in REQUIRED if name not in fields]
        if missing:
            raise ValueError(f"{path}: alias index lacks columns {', '.join(missing)}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"{path}: alias index has no rows")
    keys = set()
    for row in rows:
        # one row per tensor and expert, even if nbytes differs (the engine's key also has nbytes)
        key = (row["tensor"], int(row["expert_idx"]))
        if key in keys:
            raise ValueError(f"{path}: duplicate row {key[0]}:{key[1]}")
        if int(row["nbytes"]) <= 0 or int(row["offset"]) < 0:
            raise ValueError(f"{path}: invalid extent in row {key[0]}:{key[1]}")
        keys.add(key)
    return fields, rows


def check_path(path: str) -> None:
    """A source path the native alias reader can hold: absolute, one line, at most 511 bytes."""
    if not os.path.isabs(path) or any(char in path for char in "\t\r\n"):
        raise ValueError(f"source path must be absolute and on one line: {path!r}")
    if len(os.fsencode(path)) > MAX_PATH_BYTES:
        raise ValueError(f"source path longer than {MAX_PATH_BYTES} bytes: {path}")


def line(cells: list[str]) -> str:
    """The index line of ``cells``, refusing one the alias reader cannot hold: a cell with a tab or
    newline, or more than its line buffer."""
    if any(char in cell for cell in cells for char in "\t\r\n"):
        raise ValueError(f"an index cell holds a tab or newline: {cells}")
    text = "\t".join(cells) + "\n"
    if len(text.encode("utf-8")) > MAX_LINE_BYTES:
        raise ValueError(f"an index line is longer than {MAX_LINE_BYTES} bytes: {cells[:2]}")
    return text


def serialized(fields: list[str], row: dict[str, str]) -> str:
    return line([str(row.get(name) or "") for name in fields])


def _canonical(path: Path | str) -> str:
    return os.path.normcase(os.path.realpath(path))


def _identity(info: os.stat_result) -> tuple[int, ...]:
    # as gguf_alias: the fields an open file and its path report alike on every OS; on POSIX also
    # the inode change time, which every write moves
    same = (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns)
    return same + ((info.st_ctime_ns,) if sys.platform != "win32" else ())


def _fsync_directory(path: Path) -> None:
    # POSIX makes a rename durable through its directory; Windows does not open directories so
    if sys.platform != "win32":
        handle = os.open(path, os.O_RDONLY)
        try:
            os.fsync(handle)
        finally:
            os.close(handle)


def _read_exact(source: io.FileIO, view: memoryview, offset: int) -> None:
    source.seek(offset)
    done = 0
    while done < len(view):
        count = source.readinto(view[done:])
        if not count:
            raise ValueError(f"{source.name}: short read at byte {offset + done}")
        done += count


def _write_all(target: io.FileIO, view: memoryview) -> None:
    done = 0
    while done < len(view):
        count = target.write(view[done:])
        if not count:
            raise OSError(f"{target.name}: short write")
        done += count


def _drop_cache(handle: io.FileIO) -> None:
    # On Linux, ask the kernel to drop the pack's cached pages before the check reads them back.
    if sys.platform == "linux":
        try:
            os.posix_fadvise(handle.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
        except OSError as error:
            if error.errno not in (errno.EINVAL, errno.ENOSYS, errno.EOPNOTSUPP):
                raise


def write_pack(rows: list[dict[str, str]], pack: Path, sources: dict[str, io.FileIO]) -> list[int]:
    """Create ``pack`` (it must not exist) and copy each row's payload from its open source into
    it, 4 KiB aligned and zero padded, then read it back and compare it by SHA-256. Returns each
    row's offset in the pack; removes the pack if any step fails. One buffer of the largest row
    serves every copy."""
    largest = max((int(row["nbytes"]) for row in rows), default=0)
    buffer = memoryview(bytearray(padded(largest)))
    offsets, digests = [], []
    target = io.FileIO(pack, "x")
    try:
        with target:
            position = 0
            for row in rows:
                nbytes, extent = int(row["nbytes"]), padded(int(row["nbytes"]))
                _read_exact(sources[row["source_path"]], buffer[:nbytes], int(row["offset"]))
                buffer[nbytes:extent] = ZEROS[: extent - nbytes]
                digests.append(hashlib.sha256(buffer[:nbytes]).digest())
                _write_all(target, buffer[:extent])
                offsets.append(position)
                position += extent
            os.fsync(target.fileno())
        with io.FileIO(pack, "r") as check:
            _drop_cache(check)
            for row, offset, digest in zip(rows, offsets, digests):
                view = buffer[: int(row["nbytes"])]
                _read_exact(check, view, offset)
                if hashlib.sha256(view).digest() != digest:
                    raise RuntimeError(
                        f"{pack}: {row['tensor']}:{row['expert_idx']} reads back changed"
                    )
    except BaseException:
        pack.unlink()
        raise
    return offsets


def _existing(path: Path) -> Path:
    while not path.exists():
        path = path.parent
    return path


def stripe(
    index: Path,
    drives: list[Path],
    out: Path,
    weights: Sequence[float] | None = None,
    summary_json: Path | None = None,
) -> dict[str, Any]:
    """The alias index ``out`` with the rows of ``index`` spread over the index's own drive and
    ``drives``, and one new pack per drive in ``drives`` that receives rows. Every check runs
    before the first file or directory is created. Renaming the index into place commits the run;
    before that a failure removes the packs this run created, after it none is removed. The
    summary, if asked for, is written after the commit."""
    if not drives:
        raise ValueError("name at least one further drive")
    weights = list(weights) if weights is not None else [1.0] * (1 + len(drives))
    if len(weights) != 1 + len(drives):
        raise ValueError(
            f"give {1 + len(drives)} weights: the index's own drive first, then each drive"
        )
    fields, rows = read_index(index)
    owner = assign([unit(row) for row in rows], weights)
    on = [owner[unit(row)] for row in rows]
    targets = [Path(os.path.realpath(drive)) for drive in drives]
    if len({os.path.normcase(str(target)) for target in targets}) != len(targets):
        raise ValueError("a drive is named twice")
    generation = secrets.token_hex(16)
    packs = [target / f"{out.stem}.{generation}.pack" for target in targets]
    mine = [[k for k, d in enumerate(on) if d == drive] for drive in range(1, 1 + len(drives))]
    for pack, placed in zip(packs, mine):
        if placed:
            check_path(str(pack))
    sizes: dict[str, os.stat_result] = {}
    for row in rows:
        path = row["source_path"]
        check_path(path)
        if path not in sizes:
            sizes[path] = os.stat(path)
        if int(row["offset"]) + int(row["nbytes"]) > sizes[path].st_size:
            raise ValueError(f"{path}: row {row['tensor']}:{row['expert_idx']} ends past the file")
    # the outputs go into existing directories, not at or above a drive directory, and may replace
    # the input index, but no file the index or the new packs name
    for output in (out, summary_json):
        if output is None:
            continue
        if not output.parent.is_dir() or output.is_dir():
            raise ValueError(f"{output} must name a file in an existing directory")
        place = _canonical(output)
        for target in targets:
            if _canonical(target) == place or _canonical(target).startswith(place + os.sep):
                raise ValueError(f"{output} is a drive directory or lies above one")
    named = {_canonical(path) for path in sizes} | {_canonical(pack) for pack in packs}
    if _canonical(out) in named:
        raise ValueError(f"{out} is a source of the index or one of its packs")
    if summary_json is not None and _canonical(summary_json) in named | {
        _canonical(out),
        _canonical(index),
    }:
        raise ValueError(f"{summary_json} is the index, its output, a source or a pack")
    placed_rows = [dict(row) for row in rows]
    for pack, placed in zip(packs, mine):
        widest = str(sum(padded(int(rows[k]["nbytes"])) for k in placed))
        for k in placed:
            placed_rows[k].update(source_path=str(pack), offset=widest)
    line(fields)
    for row in placed_rows:
        serialized(fields, row)
    volumes = {info.st_dev for info in sizes.values()}
    need: dict[int, int] = {}
    room: dict[int, int] = {}
    for target, placed in zip(targets, mine):
        anchor = _existing(target)
        if not anchor.is_dir():
            raise ValueError(f"drive {target} is not a directory: {anchor} is a file")
        volume = os.stat(anchor).st_dev
        if volume in volumes:
            print(
                f"warning: {target} shares a volume with an earlier drive; it adds no bandwidth",
                file=sys.stderr,
            )
        volumes.add(volume)
        need[volume] = need.get(volume, 0) + sum(padded(int(rows[k]["nbytes"])) for k in placed)
        room[volume] = shutil.disk_usage(anchor).free
    for volume, wanted in need.items():
        if wanted > room[volume]:
            raise ValueError(f"the packs need {wanted} bytes on a volume with {room[volume]} free")
    sources: dict[str, io.FileIO] = {}
    created: list[Path] = []
    partial = out.with_name(f"{out.name}.{generation}.partial")
    partial_created = committed = False
    try:
        # every source of the index, also those whose rows all stay, open until the index is written
        for path in sizes:
            sources[path] = io.FileIO(path, "r")
            if _identity(os.fstat(sources[path].fileno())) != _identity(sizes[path]):
                raise RuntimeError(f"{path} changed before its experts were copied")
        for target, pack, placed in zip(targets, packs, mine):
            if not placed:
                continue
            missing = []  # the directories this run creates, the deepest first
            ancestor = target
            while not ancestor.exists():
                missing.append(ancestor)
                ancestor = ancestor.parent
            target.mkdir(parents=True, exist_ok=True)
            offsets = write_pack([rows[k] for k in placed], pack, sources)
            created.append(pack)
            for k, offset in zip(placed, offsets):
                moved = placed_rows[k]
                moved["offset"] = str(offset)
                if "aligned" in moved:
                    aligned = offset % ALIGNMENT == 0 and int(moved["nbytes"]) % ALIGNMENT == 0
                    moved["aligned"] = str(int(aligned))
            _fsync_directory(target)
            for directory in missing:
                _fsync_directory(directory.parent)
        with open(partial, "x", encoding="utf-8", newline="") as stream:
            partial_created = True
            stream.write(line(fields))
            for row in placed_rows:
                stream.write(serialized(fields, row))
            stream.flush()
            os.fsync(stream.fileno())
        # the last check of every source, just before the commit
        for path, source in sources.items():
            if _identity(os.fstat(source.fileno())) != _identity(sizes[path]) or _identity(
                os.stat(path)
            ) != _identity(sizes[path]):
                raise RuntimeError(f"{path} changed while the experts were copied")
        os.replace(partial, out)
        committed = True
    except BaseException:
        if not committed:
            if partial_created and partial.exists():
                partial.unlink()
            for pack in created:
                pack.unlink()
        raise
    finally:
        for source in sources.values():
            source.close()
    _fsync_directory(out.parent)
    drives_summary = [
        dict(
            path="",
            rows=sum(1 for d in on if d == 0),
            pack=None,
            bytes=sum(int(row["nbytes"]) for row, d in zip(rows, on) if d == 0),
        )
    ]
    for target, pack, placed in zip(targets, packs, mine):
        drives_summary.append(
            dict(
                path=str(target),
                rows=len(placed),
                pack=str(pack) if placed else None,
                bytes=sum(int(rows[k]["nbytes"]) for k in placed),
            )
        )
    summary = dict(
        index=str(index),
        out=str(out),
        generation=generation,
        weights=weights,
        units=len(owner),
        rows=len(rows),
        checked_rows=sum(len(placed) for placed in mine),
        drives=drives_summary,
    )
    if summary_json is not None:
        written = summary_json.with_name(f"{summary_json.name}.{generation}.partial")
        stream = open(written, "x", encoding="utf-8")
        try:
            with stream:
                stream.write(json.dumps(summary, indent=2, sort_keys=True) + "\n")
            os.replace(written, summary_json)
        finally:
            if written.exists():
                written.unlink()
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--index", type=Path, required=True, help="alias index (gguf_alias --out)")
    parser.add_argument(
        "--drive",
        type=Path,
        action="append",
        required=True,
        help="directory on a further drive for its pack; repeat for more drives",
    )
    parser.add_argument(
        "--weights", help="comma-separated: the index's own drive first, then each drive"
    )
    parser.add_argument("--out", type=Path, required=True, help="the striped alias index")
    parser.add_argument("--summary-json", type=Path)
    args = parser.parse_args(argv)
    weights = [float(value) for value in args.weights.split(",")] if args.weights else None
    summary = stripe(args.index, args.drive, args.out, weights, args.summary_json)
    for drive in summary["drives"]:
        print(f"drive={drive['path'] or '(index)'} rows={drive['rows']} bytes={drive['bytes']}")
    print(f"out={summary['out']}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1)
