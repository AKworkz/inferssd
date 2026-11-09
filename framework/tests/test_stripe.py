# SPDX-License-Identifier: Apache-2.0
"""Striping an alias index over several drives: a synthetic GGUF shard with random payloads."""

from __future__ import annotations

import csv
import json
import os
import shutil
import struct
import sys
import threading
from pathlib import Path

import pytest

from inferssd.source import gguf_alias
from inferssd.source import stripe as m

EXPERTS, LAYERS = 8, 2


def string(value: str) -> bytes:
    raw = value.encode()
    return struct.pack("<Q", len(raw)) + raw


def model(path: Path) -> None:
    """One GGUF with gate/up/down Q8_0 experts of 2 x 32 (68 bytes each, not page multiples) in
    two layers, and random payload bytes."""
    names = [
        f"blk.{layer}.ffn_{kind}_exps.weight"
        for layer in range(LAYERS)
        for kind in ("gate", "up", "down")
    ]
    nbytes = 68 * EXPERTS
    raw = b"GGUF" + struct.pack("<IQQ", 3, len(names), 0)
    for i, name in enumerate(names):
        raw += string(name) + struct.pack("<I3QIQ", 3, 32, 2, EXPERTS, 8, i * nbytes)
    data = (len(raw) + 31) // 32 * 32
    path.write_bytes(raw + bytes(data - len(raw)) + os.urandom(len(names) * nbytes))


def alias(tmp_path: Path) -> tuple[Path, Path]:
    source = tmp_path / "models" / "m.gguf"
    source.parent.mkdir()
    model(source)
    index = tmp_path / "m.tsv"
    argv = ["--model-glob", str(source), "--out", str(index), "--summary-json"]
    assert gguf_alias.main([*argv, str(tmp_path / "m.json")]) == 0
    return source, index


def rows_of(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def payload(row: dict[str, str]) -> bytes:
    with open(row["source_path"], "rb") as stream:
        stream.seek(int(row["offset"]))
        return stream.read(int(row["nbytes"]))


def test_equal_weights_give_the_kth_unit_to_drive_k_mod_n():
    owner = m.assign(range(3 * 256), [1, 1])
    assert [owner[u] for u in range(6)] == [0, 1, 0, 1, 0, 1]
    assert m.assign(range(5), [1, 1, 1]) == {0: 0, 1: 1, 2: 2, 3: 0, 4: 1}
    assert m.assign([9, 0, 5], [2, 2]) == {0: 0, 5: 1, 9: 0}  # by order, not by value
    assert m.assign([7], [1, 1, 1]) == {7: 0}  # more drives than units
    assert m.assign(range(4), [1e308, 1e308]) == {0: 0, 1: 1, 2: 0, 3: 1}


def test_weights_give_each_drive_its_share_interleaved_and_deterministic():
    units = range(3 * 256)
    owner = m.assign(units, [4.36, 3.66])
    second = [u for u in units if owner[u] == 1]
    assert abs(len(second) - 768 * 3.66 / 8.02) <= 1
    assert min(second) < 4 and max(second) > 760  # spread over the units, not one block
    assert owner == m.assign(list(reversed(units)), [4.36, 3.66])
    assert set(m.assign(units, [1, 0]).values()) == {0}
    assert sum(1 for d in m.assign(units, [1e308, 5e307]).values() if d == 1) == 256
    for weights in ([0, 0], [1, -1], [1, float("nan")], [1, float("inf")], []):
        with pytest.raises(ValueError):
            m.assign(units, weights)


def test_stripe_moves_payloads_into_aligned_checked_packs(tmp_path):
    source, index = alias(tmp_path)
    drives = [tmp_path / "ssd2" / "experts", tmp_path / "ssd3" / "experts"]
    out, summary_json = tmp_path / "m.stripe.tsv", tmp_path / "s.json"
    before = rows_of(index)
    argv = ["--index", str(index), "--drive", str(drives[0]), "--drive", str(drives[1])]
    assert m.main([*argv, "--out", str(out), "--summary-json", str(summary_json)]) == 0
    summary = json.loads(summary_json.read_text(encoding="utf-8"))
    packs = [d["pack"] for d in summary["drives"][1:]]
    assert [Path(p).name for p in packs] == [f"m.stripe.{summary['generation']}.pack"] * 2
    after = rows_of(out)
    keys = [(r["tensor"], r["expert_idx"], r["nbytes"]) for r in after]
    assert keys == [(r["tensor"], r["expert_idx"], r["nbytes"]) for r in before]
    homes: dict[tuple[int, str], str] = {}
    for old, new in zip(before, after):
        assert payload(new) == payload(old)
        expert, kind = int(new["expert_idx"]), new["kind"]
        drive = (3 * expert + m.ROLES[kind]) % 3
        # the same drive for an expert's role in every layer
        assert homes.setdefault((expert, kind), new["source_path"]) == new["source_path"]
        if drive == 0:
            assert new == old
            continue
        assert new["source_path"] == packs[drive - 1]
        # aligned keeps gguf_alias's meaning: offset and length both whole pages
        assert int(new["offset"]) % 4096 == 0 and new["aligned"] == "0"
        with open(new["source_path"], "rb") as stream:
            stream.seek(int(new["offset"]) + int(new["nbytes"]))
            assert stream.read(4096 - int(new["nbytes"])) == bytes(4096 - int(new["nbytes"]))
    on = [sum(1 for r in after if r["source_path"] == pack) for pack in packs]
    assert on == [LAYERS * EXPERTS, LAYERS * EXPERTS]  # 24 units over 3 drives, 2 layers each
    assert [Path(pack).stat().st_size for pack in packs] == [4096 * n for n in on]
    assert summary["checked_rows"] == sum(on) and summary["rows"] == len(after)
    assert [d["rows"] for d in summary["drives"]] == [len(after) - sum(on), *on]
    assert not list(tmp_path.rglob("*.partial"))
    assert all(r["source_path"] == str(source.resolve()) for r in before)


def test_a_second_run_leaves_the_first_generation_whole(tmp_path):
    _, index = alias(tmp_path)
    drive, out = tmp_path / "ssd2", tmp_path / "m.stripe.tsv"
    first = m.stripe(index, [drive], out)
    shutil.copyfile(out, tmp_path / "first.tsv")
    second = m.stripe(index, [drive], out)
    assert first["drives"][1]["pack"] != second["drives"][1]["pack"]
    assert {Path(d["pack"]).exists() for d in (first["drives"][1], second["drives"][1])} == {True}
    old, new = rows_of(tmp_path / "first.tsv"), rows_of(out)
    assert [payload(r) for r in old] == [payload(r) for r in new]  # each index reads its own packs


def test_refusals_come_before_any_file_or_directory(tmp_path, monkeypatch):
    _, index = alias(tmp_path)
    drive, out = tmp_path / "ssd2", tmp_path / "m.stripe.tsv"
    with pytest.raises(ValueError, match="give 2 weights"):
        m.stripe(index, [drive], out, [1, 1, 1])
    with pytest.raises(ValueError, match="511 bytes"):
        m.stripe(index, [tmp_path / ("x" * 600)], out)
    with pytest.raises(ValueError, match="named twice"):
        m.stripe(index, [drive, tmp_path / "other" / ".." / "ssd2"], out, [1, 1, 1])
    source = Path(rows_of(index)[0]["source_path"])
    with pytest.raises(ValueError, match="a source of the index"):
        m.stripe(index, [drive], source)
    with pytest.raises(ValueError, match="its output"):
        m.stripe(index, [drive], out, summary_json=out)
    with pytest.raises(ValueError, match="existing directory"):
        m.stripe(index, [drive], tmp_path / "missing" / "m.stripe.tsv")
    with pytest.raises(ValueError, match="existing directory"):
        m.stripe(index, [drive], out, summary_json=tmp_path / "missing" / "s.json")
    with pytest.raises(ValueError, match="lies above one"):
        m.stripe(index, [tmp_path / "out.tsv" / "experts"], tmp_path / "out.tsv")
    (tmp_path / "a-file").write_bytes(b"")
    with pytest.raises(ValueError, match="is not a directory"):
        m.stripe(index, [tmp_path / "a-file" / "experts"], out)
    wide = [dict(row, **{"x" * 5000: ""}) for row in rows_of(index)]
    write_rows(tmp_path / "wide.tsv", wide)
    with pytest.raises(ValueError, match="longer than 4095 bytes"):
        m.stripe(tmp_path / "wide.tsv", [drive], out)
    with pytest.raises(ValueError, match="tab or newline"):
        m.serialized(
            ["tensor", "expert_idx", "type"], dict(tensor="t", expert_idx="0", type="a\tb")
        )
    monkeypatch.setattr(m.shutil, "disk_usage", lambda path: type("usage", (), {"free": 4096}))
    with pytest.raises(ValueError, match="free"):
        m.stripe(index, [drive], out)
    # two drives on one volume: each pack fits the free space alone, both together do not
    monkeypatch.setattr(m.shutil, "disk_usage", lambda path: type("usage", (), {"free": 100000}))
    with pytest.raises(ValueError, match="131072 bytes"):
        m.stripe(index, [drive, tmp_path / "ssd3"], out)
    monkeypatch.undo()
    rows = rows_of(index)
    broken = tmp_path / "broken.tsv"
    for bad in (
        [dict(rows[0], offset=str(10**12)), *rows[1:]],
        [*rows, rows[0]],
        [*rows, dict(rows[0], nbytes=str(int(rows[0]["nbytes"]) + 1))],
    ):
        write_rows(broken, bad)
        with pytest.raises(ValueError, match="past the file|duplicate"):
            m.stripe(broken, [drive], out)
    assert not out.exists() and not drive.exists() and not (tmp_path / "ssd3").exists()


def test_a_payload_that_reads_back_changed_leaves_no_pack(tmp_path, monkeypatch):
    _, index = alias(tmp_path)
    drive, out = tmp_path / "ssd2", tmp_path / "m.stripe.tsv"

    def corrupt(handle):
        with open(handle.name, "r+b") as stream:
            stream.seek(1)
            byte = stream.read(1)
            stream.seek(1)
            stream.write(bytes([byte[0] ^ 0xFF]))

    monkeypatch.setattr(m, "_drop_cache", corrupt)
    with pytest.raises(RuntimeError, match="reads back changed"):
        m.stripe(index, [drive], out)
    assert not out.exists() and not list(drive.iterdir())


def test_a_source_that_changes_while_copying_publishes_nothing(tmp_path, monkeypatch):
    source, index = alias(tmp_path)
    drive, out = tmp_path / "ssd2", tmp_path / "m.stripe.tsv"
    write_pack = m.write_pack

    def then_append(rows, pack, sources):
        offsets = write_pack(rows, pack, sources)
        with open(source, "ab") as stream:
            stream.write(b"\0")
        return offsets

    monkeypatch.setattr(m, "write_pack", then_append)
    with pytest.raises(RuntimeError, match="changed while"):
        m.stripe(index, [drive], out)
    assert not out.exists() and not list(drive.iterdir())


def test_a_shard_whose_rows_all_stay_is_held_and_checked_too(tmp_path, monkeypatch):
    source, index = alias(tmp_path)
    kept = source.with_name("kept.gguf")
    shutil.copyfile(source, kept)
    rows = rows_of(index)
    for row in rows:  # with two drives the even units stay: their rows now read the copy
        if m.unit(row) % 2 == 0:
            row["source_path"] = str(kept.resolve())
    two = tmp_path / "two.tsv"
    write_rows(two, rows)
    drive, out = tmp_path / "ssd2", tmp_path / "m.stripe.tsv"
    write_pack = m.write_pack

    def then_touch(rows, pack, sources):
        offsets = write_pack(rows, pack, sources)
        with open(kept, "ab") as stream:
            stream.write(b"\0")
        return offsets

    monkeypatch.setattr(m, "write_pack", then_touch)
    with pytest.raises(RuntimeError, match="kept.gguf changed while"):
        m.stripe(two, [drive], out)
    assert not out.exists() and not list(drive.iterdir())


def test_an_existing_pack_of_the_same_name_is_never_touched(tmp_path, monkeypatch):
    _, index = alias(tmp_path)
    drive, out = tmp_path / "ssd2", tmp_path / "m.stripe.tsv"
    monkeypatch.setattr(m.secrets, "token_hex", lambda n: "f" * 2 * n)
    drive.mkdir()
    other = drive / f"m.stripe.{'f' * 32}.pack"
    other.write_bytes(b"another run's pack")
    with pytest.raises(FileExistsError):
        m.stripe(index, [drive], out)
    assert other.read_bytes() == b"another run's pack" and not out.exists()
    # nor is another run's partial index of the same name
    other.unlink()
    partial = out.with_name(f"{out.name}.{'f' * 32}.partial")
    partial.write_bytes(b"another run's index")
    with pytest.raises(FileExistsError):
        m.stripe(index, [drive], out)
    assert partial.read_bytes() == b"another run's index" and not out.exists()
    assert not list(drive.glob("*.pack"))  # this run's own pack is gone


def test_two_runs_on_one_output_leave_one_whole_generation(tmp_path, monkeypatch):
    _, index = alias(tmp_path)
    out = tmp_path / "m.stripe.tsv"
    results: list[dict] = []
    errors: list[BaseException] = []
    # both runs reach their commit before either renames its index into place
    at_commit = threading.Barrier(2, timeout=60)
    replace = os.replace

    def replace_together(source, target):
        if Path(target) == out:
            at_commit.wait()
        replace(source, target)

    monkeypatch.setattr(m.os, "replace", replace_together)

    def run(drive: Path) -> None:
        try:
            results.append(m.stripe(index, [drive], out))
        except PermissionError as error:  # Windows may refuse a rename onto a file being renamed
            errors.append(error)

    threads = [threading.Thread(target=run, args=(tmp_path / f"ssd{i}",)) for i in (2, 3)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert at_commit.n_waiting == 0 and not at_commit.broken  # both reached the commit together
    assert len(results) == 2 or (sys.platform == "win32" and results and errors)
    published = rows_of(out)
    packs = {row["source_path"] for row in published if row["source_path"].endswith(".pack")}
    assert len(packs) == 1 and packs <= {r["drives"][1]["pack"] for r in results}
    assert [payload(row) for row in published] == [payload(row) for row in rows_of(index)]
    assert not list(tmp_path.glob("*.partial"))


def test_after_the_index_is_published_its_packs_stay(tmp_path, monkeypatch):
    _, index = alias(tmp_path)
    drive, out = tmp_path / "ssd2", tmp_path / "m.stripe.tsv"
    fsync_directory = m._fsync_directory

    def fail_after_commit(path):
        if path == out.parent and out.exists():
            raise OSError("directory flush failed")
        fsync_directory(path)

    monkeypatch.setattr(m, "_fsync_directory", fail_after_commit)
    with pytest.raises(OSError, match="directory flush failed"):
        m.stripe(index, [drive], out)
    published = rows_of(out)
    assert all(Path(row["source_path"]).exists() for row in published)
    assert [payload(row) for row in published] == [payload(row) for row in rows_of(index)]
