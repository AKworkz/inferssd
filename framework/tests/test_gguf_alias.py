"""Strict Mac alias fixtures contain synthetic headers and sparse zero payloads."""

from __future__ import annotations

import csv
import io
import json
import os
import struct
import subprocess
import sys
from pathlib import Path

import pytest

from inferssd.launch.recorded import generate_alias
from inferssd.source import gguf_alias as m

EXPERT = "blk.0.ffn_up_exps.weight"


def string(value):
    value = value.encode() if isinstance(value, str) else value
    return struct.pack("<Q", len(value)) + value


def metadata(key, kind, value):
    if kind == 8:
        raw = string(value)
    elif kind == 9:
        element, values = value
        raw = struct.pack("<IQ", element, len(values))
        raw += b"".join(string(v) if element == 8 else struct.pack("<I", v) for v in values)
    else:
        raw = struct.pack({2: "<H", 4: "<I", 5: "<i"}[kind], value)
    return string(key) + struct.pack("<I", kind) + raw


def fixture(path, *, tensors=None, kv=(), version=3, payload=131072, align=32):
    # The payload budget is an independent fixture choice, not calculated from
    # the candidate type table or tensor-size implementation.
    tensors = tensors if tensors is not None else [(EXPERT, (32, 2, 256), 8, 0)]
    raw = b"GGUF" + struct.pack("<IQQ", version, len(tensors), len(kv))
    raw += b"".join(metadata(*item) for item in kv)
    for name, shape, kind, offset in tensors:
        raw += string(name) + struct.pack(f"<I{len(shape)}QIQ", len(shape), *shape, kind, offset)
    data = (len(raw) + align - 1) // align * align
    with path.open("wb") as stream:
        stream.write(raw)
        stream.truncate(data + payload)
    return raw, data


@pytest.mark.parametrize("experts", [256, 384])
@pytest.mark.parametrize("version", [2, 3])
def test_complete_valid_rows_summary_and_original_five_columns(tmp_path, experts, version):
    path = tmp_path / "model[1].gguf"
    _, data = fixture(path, version=version, tensors=[(EXPERT, (32, 2, experts), 8, 0)])
    rows, summary = m.build_rows([path], {"up"}, experts)
    assert len(rows) == experts
    assert rows[0] == dict(
        tensor=EXPERT,
        expert_idx=0,
        nbytes=68,
        source_path=str(path),
        offset=data,
        kind="up",
        layer=0,
        type="Q8_0",
        aligned=0,
        shard_index=0,
    )
    assert rows[-1]["offset"] == data + (experts - 1) * 68
    assert summary == dict(
        model_paths=[str(path)],
        model_shards=1,
        expert_tensors=1,
        entries=experts,
        n_experts=experts,
        include_kinds=["up"],
        total_payload_bytes=experts * 68,
        total_payload_gib=experts * 68 / 1024**3,
        unaligned_rows=experts,
        by_kind={"up": dict(tensors=1, entries=experts, bytes=experts * 68)},
        by_type={"Q8_0": dict(tensors=1, entries=experts, bytes=experts * 68)},
    )
    assert m.FIELDS[:5] == ("tensor", "expert_idx", "nbytes", "source_path", "offset")
    assert m.parse_model_paths([__import__("glob").escape(str(path))] * 2) == [path]


@pytest.mark.parametrize(
    "type_id,inner,expected_bytes,type_name",
    [
        (0, 32, 128, "F32"),
        (1, 32, 64, "F16"),
        (8, 32, 34, "Q8_0"),
        (9, 32, 36, "Q8_1"),
        (12, 256, 144, "Q4_K"),
        (13, 256, 176, "Q5_K"),
        (14, 256, 210, "Q6_K"),
        (30, 32, 64, "BF16"),
        (39, 32, 17, "MXFP4"),
    ],
)
def test_native_tensor_sizes(tmp_path, type_id, inner, expected_bytes, type_name):
    path = tmp_path / "model.gguf"
    fixture(path, tensors=[(EXPERT, (inner, 1, 256), type_id, 0)])
    rows, summary = m.build_rows([path], {"up"}, 256)
    assert all(row["nbytes"] == expected_bytes and row["type"] == type_name for row in rows)
    assert summary["total_payload_bytes"] == expected_bytes * 256


@pytest.mark.parametrize("alignment", [32, 64, 4096])
def test_native_ordered_layout_accepts_exact_padding(tmp_path, alignment):
    path = tmp_path / "model.gguf"
    # Three differently sized rows force real padding between tensors.
    offsets = [0, alignment, alignment * 2]
    _, data = fixture(
        path,
        tensors=[(f"dense.{i}", (1,), 0, offset) for i, offset in enumerate(offsets)],
        kv=[("general.alignment", 4, alignment)],
        align=alignment,
    )
    header = m.read_header(path)
    assert [tensor.offset for tensor in header.tensors] == [data + o for o in offsets]


@pytest.mark.parametrize("offsets", [(1,), (32,), (0, 0), (0, 4), (0, 31), (0, 33), (0, 64)])
def test_native_ordered_layout_rejects_nonzero_start_overlap_unaligned_and_gap(tmp_path, offsets):
    path = tmp_path / "bad.gguf"
    fixture(path, tensors=[(f"dense.{i}", (1,), 0, offset) for i, offset in enumerate(offsets)])
    with pytest.raises(ValueError, match="expected native ordered offset"):
        m.read_header(path)


def test_metadata_only_shard_without_data_alignment_padding_is_native_valid(tmp_path):
    a, b = (tmp_path / f"model-{i:05d}-of-00002.gguf" for i in (1, 2))
    raw, data = fixture(
        a,
        tensors=[],
        kv=[("split.no", 2, 0), ("split.count", 2, 2), ("split.tensors.count", 5, 1)],
        payload=0,
    )
    assert len(raw) < data
    a.write_bytes(raw)  # No data and no tail padding, as in the real Muse first shard.
    header = m.read_header(a)
    assert header.tensors == () and header.data_offset == data > a.stat().st_size
    fixture(b, kv=[("split.no", 2, 1)])
    rows, summary = m.build_rows([a, b], {"up"}, 256)
    assert len(rows) == 256 and all(row["shard_index"] == 1 for row in rows)
    assert summary["model_shards"] == 2 and summary["expert_tensors"] == 1


def test_nonempty_shard_without_payload_is_still_rejected(tmp_path):
    path = tmp_path / "bad.gguf"
    raw, _ = fixture(path, tensors=[("dense", (32,), 0, 0)], payload=0)
    with pytest.raises(ValueError, match="payload range exceeds source file"):
        m.read_header(path)
    path.write_bytes(raw)  # Missing alignment padding is not admitted for tensors.
    with pytest.raises(ValueError, match="data region starts beyond the source file"):
        m.read_header(path)


def test_reader_stops_before_payload_and_validates_discarded_strings(tmp_path, monkeypatch):
    path = tmp_path / "model.gguf"
    raw, data = fixture(
        path, kv=[("unused.strings", 9, (8, ["α", "中文"])), ("unused.scalars", 9, (4, [1, 2, 3]))]
    )
    reads = []
    original = m._Reader.read

    def read(reader, count):
        reads.append((reader.offset, count))
        return original(reader, count)

    monkeypatch.setattr(m._Reader, "read", read)
    header = m.read_header(path)
    assert header.data_offset == data
    assert all(offset + count <= len(raw) <= data for offset, count in reads)
    assert not any("numpy" in name for name in m.__dict__)


def test_utf8_validation_crosses_stream_chunks(tmp_path):
    path = tmp_path / "model.gguf"
    fixture(path, kv=[("unused", 8, "a" * ((1 << 20) - 1) + "α")])
    m.read_header(path)
    fixture(path, kv=[("unused", 8, b"a" * ((1 << 20) - 1) + b"\xc3")])
    with pytest.raises(UnicodeDecodeError):
        m.read_header(path)


class ShortReads(io.BytesIO):
    def read(self, count=-1):
        return super().read(min(count, 2))


def test_short_reads_are_completed_or_rejected_without_zero_fill():
    reader = m._Reader(ShortReads(struct.pack("<Q", 7)), 8)
    assert reader.take("Q") == 7
    reader = m._Reader(ShortReads(b"abcd"), 8)
    with pytest.raises(ValueError, match="short GGUF header read"):
        reader.take("Q")
    with pytest.raises(ValueError, match="truncated GGUF header"):
        m._Reader(io.BytesIO(b"x"), 1).read(1 << 64)


@pytest.mark.parametrize(
    "tensors,message",
    [
        ([(EXPERT, (32, 2, 256), 42, 0)], "unsupported native ggml type 42"),
        ([("dense", (32,), 999, 0)], "unsupported native ggml type 999"),
        ([("dense", (), 0, 0)], "invalid dimension count"),
        ([("dense", (1, 1, 1, 1, 1), 0, 0)], "invalid dimension count"),
        ([("dense", (0,), 0, 0)], "invalid dimensions"),
        ([(EXPERT, (31, 2, 256), 8, 0)], "quantized row size"),
        ([("dense", (1 << 62, 4), 0, 0)], "element count overflows"),
        ([("dense", (1 << 61,), 0, 0)], "byte count overflows"),
        ([("dense", (1 << 63,), 0, 0)], "invalid dimensions"),
        ([(EXPERT, (32, 2, 256), 8, (1 << 64) - 1)], "payload range exceeds"),
        ([("dense", (32,), 0, 131070)], "payload range exceeds"),
        ([("dense", (32,), 0, 0), ("dense", (32,), 0, 128)], "duplicated tensor"),
    ],
)
def test_all_tensors_are_strict_even_when_not_selected(tmp_path, tensors, message):
    path = tmp_path / "bad.gguf"
    fixture(path, tensors=tensors)
    with pytest.raises((ValueError, RuntimeError), match=message):
        m.build_rows([path], {"up"}, 256)


@pytest.mark.parametrize(
    "kv,message",
    [
        ([("key", 4, 1), ("key", 4, 2)], "duplicate GGUF metadata"),
        ([("GGUF.version", 4, 3)], "duplicate GGUF metadata"),
        ([("general.alignment", 8, "32")], "Bad type for general.alignment"),
        ([("general.alignment", 4, 0)], "Invalid alignment"),
        ([("general.alignment", 4, 3)], "Invalid alignment"),
        ([("general.alignment", 4, 1 << 31)], "data region starts beyond"),
        ([("split.no", 4, 0)], "Bad type for split.no"),
        ([("discarded", 8, b"\xff")], "utf-8"),
        ([("discarded", 9, (8, [b"ok", b"\xff"]))], "utf-8"),
        ([(b"bad\xff", 4, 1)], "utf-8"),
    ],
)
def test_metadata_types_alignment_duplicates_and_utf8(tmp_path, kv, message):
    path = tmp_path / "bad.gguf"
    fixture(path, kv=kv)
    with pytest.raises((ValueError, RuntimeError, UnicodeDecodeError), match=message):
        m.read_header(path)


@pytest.mark.parametrize("offset", [0, 4, 8, 16, 23, 26, 35, 47, 65, 78])
def test_truncated_header_never_publishes(tmp_path, offset):
    path = tmp_path / "bad.gguf"
    raw, _ = fixture(path)
    path.write_bytes(raw[:offset])
    out, summary = tmp_path / "alias.tsv", tmp_path / "summary.json"
    with pytest.raises((ValueError, RuntimeError)):
        m.main(
            [
                "--model-glob",
                str(path),
                "--n-experts",
                "256",
                "--out",
                str(out),
                "--summary-json",
                str(summary),
            ]
        )
    assert not out.exists() and not summary.exists()


@pytest.mark.parametrize(
    "raw,message",
    [
        (b"BAD!" + struct.pack("<IQQ", 3, 0, 0), "magic"),
        (b"GGUF" + struct.pack("<IQQ", 4, 0, 0), "unsupported GGUF version"),
        (b"GGUF" + struct.pack(">IQQ", 3, 0, 0), "big-endian"),
        (b"GGUF" + struct.pack("<IQQ", 3, 0, 1 << 63), "metadata table"),
        (b"GGUF" + struct.pack("<IQQ", 3, 1 << 63, 0), "tensor table"),
        (
            b"GGUF" + struct.pack("<IQQ", 3, 0, 1) + string("bad") + struct.pack("<I", 99),
            "unknown GGUF metadata type",
        ),
        (
            b"GGUF" + struct.pack("<IQQ", 3, 0, 1) + string("bad") + struct.pack("<IQ", 8, 1 << 63),
            "truncated GGUF header",
        ),
        (
            b"GGUF"
            + struct.pack("<IQQ", 3, 0, 1)
            + string("bad")
            + struct.pack("<IIQ", 9, 4, 1 << 63),
            "truncated GGUF header",
        ),
        (
            b"GGUF" + struct.pack("<IQQ", 3, 0, 1) + string("bad") + struct.pack("<IIQ", 9, 9, 0),
            "unsupported GGUF array",
        ),
    ],
)
def test_malformed_counts_versions_and_arrays_fail_bounded(tmp_path, raw, message):
    path = tmp_path / "bad.gguf"
    path.write_bytes(raw + bytes(128))
    with pytest.raises(ValueError, match=message):
        m.read_header(path)


@pytest.mark.parametrize("shape", [(32, 256), (32, 2, 384), (32, 256, 2)])
def test_selected_expert_geometry_is_not_inferred_or_transposed(tmp_path, shape):
    path = tmp_path / "bad.gguf"
    fixture(path, tensors=[(EXPERT, shape, 8, 0)])
    with pytest.raises(RuntimeError, match="expected 3D tensor with 256 experts"):
        m.build_rows([path], {"up"}, 256)


def test_expert_count_is_read_once_from_the_model(tmp_path):
    path = tmp_path / "model.gguf"
    fixture(
        path,
        tensors=[(EXPERT, (32, 2, 8), 8, 0), ("blk.0.ffn_gate_exps.weight", (32, 2, 8), 8, 544)],
    )
    rows, summary = m.build_rows([path], {"up", "gate"}, None)
    assert len(rows) == 16 and summary["n_experts"] == 8
    with pytest.raises(RuntimeError, match="expected 3D tensor with 6 experts"):
        m.build_rows([path], {"up"}, 6)
    mixed = tmp_path / "mixed.gguf"
    fixture(
        mixed,
        tensors=[(EXPERT, (32, 2, 8), 8, 0), ("blk.0.ffn_gate_exps.weight", (32, 2, 4), 8, 544)],
    )
    with pytest.raises(RuntimeError, match="expected 3D tensor with 8 experts"):
        m.build_rows([mixed], {"up", "gate"}, None)
    fixture(path, tensors=[(EXPERT, (32, 8), 8, 0)])
    with pytest.raises(RuntimeError, match="expected 3D tensor with routed experts"):
        m.build_rows([path], {"up"}, None)


def test_shards_and_optional_split_metadata_must_agree(tmp_path):
    a, b = (tmp_path / f"model-{i:05d}-of-00002.gguf" for i in (1, 2))
    fixture(a, kv=[("split.no", 2, 0), ("split.count", 2, 2), ("split.tensors.count", 5, 2)])
    fixture(
        b,
        tensors=[("blk.1.ffn_up_exps.weight", (32, 2, 256), 8, 0)],
        kv=[("split.no", 2, 1), ("split.count", 2, 2), ("split.tensors.count", 5, 2)],
    )
    rows, summary = m.build_rows([a, b], {"up"}, 256)
    assert len(rows) == 512 and summary["model_shards"] == 2
    with pytest.raises(RuntimeError, match="complete numbered shard set"):
        m.build_rows([a], {"up"}, 256)
    with pytest.raises(RuntimeError, match="first GGUF shard"):
        m.build_rows([b, a], {"up"}, 256)
    fixture(b, kv=[("split.no", 2, 0)])
    with pytest.raises(RuntimeError, match="inconsistent split.no"):
        m.build_rows([a, b], {"up"}, 256)
    fixture(b, kv=[("split.no", 2, 1)])
    with pytest.raises(RuntimeError, match="duplicate expert tensor across shards"):
        m.build_rows([a, b], {"up"}, 256)


def test_native_split_metadata_and_nonexpert_duplicates_are_checked(tmp_path):
    a, b = (tmp_path / f"model-{i:05d}-of-00002.gguf" for i in (1, 2))
    fixture(a)
    fixture(b)
    with pytest.raises(RuntimeError, match="missing GGUF split metadata"):
        m.build_rows([a, b], {"up"}, 256)
    fixture(
        a,
        tensors=[(EXPERT, (32, 2, 256), 8, 0), ("dense", (32,), 0, 17408)],
        kv=[("split.no", 2, 0), ("split.count", 2, 2), ("split.tensors.count", 5, 3)],
    )
    fixture(b, tensors=[("dense", (32,), 0, 0)], kv=[("split.no", 2, 1)])
    with pytest.raises(RuntimeError, match="duplicate tensor across shards: dense"):
        m.build_rows([a, b], {"up"}, 256)


def test_aligned_selection_and_runtime_name_limit(tmp_path):
    path = tmp_path / "model.gguf"
    fixture(
        path,
        tensors=[(EXPERT, (1024, 1, 256), 0, 0)],
        payload=1 << 20,
        kv=[("general.alignment", 4, 4096)],
        align=4096,
    )
    rows, summary = m.build_rows([path], {"up"}, 256, True)
    assert all(row["aligned"] == 1 for row in rows) and summary["unaligned_rows"] == 0
    long_name = "blk." + "1" * 128 + ".ffn_up_exps.weight"
    fixture(path, tensors=[(long_name, (32, 2, 256), 8, 0)])
    with pytest.raises(RuntimeError, match="too long for runtime key"):
        m.build_rows([path], {"up"}, 256)


@pytest.mark.parametrize("name", ["d" * 128, "é" * 64])
def test_all_native_tensor_names_are_byte_bounded_before_reading(tmp_path, monkeypatch, name):
    path = tmp_path / "bad.gguf"
    fixture(path, tensors=[(name, (1,), 0, 0)])
    original = m._Reader.read
    reads = []

    def read(reader, count):
        reads.append(count)
        return original(reader, count)

    monkeypatch.setattr(m._Reader, "read", read)
    with pytest.raises(RuntimeError, match="tensor name too long for runtime key: 128 UTF-8 bytes"):
        m.read_header(path)
    assert max(reads) <= 8  # The 128-byte name is rejected before its allocation/read.
    fixture(path, tensors=[("d" * 127, (1,), 0, 0)])
    assert m.read_header(path).tensors[0].name == "d" * 127


def test_legacy_type42_observation_remains_separate_from_native_admission(tmp_path):
    from inferssd.source import header

    path = tmp_path / "model.gguf"
    fixture(path, tensors=[("dense", (128, 1, 256), 42, 0)])
    assert header.read(str(path)).tensors[0].type == 42
    with pytest.raises(ValueError, match="unsupported native ggml type 42"):
        m.read_header(path)


def test_path_controls_missing_empty_and_alignment(tmp_path):
    path = tmp_path / "model.gguf"
    fixture(path)
    with pytest.raises(RuntimeError, match="unaligned offset"):
        m.build_rows([path], {"up"}, 256, True)
    with pytest.raises(RuntimeError, match="must be positive"):
        m.build_rows([path], {"up"}, 0)
    with pytest.raises(RuntimeError, match="nonempty subset"):
        m.build_rows([path], {"other"}, 256)
    with pytest.raises(RuntimeError, match="no expert tensors"):
        m.build_rows([path], {"down"}, 256)
    with pytest.raises(RuntimeError, match="matched no files"):
        m.parse_model_paths([str(tmp_path / "missing*")])


@pytest.mark.skipif(sys.platform == "win32", reason="Windows file names cannot hold tabs")
def test_a_source_path_with_a_tab_is_refused(tmp_path):
    bad = tmp_path / "bad\tpath.gguf"
    fixture(bad)
    with pytest.raises(RuntimeError, match="tabs or newlines"):
        m.read_header(bad)


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="POSIX source ownership")
def test_fifo_is_rejected_without_blocking(tmp_path):
    path = tmp_path / "source.gguf"
    os.mkfifo(path)
    with pytest.raises(ValueError, match="regular file"):
        m.read_header(path)


@pytest.mark.skipif(sys.platform == "win32", reason="Windows does not replace a file that is open")
def test_source_replacement_during_header_read_is_refused(tmp_path, monkeypatch):
    path, other = tmp_path / "source.gguf", tmp_path / "replacement.gguf"
    fixture(path)
    fixture(other)
    original = m._Reader.read
    replaced = False

    def read(reader, count):
        nonlocal replaced
        data = original(reader, count)
        if not replaced:
            replaced = True
            os.replace(other, path)
        return data

    monkeypatch.setattr(m._Reader, "read", read)
    with pytest.raises(RuntimeError, match="changed while reading"):
        m.read_header(path)


def test_module_cli_preserves_tsv_order_and_summary(tmp_path, monkeypatch, capsys):
    path = tmp_path / "model.gguf"
    fixture(
        path,
        tensors=[
            ("blk.9.ffn_up_exps.weight", (32, 2, 256), 8, 0),
            ("blk.2.ffn_gate_exps.weight", (32, 2, 256), 8, 17408),
        ],
    )
    out, summary = tmp_path / "pending.tsv", tmp_path / "pending.json"
    monkeypatch.setenv("SSD_LLAMA_REPO_HEAD", "fixture-head")
    assert (
        m.main(
            [
                "--model-glob",
                str(path),
                "--n-experts",
                "256",
                "--out",
                str(out),
                "--summary-json",
                str(summary),
            ]
        )
        == 0
    )
    with out.open(newline="") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    assert len(rows) == 512 and rows[0]["tensor"] == "blk.2.ffn_gate_exps.weight"
    assert rows[-1]["tensor"] == "blk.9.ffn_up_exps.weight"
    assert json.loads(summary.read_text())["repo_head"] == "fixture-head"
    assert capsys.readouterr().out.startswith("entries=512\nexpert_tensors=2\n")


@pytest.mark.parametrize("failure", ["build", "first-publish", "second-publish"])
def test_launch_retains_temporary_staging_and_ordered_atomic_failure_boundary(tmp_path, failure):
    cache = tmp_path / "cache"
    events = []

    def run(command):
        assert command[:4] == ["/bound/python", "-I", "-m", "inferssd.source.gguf_alias"]
        assert command[command.index("--repo") + 1] == "/repo"
        Path(command[command.index("--out") + 1]).write_text("new alias")
        if failure == "build":
            raise subprocess.CalledProcessError(1, command)
        Path(command[command.index("--summary-json") + 1]).write_text(
            '{"model_shards":1,"entries":256}'
        )

    def replace(source, target):
        events.append(Path(target).suffix)
        if failure == "first-publish" or failure == "second-publish" and len(events) == 2:
            raise OSError("publication failed")
        os.replace(source, target)

    with pytest.raises((subprocess.CalledProcessError, OSError)):
        generate_alias(
            ("/models/model.gguf",),
            "muse",
            cache,
            repo="/repo",
            python="/bound/python",
            run=run,
            replace=replace,
        )
    assert not list(cache.glob(".index-*"))
    assert events == (
        [] if failure == "build" else [".tsv"] if failure == "first-publish" else [".tsv", ".json"]
    )
    # Preserve the existing per-file atomic boundary, including a complete new
    # alias after failure of the second publication; no engine may launch then.
    assert len(list(cache.glob("*.tsv"))) == int(failure == "second-publish")
