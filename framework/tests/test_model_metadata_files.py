"""Local filesystem boundaries of the source operations, using header-only synthetic files."""

from __future__ import annotations

import ast
import contextlib
import io
import os
import shutil
import subprocess
import sys
from unittest.mock import patch

import pytest
from test_model_metadata import FIXTURES, REFERENCE_HEADER, REFERENCE_SOURCE, gguf, pack

from inferssd.source import experts, header


def original_command(argv):
    tree = ast.parse((FIXTURES / "oracle/expert_source_reference.py").read_text())
    block = next(n for n in tree.body if isinstance(n, ast.If))
    with patch.object(sys, "argv", ["expert_source.py", *argv]):
        exec(
            compile(ast.Module(body=[block], type_ignores=[]), "original-dispatch", "exec"),
            dict(vars(REFERENCE_SOURCE), __name__="__main__"),
        )


def operation(argv):
    """Run one source operation with the original command line's arguments, in process."""
    kind, *words = argv
    if kind == "index":
        experts.index(words[0], words[1] if len(words) == 3 else None, words[-1])
    elif kind == "facts":
        experts.facts(words[0], words[1] if len(words) == 2 else None)
    elif kind == "lane-args":
        experts.lane_args(words[0])
    elif kind == "dense":
        experts.dense(*words)
    else:
        experts.iolog(words[0], int(words[1]), int(words[2]), words[3], None)


@pytest.mark.parametrize("kind", ["gguf", "pack", "split"])
def test_real_files_and_output_bytes(tmp_path, kind):
    model = tmp_path / ("model-00001-of-00002.gguf" if kind == "split" else "model.gguf")
    model.write_bytes(gguf({}))
    if kind == "split":
        (tmp_path / "model-00002-of-00002.gguf").write_bytes(gguf({"tensors": []}))
    packed = tmp_path / "experts.pack"
    packed.write_bytes(pack({}))
    homes = tmp_path / "homes.tsv"
    homes.write_text("blk.0.attn_q.weight\tdrive\noutput_norm.weight\tram\n")
    target = tmp_path / "index"
    args = ["index", str(model), *([str(packed)] if kind == "pack" else []), str(target)]
    commands = [
        args,
        args,
        ["facts", str(model), *([] if kind == "pack" else [str(target / "experts.tsv")])],
        ["lane-args", str(target)],
        ["iolog", str(target), "3", "17", str(target / "reads.iolog")],
        ["dense", str(model), str(homes), str(target / "dense.tsv")],
    ]
    expected = []
    for argv in commands:
        stdout, stderr = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            original_command(argv)
        expected.append((stdout.getvalue(), stderr.getvalue()))
    original_files = {p.name: p.read_bytes() for p in target.iterdir()}
    shutil.rmtree(target)
    for argv, (stdout, stderr) in zip(commands, expected):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            operation(argv)
        assert (out.getvalue(), err.getvalue()) == (stdout, stderr)
    assert {p.name: p.read_bytes() for p in target.iterdir()} == original_files
    # The local mmap adapter produces the same NamedTuple data after unmapping.
    assert header.read_model(str(model)) == REFERENCE_HEADER.read_model(str(model))


def test_real_mtime_and_glob_escape(tmp_path):
    model = tmp_path / "model[1]-00001-of-00002.gguf"
    model.write_bytes(gguf({}))
    (tmp_path / "model[1]-00002-of-00002.gguf").write_bytes(gguf({"tensors": []}))
    # Must not include this unmatched path: the shard prefix is glob-escaped.
    (tmp_path / "model1-00002-of-00002.gguf").write_bytes(b"bad")
    assert header.shards(str(model)) == REFERENCE_HEADER.shards(str(model))
    target = str(tmp_path / "index")
    expected = REFERENCE_SOURCE.build(str(model), None, target)
    assert experts.current(target, str(model), None) == expected
    made = os.path.getmtime(f"{target}/experts.json")
    os.utime(model, (made + 1, made + 1))
    assert REFERENCE_SOURCE.current(target, str(model), None) is None
    assert experts.current(target, str(model), None) is None


def test_real_missing_input_no_artifact(tmp_path):
    missing, out = str(tmp_path / "missing.gguf"), tmp_path / "index"
    with pytest.raises(FileNotFoundError):
        operation(["index", missing, str(out)])
    # The reference creates the directory before header access; preserve that failure side effect.
    assert out.is_dir() and not list(out.iterdir())


def test_source_imports_are_inert():
    script = """
import builtins, glob, mmap, os, random
from unittest.mock import patch

def unexpected(*args, **kwargs):
    raise AssertionError('source import performed IO or random work')
with patch.object(builtins, 'open', unexpected), \
     patch.object(glob, 'glob', unexpected), \
     patch.object(mmap, 'mmap', unexpected), \
     patch.object(os, 'makedirs', unexpected), \
     patch.object(os, 'replace', unexpected), \
     patch.object(os.path, 'getmtime', unexpected), \
     patch.object(random.Random, 'choice', unexpected):
    import inferssd.source.header, inferssd.source.experts
print('inert')
"""
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=True
    )
    assert result.stdout == "inert\n"
