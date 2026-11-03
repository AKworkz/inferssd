"""Reference-first source metadata/filesystem observations; generated headers contain no weights."""

from __future__ import annotations

import ast
import contextlib
import copy
import fnmatch
import hashlib
import io
import json
import os
import random
import struct
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from reference_oracles import exact, load

FIXTURES = Path(__file__).parent / "fixtures"
REFERENCE_HEADER = load("gguf_header")
with patch.dict(sys.modules, gguf_header=REFERENCE_HEADER):
    REFERENCE_SOURCE = load("expert_source")
GOLDENS = FIXTURES / "model-metadata-goldens.json"


def gguf(config):
    def string(s):
        b = s.encode() if isinstance(s, str) else s
        return struct.pack("<Q", len(b)) + b

    def value(t, v):
        if t == 8:
            return string(v)
        if t == 9:
            et, xs = v
            return struct.pack("<IQ", et, len(xs)) + b"".join(value(et, x) for x in xs)
        if t not in REFERENCE_HEADER.SCALARS:
            return b""
        return struct.pack("<" + REFERENCE_HEADER.SCALARS[t], v)

    tensors = config.get("tensors", default_tensors())
    kv = config.get("kv", [["general.architecture", 8, "toy"], ["general.alignment", 4, 32]])
    out = config.get("magic", "GGUF").encode() + struct.pack(
        "<IQQ", config.get("version", 3), len(tensors), len(kv)
    )
    for k, t, v in kv:
        out += string(k) + struct.pack("<I", t) + value(t, v)
    offset = 0
    for name, shape, t in tensors:
        out += string(name) + struct.pack(f"<I{len(shape)}QIQ", len(shape), *shape, t, offset)
        # Input offsets are an independent fixture choice. They intentionally do
        # not depend on the candidate type table or predicted tensor size.
        offset += 65536
    return out[: config["truncate"]] if "truncate" in config else out


def default_tensors():
    return [["token_embd.weight", [512, 32], 8], ["output_norm.weight", [512], 0]] + [
        row
        for layer in (0, 1)
        for row in [
            [f"blk.{layer}.attn_q.weight", [512, 1024], 8],
            [f"blk.{layer}.attn_odd.weight", [512, 997], 8],
            [f"blk.{layer}.ffn_gate_exps.weight", [512, 256, 2], 12],
            [f"blk.{layer}.ffn_up_exps.weight", [512, 256, 2], 12],
            [f"blk.{layer}.ffn_down_exps.weight", [256, 1024, 2], 12],
        ]
    ]


def pack(config):
    rows = config.get(
        "rows",
        [
            [
                f"blk.{layer}.ffn_{role}_exps.weight",
                e,
                4096 * (1 + layer * 6 + e * 3 + i),
                4096 + i * 32,
            ]
            for layer in (0, 1)
            for e in (0, 1)
            for i, role in enumerate(("gate", "up", "down"))
        ],
    )
    raw = struct.pack(
        "<16sIIQQ",
        config.get("magic", "GGMLMOEPACKv1").encode(),
        config.get("version", 1),
        0,
        len(rows),
        0,
    )
    raw += b"".join(
        struct.pack("<128siIQQ", name.encode(), e, 0, off, size) for name, e, off, size in rows
    )
    return raw[: config["truncate"]] if "truncate" in config else raw


class Files:
    def __init__(self, case):
        self.case = case
        self.data = {
            p: gguf(cfg) for p, cfg in case.get("models", {"/case/model.gguf": {}}).items()
        }
        self.data.update({p: pack(cfg) for p, cfg in case.get("packs", {}).items()})
        self.data.update({p: text.encode() for p, text in case.get("files", {}).items()})
        self.initial = copy.deepcopy(self.data)
        self.mtimes = {p: 100 for p in self.data}
        self.mtimes.update(case.get("mtimes", {}))
        self.events, self.fds = [], {}
        self.next_fd = 7
        self.stdout, self.stderr = io.StringIO(), io.StringIO()

    def path(self, p):
        return os.path.normpath(p if p.startswith("/") else "/case/" + p)

    def fail(self, op, p):
        if self.case.get("fault") == [op, p]:
            raise OSError(f"configured {op} failure: {p}")

    def open(self, p, mode="r"):
        self.events.append(["open", p, mode])
        self.fail("open", p)
        path = self.path(p)
        if mode != "w" and path not in self.data:
            raise FileNotFoundError(f"missing synthetic file: {p}")
        if mode == "w":
            self.data[path] = b""
            self.mtimes[path] = 200
        owner = self
        base = io.BytesIO if "b" in mode else io.StringIO
        initial = self.data[path] if "b" in mode else self.data[path].decode()

        class Stream(base):
            def fileno(self):
                owner.events.append(["fileno", p])
                fd = owner.next_fd
                owner.next_fd += 1
                owner.fds[fd] = path
                return fd

            def read(self, *args):
                owner.events.append(["read", p, list(args)])
                owner.fail("read", p)
                return super().read(*args)

            def __next__(self):
                owner.events.append(["next", p])
                owner.fail("next", p)
                return super().__next__()

            def write(self, text):
                owner.events.append(["write", p, text])
                if owner.case.get("fault") == ["write", p]:
                    super().write(text[:1])
                    owner.data[path] = self.getvalue().encode()
                    raise OSError(f"configured write failure: {p}")
                result = super().write(text)
                owner.data[path] = self.getvalue().encode()
                return result

            def writelines(self, lines):
                owner.events.append(["writelines", p])
                for line in lines:
                    self.write(line)

            def close(self):
                if not self.closed:
                    owner.events.append(["close", p])
                    super().close()
                    owner.fail("close", p)

            def __del__(self):
                try:
                    self.close()
                except OSError as error:
                    owner.events.append(["ignored_close", p, str(error)])

        return Stream(initial)

    def mmap(self, fd, length, access):
        path = self.fds[fd]
        self.events.append(["mmap", path, length, access])
        self.fail("mmap", path)
        if not self.data[path]:
            raise ValueError("cannot mmap an empty file")
        owner = self

        class Mapped(bytes):
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                owner.events.append(["munmap", path])
                owner.fail("munmap", path)

        return Mapped(self.data[path])

    def glob(self, pattern):
        self.events.append(["glob", pattern])
        self.fail("glob", pattern)
        return [p for p in self.data if fnmatch.fnmatchcase(p, pattern)]

    def abspath(self, p):
        self.events.append(["abspath", p])
        self.fail("abspath", p)
        return self.path(p)

    def getmtime(self, p):
        self.events.append(["getmtime", p])
        self.fail("getmtime", p)
        if self.path(p) not in self.data:
            raise FileNotFoundError(f"missing synthetic file: {p}")
        return self.mtimes[self.path(p)]

    def isfile(self, p):
        self.events.append(["isfile", p])
        self.fail("isfile", p)
        return self.path(p) in self.data

    def makedirs(self, p, exist_ok=False):
        self.events.append(["makedirs", p, exist_ok])
        self.fail("makedirs", p)

    def replace(self, a, b):
        self.events.append(["replace", a, b])
        self.fail("replace", a)
        self.data[self.path(b)] = self.data.pop(self.path(a))
        self.mtimes[self.path(b)] = self.mtimes.pop(self.path(a))

    def rng(self, seed):
        self.events.append(["random", seed])
        rng = random.Random(seed)
        owner = self

        class RNG:
            def choice(self, xs):
                result = rng.choice(xs)
                owner.events.append(["choice", exact(xs), exact(result)])
                return result

        return RNG()

    def output(self, side, text):
        self.events.append([side, text])
        self.fail(side, "")
        return getattr(self, side).write(text)


@contextlib.contextmanager
def reference_api(fs):
    with contextlib.ExitStack() as stack:
        stack.enter_context(patch.object(REFERENCE_HEADER, "open", fs.open, create=True))
        stack.enter_context(
            patch.object(REFERENCE_HEADER, "mmap", SimpleNamespace(mmap=fs.mmap, ACCESS_READ=1))
        )
        stack.enter_context(
            patch.object(
                REFERENCE_HEADER,
                "glob",
                SimpleNamespace(glob=fs.glob, escape=__import__("glob").escape),
            )
        )
        stack.enter_context(patch.object(REFERENCE_SOURCE, "open", fs.open, create=True))
        stack.enter_context(
            patch.object(
                REFERENCE_SOURCE,
                "os",
                SimpleNamespace(
                    path=SimpleNamespace(
                        abspath=fs.abspath, getmtime=fs.getmtime, isfile=fs.isfile
                    ),
                    makedirs=fs.makedirs,
                    replace=fs.replace,
                ),
            )
        )
        stack.enter_context(
            patch.object(REFERENCE_SOURCE, "random", SimpleNamespace(Random=fs.rng))
        )
        yield REFERENCE_HEADER, REFERENCE_SOURCE


def observe(case, api=reference_api):
    fs, steps = Files(case), []
    with (
        api(fs) as (header, source),
        patch.object(sys, "stdout", SimpleNamespace(write=lambda text: fs.output("stdout", text))),
        patch.object(sys, "stderr", SimpleNamespace(write=lambda text: fs.output("stderr", text))),
    ):
        for action in case["actions"]:
            kind, *args = action
            try:
                if kind == "cli":
                    old = sys.argv
                    try:
                        sys.argv = ["expert_source.py", *args]
                        tree = ast.parse(
                            (FIXTURES / "oracle/expert_source_reference.py").read_text()
                        )
                        main = next(n for n in tree.body if isinstance(n, ast.If))
                        # Original dispatch block, original function objects and output ports.
                        scope = dict(vars(source), __name__="__main__")
                        exec(
                            compile(
                                ast.Module(body=[main], type_ignores=[]), "source-dispatch", "exec"
                            ),
                            scope,
                        )
                        value = None
                    finally:
                        sys.argv = old
                elif kind == "mtime":
                    fs.mtimes[args[0]] = args[1]
                    value = None
                elif kind == "edit":
                    fs.data[args[0]] = args[1].encode()
                    value = None
                elif kind == "delete":
                    del fs.data[args[0]]
                    value = None
                elif kind.startswith("header."):
                    value = getattr(header, kind.split(".")[1])(*args)
                else:
                    value = getattr(source, kind)(*args)
                steps.append([kind, "return", exact(value)])
            except (Exception, SystemExit) as error:
                steps.append([kind, type(error).__name__, str(error)])
    changed = {
        p: b.decode("utf-8", errors="backslashreplace")
        for p, b in fs.data.items()
        if fs.initial.get(p) != b
    }
    return {
        "steps": steps,
        "events": fs.events,
        "stdout": fs.stdout.getvalue(),
        "stderr": fs.stderr.getvalue(),
        "files": changed,
        "removed": sorted(set(fs.initial) - set(fs.data)),
    }


def runtime():
    key = f"{sys.version_info[0]}.{sys.version_info[1]}"
    if key not in ("3.9", "3.12"):
        pytest.skip("recorded reference transcripts cover Python 3.9 and 3.12 only")
    return key


def data():
    return json.loads(GOLDENS.read_text())


@pytest.fixture(scope="module")
def fixed_reference_observations():
    if os.environ.get("PYTHONHASHSEED") == "0":
        return {c["name"]: observe(c) for c in data()["cases"]}
    script = """
import json, sys
sys.path.insert(0, sys.argv[1])
from test_model_metadata import data, observe
print(json.dumps({c['name']: observe(c) for c in data()['cases']}))
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(Path(__file__).parent)],
        env=dict(os.environ, PYTHONHASHSEED="0"),
        check=True,
        text=True,
        capture_output=True,
    )
    return json.loads(result.stdout)


@pytest.mark.parametrize("case", data()["cases"], ids=lambda c: c["name"])
def test_reference_source_metadata(case, fixed_reference_observations):
    assert fixed_reference_observations[case["name"]] == case["expected"][runtime()]


def test_source_oracles():
    manifest = json.loads((FIXTURES / "oracle/source-metadata-manifest.json").read_text())
    for row in manifest["files"]:
        assert (
            hashlib.sha256((FIXTURES / "oracle" / row["file"]).read_bytes()).hexdigest()
            == row["sha256"]
        )
