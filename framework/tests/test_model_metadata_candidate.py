"""Candidate observations against frozen and dynamic whole-reference source behavior."""

from __future__ import annotations

import ast
import contextlib
import copy
import json
import os
import subprocess
import sys
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_model_metadata import FIXTURES, data, observe, runtime

from inferssd.source import experts, header

HEADER_IO = {"read", "shards", "read_model"}
SOURCE_IO = {
    "gguf_tensors",
    "gguf_entries",
    "pack_entries",
    "current",
    "index",
    "build",
    "load",
    "iolog",
    "facts",
    "lane_args",
    "dense",
}


@contextlib.contextmanager
def candidate_api(fs):
    # Same in-memory port adapter as the reference; no replacement parser or
    # candidate-derived inputs.
    port = SimpleNamespace(
        **{
            name: getattr(fs, name)
            for name in ("open", "glob", "abspath", "getmtime", "isfile", "makedirs", "replace")
        },
        map_read=fs.mmap,
    )
    h = SimpleNamespace(**vars(header))
    s = SimpleNamespace(**vars(experts))
    for name in HEADER_IO:
        setattr(h, name, partial(getattr(header, name), filesystem=port))
    for name in SOURCE_IO:
        kw = {"filesystem": port}
        if name == "iolog":
            kw["random_factory"] = fs.rng
        setattr(s, name, partial(getattr(experts, name), **kw))
    yield h, s


def candidate_observe(case):
    return observe(case, api=candidate_api)


@pytest.fixture(scope="module")
def frozen_candidate():
    script = """
import json, sys
sys.path.insert(0, sys.argv[1])
from test_model_metadata import data
from test_model_metadata_candidate import candidate_observe
print(json.dumps({c['name']: candidate_observe(c) for c in data()['cases']}))
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(Path(__file__).parent)],
        env=dict(os.environ, PYTHONHASHSEED="0"),
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)


@pytest.mark.parametrize("case", data()["cases"], ids=lambda c: c["name"])
def test_frozen_candidate(case, frozen_candidate):
    assert frozen_candidate[case["name"]] == case["expected"][runtime()]


@pytest.mark.parametrize("case", data()["cases"], ids=lambda c: c["name"])
def test_same_process_reference_equivalence(case):
    # Compare set/stat iteration dynamically as well as under the frozen seed.
    assert candidate_observe(case) == observe(case)


def test_seed17_dynamic_equivalence():
    script = """
import sys
sys.path.insert(0, sys.argv[1])
from test_model_metadata import data, observe
from test_model_metadata_candidate import candidate_observe
for case in data()['cases']:
    assert candidate_observe(case) == observe(case), case['name']
print('52 complete reference/candidate observations matched under seed 17')
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(Path(__file__).parent)],
        env=dict(os.environ, PYTHONHASHSEED="17"),
        check=True,
        capture_output=True,
        text=True,
    )
    assert "52 complete" in result.stdout


class ErasePortsAndTypes(ast.NodeTransformer):
    def visit_FunctionDef(self, node):
        for arg in node.args.args:
            arg.annotation = None
        node.returns = None
        node.args.kwonlyargs, node.args.kw_defaults = [], []
        return self.generic_visit(node)

    def visit_AnnAssign(self, node):
        return (
            ast.Assign(targets=[node.target], value=self.visit(node.value)) if node.value else None
        )

    def visit_Expr(self, node):
        if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            return None
        return self.generic_visit(node)

    def visit_Call(self, node):
        if isinstance(node.func, ast.Name) and node.func.id == "cast":
            return self.visit(node.args[1])
        node = self.generic_visit(node)
        text = ast.unparse(node.func)
        restore = {
            "filesystem.open": "open",
            "filesystem.map_read": "mmap.mmap",
            "filesystem.glob": "glob.glob",
            "filesystem.abspath": "os.path.abspath",
            "filesystem.getmtime": "os.path.getmtime",
            "filesystem.isfile": "os.path.isfile",
            "filesystem.makedirs": "os.makedirs",
            "filesystem.replace": "os.replace",
            "random_factory": "random.Random",
            "emit": "print",
        }
        if text in restore:
            node.func = ast.parse(restore[text], mode="eval").body
        node.keywords = [
            k for k in node.keywords if k.arg not in ("filesystem", "emit", "random_factory")
        ]
        return node


def normalized(node):
    return ast.dump(ErasePortsAndTypes().visit(copy.deepcopy(node)), include_attributes=False)


@pytest.mark.parametrize("old,module", [("gguf_header", header), ("expert_source", experts)])
def test_executable_ast_and_constants(old, module):
    f = ast.parse((FIXTURES / f"oracle/{old}_reference.py").read_text())
    new = ast.parse(Path(module.__file__).read_text())

    def funcs(tree):
        return {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}

    for name, node in funcs(f).items():
        assert normalized(node) == normalized(funcs(new)[name]), name
    if old == "gguf_header":
        f_cursor = next(n for n in f.body if isinstance(n, ast.ClassDef) and n.name == "_Cursor")
        n_cursor = next(n for n in new.body if isinstance(n, ast.ClassDef) and n.name == "_Cursor")
        assert normalized(f_cursor) == normalized(n_cursor)
        for name in ("Tensor", "Header"):
            old_class = next(n for n in f.body if isinstance(n, ast.ClassDef) and n.name == name)
            new_class = next(n for n in new.body if isinstance(n, ast.ClassDef) and n.name == name)
            assert [n.target.id for n in old_class.body if isinstance(n, ast.AnnAssign)] == [
                n.target.id for n in new_class.body if isinstance(n, ast.AnnAssign)
            ]
            assert {
                n.name: normalized(n) for n in old_class.body if isinstance(n, ast.FunctionDef)
            } == {n.name: normalized(n) for n in new_class.body if isinstance(n, ast.FunctionDef)}
    for name in ("TYPES", "SCALARS", "SHARD") if old == "gguf_header" else ("PAGE", "ROLE"):
        f_node = next(
            n.value for n in f.body if isinstance(n, ast.Assign) and n.targets[0].id == name
        )
        n_node = next(
            n.value for n in new.body if isinstance(n, ast.Assign) and n.targets[0].id == name
        )
        assert ast.dump(f_node) == ast.dump(n_node)
