"""Candidate bodies execute against the unchanged reference core transcripts, without OS effects."""

from __future__ import annotations

import ast
import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
from observe_ownership_reference import GROUP_CASES, LEASE_CASES, group_case, lease_case

from inferssd.platform import lease, process


class WithoutTypes(ast.NodeTransformer):
    def visit_FunctionDef(self, node):
        self.generic_visit(node)
        node.returns = None
        for arg in (*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs):
            arg.annotation = None
        return node

    def visit_AnnAssign(self, node):
        return ast.copy_location(ast.Assign(targets=[node.target], value=node.value), node)


def run_body(observer, module, case):
    # Keep the frozen observer/effects unchanged, replacing only its implementation
    # AST input. This executes installed candidate bodies, not regenerated expectations.
    tree = WithoutTypes().visit(ast.parse(Path(module.__file__).read_text()))
    tree = ast.fix_missing_locations(tree)
    namespace = dict(observer.__globals__)
    namespace["ast"] = SimpleNamespace(
        parse=lambda _text: tree,
        FunctionDef=ast.FunctionDef,
        ClassDef=ast.ClassDef,
        Module=ast.Module,
    )
    return types.FunctionType(observer.__code__, namespace)(case)


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX process groups: killpg, SIGKILL")
@pytest.mark.parametrize("case", GROUP_CASES, ids=lambda c: c["name"])
def test_candidate_group_events(case):
    assert run_body(group_case, process, case) == group_case(case)


@pytest.mark.skipif(
    sys.platform == "win32", reason="POSIX lease: a descriptor passed over a Unix socket, flock"
)
@pytest.mark.parametrize("case", LEASE_CASES, ids=lambda c: c["name"])
def test_candidate_lease_events(case):
    assert run_body(lease_case, lease, case) == lease_case(case)


def test_only_supported_core_api():
    assert lease.CORE_API_VERSION == process.CORE_API_VERSION == 1


def test_original_core_fixture_manifest_stays_frozen():
    data = json.loads((Path(__file__).parent / "fixtures/oracle/engine-manifest.json").read_text())
    assert data["ownership_base"] == "76dccf491181995981324e1e88201c87886911b9"
