"""Default shared guardian body preserves its frozen standalone caller's events."""

from __future__ import annotations

import ast
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
from observe_guardian_reference import CASES, observe
from test_ownership_core import WithoutTypes

from inferssd.platform import guardian

# The recorded caller passes these variables under their historical names.
HISTORICAL = {
    "INFERSSD_RUN_DIR": "SSDASH_RUN_DIR",
    "INFERSSD_WRAPPER_RECEIPT": "SSDASH_WRAPPER_RECEIPT",
}


class DefaultPorts(WithoutTypes):
    def visit_AnnAssign(self, node):
        return super().visit_AnnAssign(self.generic_visit(node))

    def visit_FunctionDef(self, node):
        node = super().visit_FunctionDef(node)
        node.args.kwonlyargs, node.args.kw_defaults = [], []
        return node

    def visit_Assign(self, node):
        if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            if node.targets[0].id == "common":
                return None
        return self.generic_visit(node)

    def visit_Constant(self, node):
        if isinstance(node.value, str) and node.value in HISTORICAL:
            return ast.copy_location(ast.Constant(HISTORICAL[node.value]), node)
        return node

    def visit_Name(self, node):
        if isinstance(node.ctx, ast.Load) and node.id in (
            "pass_fds",
            "lease_unit",
            "interrupted",
            "request_stop",
            "native",
        ):
            return ast.copy_location(
                ast.parse("()" if node.id == "pass_fds" else "None").body[0].value, node
            )
        return node


def candidate(case):
    tree = DefaultPorts().visit(ast.parse(Path(guardian.__file__).read_text()))
    tree = ast.fix_missing_locations(tree)
    namespace = dict(observe.__globals__)
    namespace["ast"] = SimpleNamespace(
        parse=lambda _text: tree, FunctionDef=ast.FunctionDef, Module=ast.Module
    )
    return types.FunctionType(observe.__code__, namespace)(case)


@pytest.mark.parametrize("case", CASES, ids=lambda c: c["name"])
def test_default_guardian_events(case):
    assert candidate(case) == observe(case)
