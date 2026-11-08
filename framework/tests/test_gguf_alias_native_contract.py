"""Compile the native type-trait geometry; never use the reference metadata table."""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

from inferssd.source.gguf_alias import NATIVE_TYPES


def test_strict_type_table_matches_compiled_native_traits(tmp_path):
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("native geometry contract requires a C++ compiler")
    repo = Path(__file__).resolve().parents[2]
    source = (repo / "engine/ggml/src/ggml.c").read_text(encoding="utf-8")
    traits = source.split("static const struct ggml_type_traits type_traits[GGML_TYPE_COUNT] = {")[
        1
    ].split("\n};", 1)[0]
    entries = re.findall(r"\[(GGML_TYPE_\w+|\d+)\]\s*=\s*\{([^{}]*)\}", traits)
    assert entries
    lines = []
    for enum, body in entries:
        block = re.search(r"\.blck_size\s*=\s*([^,\n]+)", body).group(1)
        size = re.search(r"\.type_size\s*=\s*([^,\n]+)", body).group(1)
        name = re.search(r'\.type_name\s*=\s*"([^"]+)"', body).group(1)
        # Retain removed entries in the compiled output as well: zero geometry
        # must remain excluded by the strict reader.
        lines.append(
            f'    std::printf("%d\\t%s\\t%lld\\t%zu\\n", int({enum}), '
            f'"{name}", static_cast<long long>({block}), static_cast<size_t>({size}));'
        )
    program = tmp_path / "native-types.cpp"
    program.write_text(
        '#include <cstdio>\n#include "ggml.h"\n'
        '#define GGML_COMMON_DECL_CPP\n#include "ggml-common.h"\n'
        "int main() {\n" + "\n".join(lines) + "\n}\n"
    )
    executable = tmp_path / "native-types"
    subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-I",
            str(repo / "engine/ggml/include"),
            "-I",
            str(repo / "engine/ggml/src"),
            str(program),
            "-o",
            str(executable),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    result = subprocess.run(
        [str(executable)], check=True, capture_output=True, text=True, timeout=10
    )
    native = {}
    excluded = set()
    for line in result.stdout.splitlines():
        number, name, block, size = line.split("\t")
        if int(block) and int(size):
            native[int(number)] = (name.upper(), int(block), int(size))
        else:
            excluded.add(int(number))
    assert native == NATIVE_TYPES
    assert excluded.isdisjoint(NATIVE_TYPES)
    assert 42 not in NATIVE_TYPES
