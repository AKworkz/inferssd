#!/usr/bin/env python3
"""Enforce the repository root layout and product ownership, not directory counts."""
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
DIRECTORIES = {"engine", "framework", "bench", "tests", "scripts", "docs", ".github"}
FILES = {
    ".clang-format", ".clang-tidy", ".dockerignore", ".editorconfig", ".gitattributes",
    ".gitignore", ".gitleaks.toml", ".gitleaksignore",
    "CHANGELOG.md", "CMakeLists.txt", "CMakePresets.json", "CONTRIBUTING.md", "LICENSE",
    "NOTICE", "README.md",
}
tracked = set(subprocess.check_output(
    ["git", "-C", str(ROOT), "ls-files"], text=True).splitlines())
errors = []
for path in sorted(tracked):
    first = path.split("/", 1)[0]
    if ("/" in path and first not in DIRECTORIES) or ("/" not in path and path not in FILES):
        errors.append("unexpected root item: " + path)
# Native tests belong with their build; retained old commands are test fixtures only.
for path in tracked:
    if path.startswith("tests/") and pathlib.Path(path).suffix in {".c", ".cpp", ".cu"}:
        errors.append("native test outside engine/tests: " + path)
    if path.startswith(("framework/compat/", "engine/tools/adaptive/")):
        errors.append("unregistered legacy production adapter: " + path)
if errors:
    print("\n".join(sorted(set(errors))), file=sys.stderr)
    sys.exit(1)
print(f"Product layout passed: {len(tracked)} tracked files")
