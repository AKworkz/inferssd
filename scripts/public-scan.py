#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Check public changes without exposing matched values in diagnostics."""
import argparse
import ipaddress
import json
import re
import subprocess
import sys
from pathlib import Path

PATH = re.compile(r"(?<![\w$}])/(?:home|Users|var/tmp)/[A-Za-z0-9_-][A-Za-z0-9_.-]*(?:/|\b)|(?<![\w$}])/root/")
LOGIN = re.compile(r"(?<![\w./-])[A-Za-z_][A-Za-z0-9_.%+-]*@([A-Za-z_][A-Za-z0-9.-]*)")
IP = re.compile(r"(?<![\w.])(?:\d{1,3}\.){3}\d{1,3}(?![\w.])")
# The company is written "WiCi AI", its GitHub owner "wici-ai/" and its domain "wici.ai".
INTERNAL = re.compile(r"HANDOFF-20\d\d|MEMORY\.md|another session|\bwici\b(?! AI|-ai/|\.ai(?![\w-]|\.\w))",
                      re.IGNORECASE)
TASK = re.compile(r"\btask[ -]?#?[0-9]+\b", re.IGNORECASE)
HAN = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")
COMMENT = re.compile(r"^\s*(?://|/\*|\*|#)")
SELF = "scripts/public-scan.py"


def git(*args):
    return subprocess.check_output(["git", *args]).decode("utf-8", errors="replace")


def added_line_text(filename, text):
    """Decode complete literal escapes while preserving every other fragment."""
    if filename.endswith(".ps1"):
        # Generated batch strings contain PowerShell `r`n escapes, not a login such
        # as the apparent suffix n@exit. Only complete double-quoted literals are
        # normalized; escaped backticks and real identifiers remain visible.
        def powershell_literal(match):
            return re.sub(r"`(.)", lambda escape: {"r": "\r", "n": "\n", "t": "\t"}.get(
                escape.group(1), escape.group(0)), match.group())
        return re.sub(r'"(?:`.|[^"`])*"', powershell_literal, text)
    if not filename.endswith(".json"):
        return text
    parts = []
    start = 0
    while start < len(text):
        opening = text.find('"', start)
        if opening < 0:
            parts.append(text[start:])
            break
        parts.append(text[start:opening])
        closing = opening + 1
        while closing < len(text):
            if text[closing] == "\\":
                closing += 2
            elif text[closing] == '"':
                break
            else:
                closing += 1
        if closing >= len(text):
            parts.append(text[opening:])
            break
        literal = text[opening:closing + 1]
        try:
            decoded = json.loads(literal)
        except ValueError:
            parts.append(literal)
        else:
            # Retain delimiters so decoding does not join adjacent fragments.
            parts.append('"' + decoded + '"')
        start = closing + 1
    return "".join(parts)


def inspect(text, location, strict=False):
    blocked = []
    if PATH.search(text):
        blocked.append("personal path")
    if any("." not in match.group(1).rstrip(".") or match.group(1).rstrip(".").endswith(".local")
           for match in LOGIN.finditer(text)):
        blocked.append("remote login identifier")
    for match in IP.finditer(text):
        try:
            address = ipaddress.ip_address(match.group())
        except ValueError:
            continue
        if address.is_private and not address.is_loopback and not address.is_unspecified:
            blocked.append("private network address")
            break
    internal = INTERNAL.search(text) or (COMMENT.search(text) and (TASK.search(text) or HAN.search(text)))
    if strict and (internal or TASK.search(text) or HAN.search(text)):
        blocked.append("internal wording")
    for kind in blocked:
        print(f"BLOCK {location}: {kind}")
    if internal and not strict:
        print(f"REPORT {location}: internal wording (review before release)")
    return len(blocked)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip", help="Skip one reviewed upstream snapshot commit")
    parser.add_argument("--ref", action="append", default=[], help="Proposed remote ref name")
    parser.add_argument("--text", action="append", default=[], help="Exported public discussion text")
    parser.add_argument("--tree", action="store_true", help="Require no internal wording in our current tree changes")
    parser.add_argument("revisions", nargs="*", help="Git revisions passed to rev-list; defaults to --all")
    args = parser.parse_args()
    failures = sum(inspect(ref, "ref", True) for ref in args.ref)
    for name in args.text:
        failures += inspect(Path(name).read_text(), f"text:{Path(name).name}", True)
    skipped = git("rev-parse", "--verify", args.skip).strip() if args.skip else None
    if args.tree:
        if not args.skip:
            parser.error("--tree requires --skip as the upstream comparison point")
        changes = [("tree", git("diff", "--no-ext-diff", "--unified=0", args.skip, "HEAD"))]
    else:
        commits = git("rev-list", *(args.revisions or ["--all"])).splitlines() if not args.text or args.revisions else []
        changes = []
        for commit in commits:
            failures += inspect(git("show", "-s", "--format=%B", commit), f"commit:{commit[:12]}", True)
            if commit != skipped:
                changes.append((commit[:12], git("show", "--format=", "--no-ext-diff", "--unified=0", commit)))
    for revision, diff in changes:
        filename = ""
        line_number = 0
        for line in diff.splitlines():
            if line.startswith("+++ b/"):
                filename = line[6:]
            elif line.startswith("@@"):
                match = re.search(r"\+(\d+)", line)
                line_number = int(match.group(1)) if match else 0
            elif line.startswith("+") and not line.startswith("+++"):
                if filename != SELF:
                    failures += inspect(added_line_text(filename, line[1:]),
                                        f"{revision}:{filename}:{line_number}", args.tree)
                line_number += 1
            elif not line.startswith("-"):
                line_number += 1
    print(f"public-scan: {failures} blocking finding(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
