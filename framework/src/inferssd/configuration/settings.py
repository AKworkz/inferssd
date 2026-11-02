"""Python mirror of the C++ registry-generated schema and settings v1 boundary."""

from __future__ import annotations

import json
import math
import re
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from importlib.resources import files
from typing import Any, Optional, Union, cast

Value = Union[int, str, tuple[int, ...], tuple[str, ...], tuple[tuple[str, float], ...]]


def schema() -> dict[str, Any]:
    """Return a fresh schema generated from registry.def, including scopes/receipts."""
    value: object = json.loads(files(__package__).joinpath("schema.json").read_text())
    if not isinstance(value, dict) or not isinstance(value.get("properties"), dict):
        raise ValueError("installed settings schema has no registry properties")
    if not value["properties"] or any(
        not isinstance(key, str) or not isinstance(prop, dict) or "kind" not in prop
        for key, prop in value["properties"].items()
    ):
        raise ValueError("installed settings schema has invalid registry properties")
    return cast(dict[str, Any], value)


@dataclass(frozen=True)
class Setting:
    key: str
    value: Value
    origin: str
    location: str
    receipt: str


@dataclass(frozen=True)
class Snapshot:
    entries: tuple[Setting, ...]
    platform: str

    def at(self, key: str) -> Setting:
        return next(item for item in self.entries if item.key == key)

    def document(self) -> dict[str, Any]:
        properties = schema()["properties"]
        values: dict[str, object] = {}
        for item in self.entries:
            kind = properties[item.key]["kind"]
            if kind == "costs":
                values[item.key] = dict(cast(tuple[tuple[str, float], ...], item.value))
            elif kind in ("cpus", "paths"):
                values[item.key] = list(cast(tuple[object, ...], item.value))
            else:
                values[item.key] = item.value
        return {
            "values": values,
            "sources": {
                item.key: {
                    "origin": item.origin,
                    "location": item.location,
                    "receipt": item.receipt,
                }
                for item in self.entries
            },
        }


PLATFORMS = ("linux", "darwin", "windows")


def host_platform() -> str:
    """The platform name of this host, the default of every platform option."""
    return {"darwin": "darwin", "win32": "windows"}.get(sys.platform, "linux")


def applies(prop: Mapping[str, Any], platform: str) -> bool:
    """A schema scope is "all" or the platform names joined by commas."""
    scope = cast(str, prop["scope"])
    return scope == "all" or platform in scope.split(",")


def integer(literal: str, low: int, high: int) -> int:
    if re.fullmatch(r"[0-9]+", literal) is None or not low <= int(literal) <= high:
        raise ValueError(f"expected decimal integer in {low}..{high}")
    return int(literal)


def cpus(literal: str) -> tuple[int, ...]:
    """An ordered CPU list of unique IDs and inclusive ranges, such as "0-2,5"."""
    result: list[int] = []
    for item in literal.split(",") if literal else []:
        endpoints = item.split("-")
        if len(endpoints) > 2:
            raise ValueError("invalid CPU list")
        low = integer(endpoints[0], 0, 2147483647)
        high = integer(endpoints[-1], 0, 2147483647)
        if high < low or high - low > 65535 or len(result) + high - low + 1 > 65536:
            raise ValueError("invalid or excessive CPU list")
        result.extend(range(low, high + 1))
    if len(set(result)) != len(result):
        raise ValueError("duplicate CPU ID")
    return tuple(result)


def typed(prop: Mapping[str, Any], literal: str) -> Value:
    literal.encode("utf-8")
    literal = literal.strip(" \t\r\n")
    kind = prop["kind"]
    if literal.startswith('"') and kind != "paths":
        if kind == "integer":
            raise ValueError("integer must not be quoted")
        decoded: object = json.loads(literal)
        if not isinstance(decoded, str):
            raise ValueError("expected string literal")
        literal = decoded
        if kind == "budget" and literal != "auto":
            raise ValueError("byte budget must not be quoted")
    if kind == "integer":
        return integer(literal, prop["minimum"], prop["maximum"])
    if kind == "budget":
        return "auto" if literal == "auto" else integer(literal, 0, prop["oneOf"][1]["maximum"])
    if kind == "paths":
        paths: object = json.loads(literal)
        if not isinstance(paths, list) or any(
            not isinstance(x, str) or not x or "\0" in x for x in paths
        ):
            raise ValueError("expected array of nonempty source paths without NUL")
        for path in paths:
            path.encode("utf-8")
        return tuple(paths)
    if kind == "cpus":
        return cpus(literal)
    costs: dict[str, float] = {}
    for word in filter(None, re.split(r"[ \t\r\n\v\f]+", literal.replace(",", " "))):
        key, equal, number = word.partition("=")
        if not equal or key not in prop["properties"] or key in costs:
            raise ValueError("unknown or duplicate cost term " + key)
        if (
            re.fullmatch(r"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?", number)
            is None
        ):
            raise ValueError("invalid cost term " + key)
        value = float(number)
        if not math.isfinite(value) or not 0 <= value <= 1e6 or key == "link_gbps" and value == 0:
            raise ValueError("invalid cost term " + key)
        costs[key] = value
    return tuple(sorted(costs.items()))


def uncomment(line: str) -> tuple[str, str]:
    quoted, escaped = False, False
    for i, character in enumerate(line):
        if escaped:
            escaped = False
            continue
        if quoted and character == "\\":
            escaped = True
            continue
        if character == '"':
            quoted = not quoted
        if not quoted and character == "#":
            return line[:i].strip(" \t\r\n"), line[i + 1 :]
    return line.strip(" \t\r\n"), ""


def parse(
    text: Optional[str] = None,
    source: str = "settings.ssd",
    *,
    overrides: Sequence[tuple[str, str]] = (),
    platform: str = "linux",
) -> Snapshot:
    source.encode("utf-8")
    if text is not None:
        text.encode("utf-8")
    if platform not in PLATFORMS:
        raise ValueError("unknown settings platform " + platform)
    properties: dict[str, Any] = schema()["properties"]
    values = {
        key: Setting(key, typed(prop, prop["default_literal"]), "default", "registry", "")
        for key, prop in properties.items()
    }

    def apply(key: str, literal: str, origin: str, location: str, receipt: str) -> None:
        try:
            prop = properties.get(key)
            if prop is None:
                raise ValueError("unknown key")
            if not applies(prop, platform):
                raise ValueError("setting is only supported on " + prop["scope"])
            value = typed(prop, literal)
        except ValueError as error:
            raise ValueError(f"{key} ({location}): {error}") from error
        values[key] = Setting(key, value, origin, location, receipt)

    if text is not None:
        lines = text.split("\n")
        if not lines or lines[0].strip(" \t\r\n") != "inferssd-settings 1":
            raise ValueError(source + ":1: expected inferssd-settings 1")
        seen = set()
        for n, line in enumerate(lines[1:], 2):
            body, receipt = uncomment(line)
            if not body:
                continue
            key, equal, literal = body.partition("=")
            key = key.strip(" \t\r\n")
            if not equal or key in seen:
                raise ValueError(f"{source}:{n}: missing '=' or duplicate key {key}")
            seen.add(key)
            apply(key, literal, "file", f"{source}:{n}", receipt)
    for key, literal in overrides:
        apply(key, literal, "cli", "--ssd-" + key.replace(".", "-").replace("_", "-"), "")
    return Snapshot(tuple(values.values()), platform)


def render(snapshot: Snapshot) -> str:
    properties = schema()["properties"]
    lines = ["inferssd-settings 1"]
    for item in snapshot.entries:
        prop = properties[item.key]
        if not applies(prop, snapshot.platform):
            continue
        literal = str(item.value)
        if prop["kind"] == "costs":
            literal = ",".join(
                f"{key}={value:.17g}"
                for key, value in cast(tuple[tuple[str, float], ...], item.value)
            )
            literal = json.dumps(literal, ensure_ascii=False)
        elif prop["kind"] == "cpus":
            literal = json.dumps(",".join(str(v) for v in cast(tuple[int, ...], item.value)))
        elif prop["kind"] == "paths":
            literal = json.dumps(item.value, ensure_ascii=False)
        elif item.value == "auto":
            literal = "auto"
        receipt = item.receipt or f" source={item.origin}; input={item.location}"
        lines.append(f"{item.key} = {literal}  #{receipt}")
    return "\n".join(lines) + "\n"
