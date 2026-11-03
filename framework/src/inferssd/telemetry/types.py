"""Typing views of native observations; these do not validate or copy input data."""

from __future__ import annotations

from typing import Any, Optional, Protocol, TextIO, TypedDict

# Legacy JSON fields are heterogeneous and retain their original permissive
# behavior. Any here is a view of that unversioned boundary, not a schema claim.
Record = dict[str, Any]
TokenKey = tuple[tuple[int, int], int]
TraceLayer = tuple[list[int], list[bool], list[bool]]


class ArmFiles(Protocol):
    def open(self, path: str, mode: str = "r", *, errors: Optional[str] = None) -> TextIO: ...

    def gzip_open(self, path: str, mode: str = "r", *, errors: Optional[str] = None) -> TextIO: ...


class ObservedArm(TypedDict):
    groups: list[list[TokenKey]]
    trace: dict[TokenKey, dict[int, TraceLayer]]
    prof: dict[TokenKey, list[float]]
    pf_graphs: list[Record]
