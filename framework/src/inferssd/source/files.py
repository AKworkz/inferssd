"""Explicit filesystem ports for legacy metadata artifacts.

The local adapter has no initialization side effects. Calls keep the original modes,
encoding defaults, mmap lifetime and replacement order; it adds no validation.
"""

from __future__ import annotations

import glob
import mmap
import os
from contextlib import AbstractContextManager
from typing import Any, BinaryIO, Literal, Protocol, TextIO, overload


class SourceFiles(Protocol):
    @overload
    def open(self, path: str, mode: Literal["rb"]) -> BinaryIO: ...
    @overload
    def open(self, path: str, mode: Literal["r", "w"] = "r") -> TextIO: ...
    def map_read(self, fd: int, length: int, *, access: int) -> AbstractContextManager[Any]: ...
    def glob(self, pattern: str) -> list[str]: ...
    def abspath(self, path: str) -> str: ...
    def getmtime(self, path: str) -> float: ...
    def isfile(self, path: str) -> bool: ...
    def makedirs(self, path: str, *, exist_ok: bool) -> None: ...
    def replace(self, source: str, destination: str) -> None: ...


class LocalFiles:
    @overload
    def open(self, path: str, mode: Literal["rb"]) -> BinaryIO: ...
    @overload
    def open(self, path: str, mode: Literal["r", "w"] = "r") -> TextIO: ...
    def open(self, path: str, mode: str = "r") -> Any:
        return open(path, mode, encoding=None if "b" in mode else "utf-8")

    def map_read(self, fd: int, length: int, *, access: int) -> mmap.mmap:
        return mmap.mmap(fd, length, access=access)

    def glob(self, pattern: str) -> list[str]:
        return glob.glob(pattern)

    def abspath(self, path: str) -> str:
        return os.path.abspath(path)

    def getmtime(self, path: str) -> float:
        return os.path.getmtime(path)

    def isfile(self, path: str) -> bool:
        return os.path.isfile(path)

    def makedirs(self, path: str, *, exist_ok: bool) -> None:
        os.makedirs(path, exist_ok=exist_ok)

    def replace(self, source: str, destination: str) -> None:
        os.replace(source, destination)


LOCAL_FILES: SourceFiles = LocalFiles()
