"""Local file operations for reading a run's outputs, with no I/O during construction."""

from __future__ import annotations

import gzip
from typing import Optional, TextIO, cast


class LocalArmFiles:
    def open(self, path: str, mode: str = "r", *, errors: Optional[str] = None) -> TextIO:
        return cast(TextIO, open(path, mode, errors=errors))

    def gzip_open(self, path: str, mode: str = "r", *, errors: Optional[str] = None) -> TextIO:
        return cast(TextIO, gzip.open(path, mode, errors=errors))
