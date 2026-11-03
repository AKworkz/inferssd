"""Import-safe framework components; importing this package performs no probes."""

from __future__ import annotations

from importlib import metadata

try:
    # The version of the installed package, the one framework/pyproject.toml names.
    __version__ = metadata.version("inferssd")
except metadata.PackageNotFoundError:  # a source tree on sys.path that was never installed
    __version__ = "0+unknown"
