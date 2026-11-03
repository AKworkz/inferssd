"""The error raised when an input is outside its explicit contract."""

from __future__ import annotations


class ContractError(ValueError):
    """An artifact is outside this version's explicit input contract."""
