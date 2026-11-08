"""Whether this host's C math library computes the floats the reference transcripts hold.

The transcripts hold log2 and pow results as glibc and Apple's libm compute them, and
Linux and macOS reproduce them bit for bit. The Windows CRT rounds some of them one or
two bits differently, so on Windows a frozen reference cannot reproduce its recording
and the product is held to the frozen reference run on this host instead.
"""

from __future__ import annotations

import sys

import pytest

RECORDED_LIBM = sys.platform != "win32"
needs_recorded_libm = pytest.mark.skipif(
    not RECORDED_LIBM,
    reason="recorded with glibc and Apple libm; the Windows CRT rounds log2 and pow apart",
)
