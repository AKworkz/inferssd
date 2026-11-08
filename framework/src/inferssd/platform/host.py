# SPDX-License-Identifier: Apache-2.0
"""This host's owned-execution mechanisms, POSIX or Windows, chosen once at import.

The shared controller, guardian and worker algorithms use only these names;
each platform provides them with the same contracts.
"""

from __future__ import annotations

import sys

if sys.platform == "win32":
    from inferssd.platform.windows.lease import (
        Lease,
        acquire_lock,
        adopt,
        borrow_lock,
        descriptor,
        link,
        replace,
        watch,
    )
    from inferssd.platform.windows.process import (
        STOP_SIGNALS,
        group_exists,
        spawn_detached,
        spawn_owned,
        stop_owned,
    )
    from inferssd.platform.windows.resources import Resources
else:
    from inferssd.platform.lease import Lease
    from inferssd.platform.posix import (
        STOP_SIGNALS,
        acquire_lock,
        adopt,
        borrow_lock,
        descriptor,
        link,
        replace,
        spawn_detached,
        spawn_owned,
        watch,
    )
    from inferssd.platform.process import group_exists, stop_owned
    from inferssd.platform.resources import Resources

__all__ = [
    "STOP_SIGNALS",
    "Lease",
    "Resources",
    "acquire_lock",
    "adopt",
    "borrow_lock",
    "descriptor",
    "group_exists",
    "link",
    "replace",
    "spawn_detached",
    "spawn_owned",
    "stop_owned",
    "watch",
]
