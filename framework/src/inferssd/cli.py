"""The inferssd command: serve, cli and profile."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from typing import Optional

from inferssd import __version__
from inferssd.app.command import add_commands as add_app_commands
from inferssd.app.command import run as run_app
from inferssd.machine.command import add_command as add_profile_command
from inferssd.machine.command import run as run_profile


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Run large GGUF models from SSD")
    parser.add_argument("--version", action="version", version=f"inferssd {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)
    add_app_commands(commands)
    add_profile_command(commands)
    args = parser.parse_args(argv)
    if args.command == "profile":
        try:
            return run_profile(args)
        except (OSError, ValueError, RuntimeError) as error:
            parser.error(str(error))
    try:
        return run_app(args)
    except KeyboardInterrupt:
        return 130
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(2, f"inferssd: {error}\n")
