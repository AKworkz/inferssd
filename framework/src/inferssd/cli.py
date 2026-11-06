"""The inferssd command: serve, cli and profile."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from typing import Optional

from inferssd.app.command import add_commands as add_app_commands
from inferssd.app.command import run as run_app
from inferssd.machine.command import add_command as add_profile_command
from inferssd.machine.command import run as run_profile
from inferssd.version import line, report


def main(argv: Optional[Sequence[str]] = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    # The version, before any command: a line for people, or with --json every field for the app.
    if arguments and arguments[0] == "--version" and set(arguments[1:]) <= {"--json"}:
        value = report()
        print(json.dumps(value) if "--json" in arguments else line(value))
        return 0
    parser = argparse.ArgumentParser(description="Run large GGUF models from SSD")
    parser.add_argument(
        "--version",
        action="store_true",
        help="print the version and exit; --json gives every field",
    )
    parser.add_argument("--json", action="store_true", help=argparse.SUPPRESS)
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
