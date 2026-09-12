"""CLI entry point for the Limen profile compiler.

Usage:
    python -m limen.profile compile examples/controller_v0.json --out out/profile
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .schema import PartnerProfile, ProfileValidationError
from .compiler import ProfileCompiler


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="limen.profile",
        description="Compile a Limen partner profile into firmware, RTL, and test targets",
    )
    sub = parser.add_subparsers(dest="command")
    compile_p = sub.add_parser("compile", help="compile a profile JSON to all targets")
    compile_p.add_argument("profile", type=Path, help="path to profile JSON file")
    compile_p.add_argument("--out", type=Path, default=Path("out"), help="output directory")
    args = parser.parse_args(argv)

    if args.command == "compile":
        try:
            profile = PartnerProfile.from_json_file(args.profile)
        except (FileNotFoundError, ProfileValidationError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        artifacts = ProfileCompiler().compile(profile)
        written = artifacts.write(args.out)
        for path in written:
            print(f"wrote {path}")
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
