"""
cellstream.cli

Command Line Interface entry point for cellstream.
"""

import sys
import argparse
from importlib.metadata import version, PackageNotFoundError


def _get_version() -> str:
    try:
        return version("cellstream")
    except PackageNotFoundError:
        return "0.1.2-dev"


def main():
    parser = argparse.ArgumentParser(
        prog="cellstream",
        description="cellstream: PyTorch-accelerated single-cell signal processing tools",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"cellstream {_get_version()}",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # Subcommand: cellstream run <config.toml>
    run_parser = subparsers.add_parser(
        "run",
        help="Run a processing job pipeline from a TOML configuration file",
    )
    run_parser.add_argument(
        "config",
        help="Path to the TOML job configuration file",
    )
    run_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate configuration and display execution plan without running processing jobs",
    )

    args = parser.parse_args()

    if args.command == "run":
        from .runner import run_pipeline
        try:
            run_pipeline(args.config, dry_run=args.dry_run)
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"Error executing pipeline: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
