#!/usr/bin/env python3
"""Minimal config-driven Stage 2 launcher for BAGEL, Lance, and Janus-Pro."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--mode", choices=("contract", "smoke", "train"), default="contract")
    parser.add_argument("--print-only", action="store_true")
    cli = parser.parse_args()
    command = [
        sys.executable, str(ROOT / "launch_stage2.py"),
        "--config", str(cli.config.resolve()), "--mode", cli.mode,
    ]
    if cli.print_only:
        command.append("--print-only")
    return subprocess.run(command, cwd=ROOT, check=True).returncode


if __name__ == "__main__":
    raise SystemExit(main())

