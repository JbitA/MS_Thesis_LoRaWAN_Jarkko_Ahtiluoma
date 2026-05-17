#!/usr/bin/env python3
"""
FILE STORY — ``run_e2e_clean.py``
================================

**Role.** Optional end-to-end smoke driver (legacy run id ``e2e_clean``); not used by canonical ``node_analysis_200trial``.

**Connects.** ``main.py`` pattern; safe to ignore for thesis archive runs.

**Developed with Cursor AI.**
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MAIN = PROJECT_ROOT / "main.py"


def main() -> None:
    p = argparse.ArgumentParser(description="Clean E2E run inside node_analysis only.")
    p.add_argument("--run-id", type=str, default="e2e_clean")
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--devices", type=int, default=50)
    p.add_argument("--days", type=int, default=100)
    p.add_argument("--releases", type=int, default=50)
    p.add_argument("--skip-survival", action="store_true")
    args = p.parse_args()

    cmd = [
        sys.executable,
        str(MAIN),
        "--seed-idata",
        "--seed-devices",
        str(int(args.devices)),
        "--seed-days",
        str(int(args.days)),
        "--seed-releases",
        str(int(args.releases)),
        "--run-id",
        args.run_id.strip(),
        "--epochs",
        str(int(args.epochs)),
    ]
    if args.skip_survival:
        cmd.append("--skip-survival")

    print("Running:", " ".join(cmd), flush=True)
    subprocess.check_call(cmd, cwd=str(PROJECT_ROOT))


if __name__ == "__main__":
    main()
