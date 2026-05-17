#!/usr/bin/env python3
"""
FILE STORY — ``run_dummy_smoke.py``
==================================

**Role.** Convenience: seed dummy idata + run short ``main.py --dummy-smoke``.

**Connects.** ``seed_dummy_smoke.py``, ``main.py``.

**Developed with Cursor AI.**
"""
from __future__ import annotations  # Forward-compatible type annotations.

import argparse  # CLI for run id, epochs, seed params, and mode flags.
import os  # Pass dummy data root and continue-on-error to children.
import subprocess  # Chain seed script and ``main.py`` without shell.
import sys  # ``sys.executable`` for reproducible Python interpreter.
from pathlib import Path  # Resolve project, pipeline, and script paths.

PROJECT_ROOT = Path(__file__).resolve().parents[2]  # ``node_analysis/`` (two levels up from ``scripts/``).
PIPELINE = PROJECT_ROOT / "pipeline"  # Pipeline subtree path.
MAIN = PROJECT_ROOT / "main.py"  # Top-level training + graphs entry point.
SEED = PIPELINE / "scripts" / "seed_dummy_smoke.py"  # Synthetic idata seeding script.
DEFAULT_RUN_ID = "node_analysis_dummy"  # Canonical output folder for smoke runs.
DEFAULT_COHORT = PIPELINE / "cohorts" / "dummy_smoke_deveui.csv"  # Allowlist written by seed script.


def main() -> None:
    """Parse CLI, optionally seed data, invoke ``main.py`` with smoke-friendly defaults."""
    p = argparse.ArgumentParser(description="Dummy-data smoke test for node_analysis.")  # Smoke test CLI.
    p.add_argument("--run-id", type=str, default=DEFAULT_RUN_ID)  # Output run directory name.
    p.add_argument("--epochs", type=int, default=2)  # Short training for fast smoke (default 2).
    p.add_argument("--devices", type=int, default=8)  # Synthetic device count forwarded to seed script.
    p.add_argument("--days", type=int, default=120)  # Synthetic days per device forwarded to seed.
    p.add_argument("--seed", type=int, default=42)  # RNG seed forwarded to seed script.
    p.add_argument("--train-only", action="store_true")  # Train without figure post-steps.
    p.add_argument("--graphs-only", action="store_true", help="Skip seed/train; rebuild figures for --run-id.")  # Figures only.
    p.add_argument("--skip-survival", action="store_true", default=True)  # Default True: skip survival pass.
    p.add_argument(  # Opt-in to slower full training stack.
        "--full-stack",
        action="store_true",
        help="Include survival training pass (slower; not required for thesis figures).",
    )
    args = p.parse_args()  # Parse smoke-test arguments.

    env = os.environ.copy()  # Base environment for all subprocess calls.
    env["HOL_ACAD_DATA_ROOT"] = str(PROJECT_ROOT / "idata" / "dummy_smoke")  # Point children at synthetic idata.
    env.setdefault("HOL_ACAD_FIG_CONTINUE_ON_ERROR", "1")  # Do not fail entire smoke on one figure error.

    if not args.graphs_only:  # Full or train-only smoke needs fresh synthetic data.
        subprocess.check_call(  # Run seed script; raise on failure.
            [
                sys.executable,  # Same Python interpreter as this script.
                str(SEED),  # Path to ``seed_dummy_smoke.py``.
                "--devices",  # Forward device count flag name.
                str(int(args.devices)),  # Device count value as string.
                "--days",  # Forward days flag name.
                str(int(args.days)),  # Days value as string.
                "--seed",  # Forward RNG seed flag name.
                str(int(args.seed)),  # Seed value as string.
            ],
            env=env,  # Child sees dummy data root (seed writes under project idata).
        )

    main_argv = [sys.executable, str(MAIN), "--run-id", args.run_id.strip(), "--epochs", str(int(args.epochs))]  # Build ``main.py`` argv.
    main_argv += ["--cohort-csv", str(DEFAULT_COHORT)]  # Always use dummy smoke allowlist.
    if args.graphs_only:  # Rebuild figures for existing run artifacts.
        main_argv.append("--graphs-only")  # Skip training in ``main.py``.
    elif args.train_only:  # Train without post-graph steps.
        main_argv.append("--train-only")  # Skip figure companion steps.
    if not args.full_stack and args.skip_survival:  # Default: skip survival unless full stack requested.
        main_argv.append("--skip-survival")  # Faster smoke path.

    print("Running:", " ".join(main_argv), flush=True)  # Log exact command for debugging.
    subprocess.check_call(main_argv, cwd=str(PROJECT_ROOT), env=env)  # Run ``main.py`` from project root.


if __name__ == "__main__":  # Entry when executed as script.
    main()  # Run end-to-end smoke test.
