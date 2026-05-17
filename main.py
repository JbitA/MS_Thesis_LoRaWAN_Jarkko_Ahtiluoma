#!/usr/bin/env python3
"""
**Role.** Sensor node analysis: one command trains
all model passes on a sensor cohort, then rebuilds eleven figures and
companion metric tables for a single ``run_id``.

**Inputs.** CLI flags (epochs, run id, cohort CSV, smoke/discover flags); optional
campus data under ``idata/``; environment may pre-set ``NODE_ANALYSIS_*``.

**Outputs.** Everything under ``outputs/<run_id>/``:

  - ``models/tables/`` — CSVs named ``node_analysis_<run_id>_<pass>_…`` (training)
  - ``models/reports/`` — orchestrator markdown report
  - ``graphs/<figure_key>/outputs/`` — thesis PNGs + companion CSV/MD (figures phase)

**Execution order (normal run).**

  1. Resolve ``run_id`` (CLI → env → auto UTC stamp).
  2. :func:`_configure_env` — mkdir outputs, set ``PYTHONPATH``, dual-write env.
  3. Optional ``discover_cohort`` script.
  4. Subprocess: ``cohort_training_orchestrator.py`` (training only; passes ``--skip-figures``).
  5. Subprocess: ``run_figures.py`` then ``run_companion_tables.py``.

**Developed by Jarkko Ahtiluoma 17.5.2026 with Cursor AI.**
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent  # node_analysis/ repository root
PIPELINE = PROJECT_ROOT / "pipeline"  # Scripts, cohorts, figures, src/
SRC = PIPELINE / "src"  # node_analysis_pipeline package root for PYTHONPATH
ORCH = SRC / "node_analysis_pipeline" / "orchestration" / "cohort_training_orchestrator.py"


def _utc_run_id() -> str:
    """Filesystem-safe UTC timestamp when the user omits ``--run-id``."""
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _configure_env(run_id: str) -> dict[str, str]:
    """Insert ``SRC`` on path, create run dirs, return env dict for subprocesses."""
    sys.path.insert(0, str(SRC))
    from node_analysis_pipeline.layout import (  # noqa: WPS433
        apply_run_env,
        run_checkpoints_dir,
        run_graphs_dir,
        run_models_dir,
        run_records_dir,
        run_reports_dir,
        run_root,
        run_tables_dir,
    )

    apply_run_env(run_id)  # Dual-write NODE_ANALYSIS_* and HOL_ACAD_* path keys
    run_root(run_id).mkdir(parents=True, exist_ok=True)
    run_graphs_dir(run_id).mkdir(parents=True, exist_ok=True)
    run_models_dir(run_id).mkdir(parents=True, exist_ok=True)
    run_tables_dir(run_id).mkdir(parents=True, exist_ok=True)
    run_reports_dir(run_id).mkdir(parents=True, exist_ok=True)
    run_checkpoints_dir(run_id).mkdir(parents=True, exist_ok=True)
    run_records_dir(run_id).mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["PYTHONPATH"] = str(SRC)  # Children import node_analysis_pipeline
    data = PROJECT_ROOT / "idata"
    if data.is_dir():
        env.setdefault("NODE_ANALYSIS_DATA_ROOT", str(data))  # Default campus telemetry root
        env.setdefault("HOL_ACAD_DATA_ROOT", str(data))  # Legacy trainer alias
    return env


def _run_discover_cohort(env: dict[str, str]) -> None:
    """Rebuild ``pipeline/cohorts/ers_co2_max_available_deveui.csv`` from idata scan."""
    subprocess.check_call([sys.executable, str(PIPELINE / "scripts" / "discover_cohort.py")], env=env)


def _run_training(env: dict[str, str], argv: list[str]) -> int:
    """Launch cohort orchestrator; return exit code (non-zero = training failure)."""
    return subprocess.call([sys.executable, str(ORCH), *argv], cwd=str(SRC), env=env)


def _run_post_graphs(env: dict[str, str]) -> None:
    """Run thesis figure build and companion tables after successful training."""
    for name, script in (
        ("thesis figures", PIPELINE / "scripts" / "run_figures.py"),
        ("companion tables", PIPELINE / "scripts" / "run_companion_tables.py"),
    ):
        if script.is_file():
            print(f"Post-step: {name}...", flush=True)
            subprocess.check_call([sys.executable, str(script)], env=env)


def main() -> None:
    p = argparse.ArgumentParser(description="Train models and emit thesis graphs in one run.")
    p.add_argument("--epochs", type=int, default=12)  # Forwarded to orchestrator → trainer
    p.add_argument("--run-id", type=str, default="")  # Empty → env or auto node_analysis_<UTC>
    p.add_argument("--resume-only", action="store_true")  # Load checkpoints; skip full retrain
    p.add_argument("--skip-figures", action="store_true")  # Legacy alias for skipping post-steps
    p.add_argument("--graphs-only", action="store_true")  # Figures/companions only; no training
    p.add_argument("--discover-cohort", action="store_true")  # Refresh DevEUI allowlist before train
    p.add_argument("--train-only", action="store_true", help="Skip graph post-steps after training.")
    p.add_argument(
        "--dummy-smoke",
        action="store_true",
        help="Use synthetic idata/dummy_smoke (runs seed script; default run-id node_analysis_dummy).",
    )
    p.add_argument("--cohort-csv", type=str, default="", help="DevEUI allowlist CSV for orchestrator.")
    p.add_argument("--skip-survival", action="store_true", help="Skip survival training pass (faster smoke).")
    args, orch_extra = p.parse_known_args()  # Unknown flags forwarded to orchestrator

    if args.dummy_smoke and not args.graphs_only:
        subprocess.check_call(
            [sys.executable, str(PIPELINE / "scripts" / "seed_dummy_smoke.py")],
            cwd=str(PROJECT_ROOT),
        )
        dummy_root = str(PROJECT_ROOT / "idata" / "dummy_smoke")
        os.environ["NODE_ANALYSIS_DATA_ROOT"] = dummy_root
        os.environ["HOL_ACAD_DATA_ROOT"] = dummy_root

    run_id = (
        args.run_id
        or os.environ.get("NODE_ANALYSIS_RUN_ID")
        or os.environ.get("HOL_ACAD_RUN_ID")
        or f"node_analysis_{_utc_run_id()}"
    ).strip().replace(":", "_")
    if args.dummy_smoke and not args.run_id:
        run_id = "node_analysis_dummy"  # Fixed folder name for local smoke tests

    env = _configure_env(run_id)
    if args.dummy_smoke:
        dummy_root = str(PROJECT_ROOT / "idata" / "dummy_smoke")
        env["NODE_ANALYSIS_DATA_ROOT"] = dummy_root
        env["HOL_ACAD_DATA_ROOT"] = dummy_root
        env.setdefault("NODE_ANALYSIS_FIG_CONTINUE_ON_ERROR", "1")  # Tolerate single-figure failures in smoke
        env.setdefault("HOL_ACAD_FIG_CONTINUE_ON_ERROR", "1")

    print(f"Run id: {run_id}", flush=True)
    print(f"Outputs: {PROJECT_ROOT / 'outputs' / run_id}", flush=True)

    if args.discover_cohort:
        _run_discover_cohort(env)

    if args.graphs_only:
        _run_post_graphs(env)
        return

    orch_argv: list[str] = ["--skip-figures"]  # Figures run here in main, not inside orchestrator
    if args.resume_only:
        orch_argv.append("--resume-only")
    orch_argv += ["--epochs", str(args.epochs), "--run-id", run_id]
    if str(args.cohort_csv).strip():
        orch_argv += ["--cohort-csv", str(args.cohort_csv).strip()]
    elif args.dummy_smoke:
        orch_argv += ["--cohort-csv", str(PIPELINE / "cohorts" / "dummy_smoke_deveui.csv")]
    if args.skip_survival or args.dummy_smoke:
        orch_argv.append("--skip-survival")
    orch_argv += orch_extra

    rc = _run_training(env, orch_argv)
    if rc != 0:
        raise SystemExit(rc)

    if not args.skip_figures and not args.train_only:
        _run_post_graphs(env)


if __name__ == "__main__":
    main()
