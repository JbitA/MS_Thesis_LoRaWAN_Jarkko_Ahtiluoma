"""
FILE STORY — ``paths.py``
=========================

**Role.** Resolves vendored trainer script path, default ``idata/``, and ``tables/`` directory.

**Connects.** ``trainer_subprocess.py``, ``cohort_training_orchestrator.py``.

**Does not** define figure PNG names (see ``figure_titles.py``).

**Developed with Cursor AI.**
"""

from __future__ import annotations  # Enable modern union syntax in type hints

import os  # Available for future env reads; primary access via env_config
from pathlib import Path  # Absolute paths to trainer, idata, and tables
from typing import Tuple  # Return type for paths() tuple of roots

from node_analysis_pipeline import env_config as env  # OUTPUT_ROOT and DATA_ROOT lookup
from node_analysis_pipeline.layout import pipeline_root, project_root, run_tables_dir  # Layout helpers

_PKG_DIR = Path(__file__).resolve().parent  # node_analysis_pipeline package directory
_SRC_DIR = _PKG_DIR.parent  # pipeline/src
_PIPELINE_ROOT = pipeline_root()  # pipeline/ (figures, runtime_vendor) — may follow env
_PROJECT_ROOT = project_root()  # node_analysis/ repo root
_BUNDLE_ROOT = _PIPELINE_ROOT  # Historical name: bundle lives under pipeline/
_THESIS_PIPELINE = _PIPELINE_ROOT  # Alias used by orchestrator imports
_REPO_ROOT = _PROJECT_ROOT  # Alias for report text and path tuples
_VENDORED_SRC = _PIPELINE_ROOT / "runtime_vendor" / "src"  # Forked trainer + telemetry code


def paths() -> Tuple[Path, Path, Path, Path]:  # Return package, bundle, pipeline, repo roots
    return _PKG_DIR, _BUNDLE_ROOT, _THESIS_PIPELINE, _REPO_ROOT  # Tuple for callers needing all roots


def trainer_script() -> Path:  # Vendored multitask_battery_halt_trainer_bundle.py path
    path = _VENDORED_SRC / "multitask_battery_halt_trainer_bundle.py"  # Canonical trainer entry
    if path.is_file():  # Ensure vendored copy exists before subprocess spawn
        return path
    raise FileNotFoundError(f"Trainer not found: {path}")  # Fail fast with explicit path


def default_tables_dir() -> Path:  # Resolve comparison CSV directory from OUTPUT_ROOT
    """Directory containing ``*_model_comparison_battery_halt.csv`` exports."""
    root = env.get_str(env.ENV_OUTPUT_ROOT, "")  # models/ root from apply_run_env
    if root:  # When orchestrator set OUTPUT_ROOT, tables are always models/tables
        return Path(root) / "tables"
    return run_tables_dir()  # Fallback: derive from run_id and project layout


def default_idata_dir() -> Path:  # LoRaWAN idata root for telemetry loading
    """LoRaWAN telemetry root (``<deveui>/application.csv`` per device)."""
    root = env.get_str(env.ENV_DATA_ROOT, "")  # Explicit idata override
    if root:  # User or discover-cohort set DATA_ROOT
        return Path(root)
    cand = _PROJECT_ROOT / "idata"  # Default repo-relative telemetry tree
    if cand.is_dir():  # Prefer existing idata directory
        return cand
    return cand  # Return path even if missing so callers can surface clear errors
