"""
FILE STORY — ``layout.py``
==========================

**Role.** Maps the active ``run_id`` to concrete directories under
``outputs/<run_id>/``. :func:`apply_run_env` is the bridge between CLI/orchestrator
and every subprocess (trainer, ``figures/build.py``).

**Directory tree created logically (mkdir in ``main._configure_env``).**

::

    outputs/<run_id>/
      models/
        tables/       ← trainer CSV exports (figures read these)
        checkpoints/  ← optional .pt saves (--resume-only)
        reports/      ← FULL_STACK_REPORT, per-pass trainer MD, figure analysis MD
        records/      ← JSON audit log per training pass (orchestrator)
      graphs/
        <figure_key>/outputs/  ← thesis PNG + companion artifacts

**Must run before training or figures** so ``OUTPUT_ROOT`` and ``GRAPH_ROOT`` match.

**Developed with Cursor AI.**
"""

from __future__ import annotations

import os
from pathlib import Path

from node_analysis_pipeline import env_config as env

_LAYOUT_FILE = Path(__file__).resolve()
_SRC_DIR = _LAYOUT_FILE.parents[1]  # pipeline/src
_PIPELINE_ROOT = _SRC_DIR.parent  # pipeline/
_PROJECT_ROOT = _PIPELINE_ROOT.parent  # node_analysis/


def project_root() -> Path:
    """Repository root (``node_analysis/``)."""
    resolved = env.get_path(env.ENV_PROJECT_ROOT)
    return resolved if resolved else _PROJECT_ROOT


def pipeline_root() -> Path:
    """``node_analysis/pipeline/`` — figures, cohorts, runtime_vendor, src."""
    return env.pipeline_root_from_env(fallback=_PIPELINE_ROOT)


def tertiary_root() -> Path:
    """Deprecated alias for :func:`pipeline_root` (legacy bundle naming)."""
    return pipeline_root()


DEFAULT_RUN_ID = "node_analysis1"  # Used when NODE_ANALYSIS_RUN_ID unset


def run_id() -> str:
    """Active run identifier (tables + graphs share this id)."""
    return (env.get_str(env.ENV_RUN_ID, DEFAULT_RUN_ID) or DEFAULT_RUN_ID).strip()


def run_root(rid: str | None = None) -> Path:
    rid = rid or run_id()
    return project_root() / "outputs" / rid  # Per-run sandbox


def run_models_dir(rid: str | None = None) -> Path:
    """Training artifacts root: tables, checkpoints, records, reports."""
    return run_root(rid) / "models"


def run_graphs_dir(rid: str | None = None) -> Path:
    """Thesis figure folders: ``<graphs>/<figure_key>/outputs/``."""
    return run_root(rid) / "graphs"


def run_tables_dir(rid: str | None = None) -> Path:
    return run_models_dir(rid) / "tables"  # All *_model_comparison_* and *_predictions_* CSVs


def run_reports_dir(rid: str | None = None) -> Path:
    return run_models_dir(rid) / "reports"


def run_checkpoints_dir(rid: str | None = None) -> Path:
    return run_models_dir(rid) / "checkpoints"


def run_records_dir(rid: str | None = None) -> Path:
    return run_models_dir(rid) / "records"


def apply_run_env(rid: str) -> None:
    """
    Set process environment for trainers and figure builders.

    Writes both ``NODE_ANALYSIS_*`` (canonical) and ``HOL_ACAD_*`` (legacy) keys.
    """
    pl = pipeline_root()
    models = run_models_dir(rid)  # Trainer OUTPUT_ROOT → models/
    graphs = run_graphs_dir(rid)  # Figure GRAPH_ROOT → graphs/
    env.set_both(env.ENV_PROJECT_ROOT, str(project_root()))
    env.set_both(env.ENV_PIPELINE_ROOT, str(pl))
    env.set_both(env.ENV_RUN_ID, rid)  # Embeds into artifact_stem(run_id, …)
    env.set_both(env.ENV_OUTPUT_ROOT, str(models))
    env.set_both(env.ENV_GRAPH_ROOT, str(graphs))
    env.set_both(env.ENV_FIG_SCAN_ROOT, str(graphs))  # Companions scan same tree
    env.set_both(env.ENV_SKIP_TRAINER_DIAGNOSTIC_FIGURES, "1")  # No models/figures/*.png (thesis PNGs only)
    os.environ.setdefault("PYTHONUTF8", "1")  # Readable trainer logs on Windows
