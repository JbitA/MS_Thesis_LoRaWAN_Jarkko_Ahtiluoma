"""
FILE STORY — ``env_config.py``
==============================

**Role.** Single registry for process-environment keys used across training,
figure building, and CLI entry points. Every path and run id should be read
through :func:`get_str`, :func:`get_path`, or :func:`set_both` here—not by
hard-coding ``HOL_ACAD_*`` in new code.

**Why two name families.** Historical bundles used ``HOL_ACAD_*``; this repo
standardizes on ``NODE_ANALYSIS_*``. :func:`get_str` tries canonical first, then
legacy, so old shells and notebooks keep working during migration.

**Train → figures contract.** ``layout.apply_run_env(run_id)`` calls
:func:`set_both` for ``RUN_ID``, ``OUTPUT_ROOT`` (→ ``outputs/<id>/models``),
and ``GRAPH_ROOT`` (→ ``outputs/<id>/graphs``). Trainers and
``figures/build.py`` inherit the same values in subprocess env.

**This file does not** define CSV artifact stems (see ``artifact_stems.py``) or
create output directories (see ``layout.py``).

**Developed with Cursor AI.**
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TypeVar

T = TypeVar("T")

# --- Canonical environment key names (string constants only) -----------------

ENV_PROJECT_ROOT = "NODE_ANALYSIS_PROJECT_ROOT"  # Repo root: node_analysis/
ENV_PIPELINE_ROOT = "NODE_ANALYSIS_PIPELINE_ROOT"  # pipeline/ (figures, src, runtime_vendor)
ENV_RUN_ID = "NODE_ANALYSIS_RUN_ID"  # Shared id for tables + graphs under outputs/<id>/
ENV_OUTPUT_ROOT = "NODE_ANALYSIS_OUTPUT_ROOT"  # Usually outputs/<id>/models (trainer writes tables here)
ENV_GRAPH_ROOT = "NODE_ANALYSIS_GRAPH_ROOT"  # outputs/<id>/graphs (thesis figure folders)
ENV_FIG_SCAN_ROOT = "NODE_ANALYSIS_FIG_SCAN_ROOT"  # Where companion scripts scan for figure dirs
ENV_DATA_ROOT = "NODE_ANALYSIS_DATA_ROOT"  # Campus idata root (*/application.csv per DevEUI)
ENV_VC_SENSOR = "NODE_ANALYSIS_VC_SENSOR"  # Optional override for ribbon figure sensor selection
ENV_FIG_CONTINUE_ON_ERROR = "NODE_ANALYSIS_FIG_CONTINUE_ON_ERROR"  # If true, one failed figure does not abort batch
ENV_COMPANION_ALLOW_NO_RECREATED = "NODE_ANALYSIS_COMPANION_ALLOW_NO_RECREATED"  # Build companion CSVs without PNG present
ENV_SKIP_TRAINER_DIAGNOSTIC_FIGURES = (
    "NODE_ANALYSIS_SKIP_TRAINER_DIAGNOSTIC_FIGURES"
)  # If true, vendored trainer skips models/figures/*.png (thesis graphs live under graphs/)

_LEGACY_MAP: dict[str, str] = {
    ENV_PROJECT_ROOT: "HOL_ACAD_PROJECT_ROOT",
    ENV_PIPELINE_ROOT: "HOL_ACAD_PIPELINE_ROOT",
    ENV_RUN_ID: "HOL_ACAD_RUN_ID",
    ENV_OUTPUT_ROOT: "HOL_ACAD_OUTPUT_ROOT",
    ENV_GRAPH_ROOT: "HOL_ACAD_GRAPH_ROOT",
    ENV_FIG_SCAN_ROOT: "HOL_ACAD_FIG_SCAN_ROOT",
    ENV_DATA_ROOT: "HOL_ACAD_DATA_ROOT",
    ENV_VC_SENSOR: "HOL_ACAD_VC_SENSOR",
    ENV_FIG_CONTINUE_ON_ERROR: "HOL_ACAD_FIG_CONTINUE_ON_ERROR",
    ENV_COMPANION_ALLOW_NO_RECREATED: "HOL_ACAD_COMPANION_ALLOW_NO_RECREATED",
    ENV_SKIP_TRAINER_DIAGNOSTIC_FIGURES: "HOL_ACAD_SKIP_TRAINER_DIAGNOSTIC_FIGURES",
}

_LEGACY_PIPELINE_ALT = "HOL_ACAD_TERTIARY_ROOT"  # Older name for pipeline/ in some bundles


def get_str(canonical: str, default: str = "") -> str:
    """Return trimmed env value: canonical name first, then legacy alias."""
    val = os.environ.get(canonical, "").strip()  # Prefer NODE_ANALYSIS_* when set
    if val:
        return val
    legacy = _LEGACY_MAP.get(canonical)  # Paired HOL_ACAD_* key, if any
    if legacy:
        val = os.environ.get(legacy, "").strip()  # Backward-compatible read path
    return val if val else default


def get_path(canonical: str, default: Path | None = None) -> Path | None:
    """Resolve an environment variable to an absolute :class:`Path`."""
    raw = get_str(canonical, "")
    if not raw:
        return default  # Caller-supplied fallback when env unset
    return Path(raw).resolve()  # Absolute path for stable joins on Windows/Linux


def set_both(canonical: str, value: str) -> None:
    """Write canonical and legacy env keys to the same value."""
    os.environ[canonical] = value  # New tooling reads NODE_ANALYSIS_*
    legacy = _LEGACY_MAP.get(canonical)
    if legacy:
        os.environ[legacy] = value  # Vendored trainer still accepts HOL_ACAD_*


def pipeline_root_from_env(*, fallback: Path) -> Path:
    """Resolve pipeline root from env (canonical → legacy → fallback)."""
    for key in (ENV_PIPELINE_ROOT, _LEGACY_PIPELINE_ALT, "HOL_ACAD_PIPELINE_ROOT"):
        raw = os.environ.get(key, "").strip()  # First non-empty wins
        if raw:
            return Path(raw).resolve()
    return fallback.resolve()  # Derived from import path when env not configured


def truthy(canonical: str) -> bool:
    """Interpret env as boolean (1/true/yes/on)."""
    return get_str(canonical, "").lower() in ("1", "true", "yes", "on")
