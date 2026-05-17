#!/usr/bin/env python3
"""
FILE STORY — ``run_companion_tables.py``
========================================

**Role.** Subprocess wrapper for ``generate_figure_companion_tables.py`` (metrics companions).

**Connects.** ``main.py`` after ``run_figures.py``.

**Developed with Cursor AI.**
"""
from __future__ import annotations

import os  # Environment copy and run-id lookup.
import subprocess  # Run companion table generator as subprocess.
import sys  # ``sys.path`` and ``sys.executable``.
from pathlib import Path  # Pipeline root resolution.

PIPELINE = Path(os.environ.get("HOL_ACAD_PIPELINE_ROOT", os.environ.get("HOL_ACAD_TERTIARY_ROOT", Path(__file__).resolve().parents[1])))  # Pipeline directory (env or relative to script).


def main() -> None:
    """Apply run env and call ``generate_figure_companion_tables.py``."""
    sys.path.insert(0, str(PIPELINE / "src"))  # Enable ``node_analysis_pipeline`` imports.
    from node_analysis_pipeline.layout import apply_run_env, run_id  # noqa: WPS433  # Deferred import after path insert.

    rid = (os.environ.get("HOL_ACAD_RUN_ID") or run_id()).strip()  # Resolve run id from env or layout helper.
    apply_run_env(rid)  # Publish run-scoped output paths to environment variables.
    env = os.environ.copy()  # Mutable env dict for child process.
    env["PYTHONPATH"] = str(PIPELINE / "src")  # Child imports pipeline modules from ``src``.
    env.setdefault("HOL_ACAD_COMPANION_ALLOW_NO_RECREATED", "1")  # Allow partial figure sets during companion gen.
    gen = PIPELINE / "figures" / "generate_figure_companion_tables.py"  # Companion metrics driver script path.
    subprocess.check_call([sys.executable, str(gen)], cwd=str(PIPELINE / "figures"), env=env)  # Run generator; fail on error.


if __name__ == "__main__":  # Direct script execution guard.
    main()  # Run companion table generation.
