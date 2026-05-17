#!/usr/bin/env python3
"""
FILE STORY — ``run_figures.py``
==============================

**Role.** Subprocess wrapper: sets ``PYTHONPATH`` and runs ``figures/build.py`` for the active run.

**Connects.** ``main.py`` (post-training), ``layout.apply_run_env``.

**Developed with Cursor AI.**
"""
from __future__ import annotations  # Modern typing without quoting forward refs.

import os  # Copy environment and read run-id / continue-on-error flags.
import subprocess  # Spawn ``figures/build.py`` as a child process.
import sys  # Insert ``SRC`` on path and use ``sys.executable``.
from pathlib import Path  # Resolve pipeline, figures, and src directories.

PIPELINE = Path(os.environ.get("HOL_ACAD_PIPELINE_ROOT", os.environ.get("HOL_ACAD_TERTIARY_ROOT", Path(__file__).resolve().parents[1])))  # Pipeline root from env or script location.
FIG = PIPELINE / "figures"  # Thesis figure builders and ``build.py`` live here.
SRC = PIPELINE / "src"  # ``node_analysis_pipeline`` package root.


def main() -> None:
    """Apply run env, then run ``figures/build.py`` with optional continue-on-error."""
    sys.path.insert(0, str(SRC))  # Import layout helpers from ``node_analysis_pipeline``.
    from node_analysis_pipeline.layout import apply_run_env, run_id  # noqa: WPS433  # Lazy import after path fix.

    rid = (os.environ.get("HOL_ACAD_RUN_ID") or run_id()).strip()  # Active run id: env override or layout default.
    apply_run_env(rid)  # Set graph/model/table paths in environment for figure code.
    env = os.environ.copy()  # Child process inherits updated run paths.
    env["PYTHONPATH"] = str(SRC)  # Figure build subprocess can import pipeline package.
    build = FIG / "build.py"  # Master script that builds all thesis figure outputs.
    try:
        subprocess.check_call([sys.executable, str(build)], cwd=str(FIG), env=env)  # Run build; raise on nonzero exit.
    except subprocess.CalledProcessError as exc:  # Capture failure exit code from child.
        if os.environ.get("HOL_ACAD_FIG_CONTINUE_ON_ERROR", "").strip().lower() in ("1", "true", "yes"):  # Smoke-tolerant mode.
            print(f"Figure build failed (exit {exc.returncode}).", flush=True)  # Log and swallow error.
        else:
            raise  # Re-raise so CI / production runs fail hard.


if __name__ == "__main__":  # Script entry when invoked directly.
    main()  # Execute figure rebuild.
