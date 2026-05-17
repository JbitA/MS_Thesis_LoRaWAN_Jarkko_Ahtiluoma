"""
FILE STORY — ``figures/util.py``
================================

**Role.** Small shared layer for all thesis figure builders: resolve where the
pipeline bundle lives, where bundled CSV assets sit, and where per-figure PNG/
companion files should be written for the active run.

**Consumers.** ``build.py``, ``recreated_from_new_data.py``,
``generate_figure_companion_tables.py``.

**Path rule.** When ``NODE_ANALYSIS_GRAPH_ROOT`` is set (via ``layout.apply_run_env``),
outputs go to ``<GRAPH_ROOT>/<figure_key>/outputs/``. Otherwise a dev fallback
writes beside ``pipeline/figures/<figure_key>/outputs/``.

**Bundled assets (read-only).**

  - ``battery_missingness_reference_series.csv`` — fixed legacy battery trace
  - ``ribbon_chart_default_sensor.csv`` — default DevEUI hint for ribbon figure

**Does not** read training tables (figures use ``ENV_OUTPUT_ROOT`` / ``tables/`` directly).

**Developed with Cursor AI.**
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

_FIGURES_DIR = Path(__file__).resolve().parent  # pipeline/figures package directory
_BUNDLE_ROOT = _FIGURES_DIR.parent  # pipeline/ when env does not override
_SRC = _BUNDLE_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from node_analysis_pipeline import env_config as env  # noqa: E402

BUNDLE_ROOT = env.pipeline_root_from_env(fallback=_BUNDLE_ROOT)  # Active pipeline root
PACKAGE_ASSETS = BUNDLE_ROOT / "assets"  # Shipped CSV assets (not run-specific)
SRC_ON_PATH = BUNDLE_ROOT / "src"  # node_analysis_pipeline import root
BATTERY_MISSINGNESS_REFERENCE_CSV = PACKAGE_ASSETS / "battery_missingness_reference_series.csv"
RIBBON_CHART_DEFAULT_SENSOR_CSV = PACKAGE_ASSETS / "ribbon_chart_default_sensor.csv"
THESIS_PIPELINE = BUNDLE_ROOT  # Legacy alias for older figure scripts
REPO_ROOT = BUNDLE_ROOT  # Legacy alias


def ensure_out(figure_key: str | Path) -> Path:
    """Create and return ``graphs/<figure_key>/outputs/`` for the active run."""
    key = figure_key.name if isinstance(figure_key, Path) else str(figure_key)
    graph_root = env.get_str(env.ENV_GRAPH_ROOT, "")
    if graph_root:
        out = Path(graph_root) / key / "outputs"  # Thesis layout under outputs/<run_id>/graphs/
    else:
        out = _FIGURES_DIR / key / "outputs"  # Local fallback next to builders
    out.mkdir(parents=True, exist_ok=True)
    return out


def copy_if_exists(src: Path, dst: Path) -> bool:
    if not src.is_file():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)  # Preserve mtime when mirroring assets into figure folder
    return True


def run_py(cwd: Path, script: Path, args: list[str] | None = None) -> None:
    cmd = [sys.executable, str(script)] + (args or [])
    subprocess.check_call(cmd, cwd=str(cwd))  # Legacy helper to spawn figure-related scripts
