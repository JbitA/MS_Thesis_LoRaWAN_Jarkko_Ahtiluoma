"""
FILE STORY — ``pipeline_assets.py``
===================================

**Role.** Path constants for bundled CSV assets (battery missingness reference, ribbon sensor hint).

**Connects.** ``figures/util.py``, ``seed_dummy_smoke.py``.

**Does not** participate in model training.

**Developed with Cursor AI.**
"""

from __future__ import annotations  # Consistent with rest of pipeline (Path types in constants).

from pathlib import Path  # Resolve asset directory relative to this module file.

_ASSETS_DIR = Path(__file__).resolve().parent  # Directory containing this module and bundled CSV files.

BATTERY_MISSINGNESS_REFERENCE_CSV = _ASSETS_DIR / "battery_missingness_reference_series.csv"  # Path to battery missingness reference series.
RIBBON_CHART_DEFAULT_SENSOR_CSV = _ASSETS_DIR / "ribbon_chart_default_sensor.csv"  # Path to default ribbon-chart sensor hint CSV.
