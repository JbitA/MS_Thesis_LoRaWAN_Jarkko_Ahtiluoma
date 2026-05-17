#!/usr/bin/env python3
"""
FILE STORY — ``discover_cohort.py``
===================================

**Role.** Scans ``idata/*/application.csv`` for devices with CO₂; writes ``pipeline/cohorts/ers_co2_max_available_deveui.csv``.

**Connects.** ``main.py --discover-cohort``.

**Developed with Cursor AI.**
"""
from __future__ import annotations  # Postponed annotations for forward-compatible typing.

import os  # Read ``HOL_ACAD_DATA_ROOT`` and pipeline root environment overrides.
from pathlib import Path  # Resolve idata root and output cohort CSV path.

import pandas as pd  # Load ``application.csv`` and emit sorted DevEUI allowlist.

PKG = Path(  # Pipeline package root (parent of ``scripts/``).
    os.environ.get(  # Prefer explicit pipeline root from environment.
        "HOL_ACAD_PIPELINE_ROOT",  # Primary env key for pipeline directory.
        os.environ.get("HOL_ACAD_TERTIARY_ROOT", Path(__file__).resolve().parents[1]),  # Fallback: script's grandparent.
    )
)


def idata_root() -> Path:
    """Resolve the sensor data root directory containing per-device ``application.csv`` files."""
    e = os.environ.get("HOL_ACAD_DATA_ROOT", "").strip()  # User-configured idata path string.
    if e:  # Non-empty env override present.
        p = Path(e)  # Convert to ``Path`` for existence checks.
        if p.is_dir():  # Use override only when it points at a real directory.
            return p  # Return configured data root.
    for cand in (PKG.parent.parent / "idata",):  # Default: ``node_analysis/idata`` relative to pipeline.
        if cand.is_dir():  # Use default when present (typical local checkout layout).
            return cand  # Return default idata directory.
    raise FileNotFoundError("Set HOL_ACAD_DATA_ROOT to the idata root (see idata/README.md).")  # No data found.


def main() -> None:
    """Walk idata, collect DevEUIs with valid CO2, write sorted cohort CSV."""
    devices: set[str] = set()  # Accumulate unique DevEUI strings across all application files.
    root = idata_root()  # Resolved sensor data root.
    for fp in sorted(root.glob("**/application.csv")):  # Every device export file under idata (sorted for stability).
        if fp.parent.name.startswith("_"):  # Skip hidden or scratch device folders (leading underscore).
            continue  # Do not ingest this device's CSV.
        df = pd.read_csv(fp, usecols=["deveui", "co2"])  # Load only columns needed for CO2 availability check.
        ok = pd.to_numeric(df["co2"], errors="coerce").notna()  # Rows where CO2 parsed to a finite number.
        dev = df.loc[ok, "deveui"].astype(str).str.strip()  # DevEUIs from rows with valid CO2, normalized strings.
        devices.update(d for d in dev.unique().tolist() if d)  # Add non-empty unique DevEUIs to the set.
    if not devices:  # Empty set means no qualifying devices in idata.
        raise RuntimeError("No ERS CO2 devices found.")  # Fail loudly so training does not run on empty cohort.
    out = PKG / "cohorts" / "ers_co2_max_available_deveui.csv"  # Canonical output allowlist path.
    out.parent.mkdir(parents=True, exist_ok=True)  # Ensure ``cohorts/`` directory exists.
    pd.DataFrame({"deveui": sorted(devices)}).to_csv(out, index=False)  # Write sorted single-column CSV (no index column).
    print(f"Wrote {len(devices)} devices -> {out}")  # Report cohort size and destination path.


if __name__ == "__main__":  # Script entry when run directly.
    main()  # Execute cohort discovery.
