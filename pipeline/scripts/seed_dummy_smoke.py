#!/usr/bin/env python3
"""
FILE STORY — ``seed_dummy_smoke.py``
====================================

**Role.** Creates synthetic ``idata/dummy_smoke/`` devices and seeds bundled asset CSVs for smoke tests.

**Connects.** ``main.py --dummy-smoke``, ``run_dummy_smoke.py``.

**Developed with Cursor AI.**
"""
from __future__ import annotations  # Enable modern type-hint syntax without runtime quotes.

import argparse  # CLI: device count, days, RNG seed.
import shutil  # Remove prior dummy idata tree before re-seeding.
from pathlib import Path  # Portable paths for idata, cohort, and assets.

import numpy as np  # Synthetic noise and clipping for battery/CO2 series.
import pandas as pd  # Write CSV uplinks and cohort allowlist.

PKG = Path(__file__).resolve().parents[1]  # ``pipeline/`` directory (parent of ``scripts/``).
PROJECT_ROOT = PKG.parent  # ``node_analysis/`` project root.
IDATA_ROOT = PROJECT_ROOT / "idata" / "dummy_smoke"  # Synthetic per-device data tree root.
COHORT_CSV = PKG / "cohorts" / "dummy_smoke_deveui.csv"  # DevEUI allowlist for dummy training runs.
ASSETS = PKG / "assets"  # Bundled figure reference CSV directory.
BATTERY_MISSINGNESS_REFERENCE_CSV = ASSETS / "battery_missingness_reference_series.csv"  # Figure trajectory asset path.


def _device_ids(n: int) -> list[str]:
    """Return ``n`` synthetic LoRaWAN DevEUI strings with deterministic suffix pattern."""
    return [f"aa-11-22-33-44-55-00-ff-fe-00-00-{i:02d}" for i in range(1, n + 1)]  # One hex-style id per device index.


def _write_application_csv(path: Path, deveui: str, *, days: int, rng: np.random.Generator) -> None:
    """Write one device's ``application.csv`` with daily synthetic uplinks."""
    t0 = pd.Timestamp("2022-01-01", tz="UTC")  # Fixed epoch for reproducible time series.
    rows: list[dict[str, object]] = []  # Accumulate uplink row dicts before DataFrame export.
    base_batt = 3.65 + float(rng.uniform(-0.05, 0.05))  # Per-device base battery voltage (V).
    for day_off in range(days):  # One calendar day per iteration.
        batt = base_batt - 0.0015 * day_off + float(rng.normal(0.0, 0.01))  # Slow drain + daily noise.
        co2 = 450.0 + float(rng.normal(0.0, 30.0))  # Random walk around nominal indoor CO2 (ppm).
        n_uplink = int(rng.integers(3, 8))  # Random uplink count per day (inclusive lower, exclusive upper).
        for k in range(n_uplink):  # Spread uplinks across the day.
            ts = t0 + pd.Timedelta(days=day_off, hours=int(k * (24 / max(n_uplink, 1))))  # Evenly spaced hours.
            rows.append(  # One LoRaWAN application uplink record.
                {
                    "time": int(ts.value // 10**6),  # Milliseconds since epoch (matches real export schema).
                    "deveui": deveui,  # Device identifier for this row.
                    "battery": round(float(np.clip(batt, 3.2, 4.0)), 4),  # Clipped Li-ion range, 4 decimal places.
                    "co2": round(float(max(co2, 300.0)), 2),  # Floor CO2 at 300 ppm, 2 decimal places.
                }
            )
    pd.DataFrame(rows).to_csv(path, index=False)  # Persist uplinks without pandas index column.


def _write_battery_missingness_reference(deveui: str, *, days: int, rng: np.random.Generator) -> None:
    """Write daily ground-truth battery series for the battery-missingness thesis figure."""
    t0 = pd.Timestamp("2022-01-01", tz="UTC")  # Align reference series start with application data.
    days_ix = pd.date_range(t0, periods=days, freq="D", tz="UTC")  # One UTC midnight per day.
    batt = 3.65 - 0.0015 * np.arange(days) + rng.normal(0.0, 0.01, size=days)  # Vectorized daily battery trajectory.
    df = pd.DataFrame(  # Columns expected by figure builder asset loader.
        {
            "sensor": deveui,  # DevEUI label for the reference sensor.
            "day": days_ix.strftime("%Y-%m-%d"),  # ISO date strings per row.
            "y_true_battery": np.clip(batt, 3.2, 4.0),  # Clipped true battery values for plotting.
        }
    )
    ASSETS.mkdir(parents=True, exist_ok=True)  # Create assets directory if missing.
    df.to_csv(BATTERY_MISSINGNESS_REFERENCE_CSV, index=False)  # Overwrite reference CSV used by figures only.


def main() -> None:
    """Parse CLI, validate constraints, seed idata + cohort + battery reference asset."""
    p = argparse.ArgumentParser(description="Seed dummy_smoke idata and cohort CSV.")  # CLI parser.
    p.add_argument("--devices", type=int, default=8, help="Number of synthetic LoRaWAN devices.")  # Default 8 devices.
    p.add_argument("--days", type=int, default=120, help="Days of uplinks per device (>= ~55 recommended).")  # Default 120 days.
    p.add_argument("--seed", type=int, default=42)  # RNG seed for reproducible smoke data.
    args = p.parse_args()  # Parse command-line arguments.

    if args.devices < 3:  # Training holdout splits need at least three devices.
        raise SystemExit("--devices must be >= 3 for sensor_holdout splits.")  # Exit with message.
    if args.days < 55:  # Model needs seq_len=21 plus halt horizon=30 days minimum.
        raise SystemExit("--days must be >= 55 (seq_len=21 + halt horizon=30).")  # Exit with message.

    rng = np.random.default_rng(int(args.seed))  # Seeded NumPy generator for all synthetic noise.
    devices = _device_ids(int(args.devices))  # Build list of synthetic DevEUIs.

    if IDATA_ROOT.exists():  # Prior dummy tree would mix old/new devices if left in place.
        shutil.rmtree(IDATA_ROOT)  # Delete entire dummy idata directory recursively.
    IDATA_ROOT.mkdir(parents=True, exist_ok=True)  # Recreate empty dummy idata root.

    for deveui in devices:  # One subdirectory per synthetic device.
        dev_dir = IDATA_ROOT / deveui  # Device-specific folder under dummy idata.
        dev_dir.mkdir(parents=True, exist_ok=True)  # Create device folder.
        _write_application_csv(dev_dir / "application.csv", deveui, days=int(args.days), rng=rng)  # Write uplink CSV.

    COHORT_CSV.parent.mkdir(parents=True, exist_ok=True)  # Ensure ``pipeline/cohorts/`` exists.
    pd.DataFrame({"deveui": devices}).to_csv(COHORT_CSV, index=False)  # Allowlist matching seeded devices.
    _write_battery_missingness_reference(devices[0], days=int(args.days), rng=rng)  # Reference series for first device.

    print(f"Wrote {len(devices)} devices under {IDATA_ROOT}")  # Confirm idata seeding.
    print(f"Wrote cohort -> {COHORT_CSV}")  # Confirm cohort CSV path.
    print(f"Wrote figure asset -> {BATTERY_MISSINGNESS_REFERENCE_CSV}")  # Confirm figure asset path.


if __name__ == "__main__":  # Direct execution entry point.
    main()  # Run seeding CLI.
