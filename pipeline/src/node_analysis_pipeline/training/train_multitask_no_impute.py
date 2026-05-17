# !/usr/bin/env python3
"""
FILE STORY — ``train_multitask_no_impute.py``
=============================================

**Role.** No-impute training pass: monkey-patches ``telemetry_sequence_pipeline`` then runs trainer in-process.

**Outputs.** Tables with stem ``…_multitask_binary_no_impute`` (used by imputed vs non-imputed figures).

**Connects.** ``multitask_battery_halt_trainer_bundle.py``, ``telemetry_sequence_pipeline.py``.

**Developed with Cursor AI.**
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[2]
_BUNDLE_ROOT = _ROOT.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _install_no_impute_pipeline(src_dir: Path):
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    import telemetry_sequence_pipeline as telemetry  # noqa: WPS433

    def load_original_daily_no_impute(
        max_sensors: int | None = None,
        *,
        deveui_allowlist: List[str] | None = None,
    ) -> pd.DataFrame:
        daily = telemetry.aggregate_daily_panel()
        if deveui_allowlist is not None:
            allow = {str(x).strip() for x in deveui_allowlist if str(x).strip()}
            daily = daily[daily["deveui"].isin(allow)].copy()
        elif max_sensors is not None:
            keep = telemetry.top_sensors_by_uplink_volume(daily, max_sensors)
            daily = daily[daily["deveui"].isin(keep)].copy()

        rows: list[pd.DataFrame] = []
        for sid, g in daily.groupby("deveui", sort=False):
            g = g.sort_values("day").copy()
            full_days = pd.date_range(g["day"].min(), g["day"].max(), freq="D", tz="UTC")
            gx = g.set_index("day").reindex(full_days).rename_axis("day").reset_index()
            gx["deveui"] = sid
            gx["daily_count"] = gx["daily_count"].fillna(0)
            # Intentionally keep battery NaNs (no interpolation/fallback).
            gx["outage_flag"] = (gx["daily_count"] == 0).astype(float)
            gx["days_since_start"] = np.arange(len(gx), dtype=float)
            gx["cum_count"] = gx["daily_count"].cumsum()
            gx["roll7_count"] = gx["daily_count"].rolling(7, min_periods=1).mean()
            gx["batt_drop_from_start"] = (
                gx["battery_mean"].iloc[0] - gx["battery_mean"]
                if pd.notna(gx["battery_mean"].iloc[0])
                else np.nan
            )
            gx["batt_roll7"] = gx["battery_mean"].rolling(7, min_periods=1).mean()
            gx["batt_delta1"] = gx["battery_mean"].diff()
            gx["count_delta1"] = gx["daily_count"].diff().fillna(0.0)
            gx["count_trend7"] = (
                gx["daily_count"]
                .rolling(7, min_periods=2)
                .apply(lambda v: float(v.iloc[-1] - v.iloc[0]), raw=False)
                .fillna(0.0)
            )
            rows.append(gx)
        return pd.concat(rows, ignore_index=True)

    def build_sequences_no_impute(df: pd.DataFrame, seq_len: int, halt_horizon_days: int) -> telemetry.SeqData:
        feature_cols = list(telemetry.SEQ_INPUT_FEATURE_COLS)
        x_list, yb_list, yh_list, day_list, sid_list, day_label_list, dt_list, yc_list = (
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
        )
        for sid, g in df.groupby("deveui", sort=False):
            g = g.sort_values("day").reset_index(drop=True)
            n = len(g)
            if n <= seq_len + 1:
                continue
            feats = g[feature_cols].values.astype(np.float32)
            daily_count = g["daily_count"].values.astype(float)
            batt = g["battery_mean"].values.astype(float)
            halt_idx = telemetry.last_active_index(daily_count)

            for i in range(0, n - seq_len - 1):
                t = i + seq_len - 1
                win = feats[i : i + seq_len]
                y_batt = float(batt[t + 1])
                if not np.isfinite(y_batt):
                    continue
                if not np.all(np.isfinite(win)):
                    continue

                yc_list.append(float(daily_count[t + 1]))
                if halt_idx is None:
                    if t > n - 1 - halt_horizon_days:
                        continue
                    y_halt = 0.0
                    dt_list.append(float("nan"))
                else:
                    dt = halt_idx - t
                    y_halt = 1.0 if 0 <= dt <= halt_horizon_days else 0.0
                    dt_list.append(float(dt))

                x_list.append(win)
                yb_list.append(y_batt)
                yh_list.append(y_halt)
                day = g["day"].iloc[t]
                day_list.append(int(pd.Timestamp(day).value))
                sid_list.append(sid)
                day_label_list.append(pd.Timestamp(day).strftime("%Y-%m-%d"))

        return telemetry.SeqData(
            x=np.asarray(x_list, dtype=np.float32),
            y_batt=np.asarray(yb_list, dtype=np.float32),
            y_halt=np.asarray(yh_list, dtype=np.float32),
            day_index=np.asarray(day_list, dtype=np.int64),
            sensor_id=np.asarray(sid_list, dtype=object),
            day_label=np.asarray(day_label_list, dtype=object),
            y_halt_dt=np.asarray(dt_list, dtype=np.float32),
            y_next_count=np.asarray(yc_list, dtype=np.float32),
        )

    telemetry.load_original_daily = load_original_daily_no_impute
    telemetry.build_sequences = build_sequences_no_impute
    return telemetry


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Bundle no-impute pass using trainer fork.")
    p.add_argument("--artifact-stem", type=str, required=True)
    p.add_argument("--halt-mode", choices=("binary", "survival"), default="binary")
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--max-sensors", type=int, default=50)
    p.add_argument(
        "--deveui-cohort-csv",
        type=str,
        default="",
        help="Optional CSV with deveui column (or first column) to force an exact cohort.",
    )
    p.add_argument("--halt-loss-weight", type=float, default=4.0)
    p.add_argument("--save-checkpoint-dir", type=str, default="")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    project = _BUNDLE_ROOT.parent
    idata = project / "idata"
    if idata.is_dir():
        os.environ.setdefault("NODE_ANALYSIS_DATA_ROOT", str(idata))
        os.environ.setdefault("HOL_ACAD_DATA_ROOT", str(idata))
    out = os.environ.get("NODE_ANALYSIS_OUTPUT_ROOT", "").strip()
    if out:
        os.environ.setdefault("HOL_ACAD_OUTPUT_ROOT", out)
    vendored_src = _BUNDLE_ROOT / "runtime_vendor" / "src"
    if vendored_src.is_dir():
        src_dir = vendored_src
    else:
        thesis_pipeline = next((p for p in _BUNDLE_ROOT.parents if p.name == "thesis_pipeline"), None)
        if thesis_pipeline is None:
            raise RuntimeError(f"Could not locate runtime_vendor or thesis_pipeline from {_BUNDLE_ROOT}")
        src_dir = thesis_pipeline / "src"

    _install_no_impute_pipeline(src_dir)
    import multitask_battery_halt_trainer_bundle as trainer  # noqa: WPS433

    run_argv = [
        "multitask_battery_halt_trainer_bundle.py",
        "--artifact-stem",
        args.artifact_stem,
        "--halt-mode",
        args.halt_mode,
        "--epochs",
        str(int(args.epochs)),
        "--max-sensors",
        str(int(args.max_sensors)),
        "--split-mode",
        "sensor_holdout",
        "--halt-bce-mix",
        "0.25",
        "--early-stopping-patience",
        "2",
        "--halt-loss-weight",
        str(float(args.halt_loss_weight)),
    ]
    if str(args.deveui_cohort_csv).strip():
        run_argv += ["--deveui-cohort-csv", str(args.deveui_cohort_csv).strip()]
    if str(args.save_checkpoint_dir).strip():
        run_argv += ["--save-checkpoint-dir", str(args.save_checkpoint_dir).strip()]

    old_argv = sys.argv[:]
    try:
        sys.argv = run_argv
        trainer.main()
    finally:
        sys.argv = old_argv


if __name__ == "__main__":
    main()
