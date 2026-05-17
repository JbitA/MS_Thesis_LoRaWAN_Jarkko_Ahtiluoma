#!/usr/bin/env python3
"""
FILE STORY — ``figures/build.py``
=================================

**Role.** CLI driver for the eleven allowlisted thesis figures. For each key in
``thesis_figure_keys.THESIS_FIGURE_KEYS``, ensures output dirs, calls
``recreated_from_new_data.make_for_figure_key``, runs PNG/companion migration,
and builds extras (LSTM-only regression, method_model_ranking tier-2 table).

**Requires.** ``layout.apply_run_env(run_id)`` already set ``GRAPH_ROOT`` and
``OUTPUT_ROOT`` (normally done in ``main.py`` before ``run_figures.py``).

**Usage.**

    python pipeline/figures/build.py
    python pipeline/figures/build.py --figure-key policy_map

**Developed with Cursor AI.**
"""

from __future__ import annotations

import argparse  # Optional single --figure-key vs build-all
import os
import sys

from pathlib import Path  # Resolve figure output directories.



import matplotlib.pyplot as plt  # Extra figures: LSTM-only regression, tier-2 table.

import numpy as np  # R² / MAE / RMSE for LSTM-only regression panel.

import pandas as pd  # Method-B tier-2 ranking table construction.



_PKG = Path(__file__).resolve().parent  # ``pipeline/figures`` package directory.

sys.path.insert(0, str(_PKG))  # Allow sibling imports (``recreated_from_new_data``, etc.).



import recreated_from_new_data as rec  # noqa: E402 — primary figure renderers (training tables).

import util  # noqa: E402 — ``ensure_out`` and bundle paths.

from figure_titles import (  # noqa: E402 — frozen display titles and companion basenames.

    FIGURE_EXTRA_TITLE,

    FIGURE_MAIN_TITLE,

    companion_extra_table_csv,

    companion_extra_table_png,

)

from thesis_figure_keys import (  # noqa: E402 — allowlist, paths, migration helpers.

    THESIS_FIGURE_KEYS,

    figure_png_path,

    migrate_graph_outputs,

    normalize_figure_key,

    remove_legacy_companion_artifacts,

    remove_legacy_figure_pngs,

)





def _build_regression_lstm(out: Path) -> None:

    """Second PNG for ``regression_4_models``: LSTM-only scatter with R²/MAE/RMSE."""

    d = rec._pred_df("lstm").dropna(subset=["y_true_battery", "y_pred_battery"])  # LSTM battery predictions only.

    y = d["y_true_battery"].to_numpy(float)  # Ground-truth voltage vector.

    p = d["y_pred_battery"].to_numpy(float)  # Predicted voltage vector.

    lo, hi = min(y.min(), p.min()), max(y.max(), p.max())  # Shared axis limits for y=x reference line.



    ybar = float(np.mean(y))  # Mean of true values (for R² denominator).

    ss_res = float(np.sum((y - p) ** 2))  # Residual sum of squares.

    ss_tot = float(np.sum((y - ybar) ** 2))  # Total sum of squares.

    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")  # Coefficient of determination.

    mae = float(np.mean(np.abs(p - y)))  # Mean absolute error (volts).

    rmse = float(np.sqrt(np.mean((p - y) ** 2)))  # Root mean squared error (volts).



    fig, ax = plt.subplots(figsize=(7.2, 6.0))  # Single-panel scatter figure.

    ax.scatter(y, p, s=6, alpha=0.25, color=rec.COL["lstm"])  # LSTM-colored points.

    ax.plot([lo, hi], [lo, hi], "k--", lw=1.2)  # Diagonal perfect-prediction reference.

    ax.set_title(FIGURE_EXTRA_TITLE["regression_4_models"][1], fontsize=14)  # “Regression LSTM only” title.

    ax.set_xlabel("True battery voltage (V)")  # X-axis label.

    ax.set_ylabel("Predicted battery voltage (V)")  # Y-axis label.

    ax.text(  # In-plot metrics annotation box.

        0.05,

        0.95,

        f"R² = {r2:.3f}\nMAE = {mae:.4f} V\nRMSE = {rmse:.4f} V",

        transform=ax.transAxes,

        va="top",

        fontsize=12,

        bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": "#bbbbbb"},

    )

    ax.grid(alpha=0.25)  # Light grid for readability.

    fig.tight_layout()  # Reduce label clipping.

    fig.savefig(figure_png_path(out, "regression_4_models", variant=1), dpi=220)  # Second PNG variant path.

    plt.close(fig)  # Free figure memory.


def _build_methodb_tier2_table(out: Path) -> None:
    """Companion ranking table for ``method_model_ranking`` (utility table PNG + CSV)."""
    figure_key = "method_model_ranking"  # Folder key for Method-B ranking figure.
    d = rec._comp_df(rec.PRIMARY_ARTIFACT_STEM).set_index("model")  # Multitask comparison table indexed by model.
    rank_df = pd.DataFrame(index=list(rec.MODELS))  # One row per backbone architecture.
    rank_df["Model"] = [rec.LABEL[m].upper() for m in rec.MODELS]  # Display labels (uppercase).
    rank_df["PR-AUC"] = d.loc[list(rec.MODELS), "halt_ap"].astype(float)  # Average precision column.
    rank_df["Recall"] = d.loc[list(rec.MODELS), "halt_recall"].astype(float)  # Recall at applied threshold.
    rank_df["F2"] = d.loc[list(rec.MODELS), "halt_f2"].astype(float)  # F2 score column.
    rank_df["Brier"] = d.loc[list(rec.MODELS), "halt_brier_tscaled"].astype(float)  # Temp-scaled Brier score.
    rank_df["FP"] = d.loc[list(rec.MODELS), "halt_fp"].astype(float)  # False positives count.
    rank_df["FN"] = d.loc[list(rec.MODELS), "halt_fn"].astype(float)  # False negatives count.
    rank_df = rank_df.sort_values(["F2", "Recall", "PR-AUC"], ascending=False).reset_index(drop=True)  # Rank order.
    rank_df.insert(0, "Rank", range(1, len(rank_df) + 1))  # 1-based rank column.
    df = rank_df[["Rank", "Model", "PR-AUC", "Recall", "F2", "Brier", "FP", "FN"]].copy()  # Final display columns.
    df["FP"] = df["FP"].round().astype(int)  # Integer FP counts for table.
    df["FN"] = df["FN"].round().astype(int)  # Integer FN counts for table.
    for c in ["PR-AUC", "Recall", "F2", "Brier"]:  # Format floating metrics to four decimals.
        df[c] = df[c].map(lambda v: f"{float(v):.4f}")

    fig, ax = plt.subplots(figsize=(13.8, 3.8))  # Wide table figure.
    ax.axis("off")  # Hide axes; table-only panel.
    cols = list(df.columns)  # Column headers for matplotlib table.
    body = df.values.tolist()  # Row cells as nested lists.
    t = ax.table(cellText=body, colLabels=cols, cellLoc="center", loc="center")  # Render grid table.
    t.auto_set_font_size(False)  # Use explicit font size below.
    t.set_fontsize(13)  # Cell text size.
    t.scale(1, 1.85)  # Row height scale factor.
    for c in range(len(cols)):  # Style header row.
        t[(0, c)].set_facecolor("#D9E1F2")
        t[(0, c)].set_text_props(weight="bold")
    rank_col = cols.index("Rank")  # Index of rank column for row shading.
    for r_i in range(1, len(body) + 1):  # Data rows (1-based table indices).
        rk = int(body[r_i - 1][rank_col])  # Rank value for this row.
        if rk == 1:  # Gold/green highlight for rank 1.
            bg = "#C8E6C9"
        elif rk == 2:  # Lighter green for rank 2.
            bg = "#E8F5E9"
        elif rk == 3:  # Amber for rank 3.
            bg = "#FFF8E1"
        else:  # Red tint for lower ranks.
            bg = "#FFEBEE"
        for c in range(len(cols)):  # Apply row background across columns.
            t[(r_i, c)].set_facecolor(bg)
    ax.set_title(FIGURE_MAIN_TITLE[figure_key], fontsize=21, pad=14)  # Main figure title above table.
    fig.tight_layout()  # Fit title and table.
    fig.savefig(out / companion_extra_table_png(figure_key, 1), dpi=220)  # Tier-2 utility table PNG.
    plt.close(fig)  # Release figure.
    (out / companion_extra_table_csv(figure_key, 1)).write_text(df.to_csv(index=False), encoding="utf-8")  # Matching CSV.


def build_figure_key(figure_key: str) -> None:

    """Build one thesis figure folder (primary PNG + extras)."""

    figure_key = normalize_figure_key(figure_key)  # Resolve legacy aliases to canonical key.

    out = util.ensure_out(figure_key)  # Create ``graphs/<key>/outputs/`` directory.

    rec.make_for_figure_key(figure_key, figure_png_path(out, figure_key))  # Primary PNG via recreated plotters.

    if figure_key == "regression_4_models":  # Extra LSTM-only scatter variant.

        _build_regression_lstm(out)

    if figure_key == "method_model_ranking":  # Extra tier-2 utility table companions.

        _build_methodb_tier2_table(out)

        remove_legacy_companion_artifacts(out, figure_key)  # Drop old generic companion names.

    remove_legacy_figure_pngs(out, figure_key)  # Remove superseded ``recreated_*.png`` names.





def main() -> None:

    p = argparse.ArgumentParser(description="Build thesis figures for one or all figure keys.")  # CLI parser.

    p.add_argument(  # Optional single-figure build.

        "--figure-key",

        "--slug",

        dest="figure_key",

        default="",

        help="Single figure key (PNG-aligned folder name); default builds all 11.",

    )

    args = p.parse_args()  # Parse argv.

    keys = [normalize_figure_key(args.figure_key)] if args.figure_key else list(THESIS_FIGURE_KEYS)  # Key list.

    for figure_key in keys:  # Build each selected figure.

        print(f"Building {figure_key}...", flush=True)  # Progress line for logs.

        build_figure_key(figure_key)  # Render PNG(s) and companions for this key.

    _src = _PKG.parent / "src"  # Pipeline source for late env_config import.

    if str(_src) not in sys.path:  # Ensure import path if not already set.

        sys.path.insert(0, str(_src))

    from node_analysis_pipeline import env_config as env  # noqa: WPS433 — read graph root after builds.



    gr = env.get_str(env.ENV_GRAPH_ROOT, "")  # Run-scoped graph directory.

    if gr:  # Migrate legacy names when writing under centralized graph root.

        migrate_graph_outputs(Path(gr))

    print("Done.", flush=True)  # Completion marker.





if __name__ == "__main__":  # Script entry when executed directly.

    main()  # Run CLI build loop.


