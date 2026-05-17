#!/usr/bin/env python3
"""
FILE STORY — ``recreated_from_new_data.py``
===========================================

**Role.** Plotting engine for all eleven allowlisted thesis figures. Called once
per figure key from ``build.py`` → :func:`make_for_figure_key`.

**Data in (two sources only).**

  1. **Training tables** — ``$OUTPUT_ROOT/tables/`` (usually ``outputs/<run_id>/models/tables/``):
     predictions, threshold sweeps, and ``*_model_comparison_battery_halt.csv`` per pass.
     Stems: ``node_analysis_<run_id>_multitask_binary`` (primary), ``…_multitask_binary_no_impute``,
     ``…_single_task_battery``, ``…_single_task_halt_focus``.

  2. **Bundled assets** — ``pipeline/assets/``:
     ``battery_missingness_reference_series.csv`` (fixed legacy trace for missingness figure);
     ``ribbon_chart_default_sensor.csv`` (default DevEUI for ribbon / time-series integration).

**Data out.** One primary PNG path per call (basename from ``figure_titles.py``); no companion CSVs here.

**Sensor choice.** Most figures auto-pick a sensor present in both imputed and no-impute tables;
``battery_missingness`` uses the reference CSV sensor; ribbon may override via ``NODE_ANALYSIS_VC_SENSOR``.

**Does not** train models or write companion metric sheets (see ``generate_figure_companion_tables.py``).

**Developed with Cursor AI.**
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_curve, precision_score

import util

import sys

_FIGURES_DIR = Path(__file__).resolve().parent
_BUNDLE_ROOT = _FIGURES_DIR.parent
_SRC = _BUNDLE_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from node_analysis_pipeline import env_config as env  # noqa: E402
from node_analysis_pipeline.artifact_stems import artifact_stem  # noqa: E402

RUN_ID = env.get_str(env.ENV_RUN_ID, "node_analysis1")
TABLES = Path(env.get_str(env.ENV_OUTPUT_ROOT, str(util.BUNDLE_ROOT / "outputs"))) / "tables"
PRIMARY_ARTIFACT_STEM = artifact_stem(RUN_ID, "multitask_binary")
NOIMP_ARTIFACT_STEM = artifact_stem(RUN_ID, "multitask_binary_no_impute")
STL_BATTERY_STEM = artifact_stem(RUN_ID, "single_task_battery")
STL_HALT_STEM = artifact_stem(RUN_ID, "single_task_halt_focus")

MODELS = ("lstm", "gru", "tcn", "transformer")
ORIGINAL_SENSOR_DEFAULT = "SYNTH_NODE_002"  # Example device for bundled figure assets (not a real LoRaWAN EUI)
SENSOR_BY_FIGURE_KEY: Dict[str, str] = {
    "battery_missingness": ORIGINAL_SENSOR_DEFAULT,
}
LABEL = {"lstm": "LSTM", "gru": "GRU", "tcn": "TCN", "transformer": "Transformer"}
COL = {"lstm": "#1565c0", "gru": "#2e7d32", "tcn": "#ef6c00", "transformer": "#6a1b9a"}
PR_SCATTER_ANNOTE_XYTEXT: Dict[str, tuple[int, int]] = {
    "lstm": (10, 14),
    "gru": (-10, 12),
    "tcn": (22, -18),
    "transformer": (-12, -30),
}

plt.rcParams.update(
    {
        "font.size": 12,
        "axes.titlesize": 18,
        "axes.labelsize": 16,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 14,
        "figure.titlesize": 19,
    }
)


# ---------------------------------------------------------------------------
# Training-table loaders (predictions, sweeps, comparison CSVs)
# ---------------------------------------------------------------------------


def _pred_df(model: str) -> pd.DataFrame:
    """Imputed multitask pass: per-day battery and halt predictions for one backbone."""
    return pd.read_csv(TABLES / f"{PRIMARY_ARTIFACT_STEM}_{model}_battery_halt_predictions.csv")


def _noimp_df(model: str) -> pd.DataFrame:
    """Non-imputed multitask pass predictions (NaN gaps preserved in features)."""
    return pd.read_csv(TABLES / f"{NOIMP_ARTIFACT_STEM}_{model}_battery_halt_predictions.csv")


def _comp_df(stem: str) -> pd.DataFrame:
    """Model comparison row set for a training pass artifact stem."""
    return pd.read_csv(TABLES / f"{stem}_model_comparison_battery_halt.csv")


def _sweep_df(model: str) -> pd.DataFrame:
    """Applied halt threshold sweep (F0.5/F1/F2 columns) for imputed pass."""
    return pd.read_csv(TABLES / f"{PRIMARY_ARTIFACT_STEM}_{model}_halt_threshold_sweep_applied.csv")


def _sweep_df_noimp(model: str) -> pd.DataFrame:
    """Applied halt threshold sweep for the no-impute pass."""
    return pd.read_csv(TABLES / f"{NOIMP_ARTIFACT_STEM}_{model}_halt_threshold_sweep_applied.csv")


def _save(fig: plt.Figure, out_path: Path, *, tight: bool = True) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if tight:
        fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Sensor selection and imputed / non-imputed alignment
# ---------------------------------------------------------------------------


def _align_sensor(model: str, sensor: str) -> pd.DataFrame:
    di = _pred_df(model)[["sensor", "day", "y_true_halt_within_h", "y_true_battery", "y_pred_halt_prob", "y_pred_battery"]].copy()
    dn = _noimp_df(model)[["sensor", "day", "y_pred_halt_prob", "y_pred_battery"]].copy()
    di = di[di["sensor"].astype(str) == sensor].copy()
    dn = dn[dn["sensor"].astype(str) == sensor].copy()
    di["day"] = pd.to_datetime(di["day"])
    dn["day"] = pd.to_datetime(dn["day"])
    d = di.merge(
        dn.rename(columns={"y_pred_halt_prob": "y_pred_halt_prob_no", "y_pred_battery": "y_pred_battery_no"})[
            ["sensor", "day", "y_pred_halt_prob_no", "y_pred_battery_no"]
        ],
        on=["sensor", "day"],
        how="inner",
    )
    return d.sort_values("day").reset_index(drop=True)


def _pick_sensor() -> str:
    common: set[str] | None = None
    for m in MODELS:
        si = set(_pred_df(m)["sensor"].astype(str).unique().tolist())
        sn = set(_noimp_df(m)["sensor"].astype(str).unique().tolist())
        s = si.intersection(sn)
        common = s if common is None else common.intersection(s)
    if not common:
        d = _pred_df("gru")
        return str(d["sensor"].astype(str).value_counts().index[0])
    # pick the most frequent common sensor in GRU predictions for stable panel length
    d = _pred_df("gru")
    vc = d[d["sensor"].astype(str).isin(common)]["sensor"].astype(str).value_counts()
    return str(vc.index[0])


def _f2_threshold_from_sweep(sweep: pd.DataFrame) -> float:
    j = int(np.argmax(pd.to_numeric(sweep["f2"], errors="coerce").fillna(-1e9).to_numpy()))
    return float(sweep.iloc[j]["threshold"])


# ---------------------------------------------------------------------------
# Figure dispatch (one branch per allowlisted thesis figure key)
# ---------------------------------------------------------------------------


def make_for_figure_key(figure_key: str, out_path: Path) -> None:
    """Render the primary PNG for ``figure_key`` to ``out_path`` (parent dirs created)."""
    sensor = SENSOR_BY_FIGURE_KEY.get(figure_key, _pick_sensor())

    if figure_key == "battery_missingness":
        # Exact original figure format/logic: single axis, true trajectory only,
        # missing-day stripes from raw iData days for the same original sensor.
        src = util.BATTERY_MISSINGNESS_REFERENCE_CSV
        df = pd.read_csv(src)
        req = {"sensor", "day", "y_true_battery"}
        if not req.issubset(df.columns):
            raise RuntimeError(f"Missing required columns in {src}")
        # Use the exact sensor present in this original source CSV (same as legacy figure logic).
        sensor_in_src = str(df["sensor"].astype(str).iloc[0])
        d = df[df["sensor"].astype(str) == sensor_in_src].copy()
        if len(d) == 0:
            raise RuntimeError(f"Sensor {sensor_in_src} not found in {src}")
        d["day"] = pd.to_datetime(d["day"])
        d = d.sort_values("day").reset_index(drop=True)
        day = d["day"].to_numpy()
        y = d["y_true_battery"].to_numpy(float)

        data_root = Path(env.get_str(env.ENV_DATA_ROOT, "")).resolve() if env.get_str(env.ENV_DATA_ROOT, "") else util.BUNDLE_ROOT.parent.parent / "idata"
        missing_day_mask = np.zeros(len(d), dtype=bool)
        if data_root.exists():
            parts = []
            for f in sorted(data_root.glob("*/application.csv")):
                rd = pd.read_csv(f, usecols=["time", "deveui"])
                rd = rd[rd["deveui"].astype(str) == sensor_in_src]
                if len(rd):
                    parts.append(rd)
            if parts:
                raw = pd.concat(parts, ignore_index=True)
                raw["time"] = pd.to_datetime(raw["time"], unit="ms", utc=True, errors="coerce")
                raw = raw.dropna(subset=["time"])
                raw_days = pd.DatetimeIndex(raw["time"].dt.floor("D").unique()).sort_values()
                observed_days = pd.DatetimeIndex(pd.to_datetime(raw_days.date))
                model_days = pd.DatetimeIndex(pd.to_datetime(pd.DatetimeIndex(day).date))
                missing_day_mask = ~model_days.isin(observed_days)

        plt.style.use("seaborn-v0_8-whitegrid")
        fig, ax = plt.subplots(1, 1, figsize=(14.8, 4.8))
        ax.plot(day, y, color="black", lw=2.0, label="True voltage trajectory")
        if np.any(missing_day_mask):
            md = day[missing_day_mask]
            ax.vlines(
                md,
                ymin=float(np.nanmin(y)),
                ymax=float(np.nanmax(y)),
                color="#f57c00",
                alpha=0.12,
                linewidth=0.8,
                label="Missing measurement",
            )
        ax.set_title("Time-series missing measurements", fontsize=24)
        ax.set_ylabel("Battery voltage (V)", fontsize=16)
        ax.set_xlabel("Time (day)", fontsize=19)
        ax.tick_params(axis="both", labelsize=14)
        ax.legend(loc="best", ncol=2, fontsize=16)
        ax.grid(alpha=0.2)
        _save(fig, out_path)
        return

    if figure_key == "threshold_f2_impute_sensitivity_comparison":
        rows = []
        for m in MODELS:
            d = _align_sensor(m, sensor)
            if len(d) == 0:
                rows.append({"model": m, "acc_i": np.nan, "pre_i": np.nan, "acc_n": np.nan, "pre_n": np.nan})
                continue
            y = d["y_true_halt_within_h"].astype(int).to_numpy()
            swi = _sweep_df(m)
            th_i = _f2_threshold_from_sweep(swi)
            th_n = _f2_threshold_from_sweep(_sweep_df_noimp(m))
            pi = d["y_pred_halt_prob"].to_numpy(float)
            pn = d["y_pred_halt_prob_no"].to_numpy(float)
            rows.append(
                {
                    "model": m,
                    "acc_i": accuracy_score(y, (pi >= th_i).astype(int)),
                    "pre_i": precision_score(y, (pi >= th_i).astype(int), zero_division=0),
                    "acc_n": accuracy_score(y, (pn >= th_n).astype(int)),
                    "pre_n": precision_score(y, (pn >= th_n).astype(int), zero_division=0),
                }
            )
        s = pd.DataFrame(rows)
        fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.8), sharex=True, sharey=True)
        # Fixed in-plot label anchors (data coordinates) to avoid overlap
        # when multiple models cluster near (1.0, 1.0).
        label_positions = {
            "lstm": (0.84, 1.009),
            "gru": (0.86, 1.003),
            "transformer": (0.88, 0.997),
            "tcn": (0.90, 0.991),
        }
        for ax, a, p, ttl in (
            (axes[0], "acc_i", "pre_i", "With feature imputation"),
            (axes[1], "acc_n", "pre_n", "Without feature imputation"),
        ):
            for _, r in s.iterrows():
                if not np.isfinite(r[p]) or not np.isfinite(r[a]):
                    continue
                ax.scatter(r[p], r[a], s=120, color=COL[r["model"]], edgecolor="black")
                tx, ty = label_positions.get(str(r["model"]), (0.86, 1.0))
                ax.annotate(
                    LABEL[r["model"]],
                    xy=(float(r[p]), float(r[a])),
                    xytext=(tx, ty),
                    textcoords="data",
                    fontsize=14,
                    ha="center",
                    va="center",
                    arrowprops={"arrowstyle": "-", "color": "#555555", "lw": 0.8},
                )
            ax.set_title(ttl, fontsize=18)
            ax.set_xlabel("Precision", fontsize=17)
            ax.set_xlim(0.4, 1.05)
            ax.set_ylim(0.98, 1.02)
            ax.tick_params(axis="both", labelsize=15)
            ax.grid(alpha=0.25)
        axes[0].set_ylabel("Accuracy", fontsize=17)
        fig.suptitle(
            "Halt prediction: accuracy vs. precision at F\u2082-optimal thresholds (single sensor)",
            fontsize=20,
        )
        _save(fig, out_path)
        return

    if figure_key in {"imputed_panel", "imputed_vs_non_imputed_halt"}:
        if figure_key == "imputed_panel":
            # Requested layout: two time-series panels only.
            # Left = all 4 model imputed predictions vs true.
            # Right = all 4 model non-imputed predictions vs true.
            fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True, sharey=False)
            ax_imp, ax_non = axes
            true_drawn = False
            for m in MODELS:
                d = _align_sensor(m, sensor)
                x = np.arange(len(d))
                if not true_drawn:
                    ax_imp.plot(x, d["y_true_battery"], "k-", lw=1.4, label="True")
                    ax_non.plot(x, d["y_true_battery"], "k-", lw=1.4, label="True")
                    true_drawn = True
                ax_imp.plot(x, d["y_pred_battery"], color=COL[m], lw=1.2, label=LABEL[m])
                ax_non.plot(x, d["y_pred_battery_no"], color=COL[m], lw=1.2, ls="--", label=LABEL[m])

            ax_imp.set_title("Imputed preprocessing", fontsize=16)
            ax_non.set_title("Non-imputed preprocessing", fontsize=16)
            ax_imp.set_ylabel("Battery voltage (V)", fontsize=15)
            ax_non.set_ylabel("Battery voltage (V)", fontsize=15)
            ax_non.set_xlabel("Time window", fontsize=15)
            ax_imp.grid(alpha=0.2)
            ax_non.grid(alpha=0.2)
            ax_imp.tick_params(axis="both", labelsize=13)
            ax_non.tick_params(axis="both", labelsize=13)
            ax_imp.legend(loc="best", fontsize=12, frameon=True)
            ax_non.legend(loc="best", fontsize=12, frameon=True)
            fig.tight_layout(rect=[0, 0.02, 1, 0.98])
            _save(fig, out_path, tight=False)
            return

        fig, axes = plt.subplots(2, 1, figsize=(13.5, 9), sharex=True, sharey=True)
        ax_imp, ax_non = axes
        ev = None
        for m in MODELS:
            d = _align_sensor(m, sensor)
            x = np.arange(len(d))
            ax_imp.plot(x, d["y_pred_halt_prob"], color=COL[m], lw=1.2, label=LABEL[m])
            ax_non.plot(x, d["y_pred_halt_prob_no"], color=COL[m], lw=1.2, ls="--", label=LABEL[m])
            if ev is None:
                ev = np.where(d["y_true_halt_within_h"].astype(int).to_numpy() == 1)[0]
        if ev is not None and len(ev):
            ax_imp.scatter(ev, np.ones_like(ev), s=8, c="black", alpha=0.6, label="True halt event")
            ax_non.scatter(ev, np.ones_like(ev), s=8, c="black", alpha=0.6, label="True halt event")

        ax_imp.set_title("Imputed preprocessing", fontsize=22)
        ax_non.set_title("Non-imputed preprocessing", fontsize=22)
        ax_imp.set_ylabel("Halt probability", fontsize=20)
        ax_non.set_ylabel("Halt probability", fontsize=20)
        ax_non.set_xlabel("Time window", fontsize=20)
        ax_imp.set_ylim(0.5, 1.0)
        ax_non.set_ylim(0.5, 1.0)
        ax_imp.grid(alpha=0.2)
        ax_non.grid(alpha=0.2)
        ax_imp.tick_params(axis="both", labelsize=18)
        ax_non.tick_params(axis="both", labelsize=18)
        ax_imp.legend(loc="upper left", fontsize=16, frameon=True)
        ax_non.legend(loc="upper left", fontsize=16, frameon=True)
        _save(fig, out_path)
        return

    if figure_key == "confusion_grid":
        fig, axes = plt.subplots(2, 2, figsize=(13, 10.5))
        axes = axes.flatten()
        tick_kw = {"axis": "both", "which": "major", "labelsize": 16}
        for ax, model in zip(axes, MODELS):
            d = _pred_df(model)
            y = d["y_true_halt_within_h"].astype(int).to_numpy()
            yhat = d["y_pred_halt_cls"].astype(int).to_numpy()
            cm = confusion_matrix(y, yhat, labels=[0, 1]).astype(float)
            cm = np.divide(cm, np.maximum(cm.sum(axis=1, keepdims=True), 1.0))
            ax.imshow(cm, vmin=0, vmax=1, cmap="Greens", aspect="equal")
            ax.set_title(LABEL[model], fontsize=17, fontweight="bold")
            ax.set_xticks([0, 1], labels=["Pred Neg", "Pred Pos"])
            ax.set_yticks([0, 1], labels=["Actual Neg", "Actual Pos"])
            ax.tick_params(**tick_kw)
            for i in range(2):
                for j in range(2):
                    ax.text(j, i, f"{cm[i, j]:.3f}", ha="center", va="center", fontsize=17, fontweight="bold")
        fig.suptitle("Halt classification confusion matrices", fontsize=20, fontweight="bold", y=0.98)
        fig.subplots_adjust(left=0.10, right=0.90, top=0.86, bottom=0.08, wspace=0.38, hspace=0.40)
        _save(fig, out_path, tight=False)
        return

    if figure_key == "pr_f2_only":
        fig, ax = plt.subplots(figsize=(7.4, 7))
        recalls: list[float] = []
        precs: list[float] = []
        taus: list[float] = []
        for model in MODELS:
            d = _pred_df(model)
            prec = float(d["y_pred_halt_cls"].sum()) / max(
                int(d["y_pred_halt_cls"].sum() + ((d["y_pred_halt_cls"] == 1) & (d["y_true_halt_within_h"] == 0)).sum()),
                1,
            )
            rec = float(((d["y_pred_halt_cls"] == 1) & (d["y_true_halt_within_h"] == 1)).sum()) / max(
                int((d["y_true_halt_within_h"] == 1).sum()), 1
            )
            recalls.append(rec)
            precs.append(prec)
            taus.append(float(_f2_threshold_from_sweep(_sweep_df(model))))
        ta = np.asarray(taus, dtype=float)
        lo, hi = float(ta.min()), float(ta.max())
        span = hi - lo
        if span < 1e-9:
            lo, hi = max(0.0, lo - 0.03), min(1.0, hi + 0.03)
        elif span < 0.05:
            pad = (0.05 - span) / 2
            lo, hi = max(0.0, lo - pad), min(1.0, hi + pad)
        norm = Normalize(vmin=lo, vmax=hi)
        cmap = plt.cm.plasma
        ax.scatter(
            recalls,
            precs,
            s=150,
            c=ta,
            cmap=cmap,
            norm=norm,
            edgecolors="black",
            linewidths=0.9,
            zorder=3,
        )
        for i, model in enumerate(MODELS):
            ox, oy = PR_SCATTER_ANNOTE_XYTEXT[model]
            ax.annotate(
                LABEL[model],
                (recalls[i], precs[i]),
                xytext=(ox, oy),
                textcoords="offset points",
                fontsize=12,
                ha="right" if ox < 0 else "left",
                va="top" if oy < 0 else "bottom",
            )
        cbar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=ax, fraction=0.046, pad=0.02)
        cbar.set_label("Probability cutoff τ")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.tick_params(axis="both", labelsize=12)
        ax.set_title("Precision-Recall F2 with probability cutoff")
        ax.grid(alpha=0.25)
        pad = 0.012
        min_span = 0.07
        r_lo, r_hi = min(recalls), max(recalls)
        p_lo, p_hi = min(precs), max(precs)
        rx = max(r_hi - r_lo, min_span)
        py = max(p_hi - p_lo, min_span)
        r_mid = (r_lo + r_hi) / 2
        p_mid = (p_lo + p_hi) / 2
        x0 = max(0.0, r_mid - rx / 2 - pad)
        x1 = min(1.0, r_mid + rx / 2 + pad)
        y0 = max(0.0, p_mid - py / 2 - pad)
        y1 = min(1.0, p_mid + py / 2 + pad)
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)
        _save(fig, out_path, tight=False)
        return

    if figure_key == "policy_map":
        fig, ax = plt.subplots(figsize=(9, 7))
        markers = {"f05": "s", "f1": "o", "f2": "^"}
        for m in MODELS:
            sw = _sweep_df(m)
            for met in ("f05", "f1", "f2"):
                j = int(np.argmax(pd.to_numeric(sw[met], errors="coerce").fillna(-1e9).to_numpy()))
                r = sw.iloc[j]
                ax.scatter(
                    float(r["recall"]),
                    float(r["precision"]),
                    s=120,
                    color=COL[m],
                    marker=markers[met],
                    edgecolors="black",
                    zorder=3,
                )
        beta_handles = [
            Line2D(
                [0],
                [0],
                linestyle="none",
                marker=markers["f05"],
                color="k",
                markerfacecolor="#bdbdbd",
                markersize=10,
                label=r"$F_{0.5}$",
            ),
            Line2D(
                [0],
                [0],
                linestyle="none",
                marker=markers["f1"],
                color="k",
                markerfacecolor="#bdbdbd",
                markersize=10,
                label=r"$F_{1}$",
            ),
            Line2D(
                [0],
                [0],
                linestyle="none",
                marker=markers["f2"],
                color="k",
                markerfacecolor="#bdbdbd",
                markersize=10,
                label=r"$F_{2}$",
            ),
        ]
        model_handles = [
            Line2D(
                [0],
                [0],
                linestyle="none",
                marker="o",
                color="k",
                markerfacecolor=COL[m],
                markersize=10,
                label=LABEL[m],
            )
            for m in MODELS
        ]
        ax.legend(
            handles=beta_handles + model_handles,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.11),
            ncol=7,
            framealpha=0.95,
            fontsize=12,
            handlelength=1.2,
            columnspacing=1.0,
            handletextpad=0.5,
        )
        ax.set_xlabel("Recall", fontsize=16)
        ax.set_ylabel("Precision", fontsize=16)
        ax.tick_params(axis="both", labelsize=15)
        ax.set_title("Precision-recall F-beta sensitivity optimization", fontsize=18)
        ax.grid(alpha=0.25)
        fig.subplots_adjust(left=0.11, right=0.95, top=0.90, bottom=0.17)
        _save(fig, out_path, tight=False)
        return


    if figure_key == "regression_4_models":
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        axes = axes.flatten()
        for ax, model in zip(axes, MODELS):
            d = _pred_df(model).dropna(subset=["y_true_battery", "y_pred_battery"])
            y = d["y_true_battery"].to_numpy(float)
            p = d["y_pred_battery"].to_numpy(float)
            lo, hi = min(y.min(), p.min()), max(y.max(), p.max())
            ax.scatter(y, p, s=5, alpha=0.25, color=COL[model])
            ax.plot([lo, hi], [lo, hi], "k--", lw=1)
            ax.set_title(LABEL[model])
            ax.set_xlabel("True battery voltage (V)")
            ax.set_ylabel("Predicted battery voltage (V)")
            ax.grid(alpha=0.2)
        fig.suptitle("Regression predicted vs true values")
        _save(fig, out_path)
        return


    if figure_key == "stl_vs_mtl_delta":
        # Match legacy layout: docs/single_sensor_comparison_method/train_single_vs_multitask_4models_v3_stratified.py
        # — 2×1 panels, bar width 0.18, yerr+capsize (here zeros: one holdout run, no seed bootstrap).
        order = ("gru", "lstm", "tcn", "transformer")
        d_mtl = _comp_df(PRIMARY_ARTIFACT_STEM).set_index("model").reindex(order)
        d_sb = _comp_df(STL_BATTERY_STEM).set_index("model").reindex(order)
        d_sh = _comp_df(STL_HALT_STEM).set_index("model").reindex(order)

        x = np.arange(len(order))
        w = 0.18
        err = np.zeros(len(order), dtype=float)

        delta_rmse = (d_mtl["battery_rmse"] - d_sb["battery_rmse"]).astype(float).to_numpy()
        delta_r2 = (d_mtl["battery_r2"] - d_sb["battery_r2"]).astype(float).to_numpy()
        delta_auc = (d_mtl["halt_auc"] - d_sh["halt_auc"]).astype(float).to_numpy()
        delta_ap = (d_mtl["halt_ap"] - d_sh["halt_ap"]).astype(float).to_numpy()
        delta_f2 = (d_mtl["halt_f2"] - d_sh["halt_f2"]).astype(float).to_numpy()
        # Legacy stratified script uses raw Brier for delta_brier_mt_minus_clsonly (not temp-scaled).
        delta_brier = (d_mtl["halt_brier"] - d_sh["halt_brier"]).astype(float).to_numpy()

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12.5, 9), sharex=True, gridspec_kw={"hspace": 0.24})
        ax1.axhline(0.0, color="#444", lw=1)
        ax1.bar(
            x - w / 2,
            delta_rmse,
            width=w,
            yerr=err,
            capsize=3,
            label="RMSE delta",
        )
        ax1.bar(
            x + w / 2,
            delta_r2,
            width=w,
            yerr=err,
            capsize=3,
            label="R² delta",
        )
        ax1.set_title("Regression task")
        ax1.set_ylabel("Delta")
        ax1.legend(loc="best", fontsize=14)
        ax1.tick_params(axis="both", labelsize=15)
        ax1.grid(alpha=0.2, axis="y")

        ax2.axhline(0.0, color="#444", lw=1)
        ax2.bar(x - 1.5 * w, delta_auc, width=w, yerr=err, capsize=3, label="AUC delta")
        ax2.bar(x - 0.5 * w, delta_ap, width=w, yerr=err, capsize=3, label="AP delta")
        ax2.bar(x + 0.5 * w, delta_f2, width=w, yerr=err, capsize=3, label="F2 delta")
        ax2.bar(x + 1.5 * w, delta_brier, width=w, yerr=err, capsize=3, label="Brier delta")
        ax2.set_title("Classification task")
        ax2.set_ylabel("Delta")
        ax2.set_xticks(x)
        ax2.set_xticklabels([LABEL[m] for m in order])
        ax2.set_xlabel("Deep learning model", fontsize=19)
        ax2.legend(loc="best", fontsize=14, ncol=2)
        ax2.tick_params(axis="both", labelsize=15)
        ax2.grid(alpha=0.2, axis="y")

        fig.suptitle("Deep learning MTL and STL performance deltas", fontsize=22)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout(rect=[0, 0.0, 1, 0.97])
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
        return

    if figure_key == "deep_learning_time_series_integration":
        # Rebuild as true vC class: one-sensor 3-panel ribbon with shaded halt band and
        # (R,H) state legend at figure level. Use mixed policy model tables:
        # transformer from SWA-off branch.
        model = "transformer"
        d = _pred_df(model).copy()
        d["day"] = pd.to_datetime(d["day"], errors="coerce")

        meta_csv = util.RIBBON_CHART_DEFAULT_SENSOR_CSV
        if not meta_csv.is_file():
            raise RuntimeError(f"Missing source metadata CSV: {meta_csv}")
        meta = pd.read_csv(meta_csv)
        sensor = str(env.get_str(env.ENV_VC_SENSOR, str(meta.loc[0, "sensor"]))).strip()

        s = d[d["sensor"].astype(str) == sensor].dropna(subset=["day"]).sort_values("day").copy()
        if s.empty:
            # New-run cohort may not include the original legacy sensor.
            # Deterministic fallback: choose a sensor with strongest end-of-life signal
            # (battery drop + halt-probability rise near sequence tail), then fallback
            # to longest timeline if scoring cannot be computed.
            best_sensor = None
            best_score = -1e18
            for sid, g in d.groupby(d["sensor"].astype(str)):
                g = g.sort_values("day").dropna(subset=["y_true_battery", "y_pred_halt_prob"])
                n = len(g)
                if n < 80:
                    continue
                b = g["y_true_battery"].to_numpy(float)
                p = g["y_pred_halt_prob"].to_numpy(float)
                k = max(10, int(0.15 * n))
                b_head = float(np.nanmedian(b[:k]))
                b_tail = float(np.nanmedian(b[-k:]))
                p_head = float(np.nanmedian(p[:k]))
                p_tail = float(np.nanmedian(p[-k:]))
                drop = b_head - b_tail
                rise = p_tail - p_head
                end_hi = float(np.mean(p[-k:] >= 0.8))
                peak = float(np.nanmax(p[-k:]))
                score = 2.0 * drop + 1.5 * rise + 0.5 * end_hi + 0.3 * peak
                if np.isfinite(score) and score > best_score:
                    best_score = score
                    best_sensor = str(sid)
            if best_sensor is not None:
                sensor = best_sensor
            else:
                sensor = str(d["sensor"].astype(str).value_counts().index[0])
            s = d[d["sensor"].astype(str) == sensor].dropna(subset=["day"]).sort_values("day").copy()
            if s.empty:
                raise RuntimeError("No usable sensor rows available in new-run transformer predictions")

        # Same threshold/ribbon spirit as original vC generator.
        tau_b_q = 0.10
        tau_h_q = 0.85
        n_ext = 20
        k_slope = 14

        y_true_b = s["y_true_battery"].to_numpy(float)
        y_pred_b = s["y_pred_battery"].to_numpy(float)
        prob = s["y_pred_halt_prob"].to_numpy(float)
        cls_eval = s["y_pred_halt_cls"].astype(int).to_numpy() if "y_pred_halt_cls" in s.columns else (prob >= 0.5).astype(int)

        tau_b = float(np.nanquantile(y_true_b, tau_b_q))
        tau_h = float(np.nanquantile(prob, tau_h_q))
        if not np.isfinite(tau_b):
            tau_b = float(np.nanmedian(y_true_b))
        if not np.isfinite(tau_h):
            tau_h = 0.5

        def _extend(series: np.ndarray, n_steps: int, k: int) -> np.ndarray:
            y = np.asarray(series, dtype=float)
            kk = min(k, len(y))
            x = np.arange(kk, dtype=float)
            slope = float(np.polyfit(x, y[-kk:], 1)[0]) if kk >= 2 else 0.0
            last = float(y[-1])
            return np.array([last + slope * (i + 1) for i in range(n_steps)], dtype=float)

        last_day = s["day"].iloc[-1]
        ext_days = pd.date_range(last_day + pd.Timedelta(days=1), periods=n_ext, freq="D")
        batt_ext = _extend(y_pred_b, n_ext, k_slope)
        risk_ext = np.clip(_extend(prob, n_ext, k_slope), 0.0, 1.0)
        cls_ext = np.full(n_ext, int(cls_eval[-1]), dtype=np.int8)

        days_all = pd.DatetimeIndex(pd.to_datetime(s["day"]).tolist() + [pd.Timestamp(t) for t in ext_days])
        batt_all = np.concatenate([y_pred_b, batt_ext])
        prob_all = np.concatenate([prob, risk_ext])
        cls_all = np.concatenate([cls_eval.astype(np.int8), cls_ext])

        r = (batt_all >= tau_b).astype(np.int8)
        h = (prob_all >= tau_h).astype(np.int8)
        states = (r * 2 + h).astype(np.int8)

        state_names = ("R0_H0", "R0_H1", "R1_H0", "R1_H1")
        state_colors = ("#6d4c41", "#c62828", "#2e7d32", "#f9a825")
        state_legend_lines = (
            "R0 H0 = Below threshold, halt flag off (mixed)",
            "R0 H1 = Below threshold, halt flag on (deprecated)",
            "R1 H0 = Above threshold, halt off (operational)",
            "R1 H1 = Above threshold, halt on (mixed)",
        )

        fig = plt.figure(figsize=(22.5, 24.0))
        gs = gridspec.GridSpec(3, 1, figure=fig, height_ratios=[2.0, 1.15, 0.52], hspace=0.14)
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
        ax3 = fig.add_subplot(gs[2, 0], sharex=ax1)

        ax1.plot(s["day"], y_true_b, color="#1f77b4", lw=1.8, label="True battery")
        ax1.plot(s["day"], y_pred_b, color="#ff7f0e", lw=1.6, alpha=0.95, label="Pred battery")
        ax1.plot(ext_days, batt_ext, color="#ff7f0e", lw=1.8, ls="--", alpha=0.95, label="Pred battery (horizon)")
        ax1.axhline(tau_b, color="#6a1b9a", ls=":", lw=1.1, label=r"$\tau_{\mathrm{batt}}$")
        ax1.axvline(last_day, color="black", lw=1.0, ls=":")
        ax1.axvspan(last_day, ext_days[-1], color="gray", alpha=0.10)
        pos_mask = (s["y_true_halt_within_h"].astype(float) > 0).values
        if pos_mask.any():
            blocks: list[tuple[int, int]] = []
            start = None
            for i, flag in enumerate(pos_mask):
                if flag and start is None:
                    start = i
                if (not flag) and (start is not None):
                    blocks.append((start, i - 1))
                    start = None
            if start is not None:
                blocks.append((start, len(pos_mask) - 1))
            for bi, (a, b) in enumerate(blocks):
                ax1.axvspan(s["day"].iloc[a], s["day"].iloc[b], color="#d62728", alpha=0.14, label="True halt window" if bi == 0 else None)
        ax1.set_ylabel("Battery voltage (V)", fontsize=24)
        ax1.grid(alpha=0.2)
        ax1.set_title("Deep learning time-series integration", fontsize=35)

        ax2.plot(s["day"], prob, color="#2ca02c", lw=2.0, alpha=0.95, label="Pred halt prob")
        ax2.plot(ext_days, risk_ext, color="#2ca02c", lw=2.0, ls="--", alpha=0.95, label="Pred halt prob (horizon)")
        ax2.axvline(last_day, color="black", lw=1.0, ls=":")
        ax2.axvspan(last_day, ext_days[-1], color="gray", alpha=0.10)
        ax2.axhspan(tau_h, 1.0, color="#3949ab", alpha=0.12, label="Halt decision band")
        ax2.axhline(tau_h, color="#1a237e", ls="-", lw=0.8, alpha=0.75)
        ax2.set_ylim(0.0, 1.0)
        ax2.set_ylabel("Halt probability", fontsize=24)
        ax2.grid(alpha=0.2)

        for i in range(len(states)):
            c = state_colors[int(states[i])]
            d0 = days_all[i]
            d1 = days_all[i + 1] if i + 1 < len(days_all) else d0 + pd.Timedelta(days=1)
            ax3.axvspan(d0, d1, color=c, alpha=0.88, lw=0)
        ax3.set_yticks([])
        ax3.set_ylabel("(R,H)\nstate", fontsize=24)
        ax3.set_xlabel("Date", fontsize=24)
        ax1.tick_params(axis="both", labelsize=19)
        ax2.tick_params(axis="both", labelsize=19)
        ax3.tick_params(axis="x", labelsize=19)

        h1, l1 = ax1.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax1.legend(h1 + h2, l1 + l2, loc="lower left", fontsize=17)
        patch_handles = [plt.Rectangle((0, 0), 1, 1, color=state_colors[i]) for i in range(4)]
        fig.legend(
            patch_handles,
            state_legend_lines,
            loc="lower center",
            ncol=2,
            bbox_to_anchor=(0.5, 0.31),
            fontsize=22.0,
            frameon=True,
            title="(R,H) state colors",
            title_fontsize=25,
            borderpad=1.2,
            labelspacing=0.9,
            handlelength=2.0,
        )
        # Use explicit subplot bounds: a second generic tight_layout() in _save()
        # can undo custom rect anchoring for this large bottom figure-legend case.
        fig.subplots_adjust(left=0.06, right=0.955, top=0.965, bottom=0.44, hspace=0.20)
        _save(fig, out_path, tight=False)
        return

    if figure_key == "method_model_ranking":
        d = _comp_df(PRIMARY_ARTIFACT_STEM).set_index("model")
        rank_df = pd.DataFrame(index=list(MODELS))
        rank_df["ap"] = d.loc[list(MODELS), "halt_ap"].astype(float)
        rank_df["recall"] = d.loc[list(MODELS), "halt_recall"].astype(float)
        rank_df["f2"] = d.loc[list(MODELS), "halt_f2"].astype(float)
        rank_df["brier"] = d.loc[list(MODELS), "halt_brier_tscaled"].astype(float)
        rank_df["fp"] = d.loc[list(MODELS), "halt_fp"].astype(float)
        rank_df["fn"] = d.loc[list(MODELS), "halt_fn"].astype(float)
        rank_df = rank_df.sort_values(["f2", "recall", "ap"], ascending=False)
        rank_df["rank"] = np.arange(1, len(rank_df) + 1, dtype=int)

        cols = ["Rank", "Model", "PR-AUC", "Recall", "F2", "Brier", "FP", "FN"]
        rows = []
        for model, r in rank_df.iterrows():
            rows.append(
                [
                    int(r["rank"]),
                    LABEL[model],
                    f"{float(r['ap']):.4f}",
                    f"{float(r['recall']):.4f}",
                    f"{float(r['f2']):.4f}",
                    f"{float(r['brier']):.4f}",
                    f"{int(round(float(r['fp'])))}",
                    f"{int(round(float(r['fn'])))}",
                ]
            )

        fig, ax = plt.subplots(figsize=(13.8, 4.6))
        ax.axis("off")
        t = ax.table(cellText=rows, colLabels=cols, cellLoc="center", loc="center")
        t.auto_set_font_size(False)
        t.set_fontsize(14)
        t.scale(1.0, 2.0)
        for c in range(len(cols)):
            t[(0, c)].set_facecolor("#D9E1F2")
            t[(0, c)].set_text_props(weight="bold")
        rank_col = cols.index("Rank")
        for r_i in range(1, len(rows) + 1):
            rk = int(rows[r_i - 1][rank_col])
            if rk == 1:
                bg = "#C8E6C9"
            elif rk == 2:
                bg = "#E8F5E9"
            elif rk == 3:
                bg = "#FFF8E1"
            else:
                bg = "#FFEBEE"
            for c in range(len(cols)):
                t[(r_i, c)].set_facecolor(bg)
        ax.set_title("Method model ranking", fontsize=21, pad=14)
        _save(fig, out_path)
        return

    raise ValueError(f"Unknown figure key: {figure_key}")
