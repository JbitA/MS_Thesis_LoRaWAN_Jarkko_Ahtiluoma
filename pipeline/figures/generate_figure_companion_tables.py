#!/usr/bin/env python3
"""
FILE STORY — ``generate_figure_companion_tables.py``
====================================================

**Role.** After thesis PNGs exist (or when ``NODE_ANALYSIS_COMPANION_ALLOW_NO_RECREATED``
is set), writes per-figure **companion artifacts** under
``graphs/<figure_key>/outputs/``: metrics CSV, markdown summary, and sheet PNG.

**Inputs.** Same training tables as ``recreated_from_new_data.py``; ribbon contingency
rows align thresholds/sensor logic with the ribbon figure renderer.

**Outputs (basenames from ``figure_titles.py``).** e.g. ``policy_map_metrics.csv``,
``stl_vs_mtl_delta_metrics.md``, extra tier-2 table for ``method_model_ranking``.

**Scan root.** ``NODE_ANALYSIS_FIG_SCAN_ROOT`` or ``GRAPH_ROOT`` — iterates
``thesis_figure_keys.iter_thesis_figure_dirs``.

**Developed with Cursor AI.**
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import util

import sys

_FIGURES_PKG = Path(__file__).resolve().parent
_BUNDLE_SRC = _FIGURES_PKG.parent / "src"
if str(_BUNDLE_SRC) not in sys.path:
    sys.path.insert(0, str(_BUNDLE_SRC))

from node_analysis_pipeline import env_config as env  # noqa: E402
from node_analysis_pipeline.artifact_stems import artifact_stem  # noqa: E402
from figure_titles import (  # noqa: E402
    FIGURE_MAIN_TITLE,
    companion_absolute_metrics_csv,
    companion_absolute_metrics_sheet_png,
    companion_absolute_metrics_sheet_title,
    companion_metrics_csv,
    companion_metrics_md,
    companion_metrics_sheet_png,
    companion_metrics_sheet_title,
)
from thesis_figure_keys import (  # noqa: E402
    THESIS_FIGURE_KEYS,
    iter_thesis_figure_dirs,
    list_figure_pngs,
    remove_legacy_companion_artifacts,
)

RUN_ID = env.get_str(env.ENV_RUN_ID, "node_analysis1")
BUNDLE_ROOT = env.pipeline_root_from_env(fallback=_FIGURES_PKG.parent)
_gr = env.get_str(env.ENV_GRAPH_ROOT, "")
FIG_SCAN_ROOT = Path(env.get_str(env.ENV_FIG_SCAN_ROOT, _gr or str(_FIGURES_PKG))).resolve()
TABLES = Path(env.get_str(env.ENV_OUTPUT_ROOT, str(BUNDLE_ROOT / "outputs"))) / "tables"

MODELS = ("lstm", "gru", "tcn", "transformer")
LABEL = {"lstm": "LSTM", "gru": "GRU", "tcn": "TCN", "transformer": "Transformer"}
# Ordered metric ids for stl_vs_mtl_delta companion CSV (`metric` column).
FIG_SINGLE_VS_MULTITASK_METRICS = (
    "delta_rmse_mtl_minus_stl_batt",
    "delta_r2_mtl_minus_stl_batt",
    "delta_auc_mtl_minus_stl_hhalt",
    "delta_ap_mtl_minus_stl_hhalt",
    "delta_f2_mtl_minus_stl_hhalt",
    "delta_brier_raw_mtl_minus_stl_hhalt",
)
# Short row titles for table_metrics_sheet.png only (no Δ / no STL baseline suffix).
DELTA_METRIC_SHEET_ROWS = {
    "delta_rmse_mtl_minus_stl_batt": "RMSE",
    "delta_r2_mtl_minus_stl_batt": "R²",
    "delta_auc_mtl_minus_stl_hhalt": "AUC",
    "delta_ap_mtl_minus_stl_hhalt": "AP",
    "delta_f2_mtl_minus_stl_hhalt": "F2",
    "delta_brier_raw_mtl_minus_stl_hhalt": "BS",
}
# Absolute MTL vs STL rows (battery metrics from STL-batt; halt metrics from STL-hhalt).
FIG_SINGLE_VS_MULTITASK_MTL_STL_SPECS: tuple[tuple[str, str, str, str], ...] = (
    ("RMSE", "battery_rmse", "stl_b", "battery_rmse"),
    ("R²", "battery_r2", "stl_b", "battery_r2"),
    ("AUC", "halt_auc", "stl_h", "halt_auc"),
    ("AP", "halt_ap", "stl_h", "halt_ap"),
    ("F2", "halt_f2", "stl_h", "halt_f2"),
    ("BS", "halt_brier", "stl_h", "halt_brier"),
)
# policy_map: rows = Fβ quantities, columns = model (after transpose).
POLICY_MAP_SWEEP_ROWS: tuple[tuple[str, str, str], ...] = (
    ("f05", "threshold", "F0.5 τ"),
    ("f05", "precision", "F0.5 precision"),
    ("f05", "recall", "F0.5 recall"),
    ("f1", "threshold", "F1 τ"),
    ("f1", "precision", "F1 precision"),
    ("f1", "recall", "F1 recall"),
    ("f2", "threshold", "F2 τ"),
    ("f2", "precision", "F2 precision"),
    ("f2", "recall", "F2 recall"),
)


def _companion_allow_no_recreated_png() -> bool:
    return env.truthy(env.ENV_COMPANION_ALLOW_NO_RECREATED)


# ---------------------------------------------------------------------------
# Ribbon sensor selection (must match recreated_from_new_data.py)
# ---------------------------------------------------------------------------


def _vc_ribbon_observed_frame(bundle_root: Path, tables: Path, run_id: str) -> pd.DataFrame:
    """Match `recreated_from_new_data.py` fig_thesis01_vC_ribbon sensor selection (no sensor id in outputs)."""
    p = tables / f"{artifact_stem(run_id, 'multitask_binary')}_transformer_battery_halt_predictions.csv"
    d = pd.read_csv(p)
    d["day"] = pd.to_datetime(d["day"], errors="coerce")
    meta_csv = util.RIBBON_CHART_DEFAULT_SENSOR_CSV
    if not meta_csv.is_file():
        raise RuntimeError(f"Missing source metadata CSV: {meta_csv}")
    meta = pd.read_csv(meta_csv)
    sensor = str(env.get_str(env.ENV_VC_SENSOR, str(meta.loc[0, "sensor"]))).strip()
    s = d[d["sensor"].astype(str) == sensor].dropna(subset=["day"]).sort_values("day").copy()
    if s.empty:
        best_sensor = None
        best_score = -1e18
        for sid, g in d.groupby(d["sensor"].astype(str)):
            g = g.sort_values("day").dropna(subset=["y_true_battery", "y_pred_halt_prob"])
            n = len(g)
            if n < 80:
                continue
            b = g["y_true_battery"].to_numpy(float)
            pr = g["y_pred_halt_prob"].to_numpy(float)
            kk = max(10, int(0.15 * n))
            b_head = float(np.nanmedian(b[:kk]))
            b_tail = float(np.nanmedian(b[-kk:]))
            p_head = float(np.nanmedian(pr[:kk]))
            p_tail = float(np.nanmedian(pr[-kk:]))
            drop = b_head - b_tail
            rise = p_tail - p_head
            end_hi = float(np.mean(pr[-kk:] >= 0.8))
            peak = float(np.nanmax(pr[-kk:]))
            score = 2.0 * drop + 1.5 * rise + 0.5 * end_hi + 0.3 * peak
            if np.isfinite(score) and score > best_score:
                best_score = score
                best_sensor = str(sid)
        if best_sensor is not None:
            sensor = best_sensor
        else:
            sensor = str(d["sensor"].astype(str).value_counts().index[0])
        s = d[d["sensor"].astype(str) == sensor].dropna(subset=["day"]).sort_values("day").copy()
    return s


def _primary_stem() -> str:
    return artifact_stem(RUN_ID, "multitask_binary")


def _mixed_policy() -> pd.DataFrame:
    return pd.read_csv(TABLES / f"{_primary_stem()}_model_comparison_battery_halt.csv")



def _markdown_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    body = []
    for _, r in df.iterrows():
        vals = []
        for c in cols:
            v = r[c]
            if isinstance(v, float):
                vals.append(f"{v:.6f}")
            else:
                vals.append(str(v))
        body.append("| " + " | ".join(vals) + " |")
    return "\n".join([header, sep] + body)


def _mixed_stems() -> tuple[str, str]:
    s = _primary_stem()
    return s, s



def _mixed_model_comparison() -> pd.DataFrame:
    mtl = _mixed_policy().set_index("model")
    return mtl.reset_index().rename(columns={"index": "model"})


def _build_table_for_figure_key(figure_key: str, base: pd.DataFrame, mtl: pd.DataFrame, stl_b: pd.DataFrame, stl_h: pd.DataFrame) -> pd.DataFrame:
    d = base.copy()
    if figure_key in {
        "imputed_vs_non_imputed_halt",
        "imputed_panel",
        "threshold_f2_impute_sensitivity_comparison",
    }:
        imp_stem = _primary_stem()
        non_stem = artifact_stem(RUN_ID, "multitask_binary_no_impute")
        d_imp = pd.read_csv(TABLES / f"{imp_stem}_model_comparison_battery_halt.csv").set_index("model")
        d_non = pd.read_csv(TABLES / f"{non_stem}_model_comparison_battery_halt.csv").set_index("model")
        rows = []
        for m in MODELS:
            imp = d_imp.loc[m]
            non = d_non.loc[m]
            rows.append(
                {
                    "model": m,
                    "imputed_f2": float(imp["halt_f2"]),
                    "nonimputed_f2": float(non["halt_f2"]),
                    "imputed_brier_tscaled": float(imp["halt_brier_tscaled"]),
                    "nonimputed_brier_tscaled": float(non["halt_brier_tscaled"]),
                    "delta_f2_imputed_minus_nonimputed": float(imp["halt_f2"] - non["halt_f2"]),
                    "delta_rmse_imputed_minus_nonimputed": float(imp["battery_rmse"] - non["battery_rmse"]),
                }
            )
        return pd.DataFrame(rows)
    if figure_key == "battery_missingness":
        p = TABLES / f"{_primary_stem()}_transformer_battery_halt_predictions.csv"
        df = pd.read_csv(p)
        sid = str(df["sensor"].astype(str).value_counts().index[0])
        g = df[df["sensor"].astype(str) == sid].copy()
        g["day"] = pd.to_datetime(g["day"], errors="coerce")
        g = g.dropna(subset=["day"]).sort_values("day")
        y = g["y_true_battery"].astype(float).to_numpy()
        k = max(10, int(0.15 * len(y)))
        drop = float(np.nanmedian(y[:k]) - np.nanmedian(y[-k:]))
        return pd.DataFrame(
            [
                {
                    "sensor_used": sid,
                    "n_days": int(len(g)),
                    "tail_battery_drop": drop,
                    "min_battery": float(np.nanmin(y)),
                    "max_battery": float(np.nanmax(y)),
                }
            ]
        )
    if figure_key == "deep_learning_time_series_integration":
        # Align thresholds with `recreated_from_new_data.py` ribbon strip (τ on true batt / pred prob).
        s = _vc_ribbon_observed_frame(BUNDLE_ROOT, TABLES, RUN_ID)
        tau_b_q = 0.10
        tau_h_q = 0.85
        y_true_b = s["y_true_battery"].to_numpy(float)
        y_pred_b = s["y_pred_battery"].to_numpy(float)
        prob = s["y_pred_halt_prob"].to_numpy(float)
        k = max(10, int(0.15 * len(s)))
        tau_b = float(np.nanquantile(y_true_b, tau_b_q))
        tau_h = float(np.nanquantile(prob, tau_h_q))
        if not np.isfinite(tau_b):
            tau_b = float(np.nanmedian(y_true_b))
        if not np.isfinite(tau_h):
            tau_h = 0.5
        batt_head = float(np.nanmedian(y_true_b[:k]))
        batt_tail = float(np.nanmedian(y_true_b[-k:]))
        p_head = float(np.nanmedian(prob[:k]))
        p_tail = float(np.nanmedian(prob[-k:]))
        tail_drop = batt_head - batt_tail
        tail_rise = p_tail - p_head
        tail_peak = float(np.nanmax(prob[-k:]))
        r = (y_pred_b >= tau_b).astype(np.int8)
        h = (prob >= tau_h).astype(np.int8)
        n00 = int(np.sum((r == 0) & (h == 0)))
        n01 = int(np.sum((r == 0) & (h == 1)))
        n10 = int(np.sum((r == 1) & (h == 0)))
        n11 = int(np.sum((r == 1) & (h == 1)))
        rows = [
            ("N observed days", str(int(len(s)))),
            ("Tail window k (days): max(10, ⌊0.15·N⌋)", str(int(k))),
            ("τ battery (V): q=0.10 of true battery (regression reference line)", f"{tau_b:.6f}"),
            ("τ halt probability: q=0.85 of predicted prob on this series (classification band)", f"{tau_h:.6f}"),
            ("Tail battery drop (V): median(early-k true) − median(late-k true)", f"{tail_drop:.6f}"),
            ("Tail risk rise: median(late-k pred prob) − median(early-k pred prob)", f"{tail_rise:.6f}"),
            ("Tail peak risk: max predicted prob in tail window", f"{tail_peak:.6f}"),
            ("(R,H) contingency — observed days only (see matrix below)", ""),
            ("R=0, H=0", str(n00)),
            ("R=0, H=1", str(n01)),
            ("R=1, H=0", str(n10)),
            ("R=1, H=1", str(n11)),
        ]
        out = pd.DataFrame(rows, columns=["quantity", "value"])
        out.attrs["vc_contingency"] = (n00, n01, n10, n11)
        out.attrs["vc_tau_b"] = tau_b
        out.attrs["vc_tau_h"] = tau_h
        return out
    # Default fallback (still result-dependent, but generic).
    return d[["model", "halt_ap", "halt_recall", "halt_f2", "halt_brier_tscaled", "battery_rmse"]].copy()


def _build_mtl_stl_absolute_table(
    mtl: pd.DataFrame,
    stl_b: pd.DataFrame,
    stl_h: pd.DataFrame,
) -> pd.DataFrame:
    """Absolute MTL vs STL: one row per architecture × training mode; columns = metrics."""
    stl_pick = {"stl_b": stl_b, "stl_h": stl_h}
    metric_cols = [spec[0] for spec in FIG_SINGLE_VS_MULTITASK_MTL_STL_SPECS]
    cols = ["training"] + metric_cols
    rows: list[dict[str, object]] = []
    for m in MODELS:
        r_mtl: dict[str, object] = {"training": f"{LABEL[m]} MTL"}
        r_stl: dict[str, object] = {"training": f"{LABEL[m]} STL"}
        for disp, mtl_col, which, stl_col in FIG_SINGLE_VS_MULTITASK_MTL_STL_SPECS:
            stl_df = stl_pick[which]
            r_mtl[disp] = float(mtl.loc[m, mtl_col])
            r_stl[disp] = float(stl_df.loc[m, stl_col])
        rows.append(r_mtl)
        rows.append(r_stl)
    return pd.DataFrame(rows, columns=cols)


def _sheet_png_mtl_stl_absolute(out: Path, df: pd.DataFrame) -> None:
    """Sheet: rows = model×(MTL|STL); columns = metrics (absolute, not deltas)."""

    def _fmt_cell(v: object) -> str:
        try:
            return f"{float(v):.4f}"
        except Exception:
            return str(v)

    tab = df.copy()
    first = tab.columns[0]
    for c in tab.columns:
        if c == first:
            continue
        tab[c] = tab[c].map(_fmt_cell)
    tab = tab.rename(columns={first: "Model"})

    fig, ax = plt.subplots(figsize=(10.2, 5.35))
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(companion_absolute_metrics_sheet_title("stl_vs_mtl_delta"), fontsize=12, pad=10)
    t = ax.table(
        cellText=tab.values.tolist(),
        colLabels=list(tab.columns),
        cellLoc="center",
        bbox=[0.03, 0.04, 0.94, 0.88],
    )
    t.auto_set_font_size(False)
    t.set_fontsize(9)
    t.scale(1, 1.55)
    for ci in range(len(tab.columns)):
        t[(0, ci)].set_facecolor("#D9E1F2")
        t[(0, ci)].set_text_props(weight="bold")
    for ri in range(1, len(tab) + 1):
        shade = "#FFFFFF" if ri % 2 else "#F7F7F7"
        for ci in range(len(tab.columns)):
            t[(ri, ci)].set_facecolor(shade)
            if ci == 0:
                t[(ri, ci)].set_text_props(weight="bold")
    fig.tight_layout()
    fig.savefig(out / companion_absolute_metrics_sheet_png("stl_vs_mtl_delta"), dpi=220)
    plt.close(fig)



def _analysis_path() -> Path:
    _out = env.get_str(env.ENV_OUTPUT_ROOT, "")
    if _out:
        return Path(_out) / "reports" / f"FIGURE_TABLE_ANALYSIS_{RUN_ID}.md"
    return BUNDLE_ROOT.parent / "outputs" / RUN_ID / "models" / "reports" / f"FIGURE_TABLE_ANALYSIS_{RUN_ID}.md"


def main() -> None:
    mtl = pd.read_csv(TABLES / f"{artifact_stem(RUN_ID, 'multitask_binary')}_model_comparison_battery_halt.csv").set_index("model")
    stl_b = pd.read_csv(TABLES / f"{artifact_stem(RUN_ID, 'single_task_battery')}_model_comparison_battery_halt.csv").set_index("model")
    stl_h = pd.read_csv(TABLES / f"{artifact_stem(RUN_ID, 'single_task_halt_focus')}_model_comparison_battery_halt.csv").set_index("model")
    base = _mixed_model_comparison()

    # Global key values requested by supervisor.
    best_ap = base.loc[base["halt_ap"].astype(float).idxmax()]
    best_rec = base.loc[base["halt_recall"].astype(float).idxmax()]
    best_f2 = base.loc[base["halt_f2"].astype(float).idxmax()]
    best_brier = base.loc[base["halt_brier_tscaled"].astype(float).idxmin()]
    delta_rows = []
    for m in MODELS:
        delta_rows.append(
            {
                "model": m,
                "delta_f2_mtl_minus_stl_hhalt": float(mtl.loc[m, "halt_f2"] - stl_h.loc[m, "halt_f2"]),
                "delta_rmse_mtl_minus_stl_batt": float(mtl.loc[m, "battery_rmse"] - stl_b.loc[m, "battery_rmse"]),
            }
        )
    delta_df = pd.DataFrame(delta_rows)
    best_delta = delta_df.loc[delta_df["delta_f2_mtl_minus_stl_hhalt"].abs().idxmax()]

    key_lines = [
        f"- Best PR-AUC: {LABEL[best_ap['model']]} = {float(best_ap['halt_ap']):.4f}",
        f"- Best recall: {LABEL[best_rec['model']]} = {float(best_rec['halt_recall']):.4f}",
        f"- Best F2: {LABEL[best_f2['model']]} = {float(best_f2['halt_f2']):.4f}",
        f"- Best Brier (temp-scaled): {LABEL[best_brier['model']]} = {float(best_brier['halt_brier_tscaled']):.4f}",
        (
            "- Main MTL/STL delta: "
            f"{LABEL[str(best_delta['model'])]} -> "
            f"ΔF2(MTL-STL_hhalt)={float(best_delta['delta_f2_mtl_minus_stl_hhalt']):+.4f}, "
            f"ΔRMSE(MTL-STL_batt)={float(best_delta['delta_rmse_mtl_minus_stl_batt']):+.4f}"
        ),
    ]

    def _fmt_cell(v):
        try:
            return f"{float(v):.4f}"
        except Exception:
            return str(v)

    def _sheet_png(fig_dir: Path, out: Path, metric_df: pd.DataFrame, key_rows: list[str]) -> None:
        d = metric_df.copy()
        if fig_dir.name == "deep_learning_time_series_integration" and list(d.columns) == ["quantity", "value"]:
            n00, n01, n10, n11 = d.attrs.get("vc_contingency", (0, 0, 0, 0))
            fig, ax = plt.subplots(figsize=(5.8, 3.4))
            ax.axis("off")
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_title(
                "Deep learning time-series contingency matrix",
                fontsize=12,
                fontweight="normal",
                pad=1,
            )
            cont = [
                ["", "H=0", "H=1"],
                ["R=0", str(n00), str(n01)],
                ["R=1", str(n10), str(n11)],
            ]
            t = ax.table(
                cellText=cont,
                cellLoc="center",
                bbox=[0.05, 0.06, 0.90, 0.84],
            )
            t.auto_set_font_size(False)
            t.set_fontsize(12)
            t.scale(1, 2.0)
            for ci in range(3):
                t[(0, ci)].set_facecolor("#D9E1F2")
                t[(0, ci)].set_text_props(weight="bold")
            for ri in range(1, 3):
                t[(ri, 0)].set_facecolor("#E8E8E8")
                t[(ri, 0)].set_text_props(weight="bold")
                for ci in range(1, 3):
                    # Match fig_thesis01_vC_ribbon (R,H) state strip colors:
                    # R0H0 brown, R0H1 red, R1H0 green, R1H1 amber.
                    state_fill = {
                        (1, 1): "#6d4c41",  # R=0,H=0
                        (1, 2): "#c62828",  # R=0,H=1
                        (2, 1): "#2e7d32",  # R=1,H=0
                        (2, 2): "#f9a825",  # R=1,H=1
                    }[(ri, ci)]
                    t[(ri, ci)].set_facecolor(state_fill)
                    t[(ri, ci)].set_text_props(weight="bold", color="white")
            fig.subplots_adjust(left=0.01, right=0.99, top=0.92, bottom=0.03)
            fig.savefig(out / companion_metrics_sheet_png(fig_dir.name), dpi=220)
            plt.close(fig)
            return

        # Single-vs-multitask delta: metrics as rows, model architectures as columns.
        if fig_dir.name == "stl_vs_mtl_delta" and "metric" in d.columns and "model" not in d.columns:
            metric_disp = d["metric"].map(lambda k: DELTA_METRIC_SHEET_ROWS.get(str(k), str(k)))
            num = d.drop(columns=["metric"]).copy()
            for c in num.columns:
                num[c] = num[c].map(_fmt_cell)
            d = pd.concat(
                [metric_disp.rename("Metric delta"), num.rename(columns={m: LABEL[m] for m in num.columns})],
                axis=1,
            )
            fig, ax = plt.subplots(figsize=(7.8, 3.4))
        elif fig_dir.name == "policy_map" and "metric" in d.columns and "model" not in d.columns:
            mcol = d["metric"]
            num = d.drop(columns=["metric"]).copy()
            for c in num.columns:
                num[c] = num[c].map(_fmt_cell)
            d = pd.concat(
                [mcol.rename("Metric"), num.rename(columns={m: LABEL[m] for m in num.columns})],
                axis=1,
            )
            fig, ax = plt.subplots(figsize=(10.8, 9.6))
        else:
            if "model" in d.columns:
                d["model"] = d["model"].map(lambda m: LABEL.get(str(m), str(m)))
            for c in d.columns:
                if c == "model":
                    continue
                d[c] = d[c].map(_fmt_cell)
            fig, ax = plt.subplots(figsize=(12, 4.2))

        ax.axis("off")
        if fig_dir.name == "stl_vs_mtl_delta" and "Metric delta" in d.columns:
            col_labels = list(d.columns)
        elif fig_dir.name == "policy_map" and "Metric" in d.columns:
            col_labels = list(d.columns)
        else:
            col_labels = [c.replace("_", " ").upper() for c in d.columns]
        if fig_dir.name == "policy_map":
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            _tb_left, _tb_w = 0.08, 0.84
            _tb_bottom, _tb_h = 0.03, 0.85
            _tb_top = _tb_bottom + _tb_h
            ax.text(
                0.5,
                _tb_top + 0.012,
                FIGURE_MAIN_TITLE["policy_map"],
                transform=ax.transAxes,
                ha="center",
                va="bottom",
                fontsize=18,
                fontweight="bold",
            )
            t = ax.table(
                cellText=d.values.tolist(),
                colLabels=col_labels,
                cellLoc="center",
                bbox=[_tb_left, _tb_bottom, _tb_w, _tb_h],
            )
        elif fig_dir.name == "stl_vs_mtl_delta" and "Metric delta" in d.columns:
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            t = ax.table(
                cellText=d.values.tolist(),
                colLabels=col_labels,
                cellLoc="center",
                bbox=[0.08, 0.08, 0.84, 0.80],
            )
        else:
            t = ax.table(cellText=d.values.tolist(), colLabels=col_labels, cellLoc="center", loc="center")
        t.auto_set_font_size(False)
        if fig_dir.name == "policy_map":
            t.set_fontsize(12)
            t.scale(1, 2.24)
        else:
            t.set_fontsize(10)
            t.scale(1, 1.5)
        for ci in range(len(col_labels)):
            t[(0, ci)].set_facecolor("#D9E1F2")
            if fig_dir.name == "policy_map":
                t[(0, ci)].set_text_props(weight="bold", fontsize=13)
            else:
                t[(0, ci)].set_text_props(weight="bold")
        for ri in range(1, len(d) + 1):
            shade = "#FFFFFF" if ri % 2 else "#F7F7F7"
            for ci in range(len(col_labels)):
                t[(ri, ci)].set_facecolor(shade)
                if fig_dir.name == "policy_map":
                    t[(ri, ci)].set_text_props(fontsize=12)
        if fig_dir.name == "policy_map":
            fig.subplots_adjust(left=0.10, right=0.90, top=0.94, bottom=0.06)
        else:
            if fig_dir.name == "stl_vs_mtl_delta":
                ax.set_title("Deep learning MTL and STL performance deltas", fontsize=12, pad=10)
            else:
                ax.set_title(f"{fig_dir.name} — result-dependent metrics sheet", fontsize=12, pad=10)
            footer_rows = key_rows
            if fig_dir.name == "stl_vs_mtl_delta":
                footer_rows = []
            if footer_rows:
                fig.text(0.01, 0.01, " | ".join(footer_rows), fontsize=8)
            fig.tight_layout()
        fig.savefig(out / companion_metrics_sheet_png(fig_dir.name), dpi=220)
        plt.close(fig)

    for fig_dir in iter_thesis_figure_dirs(FIG_SCAN_ROOT):
        out = fig_dir / "outputs"
        if not out.is_dir():
            continue
        recreated_pngs = list_figure_pngs(out, fig_dir.name)
        if not recreated_pngs and not _companion_allow_no_recreated_png():
            continue
        metric_df = _build_table_for_figure_key(fig_dir.name, base, mtl, stl_b, stl_h)
        if metric_df.empty:
            continue
        slug = fig_dir.name
        tab = out / companion_metrics_csv(slug)
        metric_df.to_csv(tab, index=False)
        md = out / companion_metrics_md(slug)
        lines = [f"# Metrics Companion — {FIGURE_MAIN_TITLE[slug]}", ""]
        lines.append("Recreated figures in this folder:")
        if recreated_pngs:
            for p in recreated_pngs:
                lines.append(f"- `{p.name}`")
        else:
            lines.append(
                "- *(no `recreated*.png` in this folder; companions generated with `HOL_ACAD_COMPANION_ALLOW_NO_RECREATED`)*"
            )
        lines += ["", "## Clean table (multitask binary pass)"]
        lines.append(_markdown_table(metric_df))
        mtl_stl_df: pd.DataFrame | None = None
        if fig_dir.name == "stl_vs_mtl_delta":
            mtl_stl_df = _build_mtl_stl_absolute_table(mtl, stl_b, stl_h)
            mtl_stl_df.to_csv(out / companion_absolute_metrics_csv(slug), index=False)
            lines += [
                "",
                "## MTL and STL (absolute, not delta)",
                "",
                "Battery metrics (RMSE, R²): MTL vs **STL trained on battery only**. "
                "Halt metrics (AUC, AP, F2, BS): MTL vs **STL trained on halt only**.",
                "",
            ]
            md_disp = mtl_stl_df.rename(columns={"training": "Model"})
            lines.append(_markdown_table(md_disp))
            lines.append("")
            lines.append("Companion sheet PNG: `{companion_absolute_metrics_sheet_png(slug)}`")
        if fig_dir.name == "deep_learning_time_series_integration" and "vc_contingency" in metric_df.attrs:
            n00, n01, n10, n11 = metric_df.attrs["vc_contingency"]
            lines += [
                "",
                "## (R,H) contingency (observed days)",
                "",
                "Rows: R from predicted battery vs τ_batt (q=0.10 of true battery). Columns: H from predicted halt prob vs τ_halt (q=0.85 of pred prob on series).",
                "",
                "|  | H=0 | H=1 |",
                "| --- | --- | --- |",
                f"| R=0 | {n00} | {n01} |",
                f"| R=1 | {n10} | {n11} |",
            ]
        if fig_dir.name != "policy_map":
            lines += ["", "## Key values (after figure/table)"]
            lines += key_lines
        md.write_text("\n".join(lines), encoding="utf-8")
        _sheet_png(fig_dir, out, metric_df, key_lines)
        if fig_dir.name == "stl_vs_mtl_delta" and mtl_stl_df is not None:
            _sheet_png_mtl_stl_absolute(out, mtl_stl_df)
        remove_legacy_companion_artifacts(out, slug)

    # Consolidated analysis text requested.
    analysis = [
        "# Figure-Level Metrics Analysis",
        "",
        f"Run id: `{RUN_ID}`",
        "",
        "Supervisor-requested key values:",
        *key_lines,
        "",
        "Temperature scaling impact view:",
        "- Compare `halt_brier` vs `halt_brier_tscaled` in each per-figure `table_metrics_clean.csv`.",
        "- Compare `halt_brier` vs `halt_brier_tscaled` in per-figure `*_metrics.csv` companions where halt metrics apply.",
    ]
    _analysis_path().write_text("\n".join(analysis), encoding="utf-8")


if __name__ == "__main__":
    main()
