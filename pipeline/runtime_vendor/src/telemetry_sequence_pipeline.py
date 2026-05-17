#!/usr/bin/env python3
"""
FILE STORY — ``telemetry_sequence_pipeline.py``
================================================

**Role.** Daily panel + sequence tensors from ``idata/*/application.csv``; label rules and feature columns.

**Connects.** Imported by trainer bundle and patched by ``train_multitask_no_impute.py``.

**Developed with Cursor AI.**
"""
from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_curve,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader, TensorDataset


# ---------------------------------------------------------------------------
# Repository paths, output roots, and SeqData container
# ---------------------------------------------------------------------------

_HERE = Path(__file__).resolve()
_BUNDLE_ROOT = _HERE.parents[2]
_THESIS_PIPELINE = next((p for p in _BUNDLE_ROOT.parents if p.name == "thesis_pipeline"), None)

_default_data_root = (
    (_BUNDLE_ROOT.parent / "idata")
    if (_BUNDLE_ROOT.parent / "idata").is_dir()
    else (_BUNDLE_ROOT / "idata")
)
_default_output_root = _BUNDLE_ROOT / "outputs"

DATA_ROOT = Path(os.environ.get("HOL_ACAD_DATA_ROOT", str(_default_data_root)))
OUTPUT_ROOT = Path(os.environ.get("HOL_ACAD_OUTPUT_ROOT", str(_default_output_root)))
OUT_TABLES = OUTPUT_ROOT / "tables"
OUT_FIGS = OUTPUT_ROOT / "figures"
OUT_REPORTS = OUTPUT_ROOT / "reports"


@dataclass
class SeqData:
    x: np.ndarray
    y_batt: np.ndarray
    y_halt: np.ndarray
    day_index: np.ndarray
    sensor_id: np.ndarray
    day_label: np.ndarray
    # Days from window end t to last active day; nan if no last_active; used for discrete survival (B04).
    y_halt_dt: np.ndarray | None = None
    # Next-day daily uplink count (B07 Poisson aux).
    y_next_count: np.ndarray | None = None


def ensure_dirs() -> None:
    OUT_TABLES.mkdir(parents=True, exist_ok=True)
    OUT_FIGS.mkdir(parents=True, exist_ok=True)
    OUT_REPORTS.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Daily panel ingestion from idata/*/application.csv
# ---------------------------------------------------------------------------


def aggregate_daily_panel() -> pd.DataFrame:
    """
    All devices: per-(deveui, day) uplink count and mean battery (before calendar reindex / features).
    Used for cohort bookkeeping (top-K by volume vs held-out list) without building full tensors.
    """
    files = sorted(
        p
        for p in DATA_ROOT.glob("*/application.csv")
        if not p.parent.name.startswith("_")
    )
    if not files:
        raise FileNotFoundError(
            f"No device application.csv files under {DATA_ROOT} "
            "(add <deveui>/application.csv; see idata/README.md)."
        )
    chunks: List[pd.DataFrame] = []
    for f in files:
        chunks.append(pd.read_csv(f, usecols=["time", "deveui", "battery"]))
    df = pd.concat(chunks, ignore_index=True)
    df["time"] = pd.to_datetime(df["time"], unit="ms", utc=True, errors="coerce")
    df = df.dropna(subset=["time", "deveui"]).copy()
    df["deveui"] = df["deveui"].astype(str)
    df["slot_15m"] = df["time"].dt.floor("15min")
    df = (
        df.groupby(["deveui", "slot_15m"], as_index=False)
        .agg(battery=("battery", "mean"))
        .rename(columns={"slot_15m": "time"})
        .sort_values(["deveui", "time"])
    )
    df["day"] = df["time"].dt.floor("D")
    return (
        df.groupby(["deveui", "day"], as_index=False)
        .agg(daily_count=("time", "size"), battery_mean=("battery", "mean"))
        .sort_values(["deveui", "day"])
    )


def top_sensors_by_uplink_volume(daily: pd.DataFrame, k: int) -> List[str]:
    """Same ordering as load_original_daily(max_sensors=k) device filter."""
    if k <= 0:
        raise ValueError("k must be positive")
    return (
        daily.groupby("deveui", as_index=False)["daily_count"]
        .sum()
        .sort_values("daily_count", ascending=False)
        .head(k)["deveui"]
        .astype(str)
        .tolist()
    )


def load_original_daily(
    max_sensors: int | None = None,
    *,
    deveui_allowlist: List[str] | None = None,
) -> pd.DataFrame:
    """
    Build per-device daily panels with calendar reindex and features.

    - ``max_sensors``: keep top-N devices by **total uplink count** (default V7 behavior when set).
    - ``deveui_allowlist``: keep exactly these ``deveui`` values (mutually exclusive with ``max_sensors`` in practice;
      if both passed, allowlist wins).
    - Neither cap nor allowlist: **all** devices in ``application.csv``.
    """
    daily = aggregate_daily_panel()
    if deveui_allowlist is not None:
        allow = {str(x).strip() for x in deveui_allowlist if str(x).strip()}
        daily = daily[daily["deveui"].isin(allow)].copy()
    elif max_sensors is not None:
        keep = top_sensors_by_uplink_volume(daily, max_sensors)
        daily = daily[daily["deveui"].isin(keep)].copy()

    rows = []
    for sid, g in daily.groupby("deveui", sort=False):
        g = g.sort_values("day").copy()
        full_days = pd.date_range(g["day"].min(), g["day"].max(), freq="D", tz="UTC")
        gx = g.set_index("day").reindex(full_days).rename_axis("day").reset_index()
        gx["deveui"] = sid
        gx["daily_count"] = gx["daily_count"].fillna(0)
        gx["battery_mean"] = gx["battery_mean"].interpolate(limit_direction="both")
        gx["battery_mean"] = gx["battery_mean"].fillna(gx["battery_mean"].median())
        gx["battery_mean"] = gx["battery_mean"].fillna(3.6)
        gx["outage_flag"] = (gx["daily_count"] == 0).astype(float)
        gx["days_since_start"] = np.arange(len(gx), dtype=float)
        gx["cum_count"] = gx["daily_count"].cumsum()
        gx["roll7_count"] = gx["daily_count"].rolling(7, min_periods=1).mean()
        gx["batt_drop_from_start"] = float(gx["battery_mean"].iloc[0]) - gx["battery_mean"]
        gx["batt_roll7"] = gx["battery_mean"].rolling(7, min_periods=1).mean()
        gx["batt_delta1"] = gx["battery_mean"].diff().fillna(0.0)
        gx["count_delta1"] = gx["daily_count"].diff().fillna(0.0)
        gx["count_trend7"] = gx["daily_count"].rolling(7, min_periods=2).apply(
            lambda v: float(v.iloc[-1] - v.iloc[0]), raw=False
        ).fillna(0.0)
        rows.append(gx)
    return pd.concat(rows, ignore_index=True)


def last_active_index(daily_count: np.ndarray) -> int | None:
    nz = np.where(daily_count > 0)[0]
    if len(nz) == 0:
        return None
    return int(nz[-1])


# Per-day channels stacked along the sequence axis (order matches tensor layout in ``SeqData.x``).
SEQ_INPUT_FEATURE_COLS: Tuple[str, ...] = (
    "daily_count",
    "outage_flag",
    "cum_count",
    "roll7_count",
    "days_since_start",
    "battery_mean",
    "batt_roll7",
    "batt_delta1",
    "batt_drop_from_start",
    "count_delta1",
    "count_trend7",
)


# ---------------------------------------------------------------------------
# Sliding-window tensors, halt labels, and temporal split / standardize
# ---------------------------------------------------------------------------


def build_sequences(df: pd.DataFrame, seq_len: int, halt_horizon_days: int) -> SeqData:
    feature_cols = list(SEQ_INPUT_FEATURE_COLS)
    x_list, yb_list, yh_list, day_list, sid_list, day_label_list, dt_list, yc_list = [], [], [], [], [], [], [], []
    for sid, g in df.groupby("deveui", sort=False):
        g = g.sort_values("day").reset_index(drop=True)
        n = len(g)
        if n <= seq_len + 1:
            continue
        feats = g[feature_cols].values.astype(np.float32)
        daily_count = g["daily_count"].values.astype(float)
        batt = g["battery_mean"].values.astype(float)
        halt_idx = last_active_index(daily_count)

        for i in range(0, n - seq_len - 1):
            t = i + seq_len - 1
            y_batt = float(batt[t + 1])  # next-day battery
            yc_list.append(float(daily_count[t + 1]))

            if halt_idx is None:
                # Right-censored: skip very-end windows where horizon is unknown.
                if t > n - 1 - halt_horizon_days:
                    continue
                y_halt = 0.0
                dt_list.append(float("nan"))
            else:
                dt = halt_idx - t
                y_halt = 1.0 if 0 <= dt <= halt_horizon_days else 0.0
                dt_list.append(float(dt))

            x_list.append(feats[i : i + seq_len])
            yb_list.append(y_batt)
            yh_list.append(y_halt)
            day = g["day"].iloc[t]
            day_list.append(int(pd.Timestamp(day).value))
            sid_list.append(sid)
            day_label_list.append(pd.Timestamp(day).strftime("%Y-%m-%d"))

    return SeqData(
        x=np.asarray(x_list, dtype=np.float32),
        y_batt=np.asarray(yb_list, dtype=np.float32),
        y_halt=np.asarray(yh_list, dtype=np.float32),
        day_index=np.asarray(day_list, dtype=np.int64),
        sensor_id=np.asarray(sid_list, dtype=object),
        day_label=np.asarray(day_label_list, dtype=object),
        y_halt_dt=np.asarray(dt_list, dtype=np.float32),
        y_next_count=np.asarray(yc_list, dtype=np.float32),
    )


def temporal_split(day_index: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    uq = np.unique(day_index)
    uq.sort()
    n = len(uq)
    d_train = uq[: int(0.70 * n)]
    d_val = uq[int(0.70 * n) : int(0.85 * n)]
    d_test = uq[int(0.85 * n) :]
    tr = np.where(np.isin(day_index, d_train))[0]
    va = np.where(np.isin(day_index, d_val))[0]
    te = np.where(np.isin(day_index, d_test))[0]
    return tr, va, te


def standardize(train_x: np.ndarray, val_x: np.ndarray, test_x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu = train_x.reshape(-1, train_x.shape[-1]).mean(axis=0, keepdims=True)
    sd = train_x.reshape(-1, train_x.shape[-1]).std(axis=0, keepdims=True) + 1e-6
    return (train_x - mu) / sd, (val_x - mu) / sd, (test_x - mu) / sd


class LSTMHead(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.lstm = nn.LSTM(input_size=in_dim, hidden_size=hidden_dim, num_layers=2, batch_first=True, dropout=0.1)
        self.reg = nn.Linear(hidden_dim, 1)
        self.cls = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h, _ = self.lstm(x)
        z = h[:, -1, :]
        return self.reg(z).squeeze(-1), self.cls(z).squeeze(-1)


class TransformerHead(nn.Module):
    def __init__(self, in_dim: int, d_model: int = 64, nhead: int = 4, layers: int = 2):
        super().__init__()
        self.in_proj = nn.Linear(in_dim, d_model)
        enc = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=128, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc, num_layers=layers)
        self.reg = nn.Linear(d_model, 1)
        self.cls = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.in_proj(x)
        h = self.encoder(z)
        u = h[:, -1, :]
        return self.reg(u).squeeze(-1), self.cls(u).squeeze(-1)


def run_train(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, epochs: int, lr: float, device: torch.device, pos_weight: float) -> nn.Module:
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    mse = nn.MSELoss()
    bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], dtype=torch.float32, device=device))
    best_state = None
    best_val = float("inf")
    for _ in range(epochs):
        model.train()
        for xb, yb_reg, yb_cls in train_loader:
            xb = xb.to(device)
            yb_reg = yb_reg.to(device)
            yb_cls = yb_cls.to(device)
            pr_reg, pr_cls = model(xb)
            loss = mse(pr_reg, yb_reg) + bce(pr_cls, yb_cls)
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        losses = []
        with torch.no_grad():
            for xb, yb_reg, yb_cls in val_loader:
                xb = xb.to(device)
                yb_reg = yb_reg.to(device)
                yb_cls = yb_cls.to(device)
                pr_reg, pr_cls = model(xb)
                losses.append(float((mse(pr_reg, yb_reg) + bce(pr_cls, yb_cls)).cpu()))
        avg = float(np.mean(losses)) if losses else float("inf")
        if avg < best_val:
            best_val = avg
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def predict(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    pr_batt, pr_halt, yb, yh = [], [], [], []
    with torch.no_grad():
        for xb, yb_reg, yb_cls in loader:
            xb = xb.to(device)
            pb, ph = model(xb)
            pr_batt.append(pb.detach().cpu().numpy())
            pr_halt.append(torch.sigmoid(ph).detach().cpu().numpy())
            yb.append(yb_reg.numpy())
            yh.append(yb_cls.numpy())
    return (
        np.concatenate(pr_batt),
        np.concatenate(pr_halt),
        np.concatenate(yb),
        np.concatenate(yh),
    )


def best_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    best_t, best_f1 = 0.5, -1.0
    for t in np.linspace(0.05, 0.95, 37):
        y_hat = (y_prob >= t).astype(int)
        f1 = f1_score(y_true.astype(int), y_hat, average="binary", zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_t = float(t)
    return best_t


def eval_metrics(yb_true: np.ndarray, yb_pred: np.ndarray, yh_true: np.ndarray, yh_prob: np.ndarray, cls_t: float) -> Dict[str, float]:
    yh_true_i = yh_true.astype(int)
    yh_hat = (yh_prob >= cls_t).astype(int)
    cm = confusion_matrix(yh_true_i, yh_hat, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    out = {
        "battery_mae": float(mean_absolute_error(yb_true, yb_pred)),
        "battery_rmse": float(np.sqrt(mean_squared_error(yb_true, yb_pred))),
        "battery_r2": float(r2_score(yb_true, yb_pred)),
        "halt_auc": float(roc_auc_score(yh_true_i, yh_prob)) if len(np.unique(yh_true_i)) > 1 else float("nan"),
        "halt_ap": float(average_precision_score(yh_true_i, yh_prob)) if len(np.unique(yh_true_i)) > 1 else float("nan"),
        "halt_brier": float(brier_score_loss(yh_true_i, yh_prob)),
        "halt_precision": float(precision_score(yh_true_i, yh_hat, zero_division=0)),
        "halt_recall": float(recall_score(yh_true_i, yh_hat, zero_division=0)),
        "halt_f1": float(f1_score(yh_true_i, yh_hat, zero_division=0)),
        "halt_specificity": float(tn / max(tn + fp, 1)),
        "halt_prevalence": float(np.mean(yh_true_i)),
        "halt_threshold": cls_t,
        "halt_tp": int(tp),
        "halt_fp": int(fp),
        "halt_tn": int(tn),
        "halt_fn": int(fn),
    }
    return out


def save_curves(y_true: np.ndarray, y_prob: np.ndarray, model_name: str) -> None:
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    fig, ax = plt.subplots(1, 2, figsize=(10, 4.2))
    ax[0].plot(fpr, tpr, lw=2, color="#1565c0")
    ax[0].plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.6)
    ax[0].set_title(f"{model_name.upper()} halt ROC")
    ax[0].set_xlabel("False positive rate")
    ax[0].set_ylabel("True positive rate")
    ax[0].grid(True, alpha=0.3)
    ax[1].plot(rec, prec, lw=2, color="#e65100")
    ax[1].set_title(f"{model_name.upper()} halt PR")
    ax[1].set_xlabel("Recall")
    ax[1].set_ylabel("Precision")
    ax[1].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_FIGS / f"v2_{model_name}_halt_roc_pr.png", dpi=180)
    plt.close(fig)


def save_battery_scatter(y_true: np.ndarray, y_pred: np.ndarray, model_name: str) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.scatter(y_true, y_pred, s=8, alpha=0.3, color="#37474f")
    lo = float(min(np.min(y_true), np.min(y_pred)))
    hi = float(max(np.max(y_true), np.max(y_pred)))
    ax.plot([lo, hi], [lo, hi], "r--", lw=1.5)
    ax.set_title(f"{model_name.upper()} next-day battery: predicted vs true")
    ax.set_xlabel("True battery mean")
    ax.set_ylabel("Predicted battery mean")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_FIGS / f"v2_{model_name}_battery_scatter.png", dpi=180)
    plt.close(fig)


def write_report(df_metrics: pd.DataFrame, n_samples: int, n_sensors: int, args: argparse.Namespace) -> None:
    best_halt = df_metrics.sort_values("halt_f1", ascending=False).iloc[0]
    best_batt = df_metrics.sort_values("battery_rmse").iloc[0]
    lines = [
        "# V2 Battery + Permanent-Halt Sequence Analysis",
        "",
        "## Objective",
        "- Regression target: next-day battery telemetry mean.",
        f"- Classification target: permanent halt within {args.halt_horizon_days} days.",
        "",
        "## Dataset and protocol",
        f"- Source: original `idata/*/application.csv`.",
        f"- Sequence length: {args.seq_len} days.",
        f"- Temporal split: 70/15/15 by calendar day.",
        f"- Sample count: {n_samples}.",
        f"- Sensors used: {n_sensors}.",
        "",
        "## Iterative engineering disclosures",
        "- Halt target is an operational label from observed telemetry tails, not a physical battery-death oracle.",
        "- Right-censored non-halt windows near series end are skipped using horizon truncation.",
        "- Hyperparameters (hidden size, layers, loss weights via class pos_weight, threshold scan) are iterative settings.",
        "",
        "## Metrics table",
        df_metrics.to_string(index=False),
        "",
        "## Key findings",
        f"- Best halt F1: `{best_halt['model']}` with F1={best_halt['halt_f1']:.4f}, recall={best_halt['halt_recall']:.4f}, precision={best_halt['halt_precision']:.4f}.",
        f"- Best battery RMSE: `{best_batt['model']}` with RMSE={best_batt['battery_rmse']:.4f}, MAE={best_batt['battery_mae']:.4f}, R2={best_batt['battery_r2']:.4f}.",
        "- AUC/AP and Brier should be interpreted jointly: discrimination and probability calibration are different properties.",
    ]
    (OUT_REPORTS / "v2_battery_halt_analysis.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="V2 LSTM/Transformer for battery + permanent halt.")
    parser.add_argument("--seq-len", type=int, default=21)
    parser.add_argument("--halt-horizon-days", type=int, default=30)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max-sensors", type=int, default=350, help="Use -1 for no cap.")
    args = parser.parse_args()

    ensure_dirs()
    torch.manual_seed(42)
    np.random.seed(42)
    device = torch.device("cpu")

    max_sensors = None if args.max_sensors is not None and args.max_sensors < 0 else args.max_sensors
    print("Loading original application.csv daily panel...")
    df = load_original_daily(max_sensors=max_sensors)
    seq = build_sequences(df, seq_len=args.seq_len, halt_horizon_days=args.halt_horizon_days)
    tr_idx, va_idx, te_idx = temporal_split(seq.day_index)

    x_tr, x_va, x_te = seq.x[tr_idx], seq.x[va_idx], seq.x[te_idx]
    yb_tr, yb_va, yb_te = seq.y_batt[tr_idx], seq.y_batt[va_idx], seq.y_batt[te_idx]
    yh_tr, yh_va, yh_te = seq.y_halt[tr_idx], seq.y_halt[va_idx], seq.y_halt[te_idx]
    x_tr, x_va, x_te = standardize(x_tr, x_va, x_te)

    train_loader = DataLoader(TensorDataset(torch.tensor(x_tr), torch.tensor(yb_tr), torch.tensor(yh_tr)), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.tensor(x_va), torch.tensor(yb_va), torch.tensor(yh_va)), batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(TensorDataset(torch.tensor(x_te), torch.tensor(yb_te), torch.tensor(yh_te)), batch_size=args.batch_size, shuffle=False)

    pos_rate = float(np.mean(yh_tr))
    pos_weight = float((1.0 - pos_rate) / max(pos_rate, 1e-6))
    models = {
        "lstm": LSTMHead(in_dim=x_tr.shape[-1]),
        "transformer": TransformerHead(in_dim=x_tr.shape[-1]),
    }

    rows = []
    for name, model in models.items():
        print(f"Training {name}...")
        model = run_train(model, train_loader, val_loader, args.epochs, args.lr, device, pos_weight)
        vb_pred, vh_prob, vb_true, vh_true = predict(model, val_loader, device)
        cls_t = best_threshold(vh_true, vh_prob)
        tb_pred, th_prob, tb_true, th_true = predict(model, test_loader, device)
        m = eval_metrics(tb_true, tb_pred, th_true, th_prob, cls_t)
        m["model"] = name
        rows.append(m)

        pd.DataFrame([m]).to_csv(OUT_TABLES / f"v2_{name}_battery_halt_metrics.csv", index=False)
        pd.DataFrame(
            {
                "sensor": seq.sensor_id[te_idx],
                "day": seq.day_label[te_idx],
                "y_true_battery": tb_true,
                "y_pred_battery": tb_pred,
                "y_true_halt_within_h": th_true.astype(int),
                "y_pred_halt_prob": th_prob,
                "y_pred_halt_cls": (th_prob >= cls_t).astype(int),
            }
        ).to_csv(OUT_TABLES / f"v2_{name}_battery_halt_predictions.csv", index=False)

        save_curves(th_true.astype(int), th_prob, name)
        save_battery_scatter(tb_true, tb_pred, name)
        print(
            f"{name}: batt_RMSE={m['battery_rmse']:.4f} batt_R2={m['battery_r2']:.4f} "
            f"halt_F1={m['halt_f1']:.4f} halt_AUC={m['halt_auc']:.4f}"
        )

    out_df = pd.DataFrame(rows)[
        [
            "model",
            "battery_mae",
            "battery_rmse",
            "battery_r2",
            "halt_auc",
            "halt_ap",
            "halt_brier",
            "halt_precision",
            "halt_recall",
            "halt_f1",
            "halt_specificity",
            "halt_prevalence",
            "halt_threshold",
            "halt_tp",
            "halt_fp",
            "halt_tn",
            "halt_fn",
        ]
    ]
    out_df.to_csv(OUT_TABLES / "v2_model_comparison_battery_halt.csv", index=False)
    write_report(
        out_df,
        n_samples=int(len(seq.x)),
        n_sensors=int(pd.Series(seq.sensor_id).nunique()),
        args=args,
    )
    print(f"Wrote: {OUT_TABLES / 'v2_model_comparison_battery_halt.csv'}")
    print(f"Wrote: {OUT_REPORTS / 'v2_battery_halt_analysis.md'}")


if __name__ == "__main__":
    main()
