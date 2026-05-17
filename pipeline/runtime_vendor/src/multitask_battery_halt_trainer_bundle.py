#!/usr/bin/env python3
"""
Multitask battery–halt deep learning trainer (optimized ensemble stack).

Developed with Cursor AI (node_analysis vendored bundle).

**FORK:** ``multitask_battery_halt_trainer_bundle.py`` — same behavior as
``multitask_battery_halt_trainer.py`` plus optional **checkpoints** (academic bundle).
The original file is unchanged; the
holistic MTL/STL bundle invokes this copy by default.

**Module:** ``multitask_battery_halt_trainer_bundle.py`` — joint **Method A** (next-day battery regression) and **Method B**
(permanent halt within horizon) on LoRaWAN sensor telemetry. Builds on ``telemetry_sequence_pipeline`` for ingestion,
labels, and windowing; adds multi-architecture heads (LSTM, GRU, Transformer, TCN, optional patch mixer), imbalance-aware
halt losses, optional survival formulation, temperature scaling, calibration exports, and thesis-oriented CSV/figure outputs.

**Data sources:** daily panels from ``telemetry_sequence_pipeline.load_original_daily`` reading ``idata/*/application.csv``;
optional device allowlist CSV via ``--deveui-cohort-csv``. **Outputs:** under ``thesis_pipeline/outputs/{tables,figures,reports}``
(relative to repository root inferred from this file’s location).

**Versioning:** Default artifact stems remain ``v7_opt_*`` for continuity with prior result files.

---
Technical reference (original V7 lineage):

**Battery target:** `--battery-target level` (default) = next-day mean voltage; `residual` = train on
next-day minus last-step raw `battery_mean`, add back at inference (wave **B01**).

**Battery loss:** `--battery-loss mse` (default) or `quantile` (B02) or `ordinal` (B03) = train-quantile bins +
cross-entropy; point prediction = **sum_k p_k · center_k**; CSV may include `y_pred_battery_bin` (argmax class).

**Halt mode:** `--halt-mode binary` (default) or `survival` (B04) = discrete-time cumulative hazards over
`0..halt_horizon_days` (K = H+1 logits); multi-BCE vs targets z_k = 1[d ≤ k]; scalar halt score = sigmoid(last logit)
aligned with legacy “halt within H”.

**Battery regime (B05):** `--battery-regime-heads gated_learned` or `gated_hand_voltage` = two battery linear heads
(plateau vs endgame) mixed by a learned sigmoid gate on `z` or a fixed voltage schedule on denormalized last-step
`battery_mean` (MSE battery only; not with quantile/ordinal).

**Training sampler (B06):** `--halt-stratified-batches` = each batch mixes halt+ / halt− rows (~`--halt-stratified-pos-fraction`);
mutually exclusive with `--weighted-sampler`.

**Count auxiliary (B07):** `--count-aux-weight W` = Poisson NLL on next-day `daily_count` (MSE battery + binary/survival halt only;
excludes quantile, ordinal, regime dual-head — see argparse checks).

**Optimizer:** AdamW with decoupled weight decay. **Calibration:** scalar temperature ``T`` on halt logits;
``q_i = max_k σ_SM(z_i/T)(k)`` (binary export uses ``σ(z_i/T)`` as ``P(y=1)``).

**Backbones:** ``backbone_architectures.py`` (forget-gate LSTM, GRU, dilated TCN+BN, multi-head Transformer).

**Patch transformer (B09):** `--patch-len P` divides `--seq-len`; replaces the default Transformer with a patch-merged encoder.

Outputs: all V5 artifacts, plus `epochs_trained`, `weight_decay`, `halt_label_smoothing` in comparison CSV.

**Progress:** each epoch prints `val_loss`, running `best`, and `no_improve` (early stop message when triggered). Use `--no-progress-epochs` for quiet logs.

**Documentation:** Inline comments and section banners describe control flow only. **Hyperparameters, training
math, and argparse defaults are unchanged** in this vendored bundle (Cursor AI / node_analysis).
"""
from __future__ import annotations  # Postponed type hints (PEP 563) for large trainer signatures.

import argparse  # CLI: epochs, cohort CSV, artifact stem, halt mode, architecture flags.
import json  # Serialize run metadata in optional run records.
import os  # NODE_ANALYSIS_OUTPUT_ROOT / HOL_ACAD_OUTPUT_ROOT and trainer env.
import sys  # sys.path for vendored src; sys.argv when invoked in-process.
from pathlib import Path  # Resolve bundle root, checkpoint dirs, CSV output paths.
from typing import Dict, List, Tuple  # Type hints for metrics dicts and index tuples.

import matplotlib.pyplot as plt  # Optional diagnostic figures under models/figures/.
import numpy as np  # Arrays for splits, metrics, temperature scaling.
import pandas as pd  # Export comparison CSVs, predictions, calibration tables.
import torch  # Training loop, checkpoints, GPU/CPU tensors.
import torch.nn as nn  # Multitask heads and backbone modules.
import torch.nn.functional as F  # Activations, BCE/MSE losses, softmax for ordinal battery.
from sklearn.metrics import (  # Hold-out metrics for battery regression and halt classification.
    average_precision_score,  # PR-AUC for halt ranking quality.
    brier_score_loss,  # Probabilistic calibration score for halt.
    confusion_matrix,  # TP/FP/TN/FN at applied threshold.
    f1_score,  # F1 at applied threshold (thesis tables).
    fbeta_score,  # F0.5/F1/F2 along threshold sweeps.
    mean_absolute_error,  # Battery MAE in volts.
    mean_squared_error,  # Battery MSE for RMSE derivation.
    precision_recall_curve,  # PR curve for policy_map-style exports.
    precision_score,  # Precision at Fβ-optimal threshold.
    r2_score,  # Battery R² on hold-out sensors.
    recall_score,  # Recall at applied threshold.
    roc_auc_score,  # Discrimination for halt logits.
    roc_curve,  # ROC points when diagnostic plots enabled.
)
from sklearn.preprocessing import KBinsDiscretizer  # Ordinal battery bins when --battery-loss ordinal.
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler  # Batched sequence training.

# Ensure vendored ``src`` is importable when trainer is launched as a script.
_SRC = Path(__file__).resolve().parent  # .../runtime_vendor/src
if str(_SRC) not in sys.path:  # Guard against duplicate insertion on re-import.
    sys.path.insert(0, str(_SRC))  # Prefer local telemetry_sequence_pipeline over site-packages.

import telemetry_sequence_pipeline as telemetry  # Daily panels, windowing, SeqData labels.
from backbone_architectures import GRUHead, LSTMHead, TCNHead, TransformerHeadV5  # Four backbone constructors.

# ---------------------------------------------------------------------------
# Path resolution and output directories (tables / figures / reports)
# ---------------------------------------------------------------------------

_HERE = Path(__file__).resolve()
BASE_DIR = _HERE.parents[2]
_THESIS_PIPELINE = next((p for p in BASE_DIR.parents if p.name == "thesis_pipeline"), None)
_default_output_root = BASE_DIR / "outputs"
if (BASE_DIR / "outputs").is_dir() is False and _THESIS_PIPELINE is not None:
    _default_output_root = _THESIS_PIPELINE / "outputs"

OUTPUT_ROOT = Path(  # Root for all trainer exports (set by node_analysis layout.apply_run_env).
    os.environ.get(  # Prefer canonical NODE_ANALYSIS_* then legacy HOL_ACAD_*.
        "NODE_ANALYSIS_OUTPUT_ROOT",
        os.environ.get("HOL_ACAD_OUTPUT_ROOT", str(_default_output_root)),
    )
)
OUT_TABLES = OUTPUT_ROOT / "tables"  # Comparison CSVs and per-model prediction files (figures read these).
OUT_FIGS = OUTPUT_ROOT / "figures"  # Trainer-internal diagnostic plots (not thesis graph allowlist).
OUT_REPORTS = OUTPUT_ROOT / "reports"  # Markdown analysis reports per artifact stem.


def _skip_trainer_diagnostic_figures() -> bool:
    """When true, do not write ROC/PR/scatter/calibration PNGs under models/figures/."""
    for key in (
        "NODE_ANALYSIS_SKIP_TRAINER_DIAGNOSTIC_FIGURES",
        "HOL_ACAD_SKIP_TRAINER_DIAGNOSTIC_FIGURES",
    ):
        if os.environ.get(key, "").strip().lower() in ("1", "true", "yes", "on"):
            return True
    return False


# ---------------------------------------------------------------------------
# Train / val / test index construction (global, per-sensor, holdout, k-fold)
# ---------------------------------------------------------------------------


def ensure_dirs() -> None:
    OUT_TABLES.mkdir(parents=True, exist_ok=True)
    if not _skip_trainer_diagnostic_figures():
        OUT_FIGS.mkdir(parents=True, exist_ok=True)
    OUT_REPORTS.mkdir(parents=True, exist_ok=True)


def per_sensor_temporal_split(sensor_id: np.ndarray, day_index: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    For each sensor, sort samples by calendar day and take the first 70% as train,
    next 15% as val, last 15% as test.

    Note: halt positives often cluster in the **tail** of a device's life, so this split
    can still leave train/val with very few positives — see `sensor_holdout` for
    device-level generalization instead.
    """
    tr: List[int] = []
    va: List[int] = []
    te: List[int] = []
    for sid in np.unique(sensor_id):
        idx = np.where(sensor_id == sid)[0]
        idx = idx[np.argsort(day_index[idx], kind="mergesort")]
        n = len(idx)
        if n <= 5:
            tr.extend(idx.tolist())
            continue
        i1 = max(1, min(n - 2, int(0.70 * n)))
        i2 = max(i1 + 1, min(n - 1, int(0.85 * n)))
        tr.extend(idx[:i1].tolist())
        va.extend(idx[i1:i2].tolist())
        te.extend(idx[i2:].tolist())
    return np.asarray(tr, dtype=np.int64), np.asarray(va, dtype=np.int64), np.asarray(te, dtype=np.int64)


def sensor_holdout_split(sensor_id: np.ndarray, seed: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Assign each device entirely to train, val, or test (70/15/15 of devices).
    Each split then contains full timelines, so halt prevalence is not forced into
    one temporal tail only — evaluates **generalization to unseen devices**.
    """
    rng = np.random.RandomState(seed)
    sensors = np.unique(sensor_id)
    rng.shuffle(sensors)
    n = len(sensors)
    i1 = max(1, min(n - 2, int(0.70 * n)))
    i2 = max(i1 + 1, min(n - 1, int(0.85 * n)))
    tr_s = set(sensors[:i1].tolist())
    va_s = set(sensors[i1:i2].tolist())
    te_s = set(sensors[i2:].tolist())
    tr = np.where(np.isin(sensor_id, list(tr_s)))[0]
    va = np.where(np.isin(sensor_id, list(va_s)))[0]
    te = np.where(np.isin(sensor_id, list(te_s)))[0]
    return tr.astype(np.int64), va.astype(np.int64), te.astype(np.int64)


def sensor_k_fold_indices(
    sensor_id: np.ndarray,
    k_folds: int,
    fold_id: int,
    seed: int = 42,
    inner_val_sensor_frac: float = 0.15,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Hold out fold `fold_id` (0 .. k_folds-1) as **test** (all windows from those sensors).
    Remaining sensors are split **by device** into train vs val (same scheme as V4).
    Used by **Primary ensemble model** (V7.2) wrapper for sensor k-fold CV.
    """
    if k_folds < 2:
        raise ValueError("k_folds must be >= 2")
    if not (0 <= fold_id < k_folds):
        raise ValueError("fold_id out of range")
    rng = np.random.RandomState(seed)
    sensors = np.unique(sensor_id)
    rng.shuffle(sensors)
    n = len(sensors)
    base = n // k_folds
    rem = n % k_folds
    folds: List[np.ndarray] = []
    start = 0
    for i in range(k_folds):
        sz = base + (1 if i < rem else 0)
        folds.append(sensors[start : start + sz])
        start += sz
    te_s = set(folds[fold_id].tolist())
    rest = np.concatenate([folds[i] for i in range(k_folds) if i != fold_id])
    if len(rest) < 2:
        raise ValueError("Not enough sensors for K-fold CV with this k_folds")
    n_va = max(1, int(inner_val_sensor_frac * len(rest)))
    n_tr = len(rest) - n_va
    tr_s = set(rest[:n_tr].tolist())
    va_s = set(rest[n_tr:].tolist())
    tr = np.where(np.isin(sensor_id, list(tr_s)))[0]
    va = np.where(np.isin(sensor_id, list(va_s)))[0]
    te = np.where(np.isin(sensor_id, list(te_s)))[0]
    return tr.astype(np.int64), va.astype(np.int64), te.astype(np.int64)


def focal_bce_with_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    alpha_pos: float = 0.75,
    gamma: float = 2.0,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    """Binary focal loss; optional label smoothing on halt targets."""
    t = targets.float()
    if label_smoothing > 0.0:
        t = t * (1.0 - 2.0 * label_smoothing) + label_smoothing
    ce = F.binary_cross_entropy_with_logits(logits, t, reduction="none")
    prob = torch.sigmoid(logits)
    p_t = prob * t + (1.0 - prob) * (1.0 - t)
    alpha_t = alpha_pos * t + (1.0 - alpha_pos) * (1.0 - t)
    loss = alpha_t * (1.0 - p_t).pow(gamma) * ce
    return loss.mean()


# ---------------------------------------------------------------------------
# Loss helpers, training loop (AdamW, early stopping), and forward passes
# ---------------------------------------------------------------------------


def pinball_loss(pred: torch.Tensor, target: torch.Tensor, taus: torch.Tensor) -> torch.Tensor:
    """pred (B, K), target (B,), taus (K,) — mean pinball over quantiles."""
    t = target.float().unsqueeze(-1)
    d = t - pred
    ta = taus.to(pred.device).float().view(1, -1)
    return torch.maximum(ta * d, (ta - 1.0) * d).mean()


def run_train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    lr: float,
    device: torch.device,
    halt_loss_weight: float,
    max_grad_norm: float,
    focal_gamma: float,
    focal_alpha_pos: float,
    halt_bce_mix: float,
    pos_weight: float,
    weight_decay: float,
    early_stopping_patience: int,
    min_delta: float,
    halt_label_smoothing: float,
    battery_loss: str = "mse",
    quantile_taus: torch.Tensor | None = None,
    halt_mode: str = "binary",
    ordinal_training: bool = False,
    count_aux_weight: float = 0.0,
    progress_epochs: bool = True,
    log_prefix: str = "",
) -> Tuple[nn.Module, int]:
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    mse = nn.MSELoss()
    pw = torch.tensor([pos_weight], dtype=torch.float32, device=device)
    caw = float(count_aux_weight)

    def batch_loss(
        pr_reg: torch.Tensor,
        pr_cls: torch.Tensor,
        yb_reg: torch.Tensor,
        yb_cls: torch.Tensor,
        y_cnt: torch.Tensor | None = None,
        log_cnt: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if halt_mode == "survival":
            lf = F.binary_cross_entropy_with_logits(pr_cls, yb_cls.float(), reduction="mean")
        else:
            lf = focal_bce_with_logits(
                pr_cls,
                yb_cls,
                alpha_pos=focal_alpha_pos,
                gamma=focal_gamma,
                label_smoothing=halt_label_smoothing,
            )
            if halt_bce_mix > 0.0:
                y_smooth = yb_cls.float()
                if halt_label_smoothing > 0.0:
                    y_smooth = y_smooth * (1.0 - 2.0 * halt_label_smoothing) + halt_label_smoothing
                bce = F.binary_cross_entropy_with_logits(pr_cls, y_smooth, pos_weight=pw)
                lf = (1.0 - halt_bce_mix) * lf + halt_bce_mix * bce
        if battery_loss == "quantile" and quantile_taus is not None:
            batt = pinball_loss(pr_reg, yb_reg, quantile_taus)
        elif battery_loss == "ordinal":
            batt = F.cross_entropy(pr_reg, yb_reg.long())
        else:
            batt = mse(pr_reg, yb_reg)
        core = batt + halt_loss_weight * lf
        if caw > 0.0 and y_cnt is not None and log_cnt is not None:
            core = core + caw * F.poisson_nll_loss(
                log_cnt.float(),
                y_cnt.float(),
                log_input=True,
                full=False,
            )
        return core

    def unpack_halt_targets(batch: tuple) -> torch.Tensor:
        if ordinal_training:
            return batch[2].to(device)
        if halt_mode == "survival" and len(batch) >= 5:
            return batch[4].to(device).float()
        return batch[2].to(device)

    best_state = None
    best_val = float("inf")
    no_improve = 0
    last_epoch = -1
    for ep in range(epochs):
        model.train()
        for batch in train_loader:
            xb = batch[0].to(device)
            yb_reg = batch[1].to(device)
            yb_cls = unpack_halt_targets(batch)
            out = model(xb)
            if caw > 0.0:
                pr_reg, pr_cls, log_cnt = out
                y_cnt = batch[4].to(device)
                loss = batch_loss(pr_reg, pr_cls, yb_reg, yb_cls, y_cnt, log_cnt)
            else:
                pr_reg, pr_cls = out  # type: ignore[misc]
                loss = batch_loss(pr_reg, pr_cls, yb_reg, yb_cls)
            opt.zero_grad()
            loss.backward()
            if max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            opt.step()
        model.eval()
        losses: List[float] = []
        with torch.no_grad():
            for batch in val_loader:
                xb = batch[0].to(device)
                yb_reg = batch[1].to(device)
                yb_cls = unpack_halt_targets(batch)
                out = model(xb)
                if caw > 0.0:
                    pr_reg, pr_cls, log_cnt = out
                    y_cnt = batch[4].to(device)
                    losses.append(float(batch_loss(pr_reg, pr_cls, yb_reg, yb_cls, y_cnt, log_cnt).cpu()))
                else:
                    pr_reg, pr_cls = out  # type: ignore[misc]
                    losses.append(float(batch_loss(pr_reg, pr_cls, yb_reg, yb_cls).cpu()))
        avg = float(np.mean(losses)) if losses else float("inf")
        last_epoch = ep
        if avg < best_val - min_delta:
            best_val = avg
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if early_stopping_patience > 0 and no_improve >= early_stopping_patience:
                if progress_epochs:
                    print(
                        f"{log_prefix}epoch {ep + 1}/{epochs} val_loss={avg:.6f} best={best_val:.6f} "
                        f"no_improve={no_improve} -> early_stop",
                        flush=True,
                    )
                break
        if progress_epochs:
            print(
                f"{log_prefix}epoch {ep + 1}/{epochs} val_loss={avg:.6f} best={best_val:.6f} no_improve={no_improve}",
                flush=True,
            )
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, last_epoch + 1


def predict(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    battery_residual: bool = False,
    battery_quantile: bool = False,
    battery_ordinal: bool = False,
    bin_centers: np.ndarray | None = None,
    halt_survival: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
    """Returns (pb_point, halt_prob, yb_level, yh, pb_quantiles_or_none, battery_bin_argmax_or_none)."""
    model.eval()
    pr_batt, pr_halt, yb, yh = [], [], [], []
    pr_qstack: List[np.ndarray] = []
    pr_bin_chunks: List[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            xb = batch[0]
            yb_task = batch[1]
            yh_task = batch[2]
            lb = batch[3] if len(batch) > 3 else None
            xb = xb.to(device)
            out = model(xb)
            if len(out) == 3:
                pb, ph = out[0], out[1]
            else:
                pb, ph = out  # type: ignore[misc]
            lb_f = lb.float() if lb is not None else None
            if battery_ordinal:
                assert bin_centers is not None and pb.dim() == 2
                probs = F.softmax(pb, dim=-1)
                bc = torch.as_tensor(bin_centers, dtype=probs.dtype, device=probs.device)
                exp_v = (probs * bc.unsqueeze(0)).sum(dim=-1)
                pr_batt.append(exp_v.detach().cpu().numpy())
                pr_bin_chunks.append(probs.argmax(dim=-1).detach().cpu().numpy().astype(np.int64))
                yb.append(batch[4].numpy().astype(np.float64))
            elif battery_quantile:
                assert pb.dim() == 2
                if battery_residual and lb_f is not None:
                    pb = pb + lb_f.to(device).unsqueeze(-1)
                k_med = pb.shape[1] // 2
                pr_batt.append(pb[:, k_med].detach().cpu().numpy())
                pr_qstack.append(pb.detach().cpu().numpy())
                yb_np = yb_task.numpy().astype(np.float64)
                if battery_residual and lb is not None:
                    yb_np = yb_np + lb.numpy().astype(np.float64)
                yb.append(yb_np)
            else:
                pb1 = pb.detach().cpu()
                if battery_residual and lb_f is not None:
                    pb1 = pb1 + lb_f
                pr_batt.append(pb1.numpy() if pb1.dim() == 1 else pb1.squeeze(-1).numpy())
                yb_np = yb_task.numpy().astype(np.float64)
                if battery_residual and lb is not None:
                    yb_np = yb_np + lb.numpy().astype(np.float64)
                yb.append(yb_np)
            if halt_survival and ph.dim() == 2:
                pr_halt.append(torch.sigmoid(ph[:, -1]).detach().cpu().numpy())
            else:
                pr_halt.append(torch.sigmoid(ph).detach().cpu().numpy())
            yh.append(yh_task.numpy())
    pb_q = np.concatenate(pr_qstack, axis=0) if pr_qstack else None
    pb_bin = np.concatenate(pr_bin_chunks, axis=0) if pr_bin_chunks else None
    return (
        np.concatenate(pr_batt),
        np.concatenate(pr_halt),
        np.concatenate(yb),
        np.concatenate(yh),
        pb_q,
        pb_bin,
    )


def predict_survival_export(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    battery_residual: bool = False,
    battery_quantile: bool = False,
    battery_ordinal: bool = False,
    bin_centers: np.ndarray | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Test/val loader with TensorDataset(x, yb, yh, lb, y_surv). Returns cumulative targets Z (N,K) and P_cum (N,K)."""
    model.eval()
    pr_batt, pr_last, yb_l, yh_l, zs, pcs = [], [], [], [], [], []
    with torch.no_grad():
        for batch in loader:
            if len(batch) < 5:
                raise ValueError("predict_survival_export requires survival TensorDataset (5-tuple batch).")
            xb = batch[0].to(device)
            yb_task = batch[1]
            yh_task = batch[2]
            lb = batch[3]
            z_surv = batch[4].numpy().astype(np.float32)
            lb_f = lb.float().to(device) if lb is not None else None
            out = model(xb)
            pb, ph = (out[0], out[1]) if len(out) == 2 else (out[0], out[1])  # type: ignore[misc]
            if ph.dim() != 2:
                raise ValueError("predict_survival_export expects halt logits (B, K).")
            p_cum = torch.sigmoid(ph).detach().cpu().numpy().astype(np.float32)
            pcs.append(p_cum)
            zs.append(z_surv)
            pr_last.append(p_cum[:, -1])
            if battery_ordinal or battery_quantile:
                raise NotImplementedError("predict_survival_export supports level/residual battery only.")
            pb1 = pb.detach().cpu()
            if battery_residual and lb_f is not None:
                pb1 = pb1 + lb_f.unsqueeze(-1) if pb1.dim() > 1 else pb1 + lb_f
            pr_batt.append(pb1.numpy() if pb1.dim() == 1 else pb1.squeeze(-1).numpy())
            yb_np = yb_task.numpy().astype(np.float64)
            if battery_residual and lb is not None:
                yb_np = yb_np + lb.numpy().astype(np.float64)
            yb_l.append(yb_np)
            yh_l.append(yh_task.numpy())
    Z = np.concatenate(zs, axis=0)
    P = np.concatenate(pcs, axis=0)
    return (
        np.concatenate(pr_batt),
        np.concatenate(pr_last),
        np.concatenate(yb_l),
        np.concatenate(yh_l),
        Z,
        P,
    )


# ---------------------------------------------------------------------------
# Halt calibration (temperature scaling), threshold sweeps, and metric exports
# ---------------------------------------------------------------------------


def collect_halt_logits(model: nn.Module, loader: DataLoader, device: torch.device, survival_last_logit: bool = False) -> np.ndarray:
    model.eval()
    chunks: List[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            xb = batch[0].to(device)
            out = model(xb)
            lg = out[1]
            arr = lg.detach().cpu().numpy()
            if survival_last_logit and arr.ndim == 2:
                chunks.append(arr[:, -1].reshape(-1))
            else:
                chunks.append(arr.reshape(-1))
    return np.concatenate(chunks)


def halt_positive_probability_scaled(logits: np.ndarray, temperature: float) -> np.ndarray:
    """P(y=1) = σ(z/T) for binary halt logit z (class-1 mass of softmax over {0, z})."""
    z = logits.astype(np.float64).reshape(-1) / max(float(temperature), 1e-6)
    return 1.0 / (1.0 + np.exp(-np.clip(z, -60.0, 60.0)))


def softmax_max_binary_probability(logits: np.ndarray, temperature: float) -> np.ndarray:
    """q_i = max_k σ_SM(z_i/T)(k) for two-class logits [0, z_i]."""
    p1 = halt_positive_probability_scaled(logits, temperature)
    return np.maximum(p1, 1.0 - p1)


def fit_temperature_min_brier(logits: np.ndarray, y: np.ndarray) -> float:
    """Scalar T>0 minimizing Brier on validation σ(z/T); ranking (AUC) unchanged under monotone scaling."""
    y = y.astype(np.float64).reshape(-1)
    logits = logits.astype(np.float64).reshape(-1)
    best_t, best_b = 1.0, float("inf")
    for t in np.logspace(-1.0, 1.0, 41):
        p = halt_positive_probability_scaled(logits, float(t))
        b = float(brier_score_loss(y, p))
        if b < best_b:
            best_b, best_t = b, float(t)
    return best_t


def calibration_bins_table(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    y = y_true.astype(np.float64).reshape(-1)
    p = np.clip(y_prob.astype(np.float64).reshape(-1), 1e-9, 1.0 - 1e-9)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    rows = []
    for b in range(n_bins):
        lo, hi = float(edges[b]), float(edges[b + 1])
        if b == n_bins - 1:
            mask = (p >= lo) & (p <= hi)
        else:
            mask = (p >= lo) & (p < hi)
        n = int(np.sum(mask))
        if n == 0:
            rows.append({"bin_lo": lo, "bin_hi": hi, "n": 0, "mean_pred": np.nan, "mean_true": np.nan})
        else:
            rows.append(
                {
                    "bin_lo": lo,
                    "bin_hi": hi,
                    "n": n,
                    "mean_pred": float(np.mean(p[mask])),
                    "mean_true": float(np.mean(y[mask])),
                }
            )
    return pd.DataFrame(rows)


def save_calibration_figure(y_true: np.ndarray, y_prob: np.ndarray, model_name: str, artifact_stem: str) -> None:
    if _skip_trainer_diagnostic_figures():
        return
    df = calibration_bins_table(y_true, y_prob, n_bins=10)
    ok = df["n"].values > 0
    if not np.any(ok):
        return
    dfo = df.loc[ok].copy()
    centers = (dfo["bin_lo"].values + dfo["bin_hi"].values) / 2.0
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    w = 0.04
    ax.bar(centers - w, dfo["mean_pred"], width=w * 2, label="Mean pred", color="#1565c0", alpha=0.75)
    ax.bar(centers + w, dfo["mean_true"], width=w * 2, label="Mean true", color="#c62828", alpha=0.75)
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Perfect")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Bin center (predicted prob)")
    ax.set_ylabel("Frequency / mean label")
    ax.set_title(f"{model_name.upper()} halt calibration (V7)")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_FIGS / f"{artifact_stem}_{model_name}_halt_calibration_plot.png", dpi=160)
    plt.close(fig)


def threshold_sweep_table(y_true: np.ndarray, y_prob: np.ndarray) -> pd.DataFrame:
    yt = y_true.astype(int)
    rows = []
    # Include low cutoffs: focal training can push scores modest; val may be sparse on positives.
    for t in np.linspace(0.01, 0.99, 99):
        y_hat = (y_prob >= t).astype(int)
        prec = precision_score(yt, y_hat, zero_division=0)
        rec = recall_score(yt, y_hat, zero_division=0)
        f1 = f1_score(yt, y_hat, zero_division=0)
        f2 = fbeta_score(yt, y_hat, beta=2.0, zero_division=0)
        f05 = fbeta_score(yt, y_hat, beta=0.5, zero_division=0)
        cm = confusion_matrix(yt, y_hat, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        spec = tn / max(tn + fp, 1)
        bal_acc = 0.5 * (rec + spec)
        rows.append(
            {
                "threshold": float(t),
                "precision": prec,
                "recall": rec,
                "f1": f1,
                "f2": f2,
                "f05": f05,
                "balanced_accuracy": bal_acc,
                "tp": int(tp),
                "fp": int(fp),
                "tn": int(tn),
                "fn": int(fn),
            }
        )
    return pd.DataFrame(rows)


def pick_threshold(df: pd.DataFrame, criterion: str) -> float:
    if criterion == "f1":
        i = int(df["f1"].values.argmax())
    elif criterion == "f2":
        i = int(df["f2"].values.argmax())
    elif criterion == "balanced_accuracy":
        i = int(df["balanced_accuracy"].values.argmax())
    else:
        raise ValueError(criterion)
    return float(df["threshold"].iloc[i])


def prevalence_matched_threshold(y_prob: np.ndarray, target_positive_rate: float) -> float:
    """Threshold such that fraction of scores >= t is approximately target_positive_rate."""
    target_positive_rate = float(np.clip(target_positive_rate, 1e-4, 0.99))
    qs = np.sort(y_prob.astype(np.float64))
    idx = int(np.floor((1.0 - target_positive_rate) * len(qs)))
    idx = int(np.clip(idx, 0, len(qs) - 1))
    return float(qs[idx])


def select_threshold_with_fallback(
    y_val: np.ndarray,
    prob_val: np.ndarray,
    y_train: np.ndarray,
    prob_train: np.ndarray,
    criterion: str,
    min_positives: int,
    global_prevalence: float,
    quantile_target_rate: float = -1.0,
) -> Tuple[float, str, pd.DataFrame]:
    """
    Prefer validation sweep; else train+val pool (reduces calendar-only shift where val has no positives).
    If the pool still lacks both classes sufficiently, use prevalence-matched score quantile on train+val probs
    (operating point only; document in report).
    """
    yv = y_val.astype(int)
    if int(np.sum(yv)) >= min_positives and int(np.sum(1 - yv)) >= min_positives:
        sweep = threshold_sweep_table(y_val, prob_val)
        return pick_threshold(sweep, criterion), "val", sweep

    yt = np.concatenate([y_train.astype(int), yv])
    pt = np.concatenate([prob_train.astype(np.float64), prob_val.astype(np.float64)])
    if int(np.sum(yt)) >= min_positives and int(np.sum(1 - yt)) >= min_positives:
        sweep = threshold_sweep_table(yt, pt)
        return pick_threshold(sweep, criterion), "train_val", sweep

    cal_prev = float(np.mean(yt)) if len(yt) else global_prevalence
    if cal_prev < 1e-8:
        cal_prev = global_prevalence
    if quantile_target_rate > 0.0:
        cal_prev = float(quantile_target_rate)
    t_q = prevalence_matched_threshold(pt, cal_prev)
    sweep = threshold_sweep_table(yt, pt)
    return t_q, f"train_val_quantile_p_cal={cal_prev:.4f}", sweep


def eval_metrics(
    yb_true: np.ndarray,
    yb_pred: np.ndarray,
    yh_true: np.ndarray,
    yh_prob: np.ndarray,
    cls_t: float,
) -> Dict[str, float]:
    yh_true_i = yh_true.astype(int)
    yh_hat = (yh_prob >= cls_t).astype(int)
    cm = confusion_matrix(yh_true_i, yh_hat, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return {
        "battery_mae": float(mean_absolute_error(yb_true, yb_pred)),
        "battery_rmse": float(np.sqrt(mean_squared_error(yb_true, yb_pred))),
        "battery_r2": float(r2_score(yb_true, yb_pred)),
        "halt_auc": float(roc_auc_score(yh_true_i, yh_prob)) if len(np.unique(yh_true_i)) > 1 else float("nan"),
        "halt_ap": float(average_precision_score(yh_true_i, yh_prob)) if len(np.unique(yh_true_i)) > 1 else float("nan"),
        "halt_brier": float(brier_score_loss(yh_true_i, yh_prob)),
        "halt_precision": float(precision_score(yh_true_i, yh_hat, zero_division=0)),
        "halt_recall": float(recall_score(yh_true_i, yh_hat, zero_division=0)),
        "halt_f1": float(f1_score(yh_true_i, yh_hat, zero_division=0)),
        "halt_f2": float(fbeta_score(yh_true_i, yh_hat, beta=2.0, zero_division=0)),
        "halt_specificity": float(tn / max(tn + fp, 1)),
        "halt_prevalence": float(np.mean(yh_true_i)),
        "halt_threshold": cls_t,
        "halt_tp": int(tp),
        "halt_fp": int(fp),
        "halt_tn": int(tn),
        "halt_fn": int(fn),
    }


def save_curves(y_true: np.ndarray, y_prob: np.ndarray, model_name: str, artifact_stem: str) -> None:
    if _skip_trainer_diagnostic_figures():
        return
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    fig, ax = plt.subplots(1, 2, figsize=(10, 4.2))
    ax[0].plot(fpr, tpr, lw=2, color="#1565c0")
    ax[0].plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.6)
    ax[0].set_title(f"{model_name.upper()} halt ROC (V7)")
    ax[0].set_xlabel("False positive rate")
    ax[0].set_ylabel("True positive rate")
    ax[0].grid(True, alpha=0.3)
    ax[1].plot(rec, prec, lw=2, color="#e65100")
    ax[1].set_title(f"{model_name.upper()} halt PR (V7)")
    ax[1].set_xlabel("Recall")
    ax[1].set_ylabel("Precision")
    ax[1].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_FIGS / f"{artifact_stem}_{model_name}_halt_roc_pr.png", dpi=180)
    plt.close(fig)


def save_battery_scatter(y_true: np.ndarray, y_pred: np.ndarray, model_name: str, artifact_stem: str) -> None:
    if _skip_trainer_diagnostic_figures():
        return
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.scatter(y_true, y_pred, s=8, alpha=0.3, color="#37474f")
    lo = float(min(np.min(y_true), np.min(y_pred)))
    hi = float(max(np.max(y_true), np.max(y_pred)))
    ax.plot([lo, hi], [lo, hi], "r--", lw=1.5)
    ax.set_title(f"{model_name.upper()} next-day battery (V7)")
    ax.set_xlabel("True battery mean")
    ax.set_ylabel("Predicted battery mean")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_FIGS / f"{artifact_stem}_{model_name}_battery_scatter.png", dpi=180)
    plt.close(fig)


def write_report(
    df_metrics: pd.DataFrame,
    n_samples: int,
    n_sensors: int,
    args: argparse.Namespace,
    sweep_notes: str,
    artifact_stem: str,
    split_mode_for_report: str | None = None,
) -> None:
    best_f2 = df_metrics.sort_values("halt_f2", ascending=False).iloc[0]
    best_batt = df_metrics.sort_values("battery_rmse").iloc[0]
    sm_rep = split_mode_for_report if split_mode_for_report is not None else str(args.split_mode)
    lines = [
        "# V7 Optimized Battery + Halt (V5 + early stop, weight decay, halt label smoothing)",
        "",
        "## Objective",
        "- Regression: next-day battery mean.",
        f"- Classification: permanent halt within {args.halt_horizon_days} days (same operational definition as V2).",
        "",
        "## What changed vs V2 / V3 / V5",
        "- Same halt loss and splits as **V3** (focal + optional BCE mix, GRU, Transformer, global / per_sensor / sensor_holdout).",
        "- **TCNHead**: stacked 1D convolutions + last-step pooling (local temporal inductive bias vs recurrent/attention).",
        "- **Temperature scaling**: single `T` fit on **validation** to minimize Brier; reported as `halt_brier_tscaled` (AUC unchanged).",
        "- **Calibration bins** CSV + reliability figure per model on **test** predictions.",
        f"- **V7 training (vs V5):** AdamW **weight_decay**, **early stopping** on composite val loss, **halt label smoothing** in focal/BCE.",
        f"- Validation threshold chosen by **`{args.threshold_criterion}`** after a dense sweep; test metrics use that threshold.",
        "- If val (then train+val) lacks enough positives/negatives for a stable F-score sweep, a **quantile threshold** matches the calibration-set positive rate (see `threshold_source`).",
        "- **Temporal shift**: halt labels can concentrate in late calendar months; test prevalence may exceed train+val prevalence — compare AUC (ranking) vs F-scores (threshold-sensitive).",
        "- Per-model CSV: `*_sweep_val.csv` and `*_sweep_applied.csv` (the table used to pick the threshold).",
        "",
        "## Dataset and protocol",
        f"- **Split mode `split_mode={sm_rep}`**: `global` = 70/15/15 by **calendar day** (V2-style); "
        "`per_sensor` = 70/15/15 along each **device timeline** (positives may still concentrate in tail splits); "
        "`sensor_holdout` = 70/15/15 **devices** — test is **unseen sensors**; train/val each see full device lives. "
        "`sensor_cv_Kfold` = one fold of **device-disjoint** k-fold (same index logic as V4); remaining devices split train/val by device.",
        f"- Sequence length: {args.seq_len}.",
        f"- Samples: {n_samples}; sensors: {n_sensors}.",
        f"- halt_loss_weight={args.halt_loss_weight}, halt_bce_mix={args.halt_bce_mix}, focal_gamma={args.focal_gamma}, focal_alpha_pos={args.focal_alpha_pos}.",
        f"- **V7:** weight_decay={args.weight_decay}, early_stopping_patience={args.early_stopping_patience}, "
        f"min_delta={args.min_delta}, halt_label_smoothing={args.halt_label_smoothing}.",
        "",
        "## Interpretation",
        "- High AUC with low F1 at a fixed threshold usually indicates **ranking signal** + **wrong operating point** under imbalance.",
        "- F2 emphasizes recall; use it when missing an imminent halt is costlier than extra alerts.",
        "- Transformer battery regression can remain weak if inductive bias mismatches the panel; compare GRU/LSTM first.",
        "- Use **`--quantile-target-rate`** (e.g. `0.16`) only as a *scenario*: fixes alert budget when calibration months have almost no halt positives but a later deployment period is riskier.",
        "- **`sensor_holdout`**: very high AUC → strong separability within cohort; confirm with **V4** `--sensor-cv-folds` if needed.",
        "- Compare **`halt_brier`** vs **`halt_brier_tscaled`**: large drop implies miscalibrated probabilities fixable by simple scaling.",
        "",
        sweep_notes,
        "",
        "## Metrics (test)",
        df_metrics.to_string(index=False),
        "",
        "## Highlights",
        f"- Best halt F2 (test): `{best_f2['model']}` — F2={best_f2['halt_f2']:.4f}, F1={best_f2['halt_f1']:.4f}, "
        f"recall={best_f2['halt_recall']:.4f}, precision={best_f2['halt_precision']:.4f}, AUC={best_f2['halt_auc']:.4f}.",
        f"- Best battery RMSE: `{best_batt['model']}` — RMSE={best_batt['battery_rmse']:.4f}, R2={best_batt['battery_r2']:.4f}.",
    ]
    (OUT_REPORTS / f"{artifact_stem}_battery_halt_analysis.md").write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# CLI entry: data load, splits, model zoo, train/eval, CSV and figure artifacts
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Bundle fork of multitask battery + halt trainer (ensemble stack + optional checkpoint dir + val grad-cosine). Original script unchanged."
    )
    parser.add_argument("--seq-len", type=int, default=21)
    parser.add_argument("--halt-horizon-days", type=int, default=30)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument(
        "--max-sensors",
        type=int,
        default=350,
        help="Cap: keep top-N devices by total uplink volume. Use -1 for **all** devices in application.csv. "
        "Ignored if --deveui-cohort-csv is set.",
    )
    parser.add_argument(
        "--deveui-cohort-csv",
        type=str,
        default="",
        help="Optional CSV with column `deveui` (or first column) listing exact devices to train on; "
        "use with sensor_cohort_heldout_inventory.py --write-balanced-cohort. Overrides --max-sensors.",
    )
    parser.add_argument("--halt-loss-weight", type=float, default=4.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--focal-alpha-pos", type=float, default=0.75)
    parser.add_argument(
        "--threshold-criterion",
        choices=("f2", "f1", "balanced_accuracy"),
        default="f2",
        help="Pick threshold on validation set.",
    )
    parser.add_argument("--weighted-sampler", action="store_true", help="Oversample halt-positive windows in training.")
    parser.add_argument(
        "--quantile-target-rate",
        type=float,
        default=-1.0,
        help="If >0, force alert-rate target for quantile threshold fallback (scenario analysis; default uses calibration/global prevalence).",
    )
    parser.add_argument(
        "--split-mode",
        choices=("global", "per_sensor", "sensor_holdout"),
        default="global",
        help="global = by calendar day; per_sensor = along each device timeline; sensor_holdout = 70/15/15 devices (unseen in test).",
    )
    parser.add_argument(
        "--halt-bce-mix",
        type=float,
        default=0.0,
        help="Blend plain weighted BCE into halt loss (0=pure focal). Can stabilize score scale.",
    )
    parser.add_argument(
        "--artifact-stem",
        type=str,
        default=None,
        help="Prefix for outputs (default: v7_opt, v7_opt_per_sensor, v7_opt_sensor_holdout).",
    )
    parser.add_argument("--no-tcn", action="store_true", help="Train only lstm, gru, transformer (no temporal CNN).")
    parser.add_argument("--weight-decay", type=float, default=2e-4, help="AdamW weight decay (V5 default was 1e-4).")
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=5,
        help="Stop if val loss does not improve for this many epochs (0 = train all epochs).",
    )
    parser.add_argument("--min-delta", type=float, default=1e-5, help="Minimum val loss improvement to reset patience.")
    parser.add_argument(
        "--halt-label-smoothing",
        type=float,
        default=0.05,
        help="Smooth 0/1 halt targets in focal/BCE (0 disables).",
    )
    parser.add_argument(
        "--battery-target",
        choices=("level", "residual"),
        default="level",
        help="level = predict next-day voltage; residual = predict (y_{t+1} - V_t) with V_t = last-step raw battery_mean, then add back at inference (B01 wave).",
    )
    parser.add_argument(
        "--battery-loss",
        choices=("mse", "quantile", "ordinal"),
        default="mse",
        help="mse | quantile (B02) | ordinal (B03) = train-fit quantile bins + CE, expected bin center as point pred.",
    )
    parser.add_argument(
        "--ordinal-bins",
        type=int,
        default=10,
        help="Number of quantile bins on train voltage for --battery-loss ordinal.",
    )
    parser.add_argument(
        "--halt-mode",
        choices=("binary", "survival"),
        default="binary",
        help="binary = focal/BCE on y_halt within H; survival (B04) = K=H+1 cumulative targets + multi-BCE.",
    )
    parser.add_argument(
        "--battery-regime-heads",
        choices=("single", "gated_learned", "gated_hand_voltage"),
        default="single",
        help="B05: single battery head (default) or dual heads mixed by learned or hand-voltage gate (MSE only).",
    )
    parser.add_argument(
        "--regime-hand-v-mid",
        type=float,
        default=3.55,
        help="Hand gate: sigmoid midpoint in volts on denormalized battery_mean (B05).",
    )
    parser.add_argument(
        "--regime-hand-v-tau",
        type=float,
        default=0.12,
        help="Hand gate: temperature in volts (B05).",
    )
    parser.add_argument(
        "--halt-stratified-batches",
        action="store_true",
        help="B06: batch sampler mixing halt-positive and halt-negative windows (not with --weighted-sampler).",
    )
    parser.add_argument(
        "--halt-stratified-pos-fraction",
        type=float,
        default=0.5,
        help="Target fraction of halt-positive rows per batch when using --halt-stratified-batches (B06).",
    )
    parser.add_argument(
        "--count-aux-weight",
        type=float,
        default=0.0,
        help="B07: Poisson NLL weight on next-day daily_count (0=off); requires MSE battery; see script exclusions.",
    )
    parser.add_argument(
        "--patch-len",
        type=int,
        default=0,
        help="B09: if >0 and divides --seq-len, Transformer backbone uses non-overlapping patches (single-head path only).",
    )
    parser.add_argument(
        "--cv-k",
        type=int,
        default=0,
        help="Primary ensemble / V7.2: if >=2 and --cv-fold-id>=0, use sensor k-fold (fold test holdout) instead of --split-mode indices.",
    )
    parser.add_argument(
        "--cv-fold-id",
        type=int,
        default=-1,
        help="Which fold to train (0 .. cv-k-1). Ignored unless --cv-k>=2.",
    )
    parser.add_argument(
        "--cv-inner-val-frac",
        type=float,
        default=0.15,
        help="Fraction of non-test sensors (by count) used for validation when using --cv-k.",
    )
    parser.add_argument(
        "--no-progress-epochs",
        action="store_true",
        help="Disable per-epoch val_loss lines during training.",
    )
    parser.add_argument(
        "--save-checkpoint-dir",
        type=str,
        default="",
        help="Bundle fork: save {artifact_stem}_{arch}_bundle.pt under this directory (empty=off).",
    )
    args = parser.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(line_buffering=True)
        except Exception:
            pass
    if hasattr(sys.stderr, "reconfigure"):
        try:
            sys.stderr.reconfigure(line_buffering=True)
        except Exception:
            pass

    ensure_dirs()
    torch.manual_seed(42)
    np.random.seed(42)
    device = torch.device("cpu")

    max_sensors = None if args.max_sensors is not None and args.max_sensors < 0 else args.max_sensors
    cohort_csv = (args.deveui_cohort_csv or "").strip()
    print("Loading application.csv daily panel (V7 optimized)...")
    if cohort_csv:
        cp = Path(cohort_csv)
        if not cp.is_file():
            raise SystemExit(f"--deveui-cohort-csv not found: {cp}")
        cdf = pd.read_csv(cp)
        col = "deveui" if "deveui" in cdf.columns else cdf.columns[0]
        allow = [d for d in cdf[col].astype(str).str.strip().tolist() if d]
        if not allow:
            raise SystemExit(
                f"--deveui-cohort-csv has no devices: {cp}\n"
                "Add DevEUI rows or regenerate with discover_cohort.py / main.py --discover-cohort."
            )
        print(f"  Using device allowlist from {cp} ({len(allow)} devices).")
        df = telemetry.load_original_daily(max_sensors=None, deveui_allowlist=allow)
    else:
        df = telemetry.load_original_daily(max_sensors=max_sensors)
        if max_sensors is not None:
            print(f"  Cohort: top-{max_sensors} devices by uplink volume (not all sensors in dataset).")
        else:
            print("  Cohort: all devices in application.csv (--max-sensors -1).")
    seq = telemetry.build_sequences(df, seq_len=args.seq_len, halt_horizon_days=args.halt_horizon_days)
    if seq.y_halt_dt is None:
        raise SystemExit("SeqData missing y_halt_dt; update telemetry_sequence_pipeline.build_sequences.")
    if args.count_aux_weight > 0.0 and seq.y_next_count is None:
        raise SystemExit("SeqData missing y_next_count; update telemetry_sequence_pipeline.build_sequences for count auxiliary.")
    # Raw last-step battery_mean for residual target (telemetry_sequence_pipeline feature_cols index 5).
    batt_idx = 5
    last_step_batt = seq.x[:, -1, batt_idx].astype(np.float32).copy()
    if args.battery_target == "residual":
        seq.y_batt = (seq.y_batt.astype(np.float32) - last_step_batt).astype(np.float32)
    global_prevalence = float(np.mean(seq.y_halt))
    n_sensors_all = int(pd.Series(seq.sensor_id).nunique())

    def _n_sensors(ix: np.ndarray) -> int:
        return int(pd.Series(seq.sensor_id[ix]).nunique())

    _stem_default = {"global": "v7_opt", "per_sensor": "v7_opt_per_sensor", "sensor_holdout": "v7_opt_sensor_holdout"}
    cv_active = int(args.cv_k) >= 2 and int(args.cv_fold_id) >= 0
    if args.artifact_stem is not None:
        artifact_stem = args.artifact_stem
    elif cv_active:
        artifact_stem = f"v7_opt_sensor_cv{int(args.cv_k)}fold_fold{int(args.cv_fold_id)}"
    else:
        artifact_stem = _stem_default[args.split_mode]
    if args.artifact_stem is None:
        batt_parts: List[str] = []
        if args.battery_target == "residual":
            batt_parts.append("residual")
        if args.battery_loss == "quantile":
            batt_parts.append("quantile")
        if args.battery_loss == "ordinal":
            batt_parts.append(f"ordinal{args.ordinal_bins}")
        if batt_parts:
            artifact_stem = f"{artifact_stem}_batt_{'_'.join(batt_parts)}"
        if args.battery_regime_heads == "gated_learned":
            artifact_stem = f"{artifact_stem}_batt_regime_gated_learned"
        elif args.battery_regime_heads == "gated_hand_voltage":
            artifact_stem = f"{artifact_stem}_batt_regime_gated_hand"
        if args.halt_stratified_batches:
            artifact_stem = f"{artifact_stem}_halt_stratified_batches"
        if args.count_aux_weight > 0.0:
            artifact_stem = f"{artifact_stem}_count_aux"
        if args.patch_len > 0:
            artifact_stem = f"{artifact_stem}_patch{args.patch_len}"
    # Always suffix survival runs so explicit --artifact-stem cannot clobber binary stems.
    if args.halt_mode == "survival" and not str(artifact_stem).endswith("_halt_survival"):
        artifact_stem = f"{artifact_stem}_halt_survival"

    split_mode_label = str(args.split_mode)
    if int(args.cv_k) >= 2 and int(args.cv_fold_id) >= 0:
        tr_idx, va_idx, te_idx = sensor_k_fold_indices(
            seq.sensor_id,
            int(args.cv_k),
            int(args.cv_fold_id),
            seed=42,
            inner_val_sensor_frac=float(args.cv_inner_val_frac),
        )
        split_mode_label = f"sensor_cv_{int(args.cv_k)}fold"
    elif args.split_mode == "global":
        tr_idx, va_idx, te_idx = telemetry.temporal_split(seq.day_index)
    elif args.split_mode == "per_sensor":
        tr_idx, va_idx, te_idx = per_sensor_temporal_split(seq.sensor_id, seq.day_index)
    else:
        tr_idx, va_idx, te_idx = sensor_holdout_split(seq.sensor_id, seed=42)

    diag_rows = [
        {
            "split": "train",
            "n_windows": int(len(tr_idx)),
            "n_sensors": _n_sensors(tr_idx),
            "halt_prevalence": float(np.mean(seq.y_halt[tr_idx])),
        },
        {
            "split": "val",
            "n_windows": int(len(va_idx)),
            "n_sensors": _n_sensors(va_idx),
            "halt_prevalence": float(np.mean(seq.y_halt[va_idx])),
        },
        {
            "split": "test",
            "n_windows": int(len(te_idx)),
            "n_sensors": _n_sensors(te_idx),
            "halt_prevalence": float(np.mean(seq.y_halt[te_idx])),
        },
    ]
    pd.DataFrame(diag_rows).to_csv(OUT_TABLES / f"{artifact_stem}_split_diagnostics.csv", index=False)

    x_tr, x_va, x_te = seq.x[tr_idx], seq.x[va_idx], seq.x[te_idx]
    yb_tr, yb_va, yb_te = seq.y_batt[tr_idx], seq.y_batt[va_idx], seq.y_batt[te_idx]
    yh_tr, yh_va, yh_te = seq.y_halt[tr_idx], seq.y_halt[va_idx], seq.y_halt[te_idx]
    lb_tr = last_step_batt[tr_idx].astype(np.float32)
    lb_va = last_step_batt[va_idx].astype(np.float32)
    lb_te = last_step_batt[te_idx].astype(np.float32)
    _flat_tr = x_tr.reshape(-1, x_tr.shape[-1])
    batt_feat_mu = float(_flat_tr[:, batt_idx].mean())
    batt_feat_std = float(_flat_tr[:, batt_idx].std() + 1e-6)
    x_tr, x_va, x_te = telemetry.standardize(x_tr, x_va, x_te)
    residual_mode = args.battery_target == "residual"
    quantile_mode = args.battery_loss == "quantile"
    ordinal_mode = args.battery_loss == "ordinal"
    if quantile_mode and ordinal_mode:
        raise SystemExit("Choose one of --battery-loss quantile or ordinal.")
    _removed_flags: list[str] = []
    if args.halt_mode != "binary":
        _removed_flags.append("--halt-mode survival")
    if args.patch_len > 0:
        _removed_flags.append("--patch-len")
    if args.battery_regime_heads != "single":
        _removed_flags.append("--battery-regime-heads")
    if args.count_aux_weight > 0.0:
        _removed_flags.append("--count-aux-weight")
    if args.halt_stratified_batches:
        _removed_flags.append("--halt-stratified-batches")
    if _removed_flags:
        raise SystemExit(
            "Experiment modules were removed from node_analysis. Unsupported flags: "
            + ", ".join(_removed_flags)
            + ". Use default binary halt and MSE/quantile/ordinal battery."
        )
    quantile_taus = torch.tensor([0.1, 0.5, 0.9], dtype=torch.float32)
    if ordinal_mode:
        n_batt_out = int(args.ordinal_bins)
    elif quantile_mode:
        n_batt_out = 3
    else:
        n_batt_out = 1

    y_level_tr = (yb_tr.astype(np.float64) + lb_tr.astype(np.float64)).astype(np.float32) if residual_mode else yb_tr.astype(np.float32)
    y_level_va = (yb_va.astype(np.float64) + lb_va.astype(np.float64)).astype(np.float32) if residual_mode else yb_va.astype(np.float32)
    y_level_te = (yb_te.astype(np.float64) + lb_te.astype(np.float64)).astype(np.float32) if residual_mode else yb_te.astype(np.float32)

    bin_centers_np: np.ndarray | None = None
    if ordinal_mode:
        enc = KBinsDiscretizer(n_bins=n_batt_out, encode="ordinal", strategy="quantile", subsample=None)
        yb_tr_d = enc.fit_transform(y_level_tr.reshape(-1, 1)).astype(np.int64).ravel()
        yb_va_d = enc.transform(y_level_va.reshape(-1, 1)).astype(np.int64).ravel()
        yb_te_d = enc.transform(y_level_te.reshape(-1, 1)).astype(np.int64).ravel()
        n_eff = int(enc.n_bins_[0])
        yb_tr_d = np.clip(yb_tr_d, 0, n_eff - 1)
        yb_va_d = np.clip(yb_va_d, 0, n_eff - 1)
        yb_te_d = np.clip(yb_te_d, 0, n_eff - 1)
        edges = enc.bin_edges_[0]
        bin_centers_np = (edges[:-1] + edges[1:]) * 0.5

    n_halt_out = 1

    pos_rate_tr = float(np.mean(yh_tr))
    pos_weight = float((1.0 - pos_rate_tr) / max(pos_rate_tr, 1e-6))

    if ordinal_mode:
        train_ds = TensorDataset(
            torch.tensor(x_tr),
            torch.tensor(yb_tr_d, dtype=torch.long),
            torch.tensor(yh_tr),
            torch.tensor(lb_tr),
            torch.tensor(y_level_tr, dtype=torch.float32),
        )
    else:
        train_ds = TensorDataset(
            torch.tensor(x_tr),
            torch.tensor(yb_tr),
            torch.tensor(yh_tr),
            torch.tensor(lb_tr),
        )
    if args.weighted_sampler:
        yh_np = yh_tr.astype(np.float64)
        n_pos = float(np.sum(yh_np))
        n_neg = float(len(yh_np) - n_pos)
        w_pos = 0.5 / max(n_pos, 1.0)
        w_neg = 0.5 / max(n_neg, 1.0)
        sample_w = np.where(yh_np > 0.5, w_pos, w_neg)
        sampler = WeightedRandomSampler(torch.tensor(sample_w, dtype=torch.double), num_samples=len(sample_w), replacement=True)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler)
    else:
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)

    if ordinal_mode:
        val_ds = TensorDataset(
            torch.tensor(x_va),
            torch.tensor(yb_va_d, dtype=torch.long),
            torch.tensor(yh_va),
            torch.tensor(lb_va),
            torch.tensor(y_level_va, dtype=torch.float32),
        )
        test_ds = TensorDataset(
            torch.tensor(x_te),
            torch.tensor(yb_te_d, dtype=torch.long),
            torch.tensor(yh_te),
            torch.tensor(lb_te),
            torch.tensor(y_level_te, dtype=torch.float32),
        )
    else:
        val_ds = TensorDataset(torch.tensor(x_va), torch.tensor(yb_va), torch.tensor(yh_va), torch.tensor(lb_va))
        test_ds = TensorDataset(torch.tensor(x_te), torch.tensor(yb_te), torch.tensor(yh_te), torch.tensor(lb_te))
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    in_dim = x_tr.shape[-1]
    models = {
        "lstm": LSTMHead(in_dim=in_dim, n_battery_out=n_batt_out, n_halt_out=n_halt_out),
        "gru": GRUHead(in_dim=in_dim, n_battery_out=n_batt_out, n_halt_out=n_halt_out),
        "transformer": TransformerHeadV5(in_dim=in_dim, n_battery_out=n_batt_out, n_halt_out=n_halt_out),
    }
    if not args.no_tcn:
        models["tcn"] = TCNHead(in_dim=in_dim, n_battery_out=n_batt_out, n_halt_out=n_halt_out)

    rows: List[Dict] = []
    sweep_notes_parts: List[str] = []
    for name, model in models.items():
        print(f"Training {name}...")
        model, epochs_done = run_train(
            model,
            train_loader,
            val_loader,
            args.epochs,
            args.lr,
            device,
            halt_loss_weight=args.halt_loss_weight,
            max_grad_norm=args.max_grad_norm,
            focal_gamma=args.focal_gamma,
            focal_alpha_pos=args.focal_alpha_pos,
            halt_bce_mix=args.halt_bce_mix,
            pos_weight=pos_weight,
            weight_decay=args.weight_decay,
            early_stopping_patience=args.early_stopping_patience,
            min_delta=args.min_delta,
            halt_label_smoothing=args.halt_label_smoothing,
            battery_loss=args.battery_loss,
            quantile_taus=quantile_taus if quantile_mode else None,
            halt_mode=args.halt_mode,
            ordinal_training=ordinal_mode,
            count_aux_weight=0.0,
            progress_epochs=not args.no_progress_epochs,
            log_prefix=f"[{name}] ",
        )
        ck_dir = (getattr(args, "save_checkpoint_dir", "") or "").strip()
        if ck_dir:
            ck_path = Path(ck_dir)
            ck_path.mkdir(parents=True, exist_ok=True)
            out_pt = ck_path / f"{artifact_stem}_{name}_bundle.pt"
            torch.save(
                {"state_dict": model.state_dict(), "model": name, "artifact_stem": artifact_stem},
                out_pt,
            )
            print(f"Wrote checkpoint: {out_pt}")
        vb_pred, vh_prob, vb_true, vh_true, _, _ = predict(
            model,
            val_loader,
            device,
            battery_residual=residual_mode,
            battery_quantile=quantile_mode,
            battery_ordinal=ordinal_mode,
            bin_centers=bin_centers_np,
            halt_survival=False,
        )
        trb_pred, trh_prob, trb_true, trh_true, _, _ = predict(
            model,
            train_loader,
            device,
            battery_residual=residual_mode,
            battery_quantile=quantile_mode,
            battery_ordinal=ordinal_mode,
            bin_centers=bin_centers_np,
            halt_survival=False,
        )
        sweep_val = threshold_sweep_table(vh_true, vh_prob)
        sweep_val.to_csv(OUT_TABLES / f"{artifact_stem}_{name}_halt_threshold_sweep_val.csv", index=False)
        cls_t, th_src, sweep_applied = select_threshold_with_fallback(
            vh_true,
            vh_prob,
            trh_true,
            trh_prob,
            args.threshold_criterion,
            min_positives=max(15, int(0.02 * len(vh_true))),
            global_prevalence=global_prevalence,
            quantile_target_rate=args.quantile_target_rate,
        )
        sweep_applied.to_csv(OUT_TABLES / f"{artifact_stem}_{name}_halt_threshold_sweep_applied.csv", index=False)
        crit_col = {"f1": "f1", "f2": "f2", "balanced_accuracy": "balanced_accuracy"}[args.threshold_criterion]
        if th_src.startswith("train_val_quantile"):
            sweep_notes_parts.append(
                f"- **{name}** ({th_src}): threshold={cls_t:.4f} "
                f"(quantile alert rate ≈ calibration positive rate; full-dataset prevalence={global_prevalence:.4f})"
            )
        else:
            br = sweep_applied.iloc[int(sweep_applied[crit_col].values.argmax())]
            cls_t = float(br["threshold"])
            sweep_notes_parts.append(
                f"- **{name}** ({th_src}): threshold={cls_t:.3f} by {args.threshold_criterion} — "
                f"P={br['precision']:.3f} R={br['recall']:.3f} F1={br['f1']:.3f} F2={br['f2']:.3f}"
            )

        tb_pred, th_prob, tb_true, th_true, tb_q, tb_bin = predict(
            model,
            test_loader,
            device,
            battery_residual=residual_mode,
            battery_quantile=quantile_mode,
            battery_ordinal=ordinal_mode,
            bin_centers=bin_centers_np,
            halt_survival=False,
        )
        m = eval_metrics(tb_true, tb_pred, th_true, th_prob, cls_t)
        lg_val = collect_halt_logits(model, val_loader, device, survival_last_logit=False)
        lg_te = collect_halt_logits(model, test_loader, device, survival_last_logit=False)
        t_scale = fit_temperature_min_brier(lg_val, vh_true)
        p_te_scaled = halt_positive_probability_scaled(lg_te, t_scale)
        m["halt_temperature"] = t_scale
        m["halt_brier_tscaled"] = float(brier_score_loss(th_true.astype(int), p_te_scaled))
        m["model"] = name
        m["threshold_criterion"] = args.threshold_criterion
        m["threshold_source"] = th_src
        m["split_mode"] = split_mode_label
        m["halt_bce_mix"] = args.halt_bce_mix
        m["epochs_trained"] = int(epochs_done)
        m["weight_decay"] = args.weight_decay
        m["halt_label_smoothing"] = args.halt_label_smoothing
        rows.append(m)

        calibration_bins_table(th_true, th_prob, n_bins=10).to_csv(
            OUT_TABLES / f"{artifact_stem}_{name}_halt_calibration_bins.csv", index=False
        )
        save_calibration_figure(th_true, th_prob, name, artifact_stem)

        pd.DataFrame([m]).to_csv(OUT_TABLES / f"{artifact_stem}_{name}_battery_halt_metrics.csv", index=False)
        pred_cols: Dict[str, np.ndarray] = {
            "sensor": seq.sensor_id[te_idx],
            "day": seq.day_label[te_idx],
            "y_true_battery": tb_true,
            "y_pred_battery": tb_pred,
            "y_true_halt_within_h": th_true.astype(int),
            "y_pred_halt_prob": th_prob,
            "y_pred_halt_prob_tscaled": p_te_scaled,
            "y_pred_halt_cls": (th_prob >= cls_t).astype(int),
        }
        if tb_q is not None:
            pred_cols["y_pred_battery_q10"] = tb_q[:, 0]
            pred_cols["y_pred_battery_q50"] = tb_q[:, 1]
            pred_cols["y_pred_battery_q90"] = tb_q[:, 2]
        if tb_bin is not None:
            pred_cols["y_pred_battery_bin"] = tb_bin
            pred_cols["y_true_battery_bin"] = yb_te_d
        pd.DataFrame(pred_cols).to_csv(OUT_TABLES / f"{artifact_stem}_{name}_battery_halt_predictions.csv", index=False)

        save_curves(th_true.astype(int), th_prob, name, artifact_stem)
        save_battery_scatter(tb_true, tb_pred, name, artifact_stem)
        print(
            f"{name}: ep={epochs_done} batt_RMSE={m['battery_rmse']:.4f} R2={m['battery_r2']:.4f} "
            f"halt_F2={m['halt_f2']:.4f} F1={m['halt_f1']:.4f} AUC={m['halt_auc']:.4f} thr={cls_t:.3f} "
            f"Brier={m['halt_brier']:.4f} Brier_T={m['halt_brier_tscaled']:.4f} T={t_scale:.3f}"
        )

    cols = [
        "model",
        "split_mode",
        "halt_bce_mix",
        "threshold_criterion",
        "threshold_source",
        "battery_mae",
        "battery_rmse",
        "battery_r2",
        "halt_auc",
        "halt_ap",
        "halt_brier",
        "halt_brier_tscaled",
        "halt_temperature",
        "halt_precision",
        "halt_recall",
        "halt_f1",
        "halt_f2",
        "halt_specificity",
        "halt_prevalence",
        "halt_threshold",
        "halt_tp",
        "halt_fp",
        "halt_tn",
        "halt_fn",
        "epochs_trained",
        "weight_decay",
        "halt_label_smoothing",
    ]
    out_df = pd.DataFrame(rows)[cols]
    out_df.to_csv(OUT_TABLES / f"{artifact_stem}_model_comparison_battery_halt.csv", index=False)
    write_report(
        out_df,
        n_samples=int(len(seq.x)),
        n_sensors=n_sensors_all,
        args=args,
        sweep_notes="### Validation operating points\n" + "\n".join(sweep_notes_parts),
        artifact_stem=artifact_stem,
        split_mode_for_report=split_mode_label,
    )
    print(f"Wrote: {OUT_TABLES / f'{artifact_stem}_model_comparison_battery_halt.csv'}")
    print(f"Wrote: {OUT_REPORTS / f'{artifact_stem}_battery_halt_analysis.md'}")


if __name__ == "__main__":
    main()
