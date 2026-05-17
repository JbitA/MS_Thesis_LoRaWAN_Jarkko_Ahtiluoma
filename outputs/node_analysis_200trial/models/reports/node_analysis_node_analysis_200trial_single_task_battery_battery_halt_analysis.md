# V7 Optimized Battery + Halt (V5 + early stop, weight decay, halt label smoothing)

> **Archive note:** Calibration **CSVs** remain in `models/tables/`. Reliability **PNGs** under `models/figures/` are disabled; thesis plots live under `outputs/.../graphs/` only.

## Objective
- Regression: next-day battery mean.
- Classification: permanent halt within 30 days (same operational definition as V2).

## What changed vs V2 / V3 / V5
- Same halt loss and splits as **V3** (focal + optional BCE mix, GRU, Transformer, global / per_sensor / sensor_holdout).
- **TCNHead**: stacked 1D convolutions + last-step pooling (local temporal inductive bias vs recurrent/attention).
- **Temperature scaling**: single `T` fit on **validation** to minimize Brier; reported as `halt_brier_tscaled` (AUC unchanged).
- **Calibration bins** CSV per model on **test** predictions (no trainer diagnostic PNGs).
- **V7 training (vs V5):** AdamW **weight_decay**, **early stopping** on composite val loss, **halt label smoothing** in focal/BCE.
- Validation threshold chosen by **`f2`** after a dense sweep; test metrics use that threshold.
- If val (then train+val) lacks enough positives/negatives for a stable F-score sweep, a **quantile threshold** matches the calibration-set positive rate (see `threshold_source`).
- **Temporal shift**: halt labels can concentrate in late calendar months; test prevalence may exceed train+val prevalence — compare AUC (ranking) vs F-scores (threshold-sensitive).
- Per-model CSV: `*_sweep_val.csv` and `*_sweep_applied.csv` (the table used to pick the threshold).

## Dataset and protocol
- **Split mode `split_mode=sensor_holdout`**: `global` = 70/15/15 by **calendar day** (V2-style); `per_sensor` = 70/15/15 along each **device timeline** (positives may still concentrate in tail splits); `sensor_holdout` = 70/15/15 **devices** — test is **unseen sensors**; train/val each see full device lives. `sensor_cv_Kfold` = one fold of **device-disjoint** k-fold (same index logic as V4); remaining devices split train/val by device.
- Sequence length: 21.
- Samples: 233128; sensors: 200.
- halt_loss_weight=0.0, halt_bce_mix=0.25, focal_gamma=2.0, focal_alpha_pos=0.75.
- **V7:** weight_decay=0.0002, early_stopping_patience=2, min_delta=1e-05, halt_label_smoothing=0.05.

## Interpretation
- High AUC with low F1 at a fixed threshold usually indicates **ranking signal** + **wrong operating point** under imbalance.
- F2 emphasizes recall; use it when missing an imminent halt is costlier than extra alerts.
- Transformer battery regression can remain weak if inductive bias mismatches the panel; compare GRU/LSTM first.
- Use **`--quantile-target-rate`** (e.g. `0.16`) only as a *scenario*: fixes alert budget when calibration months have almost no halt positives but a later deployment period is riskier.
- **`sensor_holdout`**: very high AUC → strong separability within cohort; confirm with **V4** `--sensor-cv-folds` if needed.
- Compare **`halt_brier`** vs **`halt_brier_tscaled`**: large drop implies miscalibrated probabilities fixable by simple scaling.

### Validation operating points
- **lstm** (val): threshold=0.010 by f2 — P=0.025 R=1.000 F1=0.048 F2=0.112
- **gru** (val): threshold=0.480 by f2 — P=0.077 R=0.290 F1=0.122 F2=0.187
- **transformer** (val): threshold=0.650 by f2 — P=0.096 R=0.416 F1=0.157 F2=0.250
- **tcn** (val): threshold=0.250 by f2 — P=0.025 R=1.000 F1=0.048 F2=0.112

## Metrics (test)
      model     split_mode  halt_bce_mix threshold_criterion threshold_source  battery_mae  battery_rmse  battery_r2  halt_auc  halt_ap  halt_brier  halt_brier_tscaled  halt_temperature  halt_precision  halt_recall  halt_f1  halt_f2  halt_specificity  halt_prevalence  halt_threshold  halt_tp  halt_fp  halt_tn  halt_fn  epochs_trained  weight_decay  halt_label_smoothing
       lstm sensor_holdout          0.25                  f2              val     0.002254      0.005130    0.991836  0.150047 0.014507    0.249884            0.249068          0.100000        0.025517     1.000000 0.049764 0.115768          0.000000         0.025517            0.01      870    33225        0        0              12        0.0002                  0.05
        gru sensor_holdout          0.25                  f2              val     0.002936      0.006658    0.986250  0.715893 0.165428    0.226620            0.084272          0.100000        0.080977     0.289655 0.126570 0.191141          0.913920         0.025517            0.48      252     2860    30365      618               7        0.0002                  0.05
transformer sensor_holdout          0.25                  f2              val     0.037631      0.040394    0.493844  0.585769 0.115271    0.398606            0.263512         10.000000        0.096052     0.394253 0.154470 0.243228          0.902844         0.025517            0.65      343     3228    29997      527               7        0.0002                  0.05
        tcn sensor_holdout          0.25                  f2              val     0.031105      0.046183    0.338374  0.118562 0.013674    0.233938            0.225517          0.398107        0.025521     1.000000 0.049771 0.115784          0.000150         0.025517            0.25      870    33220        5        0               7        0.0002                  0.05

## Highlights
- Best halt F2 (test): `transformer` — F2=0.2432, F1=0.1545, recall=0.3943, precision=0.0961, AUC=0.5858.
- Best battery RMSE: `lstm` — RMSE=0.0051, R2=0.9918.