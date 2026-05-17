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
- Samples: 156511; sensors: 200.
- halt_loss_weight=4.0, halt_bce_mix=0.25, focal_gamma=2.0, focal_alpha_pos=0.75.
- **V7:** weight_decay=0.0002, early_stopping_patience=2, min_delta=1e-05, halt_label_smoothing=0.05.

## Interpretation
- High AUC with low F1 at a fixed threshold usually indicates **ranking signal** + **wrong operating point** under imbalance.
- F2 emphasizes recall; use it when missing an imminent halt is costlier than extra alerts.
- Transformer battery regression can remain weak if inductive bias mismatches the panel; compare GRU/LSTM first.
- Use **`--quantile-target-rate`** (e.g. `0.16`) only as a *scenario*: fixes alert budget when calibration months have almost no halt positives but a later deployment period is riskier.
- **`sensor_holdout`**: very high AUC → strong separability within cohort; confirm with **V4** `--sensor-cv-folds` if needed.
- Compare **`halt_brier`** vs **`halt_brier_tscaled`**: large drop implies miscalibrated probabilities fixable by simple scaling.

### Validation operating points
- **lstm** (train_val): threshold=0.940 by f2 — P=0.741 R=0.903 F1=0.814 F2=0.865
- **gru** (train_val): threshold=0.920 by f2 — P=0.872 R=0.928 F1=0.899 F2=0.916
- **transformer** (train_val): threshold=0.970 by f2 — P=0.811 R=0.902 F1=0.854 F2=0.882
- **tcn** (train_val): threshold=0.920 by f2 — P=0.481 R=0.890 F1=0.624 F2=0.761

## Metrics (test)
      model     split_mode  halt_bce_mix threshold_criterion threshold_source  battery_mae  battery_rmse  battery_r2  halt_auc  halt_ap  halt_brier  halt_brier_tscaled  halt_temperature  halt_precision  halt_recall  halt_f1  halt_f2  halt_specificity  halt_prevalence  halt_threshold  halt_tp  halt_fp  halt_tn  halt_fn  epochs_trained  weight_decay  halt_label_smoothing
       lstm sensor_holdout          0.25                  f2        train_val     0.021022      0.084130   -3.176913  0.915282 0.705520    0.628514            0.284350              10.0        0.597156     0.700000 0.644501 0.676692          0.996159         0.008069            0.94      126       85    22042       54               6        0.0002                  0.05
        gru sensor_holdout          0.25                  f2        train_val     0.027905      0.033710    0.329404  0.849365 0.770077    0.631881            0.284447              10.0        0.783133     0.722222 0.751445 0.733634          0.998373         0.008069            0.92      130       36    22091       50               9        0.0002                  0.05
transformer sensor_holdout          0.25                  f2        train_val     0.156131      0.160622  -14.225131  0.960855 0.751391    0.636181            0.285030              10.0        0.645833     0.688889 0.666667 0.679825          0.996927         0.008069            0.97      124       68    22059       56               5        0.0002                  0.05
        tcn sensor_holdout          0.25                  f2        train_val     0.105092      0.406614  -96.569889  0.921795 0.297521    0.627032            0.284539              10.0        0.348442     0.683333 0.461538 0.573159          0.989605         0.008069            0.92      123      230    21897       57               5        0.0002                  0.05

## Highlights
- Best halt F2 (test): `gru` — F2=0.7336, F1=0.7514, recall=0.7222, precision=0.7831, AUC=0.8494.
- Best battery RMSE: `gru` — RMSE=0.0337, R2=0.3294.