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
- **lstm** (val): threshold=0.700 by f2 — P=0.856 R=0.963 F1=0.906 F2=0.940
- **gru** (val): threshold=0.870 by f2 — P=0.894 R=0.936 F1=0.914 F2=0.927
- **transformer** (val): threshold=0.910 by f2 — P=0.967 R=0.930 F1=0.948 F2=0.937
- **tcn** (val): threshold=0.840 by f2 — P=0.700 R=0.964 F1=0.811 F2=0.897

## Metrics (test)
      model     split_mode  halt_bce_mix threshold_criterion threshold_source  battery_mae  battery_rmse  battery_r2  halt_auc  halt_ap  halt_brier  halt_brier_tscaled  halt_temperature  halt_precision  halt_recall  halt_f1  halt_f2  halt_specificity  halt_prevalence  halt_threshold  halt_tp  halt_fp  halt_tn  halt_fn  epochs_trained  weight_decay  halt_label_smoothing
       lstm sensor_holdout          0.25                  f2              val     0.012462      0.024007    0.821221  0.940499 0.913268    0.321377            0.254584              10.0        0.736942     0.891954 0.807072 0.855945          0.991663         0.025517            0.70      776      277    32948       94               6        0.0002                  0.05
        gru sensor_holdout          0.25                  f2              val     0.014448      0.027643    0.762957  0.935844 0.918969    0.320714            0.254578              10.0        0.866740     0.904598 0.885264 0.896764          0.996358         0.025517            0.87      787      121    33104       83               5        0.0002                  0.05
transformer sensor_holdout          0.25                  f2              val     0.129447      0.133566   -4.534037  0.977341 0.934801    0.331409            0.255340              10.0        0.951673     0.882759 0.915921 0.895731          0.998826         0.025517            0.91      768       39    33186      102               7        0.0002                  0.05
        tcn sensor_holdout          0.25                  f2              val     0.034914      0.058537   -0.062948  0.973345 0.892254    0.327110            0.255589              10.0        0.672029     0.864368 0.756159 0.817569          0.988954         0.025517            0.84      752      367    32858      118               9        0.0002                  0.05

## Highlights
- Best halt F2 (test): `gru` — F2=0.8968, F1=0.8853, recall=0.9046, precision=0.8667, AUC=0.9358.
- Best battery RMSE: `lstm` — RMSE=0.0240, R2=0.8212.