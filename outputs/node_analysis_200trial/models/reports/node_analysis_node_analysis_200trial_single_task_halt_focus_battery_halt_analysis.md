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
- halt_loss_weight=100.0, halt_bce_mix=0.25, focal_gamma=2.0, focal_alpha_pos=0.75.
- **V7:** weight_decay=0.0002, early_stopping_patience=2, min_delta=1e-05, halt_label_smoothing=0.05.

## Interpretation
- High AUC with low F1 at a fixed threshold usually indicates **ranking signal** + **wrong operating point** under imbalance.
- F2 emphasizes recall; use it when missing an imminent halt is costlier than extra alerts.
- Transformer battery regression can remain weak if inductive bias mismatches the panel; compare GRU/LSTM first.
- Use **`--quantile-target-rate`** (e.g. `0.16`) only as a *scenario*: fixes alert budget when calibration months have almost no halt positives but a later deployment period is riskier.
- **`sensor_holdout`**: very high AUC → strong separability within cohort; confirm with **V4** `--sensor-cv-folds` if needed.
- Compare **`halt_brier`** vs **`halt_brier_tscaled`**: large drop implies miscalibrated probabilities fixable by simple scaling.

### Validation operating points
- **lstm** (val): threshold=0.730 by f2 — P=0.753 R=0.961 F1=0.844 F2=0.911
- **gru** (val): threshold=0.750 by f2 — P=0.781 R=0.940 F1=0.853 F2=0.903
- **transformer** (val): threshold=0.850 by f2 — P=0.900 R=0.938 F1=0.918 F2=0.930
- **tcn** (val): threshold=0.910 by f2 — P=0.676 R=0.917 F1=0.779 F2=0.856

## Metrics (test)
      model     split_mode  halt_bce_mix threshold_criterion threshold_source  battery_mae  battery_rmse  battery_r2  halt_auc  halt_ap  halt_brier  halt_brier_tscaled  halt_temperature  halt_precision  halt_recall  halt_f1  halt_f2  halt_specificity  halt_prevalence  halt_threshold  halt_tp  halt_fp  halt_tn  halt_fn  epochs_trained  weight_decay  halt_label_smoothing
       lstm sensor_holdout          0.25                  f2              val     0.029426      0.050125    0.220611  0.939253 0.904152    0.321806            0.254676              10.0        0.693950     0.896552 0.782347 0.847089          0.989646         0.025517            0.73      780      344    32881       90               5        0.0002                  0.05
        gru sensor_holdout          0.25                  f2              val     0.048679      0.150369   -6.014012  0.959016 0.903404    0.324269            0.255574              10.0        0.807281     0.866667 0.835920 0.854101          0.994582         0.025517            0.75      754      180    33045      116               3        0.0002                  0.05
transformer sensor_holdout          0.25                  f2              val     0.186173      0.196210  -10.942407  0.951624 0.920681    0.318219            0.254078              10.0        0.903600     0.894253 0.898902 0.896107          0.997502         0.025517            0.85      778       83    33142       92               8        0.0002                  0.05
        tcn sensor_holdout          0.25                  f2              val     0.146588      0.229569  -15.348312  0.966431 0.868652    0.330049            0.255700              10.0        0.689239     0.854023 0.762834 0.815050          0.989917         0.025517            0.91      743      335    32890      127               6        0.0002                  0.05

## Highlights
- Best halt F2 (test): `transformer` — F2=0.8961, F1=0.8989, recall=0.8943, precision=0.9036, AUC=0.9516.
- Best battery RMSE: `lstm` — RMSE=0.0501, R2=0.2206.