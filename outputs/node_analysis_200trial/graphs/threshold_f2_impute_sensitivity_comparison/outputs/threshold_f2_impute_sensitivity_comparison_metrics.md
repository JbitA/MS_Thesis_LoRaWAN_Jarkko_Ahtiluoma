# Metrics Companion — Threshold F2 impute sensitivity comparison

Recreated figures in this folder:
- `threshold_f2_impute_sensitivity_comparison.png`

## Clean table (multitask binary pass)
| model | imputed_f2 | nonimputed_f2 | imputed_brier_tscaled | nonimputed_brier_tscaled | delta_f2_imputed_minus_nonimputed | delta_rmse_imputed_minus_nonimputed |
| --- | --- | --- | --- | --- | --- | --- |
| lstm | 0.855945 | 0.676692 | 0.254584 | 0.284350 | 0.179254 | -0.060124 |
| gru | 0.896764 | 0.733634 | 0.254578 | 0.284447 | 0.163130 | -0.006066 |
| tcn | 0.817569 | 0.573159 | 0.255589 | 0.284539 | 0.244410 | -0.348077 |
| transformer | 0.895731 | 0.679825 | 0.255340 | 0.285030 | 0.215907 | -0.027056 |

## Key values (after figure/table)
- Best PR-AUC: Transformer = 0.9348
- Best recall: GRU = 0.9046
- Best F2: GRU = 0.8968
- Best Brier (temp-scaled): GRU = 0.2546
- Main MTL/STL delta: GRU -> ΔF2(MTL-STL_hhalt)=+0.0427, ΔRMSE(MTL-STL_batt)=+0.0210