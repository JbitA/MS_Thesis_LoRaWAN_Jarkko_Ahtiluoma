# Metrics Companion — STL vs MTL delta

Recreated figures in this folder:
- `stl_vs_mtl_delta.png`

## Clean table (multitask binary pass)
| model | halt_ap | halt_recall | halt_f2 | halt_brier_tscaled | battery_rmse |
| --- | --- | --- | --- | --- | --- |
| lstm | 0.913268 | 0.891954 | 0.855945 | 0.254584 | 0.024007 |
| gru | 0.918969 | 0.904598 | 0.896764 | 0.254578 | 0.027643 |
| transformer | 0.934801 | 0.882759 | 0.895731 | 0.255340 | 0.133566 |
| tcn | 0.892254 | 0.864368 | 0.817569 | 0.255589 | 0.058537 |

## MTL and STL (absolute, not delta)

Battery metrics (RMSE, R²): MTL vs **STL trained on battery only**. Halt metrics (AUC, AP, F2, BS): MTL vs **STL trained on halt only**.

| Model | RMSE | R² | AUC | AP | F2 | BS |
| --- | --- | --- | --- | --- | --- | --- |
| LSTM MTL | 0.024007 | 0.821221 | 0.940499 | 0.913268 | 0.855945 | 0.321377 |
| LSTM STL | 0.005130 | 0.991836 | 0.939253 | 0.904152 | 0.847089 | 0.321806 |
| GRU MTL | 0.027643 | 0.762957 | 0.935844 | 0.918969 | 0.896764 | 0.320714 |
| GRU STL | 0.006658 | 0.986250 | 0.959016 | 0.903404 | 0.854101 | 0.324269 |
| TCN MTL | 0.058537 | -0.062948 | 0.973345 | 0.892254 | 0.817569 | 0.327110 |
| TCN STL | 0.046183 | 0.338374 | 0.966431 | 0.868652 | 0.815050 | 0.330049 |
| Transformer MTL | 0.133566 | -4.534037 | 0.977341 | 0.934801 | 0.895731 | 0.331409 |
| Transformer STL | 0.040394 | 0.493844 | 0.951624 | 0.920681 | 0.896107 | 0.318219 |

Companion sheet PNG: `{companion_absolute_metrics_sheet_png(slug)}`

## Key values (after figure/table)
- Best PR-AUC: Transformer = 0.9348
- Best recall: GRU = 0.9046
- Best F2: GRU = 0.8968
- Best Brier (temp-scaled): GRU = 0.2546
- Main MTL/STL delta: GRU -> ΔF2(MTL-STL_hhalt)=+0.0427, ΔRMSE(MTL-STL_batt)=+0.0210