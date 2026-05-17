# Metrics Companion — Policy map

Recreated figures in this folder:
- `policy_map.png`

## Clean table (multitask binary pass)
| model | halt_ap | halt_recall | halt_f2 | halt_brier_tscaled | battery_rmse |
| --- | --- | --- | --- | --- | --- |
| lstm | 0.913268 | 0.891954 | 0.855945 | 0.254584 | 0.024007 |
| gru | 0.918969 | 0.904598 | 0.896764 | 0.254578 | 0.027643 |
| transformer | 0.934801 | 0.882759 | 0.895731 | 0.255340 | 0.133566 |
| tcn | 0.892254 | 0.864368 | 0.817569 | 0.255589 | 0.058537 |