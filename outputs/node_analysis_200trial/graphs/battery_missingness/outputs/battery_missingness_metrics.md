# Metrics Companion — Battery missingness

Recreated figures in this folder:
- `battery_missingness.png`

## Clean table (multitask binary pass)
| sensor_used | n_days | tail_battery_drop | min_battery | max_battery |
| --- | --- | --- | --- | --- |
| SYNTH_NODE_058 | 1196 | 0.011254 | 3.681964 | 3.705220 |

## Key values (after figure/table)
- Best PR-AUC: Transformer = 0.9348
- Best recall: GRU = 0.9046
- Best F2: GRU = 0.8968
- Best Brier (temp-scaled): GRU = 0.2546
- Main MTL/STL delta: GRU -> ΔF2(MTL-STL_hhalt)=+0.0427, ΔRMSE(MTL-STL_batt)=+0.0210