# Metrics Companion — Deep learning time-series integration

Recreated figures in this folder:
- `deep_learning_time_series_integration.png`

## Clean table (multitask binary pass)
| quantity | value |
| --- | --- |
| N observed days | 1088 |
| Tail window k (days): max(10, ⌊0.15·N⌋) | 163 |
| τ battery (V): q=0.10 of true battery (regression reference line) | 3.537968 |
| τ halt probability: q=0.85 of predicted prob on this series (classification band) | 0.584934 |
| Tail battery drop (V): median(early-k true) − median(late-k true) | 0.446931 |
| Tail risk rise: median(late-k pred prob) − median(early-k pred prob) | 0.232784 |
| Tail peak risk: max predicted prob in tail window | 0.960591 |
| (R,H) contingency — observed days only (see matrix below) |  |
| R=0, H=0 | 0 |
| R=0, H=1 | 103 |
| R=1, H=0 | 924 |
| R=1, H=1 | 61 |

## (R,H) contingency (observed days)

Rows: R from predicted battery vs τ_batt (q=0.10 of true battery). Columns: H from predicted halt prob vs τ_halt (q=0.85 of pred prob on series).

|  | H=0 | H=1 |
| --- | --- | --- |
| R=0 | 0 | 103 |
| R=1 | 924 | 61 |

## Key values (after figure/table)
- Best PR-AUC: Transformer = 0.9348
- Best recall: GRU = 0.9046
- Best F2: GRU = 0.8968
- Best Brier (temp-scaled): GRU = 0.2546
- Main MTL/STL delta: GRU -> ΔF2(MTL-STL_hhalt)=+0.0427, ΔRMSE(MTL-STL_batt)=+0.0210