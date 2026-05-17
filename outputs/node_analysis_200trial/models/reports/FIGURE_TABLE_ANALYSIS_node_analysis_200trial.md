# Figure-Level Metrics Analysis

Run id: `node_analysis_200trial`

Supervisor-requested key values:
- Best PR-AUC: Transformer = 0.9348
- Best recall: GRU = 0.9046
- Best F2: GRU = 0.8968
- Best Brier (temp-scaled): GRU = 0.2546
- Main MTL/STL delta: GRU -> ΔF2(MTL-STL_hhalt)=+0.0427, ΔRMSE(MTL-STL_batt)=+0.0210

Temperature scaling impact view:
- Compare `halt_brier` vs `halt_brier_tscaled` in per-figure `graphs/<figure_key>/outputs/*_metrics.csv` companions where halt metrics apply.