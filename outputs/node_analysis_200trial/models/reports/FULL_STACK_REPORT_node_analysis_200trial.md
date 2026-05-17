# Full academic stack report

> **Archive note (2026-05-16):** Historical log from training day. Checkpoints and `models/figures/` were removed in the lean archive. Figures were built afterward under `graphs/`. Paths mentioning `thesis_pipeline/` or `holistic_mtl_stl_academic_bundle/` refer to the upstream bundle, not this repo layout.

**run_id:** `node_analysis_200trial`
**started (UTC):** 2026-05-16T17:13:59.972020+00:00
## What was ordered (recap)

- **Thesis bundle:** dual preprocessing spec, calibration helpers, classification/regression metrics.
- **Optimizer:** AdamW (fixed in bundle trainer). **Calibration:** temperature scaling on halt logits.
- **Graphs:** one folder per figure key under `outputs/<run_id>/graphs/`; STL vs MTL is compared only as **metric deltas** in `stl_vs_mtl_delta` (and companion tables), not via formal multitask-gain formulas.
- **Statistics:** model comparison CSVs and optional checkpoints `.pt` under `outputs/<run_id>/models/`.

## Training configuration (this run)

- epochs=12, max_sensors=325, split=sensor_holdout, halt_bce_mix=0.25
- cohort_csv: *(historical run; file removed from repo — use `dummy_smoke_deveui.csv` or `--discover-cohort` for new runs)*
- checkpoints root: `E:\Oulu Time-Series models\node_analysis\outputs\node_analysis_200trial\models\checkpoints\training_node_analysis_200trial`

> **Note:** survival pass skipped (`--skip-survival`).

## Training phases

## Train step: `mtl_binary` → stem `node_analysis_node_analysis_200trial_multitask_binary`
```text
binary
```
- **return code:** 0

## Train step: `stl_battery` → stem `node_analysis_node_analysis_200trial_single_task_battery`
```text
binary
```
- **return code:** 0

## Train step: `stl_halt_heavy` → stem `node_analysis_node_analysis_200trial_single_task_halt_focus`
```text
binary
```
- **return code:** 0

## Train step: `noimp_mtl_binary` → stem `node_analysis_node_analysis_200trial_multitask_binary_no_impute` (no-impute)
- **return code:** 0

## Post-training artifact paths (key)

- **MTL binary (primary figures):** `E:\Oulu Time-Series models\node_analysis\outputs\node_analysis_200trial\models\tables\node_analysis_node_analysis_200trial_multitask_binary_model_comparison_battery_halt.csv` — exists=True
- **STL battery:** `E:\Oulu Time-Series models\node_analysis\outputs\node_analysis_200trial\models\tables\node_analysis_node_analysis_200trial_single_task_battery_model_comparison_battery_halt.csv` — exists=True
- **STL halt-heavy:** `E:\Oulu Time-Series models\node_analysis\outputs\node_analysis_200trial\models\tables\node_analysis_node_analysis_200trial_single_task_halt_focus_model_comparison_battery_halt.csv` — exists=True
- **Survival (resolved stem):** `E:\Oulu Time-Series models\node_analysis\outputs\node_analysis_200trial\models\tables\node_analysis_node_analysis_200trial_survival_halt_survival_model_comparison_battery_halt.csv` — exists=False
- **No-impute MTL binary:** `E:\Oulu Time-Series models\node_analysis\outputs\node_analysis_200trial\models\tables\node_analysis_node_analysis_200trial_multitask_binary_no_impute_model_comparison_battery_halt.csv` — exists=True

## STL vs MTL comparison
- **Figures only:** `stl_vs_mtl_delta` plots MTL minus STL metric deltas from comparison CSVs (`node_analysis_node_analysis_200trial_multitask_binary`, `node_analysis_node_analysis_200trial_single_task_battery`, `node_analysis_node_analysis_200trial_single_task_halt_focus`).

## Primary figure stem
- Thesis figures use multitask binary tables: `node_analysis_node_analysis_200trial_multitask_binary_*_{lstm,gru,transformer,tcn}_battery_halt_predictions.csv`.

## Figures (`figures/build.py`)

**Skipped** (`--skip-figures`).

## Summary

**Training:** all subprocess phases returned code 0.

**Elements touched:** only `holistic_mtl_stl_academic_bundle/` (this orchestrator + existing bundle modules + `runs/` artifacts) and **new outputs** under `thesis_pipeline/outputs/` from the **bundle trainer fork** — legacy `.py` sources outside the bundle are not edited.

**Finished (UTC):** 2026-05-16T17:53:48.416552+00:00

---
`FULL_STACK_ORCHESTRATOR_COMPLETE`
