# Artifact map — `node_analysis_200trial`

Example run for the public thesis **source-code bundle** (2026-05-16).  
Device column values are **synthetic labels** (`SYNTH_NODE_001` … `SYNTH_NODE_200`), not pseudonyms of real LoRaWAN EUIs. There is no mapping table in the repo.

**Training** produced `models/`; **figures** produced `graphs/` (11 thesis keys).

Developed with Cursor AI.

---

## A. Model creation (training)

### A.1 How it was run

| Item | Path / value |
|------|----------------|
| CLI | `node_analysis/main.py` |
| Orchestrator | `pipeline/src/node_analysis_pipeline/orchestration/cohort_training_orchestrator.py` |
| Trainer (subprocess) | `pipeline/runtime_vendor/src/multitask_battery_halt_trainer_bundle.py` |
| No-impute pass | `pipeline/src/node_analysis_pipeline/training/train_multitask_no_impute.py` |
| Cohort (historical) | not shipped — example tables use `SYNTH_NODE_*`; new runs use `--cohort-csv` or `--dummy-smoke` |
| Campus data | `node_analysis/idata/` (env: `NODE_ANALYSIS_DATA_ROOT`) |
| Report | `models/reports/FULL_STACK_REPORT_node_analysis_200trial.md` |

Training passes (all exit code 0; survival skipped):

| Pass | Artifact stem suffix |
|------|----------------------|
| Multitask binary (primary) | `node_analysis_node_analysis_200trial_multitask_binary` |
| Single-task battery | `…_single_task_battery` |
| Single-task halt focus | `…_single_task_halt_focus` |
| No-impute multitask | `…_multitask_binary_no_impute` |

Stem logic: `pipeline/src/node_analysis_pipeline/artifact_stems.py`.

### A.2 Training outputs (keep)

```
outputs/node_analysis_200trial/models/
├── tables/                    # 88 CSVs — primary inputs for all figures
│   ├── *_multitask_binary_*                    # MTL imputed (main)
│   ├── *_single_task_battery_*                # STL battery baseline
│   ├── *_single_task_halt_focus_*              # STL halt baseline
│   └── *_multitask_binary_no_impute_*          # MTL without imputation
├── reports/
│   ├── FULL_STACK_REPORT_node_analysis_200trial.md
│   ├── FIGURE_TABLE_ANALYSIS_node_analysis_200trial.md
│   └── *_battery_halt_analysis.md (one per training pass, from trainer)
└── (no checkpoints/records in lean archive — removed 2026-05-16)
```

**Not used:** `models/figures/` — trainer ROC/PR/scatter PNGs (removed; disabled on future runs).

**Lean archive (this run):** keep `models/tables/`, `models/reports/`, and `graphs/` only.

Per-model table pattern (×4 models: lstm, gru, tcn, transformer):

- `{stem}_{model}_battery_halt_predictions.csv`
- `{stem}_{model}_halt_threshold_sweep_applied.csv` (and `_val`, calibration bins, metrics)
- `{stem}_model_comparison_battery_halt.csv` (one per pass)
- `{stem}_split_diagnostics.csv`

### A.3 Training inputs (repo, not under outputs)

| Asset | Role |
|-------|------|
| `pipeline/runtime_vendor/src/telemetry_sequence_pipeline.py` | Daily panel from `idata` |
| `pipeline/runtime_vendor/src/backbone_architectures.py` | LSTM/GRU/TCN/Transformer |
| `pipeline/src/node_analysis_pipeline/layout.py` | `outputs/<run_id>/models` paths |
| `pipeline/src/node_analysis_pipeline/env_config.py` | Env keys |

---

## B. Figure creation (graphs)

### B.1 How figures are built

| Step | Script |
|------|--------|
| 1 | `pipeline/scripts/run_figures.py` → `pipeline/figures/build.py` |
| 2 | `pipeline/scripts/run_companion_tables.py` → `pipeline/figures/generate_figure_companion_tables.py` |

Shared code:

| File | Role |
|------|------|
| `pipeline/figures/recreated_from_new_data.py` | Primary PNG renderers (`make_for_figure_key`) |
| `pipeline/figures/build.py` | Orchestrates all keys; extras: LSTM regression, tier-2 utility table |
| `pipeline/figures/util.py` | Output paths + bundled CSV paths |
| `pipeline/figures/figure_titles.py` | Canonical PNG/companion basenames |
| `pipeline/figures/thesis_figure_keys.py` | Allowlist + legacy migration |
| `pipeline/assets/battery_missingness_reference_series.csv` | Battery missingness figure only |
| `pipeline/assets/ribbon_chart_default_sensor.csv` | Ribbon figure sensor hint |

### B.2 Per-figure map (code → tables → files)

| Figure key | Primary PNG | Renderer (`recreated_from_new_data.py`) | Main table stem(s) | Extra from `build.py` |
|------------|-------------|----------------------------------------|--------------------|------------------------|
| `battery_missingness` | `battery_missingness.png` | `battery_missingness` | Asset CSV + optional `idata` | — |
| `confusion_grid` | `confusion_grid.png` | `confusion_grid` | `*_multitask_binary_*_predictions` | — |
| `imputed_panel` | `imputed_panel.png` | `imputed_panel` | MTL + no-impute predictions | — |
| `imputed_vs_non_imputed_halt` | `imputed_vs_non_imputed_halt.png` | `imputed_vs_non_imputed_halt` | MTL + no-impute predictions | — |
| `threshold_f2_impute_sensitivity_comparison` | `threshold_f2_impute_sensitivity_comparison.png` | `threshold_f2_impute_sensitivity_comparison` | MTL + no-impute sweeps | — |
| `policy_map` | `policy_map.png` | `policy_map` | `*_halt_threshold_sweep_applied` | — |
| `pr_f2_only` | `pr_f2_only.png` | `pr_f2_only` | MTL predictions + sweeps | — |
| `method_model_ranking` | `method_model_ranking.png` | `method_model_ranking` | `*_multitask_binary_model_comparison` | `method_model_ranking_utility_table.png` + `.csv` |
| `regression_4_models` | `regression_4_models.png` | `regression_4_models` | MTL predictions | `regression_lstm_only.png` |
| `stl_vs_mtl_delta` | `stl_vs_mtl_delta.png` | `stl_vs_mtl_delta` | MTL + STL battery + STL halt comparisons | — |
| `deep_learning_time_series_integration` | `deep_learning_time_series_integration.png` | `deep_learning_time_series_integration` | Transformer predictions + ribbon asset | — |

`*` = `node_analysis_node_analysis_200trial_multitask_binary` (and `…_no_impute` where noted).

### B.3 Figure outputs (keep) — per key

```
outputs/node_analysis_200trial/graphs/<figure_key>/outputs/
```

| Figure key | Files to keep |
|------------|----------------|
| All 11 keys | `{key}_metrics.csv`, `{key}_metrics.md`, `{key}_metrics.png` |
| `battery_missingness` | `battery_missingness.png` |
| `confusion_grid` | `confusion_grid.png` |
| `imputed_panel` | `imputed_panel.png` |
| `imputed_vs_non_imputed_halt` | `imputed_vs_non_imputed_halt.png` |
| `threshold_f2_impute_sensitivity_comparison` | `threshold_f2_impute_sensitivity_comparison.png` |
| `policy_map` | `policy_map.png` |
| `pr_f2_only` | `pr_f2_only.png` |
| `method_model_ranking` | `method_model_ranking.png`, `method_model_ranking_utility_table.png`, `method_model_ranking_utility_table.csv` |
| `regression_4_models` | `regression_4_models.png`, `regression_lstm_only.png` |
| `stl_vs_mtl_delta` | `stl_vs_mtl_delta.png`, `stl_vs_mtl_delta_absolute_metrics.csv`, `stl_vs_mtl_delta_absolute_metrics.png` |
| `deep_learning_time_series_integration` | `deep_learning_time_series_integration.png`, `deep_learning_time_series_contingency_matrix.csv`, `.md`, `.png` |

**Total:** 49 files under `graphs/` (current clean tree).

### B.4 Do not keep (legacy names)

If any appear after old runs, delete via `build.py` migration or manually:

- `fig_*` folders, `recreated_*.png`, `method_b_tiers*`, `tier2_methodB_*`, `table_metrics_*` generics

---

## C. Integral `node_analysis` repo layout (code only)

```
node_analysis/
├── main.py
├── idata/                              # campus telemetry (read-only input)
├── pipeline/
│   ├── cohorts/dummy_smoke_deveui.csv   ← only cohort file in repo
│   ├── assets/                         # figure-only CSVs
│   ├── figures/                        # build + recreated + companions
│   ├── scripts/run_figures.py, run_companion_tables.py
│   ├── src/node_analysis_pipeline/     # env, layout, stems, orchestration, training
│   └── runtime_vendor/src/             # trainer + telemetry
└── outputs/
    └── node_analysis_200trial/         # THIS RUN ONLY (other runs removed)
```

---
