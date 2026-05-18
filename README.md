# node_analysis — LoRaWAN battery and halt models

Master's Thesis Analysis of Smart Campus Internet of Things (IoT) Sensor Network Database code.
Reproducible code for LoRaWAN battery and halt modeling. 
Trains multitask LSTM, GRU, TCN, and Transformer models and builds eleven thesis figures from one entry point.

**Developed by Jarkko Ahtiluoma 17.5.2026 with Cursor AI.**

---

## What data is in this repository (read this first)

This public tree is a **thesis source-code example**. Device labels, bundled CSVs, and the sample run under `outputs/node_analysis_200trial/` use **obviously synthetic identifiers**, not pseudonyms of real campus hardware.

| What you see | Meaning |
|--------------|---------|
| `SYNTH_NODE_001` … `SYNTH_NODE_200` | **Fake device names** for the example cohort and training tables. They are **not** LoRaWAN EUIs and **cannot** be linked to physical nodes without data that is **not** shipped here. |
| `idata/oulu-smartcampus-release-mock-*` | **Synthetic telemetry** (50 devices: `SYNTH_NODE_001`–`SYNTH_NODE_050` in the `deveui` column). Values resemble real sensor streams; **timestamps are for demonstration only**. |
| `pipeline/cohorts/dummy_smoke_deveui.csv` | **Only cohort file in the repo** — eight obviously fake ids for `python main.py --dummy-smoke`. |
| `outputs/node_analysis_200trial/` | **Example run artifacts** (tables + graphs) produced from that synthetic/example pipeline state. Metrics illustrate the workflow; they are **not** confidential campus results. |
| `aa-11-22-33-…` (dummy smoke only) | Clearly fake IDs used by `python main.py --dummy-smoke` for a minimal local test. |

---

## Using your own real dataset

You do **not** need a special API in this project—only the **on-disk layout**, a **cohort CSV**, and a **new run id**.

1. **Place telemetry** under a root directory (default `node_analysis/idata/`). The trainer loads every `*/application.csv` (folders whose names do not start with `_`). Each CSV must include at least `time` (ms UTC), `deveui`, and `battery` (see `idata/README.md`). Use your real LoRaWAN EUIs or any stable string ids in the `deveui` column.

2. **Build a cohort allowlist** — a one-column CSV named `deveui` (e.g. `pipeline/cohorts/my_campus_run.csv`), or run `python main.py --discover-cohort` to create `ers_co2_max_available_deveui.csv` from your `idata` (that file is generated locally, not shipped in the repo).

3. **Pick a new run id** (any filesystem-safe name). Outputs will go to `outputs/<run_id>/` only; they will not overwrite `node_analysis_200trial` unless you reuse that name on purpose.

4. **Train and build figures** (PowerShell example):

```powershell
cd node_analysis
python -m venv .venv
.\.venv\Scripts\activate
pip install -r pipeline\requirements.txt

# Optional if idata is elsewhere:
# $env:NODE_ANALYSIS_DATA_ROOT = "D:\campus_lorawan\idata"

python main.py --epochs 12 --run-id my_campus_2026 `
  --cohort-csv pipeline\cohorts\my_campus_run.csv --skip-survival
```

5. **Inspect results** under `outputs/my_campus_2026/models/tables/` (CSVs for figures) and `outputs/my_campus_2026/graphs/<figure_key>/outputs/` (PNG + companion metrics).

To rebuild figures without retraining:

```powershell
python main.py --graphs-only --run-id my_campus_2026
```

---

## Layout

```
main.py
idata/                          # telemetry root (see idata/README.md)
outputs/<run_id>/
  models/tables/                # training CSVs (required for figures)
  models/reports/               # markdown run logs (optional)
  graphs/<figure_key>/outputs/  # thesis PNGs + companions
pipeline/                       # source, cohorts, trainer, figures
```

Trainer **does not** write `models/figures/` (diagnostic ROC/scatter PNGs disabled). Thesis images live only under `graphs/`.

**Python map:** [`PYTHON_FILE_MAP.md`](PYTHON_FILE_MAP.md)  
**Example run layout:** [`outputs/node_analysis_200trial/ARTIFACT_MAP.md`](outputs/node_analysis_200trial/ARTIFACT_MAP.md)

---

## Requirements

- Python 3.10+
- `pip install -r pipeline/requirements.txt`

---

## Quick start (dummy smoke — only bundled cohort)

```powershell
python main.py --dummy-smoke --epochs 2
```

Uses `pipeline/cohorts/dummy_smoke_deveui.csv` (eight devices `aa-11-22-33-…`) and seeds `idata/dummy_smoke/`. Outputs go to `outputs/node_analysis_dummy/`.

The committed example under `outputs/node_analysis_200trial/` was produced with a larger synthetic cohort. Rebuild figures only:

```powershell
python main.py --graphs-only --run-id node_analysis_200trial
```

---

## Thesis figure set (11 keys)

Allowlist: `pipeline/figures/thesis_figure_keys.py`. PNG basenames: `pipeline/figures/figure_titles.py`.

| Figure key | Primary PNG |
|------------|-------------|
| `battery_missingness` | `battery_missingness.png` |
| `confusion_grid` | `confusion_grid.png` |
| `imputed_panel` | `imputed_panel.png` |
| `imputed_vs_non_imputed_halt` | `imputed_vs_non_imputed_halt.png` |
| `threshold_f2_impute_sensitivity_comparison` | `threshold_f2_impute_sensitivity_comparison.png` |
| `policy_map` | `policy_map.png` |
| `pr_f2_only` | `pr_f2_only.png` |
| `method_model_ranking` | `method_model_ranking.png` (+ `method_model_ranking_utility_table.png`) |
| `regression_4_models` | `regression_4_models.png` (+ `regression_lstm_only.png`) |
| `stl_vs_mtl_delta` | `stl_vs_mtl_delta.png` |
| `deep_learning_time_series_integration` | `deep_learning_time_series_integration.png` |

Companions: most keys ship `{key}_metrics.csv`, `.md`, `.png`. Exceptions:

- `stl_vs_mtl_delta` — also `stl_vs_mtl_delta_absolute_metrics.*`
- `method_model_ranking` — `method_model_ranking_utility_table.csv` / `.png`
- `deep_learning_time_series_integration` — `deep_learning_time_series_contingency_matrix.*`
