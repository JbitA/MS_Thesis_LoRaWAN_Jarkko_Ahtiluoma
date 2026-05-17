# `idata` (local telemetry)

## What is in this folder today

The checked-in tree is **synthetic example data** for the MS thesis **source-code bundle**, not a dump of real Oulu Smart Campus uplinks.

| Content | Description |
|---------|-------------|
| `oulu-smartcampus-release-mock-20260516-*/application.csv` | **Fake release shards**: multiple devices per file, `deveui` values `SYNTH_NODE_001` … `SYNTH_NODE_050` (numeric suffix matches the node index). Sensor fields (battery, CO₂, etc.) are **simulated** for pipeline testing. |
| `_schema_template/` | Column documentation and a header-only `application.csv` example. **Ignored** by training (folder name starts with `_`). |
| `dummy_smoke/` | Created only when you run `python main.py --dummy-smoke` — eight devices with ids `aa-11-22-33-…` (obviously not real hardware). |

**There is no pseudonym map** and no real LoRaWAN EUIs in the example shards. Names are chosen so readers can see they are placeholders.

---

## Layout the trainer expects

The ingestion code (`telemetry_sequence_pipeline.py`) scans:

```
<DATA_ROOT>/*/application.csv
```

Any subfolder (except names starting with `_`) may hold one `application.csv`. Rows from all files are merged; the **`deveui` column** is the device key (string). Folder names do **not** have to equal `deveui`, but using one folder per device is a common layout for real exports:

```
idata/
  <your-device-id>/
    application.csv
  _schema_template/          # optional documentation only
```

The bundled mock release uses **date-stamped folders** instead; that is also valid because ids live **inside** the CSV.

---

## Columns used by this repo

| Column | Dtype | Used by |
|--------|-------|---------|
| `time` | int64 (ms since Unix epoch, UTC) | Training pipeline |
| `deveui` | string | Training pipeline, cohort CSV |
| `battery` | float64 (volts) | Battery regression target |
| `co2` | float64 (ppm) | `python main.py --discover-cohort` (ERS-style filter) |

Extra columns in campus exports are ignored unless you extend the pipeline.

---

## Using your own real dataset

1. Copy or mount your telemetry so each device has uplinks in `application.csv` files under one root (default `node_analysis/idata/`, or set `NODE_ANALYSIS_DATA_ROOT`).

2. Put **your** device identifiers in the `deveui` column (typical: dashed LoRaWAN EUIs from your network server).

3. Write `pipeline/cohorts/my_devices.csv` with a single column `deveui` listing the devices to include.

4. Run training with a **new** run id (see root `README.md`):

```powershell
$env:NODE_ANALYSIS_DATA_ROOT = "E:\path\to\your\idata"   # if not default
python main.py --run-id my_real_run_2026 --cohort-csv pipeline\cohorts\my_devices.csv --epochs 12
```

Outputs appear under `outputs/my_real_run_2026/` only. Do not commit real `idata` to a public repository unless your data-protection review allows it.

---

## Environment

```powershell
$env:NODE_ANALYSIS_DATA_ROOT = "E:\path\to\node_analysis\idata"
```

If unset, `main.py` uses `node_analysis/idata` when that directory exists.
