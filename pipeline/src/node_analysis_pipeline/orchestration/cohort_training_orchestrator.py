#!/usr/bin/env python3
"""
FILE STORY — ``cohort_training_orchestrator.py``
================================================

**Role.** Runs the full **training** stack for one DevEUI cohort: five passes in
sequence, stem migration, and a markdown audit report. Invoked as a subprocess
from ``main.py`` (which handles figures separately when using the default flow).

**Training passes (in order).**

  1. Multitask binary (imputed features) — primary tables for most figures
  2. Single-task battery (STL regression baseline)
  3. Single-task halt focus (STL classification baseline)
  4. Survival halt (optional; ``--skip-survival``)
  5. Multitask binary **no impute** — via ``train_multitask_no_impute.py`` in-process patch

**Outputs.** ``outputs/<run_id>/models/tables/node_analysis_<run_id>_*``;
``models/reports/FULL_STACK_REPORT_<run_id>.md``; optional checkpoints.

**Figures.** Unless ``--skip-figures``, calls ``pipeline/figures/build.py`` at end
(``main.py`` normally passes ``--skip-figures`` and runs figures itself afterward).

**Developed with Cursor AI.**
"""

from __future__ import annotations  # Typing for optional paths and step tuples

import argparse  # CLI: epochs, run_id, resume-only, skip-figures, skip-survival
import os  # Environment for no-impute subprocess and figure phase
import subprocess  # Spawn train_multitask_no_impute and figures/build.py
import sys  # sys.path for src imports; sys.executable for children
import traceback  # Print stack trace on unhandled failure before re-raise
from datetime import datetime, timezone  # UTC timestamps in markdown report
from pathlib import Path  # Cohort CSV, checkpoints, report paths
from typing import List, Optional, Tuple  # Step lists, optional checkpoint dirs

_ROOT = Path(__file__).resolve().parents[2]  # .../pipeline/src
if str(_ROOT) not in sys.path:  # Allow imports when executed as script from pipeline/
    sys.path.insert(0, str(_ROOT))

from node_analysis_pipeline.artifact_stems import (  # noqa: E402
    artifact_stem,
    migrate_table_stems,
    survival_resolved_stem,
)
from node_analysis_pipeline.layout import DEFAULT_RUN_ID, run_checkpoints_dir, run_reports_dir  # noqa: E402
from node_analysis_pipeline.paths import _BUNDLE_ROOT, _THESIS_PIPELINE, default_tables_dir  # noqa: E402
from node_analysis_pipeline.training.trainer_subprocess import TrainerInvocation, invoke_trainer  # noqa: E402

_MARKER_DONE = "FULL_STACK_ORCHESTRATOR_COMPLETE"  # Sentinel printed and embedded in report


def _utc_id() -> str:  # Generate timestamp run_id when CLI omits --run-id
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")  # Fallback run_id when none provided


def _resolve_cohort_csv(cli_value: str) -> Path:
    """Cohort allowlist: explicit ``--cohort-csv``, else discover output if present."""
    if str(cli_value).strip():
        return Path(cli_value.strip())
    discovered = _BUNDLE_ROOT / "cohorts" / "ers_co2_max_available_deveui.csv"
    if discovered.is_file():
        return discovered
    raise SystemExit(
        "No cohort CSV: pass --cohort-csv, run main.py --discover-cohort first, or use --dummy-smoke "
        "(uses pipeline/cohorts/dummy_smoke_deveui.csv)."
    )


def _cohort_device_count(cohort_csv: Path) -> int:  # Count unique non-empty DevEUIs in cohort
    import pandas as pd  # Lazy import — orchestrator starts fast without pandas

    df = pd.read_csv(cohort_csv)  # Load allowlist
    col = "deveui" if "deveui" in df.columns else df.columns[0]  # Standard or first column
    s = df[col].astype(str).str.strip()  # Normalize DevEUI strings
    return int(s[(s != "") & (s.str.lower() != "nan")].nunique())  # Count non-empty unique devices


def _shared_binary_args(  # Build common CLI flags for imputed trainer passes
    *,
    epochs: int,
    max_sensors: int,
    halt_loss_weight: float,
    save_checkpoint_dir: Optional[Path],
    cohort_csv: Path,
) -> List[str]:
    out: List[str] = [  # CLI fragment shared by MTL/STL/survival subprocess passes
        "--epochs",
        str(epochs),
        "--max-sensors",
        str(max_sensors),
        "--split-mode",
        "sensor_holdout",
        "--halt-bce-mix",
        "0.25",
        "--early-stopping-patience",
        "2",
        "--halt-loss-weight",
        str(halt_loss_weight),
        "--deveui-cohort-csv",
        str(cohort_csv),
    ]
    if save_checkpoint_dir is not None:  # Optional per-pass checkpoint directory
        out += ["--save-checkpoint-dir", str(save_checkpoint_dir)]
    return out


def _run_train(  # Invoke vendored trainer subprocess for one imputed pass
    *,
    pass_name: str,
    halt_mode: str,
    artifact_stem: str,
    extra: List[str],
    log: List[str],
) -> int:
    log.append(f"\n## Train step: `{pass_name}` → stem `{artifact_stem}`\n```text\n{halt_mode}\n```\n")
    inv = TrainerInvocation(  # Package argv for vendored trainer subprocess
        pass_name=pass_name,
        halt_mode=halt_mode,
        artifact_stem=artifact_stem,
        extra_args=extra,
        dry_run=False,
        stream_output=True,
    )
    rc = invoke_trainer(inv)  # Run bundle; return code 0 = success
    log.append(f"- **return code:** {rc}\n")
    return rc


def _run_train_noimpute(  # Spawn no-impute script as separate subprocess
    *,
    pass_name: str,
    halt_mode: str,
    artifact_stem: str,
    epochs: int,
    max_sensors: int,
    halt_loss_weight: float,
    save_checkpoint_dir: Optional[Path],
    cohort_csv: Path,
    log: List[str],
) -> int:
    script = _ROOT / "node_analysis_pipeline" / "training" / "train_multitask_no_impute.py"
    cmd = [  # Separate subprocess — monkey-patches telemetry before in-process trainer
        sys.executable,
        str(script),
        "--artifact-stem",
        artifact_stem,
        "--halt-mode",
        halt_mode,
        "--epochs",
        str(int(epochs)),
        "--max-sensors",
        str(int(max_sensors)),
        "--halt-loss-weight",
        str(float(halt_loss_weight)),
        "--deveui-cohort-csv",
        str(cohort_csv),
    ]
    if save_checkpoint_dir is not None:
        cmd += ["--save-checkpoint-dir", str(save_checkpoint_dir)]
    env = {**os.environ, "PYTHONPATH": str(_ROOT)}  # Propagate run env + src on PYTHONPATH
    log.append(f"\n## Train step: `{pass_name}` → stem `{artifact_stem}` (no-impute)\n")
    proc = subprocess.run(cmd, cwd=str(_ROOT), env=env)  # cwd=src so relative imports resolve
    log.append(f"- **return code:** {int(proc.returncode)}\n")
    return int(proc.returncode)


def _figures_phase(log: List[str]) -> None:  # Run pipeline/figures/build.py for thesis PNGs
    import sys  # Local import avoids shadowing module-level sys in type checkers

    fig_root = _BUNDLE_ROOT / "figures"  # pipeline/figures — hosts build.py and thesis_figure_keys
    if str(fig_root) not in sys.path:
        sys.path.insert(0, str(fig_root))  # Import thesis_figure_keys from figures package
    from thesis_figure_keys import THESIS_FIGURE_KEYS  # noqa: WPS433  # Allowlist of 11 thesis figures

    log.append("\n## Figures (`outputs/<run_id>/graphs/<figure_key>/outputs`)\n")
    log.append(
        f"- **Allowlist ({len(THESIS_FIGURE_KEYS)} keys):** " + ", ".join(f"`{k}`" for k in THESIS_FIGURE_KEYS) + "\n"
    )
    env = {**os.environ, "PYTHONPATH": str(_ROOT)}  # Figure scripts may import node_analysis_pipeline
    env.setdefault("HOL_ACAD_PIPELINE_ROOT", str(_BUNDLE_ROOT))  # Legacy figure code path
    env.setdefault("HOL_ACAD_TERTIARY_ROOT", str(_BUNDLE_ROOT))
    env.setdefault("HOL_ACAD_OUTPUT_ROOT", os.environ.get("HOL_ACAD_OUTPUT_ROOT", str(_BUNDLE_ROOT / "outputs")))
    env.setdefault("HOL_ACAD_RUN_ID", os.environ.get("HOL_ACAD_RUN_ID", DEFAULT_RUN_ID))
    gr = os.environ.get("HOL_ACAD_GRAPH_ROOT", "").strip()  # Use graph root from training phase if set
    if gr:
        env.setdefault("HOL_ACAD_FIG_SCAN_ROOT", gr)  # Scan trained-run graphs folder for PNGs
    build_py = fig_root / "build.py"  # Batch builder for all thesis figures
    log.append(f"\n### `figures/build.py`\n")
    try:
        subprocess.run(
            [sys.executable, str(build_py)],
            cwd=str(fig_root),  # Figures expect cwd = pipeline/figures
            env=env,
            check=True,  # Raise on non-zero exit — caught below for report
        )
        log.append("- **status:** ok\n")
    except subprocess.CalledProcessError as e:
        log.append(f"- **status:** failed (exit {e.returncode})\n")


def main() -> None:  # Full stack entry: train passes → migrate → figures → report
    p = argparse.ArgumentParser(description="Run full bundle stack: train → figures.")
    p.add_argument("--epochs", type=int, default=12, help="Training epochs per phase (AdamW + early stopping).")
    p.add_argument("--max-sensors", type=int, default=325, help="Cap when cohort CSV absent; ignored if allowlist is set.")
    p.add_argument(
        "--run-id",
        type=str,
        default=DEFAULT_RUN_ID,
        help="Run id → node_analysis_<run_id>_* table prefixes.",
    )
    p.add_argument(
        "--cohort-csv",
        type=str,
        default="",
        help="DevEUI allowlist CSV (required unless discover-cohort output exists; dummy-smoke uses dummy_smoke_deveui.csv).",
    )
    p.add_argument(
        "--resume-only",
        action="store_true",
        help="Skip all training; rebuild FULL_STACK_REPORT and figures for an existing run_id.",
    )
    p.add_argument("--skip-figures", action="store_true", help="Skip figures phase (e.g. after they already succeeded).")
    p.add_argument(
        "--skip-survival",
        action="store_true",
        help="Omit survival training pass (faster; not used by thesis figure allowlist).",
    )
    args = p.parse_args()  # Consume CLI after all flags registered

    run_id = (args.run_id or _utc_id()).strip().replace(":", "_")  # Filesystem-safe run identifier
    cohort_csv = _resolve_cohort_csv(args.cohort_csv)
    if not cohort_csv.is_file():  # Cohort required for reproducible thesis device set
        raise SystemExit(f"Cohort CSV not found: {cohort_csv}")
    n_cohort = _cohort_device_count(cohort_csv)  # Validate non-empty allowlist
    if n_cohort == 0:
        raise SystemExit(
            f"Cohort file is empty: {cohort_csv}\n"
            "Populate DevEUIs (one per line) or run: python main.py --discover-cohort\n"
            "(requires HOL_ACAD_DATA_ROOT pointing at idata with application.csv files)."
        )

    tables = default_tables_dir()  # outputs/<run_id>/models/tables after apply_run_env
    ck_root = run_checkpoints_dir(run_id) / f"training_{run_id}"  # Per-run checkpoint subtree
    ck_root.mkdir(parents=True, exist_ok=True)  # Create before trainers write .pt files

    log: List[str] = []  # Markdown report sections accumulated in memory
    failures: List[str] = []  # Pass names with non-zero return codes

    log.append(f"# Full academic stack report\n\n**run_id:** `{run_id}`\n")
    log.append(f"**started (UTC):** {datetime.now(timezone.utc).isoformat()}\n")
    if args.resume_only:
        log.append("\n> **Mode:** `--resume-only` (training subprocesses were not started in this invocation).\n")
    log.append("## What was ordered (recap)\n\n")
    log.append(
        "- **Thesis bundle:** dual preprocessing spec, calibration helpers, classification/regression metrics.\n"
    )
    log.append("- **Optimizer:** AdamW (fixed in bundle trainer). **Calibration:** temperature scaling on halt logits.\n")
    log.append(
        "- **Graphs:** one folder per figure key under `outputs/<run_id>/graphs/`; STL vs MTL is compared only as "
        "**metric deltas** in `stl_vs_mtl_delta` (and companion tables), not via formal multitask-gain formulas.\n"
    )
    log.append(
        "- **Statistics:** model comparison CSVs and optional checkpoints `.pt` under `outputs/<run_id>/models/`.\n\n"
    )

    log.append("## Training configuration (this run)\n\n")
    log.append(f"- epochs={args.epochs}, max_sensors={args.max_sensors}, split=sensor_holdout, halt_bce_mix=0.25\n")
    log.append(f"- cohort_csv: `{cohort_csv}`\n")
    log.append(f"- checkpoints root: `{ck_root}`\n")

    e = int(args.epochs)  # Epoch count reused across passes
    ms = int(args.max_sensors)  # Sensor cap reused across passes
    if not args.resume_only and e < 1:  # Training would be meaningless with zero epochs
        raise SystemExit("--epochs must be >= 1")

    stem_mtl = artifact_stem(run_id, "multitask_binary")  # Primary thesis figure tables
    stem_sb = artifact_stem(run_id, "single_task_battery")  # STL battery baseline
    stem_sh = artifact_stem(run_id, "single_task_halt_focus")  # STL halt-heavy baseline
    stem_sv = artifact_stem(run_id, "survival")  # Survival pass (trainer adds _halt_survival)
    stem_noimp_mtl = artifact_stem(run_id, "multitask_binary_no_impute")  # No-impute comparison pass

    steps: List[Tuple[str, str, str, List[str]]] = [  # (pass_name, halt_mode, stem, extra_argv)
        (
            "mtl_binary",
            "binary",
            stem_mtl,
            _shared_binary_args(
                epochs=e,
                max_sensors=ms,
                halt_loss_weight=4.0,
                save_checkpoint_dir=ck_root / "mtl_binary",
                cohort_csv=cohort_csv,
            ),
        ),
        (
            "stl_battery",
            "binary",
            stem_sb,
            _shared_binary_args(
                epochs=e,
                max_sensors=ms,
                halt_loss_weight=0.0,
                save_checkpoint_dir=ck_root / "stl_battery",
                cohort_csv=cohort_csv,
            ),
        ),
        (
            "stl_halt_heavy",
            "binary",
            stem_sh,
            _shared_binary_args(
                epochs=e,
                max_sensors=ms,
                halt_loss_weight=100.0,
                save_checkpoint_dir=ck_root / "stl_halt_heavy",
                cohort_csv=cohort_csv,
            ),
        ),
        (
            "survival",
            "survival",
            stem_sv,
            _shared_binary_args(
                epochs=e,
                max_sensors=ms,
                halt_loss_weight=4.0,
                save_checkpoint_dir=ck_root / "survival",
                cohort_csv=cohort_csv,
            ),
        ),
    ]
    if args.skip_survival:  # Optional faster run without survival tables
        steps = [s for s in steps if s[0] != "survival"]
        log.append("\n> **Note:** survival pass skipped (`--skip-survival`).\n")

    log.append("\n## Training phases\n")
    if not args.resume_only:  # Full train: invoke each subprocess pass
        for pass_name, halt_mode, stem, extra in steps:
            rc = _run_train(pass_name=pass_name, halt_mode=halt_mode, artifact_stem=stem, extra=extra, log=log)
            if rc != 0:
                failures.append(f"{pass_name} (rc={rc})")
        rc = _run_train_noimpute(
            pass_name="noimp_mtl_binary",
            halt_mode="binary",
            artifact_stem=stem_noimp_mtl,
            epochs=e,
            max_sensors=ms,
            halt_loss_weight=4.0,
            save_checkpoint_dir=ck_root / "noimp_mtl_binary",
            cohort_csv=cohort_csv,
            log=log,
        )
        if rc != 0:
            failures.append(f"noimp_mtl_binary (rc={rc})")
    else:
        log.append(
            "\nAll training steps **skipped**; expecting comparison CSVs and checkpoints from a prior full run "
            f"for `run_id={run_id}`.\n"
        )

    # Survival trainer appends ``_halt_survival`` to the artifact stem.
    stem_sv_resolved = survival_resolved_stem(run_id)  # On-disk survival comparison CSV prefix
    migrate_table_stems(tables, run_id)  # Rename legacy hol_acad_fs_* if present

    log.append("\n## Post-training artifact paths (key)\n\n")
    for label, stem in [
        ("MTL binary (primary figures)", stem_mtl),
        ("STL battery", stem_sb),
        ("STL halt-heavy", stem_sh),
        ("Survival (resolved stem)", stem_sv_resolved),
        ("No-impute MTL binary", stem_noimp_mtl),
    ]:
        cmp = tables / f"{stem}_model_comparison_battery_halt.csv"  # Key existence check per pass
        log.append(f"- **{label}:** `{cmp}` — exists={cmp.is_file()}\n")

    log.append("\n## STL vs MTL comparison\n")
    log.append(
        "- **Figures only:** `stl_vs_mtl_delta` plots MTL minus STL metric deltas from comparison CSVs "
        f"(`{stem_mtl}`, `{stem_sb}`, `{stem_sh}`).\n"
    )

    log.append("\n## Primary figure stem\n")
    log.append(
        f"- Thesis figures use multitask binary tables: `{stem_mtl}_*_{{lstm,gru,transformer,tcn}}_battery_halt_predictions.csv`.\n"
    )

    os.environ["HOL_ACAD_RUN_ID"] = run_id  # Figures phase reads run id from env
    if not args.skip_figures:
        _figures_phase(log)  # Build all thesis PNGs under graphs/
    else:
        log.append("\n## Figures (`figures/build.py`)\n\n**Skipped** (`--skip-figures`).\n")

    log.append("\n## Summary\n\n")
    if args.resume_only:
        log.append(
            "**Training:** not re-run in this invocation (`--resume-only`); comparison CSVs on disk from the earlier bundle run were used.\n\n"
        )
    elif failures:
        log.append("**Training failures:** " + ", ".join(failures) + "\n\n")
    else:
        log.append("**Training:** all subprocess phases returned code 0.\n\n")

    log.append(
        "**Elements touched:** `node_analysis/pipeline/` (orchestrator, figures, vendored trainer) and "
        "**new outputs** under `node_analysis/outputs/<run_id>/`.\n"
    )

    log.append(f"\n**Finished (UTC):** {datetime.now(timezone.utc).isoformat()}\n")
    log.append(f"\n---\n`{_MARKER_DONE}`\n")

    out_md = run_reports_dir(run_id) / f"FULL_STACK_REPORT_{run_id}.md"  # Persistent markdown audit
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("".join(log), encoding="utf-8")  # Write full report atomically
    print(f"Wrote {out_md}", flush=True)  # User-visible confirmation
    print(_MARKER_DONE, flush=True)  # Sentinel for CI/log parsers


if __name__ == "__main__":  # python -m or direct script execution
    try:
        main()
    except Exception:
        traceback.print_exc()  # Full stack to stderr before exit
        raise
