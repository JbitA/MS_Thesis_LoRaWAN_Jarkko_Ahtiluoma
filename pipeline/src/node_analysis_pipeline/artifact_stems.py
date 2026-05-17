"""
FILE STORY — ``artifact_stems.py``
==================================

**Role.** Defines how training CSV/checkpoint **filename prefixes** are built so
orchestrator, vendored trainer, figure code, and migration utilities all agree on
disk names.

**Canonical pattern.** ``node_analysis_<run_id>_<pass>`` where ``pass`` is one of:
``multitask_binary``, ``single_task_battery``, ``single_task_halt_focus``,
``survival``, ``multitask_binary_no_impute``.

**Examples (run_id = ``node_analysis_200trial``).**

  - ``…_multitask_binary_model_comparison_battery_halt.csv``
  - ``…_multitask_binary_lstm_battery_halt_predictions.csv``
  - ``…_survival_halt_survival_transformer_survival_per_k_metrics.csv`` (survival adds suffix)

**Legacy.** ``hol_acad_fs_<run_id>_<short>`` — :func:`migrate_table_stems` renames when safe.

**Readers.** ``figures/recreated_from_new_data.py``, ``generate_figure_companion_tables.py``,
``cohort_training_orchestrator`` (via env OUTPUT_ROOT + stem helpers).

**Developed with Cursor AI.**
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Mapping

TrainingPassKind = Literal[
    "multitask_binary",
    "single_task_battery",
    "single_task_halt_focus",
    "survival",
    "multitask_binary_no_impute",
]

_LEGACY_PASS_SUFFIX: Mapping[TrainingPassKind, str] = {
    "multitask_binary": "mtl_bin",
    "single_task_battery": "stl_batt",
    "single_task_halt_focus": "stl_hhalt",
    "survival": "surv",
    "multitask_binary_no_impute": "noimp_mtl_bin",
}

_PASS_SEGMENT: Mapping[TrainingPassKind, str] = {
    "multitask_binary": "multitask_binary",
    "single_task_battery": "single_task_battery",
    "single_task_halt_focus": "single_task_halt_focus",
    "survival": "survival",
    "multitask_binary_no_impute": "multitask_binary_no_impute",
}


def artifact_stem(run_id: str, pass_kind: TrainingPassKind) -> str:
    """Canonical CSV/checkpoint prefix for a training pass."""
    rid = run_id.strip().replace(":", "_")  # Safe on Windows paths
    return f"node_analysis_{rid}_{_PASS_SEGMENT[pass_kind]}"


def legacy_artifact_stem(run_id: str, pass_kind: TrainingPassKind) -> str:
    """Historical ``hol_acad_fs_*`` prefix (migration / resume-only runs)."""
    rid = run_id.strip().replace(":", "_")
    return f"hol_acad_fs_{rid}_{_LEGACY_PASS_SUFFIX[pass_kind]}"


def survival_resolved_stem(run_id: str) -> str:
    """Stem after survival trainer appends ``_halt_survival`` to filenames."""
    return f"{artifact_stem(run_id, 'survival')}_halt_survival"


def legacy_survival_resolved_stem(run_id: str) -> str:
    return f"{legacy_artifact_stem(run_id, 'survival')}_halt_survival"


def comparison_csv_path(tables_dir: Path, run_id: str, pass_kind: TrainingPassKind) -> Path:
    """Return model comparison CSV path, preferring canonical then legacy stem."""
    name = "{stem}_model_comparison_battery_halt.csv"
    for stem in (artifact_stem(run_id, pass_kind), legacy_artifact_stem(run_id, pass_kind)):
        p = tables_dir / name.format(stem=stem)
        if p.is_file():
            return p  # Support mixed migration state on disk
    return tables_dir / name.format(stem=artifact_stem(run_id, pass_kind))


def predictions_csv_path(tables_dir: Path, run_id: str, pass_kind: TrainingPassKind, model: str) -> Path:
    """Per-model prediction export (imputed multitask by default)."""
    name = "{stem}_{model}_battery_halt_predictions.csv"
    for stem in (artifact_stem(run_id, pass_kind), legacy_artifact_stem(run_id, pass_kind)):
        p = tables_dir / name.format(stem=stem, model=model)
        if p.is_file():
            return p
    return tables_dir / name.format(stem=artifact_stem(run_id, pass_kind), model=model)


def migrate_table_stems(tables_dir: Path, run_id: str) -> list[tuple[str, str]]:
    """
    Rename ``hol_acad_fs_*`` table files to ``node_analysis_*`` when safe.

    Only renames when the destination does not already exist. Returns list of
    ``(old_name, new_name)`` pairs.
    """
    if not tables_dir.is_dir():
        return []
    renames: list[tuple[str, str]] = []
    for pass_kind in _PASS_SEGMENT:
        old_prefix = legacy_artifact_stem(run_id, pass_kind)
        new_prefix = artifact_stem(run_id, pass_kind)
        for path in sorted(tables_dir.glob(f"{old_prefix}*")):
            new_name = path.name.replace(old_prefix, new_prefix, 1)
            dest = tables_dir / new_name
            if dest.is_file():
                continue  # Never overwrite an existing canonical file
            path.rename(dest)
            renames.append((path.name, new_name))
    old_surv = legacy_survival_resolved_stem(run_id)
    new_surv = survival_resolved_stem(run_id)
    for path in sorted(tables_dir.glob(f"{old_surv}*")):
        new_name = path.name.replace(old_surv, new_surv, 1)
        dest = tables_dir / new_name
        if dest.is_file():
            continue
        path.rename(dest)
        renames.append((path.name, new_name))
    return renames
