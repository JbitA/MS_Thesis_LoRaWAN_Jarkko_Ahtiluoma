"""
FILE STORY — ``thesis_figure_keys.py``
======================================

**Role.** Allowlist of 11 figure folder keys; legacy ``fig_*`` migration; PNG path helpers.

**Connects.** ``figure_titles.py`` (basenames), ``build.py``, ``run_figures.py``.

**Developed with Cursor AI.**
"""

from __future__ import annotations



import shutil  # Directory merge, rename, and rmtree during migration.

from pathlib import Path  # Graph root and per-figure output paths.

from typing import Iterator  # ``iter_thesis_figure_dirs`` yield type.



from figure_titles import (  # Canonical PNG/CSV basenames derived from display titles.

    FIGURE_MAIN_TITLE,

    all_companion_table_png_basenames,

    companion_artifact_names,

    figure_png_names_for_key,

)



# ---------------------------------------------------------------------------

# Allowlisted thesis figures (folder key = primary PNG stem)

# ---------------------------------------------------------------------------

THESIS_FIGURE_KEYS: tuple[str, ...] = (

    "battery_missingness",

    "confusion_grid",

    "imputed_panel",

    "imputed_vs_non_imputed_halt",

    "threshold_f2_impute_sensitivity_comparison",

    "policy_map",

    "pr_f2_only",

    "method_model_ranking",

    "regression_4_models",

    "stl_vs_mtl_delta",

    "deep_learning_time_series_integration",

)



# Deprecated alias (historical name: "slug").

THESIS_FIGURE_SLUGS: tuple[str, ...] = THESIS_FIGURE_KEYS  # Same tuple; old import name.



THESIS_FIGURE_PNG_NAMES: dict[str, tuple[str, ...]] = {

    key: figure_png_names_for_key(key) for key in THESIS_FIGURE_KEYS  # Primary + extra PNGs per key.

}



# ---------------------------------------------------------------------------

# Legacy folder / filename maps (migration only; thesis PNG names unchanged)

# ---------------------------------------------------------------------------

# Old ``fig_*`` directory names → current figure keys (for migration only).

LEGACY_FIGURE_DIR_TO_KEY: dict[str, str] = {

    "fig_a_only_battery_missingness": "battery_missingness",

    "fig_confusion_grid_consolidated": "confusion_grid",

    "fig_dl_combined_imputed_panel": "imputed_panel",

    "fig_dl_halt_imputed_nonimputed": "imputed_vs_non_imputed_halt",

    "fig_dl_individual_sensor_f2": "threshold_f2_impute_sensitivity_comparison",

    "fig_graph03_policy_map": "policy_map",

    "fig_graph03_pr_f2_only": "pr_f2_only",

    "fig_methodb_tier_confusion": "method_model_ranking",

    "method_b_tiers": "method_model_ranking",

    "fig_regression_pred_vs_true": "regression_4_models",

    "fig_single_vs_multitask_delta": "stl_vs_mtl_delta",

    "fig_thesis01_vC_ribbon": "deep_learning_time_series_integration",

}



# Reverse map: accept legacy key strings in CLI/scripts during transition.

LEGACY_KEY_ALIASES: dict[str, str] = {**LEGACY_FIGURE_DIR_TO_KEY, **{v: v for v in THESIS_FIGURE_KEYS}}



# Former pipeline figures — removed; delete directories if found.

EXCLUDED_GRAPH_KEYS: frozenset[str] = frozenset(

    {

        "fig_calibration_evaluations",

        "fig_graph03_pr_latest_run",

        "fig_survival_horizon_brier",

        "calibration_evaluations",

        "graph03_pr_latest_run",

        "survival_horizon_brier",

    }

)



EXCLUDED_GRAPH_SLUGS: frozenset[str] = EXCLUDED_GRAPH_KEYS  # Deprecated alias for excluded keys.



GENERIC_LEGACY_COMPANION: tuple[str, ...] = (  # Pre-title-based companion basenames to delete when superseded.

    "table_metrics_clean.csv",

    "table_metrics_clean.md",

    "table_metrics_sheet.png",

    "table_metrics_mtl_stl_clean.csv",

    "table_metrics_mtl_stl_sheet.png",

    "tier2_methodB_utility_ranking_table.png",

    "tier2_methodB_model_selection_recreated.csv",

)



# Legacy PNG filenames that may exist before title-based rename.

LEGACY_RECREATED_PNG: dict[str, tuple[str, ...]] = {

    "battery_missingness": (

        "recreated_fig_a_only_battery_missingness.png",

        "time_series_missing_measurements.png",

    ),

    "confusion_grid": (

        "recreated_fig_confusion_grid_consolidated.png",

        "confusion_grid.png",

        "halt_classification_confusion_matrices.png",

    ),

    "imputed_panel": (

        "recreated_fig_dl_combined_imputed_panel.png",

        "imputed_panel.png",

        "battery_voltage_imputed_and_non_imputed_preprocessing.png",

        "imputed_preprocessing_and_non_imputed_preprocessing_battery_voltage_v.png",

    ),

    "imputed_vs_non_imputed_halt": (

        "recreated_fig_dl_halt_imputed_nonimputed.png",

        "imputed_vs_non_imputed_halt.png",

        "halt_probability_imputed_and_non_imputed_preprocessing.png",

        "imputed_preprocessing_and_non_imputed_preprocessing_halt_probability.png",

    ),

    "threshold_f2_impute_sensitivity_comparison": (

        "recreated_fig_dl_individual_sensor_f2.png",

        "per_sensor_f2.png",

        "halt_prediction_accuracy_vs_precision_at_f2_optimal_thresholds_single_sensor.png",

    ),

    "policy_map": (

        "recreated_fig_graph03_policy_map.png",

        "policy_map.png",

        "precision_recall_f_beta_sensitivity_optimization.png",

    ),

    "pr_f2_only": (

        "recreated_fig_graph03_pr_f2_only.png",

        "pr_f2_only.png",

        "precision_recall_f2_with_probability_cutoff.png",

    ),

    "method_model_ranking": (

        "recreated_fig_methodb_tier_confusion.png",

        "method_b_tiers.png",

        "method_model_ranking.png",

        "halt_classification_f2_model_ranking.png",

    ),

    "regression_4_models": (

        "recreated_fig_regression_pred_vs_true.png",

        "recreated_fig_regression_pred_vs_true_lstm.png",

        "regression_4_models.png",

        "regression_lstm_only.png",

        "regression_predicted_vs_true_values.png",

        "lstm_regression_predicted_and_true_values.png",

    ),

    "stl_vs_mtl_delta": (

        "recreated_fig_single_vs_multitask_delta.png",

        "stl_vs_mtl_delta.png",

        "deep_learning_mtl_and_stl_performance_deltas.png",

    ),

    "deep_learning_time_series_integration": (

        "recreated_fig_thesis01_vC_ribbon.png",

        "thesis_ribbon.png",

        "task_unified_presentation.png",

        "deep_learning_time_series_integration.png",

    ),

}





# ---------------------------------------------------------------------------

# Key normalization

# ---------------------------------------------------------------------------

def normalize_figure_key(name: str) -> str:

    """Map legacy ``fig_*`` or current key to a canonical ``THESIS_FIGURE_KEYS`` entry."""

    key = LEGACY_KEY_ALIASES.get(name.strip(), name.strip())  # Resolve alias or strip input.

    if key not in THESIS_FIGURE_KEYS:  # Reject unknown keys outside allowlist.

        raise KeyError(f"Unknown figure key {name!r}; allowlist: {THESIS_FIGURE_KEYS}")

    return key  # Canonical figure key string.





# ---------------------------------------------------------------------------

# Directory migration (fig_* folders → PNG-aligned keys)

# ---------------------------------------------------------------------------

def migrate_figure_directories(graph_root: Path) -> list[tuple[str, str]]:

    """Rename legacy ``fig_*`` graph folders to PNG-aligned figure keys."""

    renames: list[tuple[str, str]] = []  # Log of (old_name, new_name) renames.

    if not graph_root.is_dir():  # Nothing to migrate if graph root missing.

        return renames

    for old_name, new_name in LEGACY_FIGURE_DIR_TO_KEY.items():  # Each legacy folder mapping.

        old_p = graph_root / old_name  # Legacy directory path.

        new_p = graph_root / new_name  # Target canonical key directory.

        if not old_p.is_dir():  # Skip if legacy folder absent.

            continue

        if new_p.is_dir():  # Target exists: merge outputs then remove legacy.

            # Merge outputs/ from legacy into new if new exists but is empty-ish

            old_out = old_p / "outputs"  # Legacy figure artifacts folder.

            new_out = new_p / "outputs"  # Canonical outputs folder.

            if old_out.is_dir():  # Only merge when legacy has outputs.

                new_out.mkdir(parents=True, exist_ok=True)  # Ensure destination exists.

                for f in old_out.iterdir():  # Each file in legacy outputs.

                    dest = new_out / f.name  # Same basename under new folder.

                    if not dest.exists():  # Do not overwrite existing canonical files.

                        shutil.move(str(f), str(dest))  # Move file into canonical tree.

            shutil.rmtree(old_p, ignore_errors=True)  # Remove legacy folder after merge.

            renames.append((old_name, f"{new_name} (merged)"))  # Record merge event.

        else:

            old_p.rename(new_p)  # Simple rename when target absent.

            renames.append((old_name, new_name))  # Record rename event.

    return renames  # Migration log for callers.





# ---------------------------------------------------------------------------

# Figure PNG migration (legacy basenames → title-derived names)

# ---------------------------------------------------------------------------

def _figure_png_sources(out_dir: Path, figure_key: str) -> list[Path]:

    """Legacy / orphan figure PNGs under ``out_dir`` (not companion tables)."""

    targets = set(THESIS_FIGURE_PNG_NAMES.get(figure_key, ()))  # Canonical PNG names to keep.

    table_pngs = all_companion_table_png_basenames()  # Companion sheets excluded from figure sources.

    sources: list[Path] = []  # Candidate legacy paths in priority order.

    seen: set[str] = set()  # Deduplicate by basename.



    def _add(path: Path) -> None:

        if not path.is_file() or path.name in seen or path.name in targets:  # Skip invalid/duplicate/canonical.

            return

        if path.name in table_pngs or path.name.startswith("table_metrics"):  # Skip companion table PNGs.

            return

        seen.add(path.name)  # Mark basename consumed.

        sources.append(path)  # Queue as rename source.



    for name in LEGACY_RECREATED_PNG.get(figure_key, ()):  # Known legacy names first (priority).

        _add(out_dir / name)

    for path in sorted(out_dir.glob("recreated*.png")):  # Any other recreated_* PNGs.

        _add(path)

    for path in sorted(out_dir.glob("*.png")):  # Residual PNGs last resort.

        _add(path)

    return sources  # Ordered candidate sources for migrate_figure_pngs.





def migrate_figure_pngs(out_dir: Path, figure_key: str) -> list[tuple[str, str]]:

    """Rename legacy figure PNGs to canonical thesis title basenames."""

    targets = THESIS_FIGURE_PNG_NAMES.get(figure_key, ())  # Expected canonical filenames.

    if not targets or not out_dir.is_dir():  # No-op when unconfigured or missing dir.

        return []

    sources = _figure_png_sources(out_dir, figure_key)  # Legacy/orphan PNG paths.

    renames: list[tuple[str, str]] = []  # (old_basename, new_basename) log.

    for i, target in enumerate(targets):  # One source per target variant index.

        dest = out_dir / target  # Destination canonical path.

        if dest.is_file():  # Already canonical; skip.

            continue

        if i >= len(sources):  # No more legacy sources for extra variants.

            break

        src = sources[i]  # i-th legacy file maps to i-th canonical name.

        if src.name == target:  # Already correct basename.

            continue

        src.rename(dest)  # Rename in place to thesis basename.

        renames.append((src.name, target))  # Log rename for reporting.

    return renames





# ---------------------------------------------------------------------------

# Companion artifact migration (generic names → title-derived names)

# ---------------------------------------------------------------------------

def _companion_rename_pairs(figure_key: str) -> list[tuple[str, str]]:

    from figure_titles import (  # noqa: WPS433 — local import avoids circular import at module load.

        companion_absolute_metrics_csv,

        companion_absolute_metrics_sheet_png,

        companion_extra_table_csv,

        companion_extra_table_png,

        companion_metrics_csv,

        companion_metrics_md,

        companion_metrics_sheet_png,

    )



    pairs: list[tuple[str, str]] = [  # (legacy_basename, canonical_basename) for standard companions.

        ("table_metrics_clean.csv", companion_metrics_csv(figure_key)),

        ("table_metrics_clean.md", companion_metrics_md(figure_key)),

        ("table_metrics_sheet.png", companion_metrics_sheet_png(figure_key)),

    ]

    if figure_key == "stl_vs_mtl_delta":  # Extra absolute MTL/STL companions for this figure only.

        pairs.extend(

            [

                ("table_metrics_mtl_stl_clean.csv", companion_absolute_metrics_csv(figure_key)),

                ("table_metrics_mtl_stl_sheet.png", companion_absolute_metrics_sheet_png(figure_key)),

            ]

        )

    if figure_key == "method_model_ranking":  # Tier-2 utility table companions.

        pairs.extend(

            [

                ("tier2_methodB_utility_ranking_table.png", companion_extra_table_png(figure_key, 1)),

                ("tier2_methodB_model_selection_recreated.csv", companion_extra_table_csv(figure_key, 1)),

            ]

        )

    return pairs  # Full rename list for this figure key.





def migrate_companion_artifacts(out_dir: Path, figure_key: str) -> list[tuple[str, str]]:

    if not out_dir.is_dir():  # No outputs folder to migrate.

        return []

    renames: list[tuple[str, str]] = []  # Log of companion renames.

    for src_name, dst_name in _companion_rename_pairs(figure_key):  # Each legacy→canonical pair.

        if src_name == dst_name:  # Already canonical basename.

            continue

        src = out_dir / src_name  # Legacy companion path.

        dst = out_dir / dst_name  # Canonical companion path.

        if src.is_file() and not dst.is_file():  # Rename only when dest absent.

            src.rename(dst)  # In-place rename to thesis basename.

            renames.append((src_name, dst_name))  # Log event.

    return renames





# ---------------------------------------------------------------------------

# Cleanup: excluded dirs and superseded legacy files

# ---------------------------------------------------------------------------

def remove_excluded_graph_dirs(graph_root: Path) -> list[str]:

    """Delete non-allowlisted graph directories (legacy ``fig_*`` and excluded keys)."""

    removed: list[str] = []  # Names of deleted directories.

    if not graph_root.is_dir():  # Nothing to clean.

        return removed

    allow = set(THESIS_FIGURE_KEYS)  # Fast membership for allowlisted keys.

    for folder in sorted(graph_root.iterdir()):  # Each child of graph root.

        if not folder.is_dir():  # Only consider directories.

            continue

        name = folder.name  # Folder basename.

        if name in allow:  # Keep thesis figure folders.

            continue

        if name in EXCLUDED_GRAPH_KEYS:  # Explicitly excluded legacy figures.

            shutil.rmtree(folder, ignore_errors=True)  # Delete excluded folder.

            removed.append(name)  # Log removal.

            continue

        if name.startswith("fig_"):  # Any remaining fig_* not in LEGACY map.

            shutil.rmtree(folder, ignore_errors=True)  # Delete orphan legacy folder.

            removed.append(name)  # Log removal.

    return removed





def remove_legacy_figure_pngs(out_dir: Path, figure_key: str) -> None:

    current = set(THESIS_FIGURE_PNG_NAMES.get(figure_key, ()))  # Canonical figure PNG basenames.

    for name in LEGACY_RECREATED_PNG.get(figure_key, ()):  # Each known legacy basename.

        if name in current:  # Never delete a name that is still canonical.

            continue

        path = out_dir / name  # Legacy file path.

        if path.is_file():  # Delete only if present.

            path.unlink()  # Remove superseded figure PNG.





def remove_legacy_companion_artifacts(out_dir: Path, figure_key: str) -> None:

    current = set(companion_artifact_names(figure_key))  # Canonical companion basenames for key.

    for name in GENERIC_LEGACY_COMPANION:  # Generic pre-migration companion names.

        if name in current:  # Never delete if still the canonical name.

            continue

        path = out_dir / name  # Legacy companion path.

        if path.is_file():  # Delete only if present.

            path.unlink()  # Remove superseded companion file.





# ---------------------------------------------------------------------------

# Orchestrated migration + path resolution API

# ---------------------------------------------------------------------------

def migrate_graph_outputs(graph_root: Path) -> dict[str, list[tuple[str, str]]]:

    """Rename folders, figure PNGs, and companion files to canonical thesis names."""

    migrate_figure_directories(graph_root)  # fig_* → figure_key folders.

    remove_excluded_graph_dirs(graph_root)  # Drop excluded/orphan fig_* dirs.

    log: dict[str, list[tuple[str, str]]] = {}  # Per-key rename events.

    for folder in iter_thesis_figure_dirs(graph_root):  # Each allowlisted figure folder.

        figure_key = folder.name  # Folder name equals figure key.

        out = folder / "outputs"  # Per-figure artifact directory.

        if not out.is_dir():  # Skip keys with no outputs yet.

            continue

        events: list[tuple[str, str]] = []  # Renames for this key.

        events.extend(migrate_figure_pngs(out, figure_key))  # Figure PNG renames.

        events.extend(migrate_companion_artifacts(out, figure_key))  # Companion renames.

        remove_legacy_figure_pngs(out, figure_key)  # Delete leftover legacy figure PNGs.

        remove_legacy_companion_artifacts(out, figure_key)  # Delete leftover legacy companions.

        if events:  # Record only keys with changes.

            log[figure_key] = events

    return log  # Full migration log keyed by figure_key.





def figure_png_path(out_dir: Path, figure_key: str, *, variant: int = 0) -> Path:

    names = THESIS_FIGURE_PNG_NAMES.get(figure_key)  # Tuple of canonical PNG basenames.

    if not names:  # Unknown figure key.

        raise KeyError(f"No PNG name mapping for figure key {figure_key!r}")

    if variant < 0 or variant >= len(names):  # Bounds-check variant index.

        raise IndexError(f"variant {variant} out of range for {figure_key!r} ({len(names)} name(s))")

    return out_dir / names[variant]  # Full path to primary or extra figure PNG.





def list_figure_pngs(out_dir: Path, figure_key: str) -> list[Path]:

    found: list[Path] = []  # Canonical figure PNG paths that exist.

    for name in THESIS_FIGURE_PNG_NAMES.get(figure_key, ()):  # Each expected basename.

        p = out_dir / name

        if p.is_file():  # Collect existing canonical files.

            found.append(p)

    if found:  # Prefer canonical names when present.

        return found

    legacy = sorted(out_dir.glob("recreated*.png"))  # Fallback: any recreated_* PNGs.

    if legacy:

        return legacy

    table_pngs = all_companion_table_png_basenames()  # Exclude companion sheets from fallback.

    return sorted(

        p for p in out_dir.glob("*.png") if p.name not in table_pngs and not p.name.startswith("table_metrics")

    )  # Last resort: non-table PNGs only.





def iter_thesis_figure_dirs(fig_root: Path) -> Iterator[Path]:

    """Yield ``<fig_root>/<figure_key>`` for each allowlisted key (mkdir if missing)."""

    for figure_key in THESIS_FIGURE_KEYS:  # Stable iteration order.

        folder = fig_root / figure_key  # Per-figure directory.

        folder.mkdir(parents=True, exist_ok=True)  # Ensure folder exists for builders.

        yield folder  # Path to ``graphs/<figure_key>/``.





def is_thesis_figure_key(name: str) -> bool:

    try:

        normalize_figure_key(name)  # Validates against allowlist + aliases.

        return True  # Accepted key.

    except KeyError:

        return False  # Unknown or legacy-only name.





def is_thesis_figure_slug(name: str) -> bool:

    """Deprecated alias for :func:`is_thesis_figure_key`."""

    return is_thesis_figure_key(name)  # Delegate to canonical predicate.





def main_title(figure_key: str) -> str:

    return FIGURE_MAIN_TITLE[normalize_figure_key(figure_key)]  # Human-readable title for display/PNG derivation.


