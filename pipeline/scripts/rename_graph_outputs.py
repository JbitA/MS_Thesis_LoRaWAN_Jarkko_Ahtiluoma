#!/usr/bin/env python3
"""
FILE STORY — ``rename_graph_outputs.py``
========================================

**Role.** Maintenance utility: rename/migrate graph output files to canonical thesis basenames.

**Connects.** ``thesis_figure_keys.migrate_graph_outputs``.

**Developed with Cursor AI.**
"""
from __future__ import annotations  # Annotations as strings where needed for forward refs.

import argparse  # CLI: graph root, run id, table migration flag.
import sys  # Mutate ``sys.path`` before local imports.
from pathlib import Path  # Resolve pipeline, graph root, and table directories.

PIPELINE = Path(__file__).resolve().parents[1]  # ``pipeline/`` directory containing ``figures/`` and ``src/``.
FIG = PIPELINE / "figures"  # Figure builders and ``thesis_figure_keys`` module location.
SRC = PIPELINE / "src"  # ``node_analysis_pipeline`` package root.
sys.path.insert(0, str(FIG))  # Import ``thesis_figure_keys`` from figures tree.
sys.path.insert(0, str(SRC))  # Import ``node_analysis_pipeline`` from src tree.

from node_analysis_pipeline import env_config as env  # noqa: E402  # Env var names and getters (import after path).
from node_analysis_pipeline.artifact_stems import migrate_table_stems  # noqa: E402  # Rename legacy table CSV stems.
from thesis_figure_keys import migrate_graph_outputs, remove_excluded_graph_dirs  # noqa: E402  # Graph folder/file renames.


def main() -> None:
    """Resolve graph root, optionally migrate tables, remove excluded dirs, rename outputs."""
    p = argparse.ArgumentParser(description="Migrate graph PNG/CSV/MD names and folder keys.")  # CLI parser.
    p.add_argument(  # Explicit graphs directory override.
        "--graph-root",
        type=Path,
        default=None,
        help="Path to outputs/<run_id>/graphs (default: NODE_ANALYSIS_GRAPH_ROOT).",
    )
    p.add_argument(  # Run id used when graph-root omitted.
        "--run-id",
        default=env.get_str(env.ENV_RUN_ID, ""),  # Default from current environment run id.
        help="Run id when graph-root omitted.",
    )
    p.add_argument(  # Also rename model table CSVs under ``outputs/<run_id>/models/tables``.
        "--migrate-tables",
        action="store_true",
        help="Also rename hol_acad_fs_* CSVs to node_analysis_* under models/tables.",
    )
    args = p.parse_args()  # Parse migration CLI arguments.

    if args.graph_root is not None:  # User supplied explicit graph tree path.
        graph_root = args.graph_root.resolve()  # Absolute normalized graph root.
    elif args.run_id:  # Infer graph root from run id under project outputs.
        graph_root = (PIPELINE.parent / "outputs" / args.run_id / "graphs").resolve()  # ``node_analysis/outputs/<run_id>/graphs``.
    else:  # Fall back to environment variable for graph root.
        gr = env.get_str(env.ENV_GRAPH_ROOT, "")  # Read ``NODE_ANALYSIS_GRAPH_ROOT`` (or legacy alias).
        if not gr:  # Missing both CLI run id and env graph root.
            raise SystemExit("Provide --graph-root or --run-id (or set NODE_ANALYSIS_GRAPH_ROOT).")  # Usage error.
        graph_root = Path(gr).resolve()  # Use env-provided path, absolutized.

    if not graph_root.is_dir():  # Nothing to migrate if path missing.
        raise SystemExit(f"Graph root not found: {graph_root}")  # Fail with clear message.

    if args.migrate_tables and args.run_id:  # Optional table stem migration alongside graphs.
        tables = PIPELINE.parent / "outputs" / args.run_id / "models" / "tables"  # Metric CSV directory for run.
        renamed = migrate_table_stems(tables, args.run_id)  # Returns list of (old, new) renames performed.
        if renamed:  # Only log when at least one file was renamed.
            print(f"Renamed {len(renamed)} table file(s) under {tables}", flush=True)  # Summary of table migration.

    removed = remove_excluded_graph_dirs(graph_root)  # Delete deprecated figure folders; return removed names.
    if removed:  # Log excluded folder removals.
        print(f"Removed excluded folders: {', '.join(removed)}", flush=True)  # Comma-separated folder keys.

    log = migrate_graph_outputs(graph_root)  # Per-figure-key list of (src, dst) rename events.
    if not log and not removed:  # Already canonical layout.
        print(f"No renames needed under {graph_root}", flush=True)  # Idempotent success message.
        return  # Exit early when nothing changed.

    print(f"Migrated under {graph_root}:", flush=True)  # Header for per-figure migration log.
    for figure_key, events in log.items():  # Iterate canonical figure keys.
        print(f"  {figure_key}:", flush=True)  # Subheader for one figure directory.
        for src, dst in events:  # Each file rename within that figure folder.
            print(f"    {src} -> {dst}", flush=True)  # Indented before/after path pair.


if __name__ == "__main__":  # Script entry guard.
    main()  # Run graph (and optional table) migration.
