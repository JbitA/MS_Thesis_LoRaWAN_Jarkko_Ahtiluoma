"""
FILE STORY — ``trainer_subprocess.py``
======================================

**Role.** Spawns ``multitask_battery_halt_trainer_bundle.py`` with ``--artifact-stem``; writes ``RUN_RECORD_*.json``.

**Connects.** ``paths.trainer_script()``, ``env_config.ENV_OUTPUT_ROOT``.

**Developed with Cursor AI.**
"""

from __future__ import annotations  # Modern typing for dataclass and optional types

import json  # Serialize TrainerInvocation metadata to RUN_RECORD JSON
import os  # Copy and augment environ before subprocess.run
import subprocess  # Execute vendored trainer as child process
import sys  # sys.executable and stderr printing on failure
from dataclasses import asdict, dataclass  # asdict unused but available for record dumps
from datetime import datetime, timezone  # UTC timestamps in run records
from pathlib import Path  # Paths for script, records, and bundle root
from typing import Any, Dict, List, Optional  # Record payload and optional checkpoint dir typing

from node_analysis_pipeline import env_config as env  # OUTPUT_ROOT for record directory
from node_analysis_pipeline.paths import _BUNDLE_ROOT, trainer_script  # Resolve vendored trainer path


@dataclass  # Immutable invocation bundle for subprocess trainer
class TrainerInvocation:
    pass_name: str  # Human label for logs and RUN_RECORD (e.g. mtl_binary)
    halt_mode: str  # binary or survival — forwarded as --halt-mode
    artifact_stem: str  # node_analysis_<run_id>_<pass> prefix for CSV outputs
    extra_args: List[str]  # Epochs, cohort CSV, checkpoint dir, loss weights, etc.
    dry_run: bool = False  # If True, log argv only and skip subprocess
    # If True (default), child stdout/stderr go to the console so epoch logs appear live.
    # If False, output is buffered and the last ~8k chars are stored in RUN_RECORD on failure.
    stream_output: bool = True  # Live console vs captured tails for debugging


def write_run_record(bundle_root: Path, payload: Dict[str, Any]) -> Path:  # Persist JSON audit of trainer run
    out_root = env.get_str(env.ENV_OUTPUT_ROOT, "")  # models/ root from apply_run_env
    rec_dir = Path(out_root) / "records" if out_root else bundle_root / "runs"  # Prefer thesis records/
    rec_dir.mkdir(parents=True, exist_ok=True)  # Ensure directory exists before write
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")  # Unique UTC suffix per invocation
    path = rec_dir / f"RUN_RECORD_{payload.get('pass_name','x')}_{ts}.json"  # One JSON audit file per run
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")  # Persist argv and return metadata
    return path


def invoke_trainer(inv: TrainerInvocation) -> int:  # Spawn vendored trainer; return exit code
    script = trainer_script()  # Absolute path to multitask_battery_halt_trainer_bundle.py
    if not script.is_file():  # Guard before subprocess — clearer than child import error
        raise FileNotFoundError(f"Trainer not found: {script}")

    argv = [sys.executable, str(script)]  # Same Python interpreter as orchestrator
    argv += inv.extra_args  # Orchestrator-supplied training hyperparameters and paths
    argv += ["--halt-mode", inv.halt_mode]  # binary vs survival halt head
    argv += ["--artifact-stem", inv.artifact_stem]  # CSV/checkpoint filename prefix

    record: Dict[str, Any] = {  # Audit payload written regardless of success/failure
        "pass_name": inv.pass_name,
        "halt_mode": inv.halt_mode,
        "artifact_stem": inv.artifact_stem,
        "argv": argv,
        "trainer_script": str(script),
        "ts_utc": datetime.now(timezone.utc).isoformat(),
    }

    if inv.dry_run:  # CI/docs: record intent without GPU/time cost
        write_run_record(_BUNDLE_ROOT, record)
        print("[dry-run] would execute:", " ".join(argv))
        return 0

    env = os.environ.copy()  # Inherit orchestrator env (RUN_ID, OUTPUT_ROOT, etc.)
    env.setdefault("PYTHONUTF8", "1")  # UTF-8 stdout/stderr for trainer logs
    models_default = str(_BUNDLE_ROOT.parent / "outputs" / "models")  # Fallback if OUTPUT_ROOT unset
    env.setdefault("NODE_ANALYSIS_OUTPUT_ROOT", models_default)
    env.setdefault("HOL_ACAD_OUTPUT_ROOT", models_default)  # Legacy trainer reads HOL_ACAD_*
    default_data = _BUNDLE_ROOT.parent / "idata"  # Default LoRaWAN telemetry tree
    env.setdefault("NODE_ANALYSIS_DATA_ROOT", str(default_data))
    env.setdefault("HOL_ACAD_DATA_ROOT", str(default_data))
    if inv.stream_output:  # Live epoch progress on parent console
        proc = subprocess.run(argv, cwd=str(script.parent), env=env)  # cwd = runtime_vendor/src
        record["returncode"] = proc.returncode
        record["stdout_stderr"] = "streamed_to_console"
    else:  # Capture output for failure diagnosis in RUN_RECORD
        proc = subprocess.run(argv, cwd=str(script.parent), env=env, capture_output=True, text=True)
        record["returncode"] = proc.returncode
        record["stdout_tail"] = proc.stdout[-8000:]  # Last 8k chars avoid huge JSON files
        record["stderr_tail"] = proc.stderr[-8000:]
    write_run_record(_BUNDLE_ROOT, record)  # Always persist audit trail
    if proc.returncode != 0 and not inv.stream_output:  # Surface recent stderr when buffered
        print(proc.stderr[-4000:], file=sys.stderr)
    return int(proc.returncode)  # Propagate to orchestrator failure list
