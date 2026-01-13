"""
core/runtime/manifest.py

Run provenance utilities (traceability contract).

Role in the pipeline
--------------------
This module owns the minimal, reader-facing artifacts that make a run
auditable and reproducible without reading code:

  1) config.normalized.json
     The exact normalized configuration that was executed (diff-friendly).
     This is the strongest reproducibility artifact.

  2) manifest.json
     A small, stable execution/provenance contract intended for external readers
     and lightweight tooling. It complements (does not replace) the repo-native
     case_manifest.json written elsewhere.

Design constraints
------------------
- Single source of truth for the configuration hash:
  the caller MUST pass cfg_hash computed by core.config.manifest.compute_config_hash.
  This module MUST NOT recompute a second hash.

- Dependency-minimal and deterministic:
  only standard library, stable JSON formatting.

- Failure-friendly:
  the manifest format supports status = "started" | "completed" | "failed"
  and an optional error string, so a run can leave a meaningful trace even
  if an engine crashes mid-execution.

Schema
------
manifest.json includes, at minimum:
- run_id, case_path, config_hash
- git_commit (best effort)
- engine.requested / engine.selected
- execution.mode / execution.seed / execution.seed_source / execution.assumptions
- validity (minimal contract, conservative by default)
- schema.name / schema.dir (if available)
- status (+ error when failed)
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional


def git_commit_hash() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            .decode("utf-8")
            .strip()
        )
    except Exception:
        return "unknown"


def write_normalized_config(out_dir: Path, cfg: Dict[str, Any]) -> Path:
    """
    Persist the normalized config that actually ran.
    This is the strongest reproducibility artifact.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / "config.normalized.json"
    p.write_text(json.dumps(cfg, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return p


def write_manifest(
    *,
    out_dir: Path,
    run_id: str,
    case_path: str,
    cfg_hash: str,
    engine_requested: str,
    engine_selected: str,
    seed: int,
    seed_source: str,
    assumptions: List[str],
    validity: Dict[str, Any],
    schema_name: str = "",
    schema_dir: str = "",
    status: str = "completed",  # "started" | "completed" | "failed"
    error: Optional[str] = None,
) -> Path:
    """
    Write a minimal manifest.json.

    NOTE: cfg_hash must come from core.config.manifest.compute_config_hash
    (caller responsibility). We do not recompute hashes here.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    payload: Dict[str, Any] = {
        "run_id": run_id,
        "case_path": case_path,
        "config_hash": str(cfg_hash),
        "git_commit": git_commit_hash(),
        "engine": {
            "requested": str(engine_requested).lower().strip(),
            "selected": str(engine_selected).lower().strip(),
        },
        "execution": {
            "mode": str(engine_selected).lower().strip(),
            "seed": int(seed),
            "seed_source": str(seed_source),
            "assumptions": assumptions if assumptions else ["unspecified"],
        },
        "validity": validity,
        "schema": {
            "name": str(schema_name),
            "dir": str(schema_dir),
        },
        "status": str(status),
    }

    if error:
        payload["error"] = str(error)

    p = out_dir / "manifest.json"
    p.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return p