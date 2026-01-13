"""
core/runtime/manifest.py

Run manifest + traceability utilities.

Every run MUST emit a minimal manifest that captures:
- execution mode (model_based | monte_carlo | signal_level)
- seed
- config hash (stable)
- git commit hash
- assumptions (human-facing)
- validity contract (what stats model / clutter regime / limits)

This is intentionally small and dependency-free.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Dict, List


def git_commit_hash() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            .decode("utf-8")
            .strip()
        )
    except Exception:
        return "unknown"


def hash_config(cfg: Dict[str, Any]) -> str:
    blob = json.dumps(cfg, sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:12]


def write_manifest(
    *,
    out_dir: Path,
    cfg: Dict[str, Any],
    execution_mode: str,
    seed: int,
    assumptions: List[str],
    validity: Dict[str, Any],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "execution_mode": str(execution_mode).lower().strip(),
        "seed": int(seed),
        "config_hash": hash_config(cfg),
        "git_commit": git_commit_hash(),
        "assumptions": assumptions if assumptions else ["unspecified"],
        "validity": validity,
    }

    (out_dir / "manifest.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )