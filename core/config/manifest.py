"""
core/config/manifest.py

Case manifest generation for reproducibility, WITHOUT leaking absolute paths.

Goals
-----
- Keep runs reproducible (config, seed, git hash, package version, CLI extras).
- Avoid leaking machine-specific absolute paths (e.g., /Users/.../venv/bin/python).
- Provide a stable config hash so that results can be traced to the exact inputs.

Policy
------
- Any path under project_root is stored as "${PROJECT_ROOT}/<relative>".
- Any path outside project_root is reduced to:
    "<ABSOLUTE_PATH_REDACTED>:<basename>"

Notes
-----
- The config hash is computed over a canonical JSON serialization of the *sanitized* config
  (so it is stable across machines even if absolute paths differ).
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

try:
    # Python 3.8+: standard library
    from importlib.metadata import version as pkg_version  # type: ignore
except Exception:  # pragma: no cover
    pkg_version = None  # type: ignore


class ManifestError(RuntimeError):
    """Raised when manifest generation or writing fails."""


@dataclass(frozen=True)
class GitInfo:
    """Minimal git identity for traceability."""

    hash: str
    dirty: Optional[bool]


def _safe_path_str(value: str, project_root: Optional[Path]) -> str:
    """
    Sanitize a path-like string to avoid leaking absolute machine paths.

    Rules
    -----
    - If inside project_root -> "${PROJECT_ROOT}/<relpath>"
    - Else -> "<ABSOLUTE_PATH_REDACTED>:<basename>"

    Important
    ---------
    We treat strings as "path-like" if they either:
      - are absolute paths, OR
      - contain a path separator, OR
      - start with '.' or '~'
    """
    if project_root is None:
        return value

    # Heuristic: only sanitize things that look like paths.
    looks_like_path = (
        value.startswith("/")
        or value.startswith("~")
        or value.startswith(".")
        or ("/" in value)
    )
    if not looks_like_path:
        return value

    try:
        p = Path(value).expanduser()
    except Exception:
        return value

    try:
        pr = project_root.resolve()
        rp = p.resolve()
        try:
            rel = rp.relative_to(pr)
            return str(Path("${PROJECT_ROOT}") / rel)
        except Exception:
            return f"<ABSOLUTE_PATH_REDACTED>:{rp.name}"
    except Exception:
        return value


def build_case_manifest(
    cfg: Dict[str, Any],
    *,
    seed: Optional[int],
    extras: Optional[Dict[str, Any]],
    project_root: Optional[Path],
    engine_package: str = "radar_pipeline",
    include_identity: bool = True,
) -> Dict[str, Any]:
    """
    Build an in-memory case manifest dict.

    NOTE
    ----
    - Paths are sanitized to avoid leaking absolute machine locations.
    - The config hash is computed over the sanitized config for stability across machines.
    """
    pr = project_root.resolve() if project_root is not None else None

    env: Dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        # Store these but sanitized; they are still useful for debugging.
        "executable": _safe_path_str(sys.executable, pr),
        "cwd": _safe_path_str(str(Path.cwd()), pr),
    }
    if include_identity:
        env["user"] = os.environ.get("USER") or os.environ.get("USERNAME") or ""
        env["hostname"] = platform.node()

    pkg_ver = ""
    if pkg_version is not None:
        try:
            pkg_ver = pkg_version(engine_package)
        except Exception:
            pkg_ver = ""

    cfg_hash = compute_config_hash(cfg, project_root=pr)

    manifest: Dict[str, Any] = {
        "engine_package": engine_package,
        "engine_version": pkg_ver,
        "seed": seed,
        "config_hash": cfg_hash,
        "case": cfg,
        "git": _git_info(pr).__dict__ if pr is not None else {"hash": "", "dirty": None},
        "environment": env,
        "extras": extras or {},
    }

    # Final sanitization pass (covers any stray path-like strings).
    manifest = _walk_sanitize(manifest, pr)
    return manifest

def _walk_sanitize(obj: Any, project_root: Optional[Path]) -> Any:
    """Recursively sanitize path-like strings inside nested dict/list structures."""
    if isinstance(obj, dict):
        return {str(k): _walk_sanitize(v, project_root) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_walk_sanitize(v, project_root) for v in obj]
    if isinstance(obj, tuple):
        return [_walk_sanitize(v, project_root) for v in obj]
    if isinstance(obj, Path):
        return _safe_path_str(str(obj), project_root)
    if isinstance(obj, str):
        return _safe_path_str(obj, project_root)
    return obj


def _git_info(project_root: Path) -> GitInfo:
    """Best-effort git hash and dirty flag; returns empty fields if unavailable."""
    try:
        # Hash
        h = (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=str(project_root),
                stderr=subprocess.DEVNULL,
                text=True,
            )
            .strip()
        )
        # Dirty?
        dirty = (
            subprocess.call(
                ["git", "diff", "--quiet"],
                cwd=str(project_root),
                stderr=subprocess.DEVNULL,
            )
            != 0
        )
        return GitInfo(hash=h, dirty=dirty)
    except Exception:
        return GitInfo(hash="", dirty=None)


def _canonical_json(obj: Any) -> str:
    """
    Canonical JSON string for hashing.

    Implementation details
    ----------------------
    - sort_keys=True: stable key order
    - separators=(',', ':'): no whitespace differences
    - ensure_ascii=True: stable encoding across terminals
    """
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def compute_config_hash(cfg: Dict[str, Any], *, project_root: Optional[Path]) -> str:
    """
    Compute a stable SHA-256 hash for a loaded case configuration.

    Important
    ---------
    The hash is computed over a *sanitized* form of cfg so that absolute paths do not
    change the hash across machines.
    """
    sanitized_cfg = _walk_sanitize(cfg, project_root)
    payload = _canonical_json(sanitized_cfg).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _write_json_deterministic(path: Path, payload: Dict[str, Any]) -> None:
    """Write JSON with stable formatting for clean diffs and reliable hashing."""
    text = json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=False)
    path.write_text(text + "\n", encoding="utf-8")


def write_case_manifest(
    cfg: Dict[str, Any],
    *,
    output_dir: Path,
    seed: Optional[int],
    extras: Optional[Dict[str, Any]],
    project_root: Optional[Path],
    engine_package: str = "radar_pipeline",
    filename: str = "case_manifest.json",
    include_identity: bool = True,
) -> Dict[str, Any]:
    """
    Build and write a case manifest JSON file to output_dir.

    Returns
    -------
    dict
        The manifest payload that was written.
    """
    try:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / filename

        manifest = build_case_manifest(
            cfg,
            seed=seed,
            extras=extras,
            project_root=project_root,
            engine_package=engine_package,
            include_identity=include_identity,
        )

        _write_json_deterministic(out_path, manifest)
        return manifest
    except Exception as exc:
        raise ManifestError(f"Failed to write manifest JSON to {output_dir}: {exc}") from exc