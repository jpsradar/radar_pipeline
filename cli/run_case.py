"""
cli/run_case.py

Reproducible case runner for the radar pipeline.

Why this module exists
----------------------
This file is the "orchestrator" for a single runnable case:
- Load + validate a case config (schema-driven)
- Resolve engine selection (explicit or auto)
- Derive a stable run identity (case stem + engine + seed + config hash)
- Write provenance artifacts (manifest + metrics + optional HTML report)

This module MUST remain boring and deterministic.
It should not implement radar math; it only wires contracts and I/O.

Contract: Execution metadata is mandatory
----------------------------------------
To prevent "it ran but we don't know what it meant", every run MUST persist a
human-auditable execution contract into outputs.

We enforce and persist:
- mode:   the resolved engine actually executed ("model_based" | "monte_carlo" | "signal_level")
- seed:   the integer seed actually used (user-provided or derived)
- assumptions: the engineering assumptions declared in the case config

Where it is written:
- metrics.json        : top-level "execution" block (always present)
- case_manifest.json  : includes seed + extras; metrics mirrors execution for consumers

Engine/config consistency
-------------------------
If the case config declares execution.mode, we require it to match the resolved engine
(we do NOT silently switch meaning). This prevents accidental mismatches between config
intent and CLI invocation.

Path redaction
--------------
Console output avoids absolute paths:
- If inside project root: ${PROJECT_ROOT}/...
- Else: basename only
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from core.config.loaders import ConfigError, LoadOptions, load_case
from core.config.manifest import compute_config_hash, write_case_manifest


# ---------------------------------------------------------------------
# Run identity / directory naming
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class RunIdentity:
    """Immutable identity for a run; used for directory naming and traceability."""

    case_stem: str
    engine: str
    seed: int
    config_hash: str

    @property
    def short_hash(self) -> str:
        return self.config_hash[:8]

    @property
    def run_id(self) -> str:
        return f"{self.case_stem}__{self.engine}__seed{self.seed}__cfg{self.short_hash}"


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="run_case",
        description="Run a single radar performance/simulation case (reproducible).",
    )

    parser.add_argument("--case", required=True, help="Path to case YAML/JSON.")
    parser.add_argument("--out", default="results/cases", help="Base output dir (default: results/cases).")

    parser.add_argument("--name", default=None, help="Optional display name (report title only).")
    parser.add_argument("--seed", type=int, default=None, help="Optional RNG seed (default: derived from config).")
    parser.add_argument("--schema-dir", default="configs/schemas", help="Schema folder (default: configs/schemas).")
    parser.add_argument("--schema", default="case.schema.json", help="Schema filename (default: case.schema.json).")

    parser.add_argument(
        "--engine",
        default="auto",
        choices=["auto", "model_based", "monte_carlo", "signal_level"],
        help="Engine selection. 'auto' picks based on presence of a monte_carlo block.",
    )

    parser.add_argument("--strict", action="store_true", help="Fail on unknown config fields.")
    parser.add_argument("--report", action="store_true", help="Generate report.html in the run directory.")
    parser.add_argument("--overwrite", action="store_true", help="Allow overwriting an existing run directory.")

    return parser.parse_args()


def _auto_select_engine(cfg: Dict[str, Any]) -> str:
    """
    Minimal, deterministic engine auto-selection.

    Rule
    ----
    - If cfg has a 'monte_carlo' dict: treat as a Monte Carlo experiment case.
    - Else: treat as a model_based performance case.
    """
    mc = cfg.get("monte_carlo", None)
    return "monte_carlo" if isinstance(mc, dict) else "model_based"


def _derive_seed(config_hash: str) -> int:
    """Derive a stable default seed from the config hash (bounded to signed 32-bit range)."""
    return int(config_hash[:8], 16) % (2**31 - 1)


def _ensure_empty_dir(path: Path, *, overwrite: bool) -> None:
    """
    Ensure output directory exists and is safe to write into.

    If overwrite=False and the directory exists, we refuse (reproducibility guard).
    """
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"Output directory already exists: {path}\n"
            "Refusing to overwrite. Use --overwrite for local iteration, or change the seed/config."
        )
    path.mkdir(parents=True, exist_ok=True)


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    """Write a JSON payload with stable formatting (diff-friendly)."""
    path.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------
# Execution contract helpers
# ---------------------------------------------------------------------


def _get_cfg_execution_mode(cfg: Dict[str, Any]) -> Optional[str]:
    """
    Read cfg.execution.mode if present.

    Returns
    -------
    str | None
        Lowercased mode string, or None if not present.
    """
    ex = cfg.get("execution", None)
    if not isinstance(ex, dict):
        return None
    mode = ex.get("mode", None)
    if mode is None:
        return None
    return str(mode).lower().strip()


def _get_cfg_assumptions(cfg: Dict[str, Any]) -> List[str]:
    """
    Read cfg.assumptions as a list of strings.

    The schema normally enforces this, but we keep this resilient because
    this function is part of the CLI contract layer.
    """
    raw = cfg.get("assumptions", None)
    if raw is None:
        return []
    if not isinstance(raw, list):
        return [str(raw)]
    out: List[str] = []
    for x in raw:
        s = str(x).strip()
        if s:
            out.append(s)
    return out


def _assert_engine_matches_execution_mode(*, engine_selected: str, cfg_mode: Optional[str]) -> None:
    """
    Enforce that cfg.execution.mode (if declared) matches the resolved engine.

    We do NOT silently reinterpret the config; mismatches must fail loudly.
    """
    if cfg_mode is None:
        return
    eng = str(engine_selected).lower().strip()
    if eng != cfg_mode:
        raise ValueError(
            "Engine/config mismatch: config declares execution.mode="
            f"'{cfg_mode}', but resolved engine is '{eng}'. "
            "Fix either the case config or the CLI --engine selection."
        )


def _inject_execution_contract(
    *,
    metrics: Dict[str, Any],
    engine_selected: str,
    seed: int,
    assumptions: List[str],
) -> Dict[str, Any]:
    """
    Inject a mandatory, top-level execution contract into metrics.

    This keeps consumers (reports/validation) from depending on CLI internals
    or manifest parsing for the most important provenance fields.
    """
    metrics_out = dict(metrics)
    metrics_out["execution"] = {
        "mode": str(engine_selected).lower().strip(),
        "seed": int(seed),
        "assumptions": assumptions if assumptions else ["unspecified"],
    }
    return metrics_out


# ---------------------------------------------------------------------
# Engine dispatch
# ---------------------------------------------------------------------


def _run_engine(engine: str, cfg: Dict[str, Any], seed: int) -> Dict[str, Any]:
    engine_n = str(engine).lower().strip()

    if engine_n == "model_based":
        from core.simulation.model_based import run_model_based_case  # type: ignore

        return run_model_based_case(cfg=cfg, seed=seed)

    if engine_n == "monte_carlo":
        # IMPORTANT: new unified entrypoint supports task=pfa|pd|pfa_pd,
        # but remains backward compatible (default task is pfa).
        from core.simulation.monte_carlo import run_monte_carlo  # type: ignore

        return run_monte_carlo(cfg=cfg, seed=seed)

    if engine_n == "signal_level":
        from core.simulation.signal_level import run_signal_level_case  # type: ignore

        return run_signal_level_case(cfg=cfg, seed=seed)

    raise ValueError(f"Unknown engine: {engine}")


def _write_html_report(out_dir: Path, title: str | None = None) -> None:
    """
    Generate report.html for a single case directory.

    Note
    ----
    Report generation is a pure post-process; it should not rerun simulations.
    """
    from reports.case_generators import generate_case_report_html  # type: ignore

    case_dir = out_dir
    metrics_path = case_dir / "metrics.json"
    manifest_path = case_dir / "case_manifest.json"
    manifest_arg = manifest_path if manifest_path.exists() else None

    generate_case_report_html(
        case_dir=case_dir,
        metrics_path=metrics_path,
        manifest_path=manifest_arg,
        out_dir=case_dir,
        title=title,
    )


def _pretty_path(path: Path, project_root: Path) -> str:
    """
    Render a path without leaking absolute filesystem locations.

    - If path is inside project_root: show as ${PROJECT_ROOT}/...
    - Else: show only the basename.
    """
    try:
        rel = path.resolve().relative_to(project_root.resolve())
        return str(Path("${PROJECT_ROOT}") / rel)
    except Exception:
        return path.name


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------


def main() -> int:
    args = _parse_args()

    project_root = Path.cwd().resolve()

    case_path = Path(args.case)
    out_base = Path(args.out)

    # ---- Load + validate config (schema-driven) ----
    try:
        cfg = load_case(
            case_path,
            schema_dir=args.schema_dir,
            schema_name=args.schema,
            options=LoadOptions(strict=bool(args.strict), resolve_paths=True, normalize_units=True),
        )
    except ConfigError as exc:
        print(f"[ERROR] Config load/validation failed: {exc}")
        return 2
    except Exception as exc:
        print(f"[ERROR] Unexpected error while loading config: {exc}")
        return 3

    # ---- Resolve engine ----
    selected_engine = args.engine
    if str(selected_engine).lower().strip() == "auto":
        selected_engine = _auto_select_engine(cfg)

    # ---- Enforce execution.mode contract if present ----
    cfg_mode = _get_cfg_execution_mode(cfg)
    _assert_engine_matches_execution_mode(engine_selected=selected_engine, cfg_mode=cfg_mode)

    # ---- Compute stable run identity ----
    cfg_hash = compute_config_hash(cfg, project_root=project_root)
    seed = int(args.seed) if args.seed is not None else _derive_seed(cfg_hash)

    ident = RunIdentity(case_stem=case_path.stem, engine=selected_engine, seed=seed, config_hash=cfg_hash)
    out_dir = (out_base / ident.run_id).resolve()

    # ---- Create output directory safely ----
    try:
        _ensure_empty_dir(out_dir, overwrite=bool(args.overwrite))
    except Exception as exc:
        msg = str(exc).replace(str(project_root), "${PROJECT_ROOT}")
        print(f"[ERROR] {msg}")
        return 4

    # ---- Write manifest (provenance) ----
    extras = {
        "cli_args": vars(args),
        "output_dir": str(Path(args.out) / ident.run_id),
        "engine_requested": args.engine,
        "engine_selected": selected_engine,
        "run_id": ident.run_id,
        "seed_source": "user" if args.seed is not None else "derived_from_config_hash",
        "display_name": args.name or "",
        "execution_mode_declared": cfg_mode or "",
        "assumptions_declared": _get_cfg_assumptions(cfg),
    }
    try:
        write_case_manifest(
            cfg,
            output_dir=out_dir,
            seed=seed,
            extras=extras,
            project_root=project_root,
            engine_package="radar-pipeline",
        )
    except Exception as exc:
        print(f"[ERROR] Failed to write case_manifest.json: {exc}")
        return 5

    # ---- Run engine + write metrics ----
    try:
        metrics = _run_engine(selected_engine, cfg, seed=seed)

        assumptions = _get_cfg_assumptions(cfg)
        metrics = _inject_execution_contract(
            metrics=metrics,
            engine_selected=selected_engine,
            seed=seed,
            assumptions=assumptions,
        )

        _write_json(out_dir / "metrics.json", metrics)
    except Exception as exc:
        print(f"[ERROR] Engine run failed: {exc}")
        return 6

    # ---- Optional report ----
    if args.report:
        try:
            _write_html_report(out_dir=out_dir, title=args.name)
        except Exception as exc:
            print(f"[ERROR] Failed to write report.html: {exc}")
            return 7

    print(f"[OK] Case complete: {_pretty_path(out_dir, project_root)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())