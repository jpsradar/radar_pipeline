"""
cli/run_case.py

Reproducible case runner (single-case orchestrator).

Role in the pipeline
--------------------
This module is the entrypoint that turns a case config into a fully traceable run.
It is intentionally boring: it wires configuration, execution, and artifacts.
It must not implement radar math.

Responsibilities
----------------
- Load + validate the case configuration (schema-driven).
- Resolve the execution engine (explicit or auto).
- Derive a stable run identity:
    <case_stem>__<engine>__seed<seed>__cfg<config_hash_prefix>
- Persist provenance and contracts:
    - config.normalized.json  : the normalized config that actually ran
    - manifest.json           : minimal, stable execution/provenance contract
    - case_manifest.json      : repo-native, richer provenance (existing format)
    - metrics.json            : engine outputs + injected execution/validity blocks

Execution contract (mandatory)
------------------------------
Every run MUST write an auditable contract to disk, independent of reports:
- execution.mode        : engine actually executed (model_based | monte_carlo | signal_level)
- execution.seed        : seed used
- execution.assumptions : assumptions declared in the case config (or "unspecified")
- validity              : minimal, conservative validity contract (top-level)

Engine/config consistency
-------------------------
If the case config declares execution.mode, it MUST match the resolved engine.
Mismatches fail loudly (we do not silently reinterpret config intent).

Failure behavior
----------------
manifest.json and config.normalized.json are written before running the engine.
If execution fails, the manifest is updated with status="failed" and an error string.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from core.config.loaders import ConfigError, LoadOptions, load_case
from core.config.manifest import compute_config_hash, write_case_manifest
from core.runtime.manifest import write_manifest, write_normalized_config


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

    # Seed precedence:
    #   1) --seed (explicit)
    #   2) $SEED
    #   3) derived from config hash
    parser.add_argument("--seed", type=int, default=None, help="Optional RNG seed (overrides $SEED).")

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
    mc = cfg.get("monte_carlo", None)
    return "monte_carlo" if isinstance(mc, dict) else "model_based"


def _derive_seed(config_hash: str) -> int:
    return int(config_hash[:8], 16) % (2**31 - 1)


def _seed_from_env() -> Optional[int]:
    raw = os.environ.get("SEED", "").strip()
    if not raw:
        return None
    try:
        return int(raw)
    except Exception:
        return None


def _ensure_empty_dir(path: Path, *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"Output directory already exists: {path}\n"
            "Refusing to overwrite. Use --overwrite for local iteration, or change the seed/config."
        )
    path.mkdir(parents=True, exist_ok=True)


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------
# Execution contract helpers
# ---------------------------------------------------------------------


def _get_cfg_execution_mode(cfg: Dict[str, Any]) -> Optional[str]:
    ex = cfg.get("execution", None)
    if not isinstance(ex, dict):
        return None
    mode = ex.get("mode", None)
    if mode is None:
        return None
    return str(mode).lower().strip()


def _get_cfg_assumptions(cfg: Dict[str, Any]) -> List[str]:
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
    validity: Dict[str, Any],
) -> Dict[str, Any]:
    metrics_out = dict(metrics)
    metrics_out["execution"] = {
        "mode": str(engine_selected).lower().strip(),
        "seed": int(seed),
        "assumptions": assumptions if assumptions else ["unspecified"],
    }
    # validity at top-level (same intent as manifest.json)
    metrics_out["validity"] = validity
    return metrics_out


def _infer_validity(cfg: Dict[str, Any]) -> Dict[str, Any]:
    stat_model = "unknown"
    clutter = "unknown"

    mc = cfg.get("monte_carlo", {}) if isinstance(cfg.get("monte_carlo", None), dict) else {}
    bg = mc.get("background", {}) if isinstance(mc.get("background", None), dict) else {}

    if bg:
        if "model" in bg:
            stat_model = str(bg.get("model"))
        hetero = bg.get("hetero", {})
        if isinstance(hetero, dict) and bool(hetero.get("enabled", False)):
            clutter = "heterogeneous"
        else:
            clutter = "homogeneous"

    return {
        "stat_model": stat_model,
        "clutter": clutter,
        "limits": [
            "assumptions are as-declared in case YAML",
            "validity is scenario-dependent",
        ],
    }


# ---------------------------------------------------------------------
# Engine dispatch
# ---------------------------------------------------------------------


def _run_engine(engine: str, cfg: Dict[str, Any], seed: int) -> Dict[str, Any]:
    engine_n = str(engine).lower().strip()

    if engine_n == "model_based":
        from core.simulation.model_based import run_model_based_case  # type: ignore
        return run_model_based_case(cfg=cfg, seed=seed)

    if engine_n == "monte_carlo":
        from core.simulation.monte_carlo import run_monte_carlo  # type: ignore
        return run_monte_carlo(cfg=cfg, seed=seed)

    if engine_n == "signal_level":
        from core.simulation.signal_level import run_signal_level_case  # type: ignore
        return run_signal_level_case(cfg=cfg, seed=seed)

    raise ValueError(f"Unknown engine: {engine}")


def _write_html_report(out_dir: Path, title: str | None = None) -> None:
    from reports.case_generators import generate_case_report_html  # type: ignore

    metrics_path = out_dir / "metrics.json"
    manifest_path = out_dir / "case_manifest.json"
    manifest_arg = manifest_path if manifest_path.exists() else None

    generate_case_report_html(
        case_dir=out_dir,
        metrics_path=metrics_path,
        manifest_path=manifest_arg,
        out_dir=out_dir,
        title=title,
    )


def _pretty_path(path: Path, project_root: Path) -> str:
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

    # ---- Load + validate config ----
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

    if args.seed is not None:
        seed = int(args.seed)
        seed_source = "user"
    else:
        env_seed = _seed_from_env()
        if env_seed is not None:
            seed = int(env_seed)
            seed_source = "env(SEED)"
        else:
            seed = _derive_seed(cfg_hash)
            seed_source = "derived_from_config_hash"

    ident = RunIdentity(case_stem=case_path.stem, engine=selected_engine, seed=seed, config_hash=cfg_hash)
    out_dir = (out_base / ident.run_id).resolve()

    # ---- Create output directory safely ----
    try:
        _ensure_empty_dir(out_dir, overwrite=bool(args.overwrite))
    except Exception as exc:
        msg = str(exc).replace(str(project_root), "${PROJECT_ROOT}")
        print(f"[ERROR] {msg}")
        return 4

    assumptions = _get_cfg_assumptions(cfg)
    validity = _infer_validity(cfg)

    # ---- Write normalized config + manifest early (even if engine fails) ----
    write_normalized_config(out_dir, cfg)
    write_manifest(
        out_dir=out_dir,
        run_id=ident.run_id,
        case_path=str(case_path),
        cfg_hash=cfg_hash,
        engine_requested=str(args.engine),
        engine_selected=str(selected_engine),
        seed=int(seed),
        seed_source=str(seed_source),
        assumptions=assumptions,
        validity=validity,
        schema_name=str(args.schema),
        schema_dir=str(args.schema_dir),
        status="started",
        error=None,
    )

    # ---- Write case_manifest.json (repo-native provenance) ----
    extras = {
        "cli_args": vars(args),
        "output_dir": str(Path(args.out) / ident.run_id),
        "engine_requested": args.engine,
        "engine_selected": selected_engine,
        "run_id": ident.run_id,
        "seed_source": seed_source,
        "display_name": args.name or "",
        "execution_mode_declared": cfg_mode or "",
        "assumptions_declared": assumptions,
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
        # mark manifest failed
        write_manifest(
            out_dir=out_dir,
            run_id=ident.run_id,
            case_path=str(case_path),
            cfg_hash=cfg_hash,
            engine_requested=str(args.engine),
            engine_selected=str(selected_engine),
            seed=int(seed),
            seed_source=str(seed_source),
            assumptions=assumptions,
            validity=validity,
            schema_name=str(args.schema),
            schema_dir=str(args.schema_dir),
            status="failed",
            error=f"Failed to write case_manifest.json: {exc}",
        )
        print(f"[ERROR] Failed to write case_manifest.json: {exc}")
        return 5

    # ---- Run engine + write metrics (+ contracts) ----
    try:
        metrics = _run_engine(selected_engine, cfg, seed=seed)
        metrics = _inject_execution_contract(
            metrics=metrics,
            engine_selected=selected_engine,
            seed=seed,
            assumptions=assumptions,
            validity=validity,
        )
        _write_json(out_dir / "metrics.json", metrics)

        # finalize manifest
        write_manifest(
            out_dir=out_dir,
            run_id=ident.run_id,
            case_path=str(case_path),
            cfg_hash=cfg_hash,
            engine_requested=str(args.engine),
            engine_selected=str(selected_engine),
            seed=int(seed),
            seed_source=str(seed_source),
            assumptions=assumptions,
            validity=validity,
            schema_name=str(args.schema),
            schema_dir=str(args.schema_dir),
            status="completed",
            error=None,
        )

    except Exception as exc:
        write_manifest(
            out_dir=out_dir,
            run_id=ident.run_id,
            case_path=str(case_path),
            cfg_hash=cfg_hash,
            engine_requested=str(args.engine),
            engine_selected=str(selected_engine),
            seed=int(seed),
            seed_source=str(seed_source),
            assumptions=assumptions,
            validity=validity,
            schema_name=str(args.schema),
            schema_dir=str(args.schema_dir),
            status="failed",
            error=str(exc),
        )
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