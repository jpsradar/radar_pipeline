"""
cli/run_sweep.py

Deterministic grid sweep runner for the radar pipeline.

Changes in this revision
------------------------
- Print paths relative to project root to avoid leaking absolute paths.
- Treat run_grid_sweep() return value as a LIST (repo contract).
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import yaml

from core.config.loaders import load_case, LoadOptions, ConfigError
from core.config.manifest import compute_config_hash, write_case_manifest
from sweeps.grid import run_grid_sweep


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="run_sweep",
        description="Run a deterministic grid sweep over a radar case.",
    )
    parser.add_argument("--case", required=True, help="Base case YAML/JSON file (performance case).")
    parser.add_argument("--sweep", required=True, help="Sweep spec YAML/JSON file (grid definition).")
    parser.add_argument("--out", default="results/sweeps", help="Output directory (default: results/sweeps).")
    parser.add_argument(
        "--engine",
        default="model_based",
        choices=["model_based", "monte_carlo", "signal_level"],
        help="Engine per sweep point (default: model_based).",
    )
    parser.add_argument("--seed", type=int, default=123, help="Deterministic seed (default: 123).")
    parser.add_argument("--overwrite", action="store_true", help="Allow overwriting an existing run directory.")
    parser.add_argument("--report", action="store_true", help="Generate a report in '<out_dir>/report/'.")
    parser.add_argument("--objectives", default=None, help="Optional objectives JSON (reporting only).")
    return parser.parse_args()


def _load_yaml_or_json(path: Path) -> Dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        obj = json.loads(text)
    else:
        obj = yaml.safe_load(text)
    if not isinstance(obj, dict):
        raise ValueError(f"Sweep spec must be a dict/object, got: {type(obj).__name__}")
    return obj


def _parse_objectives(s: Optional[str]) -> Optional[Dict[str, Any]]:
    if not s:
        return None
    obj = json.loads(s)
    if not isinstance(obj, dict):
        raise ValueError("Objectives must be a JSON object (dict).")
    return obj


def _make_runner(engine: str, seed: int) -> Callable[[Dict[str, Any]], Dict[str, Any]]:
    engine_n = str(engine).lower().strip()

    if engine_n == "model_based":
        from core.simulation.model_based import run_model_based_case  # type: ignore

        def _runner(cfg: Dict[str, Any]) -> Dict[str, Any]:
            return run_model_based_case(cfg=cfg, seed=int(seed))

        return _runner

    if engine_n == "monte_carlo":
        from core.simulation.monte_carlo import run_pfa_monte_carlo  # type: ignore

        def _runner(cfg: Dict[str, Any]) -> Dict[str, Any]:
            return run_pfa_monte_carlo(cfg=cfg, seed=int(seed))

        return _runner

    if engine_n == "signal_level":
        from core.simulation.signal_level import run_signal_level_case  # type: ignore

        def _runner(cfg: Dict[str, Any]) -> Dict[str, Any]:
            return run_signal_level_case(cfg=cfg, seed=int(seed))

        return _runner

    raise ValueError(f"Unknown engine: {engine}")


def _write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def _resolve_out_dir(
    out_base: Path,
    *,
    case_stem: str,
    engine: str,
    seed: int,
    base_case_hash: str,
    sweep_hash: str,
) -> Path:
    run_id = f"sweep__{case_stem}__{engine}__seed{seed}__cfg{base_case_hash[:8]}__sw{sweep_hash[:8]}"
    return (out_base / run_id).resolve()


def _write_report(out_dir: Path, sweep_obj: Dict[str, Any], objectives: Optional[Dict[str, Any]]) -> None:
    from reports.generators import make_sweep_report  # type: ignore

    rep_dir = out_dir / "report"
    rep_dir.mkdir(parents=True, exist_ok=True)

    report = make_sweep_report(sweep_obj=sweep_obj, objectives=objectives, out_dir=rep_dir)
    _write_json(rep_dir / "report.json", report)

    html = report.get("html", "")
    if html:
        (rep_dir / "report.html").write_text(html, encoding="utf-8")


def _pretty_path(path: Path, project_root: Path) -> str:
    try:
        rel = path.resolve().relative_to(project_root.resolve())
        return str(Path("${PROJECT_ROOT}") / rel)
    except Exception:
        return path.name


def main() -> int:
    args = _parse_args()
    project_root = Path.cwd().resolve()

    case_path = Path(args.case)
    sweep_path = Path(args.sweep)
    out_base = Path(args.out)

    try:
        objectives = _parse_objectives(args.objectives) if args.report else None
    except Exception as exc:
        print(f"[ERROR] Invalid --objectives: {exc}")
        return 2

    try:
        base_cfg = load_case(
            case_path,
            schema_dir="configs/schemas",
            schema_name="case.schema.json",
            options=LoadOptions(strict=True, resolve_paths=True, normalize_units=True),
        )
    except ConfigError as exc:
        print(f"[ERROR] Failed to load/validate base case: {exc}")
        return 3
    except Exception as exc:
        print(f"[ERROR] Unexpected error while loading base case: {exc}")
        return 4

    try:
        sweep_spec = _load_yaml_or_json(sweep_path)
    except Exception as exc:
        print(f"[ERROR] Failed to load sweep spec: {exc}")
        return 5

    base_case_hash = compute_config_hash(base_cfg, project_root=project_root)
    sweep_hash = compute_config_hash(sweep_spec, project_root=project_root)

    out_dir = _resolve_out_dir(
        out_base,
        case_stem=case_path.stem,
        engine=str(args.engine),
        seed=int(args.seed),
        base_case_hash=base_case_hash,
        sweep_hash=sweep_hash,
    )

    if out_dir.exists() and any(out_dir.iterdir()) and not bool(args.overwrite):
        msg = (
            f"Output directory already exists and is not empty: {_pretty_path(out_dir, project_root)}\n"
            "Refusing to overwrite. Use --overwrite for local iteration, or change the seed/sweep/config."
        )
        print(f"[ERROR] {msg}")
        return 6

    out_dir.mkdir(parents=True, exist_ok=True)

    sweep_manifest_cfg: Dict[str, Any] = {"base_case": base_cfg, "sweep_spec": sweep_spec}
    extras = {
        "cli_args": vars(args),
        "engine_selected": str(args.engine),
        "base_case_hash": base_case_hash,
        "sweep_hash": sweep_hash,
        "case_path": str(case_path),
        "sweep_path": str(sweep_path),
        "out_dir": str(out_dir),  # will be sanitized in manifest
    }
    try:
        write_case_manifest(
            sweep_manifest_cfg,
            output_dir=out_dir,
            seed=int(args.seed),
            extras=extras,
            project_root=project_root,
            engine_package="radar-pipeline",
            filename="sweep_queries.json",  # consistent naming in repo outputs
        )
    except Exception as exc:
        print(f"[ERROR] Failed to write sweep manifest: {exc}")
        return 7

    try:
        runner = _make_runner(engine=args.engine, seed=int(args.seed))
        results: List[Dict[str, Any]] = run_grid_sweep(base_cfg=base_cfg, sweep_spec=sweep_spec, runner=runner)
    except Exception as exc:
        print(f"[ERROR] Sweep execution failed: {exc}")
        return 8

    sweep_obj: Dict[str, Any] = {
        "meta": {
            "case_path": str(case_path),
            "sweep_path": str(sweep_path),
            "engine": str(args.engine),
            "seed": int(args.seed),
            "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "base_case_hash": base_case_hash,
            "sweep_hash": sweep_hash,
        },
        "n_points": int(len(results)),
        "results": results,
    }
    _write_json(out_dir / "sweep.json", sweep_obj)

    if args.report:
        try:
            _write_report(out_dir=out_dir, sweep_obj=sweep_obj, objectives=objectives)
        except Exception as exc:
            print(f"[ERROR] Failed to write sweep report: {exc}")
            return 9

    print(f"[OK] Sweep complete: {_pretty_path(out_dir, project_root)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())