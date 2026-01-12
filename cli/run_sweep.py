"""
cli/run_sweep.py

Deterministic grid sweep runner for the radar pipeline.

This script intentionally emphasizes reproducibility:
- Inputs are schema-validated (base case) and parsed deterministically (sweep spec).
- Output directory naming is stable and derived from hashes, not timestamps.
- A sweep manifest is written before execution starts.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

from core.config.loaders import load_case, LoadOptions, ConfigError
from core.config.manifest import compute_config_hash, write_case_manifest
from sweeps.grid import run_grid_sweep


# ---------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="run_sweep",
        description="Run a deterministic grid sweep over a radar case.",
    )

    parser.add_argument("--case", required=True, help="Base case YAML/JSON file (performance case).")
    parser.add_argument("--sweep", required=True, help="Sweep specification YAML/JSON file (grid definition).")

    parser.add_argument(
        "--out",
        required=True,
        help=(
            "Output path. If it ends with .json, it is treated as the sweep JSON file path. "
            "Otherwise it is treated as an output directory and 'sweep.json' is written inside."
        ),
    )

    parser.add_argument(
        "--engine",
        default="model_based",
        choices=["model_based", "monte_carlo", "signal_level"],
        help="Simulation engine to run for each sweep point (default: model_based).",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=123,
        help="Seed used to make the sweep deterministic (default: 123).",
    )

    parser.add_argument(
        "--name",
        default=None,
        help=("Optional display name (only affects reporting text). Directory naming remains stable."),
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow writing into an existing non-empty output directory (dangerous; local iteration only).",
    )

    parser.add_argument(
        "--report",
        action="store_true",
        help=(
            "Generate a report in '<out_dir>/report/' (report.json + report.html + plots/). "
            "Requires reports.generators."
        ),
    )

    parser.add_argument(
        "--objectives",
        default=None,
        help="Optional JSON string of objectives (used only for reporting).",
    )

    return parser.parse_args()


def _resolve_out(
    args_out: str,
    *,
    case_stem: str,
    engine: str,
    seed: int,
    base_case_hash: str,
    sweep_hash: str,
) -> Tuple[Path, Path]:
    """
    Resolve output paths for a sweep run using a *stable* naming convention.

    Directory naming
    ----------------
    sweep__<case_stem>__<engine>__seed<seed>__cfg<hash8>__sw<hash8>
    """
    out = Path(args_out)

    run_id = f"sweep__{case_stem}__{engine}__seed{seed}__cfg{base_case_hash[:8]}__sw{sweep_hash[:8]}"

    # File mode: explicit .json path
    if out.suffix.lower() == ".json":
        out_dir = out.parent / run_id
        sweep_json_path = out  # keep the user-requested JSON path
        out_dir.mkdir(parents=True, exist_ok=True)
        return out_dir.resolve(), sweep_json_path.resolve()

    # Directory mode: create a run folder inside `out`
    out_dir = (out / run_id).resolve()
    sweep_json_path = out_dir / "sweep.json"
    return out_dir, sweep_json_path


def _write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def _default_run_name(path: Path) -> str:
    # Kept for backward compatibility in older outputs; not used for naming anymore.
    ts = _dt.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    return f"{path.stem}__{ts}"


def _make_runner(engine: str, seed: int) -> Callable[[Dict[str, Any]], Dict[str, Any]]:
    engine_n = str(engine).lower().strip()

    if engine_n == "model_based":
        from core.simulation.model_based import run_model_based  # type: ignore
        return lambda cfg: run_model_based(cfg)

    if engine_n == "monte_carlo":
        from core.simulation.monte_carlo import run_monte_carlo  # type: ignore
        return lambda cfg: run_monte_carlo(cfg, seed=seed)

    if engine_n == "signal_level":
        from core.simulation.signal_level import run_signal_level  # type: ignore
        return lambda cfg: run_signal_level(cfg, seed=seed)

    raise ValueError(f"Unknown engine: {engine}")


def _parse_objectives(s: Optional[str]) -> Optional[Dict[str, Any]]:
    if not s:
        return None
    obj = json.loads(s)
    if not isinstance(obj, dict):
        raise ValueError("Objectives must be a JSON object (dict).")
    return obj


def _write_report(out_dir: Path, sweep_obj: Dict[str, Any], objectives: Optional[Dict[str, Any]]) -> None:
    from reports.generators import make_sweep_report  # type: ignore

    rep_dir = out_dir / "report"
    rep_dir.mkdir(parents=True, exist_ok=True)

    report = make_sweep_report(sweep_obj=sweep_obj, objectives=objectives, out_dir=rep_dir)
    _write_json(rep_dir / "report.json", report)

    html = report.get("html", "")
    if html:
        (rep_dir / "report.html").write_text(html, encoding="utf-8")


# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------

def main() -> int:
    args = _parse_args()

    case_path = Path(args.case)
    sweep_path = Path(args.sweep)

    try:
        objectives = _parse_objectives(args.objectives) if args.report else None
    except Exception as exc:
        print(f"[ERROR] Invalid --objectives: {exc}")
        return 2

    # Load base case with strict validation.
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

    # Load sweep spec with relaxed validation (it is not a performance case).
    try:
        sweep_spec = load_case(
            sweep_path,
            options=LoadOptions(strict=False, resolve_paths=True, normalize_units=False),
        )
    except Exception as exc:
        print(f"[ERROR] Failed to load sweep spec: {exc}")
        return 5

    # Resolve stable output directory name from config hashes.
    project_root = Path.cwd()
    base_case_hash = compute_config_hash(base_cfg, project_root=project_root)
    sweep_hash = compute_config_hash(sweep_spec, project_root=project_root)

    out_dir, sweep_json_path = _resolve_out(
        args.out,
        case_stem=case_path.stem,
        engine=str(args.engine),
        seed=int(args.seed),
        base_case_hash=base_case_hash,
        sweep_hash=sweep_hash,
    )

    # Refuse to overwrite by default (same policy as run_case).
    if out_dir.exists() and any(out_dir.iterdir()) and not bool(getattr(args, "overwrite", False)):
        print(
            f"[ERROR] Output directory already exists and is not empty: {out_dir}\n"
            "Refusing to overwrite. Use --overwrite for local iteration, or change the seed/sweep/config."
        )
        return 6

    out_dir.mkdir(parents=True, exist_ok=True)

    # Write a sweep manifest early for traceability, even if execution fails later.
    sweep_manifest_cfg: Dict[str, Any] = {
        "base_case": base_cfg,
        "sweep_spec": sweep_spec,
    }
    extras = {
        "cli_args": vars(args),
        "run_id_dir": str(out_dir),
        "engine_selected": str(args.engine),
        "base_case_hash": base_case_hash,
        "sweep_hash": sweep_hash,
        "case_path": str(case_path),
        "sweep_path": str(sweep_path),
        "sweep_json_path": str(sweep_json_path),
    }
    try:
        write_case_manifest(
            sweep_manifest_cfg,
            output_dir=out_dir,
            seed=int(args.seed),
            extras=extras,
            project_root=project_root,
            engine_package="radar-pipeline",
            filename="sweep_manifest.json",
        )
    except Exception as exc:
        print(f"[ERROR] Failed to write sweep_manifest.json: {exc}")
        return 7

    # Run sweep.
    try:
        runner = _make_runner(engine=args.engine, seed=int(args.seed))
        results = run_grid_sweep(base_cfg=base_cfg, sweep_spec=sweep_spec, runner=runner)
    except Exception as exc:
        print(f"[ERROR] Sweep execution failed: {exc}")
        return 8

    # Write sweep artifact.
    sweep_obj: Dict[str, Any] = {
        "meta": {
            "case_path": str(case_path),
            "sweep_path": str(sweep_path),
            "engine": str(args.engine),
            "seed": int(args.seed),
            "timestamp_utc": _dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
            "base_case_hash": base_case_hash,
            "sweep_hash": sweep_hash,
        },
        "n_points": int(results.get("n_points", 0)),
        "results": results.get("results", []),
    }
    _write_json(sweep_json_path, sweep_obj)

    # Optional report.
    if args.report:
        try:
            _write_report(out_dir=out_dir, sweep_obj=sweep_obj, objectives=objectives)
        except Exception as exc:
            print(f"[ERROR] Failed to write sweep report: {exc}")
            return 9

    print(f"[OK] Sweep complete: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())