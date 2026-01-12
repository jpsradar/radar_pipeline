"""
cli/run_case.py

Reproducible case runner for the radar pipeline.

Changes in this revision
------------------------
- Print paths relative to project root (or CWD) to avoid leaking absolute paths.
- Keep stable run directory naming and overwrite-protection behavior.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

from core.config.loaders import ConfigError, LoadOptions, load_case
from core.config.manifest import compute_config_hash, write_case_manifest


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
    mc = cfg.get("monte_carlo", None)
    return "monte_carlo" if isinstance(mc, dict) else "model_based"


def _derive_seed(config_hash: str) -> int:
    return int(config_hash[:8], 16) % (2**31 - 1)


def _ensure_empty_dir(path: Path, *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"Output directory already exists: {path}\n"
            "Refusing to overwrite. Use --overwrite for local iteration, or change the seed/config."
        )
    path.mkdir(parents=True, exist_ok=True)


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def _run_engine(engine: str, cfg: Dict[str, Any], seed: int) -> Dict[str, Any]:
    engine_n = str(engine).lower().strip()

    if engine_n == "model_based":
        from core.simulation.model_based import run_model_based_case  # type: ignore

        return run_model_based_case(cfg=cfg, seed=seed)

    if engine_n == "monte_carlo":
        from core.simulation.monte_carlo import run_pfa_monte_carlo  # type: ignore

        return run_pfa_monte_carlo(cfg=cfg, seed=seed)

    if engine_n == "signal_level":
        from core.simulation.signal_level import run_signal_level_case  # type: ignore

        return run_signal_level_case(cfg=cfg, seed=seed)

    raise ValueError(f"Unknown engine: {engine}")


def _write_html_report(out_dir: Path, title: str | None = None) -> None:
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


def main() -> int:
    args = _parse_args()

    project_root = Path.cwd().resolve()

    case_path = Path(args.case)
    out_base = Path(args.out)

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

    selected_engine = args.engine
    if str(selected_engine).lower().strip() == "auto":
        selected_engine = _auto_select_engine(cfg)

    cfg_hash = compute_config_hash(cfg, project_root=project_root)
    seed = int(args.seed) if args.seed is not None else _derive_seed(cfg_hash)

    ident = RunIdentity(case_stem=case_path.stem, engine=selected_engine, seed=seed, config_hash=cfg_hash)
    out_dir = (out_base / ident.run_id).resolve()

    try:
        _ensure_empty_dir(out_dir, overwrite=bool(args.overwrite))
    except Exception as exc:
        # Keep the error message but redact absolute paths in the printed line below.
        msg = str(exc).replace(str(project_root), "${PROJECT_ROOT}")
        print(f"[ERROR] {msg}")
        return 4

    extras = {
        "cli_args": vars(args),
        "output_dir": str(Path(args.out) / ident.run_id),
        "engine_requested": args.engine,
        "engine_selected": selected_engine,
        "run_id": ident.run_id,
        "seed_source": "user" if args.seed is not None else "derived_from_config_hash",
        "display_name": args.name or "",
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

    try:
        metrics = _run_engine(selected_engine, cfg, seed=seed)
        _write_json(out_dir / "metrics.json", metrics)
    except Exception as exc:
        print(f"[ERROR] Engine run failed: {exc}")
        return 6

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