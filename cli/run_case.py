"""
cli/run_case.py

Reproducible case runner for the radar pipeline.

Key properties
--------------
- Schema validation and unit normalization happen at load time.
- Execution is deterministic given (normalized config, seed, engine).
- Outputs are written to a run directory with a *stable* name:
    <case_stem>__<engine>__seed<seed>__cfg<hash8>

Why the stable name matters
---------------------------
A stable run directory name enables:
- idempotent reruns (no hidden timestamp variance)
- traceability (run folder name encodes the minimal identity)
- clean diffs for reports/metrics across revisions

Design constraints
------------------
- No absolute paths are written into the manifest or HTML report.
- If the output directory already exists, the CLI errors out unless --overwrite is used.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

from core.config.loaders import ConfigError, LoadOptions, load_case
from core.config.manifest import compute_config_hash, write_case_manifest

# Engines are imported lazily inside _run_engine to keep CLI import time low.


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


# ----------------------------
# CLI
# ----------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="run_case",
        description="Run a single radar performance/simulation case (reproducible).",
    )

    parser.add_argument("--case", required=True, help="Path to case YAML/JSON.")
    parser.add_argument("--out", default="results/cases", help="Base output dir (default: results/cases).")

    parser.add_argument(
        "--name",
        default=None,
        help="Optional display name (only affects reporting text). Directory naming remains stable.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional RNG seed. If omitted, derived from config hash.",
    )
    parser.add_argument("--schema-dir", default="configs/schemas", help="Schema folder (default: configs/schemas).")

    parser.add_argument(
        "--schema",
        default="case.schema.json",
        help="Schema filename (default: case.schema.json).",
    )

    parser.add_argument(
        "--engine",
        default="auto",
        choices=["auto", "model_based", "monte_carlo", "signal_level"],
        help="Engine selection. 'auto' picks based on presence of a monte_carlo block.",
    )

    parser.add_argument(
        "--strict",
        action="store_true",
        help="If set, fail on unknown config fields (schema strictness).",
    )

    parser.add_argument(
        "--report",
        action="store_true",
        help="If set, generate a self-contained HTML report in the run directory.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="If set, allow writing into an existing run directory (dangerous; use for local iteration only).",
    )

    return parser.parse_args()


def _auto_select_engine(cfg: Dict[str, Any]) -> str:
    """Auto-select engine based on the loaded case structure."""
    mc = cfg.get("monte_carlo", None)
    if isinstance(mc, dict):
        return "monte_carlo"
    return "model_based"


def _derive_seed(config_hash: str) -> int:
    """Derive a deterministic seed from the config hash when the user does not provide one."""
    # Use the first 8 hex chars (32-bit) and keep it within signed 31-bit range for NumPy compatibility.
    return int(config_hash[:8], 16) % (2**31 - 1)


def _ensure_empty_dir(path: Path, *, overwrite: bool) -> None:
    """Create an output directory, refusing to overwrite unless explicitly allowed."""
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"Output directory already exists: {path}\n"
            "Refusing to overwrite. Use --overwrite for local iteration, or change the seed/config."
        )
    path.mkdir(parents=True, exist_ok=True)


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    """Write JSON with stable formatting for readable diffs."""
    text = json.dumps(payload, sort_keys=True, indent=2)
    path.write_text(text + "\n", encoding="utf-8")


# ----------------------------
# Engine dispatch
# ----------------------------

def _run_engine(engine: str, cfg: Dict[str, Any], seed: int) -> Dict[str, Any]:
    engine_n = str(engine).lower().strip()

    if engine_n == "model_based":
        from core.simulation.model_based import run_model_based  # type: ignore
        return run_model_based(cfg)

    if engine_n == "monte_carlo":
        from core.simulation.monte_carlo import run_monte_carlo  # type: ignore
        return run_monte_carlo(cfg, seed=seed)

    if engine_n == "signal_level":
        from core.simulation.signal_level import run_signal_level  # type: ignore
        return run_signal_level(cfg, seed=seed)

    raise ValueError(f"Unknown engine: {engine}")


def _write_html_report(out_dir: Path, metrics: Dict[str, Any]) -> None:
    """Generate a self-contained HTML report (best-effort)."""
    from reports.make_case_report import make_case_report_html  # type: ignore

    html = make_case_report_html(out_dir=out_dir, metrics=metrics)
    (out_dir / "report.html").write_text(html, encoding="utf-8")


# ----------------------------
# Main
# ----------------------------

def main() -> int:
    args = _parse_args()

    case_path = Path(args.case)
    out_base = Path(args.out)

    # 1) Load + validate config first (so naming can be derived from the true normalized config).
    try:
        cfg = load_case(
            case_path,
            schema_dir=args.schema_dir,
            schema_name=args.schema,
            options=LoadOptions(strict=args.strict, resolve_paths=True, normalize_units=True),
        )
    except ConfigError as exc:
        print(f"[ERROR] Config load/validation failed: {exc}")
        return 2
    except Exception as exc:
        print(f"[ERROR] Unexpected error while loading config: {exc}")
        return 3

    # 2) Resolve engine.
    selected_engine = args.engine
    if str(selected_engine).lower().strip() == "auto":
        selected_engine = _auto_select_engine(cfg)

    # 3) Compute stable config hash and seed.
    project_root = Path.cwd()
    cfg_hash = compute_config_hash(cfg, project_root=project_root)

    seed = int(args.seed) if args.seed is not None else _derive_seed(cfg_hash)

    ident = RunIdentity(case_stem=case_path.stem, engine=selected_engine, seed=seed, config_hash=cfg_hash)
    out_dir = (out_base / ident.run_id).resolve()

    # 4) Create output directory (refuse overwrite by default).
    try:
        _ensure_empty_dir(out_dir, overwrite=bool(args.overwrite))
    except Exception as exc:
        print(f"[ERROR] {exc}")
        return 4

    # 5) Write manifest early (before running engines) so partial runs are still traceable.
    extras = {
        "cli_args": vars(args),
        "output_dir": str(Path(args.out) / ident.run_id),  # relative user-facing path
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

    # 6) Run engine and write metrics.
    try:
        metrics = _run_engine(selected_engine, cfg, seed=seed)
        _write_json(out_dir / "metrics.json", metrics)
    except Exception as exc:
        print(f"[ERROR] Engine run failed: {exc}")
        return 6

    # 7) Optional report.
    if args.report:
        try:
            _write_html_report(out_dir=out_dir, metrics=metrics)
        except Exception as exc:
            print(f"[ERROR] Failed to write report.html: {exc}")
            return 7

    print(f"[OK] Case complete: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())