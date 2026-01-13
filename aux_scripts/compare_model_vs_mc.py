"""
aux_scripts/compare_model_vs_mc.py

Model vs Monte Carlo comparison runner (no notebooks).

Engineering intent
------------------
This script produces a small, reviewable artifact that answers:

  "If I ask for Pfa = X in the model, what FAR do I *actually* observe empirically?"

It compares:
- Model-based FAR per second (derived from configured Pfa + RD grid evaluation rate)
- Monte Carlo empirical Pfa (with Wilson 95% CI), converted to FAR per second using the
  same cell-evaluation rate from the model run.

Why this matters
----------------
- Pfa is a per-test probability; FAR is a system-level rate.
- Even if Pfa is "held fixed", the observed FAR depends on how often the detector is evaluated.
- This is a clean model-vs-experiment story and is easy to regression-check.

Outputs
-------
Writes into:
  results/comparisons/<run_id>/
    - comparison.csv
    - comparison.json

Usage
-----
python aux_scripts/compare_model_vs_mc.py \
  --case configs/cases/demo_pd_noise.yaml \
  --seed 123 \
  --strict \
  --overwrite

Notes
-----
- Requires the case to be a performance_case (radar/antenna/receiver/target present).
- The Monte Carlo experiment here is focused on *false alarms* (Pfa/FAR), not Pd.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


from core.config.loaders import ConfigError, LoadOptions, load_case
from core.config.manifest import compute_config_hash


@dataclass(frozen=True)
class CompareIdentity:
    """Immutable identity for a comparison run (directory naming + traceability)."""

    case_stem: str
    seed: int
    config_hash: str

    @property
    def short_hash(self) -> str:
        return self.config_hash[:8]

    @property
    def run_id(self) -> str:
        return f"{self.case_stem}__model_vs_mc__seed{self.seed}__cfg{self.short_hash}"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="compare_model_vs_mc",
        description="Compare model_based FAR vs Monte Carlo empirical Pfa (converted to FAR/s).",
    )
    p.add_argument("--case", required=True, help="Base performance case YAML/JSON.")
    p.add_argument("--seed", type=int, default=123, help="Deterministic seed tag (default: 123).")

    # Monte Carlo controls (kept here to avoid editing configs for quick iteration)
    p.add_argument("--mc-trials", type=int, default=200_000, help="Monte Carlo trials (default: 200000).")
    p.add_argument("--mc-n-ref", type=int, default=32, help="CA-CFAR Nref for independent MC (default: 32).")

    p.add_argument("--out", default="results/comparisons", help="Base output folder (default: results/comparisons).")
    p.add_argument("--schema-dir", default="configs/schemas", help="Schema folder (default: configs/schemas).")
    p.add_argument("--schema", default="case.schema.json", help="Schema filename (default: case.schema.json).")
    p.add_argument("--strict", action="store_true", help="Fail on unknown config fields.")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing comparison directory.")
    return p.parse_args()


def _ensure_dir(path: Path, *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"Output directory already exists: {path}\n"
            "Refusing to overwrite. Use --overwrite for local iteration."
        )
    path.mkdir(parents=True, exist_ok=True)


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def _deepcopy_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    # Simple, dependable deep copy for config dicts.
    return json.loads(json.dumps(cfg))


def _pretty_path(path: Path, project_root: Path) -> str:
    try:
        rel = path.resolve().relative_to(project_root.resolve())
        return str(Path("${PROJECT_ROOT}") / rel)
    except Exception:
        return path.name


def _extract_detection_pfa(cfg: Dict[str, Any]) -> float:
    det = cfg.get("detection", None)
    if not isinstance(det, dict) or "pfa" not in det:
        raise ValueError("Base case must define detection.pfa (required for FAR comparison).")
    return float(det["pfa"])


def _extract_model_far_per_second(metrics: Dict[str, Any]) -> Optional[float]:
    far = metrics.get("far", None)
    if not isinstance(far, dict):
        return None
    v = far.get("per_second", None)
    return None if v is None else float(v)


def _extract_cells_per_second(metrics: Dict[str, Any]) -> Optional[float]:
    geom = metrics.get("geometry", None)
    if not isinstance(geom, dict):
        return None

    cpis = geom.get("cpis_per_second", None)
    cells = geom.get("cells_per_cpi", None)
    if cpis is None or cells is None:
        return None
    try:
        return float(cpis) * float(cells)
    except Exception:
        return None


def _extract_wilson_ci(metrics: Dict[str, Any]) -> Optional[Tuple[float, float]]:
    ci = metrics.get("confidence_intervals", None)
    if not isinstance(ci, dict):
        return None
    w = ci.get("wilson_95", None)
    if not isinstance(w, dict):
        return None
    lo = w.get("low", None)
    hi = w.get("high", None)
    if lo is None or hi is None:
        return None
    return (float(lo), float(hi))


def _run_model_based(cfg: Dict[str, Any], seed: int) -> Dict[str, Any]:
    from core.simulation.model_based import run_model_based_case  # type: ignore

    return run_model_based_case(cfg=cfg, seed=seed)


def _run_mc_pfa(cfg_monte_carlo_case: Dict[str, Any], seed: int) -> Dict[str, Any]:
    from core.simulation.monte_carlo import run_pfa_monte_carlo  # type: ignore

    return run_pfa_monte_carlo(cfg=cfg_monte_carlo_case, seed=seed)


def main() -> int:
    args = _parse_args()
    project_root = Path.cwd().resolve()

    case_path = Path(args.case)

    try:
        cfg_base = load_case(
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

    # Force model_based for the model run (explicit, avoids "auto" ambiguity)
    cfg_model = _deepcopy_cfg(cfg_base)
    cfg_model.setdefault("execution", {})
    if isinstance(cfg_model["execution"], dict):
        cfg_model["execution"]["mode"] = "model_based"

    try:
        pfa_target = _extract_detection_pfa(cfg_model)
    except Exception as exc:
        print(f"[ERROR] {exc}")
        return 4

    # Stable identity from the normalized base cfg
    cfg_hash = compute_config_hash(cfg_model, project_root=project_root)
    ident = CompareIdentity(case_stem=case_path.stem, seed=int(args.seed), config_hash=cfg_hash)

    out_dir = (Path(args.out) / ident.run_id).resolve()
    try:
        _ensure_dir(out_dir, overwrite=bool(args.overwrite))
    except Exception as exc:
        msg = str(exc).replace(str(project_root), "${PROJECT_ROOT}")
        print(f"[ERROR] {msg}")
        return 5

    # 1) Model run
    try:
        m_model = _run_model_based(cfg_model, seed=int(args.seed))
    except Exception as exc:
        print(f"[ERROR] model_based run failed: {exc}")
        return 6

    far_model = _extract_model_far_per_second(m_model)
    cells_per_second = _extract_cells_per_second(m_model)

    if cells_per_second is None:
        print("[ERROR] Could not extract cells_per_second from model metrics.geometry.")
        return 7

    # 2) Monte Carlo Pfa run (independent CA-CFAR by default)
    cfg_mc = {
        "execution": {"mode": "monte_carlo"},
        "assumptions": [
            "Independent trials Monte Carlo: CUT and reference samples are IID under the declared background model.",
            "This experiment estimates empirical Pfa (with Wilson 95% CI); it does not estimate Pd.",
        ],
        "monte_carlo": {
            "seed": int(args.seed),
            "pfa": float(pfa_target),
            "n_trials": int(args.mc_trials),
            "detector": "ca_cfar_independent",
            "n_ref": int(args.mc_n_ref),
            "background": {
                "model": "exponential",
                "mean_power": 1.0,
                "params": {},
                "hetero": {"enabled": False},
            },
        },
    }

    try:
        m_mc = _run_mc_pfa(cfg_mc, seed=int(args.seed))
    except Exception as exc:
        print(f"[ERROR] monte_carlo run failed: {exc}")
        return 8

    pfa_emp = float(m_mc.get("pfa_empirical", float("nan")))
    far_emp = pfa_emp * float(cells_per_second)
    wilson = _extract_wilson_ci(m_mc)

    # Build comparison table
    rows = [
        {
            "metric": "pfa_per_test",
            "model": float(pfa_target),
            "monte_carlo": float(pfa_emp),
            "delta": float(pfa_emp - float(pfa_target)),
        },
        {
            "metric": "far_per_second",
            "model": None if far_model is None else float(far_model),
            "monte_carlo": float(far_emp),
            "delta": None if far_model is None else float(far_emp - float(far_model)),
        },
    ]

    # Write CSV
    csv_path = out_dir / "comparison.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["metric", "model", "monte_carlo", "delta"])
        w.writeheader()
        w.writerows(rows)

    # Write JSON
    payload = {
        "run_id": ident.run_id,
        "base_case": str(case_path),
        "seed": int(args.seed),
        "pfa_target": float(pfa_target),
        "model": {
            "far_per_second": None if far_model is None else float(far_model),
            "cells_per_second": float(cells_per_second),
        },
        "monte_carlo": {
            "n_trials": int(args.mc_trials),
            "n_ref": int(args.mc_n_ref),
            "pfa_empirical": float(pfa_emp),
            "far_per_second": float(far_emp),
            "wilson_95": None if wilson is None else {"low": wilson[0], "high": wilson[1]},
        },
        "table": rows,
    }
    _write_json(out_dir / "comparison.json", payload)

    print(f"[OK] Comparison complete: {_pretty_path(out_dir, project_root)}")
    print("[OK] Artifacts: comparison.csv, comparison.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())