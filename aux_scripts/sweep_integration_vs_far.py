"""
aux_scripts/sweep_integration_vs_far.py

Reproducible trade-off sweep (NO notebooks):
Integration (N pulses) vs system false-alarm rate (FAR) and Pd.

Why this script exists
----------------------
This repo requires trade-offs to be demonstrated in a way that is:
- deterministic (seeded, commit-stable)
- automation-friendly (single command, artifacts written to disk)
- reportable (CSV + PNG plot + JSON summary)
- independent of notebook environments

This sweep targets a concrete engineering tension:
- Increasing noncoherent integration pulses N tends to increase detection performance (Pd)
  because effective SNR improves.
- But increasing N also increases CPI duration, which reduces CPIs per second.
  If Pfa per RD-cell is held fixed, system-level FAR per second should decrease
  because the detector is evaluated fewer times per second.

What the script does
--------------------
1) Loads a base case YAML via the existing loader + schema (strict, normalized).
2) Forces execution.mode = "model_based" and keeps the RD grid fixed.
3) Sweeps detection.n_pulses over a list (default: [1, 2, 4, 8, 16, 32]).
4) Runs the model_based engine for each N and extracts:
   - FAR per second (metrics["far"]["per_second"])          [system-level proxy]
   - Pd at the first range point (metrics["detection"]["pd"][0])
   - CPI duration and CPIs/sec (metrics["geometry"])
5) Writes artifacts into results/sweeps/<run_id>/:
   - sweep.csv       (table for diffing / spreadsheets)
   - sweep.json      (full extracted arrays + run metadata)
   - sweep.png       (simple plot: Pd and FAR/s vs N, separate y-axes)

Usage
-----
python aux_scripts/sweep_integration_vs_far.py \
  --case configs/cases/demo_pd_noise.yaml \
  --seed 123 \
  --out results/sweeps \
  --overwrite \
  --strict

Notes
-----
- The base case must be a valid "performance_case" (has radar/antenna/receiver/target).
- This script intentionally does NOT depend on cli/run_sweep.py to avoid format drift.
- Plot styling uses matplotlib defaults (no explicit colors), per project conventions.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np

from core.config.loaders import ConfigError, LoadOptions, load_case
from core.config.manifest import compute_config_hash


@dataclass(frozen=True)
class SweepIdentity:
    """Immutable identity for a sweep run; used for directory naming and traceability."""

    case_stem: str
    seed: int
    config_hash: str

    @property
    def short_hash(self) -> str:
        return self.config_hash[:8]

    @property
    def run_id(self) -> str:
        return f"{self.case_stem}__integration_vs_far__seed{self.seed}__cfg{self.short_hash}"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="sweep_integration_vs_far",
        description="Trade-off sweep: integration pulses N vs FAR/s and Pd (model_based).",
    )
    p.add_argument("--case", required=True, help="Base case YAML/JSON (performance_case).")
    p.add_argument("--seed", type=int, default=123, help="Deterministic seed tag (default: 123).")
    p.add_argument(
        "--n-pulses",
        default="1,2,4,8,16,32",
        help="Comma-separated list of N values (default: 1,2,4,8,16,32).",
    )
    p.add_argument("--out", default="results/sweeps", help="Base output folder (default: results/sweeps).")
    p.add_argument("--schema-dir", default="configs/schemas", help="Schema folder (default: configs/schemas).")
    p.add_argument("--schema", default="case.schema.json", help="Schema filename (default: case.schema.json).")
    p.add_argument("--strict", action="store_true", help="Fail on unknown config fields.")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing sweep directory.")
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
    """
    Dependable deep copy for config dicts.

    We avoid copy.deepcopy() because configs can contain Path-like objects in some pipelines.
    JSON round-trip ensures we only keep JSON-serializable primitives (as intended by schema).
    """
    return json.loads(json.dumps(cfg))


def _run_model_based(cfg: Dict[str, Any], seed: int) -> Dict[str, Any]:
    from core.simulation.model_based import run_model_based_case  # type: ignore

    return run_model_based_case(cfg=cfg, seed=seed)


def _extract_first_pd(metrics: Dict[str, Any]) -> Optional[float]:
    det = metrics.get("detection", None)
    if not isinstance(det, dict):
        return None
    pd = det.get("pd", None)
    if not isinstance(pd, list) or len(pd) == 0:
        return None
    try:
        return float(pd[0])
    except Exception:
        return None


def _extract_far_per_second(metrics: Dict[str, Any]) -> Optional[float]:
    far = metrics.get("far", None)
    if not isinstance(far, dict):
        return None
    x = far.get("per_second", None)
    if x is None:
        return None
    try:
        return float(x)
    except Exception:
        return None


def _extract_geometry(metrics: Dict[str, Any]) -> Dict[str, Optional[float]]:
    g = metrics.get("geometry", None)
    if not isinstance(g, dict):
        return {"cpi_duration_s": None, "cpis_per_second": None, "cells_per_cpi": None}

    def f(key: str) -> Optional[float]:
        v = g.get(key, None)
        if v is None:
            return None
        try:
            return float(v)
        except Exception:
            return None

    return {
        "cpi_duration_s": f("cpi_duration_s"),
        "cpis_per_second": f("cpis_per_second"),
        "cells_per_cpi": f("cells_per_cpi"),
    }


def _plot(out_png: Path, n_list: List[int], pd_list: List[Optional[float]], far_list: List[Optional[float]]) -> None:
    """
    Plot integration trade-offs without dual-axis ambiguity.

    The first panel shows normalized trends for comparison. The lower panels
    preserve the physical units for Pd and FAR separately.
    """
    n = np.asarray(n_list, dtype=float)
    pd = np.array([np.nan if v is None else float(v) for v in pd_list], dtype=float)
    far = np.array([np.nan if v is None else float(v) for v in far_list], dtype=float)

    pd_norm = pd / max(float(np.nanmax(pd)), 1e-12)
    far_norm = far / max(float(np.nanmax(far)), 1e-12)

    fig, (ax_norm, ax_pd, ax_far) = plt.subplots(
        3,
        1,
        sharex=True,
        figsize=(9.5, 8.0),
        gridspec_kw={"height_ratios": [1.35, 1.0, 1.0]},
    )

    ax_norm.plot(n, pd_norm, marker="o", linewidth=2.0, label="Pd normalized")
    ax_norm.plot(n, far_norm, marker="s", linewidth=2.0, label="FAR/s normalized")
    ax_norm.set_title("Radar System Trade-off: Detection vs False-Alarm Load")
    ax_norm.set_ylabel("Normalized metric")
    ax_norm.set_ylim(-0.03, 1.08)
    ax_norm.grid(True, which="both", linestyle="--", alpha=0.35)
    ax_norm.legend(loc="best")

    ax_pd.plot(n, np.maximum(pd, 1e-12), marker="o", linewidth=2.0)
    ax_pd.set_yscale("log")
    ax_pd.set_ylabel("Pd")
    ax_pd.set_title("Detection probability at first range point")
    ax_pd.grid(True, which="both", linestyle="--", alpha=0.35)

    ax_far.plot(n, np.maximum(far, 1e-12), marker="s", linewidth=2.0)
    ax_far.set_yscale("log")
    ax_far.set_ylabel("FAR / second")
    ax_far.set_xlabel("N pulses (noncoherent integration)")
    ax_far.set_title("Operational false-alarm load")
    ax_far.grid(True, which="both", linestyle="--", alpha=0.35)

    ax_far.set_xscale("log", base=2)
    ax_far.set_xticks(n_list)
    ax_far.get_xaxis().set_major_formatter(plt.ScalarFormatter())

    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def main() -> int:
    args = _parse_args()
    project_root = Path.cwd().resolve()

    # Parse the N list deterministically from CLI input
    try:
        n_list = [int(x.strip()) for x in str(args.n_pulses).split(",") if x.strip()]
    except Exception:
        print("[ERROR] Invalid --n-pulses. Expected comma-separated integers.")
        return 2
    if not n_list or any(n <= 0 for n in n_list):
        print("[ERROR] All N values must be >= 1, and list must be non-empty.")
        return 2

    case_path = Path(args.case)

    # Load and strictly validate the base case via the canonical loader + schema
    try:
        cfg_base = load_case(
            case_path,
            schema_dir=args.schema_dir,
            schema_name=args.schema,
            options=LoadOptions(strict=bool(args.strict), resolve_paths=True, normalize_units=True),
        )
    except ConfigError as exc:
        print(f"[ERROR] Config load/validation failed: {exc}")
        return 3
    except Exception as exc:
        print(f"[ERROR] Unexpected error while loading config: {exc}")
        return 4

    # Force model_based execution (this trade-off is defined by model_based FAR conversion)
    cfg_base = _deepcopy_cfg(cfg_base)
    cfg_base.setdefault("execution", {})
    if isinstance(cfg_base["execution"], dict):
        cfg_base["execution"]["mode"] = "model_based"

    # Detection must exist and must define pfa so FAR and Pd can be produced
    cfg_base.setdefault("detection", {})
    if not isinstance(cfg_base["detection"], dict):
        print("[ERROR] cfg['detection'] must be a dict.")
        return 5
    if "pfa" not in cfg_base["detection"]:
        print("[ERROR] Base case must provide detection.pfa for this sweep.")
        return 5

    # Normalize integration mode (defensive)
    integ = str(cfg_base["detection"].get("integration", "noncoherent")).lower().strip()
    cfg_base["detection"]["integration"] = integ

    # Compute a stable hash for sweep identity.
    # We intentionally incorporate the N-list into the config hash so two sweeps with
    # different N grids do not collide in run_id.
    cfg_for_hash = _deepcopy_cfg(cfg_base)
    cfg_for_hash["_sweep_intent"] = {"n_pulses": n_list}
    cfg_hash = compute_config_hash(cfg_for_hash, project_root=project_root)

    ident = SweepIdentity(case_stem=case_path.stem, seed=int(args.seed), config_hash=cfg_hash)

    out_base = Path(args.out)
    out_dir = (out_base / ident.run_id).resolve()

    try:
        _ensure_dir(out_dir, overwrite=bool(args.overwrite))
    except Exception as exc:
        msg = str(exc).replace(str(project_root), "${PROJECT_ROOT}")
        print(f"[ERROR] {msg}")
        return 6

    rows: List[Dict[str, Any]] = []
    pd_list: List[Optional[float]] = []
    far_list: List[Optional[float]] = []

    for n_pulses in n_list:
        cfg = _deepcopy_cfg(cfg_base)
        cfg.setdefault("detection", {})
        cfg["detection"]["n_pulses"] = int(n_pulses)
        cfg["detection"]["integration"] = str(cfg["detection"].get("integration", "noncoherent")).lower().strip()

        m = _run_model_based(cfg, seed=int(args.seed))

        pd0 = _extract_first_pd(m)
        far_s = _extract_far_per_second(m)
        geom = _extract_geometry(m)

        pd_list.append(pd0)
        far_list.append(far_s)

        rows.append(
            {
                "n_pulses": int(n_pulses),
                "pd_at_first_range": pd0,
                "far_per_second": far_s,
                "cpi_duration_s": geom["cpi_duration_s"],
                "cpis_per_second": geom["cpis_per_second"],
                "cells_per_cpi": geom["cells_per_cpi"],
            }
        )

        print(f"[OK] N={n_pulses:<3d}  Pd@R0={pd0!s:<12}  FAR/s={far_s!s:<12}  CPI={geom['cpi_duration_s']}")

    if not rows:
        print("[ERROR] No sweep rows produced (unexpected).")
        return 7

    # Write CSV (stable column order)
    csv_path = out_dir / "sweep.csv"
    fieldnames = ["n_pulses", "pd_at_first_range", "far_per_second", "cpi_duration_s", "cpis_per_second", "cells_per_cpi"]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    # Write JSON summary (includes full extracted rows + intent metadata)
    summary = {
        "run_id": ident.run_id,
        "seed": int(args.seed),
        "base_case": str(case_path),
        "intent": {"n_pulses": n_list, "engine": "model_based"},
        "rows": rows,
    }
    _write_json(out_dir / "sweep.json", summary)

    # Plot
    png_path = out_dir / "sweep.png"
    _plot(png_path, n_list=n_list, pd_list=pd_list, far_list=far_list)

    # Print a path without leaking absolute FS locations
    try:
        rel = out_dir.resolve().relative_to(project_root.resolve())
        pretty = str(Path("${PROJECT_ROOT}") / rel)
    except Exception:
        pretty = out_dir.name

    print(f"[OK] Sweep complete: {pretty}")
    print("[OK] Artifacts: sweep.csv, sweep.json, sweep.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())