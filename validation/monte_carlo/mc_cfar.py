"""
validation/monte_carlo/mc_cfar.py

Monte Carlo CFAR validation runner (standalone utility, deterministic, report-friendly).

Purpose
-------
This module provides a small, explicit CLI for running CA-CFAR Monte Carlo experiments
*without* going through the full case runner. It is intended for engineering validation:

- Verify CA-CFAR Pfa control under homogeneous exponential background.
- Characterize sensitivity under heterogeneous backgrounds (piecewise mean multipliers).
- Produce stable JSON artifacts suitable for CI, debugging, and report inclusion.

It is not a replacement for:
- `cli/run_case.py` (pipeline entrypoint)
- `validation/sanity_checks.py` (golden harness)

It exists because it is useful to quickly answer questions like:
- "Does Pfa stay near target for n_ref=16 vs 32 vs 64?"
- "What happens to Pfa under a known heterogeneity pattern?"
- "Do both detector modes run and produce plausible outputs?"

Inputs
------
Command line arguments (see `--help`):
- --pfa: Target Pfa in (0,1)
- --n-trials: Number of Monte Carlo trials
- --detector: "ca_cfar_independent" or "ca_cfar_1d_sliding"
- --n-ref: Number of reference cells (for independent) or reference per-side (depending on core impl)
- --mean-power: Background mean power (>0), canonical noise level
- --hetero: Enable heterogeneous segments (piecewise multipliers)
- --segments: Comma-separated multipliers, e.g. "0.7,1.0,1.8"
- --weights:  Comma-separated weights, e.g. "0.25,0.50,0.25" (must sum to 1, will be normalized)
- --seed: RNG seed (int)
- --out: Optional output JSON path. If omitted, prints JSON to stdout.

Outputs
-------
A JSON object (dict) matching the core Monte Carlo engine output, plus a small wrapper header:
- wrapper: metadata about this run (args, timestamp_utc)
- result:  the dict returned by `core.simulation.monte_carlo.run_pfa_monte_carlo`

Dependencies
------------
- Python 3.10+
- NumPy
- This repository modules:
  - core.simulation.monte_carlo.run_pfa_monte_carlo

Execution
---------
Examples:

Homogeneous CA-CFAR Pfa check:
    python -m validation.monte_carlo.mc_cfar --pfa 1e-3 --n-trials 200000 --detector ca_cfar_independent --n-ref 32 --seed 123

Heterogeneous "spot" characterization (piecewise multipliers):
    python -m validation.monte_carlo.mc_cfar --pfa 1e-3 --n-trials 200000 --detector ca_cfar_independent --n-ref 32 --hetero \
        --segments 0.7,1.0,1.8 --weights 0.25,0.50,0.25 --seed 124

Exit Codes
----------
0 : success
2 : invalid arguments / configuration error
3 : unexpected runtime error

Design Notes
------------
- Deterministic given (args, seed).
- No plotting. No file I/O unless --out is provided.
- Keeps the configuration explicit and schema-compatible with the existing case schema.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import math
from typing import Any, Dict, List, Optional, Tuple

from core.simulation.monte_carlo import run_pfa_monte_carlo


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="mc_cfar",
        description="Standalone Monte Carlo runner for CA-CFAR Pfa validation.",
    )

    p.add_argument("--pfa", type=float, required=True, help="Target Pfa in (0,1). Example: 1e-3")
    p.add_argument("--n-trials", type=int, required=True, help="Number of Monte Carlo trials (>=1).")
    p.add_argument(
        "--detector",
        type=str,
        required=True,
        choices=["ca_cfar_independent", "ca_cfar_1d_sliding"],
        help="Detector mode (must match core engine enum).",
    )
    p.add_argument("--n-ref", type=int, default=32, help="Number of reference cells (>=1). Default: 32")
    p.add_argument("--mean-power", type=float, default=1.0, help="Background mean power (>0). Default: 1.0")
    p.add_argument("--seed", type=int, default=123, help="RNG seed (int). Default: 123")

    p.add_argument(
        "--hetero",
        action="store_true",
        help="Enable heterogeneous piecewise mean multipliers (segments).",
    )
    p.add_argument(
        "--segments",
        type=str,
        default="0.7,1.0,1.8",
        help="Comma-separated multipliers for hetero segments. Default: '0.7,1.0,1.8'",
    )
    p.add_argument(
        "--weights",
        type=str,
        default="0.25,0.50,0.25",
        help="Comma-separated weights for segments. Default: '0.25,0.50,0.25'",
    )

    p.add_argument(
        "--out",
        type=str,
        default=None,
        help="Optional output JSON file path. If omitted, prints JSON to stdout.",
    )

    return p.parse_args()


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _utc_now_str() -> str:
    return _dt.datetime.now(tz=_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse_csv_floats(s: str) -> List[float]:
    items = [x.strip() for x in s.split(",") if x.strip() != ""]
    return [float(x) for x in items]


def _normalize_weights(w: List[float]) -> List[float]:
    if len(w) == 0:
        raise ValueError("weights list must be non-empty")
    if any((not math.isfinite(x) or x < 0.0) for x in w):
        raise ValueError(f"weights must be finite and >= 0, got {w}")
    s = float(sum(w))
    if s <= 0.0:
        raise ValueError(f"weights sum must be > 0, got sum={s}")
    return [float(x / s) for x in w]


def _make_segments(n_trials: int, multipliers: List[float], weights: List[float]) -> List[Dict[str, Any]]:
    if n_trials <= 0:
        raise ValueError(f"n_trials must be >= 1, got {n_trials}")
    if len(multipliers) != len(weights):
        raise ValueError(f"segments and weights must have same length, got {len(multipliers)} vs {len(weights)}")
    if any((not math.isfinite(m) or m <= 0.0) for m in multipliers):
        raise ValueError(f"segment multipliers must be finite and > 0, got {multipliers}")

    w_norm = _normalize_weights(weights)

    # Allocate integer counts deterministically, forcing exact sum = n_trials.
    counts = [int(round(w * n_trials)) for w in w_norm]
    # Fix rounding drift: adjust the last element to force exact sum.
    drift = n_trials - sum(counts)
    counts[-1] += drift

    if any(c <= 0 for c in counts):
        raise ValueError(f"segment counts must all be >= 1; got counts={counts} for n_trials={n_trials}")

    if sum(counts) != n_trials:
        raise ValueError("internal error: segment counts do not sum to n_trials")

    return [{"value": float(m), "count": int(c)} for m, c in zip(multipliers, counts)]


def _build_cfg(args: argparse.Namespace) -> Dict[str, Any]:
    pfa = float(args.pfa)
    if not (0.0 < pfa < 1.0):
        raise ValueError(f"--pfa must be in (0,1), got {pfa}")

    n_trials = int(args.n_trials)
    if n_trials < 1:
        raise ValueError(f"--n-trials must be >= 1, got {n_trials}")

    n_ref = int(args.n_ref)
    if n_ref < 1:
        raise ValueError(f"--n-ref must be >= 1, got {n_ref}")

    mean_power = float(args.mean_power)
    if not (math.isfinite(mean_power) and mean_power > 0.0):
        raise ValueError(f"--mean-power must be finite and > 0, got {mean_power}")

    hetero_enabled = bool(args.hetero)

    hetero: Dict[str, Any] = {"enabled": hetero_enabled, "mode": "multiply"}
    if hetero_enabled:
        multipliers = _parse_csv_floats(str(args.segments))
        weights = _parse_csv_floats(str(args.weights))
        hetero["mean_multiplier_segments"] = _make_segments(n_trials=n_trials, multipliers=multipliers, weights=weights)
    else:
        hetero["mean_multiplier"] = 1.0

    cfg: Dict[str, Any] = {
        "monte_carlo": {
            "pfa": pfa,
            "n_trials": n_trials,
            "detector": str(args.detector),
            "n_ref": n_ref,
            "background": {
                "model": "exponential",
                "mean_power": mean_power,
                "params": {},
                "hetero": hetero,
            },
            "seed": int(args.seed),
        }
    }
    return cfg


def _write_json(path: str, obj: Dict[str, Any]) -> None:
    text = json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text + "\n")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> int:
    args = _parse_args()

    try:
        cfg = _build_cfg(args)
    except Exception as exc:
        print(f"[ERROR] Invalid arguments: {exc}")
        return 2

    try:
        result = run_pfa_monte_carlo(cfg, seed=int(args.seed))
    except Exception as exc:
        print(f"[ERROR] Monte Carlo execution failed: {exc}")
        return 3

    payload: Dict[str, Any] = {
        "wrapper": {
            "tool": "validation.monte_carlo.mc_cfar",
            "timestamp_utc": _utc_now_str(),
            "args": vars(args),
        },
        "result": result,
    }

    if args.out:
        try:
            _write_json(str(args.out), payload)
            print(f"[OK] Wrote: {args.out}")
        except Exception as exc:
            print(f"[ERROR] Failed to write JSON: {exc}")
            return 3
    else:
        print(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())