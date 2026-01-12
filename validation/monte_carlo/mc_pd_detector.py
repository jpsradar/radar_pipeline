"""
validation/monte_carlo/mc_pd_detector.py

Monte Carlo Pd / Pfa validation for the square-law energy detector (standalone utility).

Purpose
-------
This module runs empirical detection experiments for the pipeline's energy detector,
providing a deterministic validation path for:

- Pfa calibration of the threshold (under H0 noise-only).
- Pd vs SNR behavior (under H1 signal+noise) for noncoherent integration.

It is meant to support engineering review and functional validation, complementing:
- `core.detection.thresholds` (threshold calculators)
- `core.simulation.model_based` (closed-form Pd computation)

This tool intentionally avoids "magic":
- It uses explicit noncoherent energy: sum(|z_k|^2) over K pulses (complex baseband samples).
- It models H0 and H1 using complex circular Gaussian noise and a deterministic complex tone
  per pulse (coherent across pulses) with a random phase per trial (to avoid phase-lock artifacts).

Inputs
------
Command line arguments (see `--help`):
- --pfa: Target Pfa in (0,1)
- --n-pulses: Number of integrated pulses (K >= 1)
- --n-trials: Monte Carlo trials (>= 1)
- --snr-db: Comma-separated SNR points in dB for Pd curve (H1). Example: "-10,-5,0,5,10"
- --seed: RNG seed
- --ci: If set, prints Wilson 95% confidence interval for Pfa and Pd estimates.

Noise / Signal Model (explicit)
-------------------------------
Let noise per pulse be z_k ~ CN(0, sigma_n^2) where E[|z_k|^2] = sigma_n^2.
We set sigma_n^2 = 1.0 by default (dimensionless), and enforce SNR via signal power.

For a requested linear SNR = Ps / Pn:
- Pn = E[|z_k|^2] = 1.0
- Choose deterministic signal amplitude A such that |A|^2 = Ps = SNR * Pn

Then per pulse under H1:
  y_k = A * exp(j*phi) + z_k

Noncoherent energy statistic:
  T = sum_{k=1..K} |y_k|^2

Threshold
---------
We compute the energy threshold using the repository's threshold function:
- `core.detection.thresholds.energy_threshold_noncoherent(pfa, n_pulses)`

If that function is absent in your repo version, this script will fail loudly
(and that's correct: you need the threshold API stable for v1).

Outputs
-------
JSON object printed to stdout (or written via --out):
- configuration (args, timestamp_utc)
- threshold used
- Pfa empirical (H0) + optional Wilson CI
- Pd empirical per SNR point + optional Wilson CI per point

Dependencies
------------
- Python 3.10+
- NumPy
- Repository module: core.detection.thresholds

Execution
---------
Example:
    python -m validation.monte_carlo.mc_pd_detector --pfa 1e-6 --n-pulses 16 --n-trials 200000 --snr-db -10,-5,0,5,10 --seed 123 --ci

Exit Codes
----------
0 : success
2 : invalid arguments
3 : runtime error

Design Notes
------------
- Deterministic given (args, seed).
- No plotting, no external I/O unless --out is provided.
- Empirical checks are intentionally strict about numeric finiteness and structure.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from core.detection import thresholds as _thresholds


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    """
    Parse CLI arguments for the Pd Monte Carlo detector validation tool.

    This CLI is designed to be:
    - Scriptable (CI / batch friendly)
    - Explicit (no hidden defaults)
    - Robust to negative SNR values

    Notes on --snr-db
    -----------------
    SNR values are provided as a single comma-separated string.
    Because values may be negative, users should either:
      - use --snr-db="-10,-5,0,5,10"
      - or use --snr-db=<value>

    Examples
    --------
    python -m validation.monte_carlo.mc_pd_detector \\
        --pfa 1e-6 --n-pulses 16 --n-trials 200000 \\
        --snr-db="-10,-5,0,5,10" --seed 123 --ci
    """
    parser = argparse.ArgumentParser(
        prog="mc_pd_detector",
        description=(
            "Monte Carlo validation of detection probability (Pd) versus SNR "
            "for a fixed Pfa and pulse integration setting."
        ),
    )

    parser.add_argument(
        "--pfa",
        type=float,
        required=True,
        help="Target probability of false alarm (e.g. 1e-6).",
    )

    parser.add_argument(
        "--n-pulses",
        type=int,
        required=True,
        help="Number of pulses integrated by the detector.",
    )

    parser.add_argument(
        "--n-trials",
        type=int,
        required=True,
        help="Number of Monte Carlo trials per SNR point.",
    )

    parser.add_argument(
        "--snr-db",
        type=str,
        required=True,
        metavar="SNR_DB",
        help=(
            "Comma-separated list of SNR values in dB. "
            "Example: -10,-5,0,5,10. "
            "If negative values are used, quote the argument or use --snr-db=<value>."
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional RNG seed for reproducibility.",
    )

    parser.add_argument(
        "--ci",
        action="store_true",
        help="If set, compute and report confidence intervals on Pd.",
    )

    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Optional path to write JSON output. If omitted, prints to stdout.",
    )

    args = parser.parse_args()

    # -------------------------------------------------
    # Post-parse validation and normalization
    # -------------------------------------------------
    if not (0.0 < args.pfa < 1.0):
        parser.error(f"--pfa must be in (0,1), got {args.pfa}")

    if args.n_pulses <= 0:
        parser.error(f"--n-pulses must be positive, got {args.n_pulses}")

    if args.n_trials <= 0:
        parser.error(f"--n-trials must be positive, got {args.n_trials}")

    # Parse SNR list explicitly (kept as string in argparse)
    try:
        snr_vals = [float(x) for x in args.snr_db.split(",")]
    except Exception:
        parser.error(f"--snr-db must be a comma-separated list of numbers, got '{args.snr_db}'")

    if len(snr_vals) == 0:
        parser.error("--snr-db must contain at least one SNR value")

    if not all(math.isfinite(x) for x in snr_vals):
        parser.error(f"--snr-db contains non-finite values: {snr_vals}")

    # Attach parsed list back onto args for downstream use
    args.snr_db_list = snr_vals

    return args

# ---------------------------------------------------------------------
# Confidence interval (Wilson)
# ---------------------------------------------------------------------

def wilson_ci_95(k: int, n: int) -> Dict[str, float]:
    if n <= 0:
        raise ValueError("n must be positive for Wilson CI")
    if k < 0 or k > n:
        raise ValueError("k must satisfy 0 <= k <= n for Wilson CI")

    z = 1.959963984540054  # 95% two-sided
    phat = k / n
    denom = 1.0 + (z * z) / n
    center = (phat + (z * z) / (2.0 * n)) / denom
    half = (z / denom) * math.sqrt((phat * (1.0 - phat) / n) + (z * z) / (4.0 * n * n))
    return {"low": float(max(0.0, center - half)), "high": float(min(1.0, center + half))}


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _utc_now_str() -> str:
    return _dt.datetime.now(tz=_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse_csv_floats(s: str) -> List[float]:
    items = [x.strip() for x in s.split(",") if x.strip() != ""]
    return [float(x) for x in items]


def _require_threshold_api() -> Any:
    """
    Require a stable threshold API in core.detection.thresholds.

    Expected function:
        energy_threshold_noncoherent(pfa: float, n_pulses: int) -> float
    """
    fn = getattr(_thresholds, "energy_threshold_noncoherent", None)
    if fn is None or not callable(fn):
        raise RuntimeError(
            "Missing threshold API: core.detection.thresholds.energy_threshold_noncoherent(pfa, n_pulses) is required."
        )
    return fn


def _complex_gaussian(rng: np.random.Generator, shape: Tuple[int, ...], noise_power: float) -> np.ndarray:
    """
    Generate CN(0, noise_power) samples where E[|z|^2] = noise_power.
    """
    if not (math.isfinite(noise_power) and noise_power > 0.0):
        raise ValueError(f"noise_power must be finite and > 0, got {noise_power}")
    sigma2 = noise_power / 2.0
    scale = math.sqrt(sigma2)
    x = rng.normal(0.0, scale, size=shape)
    y = rng.normal(0.0, scale, size=shape)
    return x + 1j * y


def _write_json(path: str, obj: Dict[str, Any]) -> None:
    text = json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text + "\n")


# ---------------------------------------------------------------------
# Core experiment
# ---------------------------------------------------------------------

def run_mc_pd_detector(
    *,
    pfa: float,
    n_pulses: int,
    n_trials: int,
    snr_db: List[float],
    seed: int,
    include_ci: bool,
) -> Dict[str, Any]:
    if not (0.0 < pfa < 1.0):
        raise ValueError(f"pfa must be in (0,1), got {pfa}")
    if n_pulses < 1:
        raise ValueError(f"n_pulses must be >= 1, got {n_pulses}")
    if n_trials < 1:
        raise ValueError(f"n_trials must be >= 1, got {n_trials}")
    if len(snr_db) == 0:
        raise ValueError("snr_db list must be non-empty")

    thr_fn = _require_threshold_api()
    thr = float(thr_fn(float(pfa), int(n_pulses)))
    if not (math.isfinite(thr) and thr > 0.0):
        raise RuntimeError(f"Computed threshold must be finite and > 0, got {thr}")

    rng = np.random.default_rng(int(seed))

    # Noise model: dimensionless power per pulse.
    noise_power = 1.0

    # -------------------------
    # H0: estimate Pfa empirical
    # -------------------------
    z0 = _complex_gaussian(rng, (n_trials, n_pulses), noise_power=noise_power)
    t0 = np.sum(np.abs(z0) ** 2, axis=1)
    k0 = int(np.sum(t0 > thr))
    pfa_emp = float(k0 / n_trials)

    out: Dict[str, Any] = {
        "engine": "mc_pd_detector",
        "threshold": {
            "method": "energy_threshold_noncoherent",
            "pfa_target": float(pfa),
            "n_pulses": int(n_pulses),
            "threshold": float(thr),
            "noise_power_per_pulse": float(noise_power),
        },
        "pfa_h0": {
            "n_trials": int(n_trials),
            "exceedances": int(k0),
            "pfa_empirical": float(pfa_emp),
        },
        "pd_h1": {
            "n_trials": int(n_trials),
            "snr_db": [float(x) for x in snr_db],
            "pd_empirical": [],
        },
    }

    if include_ci:
        out["pfa_h0"]["wilson_95"] = wilson_ci_95(k0, n_trials)

    # -------------------------
    # H1: estimate Pd per SNR
    # -------------------------
    pd_list: List[float] = []
    ci_list: List[Dict[str, float]] = []

    for snr_db_i in snr_db:
        snr_lin = 10.0 ** (float(snr_db_i) / 10.0)
        if not (math.isfinite(snr_lin) and snr_lin >= 0.0):
            raise ValueError(f"Invalid SNR: {snr_db_i} dB")

        # Choose deterministic amplitude such that |A|^2 = Ps = snr_lin * Pn.
        a = math.sqrt(max(0.0, snr_lin * noise_power))

        # Random phase per trial to avoid trivial coherence artifacts.
        phi = rng.uniform(0.0, 2.0 * math.pi, size=(n_trials, 1))
        s = a * (np.cos(phi) + 1j * np.sin(phi))  # (n_trials,1)

        z1 = _complex_gaussian(rng, (n_trials, n_pulses), noise_power=noise_power)
        y = z1 + s  # broadcast over pulses
        t1 = np.sum(np.abs(y) ** 2, axis=1)

        k1 = int(np.sum(t1 > thr))
        pd_emp = float(k1 / n_trials)

        pd_list.append(pd_emp)
        if include_ci:
            ci_list.append(wilson_ci_95(k1, n_trials))

    out["pd_h1"]["pd_empirical"] = [float(x) for x in pd_list]
    if include_ci:
        out["pd_h1"]["wilson_95"] = ci_list

    return out


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> int:
    args = _parse_args()

    try:
        snr_db = _parse_csv_floats(str(args.snr_db))
        payload = run_mc_pd_detector(
            pfa=float(args.pfa),
            n_pulses=int(args.n_pulses),
            n_trials=int(args.n_trials),
            snr_db=snr_db,
            seed=int(args.seed),
            include_ci=bool(args.ci),
        )
    except Exception as exc:
        print(f"[ERROR] {exc}")
        return 2

    wrapper = {
        "wrapper": {
            "tool": "validation.monte_carlo.mc_pd_detector",
            "timestamp_utc": _utc_now_str(),
            "args": vars(args),
        },
        "result": payload,
    }

    if args.out:
        try:
            _write_json(str(args.out), wrapper)
            print(f"[OK] Wrote: {args.out}")
        except Exception as exc:
            print(f"[ERROR] Failed to write JSON: {exc}")
            return 3
    else:
        print(json.dumps(wrapper, indent=2, sort_keys=True, ensure_ascii=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())