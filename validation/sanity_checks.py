"""
validation/sanity_checks.py

Validation harness for the radar pipeline (professional, reproducible sanity checks).

What this module does
---------------------
Provides a small set of deterministic, automation-friendly checks to prevent
silent regressions in core radar performance math.

This is NOT a full test suite replacement.
This is the "golden sanity harness" that must pass before you trust any sweep/report.

Checks included (v1)
--------------------
1) Radar equation scaling sanity:
   - Received power must scale ~ 1/R^4 for monostatic radar equation.
   - We validate the ratio Pr(R1)/Pr(R2) against (R2/R1)^4.

2) Pd monotonicity sanity:
   - For fixed Pfa and integration settings, Pd must be non-decreasing with SNR.
   - This catches broken threshold/statistics wiring.

3) CA-CFAR homogeneous Pfa Monte Carlo (golden property):
   - Under homogeneous exponential background, CA-CFAR (independent mode) should
     maintain the requested Pfa (within statistical tolerance).
   - We verify that:
       - empirical Pfa is close to target (absolute tolerance)
       - target lies within Wilson 95% confidence interval

4) Two Monte Carlo "spot" runs:
   - Homogeneous baseline (fast)
   - Heterogeneous segments (fast) to expose non-homogeneity sensitivity
     (we do NOT require it to match Pfa; we just record behavior)

5) Signal-level engine smoke + invariants (v1):
   - Ensures the signal_level engine runs deterministically and produces a sane RD map summary.
   - Validates core invariants:
       * correct grid dimensions
       * finite, positive noise power model
       * finite RD stats (mean/median/max)
       * injected target bin power is finite
       * optional CFAR outputs are finite and structurally consistent

How to run
----------
Fast (default):
    python -m validation.sanity_checks

Full (larger MC):
    python -m validation.sanity_checks --full

Exit codes
----------
0 : all checks passed
2 : at least one check failed
3 : unexpected error / exception
"""

from __future__ import annotations

import argparse
from typing import Any, Dict, List
import math

import numpy as np

from core.simulation.model_based import run_model_based_case
from core.simulation.monte_carlo import run_pfa_monte_carlo
from core.simulation.signal_level import run_signal_level_case


# ---------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------

class SanityCheckError(AssertionError):
    """Raised when a sanity check fails."""


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="sanity_checks",
        description="Run radar pipeline sanity checks (golden validation harness).",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Run slower checks with larger Monte Carlo sizes.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=123,
        help="Base RNG seed for deterministic validation runs (default: 123).",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------
# Public entrypoint
# ---------------------------------------------------------------------

def main() -> int:
    args = _parse_args()

    try:
        run_all_checks(full=args.full, seed=args.seed)
    except SanityCheckError as exc:
        print(f"[FAIL] {exc}")
        return 2
    except Exception as exc:
        print(f"[ERROR] Unexpected exception: {exc}")
        return 3

    print("[OK] All sanity checks passed.")
    return 0


def run_all_checks(*, full: bool, seed: int) -> None:
    """
    Run the full sanity harness.

    Parameters
    ----------
    full : bool
        If True, run larger Monte Carlo trial counts.
    seed : int
        Base seed for reproducibility.
    """
    print(f"[INFO] Running sanity checks (full={full}, seed={seed})")

    _check_radar_equation_scaling()
    _check_pd_monotonicity()

    # Golden MC check sizes
    n_trials_homo = 2_000_000 if full else 200_000
    n_trials_hetero = 2_000_000 if full else 200_000

    _check_cfar_homogeneous_pfa(seed=seed, n_trials=n_trials_homo)
    _spot_cfar_heterogeneous(seed=seed + 1, n_trials=n_trials_hetero)

    # Signal-level smoke/invariants (keep it fast even in full mode)
    _check_signal_level_smoke(seed=seed + 2)


# ---------------------------------------------------------------------
# 1) Radar equation scaling sanity
# ---------------------------------------------------------------------

def _check_radar_equation_scaling() -> None:
    """
    Validate monostatic radar equation scaling ~ 1/R^4.

    We run model_based at two ranges and compare received power ratio.
    """
    cfg = _base_model_based_cfg()
    cfg["metrics"] = {"ranges_m": [10_000.0, 20_000.0]}  # 10 km, 20 km

    m = run_model_based_case(cfg, seed=None)
    pr_w = m.get("received_power_w", None)
    if not isinstance(pr_w, list) or len(pr_w) != 2:
        raise SanityCheckError("model_based metrics missing received_power_w (expected length 2)")

    pr1 = float(pr_w[0])
    pr2 = float(pr_w[1])
    if pr1 <= 0.0 or pr2 <= 0.0:
        raise SanityCheckError("received power must be positive")

    ratio_emp = pr1 / pr2
    ratio_theory = (20_000.0 / 10_000.0) ** 4  # (R2/R1)^4

    # Allow small floating error; this should be extremely tight.
    rel_err = abs(ratio_emp - ratio_theory) / ratio_theory
    if rel_err > 1e-10:
        raise SanityCheckError(
            f"Radar equation scaling failed: empirical={ratio_emp:.6g}, theory={ratio_theory:.6g}, rel_err={rel_err:.3e}"
        )

    print("[PASS] Radar equation scaling (~1/R^4)")


# ---------------------------------------------------------------------
# 2) Pd monotonicity sanity
# ---------------------------------------------------------------------

def _check_pd_monotonicity() -> None:
    """
    Pd must be non-decreasing with SNR for fixed Pfa and integration settings.

    We force SNR variation via range changes (closer range => higher SNR).
    """
    cfg = _base_model_based_cfg()
    cfg["detection"] = {"pfa": 1e-6, "n_pulses": 16, "integration": "noncoherent"}
    cfg["metrics"] = {"ranges_m": [40_000.0, 30_000.0, 20_000.0, 15_000.0, 10_000.0]}

    m = run_model_based_case(cfg, seed=None)
    det = m.get("detection", None)
    if not isinstance(det, dict) or "pd" not in det:
        raise SanityCheckError("model_based detection output missing Pd array")

    pd = det["pd"]
    if not isinstance(pd, list) or len(pd) < 2:
        raise SanityCheckError("Pd must be a non-empty list")

    pd_arr = np.asarray(pd, dtype=float)
    if np.any(~np.isfinite(pd_arr)):
        raise SanityCheckError("Pd contains non-finite values")

    # Range decreases in our list => SNR increases => Pd should be non-decreasing.
    # Allow tiny numerical noise (strict but safe).
    if np.any(np.diff(pd_arr) < -1e-12):
        raise SanityCheckError(f"Pd is not monotonic non-decreasing with SNR: Pd={pd_arr.tolist()}")

    print("[PASS] Pd monotonicity vs SNR")


# ---------------------------------------------------------------------
# 3) CA-CFAR homogeneous Pfa golden property
# ---------------------------------------------------------------------

def _check_cfar_homogeneous_pfa(*, seed: int, n_trials: int) -> None:
    """
    Golden check:
    CA-CFAR independent mode under homogeneous exponential background should meet target Pfa.

    Acceptance criteria (professional & statistically grounded)
    ----------------------------------------------------------
    - Wilson 95% CI must contain the target Pfa
    - |pfa_emp - pfa_target| must be within an absolute tolerance based on binomial stdev
      (we use 5-sigma as a conservative bound for CI mismatch)
    """
    pfa_target = 1e-3
    n_ref = 32

    cfg: Dict[str, Any] = {
        "monte_carlo": {
            "pfa": pfa_target,
            "n_trials": int(n_trials),
            "detector": "ca_cfar_independent",
            "n_ref": int(n_ref),
            "background": {
                "model": "exponential",
                "mean_power": 1.0,
                "params": {},
                "hetero": {"enabled": False, "mode": "multiply", "mean_multiplier": 1.0},
            },
            "seed": int(seed),
        }
    }

    m = run_pfa_monte_carlo(cfg, seed=seed)

    p_emp = float(m.get("pfa_empirical", float("nan")))
    if not math.isfinite(p_emp):
        raise SanityCheckError("Monte Carlo metrics missing finite pfa_empirical")

    ci = m.get("confidence_intervals", None)
    if not isinstance(ci, dict) or "wilson_95" not in ci:
        raise SanityCheckError("Monte Carlo metrics missing Wilson confidence interval")

    wil = ci["wilson_95"]
    low = float(wil.get("low", float("nan")))
    high = float(wil.get("high", float("nan")))
    if not (math.isfinite(low) and math.isfinite(high)):
        raise SanityCheckError("Wilson CI bounds must be finite")

    if not (low <= pfa_target <= high):
        raise SanityCheckError(
            f"CA-CFAR homogeneous Pfa check failed: target {pfa_target} not in Wilson95 [{low:.6g}, {high:.6g}] "
            f"(empirical={p_emp:.6g}, n_trials={n_trials})"
        )

    # Additional absolute tolerance gate: 5-sigma binomial stdev around target
    sigma = math.sqrt(pfa_target * (1.0 - pfa_target) / n_trials)
    tol = 5.0 * sigma
    if abs(p_emp - pfa_target) > tol:
        raise SanityCheckError(
            f"CA-CFAR homogeneous Pfa too far from target: emp={p_emp:.6g}, target={pfa_target:.6g}, tol={tol:.3e}"
        )

    print(f"[PASS] CA-CFAR homogeneous Pfa (n_trials={n_trials}, emp={p_emp:.6g})")


# ---------------------------------------------------------------------
# 4) MC spot run: heterogeneous behavior (record-only)
# ---------------------------------------------------------------------

def _make_segments_for_trials(n_trials: int) -> List[Dict[str, Any]]:
    """
    Build a deterministic piecewise-constant multiplier pattern whose counts sum to n_trials.

    Design intent
    -------------
    - Keep the same "shape" as the demo: 0.7 / 1.0 / 1.8 with 25% / 50% / 25%.
    - Always satisfy the Monte Carlo contract: sum(counts) == n_trials.
    """
    if n_trials <= 0:
        raise SanityCheckError(f"n_trials must be positive, got {n_trials}")

    c1 = int(round(0.25 * n_trials))
    c2 = int(round(0.50 * n_trials))
    c3 = n_trials - c1 - c2  # force exact sum

    if c1 <= 0 or c2 <= 0 or c3 <= 0:
        raise SanityCheckError(f"Invalid segment counts after scaling: c1={c1}, c2={c2}, c3={c3}")

    return [
        {"value": 0.7, "count": int(c1)},
        {"value": 1.0, "count": int(c2)},
        {"value": 1.8, "count": int(c3)},
    ]


def _spot_cfar_heterogeneous(*, seed: int, n_trials: int) -> None:
    """
    Run a deterministic heterogeneous Monte Carlo "spot" to expose non-homogeneity effects.

    Important
    ---------
    We do NOT enforce that heterogeneous runs meet the target Pfa.
    The point is to produce a reproducible datapoint and ensure the code path works.
    """
    pfa_target = 1e-3
    n_ref = 32

    segments = _make_segments_for_trials(n_trials=n_trials)

    cfg: Dict[str, Any] = {
        "monte_carlo": {
            "pfa": pfa_target,
            "n_trials": int(n_trials),
            "detector": "ca_cfar_independent",
            "n_ref": int(n_ref),
            "background": {
                "model": "exponential",
                "mean_power": 1.0,
                "params": {},
                "hetero": {
                    "enabled": True,
                    "mode": "multiply",
                    "mean_multiplier_segments": segments,
                },
            },
            "seed": int(seed),
        }
    }

    m = run_pfa_monte_carlo(cfg, seed=seed)
    p_emp = float(m.get("pfa_empirical", float("nan")))
    if not math.isfinite(p_emp):
        raise SanityCheckError("Heterogeneous Monte Carlo produced non-finite pfa_empirical")

    print(f"[PASS] MC spot (hetero path executes) (n_trials={n_trials}, emp={p_emp:.6g})")


# ---------------------------------------------------------------------
# 5) Signal-level smoke + invariants
# ---------------------------------------------------------------------

def _check_signal_level_smoke(*, seed: int) -> None:
    """
    Signal-level engine smoke test + invariants (v1).

    This check is intentionally "boring" and strict:
    - It ensures the signal_level engine executes on a valid performance-style config.
    - It validates key invariants that must hold for the RD-map artifact to be trusted.
    """
    cfg = _base_signal_level_cfg()

    # Keep detection enabled so the CFAR code path is exercised.
    cfg["detection"] = {"pfa": 1e-3}

    m = run_signal_level_case(cfg, seed=int(seed))

    if not isinstance(m, dict):
        raise SanityCheckError("signal_level must return a dict metrics object")

    if m.get("engine") != "signal_level":
        raise SanityCheckError(f"signal_level metrics.engine must be 'signal_level', got {m.get('engine')}")

    rd_grid = m.get("rd_grid", None)
    if not isinstance(rd_grid, dict):
        raise SanityCheckError("signal_level metrics missing rd_grid dict")

    n_r = int(rd_grid.get("n_range_bins", -1))
    n_d = int(rd_grid.get("n_doppler_bins", -1))
    if n_r != cfg["geometry"]["n_range_bins"] or n_d != cfg["geometry"]["n_doppler_bins"]:
        raise SanityCheckError(
            f"signal_level rd_grid size mismatch: got ({n_r},{n_d}), "
            f"expected ({cfg['geometry']['n_range_bins']},{cfg['geometry']['n_doppler_bins']})"
        )

    tb = rd_grid.get("target_bin", None)
    if not isinstance(tb, dict) or "r" not in tb or "d" not in tb:
        raise SanityCheckError("signal_level rd_grid missing target_bin {r,d}")

    tr = int(tb["r"])
    td = int(tb["d"])
    if not (0 <= tr < n_r and 0 <= td < n_d):
        raise SanityCheckError(f"signal_level target_bin out of bounds: (r={tr}, d={td}) for grid ({n_r},{n_d})")

    noise = m.get("noise_model", None)
    if not isinstance(noise, dict):
        raise SanityCheckError("signal_level metrics missing noise_model dict")

    noise_power_w = float(noise.get("noise_power_w", float("nan")))
    if not (math.isfinite(noise_power_w) and noise_power_w > 0.0):
        raise SanityCheckError(f"signal_level noise_power_w must be finite and > 0, got {noise_power_w}")

    stats = m.get("rd_power_map_stats", None)
    if not isinstance(stats, dict):
        raise SanityCheckError("signal_level metrics missing rd_power_map_stats dict")

    for k in ("mean", "median", "max", "p90", "p99"):
        v = float(stats.get(k, float("nan")))
        if not math.isfinite(v):
            raise SanityCheckError(f"signal_level rd_power_map_stats.{k} must be finite, got {v}")
        if v < 0.0:
            raise SanityCheckError(f"signal_level rd_power_map_stats.{k} must be >= 0, got {v}")

    inj = m.get("target_injection", None)
    if not isinstance(inj, dict):
        raise SanityCheckError("signal_level metrics missing target_injection dict")

    pr_w = float(inj.get("received_power_w", float("nan")))
    snr_lin = float(inj.get("snr_injected_lin", float("nan")))
    if not (math.isfinite(pr_w) and pr_w >= 0.0):
        raise SanityCheckError(f"signal_level received_power_w must be finite and >= 0, got {pr_w}")
    if not (math.isfinite(snr_lin) and snr_lin >= 0.0):
        raise SanityCheckError(f"signal_level snr_injected_lin must be finite and >= 0, got {snr_lin}")

    # Optional detection output checks (we enabled it in cfg)
    det = m.get("detection", None)
    if not isinstance(det, dict):
        raise SanityCheckError("signal_level detection dict missing (expected when detection.pfa is provided)")

    alpha = float(det.get("alpha", float("nan")))
    if not (math.isfinite(alpha) and alpha > 0.0):
        raise SanityCheckError(f"signal_level detection.alpha must be finite and > 0, got {alpha}")

    td_flag = det.get("target_detected", None)
    if not isinstance(td_flag, bool):
        raise SanityCheckError("signal_level detection.target_detected must be a boolean")

    thr = det.get("target_threshold", None)
    if thr is not None:
        thr_f = float(thr)
        if not math.isfinite(thr_f):
            raise SanityCheckError("signal_level detection.target_threshold must be finite when present")

    print("[PASS] Signal-level engine smoke + invariants")


# ---------------------------------------------------------------------
# Helpers: base configs
# ---------------------------------------------------------------------

def _base_model_based_cfg() -> Dict[str, Any]:
    """
    Minimal deterministic model_based config used by sanity checks.

    Notes
    -----
    Values are chosen to be physically plausible but not tied to a specific radar.
    """
    return {
        "radar": {
            "fc_hz": 10.0e9,
            "tx_power_w": 1.0e3,
            "prf_hz": 1.0e3,
        },
        "antenna": {
            "gain_tx_db": 30.0,
            "gain_rx_db": 30.0,
        },
        "receiver": {
            "bw_hz": 5.0e6,
            "nf_db": 5.0,
            "temperature_k": 290.0,
        },
        "target": {"rcs_sqm": 1.0},
        "environment": {"system_losses_db": 0.0},
        "geometry": {
            "n_range_bins": 256,
            "n_doppler_bins": 16,
            "beams_per_scan": 1,
            "n_cpi_per_dwell": 1,
        },
    }


def _base_signal_level_cfg() -> Dict[str, Any]:
    """
    Minimal deterministic performance-style config for signal_level sanity checks.

    Design intent
    -------------
    - Uses only performance_case sections (no schema changes, no new config files).
    - Geometry is sized to meet CFAR neighborhood assumptions (>= 5x5).
    - Keeps values plausible and deterministic.
    """
    return {
        "radar": {
            "fc_hz": 10.0e9,
            "tx_power_w": 1.0e3,
            "prf_hz": 1.0e3,
            "duty_cycle": 0.1,
        },
        "antenna": {
            "gain_tx_db": 30.0,
            "gain_rx_db": 30.0,
        },
        "receiver": {
            "bw_hz": 5.0e6,
            "nf_db": 5.0,
            "temperature_k": 290.0,
        },
        "target": {"rcs_sqm": 1.0},
        "environment": {"system_losses_db": 0.0},
        "geometry": {
            "n_range_bins": 64,
            "n_doppler_bins": 32,
            "beams_per_scan": 1,
            "n_cpi_per_dwell": 1,
        },
        # Range is only used to compute Pr for the injected target; no bin-mapping is assumed.
        "scenario": {"range_m": 10_000.0},
    }


if __name__ == "__main__":
    raise SystemExit(main())