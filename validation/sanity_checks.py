"""
validation/sanity_checks.py

Golden validation harness for the radar_pipeline.

Why this exists
--------------
This repository has three kinds of logic that can silently drift apart over time:

1) Contracts (schemas + loader + CLI strict mode)
   - Ensures "what we think we ran" is exactly what we ran.
   - Prevents schema/$ref/resolver breakages and config/engine mismatches.

2) Deterministic physics/statistics (model_based)
   - Radar equation and receiver noise power must remain physically consistent.
   - Detection math (chi2 / ncx2) must remain correctly wired to thresholds and SNR/SINR.

3) Empirical behavior (monte_carlo + signal_level)
   - CFAR and RD-map generation must execute reproducibly and remain numerically sane.
   - “Golden” Monte Carlo properties guard against regressions that unit tests miss.

What this harness guarantees (v1)
---------------------------------
If `python -m validation.sanity_checks` passes, then:

A) End-to-end contract gates pass
   - cli.run_case runs canonical cases under --strict
   - produced metrics.json includes required contracts:
       * validity (all engines)
       * detection.contract (model_based with detection enabled)
       * sinr_db emission and degradation for jammer demo case

B) Physics/statistics invariants hold
   - Monostatic radar equation scales ~ 1/R^4
   - Pd is monotonic with increasing SNR/SINR (for fixed Pfa and integration settings)

C) Monte Carlo golden behavior remains correct
   - CA-CFAR independent mode under homogeneous exponential noise maintains target Pfa
     within statistically justified tolerance (Wilson 95% CI + 5-sigma bound)

D) Signal-level engine remains usable
   - RD grid shape is correct
   - noise model and RD statistics are finite and sane
   - optional detection outputs are structurally consistent

Non-goals (intentional)
-----------------------
- This is not a full unit test suite replacement.
- This is not a benchmarking tool.
- This does not attempt to validate high-fidelity waveform models.

How to run
----------
Fast (default):
    python -m validation.sanity_checks

Full (slower MC sizes):
    python -m validation.sanity_checks --full

Exit codes
----------
0 : all checks passed
2 : at least one check failed (assertion-style failure)
3 : unexpected exception / error
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from core.simulation.model_based import run_model_based_case
from core.simulation.monte_carlo import run_pfa_monte_carlo


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

    Gates are ordered:
    0) End-to-end contract checks (schema + strict + metrics contracts)
    1) Core math invariants (radar equation scaling)
    2) Pd monotonicity (model-based)
    3) MC golden property (CA-CFAR homogeneous)
    4) MC hetero spot (executes path, record-only)
    5) Signal-level smoke (optional but recommended)
    """
    print(f"[INFO] Running sanity checks (full={full}, seed={seed})")

    # 0) End-to-end CLI contract checks (this is what keeps phase-1 honest)
    _check_end_to_end_case_contracts(seed=seed)

    # 1) In-memory math checks
    _check_radar_equation_scaling()
    _check_pd_monotonicity()

    # Golden MC check sizes
    n_trials_homo = 2_000_000 if full else 200_000
    n_trials_hetero = 2_000_000 if full else 200_000

    _check_cfar_homogeneous_pfa(seed=seed, n_trials=n_trials_homo)
    _spot_cfar_heterogeneous(seed=seed + 1, n_trials=n_trials_hetero)

    # 5) Signal-level smoke/invariants
    _check_signal_level_smoke(seed=seed + 2)


# ---------------------------------------------------------------------
# 0) End-to-end contract checks (CLI + schema + strict)
# ---------------------------------------------------------------------

def _run_cli_case(*, case_path: str, seed: int) -> None:
    """
    Run cli.run_case in strict mode. Raises SanityCheckError on failure.
    """
    cmd = [
        sys.executable,
        "-m",
        "cli.run_case",
        "--case",
        case_path,
        "--engine",
        "auto",
        "--seed",
        str(int(seed)),
        "--overwrite",
        "--strict",
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if proc.returncode != 0:
        raise SanityCheckError(
            "cli.run_case failed.\n"
            f"case={case_path}\n"
            f"exit_code={proc.returncode}\n"
            "----- output -----\n"
            f"{proc.stdout}"
        )


def _find_run_dir(prefix: str) -> Path:
    """
    Find the first run dir under results/cases matching prefix, sorted by name for determinism.
    """
    root = Path("results/cases")
    if not root.exists():
        raise SanityCheckError("results/cases does not exist (cli.run_case did not produce outputs)")
    hits = sorted([x for x in root.iterdir() if x.is_dir() and x.name.startswith(prefix)], key=lambda x: x.name)
    if not hits:
        raise SanityCheckError(f"Missing run dir for prefix: {prefix}")
    return hits[0]


def _load_metrics(run_dir: Path) -> Dict[str, Any]:
    p = run_dir / "metrics.json"
    if not p.exists():
        raise SanityCheckError(f"Missing metrics.json: {p}")
    return json.loads(p.read_text(encoding="utf-8"))


def _check_end_to_end_case_contracts(*, seed: int) -> None:
    """
    End-to-end contract enforcement (Phase 1 gates).

    This ensures:
    - schema resolution works
    - strict validation works
    - engine runs
    - metrics include required contracts (validity + detection contract where applicable)
    """
    # Clean results to avoid picking old runs
    if Path("results").exists():
        # safe, deterministic
        subprocess.run(["rm", "-rf", "results"], check=False)

    # Run the canonical cases
    _run_cli_case(case_path="configs/cases/demo_pd_noise.yaml", seed=seed)
    _run_cli_case(case_path="configs/cases/demo_clutter_cfar.yaml", seed=seed)
    _run_cli_case(case_path="configs/cases/demo_jamming_sinr.yaml", seed=seed)

    # 0a) demo_pd_noise: validity + detection.contract
    run0 = _find_run_dir(f"demo_pd_noise__model_based__seed{seed}__cfg")
    m0 = _load_metrics(run0)
    if "validity" not in m0:
        raise SanityCheckError(f"{run0.name}: metrics.json missing top-level validity")
    det0 = m0.get("detection", None)
    if not isinstance(det0, dict) or "contract" not in det0:
        raise SanityCheckError(f"{run0.name}: metrics.json missing detection.contract (required for model-vs-mc)")
    if not isinstance(det0["contract"], dict):
        raise SanityCheckError(f"{run0.name}: detection.contract must be a dict")

    # 0b) demo_clutter_cfar: validity must exist (monte carlo engine)
    run1 = _find_run_dir(f"demo_clutter_cfar__monte_carlo__seed{seed}__cfg")
    m1 = _load_metrics(run1)
    if "validity" not in m1:
        raise SanityCheckError(f"{run1.name}: metrics.json missing top-level validity")
    if "detector" not in m1:
        raise SanityCheckError(f"{run1.name}: metrics.json missing detector field")

    # 0c) demo_jamming_sinr: sinr_db exists and shows degradation vs snr_db
    run2 = _find_run_dir(f"demo_jamming_sinr__model_based__seed{seed}__cfg")
    m2 = _load_metrics(run2)
    if "snr_db" not in m2 or "sinr_db" not in m2:
        raise SanityCheckError(f"{run2.name}: expected both snr_db and sinr_db in metrics.json")
    snr0 = float(m2["snr_db"][0])
    sinr0 = float(m2["sinr_db"][0])
    if not (sinr0 < snr0):
        raise SanityCheckError(
            f"{run2.name}: expected SINR < SNR when jammer active. snr_db[0]={snr0}, sinr_db[0]={sinr0}"
        )

    print("[PASS] End-to-end strict case contracts (pd_noise + clutter_cfar + jamming_sinr)")


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

    Acceptance:
    - Wilson 95% CI contains the target
    - |p_emp - p_target| within 5-sigma binomial tolerance
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
            f"CA-CFAR homogeneous Pfa failed: target {pfa_target} not in Wilson95 [{low:.6g}, {high:.6g}] "
            f"(empirical={p_emp:.6g}, n_trials={n_trials})"
        )

    sigma = math.sqrt(pfa_target * (1.0 - pfa_target) / n_trials)
    tol = 5.0 * sigma
    if abs(p_emp - pfa_target) > tol:
        raise SanityCheckError(
            f"CA-CFAR homogeneous Pfa too far: emp={p_emp:.6g}, target={pfa_target:.6g}, tol={tol:.3e}"
        )

    print(f"[PASS] CA-CFAR homogeneous Pfa (n_trials={n_trials}, emp={p_emp:.6g})")


# ---------------------------------------------------------------------
# 4) MC spot run: heterogeneous behavior (record-only)
# ---------------------------------------------------------------------

def _make_segments_for_trials(n_trials: int) -> List[Dict[str, Any]]:
    if n_trials <= 0:
        raise SanityCheckError(f"n_trials must be positive, got {n_trials}")

    c1 = int(round(0.25 * n_trials))
    c2 = int(round(0.50 * n_trials))
    c3 = n_trials - c1 - c2  # force exact sum

    if c1 <= 0 or c2 <= 0 or c3 <= 0:
        raise SanityCheckError(f"Invalid segment counts: c1={c1}, c2={c2}, c3={c3}")

    return [
        {"value": 0.7, "count": int(c1)},
        {"value": 1.0, "count": int(c2)},
        {"value": 1.8, "count": int(c3)},
    ]


def _spot_cfar_heterogeneous(*, seed: int, n_trials: int) -> None:
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

    print(f"[PASS] MC spot (hetero executes) (n_trials={n_trials}, emp={p_emp:.6g})")


# ---------------------------------------------------------------------
# 5) Signal-level smoke + invariants
# ---------------------------------------------------------------------

def _check_signal_level_smoke(*, seed: int) -> None:
    """
    Signal-level engine smoke test + invariants.

    If your repo intentionally removes/renames signal_level, fail loudly: this harness
    is meant to catch that regression.
    """
    try:
        from core.simulation.signal_level import run_signal_level_case  # local import for clearer failure
    except Exception as exc:
        raise SanityCheckError(f"signal_level import failed: {exc}")

    cfg = _base_signal_level_cfg()
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
        "scenario": {"range_m": 10_000.0},
    }


if __name__ == "__main__":
    raise SystemExit(main())