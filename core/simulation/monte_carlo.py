"""
core/simulation/monte_carlo.py

Monte Carlo simulation utilities for radar performance validation.

What this module does
---------------------
Provides reproducible Monte Carlo runners to estimate:
- Empirical Pfa for a chosen detector (e.g., CA-CFAR) under specified background models
- Confidence intervals for report-ready validation

Primary use case (LinkedIn / notebook demo)
-------------------------------------------
- Show that CA-CFAR maintains target Pfa under:
  (a) homogeneous exponential noise
  (b) heterogeneous clutter (mean scaling map / segments)
- Show CFAR brittleness under non-homogeneous backgrounds

Inputs (cfg dict - recommended fields)
--------------------------------------
cfg["monte_carlo"]:
  - n_trials: int (number of trials)
  - pfa: float (target false alarm probability)
  - detector: "ca_cfar_independent" | "ca_cfar_1d_sliding"
  - n_ref: int (for independent CA-CFAR)
  - n_train: int, n_guard: int, n_cells: int (for 1d sliding CA-CFAR)
  - background:
      - model: "exponential" | "weibull" | "lognormal" | "k"
      - params: dict (model-specific parameters)
      - mean_power: float (optional; used by exponential/k helper paths)
      - hetero:
          - enabled: bool
          - mode: "multiply" (v1)
          - mean_multiplier: float (scalar scale)
          - mean_multiplier_segments: list[{value: float, count: int}] (compact per-trial pattern)

Outputs
-------
A metrics dict suitable to be written as metrics.json:
- pfa_target, pfa_empirical
- n_trials, n_false_alarms
- confidence intervals (normal approx and Wilson score)
- detector metadata (alpha, n_ref / n_train/n_guard)
- background metadata (model name, parameters, heterogeneity)

CLI usage
---------
Called by cli/run_case.py when engine == "monte_carlo" or when auto-detected.
Example:
    python -m cli.run_case --case configs/cases/demo_clutter_cfar.yaml --seed 123
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union, List
import math

import numpy as np

from core.environment.clutter_models import (
    ExponentialPower,
    WeibullPower,
    LognormalPower,
    KPower,
    apply_mean_scaling,
    k_from_mean_and_shape,
)
from core.detection.cfar import ca_cfar_alpha
from core.detection.detectors import detect_ca_cfar_1d_sliding


class MonteCarloError(ValueError):
    """Raised when Monte Carlo configuration is invalid."""


# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------

def run_pfa_monte_carlo(cfg: Dict[str, Any], seed: Optional[int] = None) -> Dict[str, Any]:
    """
    Estimate empirical Pfa for a detector under a specified background model.

    Parameters
    ----------
    cfg : dict
        Case configuration dictionary.
    seed : int | None
        RNG seed for reproducibility. If cfg["monte_carlo"]["seed"] exists, it overrides this.

    Returns
    -------
    dict
        Metrics dictionary with empirical Pfa and confidence intervals.
    """
    mc = _require_section(cfg, "monte_carlo")

    rng_seed = mc.get("seed", seed)
    rng = np.random.default_rng(None if rng_seed is None else int(rng_seed))

    pfa_target = _require_prob(mc, "pfa")
    n_trials = _require_int(mc, "n_trials", min_value=1, default=100_000)

    detector = str(mc.get("detector", "ca_cfar_independent")).lower().strip()
    bg = _require_section(mc, "background")

    dist = _build_background_distribution(bg)

    # --- Heterogeneity parsing (professional: compact patterns supported) ---
    hetero = bg.get("hetero", None)
    mean_multiplier: Optional[Union[float, np.ndarray]] = None
    if isinstance(hetero, dict) and bool(hetero.get("enabled", False)):
        mode = str(hetero.get("mode", "multiply")).lower().strip()
        if mode != "multiply":
            raise MonteCarloError("background.hetero.mode must be 'multiply' (v1)")

        if "mean_multiplier_segments" in hetero:
            mean_multiplier = _expand_mean_multiplier_segments(
                segments=hetero["mean_multiplier_segments"],
                n_trials=n_trials,
            )
        elif "mean_multiplier" in hetero:
            mean_multiplier = float(hetero["mean_multiplier"])
        else:
            raise MonteCarloError(
                "background.hetero.enabled=true requires mean_multiplier or mean_multiplier_segments"
            )

    if detector == "ca_cfar_independent":
        n_ref = _require_int(mc, "n_ref", min_value=1, default=32)
        metrics = _run_independent_ca_cfar_pfa(
            rng=rng,
            dist=dist,
            pfa=pfa_target,
            n_trials=n_trials,
            n_ref=n_ref,
            mean_multiplier=mean_multiplier,
        )
        return _decorate_metrics(metrics, cfg=cfg)

    if detector == "ca_cfar_1d_sliding":
        n_train = _require_int(mc, "n_train", min_value=1, default=16)
        n_guard = _require_int(mc, "n_guard", min_value=0, default=0)
        n_cells = _require_int(mc, "n_cells", min_value=(2 * n_train + 2 * n_guard + 3), default=512)

        metrics = _run_sliding_ca_cfar_pfa(
            rng=rng,
            dist=dist,
            pfa=pfa_target,
            n_trials=n_trials,
            n_cells=n_cells,
            n_train=n_train,
            n_guard=n_guard,
            mean_multiplier=mean_multiplier,
        )
        return _decorate_metrics(metrics, cfg=cfg)

    raise MonteCarloError(f"Unknown detector: {detector}")


# ---------------------------------------------------------------------
# Independent CA-CFAR Pfa estimation (fast, vectorized)
# ---------------------------------------------------------------------

def _run_independent_ca_cfar_pfa(
    *,
    rng: np.random.Generator,
    dist: Any,
    pfa: float,
    n_trials: int,
    n_ref: int,
    mean_multiplier: Optional[Union[float, np.ndarray]],
) -> Dict[str, Any]:
    """
    Independent-trial CA-CFAR Pfa estimation.

    Each trial draws:
    - CUT power ~ background distribution
    - Nref reference powers ~ background distribution
    Then tests: CUT > alpha * mean(ref)

    This isolates CA-CFAR thresholding math from window correlation effects.
    """
    alpha = ca_cfar_alpha(pfa=pfa, n_ref=n_ref)

    x_cut = dist.sample(rng, size=n_trials)                 # shape: (n_trials,)
    x_ref = dist.sample(rng, size=(n_trials, n_ref))        # shape: (n_trials, n_ref)

    if mean_multiplier is not None:
        # Apply heterogeneity as multiplicative mean scaling.
        # - scalar: global scale
        # - vector length n_trials: per-trial scale; broadcast to refs
        m = np.asarray(mean_multiplier, dtype=float)
        x_cut = apply_mean_scaling(x_cut, m)
        if m.ndim == 1 and m.size == n_trials:
            x_ref = apply_mean_scaling(x_ref, m.reshape(-1, 1))
        else:
            x_ref = apply_mean_scaling(x_ref, m)

    z = np.mean(x_ref, axis=1)
    thr = alpha * z
    fa = x_cut > thr

    n_fa = int(np.sum(fa))
    pfa_emp = float(n_fa / n_trials)

    ci_norm = _ci_normal_approx(p=pfa_emp, n=n_trials, z=1.96)
    ci_wilson = _ci_wilson(p=pfa_emp, n=n_trials, z=1.96)

    return {
        "task": "pfa_monte_carlo",
        "detector": "ca_cfar_independent",
        "pfa_target": float(pfa),
        "pfa_empirical": pfa_emp,
        "n_trials": int(n_trials),
        "n_false_alarms": n_fa,
        "alpha": float(alpha),
        "n_ref": int(n_ref),
        "confidence_intervals": {
            "normal_95": {"low": ci_norm[0], "high": ci_norm[1]},
            "wilson_95": {"low": ci_wilson[0], "high": ci_wilson[1]},
        },
    }


# ---------------------------------------------------------------------
# Sliding 1D CA-CFAR Pfa estimation (includes overlap/edges)
# ---------------------------------------------------------------------

def _run_sliding_ca_cfar_pfa(
    *,
    rng: np.random.Generator,
    dist: Any,
    pfa: float,
    n_trials: int,
    n_cells: int,
    n_train: int,
    n_guard: int,
    mean_multiplier: Optional[Union[float, np.ndarray]],
) -> Dict[str, Any]:
    """
    Sliding-window CA-CFAR Pfa estimation.

    For each trial:
    - Generate a 1D power line x of length n_cells
    - Apply CA-CFAR sliding detector
    - Compute false alarms only where threshold is defined (non-edge CUTs)

    Notes
    -----
    This is closer to real use (RD line), but includes correlation due to window overlap.
    """
    alpha = ca_cfar_alpha(pfa=pfa, n_ref=2 * n_train)

    total_cuts = 0
    total_fa = 0

    for _ in range(n_trials):
        x = dist.sample(rng, size=n_cells)

        if mean_multiplier is not None:
            # For sliding mode, heterogeneity must be broadcastable to (n_cells,).
            # - scalar: ok
            # - vector: must match n_cells
            m = np.asarray(mean_multiplier, dtype=float)
            x = apply_mean_scaling(x, m)

        res = detect_ca_cfar_1d_sliding(x_power=x, pfa=pfa, n_train=n_train, n_guard=n_guard)
        valid = np.isfinite(res.threshold)
        total_cuts += int(np.sum(valid))
        total_fa += int(np.sum(res.detections & valid))

    pfa_emp = float(total_fa / max(total_cuts, 1))

    ci_norm = _ci_normal_approx(p=pfa_emp, n=total_cuts, z=1.96)
    ci_wilson = _ci_wilson(p=pfa_emp, n=total_cuts, z=1.96)

    return {
        "task": "pfa_monte_carlo",
        "detector": "ca_cfar_1d_sliding",
        "pfa_target": float(pfa),
        "pfa_empirical": pfa_emp,
        "n_trials": int(n_trials),
        "n_cells": int(n_cells),
        "n_train": int(n_train),
        "n_guard": int(n_guard),
        "alpha": float(alpha),
        "n_cuts_evaluated": int(total_cuts),
        "n_false_alarms": int(total_fa),
        "confidence_intervals": {
            "normal_95": {"low": ci_norm[0], "high": ci_norm[1]},
            "wilson_95": {"low": ci_wilson[0], "high": ci_wilson[1]},
        },
    }


# ---------------------------------------------------------------------
# Heterogeneity helpers
# ---------------------------------------------------------------------

def _expand_mean_multiplier_segments(
    *,
    segments: Any,
    n_trials: int,
) -> np.ndarray:
    """
    Expand a compact heterogeneity definition into a per-trial multiplier vector.

    Expected format
    ---------------
    segments: list of dicts, each with:
      - value: float > 0
      - count: int >= 1

    The sum of all counts MUST equal n_trials.

    Returns
    -------
    np.ndarray
        Vector of length n_trials (float), suitable for per-trial scaling.
    """
    if not isinstance(segments, list) or len(segments) == 0:
        raise MonteCarloError("mean_multiplier_segments must be a non-empty list")

    values: List[np.ndarray] = []
    total = 0

    for seg in segments:
        if not isinstance(seg, dict):
            raise MonteCarloError(f"Invalid hetero segment (must be dict): {seg}")

        try:
            v = float(seg["value"])
            c = int(seg["count"])
        except Exception as exc:
            raise MonteCarloError(f"Invalid hetero segment: {seg}") from exc

        if not math.isfinite(v) or v <= 0.0 or c <= 0:
            raise MonteCarloError(f"Invalid hetero segment values: {seg}")

        values.append(np.full(c, v, dtype=float))
        total += c

    if total != n_trials:
        raise MonteCarloError(
            f"Sum of hetero segment counts ({total}) != n_trials ({n_trials})"
        )

    return np.concatenate(values)


# ---------------------------------------------------------------------
# Background distribution factory
# ---------------------------------------------------------------------

def _build_background_distribution(bg: Dict[str, Any]) -> Any:
    """
    Build a clutter/noise power distribution from config.

    Supported models
    ----------------
    - exponential: mean_power (from bg.mean_power or params.mean_power, default 1.0)
    - weibull: params.shape_k, params.scale_lam
    - lognormal: params.mu_ln, params.sigma_ln
    - k: params.shape_v and either params.scale_theta or mean_power
    """
    model = str(bg.get("model", "exponential")).lower().strip()
    params = bg.get("params", {})
    if params is None:
        params = {}
    if not isinstance(params, dict):
        raise MonteCarloError("background.params must be a dict")

    if model == "exponential":
        mean_power = bg.get("mean_power", params.get("mean_power", 1.0))
        return ExponentialPower(mean_power=float(mean_power))

    if model == "weibull":
        k = params.get("shape_k", None)
        lam = params.get("scale_lam", None)
        if k is None or lam is None:
            raise MonteCarloError("weibull requires params: shape_k, scale_lam")
        return WeibullPower(shape_k=float(k), scale_lam=float(lam))

    if model == "lognormal":
        mu = params.get("mu_ln", None)
        sig = params.get("sigma_ln", None)
        if mu is None or sig is None:
            raise MonteCarloError("lognormal requires params: mu_ln, sigma_ln")
        return LognormalPower(mu_ln=float(mu), sigma_ln=float(sig))

    if model == "k":
        v = params.get("shape_v", None)
        if v is None:
            raise MonteCarloError("k requires params: shape_v, and either scale_theta or mean_power")
        if "scale_theta" in params:
            return KPower(shape_v=float(v), scale_theta=float(params["scale_theta"]))
        mean_power = bg.get("mean_power", params.get("mean_power", None))
        if mean_power is None:
            raise MonteCarloError("k requires either params.scale_theta or mean_power")
        return k_from_mean_and_shape(mean_power=float(mean_power), shape_v=float(v))

    raise MonteCarloError(f"Unsupported background model: {model}")


# ---------------------------------------------------------------------
# Confidence intervals
# ---------------------------------------------------------------------

def _ci_normal_approx(*, p: float, n: int, z: float) -> Tuple[float, float]:
    """
    Normal approximation CI for a binomial proportion.

    Notes
    -----
    This is OK for large n, but poor for tiny p. Wilson is included for robustness.
    """
    if n <= 0:
        return (0.0, 1.0)
    p = float(p)
    se = math.sqrt(max(p * (1.0 - p), 0.0) / n)
    return (max(0.0, p - z * se), min(1.0, p + z * se))


def _ci_wilson(*, p: float, n: int, z: float) -> Tuple[float, float]:
    """
    Wilson score interval for a binomial proportion (better for small p).
    """
    if n <= 0:
        return (0.0, 1.0)
    p = float(p)
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p + z2 / (2.0 * n)) / denom
    half = (z / denom) * math.sqrt((p * (1.0 - p) / n) + (z2 / (4.0 * n * n)))
    return (max(0.0, center - half), min(1.0, center + half))


# ---------------------------------------------------------------------
# Metrics decoration / traceability
# ---------------------------------------------------------------------

def _decorate_metrics(metrics: Dict[str, Any], *, cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Attach traceability fields so notebooks/reports can show "what was run".

    Important
    ---------
    This function is intentionally a shallow decorator:
    - It does not rewrite the core metrics
    - It only adds background/hetero metadata for reporting
    """
    mc = cfg.get("monte_carlo", {})
    bg = mc.get("background", {}) if isinstance(mc, dict) else {}

    metrics_out = dict(metrics)
    metrics_out["background"] = {
        "model": bg.get("model", "exponential"),
        "mean_power": bg.get("mean_power", None),
        "params": bg.get("params", {}),
        "hetero": bg.get("hetero", None),
    }
    return metrics_out


# ---------------------------------------------------------------------
# Config parsing helpers
# ---------------------------------------------------------------------

def _require_section(d: Dict[str, Any], key: str) -> Dict[str, Any]:
    sec = d.get(key, None)
    if not isinstance(sec, dict):
        raise MonteCarloError(f"Missing or invalid section '{key}' (must be a dict)")
    return sec


def _require_prob(d: Dict[str, Any], key: str) -> float:
    v = d.get(key, None)
    if v is None:
        raise MonteCarloError(f"Missing required probability '{key}'")
    try:
        p = float(v)
    except Exception as exc:
        raise MonteCarloError(f"'{key}' must be numeric, got {v}") from exc
    if not math.isfinite(p) or not (0.0 < p < 1.0):
        raise MonteCarloError(f"'{key}' must be in (0,1), got {p}")
    return p


def _require_int(d: Dict[str, Any], key: str, *, min_value: int, default: Optional[int] = None) -> int:
    if key not in d:
        if default is None:
            raise MonteCarloError(f"Missing required integer '{key}'")
        x = int(default)
    else:
        try:
            x = int(d[key])
        except Exception as exc:
            raise MonteCarloError(f"'{key}' must be an integer, got {d[key]}") from exc

    if x < min_value:
        raise MonteCarloError(f"'{key}' must be >= {min_value}, got {x}")
    return x