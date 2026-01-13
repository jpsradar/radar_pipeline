"""
core/simulation/monte_carlo.py

Monte Carlo simulation utilities for radar performance validation.

Purpose
-------
This module provides reproducible Monte Carlo estimators for false-alarm behavior
(Pfa / FAR surrogates) under different background power statistics and detector choices.

It exists to make "model vs experiment" arguments defensible:
- We do not only compute theoretical CFAR scalings.
- We also estimate empirical Pfa and confidence intervals, producing artifacts that
  can be shown in reports and compared across commits (regression friendly).

Key behaviors (v2)
------------------
1) Detector choices (current)
   - ca_cfar_independent:
       Independent trials. Each trial draws:
         CUT power ~ background distribution
         Nref reference powers ~ background distribution
       Then tests: CUT > alpha * mean(ref)
       This isolates the CA-CFAR math from sliding-window correlation effects.
   - ca_cfar_1d_sliding:
       Sliding-window CA-CFAR over a 1D power line. More realistic but correlated.

2) Background models (power domain)
   - exponential, weibull, lognormal, k
   All are interpreted as distributions of POWER (nonnegative scalars).

3) Heterogeneity (IMPORTANT CONCEPT)
   Heterogeneity is a mechanism to intentionally violate CFAR assumptions.

   In v1, heterogeneity was implemented as a multiplicative scaling applied to BOTH
   CUT and reference samples in each trial. This is scale-invariant for CA-CFAR and
   therefore DOES NOT break CFAR (it should not change Pfa).

   In v2, we add an explicit control:
     background.hetero.apply_to: "both" | "cut" | "ref"

   - "both" (default, backward compatible): scale CUT and references together
     -> preserves scale invariance -> CFAR still holds.
   - "cut": scale CUT only (references unchanged)
     -> violates homogeneity assumption -> CFAR breaks (Pfa drifts).
   - "ref": scale references only (CUT unchanged)
     -> also violates assumptions -> CFAR breaks.

   For ca_cfar_1d_sliding, "cut/ref" separation is not meaningful because CUT and
   references come from the same line sample set; we therefore only allow "both".

Outputs
-------
Returns a dict suitable to be written as metrics.json. It includes:
- pfa_target, pfa_empirical
- counts, confidence intervals
- detector metadata (alpha, n_ref / n_train/n_guard)
- background metadata (model name, parameters, hetero definition)
- validity contract (top-level "validity") to document model assumptions/limits

"""

from __future__ import annotations

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
        Case configuration dictionary (validated upstream by schema/loader).
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

    # --- Heterogeneity parsing (v2: apply_to support) ---
    hetero = bg.get("hetero", None)

    mean_multiplier: Optional[Union[float, np.ndarray]] = None
    apply_to = "both"  # default for backward compatibility

    if isinstance(hetero, dict) and bool(hetero.get("enabled", False)):
        mode = str(hetero.get("mode", "multiply")).lower().strip()
        if mode != "multiply":
            raise MonteCarloError("background.hetero.mode must be 'multiply' (v1)")

        # v2 extension: where to apply the scaling (cut/ref/both)
        apply_to = str(hetero.get("apply_to", "both")).lower().strip()
        if apply_to not in {"both", "cut", "ref"}:
            raise MonteCarloError("background.hetero.apply_to must be one of: both | cut | ref")

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

    # Detector routing
    if detector == "ca_cfar_independent":
        n_ref = _require_int(mc, "n_ref", min_value=1, default=32)
        metrics = _run_independent_ca_cfar_pfa(
            rng=rng,
            dist=dist,
            pfa=pfa_target,
            n_trials=n_trials,
            n_ref=n_ref,
            mean_multiplier=mean_multiplier,
            hetero_apply_to=apply_to,
        )
        return _decorate_metrics(metrics, cfg=cfg)

    if detector == "ca_cfar_1d_sliding":
        # For sliding mode, apply_to separation is not meaningful: all samples live on one line.
        # We accept only "both" to avoid implying behavior we cannot implement.
        if apply_to != "both":
            raise MonteCarloError(
                "background.hetero.apply_to must be 'both' for detector=ca_cfar_1d_sliding"
            )

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
    hetero_apply_to: str,
) -> Dict[str, Any]:
    """
    Independent-trial CA-CFAR Pfa estimation.

    Each trial draws:
    - CUT power ~ background distribution
    - Nref reference powers ~ background distribution
    Then tests: CUT > alpha * mean(ref)

    Heterogeneity semantics
    ----------------------
    If mean_multiplier is provided, it is applied according to hetero_apply_to:

    - both (default): scale CUT and refs together (scale invariant for CA-CFAR)
    - cut:  scale CUT only  (violates homogeneity assumption -> CFAR breaks)
    - ref:  scale refs only  (violates homogeneity assumption -> CFAR breaks)
    """
    alpha = ca_cfar_alpha(pfa=pfa, n_ref=n_ref)

    x_cut = dist.sample(rng, size=n_trials)          # shape: (n_trials,)
    x_ref = dist.sample(rng, size=(n_trials, n_ref)) # shape: (n_trials, n_ref)

    if mean_multiplier is not None:
        m = np.asarray(mean_multiplier, dtype=float)

        # CUT scaling
        if hetero_apply_to in {"both", "cut"}:
            x_cut = apply_mean_scaling(x_cut, m)

        # REF scaling
        if hetero_apply_to in {"both", "ref"}:
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
    Attach traceability + validity fields so reports can show:
    - what was run (background, detector)
    - under what modeling assumptions/limits (validity contract)
    """
    mc = cfg.get("monte_carlo", {})
    bg = mc.get("background", {}) if isinstance(mc, dict) else {}

    # Detector name must be forwarded explicitly because validity_for_monte_carlo
    # requires it as a keyword-only argument.
    detector = None
    if isinstance(metrics, dict):
        detector = metrics.get("detector", None)
    if detector is None and isinstance(mc, dict):
        detector = mc.get("detector", None)
    detector = str(detector) if detector is not None else "unknown"

    from core.contracts.validity import validity_for_monte_carlo  # type: ignore

    metrics_out = dict(metrics)

    metrics_out["background"] = {
        "model": bg.get("model", "exponential"),
        "mean_power": bg.get("mean_power", None),
        "params": bg.get("params", {}),
        "hetero": bg.get("hetero", None),
    }

    # New: validity contract (required across engines)
    metrics_out["validity"] = validity_for_monte_carlo(cfg, detector=detector)

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