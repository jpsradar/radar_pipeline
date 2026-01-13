"""
core/simulation/monte_carlo.py

Monte Carlo simulation utilities for radar performance validation.

Purpose
-------
This module provides reproducible Monte Carlo estimators for detection/false-alarm
behavior under different background power statistics and detector choices.

Design intent (repo-level professionalism)
------------------------------------------
- Model vs experiment is explicit: analytic scalings exist, but empirical Monte Carlo
  is used to validate assumptions and expose failure modes.
- Outputs are traceable: detector/background metadata + confidence intervals + validity.

Supported tasks
---------------
monte_carlo.task (default: "pfa"):
- "pfa"     : estimate empirical Pfa under background-only conditions
- "pd"      : estimate empirical Pd under background + target (Swerling 0/1/2/3/4)
- "pfa_pd"  : run both and return a combined metrics dict

Detectors (independent-trials)
------------------------------
- ca_cfar_independent:
    CUT and Nref references sampled iid per trial, threshold is:
      thr = alpha(pfa, n_ref) * mean(ref)
- os_cfar_independent:
    CUT and Nref references sampled iid per trial, threshold is:
      thr = alpha * OS_k(ref)
    alpha can be user-provided or MC-calibrated under the declared homogeneous background model.

Sliding CA-CFAR
---------------
- ca_cfar_1d_sliding exists for Pfa-only realism checks.
  Pd is intentionally NOT implemented in sliding mode (would require a structured
  target injection model along the line and careful counting contracts).

Background models (power domain)
--------------------------------
- exponential, weibull, lognormal, k
All interpreted as distributions of POWER (nonnegative scalars).

Heterogeneity
-------------
background.hetero.apply_to: "both" | "cut" | "ref"
- "both" preserves scale invariance (CFAR should hold)
- "cut"/"ref" intentionally violates homogeneity (CFAR breaks)

Target fluctuation (Pd task)
----------------------------
monte_carlo.pd.swerling:
- "swerling0": nonfluctuating (constant within and across CPI)
- "swerling1": slow fluctuation (one exponential power multiplier per trial)
- "swerling2": fast fluctuation (exponential multiplier per pulse)
- "swerling3": slow fluctuation (gamma(k=2) multiplier per trial)
- "swerling4": fast fluctuation (gamma(k=2) multiplier per pulse)

Integration for Pd task
-----------------------
Power-domain noncoherent integration:
- For each pulse: sample background power for CUT/refs
- Inject target power into CUT only
- Sum powers across N pulses
- Apply CFAR thresholding to the integrated CUT and integrated refs

Coherent integration is NOT represented in this power-domain MC module.
If requested, we fail loudly.

Outputs
-------
All estimators return:
- point estimate (pfa_empirical or pd_empirical)
- binomial confidence intervals (normal_95 + wilson_95)
- detector + background metadata
- validity contract at top-level "validity" (decorated)
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from core.detection.cfar import ca_cfar_alpha
from core.detection.detectors import detect_ca_cfar_1d_sliding
from core.environment.clutter_models import (
    ExponentialPower,
    KPower,
    LognormalPower,
    WeibullPower,
    apply_mean_scaling,
    k_from_mean_and_shape,
)


class MonteCarloError(ValueError):
    """Raised when Monte Carlo configuration is invalid."""


# ---------------------------------------------------------------------
# Public API (unified entrypoint)
# ---------------------------------------------------------------------


def run_monte_carlo(cfg: Dict[str, Any], seed: Optional[int] = None) -> Dict[str, Any]:
    """
    Unified Monte Carlo entrypoint.

    Backward compatibility
    ----------------------
    If monte_carlo.task is absent, defaults to "pfa" (legacy behavior).
    """
    mc = _require_section(cfg, "monte_carlo")
    task = str(mc.get("task", "pfa")).lower().strip()

    if task == "pfa":
        return run_pfa_monte_carlo(cfg=cfg, seed=seed)

    if task == "pd":
        metrics = run_pd_monte_carlo(cfg=cfg, seed=seed)
        return metrics

    if task == "pfa_pd":
        pfa_metrics = run_pfa_monte_carlo(cfg=cfg, seed=seed)
        pd_metrics = run_pd_monte_carlo(cfg=cfg, seed=seed)

        out: Dict[str, Any] = {
            "task": "pfa_pd_monte_carlo",
            "pfa": pfa_metrics,
            "pd": pd_metrics,
        }
        return _decorate_metrics(out, cfg=cfg)

    raise MonteCarloError("monte_carlo.task must be one of: pfa | pd | pfa_pd")


# ---------------------------------------------------------------------
# Legacy public API (kept stable)
# ---------------------------------------------------------------------


def run_pfa_monte_carlo(cfg: Dict[str, Any], seed: Optional[int] = None) -> Dict[str, Any]:
    """
    Estimate empirical Pfa for a detector under a specified background model.

    Notes
    -----
    This is the legacy entrypoint used by older cases; it remains stable.
    """
    mc = _require_section(cfg, "monte_carlo")

    rng_seed = mc.get("seed", seed)
    rng = np.random.default_rng(None if rng_seed is None else int(rng_seed))

    pfa_target = _require_prob(mc, "pfa")
    n_trials = _require_int(mc, "n_trials", min_value=1, default=100_000)

    detector = str(mc.get("detector", "ca_cfar_independent")).lower().strip()
    bg = _require_section(mc, "background")
    dist = _build_background_distribution(bg)

    mean_multiplier, apply_to = _parse_hetero(bg=bg, n_trials=n_trials)

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
        if apply_to != "both":
            raise MonteCarloError("background.hetero.apply_to must be 'both' for detector=ca_cfar_1d_sliding")

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

    if detector == "os_cfar_independent":
        n_ref = _require_int(mc, "n_ref", min_value=1, default=32)
        k = _resolve_os_rank(n_ref=n_ref, rank_k=mc.get("rank_k", None), rank_frac=mc.get("rank_frac", None))

        alpha, alpha_cal = _resolve_or_calibrate_os_alpha(
            mc=mc,
            dist=dist,
            pfa_target=pfa_target,
            n_ref=n_ref,
            rank_k=k,
            rng_seed=rng_seed,
        )

        metrics = _run_independent_os_cfar_pfa(
            rng=rng,
            dist=dist,
            pfa=pfa_target,
            n_trials=n_trials,
            n_ref=n_ref,
            rank_k=k,
            alpha=alpha,
            mean_multiplier=mean_multiplier,
            hetero_apply_to=apply_to,
        )
        metrics["alpha_calibration"] = alpha_cal
        return _decorate_metrics(metrics, cfg=cfg)

    raise MonteCarloError(f"Unknown detector: {detector}")


def run_pd_monte_carlo(cfg: Dict[str, Any], seed: Optional[int] = None) -> Dict[str, Any]:
    """
    Estimate empirical Pd for an independent-trials CFAR detector in noise/clutter
    with an injected target governed by a Swerling fluctuation model.

    Requirements
    ------------
    - monte_carlo.pfa must be present (threshold scaling contract).
    - monte_carlo.detector must be ca_cfar_independent or os_cfar_independent.
    - monte_carlo.pd must exist and provide:
        snr_db, n_pulses, integration, swerling
    - background.mean_power must be present (used to convert SNR to mean target power).
      (For non-exponential clutter models, this is still a reasonable "power scale" contract.)
    """
    mc = _require_section(cfg, "monte_carlo")

    rng_seed = mc.get("seed", seed)
    rng = np.random.default_rng(None if rng_seed is None else int(rng_seed))

    pfa_target = _require_prob(mc, "pfa")
    n_trials = _require_int(mc, "n_trials", min_value=1, default=100_000)

    detector = str(mc.get("detector", "ca_cfar_independent")).lower().strip()
    if detector not in {"ca_cfar_independent", "os_cfar_independent"}:
        raise MonteCarloError("Pd task supports only: ca_cfar_independent | os_cfar_independent")

    bg = _require_section(mc, "background")
    dist = _build_background_distribution(bg)

    # Use mean_power as the physical "noise/clutter power scale" for SNR conversion.
    if "mean_power" not in bg:
        raise MonteCarloError("Pd task requires background.mean_power (power scale for SNR conversion)")
    mean_power = float(bg["mean_power"])
    if not (math.isfinite(mean_power) and mean_power > 0.0):
        raise MonteCarloError(f"background.mean_power must be > 0, got {mean_power}")

    mean_multiplier, apply_to = _parse_hetero(bg=bg, n_trials=n_trials)

    pd_cfg = _require_section(mc, "pd")
    snr_db = _require_number(pd_cfg, "snr_db")
    snr_lin = 10.0 ** (float(snr_db) / 10.0)

    n_pulses = _require_int(pd_cfg, "n_pulses", min_value=1, default=1)
    integration = str(pd_cfg.get("integration", "noncoherent")).lower().strip()
    if integration != "noncoherent":
        raise MonteCarloError("Pd task supports only power-domain noncoherent integration")

    swerling = str(pd_cfg.get("swerling", "swerling1")).lower().strip()
    if swerling not in {"swerling0", "swerling1", "swerling2", "swerling3", "swerling4"}:
        raise MonteCarloError("pd.swerling must be one of: swerling0|swerling1|swerling2|swerling3|swerling4")

    # Detector setup
    n_ref = _require_int(mc, "n_ref", min_value=1, default=32)

    if detector == "ca_cfar_independent":
        alpha = ca_cfar_alpha(pfa=pfa_target, n_ref=n_ref)
        alpha_meta: Dict[str, Any] = {"method": "analytic_ca_cfar_alpha"}
        k_rank = None
    else:
        k_rank = _resolve_os_rank(n_ref=n_ref, rank_k=mc.get("rank_k", None), rank_frac=mc.get("rank_frac", None))
        alpha, alpha_meta = _resolve_or_calibrate_os_alpha(
            mc=mc,
            dist=dist,
            pfa_target=pfa_target,
            n_ref=n_ref,
            rank_k=k_rank,
            rng_seed=rng_seed,
        )

    # Run Pd experiment
    pd_emp, n_det, ci_norm, ci_wilson = _run_independent_cfar_pd(
        rng=rng,
        dist=dist,
        detector=detector,
        pfa=pfa_target,
        n_trials=n_trials,
        n_ref=n_ref,
        alpha=float(alpha),
        os_rank_k=None if k_rank is None else int(k_rank),
        mean_multiplier=mean_multiplier,
        hetero_apply_to=apply_to,
        n_pulses=int(n_pulses),
        swerling=swerling,
        mean_power=mean_power,
        snr_lin=float(snr_lin),
    )

    out: Dict[str, Any] = {
        "task": "pd_monte_carlo",
        "detector": detector,
        "pfa_target": float(pfa_target),
        "pd_empirical": float(pd_emp),
        "n_trials": int(n_trials),
        "n_detections": int(n_det),
        "n_ref": int(n_ref),
        "alpha": float(alpha),
        "alpha_meta": alpha_meta,
        "pd_config": {
            "snr_db": float(snr_db),
            "n_pulses": int(n_pulses),
            "integration": "noncoherent",
            "swerling": swerling,
        },
        "confidence_intervals": {
            "normal_95": {"low": ci_norm[0], "high": ci_norm[1]},
            "wilson_95": {"low": ci_wilson[0], "high": ci_wilson[1]},
        },
    }
    if k_rank is not None:
        out["rank_k"] = int(k_rank)

    return _decorate_metrics(out, cfg=cfg)


# ---------------------------------------------------------------------
# Internal: heterogeneity parsing
# ---------------------------------------------------------------------


def _parse_hetero(*, bg: Dict[str, Any], n_trials: int) -> Tuple[Optional[Union[float, np.ndarray]], str]:
    hetero = bg.get("hetero", None)

    mean_multiplier: Optional[Union[float, np.ndarray]] = None
    apply_to = "both"

    if isinstance(hetero, dict) and bool(hetero.get("enabled", False)):
        mode = str(hetero.get("mode", "multiply")).lower().strip()
        if mode != "multiply":
            raise MonteCarloError("background.hetero.mode must be 'multiply'")

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

    return mean_multiplier, apply_to


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
    alpha = ca_cfar_alpha(pfa=pfa, n_ref=n_ref)

    x_cut = dist.sample(rng, size=n_trials)
    x_ref = dist.sample(rng, size=(n_trials, n_ref))

    if mean_multiplier is not None:
        m = np.asarray(mean_multiplier, dtype=float)

        if hetero_apply_to in {"both", "cut"}:
            x_cut = apply_mean_scaling(x_cut, m)

        if hetero_apply_to in {"both", "ref"}:
            if m.ndim == 1 and m.size == n_trials:
                x_ref = apply_mean_scaling(x_ref, m.reshape(-1, 1))
            else:
                x_ref = apply_mean_scaling(x_ref, m)

    thr = float(alpha) * np.mean(x_ref, axis=1)
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
# Sliding 1D CA-CFAR Pfa estimation
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
    alpha = ca_cfar_alpha(pfa=pfa, n_ref=2 * n_train)

    total_cuts = 0
    total_fa = 0

    for _ in range(n_trials):
        x = dist.sample(rng, size=n_cells)

        if mean_multiplier is not None:
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
# OS-CFAR helpers + estimation
# ---------------------------------------------------------------------


def _kth_order_statistic(x: np.ndarray, *, k: int) -> np.ndarray:
    if x.ndim != 2:
        raise MonteCarloError(f"_kth_order_statistic expects 2D array, got shape={x.shape}")
    n_ref = x.shape[1]
    if not (1 <= k <= n_ref):
        raise MonteCarloError(f"k must satisfy 1 <= k <= n_ref (n_ref={n_ref}), got {k}")
    idx = k - 1
    return np.partition(x, idx, axis=1)[:, idx]


def _resolve_os_rank(*, n_ref: int, rank_k: Any, rank_frac: Any) -> int:
    if rank_k is not None and rank_frac is not None:
        raise MonteCarloError("OS-CFAR: provide only one of rank_k or rank_frac")

    if rank_k is None and rank_frac is None:
        rank_frac = 0.75

    if rank_k is not None:
        try:
            k = int(rank_k)
        except Exception as exc:
            raise MonteCarloError(f"OS-CFAR: rank_k must be int, got {rank_k}") from exc
    else:
        try:
            rf = float(rank_frac)
        except Exception as exc:
            raise MonteCarloError(f"OS-CFAR: rank_frac must be numeric, got {rank_frac}") from exc
        if not (0.0 < rf <= 1.0):
            raise MonteCarloError(f"OS-CFAR: rank_frac must be in (0,1], got {rf}")
        k = int(math.ceil(rf * n_ref))

    if not (1 <= k <= n_ref):
        raise MonteCarloError(f"OS-CFAR: rank_k must be in [1,{n_ref}], got {k}")
    return k


def _resolve_or_calibrate_os_alpha(
    *,
    mc: Dict[str, Any],
    dist: Any,
    pfa_target: float,
    n_ref: int,
    rank_k: int,
    rng_seed: Optional[int],
) -> Tuple[float, Dict[str, Any]]:
    alpha_override = mc.get("alpha", None)
    if alpha_override is not None:
        try:
            alpha = float(alpha_override)
        except Exception as exc:
            raise MonteCarloError(f"alpha must be numeric, got {alpha_override}") from exc
        if not (math.isfinite(alpha) and alpha > 0.0):
            raise MonteCarloError(f"alpha must be finite and > 0, got {alpha}")
        return float(alpha), {"method": "user_override"}

    n_cal = _require_int(mc, "alpha_calibration_n", min_value=10_000, default=200_000)
    cal_seed = mc.get("alpha_calibration_seed", rng_seed)
    alpha, alpha_cal = _calibrate_os_cfar_alpha(
        dist=dist,
        pfa_target=pfa_target,
        n_ref=n_ref,
        rank_k=rank_k,
        n_cal=n_cal,
        seed=None if cal_seed is None else int(cal_seed),
    )
    return float(alpha), alpha_cal


def _run_independent_os_cfar_pfa(
    *,
    rng: np.random.Generator,
    dist: Any,
    pfa: float,
    n_trials: int,
    n_ref: int,
    rank_k: int,
    alpha: float,
    mean_multiplier: Optional[Union[float, np.ndarray]],
    hetero_apply_to: str,
) -> Dict[str, Any]:
    if not (math.isfinite(alpha) and alpha > 0.0):
        raise MonteCarloError(f"alpha must be finite and > 0, got {alpha}")

    x_cut = dist.sample(rng, size=n_trials)
    x_ref = dist.sample(rng, size=(n_trials, n_ref))

    if mean_multiplier is not None:
        m = np.asarray(mean_multiplier, dtype=float)

        if hetero_apply_to in {"both", "cut"}:
            x_cut = apply_mean_scaling(x_cut, m)

        if hetero_apply_to in {"both", "ref"}:
            if m.ndim == 1 and m.size == n_trials:
                x_ref = apply_mean_scaling(x_ref, m.reshape(-1, 1))
            else:
                x_ref = apply_mean_scaling(x_ref, m)

    os_k = _kth_order_statistic(x_ref, k=rank_k)
    thr = float(alpha) * os_k
    fa = x_cut > thr

    n_fa = int(np.sum(fa))
    pfa_emp = float(n_fa / n_trials)

    ci_norm = _ci_normal_approx(p=pfa_emp, n=n_trials, z=1.96)
    ci_wilson = _ci_wilson(p=pfa_emp, n=n_trials, z=1.96)

    return {
        "task": "pfa_monte_carlo",
        "detector": "os_cfar_independent",
        "pfa_target": float(pfa),
        "pfa_empirical": pfa_emp,
        "n_trials": int(n_trials),
        "n_false_alarms": n_fa,
        "alpha": float(alpha),
        "n_ref": int(n_ref),
        "rank_k": int(rank_k),
        "confidence_intervals": {
            "normal_95": {"low": ci_norm[0], "high": ci_norm[1]},
            "wilson_95": {"low": ci_wilson[0], "high": ci_wilson[1]},
        },
    }


def _calibrate_os_cfar_alpha(
    *,
    dist: Any,
    pfa_target: float,
    n_ref: int,
    rank_k: int,
    n_cal: int,
    seed: Optional[int],
) -> Tuple[float, Dict[str, Any]]:
    if n_cal < 10_000:
        raise MonteCarloError(f"alpha_calibration_n too small (min 10000), got {n_cal}")

    rng = np.random.default_rng(None if seed is None else int(seed))

    x_cut = dist.sample(rng, size=n_cal)
    x_ref = dist.sample(rng, size=(n_cal, n_ref))
    os_k = _kth_order_statistic(x_ref, k=rank_k)

    def pfa_for(a: float) -> float:
        return float(np.mean(x_cut > (a * os_k)))

    lo = 1e-6
    hi = 1e6

    p_lo = pfa_for(lo)
    p_hi = pfa_for(hi)

    if p_lo < pfa_target:
        lo = 1e-12
        p_lo = pfa_for(lo)
    if p_hi > pfa_target:
        hi = 1e12
        p_hi = pfa_for(hi)

    if not (p_lo >= pfa_target >= p_hi):
        raise MonteCarloError(
            "Failed to bracket alpha for OS-CFAR calibration: "
            f"pfa(lo)={p_lo:.3g}, pfa(hi)={p_hi:.3g}, target={pfa_target:.3g}"
        )

    log_lo = math.log(lo)
    log_hi = math.log(hi)

    for _ in range(60):
        log_mid = 0.5 * (log_lo + log_hi)
        mid = math.exp(log_mid)
        p_mid = pfa_for(mid)
        if p_mid > pfa_target:
            log_lo = log_mid
        else:
            log_hi = log_mid

    alpha = math.exp(0.5 * (log_lo + log_hi))
    p_final = pfa_for(alpha)

    meta = {
        "method": "mc_calibration",
        "n_cal": int(n_cal),
        "seed": None if seed is None else int(seed),
        "rank_k": int(rank_k),
        "n_ref": int(n_ref),
        "pfa_target": float(pfa_target),
        "pfa_calibrated": float(p_final),
    }
    return float(alpha), meta


# ---------------------------------------------------------------------
# Pd experiment core (independent trials, noncoherent integration)
# ---------------------------------------------------------------------


def _run_independent_cfar_pd(
    *,
    rng: np.random.Generator,
    dist: Any,
    detector: str,
    pfa: float,
    n_trials: int,
    n_ref: int,
    alpha: float,
    os_rank_k: Optional[int],
    mean_multiplier: Optional[Union[float, np.ndarray]],
    hetero_apply_to: str,
    n_pulses: int,
    swerling: str,
    mean_power: float,
    snr_lin: float,
) -> Tuple[float, int, Tuple[float, float], Tuple[float, float]]:
    # Sample background powers per pulse, then integrate (sum) across pulses.
    # Shapes:
    #   cut_bg: (n_trials, n_pulses)
    #   ref_bg: (n_trials, n_ref, n_pulses)
    cut_bg = dist.sample(rng, size=(n_trials, n_pulses))
    ref_bg = dist.sample(rng, size=(n_trials, n_ref, n_pulses))

    # Apply heterogeneity scaling (same semantics as Pfa).
    if mean_multiplier is not None:
        m = np.asarray(mean_multiplier, dtype=float)

        if hetero_apply_to in {"both", "cut"}:
            cut_bg = apply_mean_scaling(cut_bg, m.reshape(-1, 1) if (m.ndim == 1 and m.size == n_trials) else m)

        if hetero_apply_to in {"both", "ref"}:
            if m.ndim == 1 and m.size == n_trials:
                ref_bg = apply_mean_scaling(ref_bg, m.reshape(-1, 1, 1))
            else:
                ref_bg = apply_mean_scaling(ref_bg, m)

    # Target injection: power-domain additive target power into CUT only.
    # Mean target power per pulse = snr_lin * mean_power.
    mean_sig = float(snr_lin) * float(mean_power)

    mult = _swerling_multiplier(rng=rng, n_trials=n_trials, n_pulses=n_pulses, swerling=swerling)
    sig = mean_sig * mult  # shape: (n_trials, n_pulses)

    cut = np.sum(cut_bg + sig, axis=1)  # (n_trials,)
    ref = np.sum(ref_bg, axis=2)  # (n_trials, n_ref)

    if detector == "ca_cfar_independent":
        thr = float(alpha) * np.mean(ref, axis=1)
    elif detector == "os_cfar_independent":
        if os_rank_k is None:
            raise MonteCarloError("internal error: os_rank_k is required for OS-CFAR Pd")
        os_k = _kth_order_statistic(ref, k=int(os_rank_k))
        thr = float(alpha) * os_k
    else:
        raise MonteCarloError(f"internal error: unsupported detector for Pd: {detector}")

    det = cut > thr
    n_det = int(np.sum(det))
    pd_emp = float(n_det / n_trials)

    ci_norm = _ci_normal_approx(p=pd_emp, n=n_trials, z=1.96)
    ci_wilson = _ci_wilson(p=pd_emp, n=n_trials, z=1.96)
    return pd_emp, n_det, ci_norm, ci_wilson


def _swerling_multiplier(
    *,
    rng: np.random.Generator,
    n_trials: int,
    n_pulses: int,
    swerling: str,
) -> np.ndarray:
    """
    Return per-pulse power multipliers with mean 1.0.

    Output shape is (n_trials, n_pulses).
    """
    if swerling == "swerling0":
        return np.ones((n_trials, n_pulses), dtype=float)

    # Exponential with mean 1 (gamma(k=1, theta=1))
    if swerling == "swerling1":
        v = rng.exponential(scale=1.0, size=n_trials).reshape(-1, 1)
        return np.repeat(v, n_pulses, axis=1)

    if swerling == "swerling2":
        return rng.exponential(scale=1.0, size=(n_trials, n_pulses))

    # Gamma(k=2, theta=0.5) => mean 1
    if swerling == "swerling3":
        v = rng.gamma(shape=2.0, scale=0.5, size=n_trials).reshape(-1, 1)
        return np.repeat(v, n_pulses, axis=1)

    if swerling == "swerling4":
        return rng.gamma(shape=2.0, scale=0.5, size=(n_trials, n_pulses))

    raise MonteCarloError(f"Unsupported swerling model: {swerling}")


# ---------------------------------------------------------------------
# Heterogeneity helpers
# ---------------------------------------------------------------------


def _expand_mean_multiplier_segments(*, segments: Any, n_trials: int) -> np.ndarray:
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
        raise MonteCarloError(f"Sum of hetero segment counts ({total}) != n_trials ({n_trials})")

    return np.concatenate(values)


# ---------------------------------------------------------------------
# Background distribution factory
# ---------------------------------------------------------------------


def _build_background_distribution(bg: Dict[str, Any]) -> Any:
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
    if n <= 0:
        return (0.0, 1.0)
    p = float(p)
    se = math.sqrt(max(p * (1.0 - p), 0.0) / n)
    return (max(0.0, p - z * se), min(1.0, p + z * se))


def _ci_wilson(*, p: float, n: int, z: float) -> Tuple[float, float]:
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
    mc = cfg.get("monte_carlo", {})
    bg = mc.get("background", {}) if isinstance(mc, dict) else {}

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


def _require_number(d: Dict[str, Any], key: str) -> float:
    v = d.get(key, None)
    if v is None:
        raise MonteCarloError(f"Missing required number '{key}'")
    try:
        x = float(v)
    except Exception as exc:
        raise MonteCarloError(f"'{key}' must be numeric, got {v}") from exc
    if not math.isfinite(x):
        raise MonteCarloError(f"'{key}' must be finite, got {x}")
    return x


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