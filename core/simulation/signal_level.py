"""
core/simulation/signal_level.py

Selective signal-level engine for radar spot validation (IQ / Range-Doppler).

What this module does
---------------------
This module provides a *selective*, higher-fidelity simulation path intended to validate
(or challenge) the model-based engine in a small number of carefully chosen cases.

It is NOT designed for massive sweeps. It is designed for questions like:
- "Does the RD-map behavior agree with closed-form expectations?"
- "Are sidelobes / leakage / windowing / binning assumptions hiding failure modes?"
- "Does CFAR behave as expected on a synthetic RD power map?"

In v1, this engine focuses on producing a consistent synthetic Range-Doppler (RD) map:
- RD grid defined by geometry.n_range_bins and geometry.n_doppler_bins
- Noise floor consistent with receiver k*T*B and Noise Figure (NF)
- Single injected target (deterministic amplitude) placed in a chosen RD bin
- Optional CA-CFAR on the RD power map (simplified 2D ring neighborhood)

Scope (v1)
----------
- Produces a synthetic RD power map with:
  - Complex circular Gaussian noise
  - Single deterministic target injection into one RD bin
- Optional CA-CFAR-like detection on the RD map

Non-goals (explicit)
--------------------
- This does NOT simulate raw wideband fast-time IQ.
- This does NOT perform pulse compression / matched filtering.
- This does NOT map physical range to bins (no waveform/timing schema yet).

Inputs (cfg dict)
-----------------
Designed to work with the existing strict schema (performance_case).
Uses only already-defined sections:

Required:
- cfg["radar"]["fc_hz"]
- cfg["radar"]["tx_power_w"]
- cfg["antenna"]["gain_tx_db"] (gain_rx_db optional)
- cfg["receiver"]["bw_hz"], cfg["receiver"]["nf_db"]
- cfg["target"]["rcs_sqm"]

Optional / recommended:
- cfg["environment"]["system_losses_db"] (default 0)
- cfg["geometry"]["n_range_bins"], cfg["geometry"]["n_doppler_bins"] (defaults if missing)
- cfg["scenario"]["range_m"] or cfg["metrics"]["ranges_m"] (used only to compute Pr for the injected target)
- cfg["detection"]["pfa"] (if provided, enables CA-CFAR thresholding)

Outputs (metrics dict)
----------------------
Metrics dictionary suitable for metrics.json:
- "engine": "signal_level"
- "rd_grid": grid sizes and chosen target bin
- "noise_model": N, sigma2
- "target_injection": range used for Pr, Pr, injected amplitude, and implied SNR in the injected RD bin
- "rd_power_map_stats": summary stats (mean/median/percentiles/max)
- "detection": optional CA-CFAR summary
- "crosscheck_model_based": optional comparison vs model_based SNR for the same range point

CLI usage
---------
Called by cli/run_case.py when --engine signal_level is selected:

    python -m cli.run_case --case configs/cases/demo_pd_noise.yaml --engine signal_level --seed 123

Design notes (professional)
---------------------------
- Deterministic output given (cfg, seed)
- No file I/O, no plotting
- Explicit assumptions are surfaced in metrics["assumptions"]
- No hidden state

Dependencies
------------
- NumPy
- SciPy is NOT required.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple
import math

import numpy as np

from core.config.units import (
    db_to_lin_power,
    lin_to_db_power,
    k_boltzmann,
)
from core.detection.cfar import ca_cfar_alpha
from core.simulation.model_based import run_model_based_case


class SignalLevelError(ValueError):
    """Raised when signal-level configuration is invalid or inconsistent."""


@dataclass(frozen=True)
class RDGrid:
    """
    Range-Doppler grid definition for the synthetic RD map.

    Attributes
    ----------
    n_range_bins : int
        Number of range bins (fast-time / matched-filter bins).
    n_doppler_bins : int
        Number of Doppler bins (slow-time FFT bins).
    """
    n_range_bins: int
    n_doppler_bins: int


# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------

def run_signal_level_case(cfg: Dict[str, Any], seed: Optional[int] = None) -> Dict[str, Any]:
    """
    Run a selective signal-level simulation and produce RD-map metrics.

    Parameters
    ----------
    cfg : dict
        Validated and normalized performance case configuration.
    seed : int | None
        RNG seed for reproducibility.

    Returns
    -------
    dict
        Metrics dictionary suitable for writing to metrics.json.
    """
    radar = _require_section(cfg, "radar")
    antenna = _require_section(cfg, "antenna")
    receiver = _require_section(cfg, "receiver")
    target = _require_section(cfg, "target")

    # Minimal required fields
    fc_hz = _require_pos_float(radar, "fc_hz")
    tx_power_w = _require_pos_float(radar, "tx_power_w")

    gain_tx_db = _require_float(antenna, "gain_tx_db", default=0.0)
    gain_rx_db = _require_float(antenna, "gain_rx_db", default=gain_tx_db)

    bw_hz = _require_pos_float(receiver, "bw_hz")
    nf_db = _require_nonneg_float(receiver, "nf_db", default=0.0)
    temperature_k = _require_pos_float(receiver, "temperature_k", default=290.0)

    sigma_sqm = _require_pos_float(target, "rcs_sqm")

    env = cfg.get("environment", {}) if isinstance(cfg.get("environment", {}), dict) else {}
    system_losses_db = _require_nonneg_float(env, "system_losses_db", default=0.0)

    # Geometry defines RD grid size.
    geom = cfg.get("geometry", {}) if isinstance(cfg.get("geometry", {}), dict) else {}
    grid = RDGrid(
        n_range_bins=int(geom.get("n_range_bins", 256) or 256),
        n_doppler_bins=int(geom.get("n_doppler_bins", 64) or 64),
    )
    if grid.n_range_bins <= 0 or grid.n_doppler_bins <= 0:
        raise SignalLevelError("geometry.n_range_bins and geometry.n_doppler_bins must be positive integers")

    # Detection (optional): if pfa exists, we run a CA-CFAR-like detector on the RD power map.
    det = cfg.get("detection", {}) if isinstance(cfg.get("detection", {}), dict) else {}
    pfa = det.get("pfa", None)

    rng = np.random.default_rng(None if seed is None else int(seed))

    # -----------------------------------------------------------------
    # 1) Noise model (consistent with the performance engine)
    # -----------------------------------------------------------------
    # Receiver noise power at input:
    #   N = k * T * B * F
    #
    # Modeling assumption (explicit):
    # - In this synthetic RD map, we interpret N as the *mean power per RD cell*.
    # - This is not a waveform-derived calibration; it is a consistency assumption.
    noise_factor = float(db_to_lin_power(float(nf_db)))
    noise_power_w = k_boltzmann() * float(temperature_k) * float(bw_hz) * noise_factor

    # Complex circular Gaussian:
    #   z = x + j y, x,y ~ N(0, sigma2)
    # Then E[|z|^2] = 2*sigma2.
    # Set E[|z|^2] = noise_power_w => sigma2 = noise_power_w / 2.
    if not math.isfinite(noise_power_w) or noise_power_w <= 0.0:
        raise SignalLevelError(f"Computed noise_power_w must be finite and > 0, got {noise_power_w}")
    sigma2 = noise_power_w / 2.0

    # -----------------------------------------------------------------
    # 2) Create a synthetic RD map with noise
    # -----------------------------------------------------------------
    rd_complex = _complex_gaussian(
        rng=rng,
        shape=(grid.n_range_bins, grid.n_doppler_bins),
        sigma2=sigma2,
    )

    # -----------------------------------------------------------------
    # 3) Inject a deterministic target into a selected RD bin (spot validation)
    # -----------------------------------------------------------------
    target_bin_r, target_bin_d = _choose_target_bin(cfg, grid)
    target_range_m = _choose_target_range_m(cfg)

    pr_w = _received_power_w_single_bin(
        fc_hz=fc_hz,
        tx_power_w=tx_power_w,
        gain_tx_db=gain_tx_db,
        gain_rx_db=gain_rx_db,
        sigma_sqm=sigma_sqm,
        system_losses_db=system_losses_db,
        range_m=target_range_m,
    )

    # Map Pr into a complex bin injection:
    # Since power in a bin is |z|^2, choose amplitude A such that A^2 = Pr.
    amp = math.sqrt(max(pr_w, 0.0))

    # Random phase, deterministic via RNG seed
    phase = float(rng.uniform(0.0, 2.0 * math.pi))
    rd_complex[target_bin_r, target_bin_d] += amp * (math.cos(phase) + 1j * math.sin(phase))

    # RD power map
    rd_power = np.abs(rd_complex) ** 2

    # Implied injected SNR in the RD cell under this modeling assumption
    snr_injected_lin = float(pr_w / noise_power_w)
    snr_injected_db = float(lin_to_db_power(max(snr_injected_lin, np.finfo(float).tiny)))

    # -----------------------------------------------------------------
    # 3.5) Cross-check against model_based (same range point)
    # -----------------------------------------------------------------
    # Purpose:
    # - Keep this engine honest: for the same configuration and the same single range point,
    #   compare this engine's implied SNR (Pr/N) against the model_based engine's SNR output.
    #
    # Non-fatal by design:
    # - A cross-check failure should not block RD-map generation; it should be recorded clearly.
    crosscheck_model_based: Dict[str, Any]
    try:
        cfg_mb = dict(cfg)
        cfg_mb["metrics"] = {"ranges_m": [float(target_range_m)]}

        mb = run_model_based_case(cfg_mb, seed=None)

        snr_mb_lin = _extract_single_snr_lin(mb)
        snr_mb_db = float(lin_to_db_power(max(snr_mb_lin, np.finfo(float).tiny)))

        delta_db = float(snr_injected_db - snr_mb_db)
        rel_err_lin = float(abs(snr_injected_lin - snr_mb_lin) / max(abs(snr_mb_lin), 1e-30))

        crosscheck_model_based = {
            "ok": True,
            "range_m": float(target_range_m),
            "snr_signal_level_lin": float(snr_injected_lin),
            "snr_signal_level_db": float(snr_injected_db),
            "snr_model_based_lin": float(snr_mb_lin),
            "snr_model_based_db": float(snr_mb_db),
            "delta_db_signal_minus_model": float(delta_db),
            "relative_error_lin": float(rel_err_lin),
        }
    except Exception as exc:
        crosscheck_model_based = {
            "ok": False,
            "range_m": float(target_range_m),
            "error": f"{type(exc).__name__}: {exc}",
        }

    # -----------------------------------------------------------------
    # 4) Optional CA-CFAR-like detection (simplified 2D neighborhood)
    # -----------------------------------------------------------------
    detection_out = None
    if pfa is not None:
        pfa_f = float(pfa)
        if not (0.0 < pfa_f < 1.0):
            raise SignalLevelError(f"detection.pfa must be in (0,1), got {pfa_f}")

        # v1: fixed effective reference count (explicit). Later: parameterize in schema.
        n_ref = 32
        if n_ref <= 0:
            raise SignalLevelError("Internal error: n_ref must be positive")
        alpha = float(ca_cfar_alpha(pfa=pfa_f, n_ref=n_ref))
        if not math.isfinite(alpha) or alpha <= 0.0:
            raise SignalLevelError(f"Computed CA-CFAR alpha must be finite and > 0, got {alpha}")

        det_map, thr_map = _detect_cfar_2d_ring(
            rd_power=rd_power,
            alpha=alpha,
            n_ref=n_ref,
            rng=rng,
        )

        n_det = int(np.sum(det_map))

        thr_cut = float(thr_map[target_bin_r, target_bin_d])
        detection_out = {
            "method": "ca_cfar_2d_ring_v1",
            "pfa_target": pfa_f,
            "n_ref_effective": int(n_ref),
            "alpha": float(alpha),
            "detections_total": n_det,
            "target_bin": {"r": int(target_bin_r), "d": int(target_bin_d)},
            "target_detected": bool(det_map[target_bin_r, target_bin_d]),
            "target_threshold": thr_cut if math.isfinite(thr_cut) else None,
            "target_cell_power": float(rd_power[target_bin_r, target_bin_d]),
        }

    # -----------------------------------------------------------------
    # Package metrics (traceable, report-friendly)
    # -----------------------------------------------------------------
    stats = _rd_power_stats(rd_power)

    metrics: Dict[str, Any] = {
        "engine": "signal_level",
        "assumptions": {
            "rd_map_is_synthetic": True,
            "no_waveform_or_pulse_compression": True,
            "no_physical_range_to_bin_mapping": True,
            "noise_power_interpreted_as_mean_power_per_rd_cell": True,
            "single_target_injected_into_one_rd_bin": True,
        },
        "rd_grid": {
            "n_range_bins": int(grid.n_range_bins),
            "n_doppler_bins": int(grid.n_doppler_bins),
            "target_bin": {"r": int(target_bin_r), "d": int(target_bin_d)},
        },
        "noise_model": {
            "noise_power_w": float(noise_power_w),
            "noise_power_dbw": float(lin_to_db_power(max(noise_power_w, np.finfo(float).tiny))),
            "sigma2_per_quadrature": float(sigma2),
        },
        "target_injection": {
            "range_m": float(target_range_m),
            "received_power_w": float(pr_w),
            "received_power_dbw": float(lin_to_db_power(max(pr_w, np.finfo(float).tiny))),
            "injected_amplitude": float(amp),
            "snr_injected_lin": float(snr_injected_lin),
            "snr_injected_db": float(snr_injected_db),
        },
        "rd_power_map_stats": stats,
        "crosscheck_model_based": crosscheck_model_based,
    }

    if detection_out is not None:
        metrics["detection"] = detection_out

    return metrics


# ---------------------------------------------------------------------
# RD map helpers
# ---------------------------------------------------------------------

def _complex_gaussian(*, rng: np.random.Generator, shape: Tuple[int, int], sigma2: float) -> np.ndarray:
    """
    Generate complex circular Gaussian noise with given quadrature variance sigma2.

    Parameters
    ----------
    rng : np.random.Generator
        Random number generator.
    shape : tuple[int,int]
        Output shape (n_range_bins, n_doppler_bins).
    sigma2 : float
        Variance of each quadrature component.

    Returns
    -------
    np.ndarray (complex128)
        Complex noise array.
    """
    if sigma2 < 0.0 or not math.isfinite(float(sigma2)):
        raise SignalLevelError(f"sigma2 must be finite and >= 0, got {sigma2}")
    scale = math.sqrt(sigma2)
    x = rng.normal(loc=0.0, scale=scale, size=shape)
    y = rng.normal(loc=0.0, scale=scale, size=shape)
    return x + 1j * y


def _rd_power_stats(rd_power: np.ndarray) -> Dict[str, Any]:
    """
    Compute robust summary statistics for an RD power map.

    Notes
    -----
    These stats are designed for metrics.json (human diff-friendly):
    - mean/median show global floor behavior
    - p90/p99 show tail behavior
    - max indicates peak strength (target or noise outlier)
    """
    x = np.asarray(rd_power, dtype=float)
    return {
        "shape": [int(x.shape[0]), int(x.shape[1])],
        "mean": float(np.mean(x)),
        "median": float(np.median(x)),
        "p90": float(np.percentile(x, 90.0)),
        "p99": float(np.percentile(x, 99.0)),
        "max": float(np.max(x)),
    }


def _detect_cfar_2d_ring(
    *,
    rd_power: np.ndarray,
    alpha: float,
    n_ref: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simplified 2D CA-CFAR detection on an RD map using a small neighborhood "ring".

    IMPORTANT (v1 limitations)
    --------------------------
    - Fixed neighborhood size (5x5 window) excluding the CUT.
    - Deterministic subsampling if available references exceed n_ref.
    - Edge cells are excluded (threshold remains NaN).
    - Adequate for spot validation and wiring checks; not a production CFAR implementation.

    Parameters
    ----------
    rd_power : np.ndarray
        2D RD power map.
    alpha : float
        CA-CFAR scaling factor (threshold multiplier).
    n_ref : int
        Effective number of reference cells used to estimate mean background.
    rng : np.random.Generator
        RNG used only if subsampling references is required.

    Returns
    -------
    detections : np.ndarray (bool)
        Detection map.
    threshold : np.ndarray (float)
        Threshold map (NaN on excluded edges).
    """
    if n_ref <= 0:
        raise SignalLevelError(f"n_ref must be >= 1, got {n_ref}")
    if not math.isfinite(alpha) or alpha <= 0.0:
        raise SignalLevelError(f"alpha must be finite and > 0, got {alpha}")

    x = np.asarray(rd_power, dtype=float)
    if x.ndim != 2:
        raise SignalLevelError("rd_power must be a 2D array")

    nr, nd = x.shape

    # Fixed neighborhood: 5x5 centered on CUT
    half = 2
    if nr < (2 * half + 1) or nd < (2 * half + 1):
        raise SignalLevelError("RD grid too small for CFAR neighborhood (need at least 5x5)")

    det = np.zeros_like(x, dtype=bool)
    thr = np.full_like(x, fill_value=np.nan, dtype=float)

    cut_idx = (2 * half + 1) * half + half  # center element of flattened 5x5

    for r in range(half, nr - half):
        for d in range(half, nd - half):
            win = x[r - half : r + half + 1, d - half : d + half + 1].reshape(-1)

            # Exclude CUT
            refs = np.delete(win, cut_idx)

            # Enforce n_ref by deterministic subsampling if needed
            if refs.size > n_ref:
                idx = rng.choice(refs.size, size=n_ref, replace=False)
                refs_eff = refs[idx]
            else:
                refs_eff = refs

            mu = float(np.mean(refs_eff))
            t = float(alpha * mu)
            thr[r, d] = t
            det[r, d] = bool(x[r, d] > t)

    return det, thr


# ---------------------------------------------------------------------
# Radar-equation-like injection helpers (single-bin received power)
# ---------------------------------------------------------------------

def _received_power_w_single_bin(
    *,
    fc_hz: float,
    tx_power_w: float,
    gain_tx_db: float,
    gain_rx_db: float,
    sigma_sqm: float,
    system_losses_db: float,
    range_m: float,
) -> float:
    """
    Monostatic radar equation received power for a single range point.

    Pr = Pt * Gt * Gr * (lambda^2 * sigma) / ((4*pi)^3 * R^4 * L)

    Notes
    -----
    This intentionally mirrors core/simulation/model_based.py behavior, but stays scalar
    to avoid tight coupling and to keep import-time dependencies minimal.
    """
    if range_m <= 0.0 or not math.isfinite(range_m):
        raise SignalLevelError(f"range_m must be finite and > 0, got {range_m}")

    wavelength_m = 299_792_458.0 / float(fc_hz)
    gt = float(db_to_lin_power(float(gain_tx_db)))
    gr = float(db_to_lin_power(float(gain_rx_db)))
    losses = float(db_to_lin_power(float(system_losses_db)))

    numerator = float(tx_power_w) * gt * gr * (wavelength_m ** 2) * float(sigma_sqm)
    denom = ((4.0 * math.pi) ** 3) * (float(range_m) ** 4) * losses
    return float(numerator / denom)


def _choose_target_range_m(cfg: Dict[str, Any]) -> float:
    """
    Choose a target range for injection (used only to compute Pr).

    Priority:
    - scenario.range_m if present
    - metrics.ranges_m median if present
    - default: 10 km
    """
    scenario = cfg.get("scenario", {}) if isinstance(cfg.get("scenario", {}), dict) else {}
    if "range_m" in scenario:
        r = float(scenario["range_m"])
        if not math.isfinite(r) or r <= 0.0:
            raise SignalLevelError("scenario.range_m must be finite and > 0")
        return r

    metrics = cfg.get("metrics", {}) if isinstance(cfg.get("metrics", {}), dict) else {}
    if "ranges_m" in metrics and isinstance(metrics["ranges_m"], list) and len(metrics["ranges_m"]) > 0:
        arr = np.asarray([float(x) for x in metrics["ranges_m"]], dtype=float)
        arr = arr[np.isfinite(arr) & (arr > 0.0)]
        if arr.size > 0:
            return float(np.median(arr))

    return 10_000.0


def _choose_target_bin(cfg: Dict[str, Any], grid: RDGrid) -> Tuple[int, int]:
    """
    Choose a target RD bin location.

    v1 rule (explicit):
    - Place target at the center Doppler bin
    - Place target at the middle range bin

    Rationale:
    - No physical range-to-bin mapping exists in the schema yet.
    - This choice is deterministic and reviewable.
    """
    r_bin = int(grid.n_range_bins // 2)
    d_bin = int(grid.n_doppler_bins // 2)
    return r_bin, d_bin


def _extract_single_snr_lin(metrics: Dict[str, Any]) -> float:
    """
    Extract a single-point SNR (linear) from model_based metrics output.

    Accepted shapes (by design tolerance):
    - metrics["snr_lin"] = [x]
    - metrics["snr"]["lin"] = [x]
    - metrics["snr_db"] = [x]  (converted to linear)
    - metrics["snr"]["db"] = [x] (converted to linear)

    Raises
    ------
    SignalLevelError if a single SNR point cannot be extracted.
    """
    # snr_lin (top-level)
    v = metrics.get("snr_lin", None)
    if isinstance(v, list) and len(v) == 1:
        x = float(v[0])
        if not math.isfinite(x) or x <= 0.0:
            raise SignalLevelError(f"model_based snr_lin must be finite and > 0, got {x}")
        return x

    # snr dict
    snr = metrics.get("snr", None)
    if isinstance(snr, dict):
        v2 = snr.get("lin", None)
        if isinstance(v2, list) and len(v2) == 1:
            x = float(v2[0])
            if not math.isfinite(x) or x <= 0.0:
                raise SignalLevelError(f"model_based snr.lin must be finite and > 0, got {x}")
            return x

    # snr_db (top-level)
    vdb = metrics.get("snr_db", None)
    if isinstance(vdb, list) and len(vdb) == 1:
        db = float(vdb[0])
        if not math.isfinite(db):
            raise SignalLevelError(f"model_based snr_db must be finite, got {db}")
        lin = float(db_to_lin_power(db))
        if not math.isfinite(lin) or lin <= 0.0:
            raise SignalLevelError(f"model_based snr_db converted to invalid linear SNR: {lin}")
        return lin

    # snr dict db
    if isinstance(snr, dict):
        vdb2 = snr.get("db", None)
        if isinstance(vdb2, list) and len(vdb2) == 1:
            db = float(vdb2[0])
            if not math.isfinite(db):
                raise SignalLevelError(f"model_based snr.db must be finite, got {db}")
            lin = float(db_to_lin_power(db))
            if not math.isfinite(lin) or lin <= 0.0:
                raise SignalLevelError(f"model_based snr.db converted to invalid linear SNR: {lin}")
            return lin

    raise SignalLevelError("Unable to extract single-point SNR from model_based metrics")


# ---------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------

def _require_section(cfg: Dict[str, Any], key: str) -> Dict[str, Any]:
    sec = cfg.get(key, None)
    if not isinstance(sec, dict):
        raise SignalLevelError(f"Missing or invalid cfg['{key}'] section (must be a dict)")
    return sec


def _require_float(section: Dict[str, Any], key: str, *, default: Optional[float] = None) -> float:
    if key not in section:
        if default is None:
            raise SignalLevelError(f"Missing required field '{key}'")
        return float(default)
    try:
        v = float(section[key])
    except Exception as exc:
        raise SignalLevelError(f"Field '{key}' must be numeric, got {section[key]}") from exc
    if not math.isfinite(v):
        raise SignalLevelError(f"Field '{key}' must be finite, got {section[key]}")
    return v


def _require_pos_float(section: Dict[str, Any], key: str, *, default: Optional[float] = None) -> float:
    v = _require_float(section, key, default=default)
    if v <= 0.0:
        raise SignalLevelError(f"Field '{key}' must be > 0, got {v}")
    return v


def _require_nonneg_float(section: Dict[str, Any], key: str, *, default: Optional[float] = None) -> float:
    v = _require_float(section, key, default=default)
    if v < 0.0:
        raise SignalLevelError(f"Field '{key}' must be >= 0, got {v}")
    return v