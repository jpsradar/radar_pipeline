"""
core/simulation/model_based.py

Fast physical-statistical engine (model-based) for radar performance trade studies.

What this module does (serious baseline)
----------------------------------------
- Computes a monostatic radar equation budget (received power vs range).
- Computes receiver noise power using k*T*B and Noise Figure (NF).
- Produces SNR in linear and dB (noise-limited baseline; SINR is an extension hook).
- Computes detection performance (Pd) for a square-law energy detector using
  chi-square / noncentral chi-square statistics (no approximations).

Scope (v1)
----------
- Primary use: performance and trade sweeps (fast).
- Assumes a pulsed (or pulse-Doppler) style matched-filter output per range cell.
- Supports:
  - Noncoherent integration: sum of pulse energies (chi-square with 2N DOF).
  - Coherent-like baseline: 2-DOF energy test with boosted SNR_total = N * SNR_per_pulse.

Contract for model-vs-experiment comparisons
--------------------------------------------
When detection.pfa is provided, this engine MUST emit a fully specified detector contract
inside metrics["detection"], including:
- the test statistic definition
- the assumed H0/H1 distributions and parameters
- the threshold used
- the resulting Pd curve

This is required so a Monte Carlo experiment can replicate the exact same detector.

Dependencies
------------
- NumPy
- SciPy (chi-square and noncentral chi-square)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional
import math

import numpy as np

try:
    from scipy.stats import chi2, ncx2  # type: ignore
except Exception as exc:  # pragma: no cover
    raise ImportError(
        "SciPy is required for detection statistics. Install with: pip install scipy"
    ) from exc

from core.config.units import (
    db_to_lin_power,
    lin_to_db_power,
    k_boltzmann,
)

# Geometry + FAR (system-level counts)
from core.geometry.dwell import make_cpi, make_dwell, cpis_per_second
from core.geometry.counts import make_rd_grid, make_scan_geometry
from core.detection.far_conversion import FARInputs, convert_pfa_to_far


# ----------------------------
# Internal data containers
# ----------------------------

@dataclass(frozen=True)
class BudgetTerms:
    """Holds intermediate budget terms for traceability."""
    wavelength_m: float
    gt_lin: float
    gr_lin: float
    losses_lin: float
    noise_power_w: float


# ----------------------------
# Public API
# ----------------------------

def run_model_based_case(cfg: Dict[str, Any], seed: Optional[int] = None) -> Dict[str, Any]:
    """
    Run a model-based performance evaluation for the provided config.

    Parameters
    ----------
    cfg : dict
        Validated and normalized case configuration.
    seed : int | None
        Reserved for future stochastic features. Not used yet.

    Returns
    -------
    dict
        Metrics dictionary to be written to metrics.json.
    """
    # NOTE: Engine is deterministic. Seed is stored by CLI for provenance.
    _ = seed

    # --- Extract and validate minimal required parameters ---
    radar = _require_section(cfg, "radar")
    antenna = _require_section(cfg, "antenna")
    receiver = _require_section(cfg, "receiver")
    target = _require_section(cfg, "target")

    fc_hz = _require_pos_float(radar, "fc_hz")
    tx_power_w = _require_pos_float(radar, "tx_power_w")

    # PRF is used for explicit CPI timing and FAR conversion.
    prf_hz = _require_pos_float(radar, "prf_hz", default=1_000.0)

    gain_tx_db = _require_float(antenna, "gain_tx_db", default=0.0)
    gain_rx_db = _require_float(antenna, "gain_rx_db", default=gain_tx_db)

    bw_hz = _require_pos_float(receiver, "bw_hz")
    nf_db = _require_nonneg_float(receiver, "nf_db", default=0.0)
    temperature_k = _require_pos_float(receiver, "temperature_k", default=290.0)

    sigma_sqm = _require_pos_float(target, "rcs_sqm")

    env = cfg.get("environment", {}) if isinstance(cfg.get("environment", {}), dict) else {}
    system_losses_db = _require_nonneg_float(env, "system_losses_db", default=0.0)

    # Ranges: metrics.ranges_m (vector) preferred; else scenario.range_m; else default.
    ranges_m = _get_ranges_m(cfg)

    # Detection settings (optional)
    det = cfg.get("detection", {}) if isinstance(cfg.get("detection", {}), dict) else {}
    pfa = det.get("pfa", None)
    n_pulses = int(det.get("n_pulses", 1) or 1)
    integration = str(det.get("integration", "noncoherent")).lower().strip()

    if n_pulses <= 0:
        raise ValueError(f"detection.n_pulses must be >= 1, got {n_pulses}")
    if integration not in {"noncoherent", "coherent"}:
        raise ValueError("detection.integration must be 'noncoherent' or 'coherent'")

    # -----------------------------------------------------------------
    # Geometry + counts (explicit, to enable system-level FAR)
    # -----------------------------------------------------------------
    geometry = cfg.get("geometry", {}) if isinstance(cfg.get("geometry", {}), dict) else {}

    n_range_bins = int(geometry.get("n_range_bins", 256))
    n_doppler_bins = int(geometry.get("n_doppler_bins", n_pulses))
    beams_per_scan = int(geometry.get("beams_per_scan", 1))
    n_cpi_per_dwell = int(geometry.get("n_cpi_per_dwell", 1))

    cpi = make_cpi(prf_hz=prf_hz, n_pulses=n_pulses)
    dwell = make_dwell(cpi=cpi, n_cpi=n_cpi_per_dwell)
    cpis_rate_hz = cpis_per_second(cpi)

    rd_grid = make_rd_grid(n_range_bins=n_range_bins, n_doppler_bins=n_doppler_bins)
    scan_geom = make_scan_geometry(beams_per_scan=beams_per_scan, dwell_time_s=dwell.dwell_time_s)

    # --- Compute budget and SNR vs range ---
    terms = _compute_budget_terms(
        fc_hz=fc_hz,
        gain_tx_db=gain_tx_db,
        gain_rx_db=gain_rx_db,
        nf_db=nf_db,
        temperature_k=temperature_k,
        bw_hz=bw_hz,
        system_losses_db=system_losses_db,
    )

    pr_w = _received_power_w(
        tx_power_w=tx_power_w,
        gt_lin=terms.gt_lin,
        gr_lin=terms.gr_lin,
        wavelength_m=terms.wavelength_m,
        sigma_sqm=sigma_sqm,
        range_m=ranges_m,
        losses_lin=terms.losses_lin,
    )

    # Noise-limited baseline (SINR extension would modify denominator)
    snr_lin = pr_w / terms.noise_power_w
    snr_db = lin_to_db_power(np.maximum(snr_lin, np.finfo(float).tiny))

    # --- Detection performance (optional) ---
    pd = None
    threshold = None
    far = None
    detection_contract: Optional[Dict[str, Any]] = None

    if pfa is not None:
        pfa_f = float(pfa)
        if not (0.0 < pfa_f < 1.0):
            raise ValueError(f"detection.pfa must be in (0, 1), got {pfa_f}")

        # Detector contract (explicit, for model-vs-experiment comparisons)
        # Statistic is always in the POWER/ENERGY domain:
        #   T = sum_{i=1..N} |z_i|^2    (noncoherent)
        # or coherent-like baseline:
        #   T = |z|^2 with boosted SNR_total = N * SNR_per_pulse (df=2)
        detection_contract = {
            "pfa_target": pfa_f,
            "integration": integration,
            "n_pulses": int(n_pulses),
            "statistic": {
                "name": "energy_sum",
                "definition": "T = sum_{i=1..N} |z_i|^2 (power/energy domain)",
                "normalization": "unit-variance convention for chi-square family (SciPy chi2/ncx2)",
            },
            "h0": {},
            "h1": {},
        }

        if integration == "noncoherent":
            # Under H0 => chi-square with 2N DOF
            # Under H1 => noncentral chi-square with 2N DOF, nc = 2*N*SNR_per_pulse
            threshold = _threshold_noncoherent(pfa=pfa_f, n_pulses=n_pulses)
            pd = _pd_noncoherent(threshold=threshold, snr_lin=snr_lin, n_pulses=n_pulses)

            detection_contract["h0"] = {"family": "chi2", "df": int(2 * n_pulses)}
            detection_contract["h1"] = {
                "family": "ncx2",
                "df": int(2 * n_pulses),
                "noncentrality": "nc = 2*N*SNR_per_pulse",
            }
        else:
            # Coherent-like baseline: df=2 energy detector with boosted SNR_total=N*SNR
            threshold = _threshold_coherent(pfa=pfa_f)
            pd = _pd_coherent(threshold=threshold, snr_lin=snr_lin, n_pulses=n_pulses)

            detection_contract["h0"] = {"family": "chi2", "df": 2}
            detection_contract["h1"] = {
                "family": "ncx2",
                "df": 2,
                "noncentrality": "nc = 2*SNR_total, with SNR_total = N*SNR_per_pulse",
            }

        detection_contract["threshold"] = float(threshold)

        # System-level FAR conversion (explicit counts)
        far_inputs = FARInputs(
            pfa=pfa_f,
            domain="rd_cell",
            cells_per_cpi=rd_grid.cells_per_cpi,
            cpis_per_second=cpis_rate_hz,
            beams_per_scan=scan_geom.beams_per_scan,
            scans_per_second=scan_geom.scans_per_second,
        )
        far = convert_pfa_to_far(far_inputs)

    # --- Package metrics with traceability ---
    metrics: Dict[str, Any] = {
        "engine": "model_based",
        "inputs_summary": {
            "fc_hz": fc_hz,
            "tx_power_w": tx_power_w,
            "prf_hz": prf_hz,
            "gain_tx_db": gain_tx_db,
            "gain_rx_db": gain_rx_db,
            "bw_hz": bw_hz,
            "nf_db": nf_db,
            "temperature_k": temperature_k,
            "rcs_sqm": sigma_sqm,
            "system_losses_db": system_losses_db,
            "n_pulses": n_pulses,
            "integration": integration,
            "pfa": float(pfa) if pfa is not None else None,
        },
        "ranges_m": ranges_m.tolist(),
        "budget": {
            "wavelength_m": terms.wavelength_m,
            "gt_lin": terms.gt_lin,
            "gr_lin": terms.gr_lin,
            "losses_lin": terms.losses_lin,
            "noise_power_w": terms.noise_power_w,
            "noise_power_dbw": lin_to_db_power(terms.noise_power_w),
        },
        "received_power_w": pr_w.tolist(),
        "received_power_dbw": lin_to_db_power(np.maximum(pr_w, np.finfo(float).tiny)).tolist(),
        "snr_lin": snr_lin.tolist(),
        "snr_db": snr_db.tolist(),
        "geometry": {
            "n_range_bins": rd_grid.n_range_bins,
            "n_doppler_bins": rd_grid.n_doppler_bins,
            "cells_per_cpi": rd_grid.cells_per_cpi,
            "cpi_duration_s": cpi.duration_s,
            "n_cpi_per_dwell": dwell.n_cpi,
            "dwell_time_s": dwell.dwell_time_s,
            "cpis_per_second": cpis_rate_hz,
            "beams_per_scan": scan_geom.beams_per_scan,
            "scan_time_s": scan_geom.scan_time_s,
            "scans_per_second": scan_geom.scans_per_second,
        },
    }

    if pd is not None and threshold is not None and detection_contract is not None:
        # NOTE: Keep a flat Pd array for plotting, but also ship the full detector contract.
        metrics["detection"] = {
            "threshold": float(threshold),
            "pd": pd.tolist(),
            "pfa_target": detection_contract["pfa_target"],
            "integration": detection_contract["integration"],
            "n_pulses": detection_contract["n_pulses"],
            "contract": detection_contract,
        }

    if far is not None:
        metrics["far"] = {
            "per_second": far.far_per_second,
            "per_scan": far.far_per_scan,
            "breakdown": far.breakdown,
        }

    return metrics


# ----------------------------
# Budget math
# ----------------------------

def _compute_budget_terms(
    *,
    fc_hz: float,
    gain_tx_db: float,
    gain_rx_db: float,
    nf_db: float,
    temperature_k: float,
    bw_hz: float,
    system_losses_db: float,
) -> BudgetTerms:
    """Compute deterministic budget terms for the radar equation and receiver noise."""
    wavelength_m = 299_792_458.0 / fc_hz

    gt_lin = db_to_lin_power(gain_tx_db)
    gr_lin = db_to_lin_power(gain_rx_db)

    losses_lin = db_to_lin_power(system_losses_db)
    noise_factor_lin = db_to_lin_power(nf_db)

    noise_power_w = k_boltzmann() * temperature_k * bw_hz * noise_factor_lin

    return BudgetTerms(
        wavelength_m=wavelength_m,
        gt_lin=gt_lin,
        gr_lin=gr_lin,
        losses_lin=losses_lin,
        noise_power_w=noise_power_w,
    )


def _received_power_w(
    *,
    tx_power_w: float,
    gt_lin: float,
    gr_lin: float,
    wavelength_m: float,
    sigma_sqm: float,
    range_m: np.ndarray,
    losses_lin: float,
) -> np.ndarray:
    """
    Monostatic radar equation (received power).

    Pr = Pt * Gt * Gr * (lambda^2 * sigma) / ( (4*pi)^3 * R^4 * L )
    """
    r = np.asarray(range_m, dtype=float)
    if np.any(r <= 0.0):
        raise ValueError("All ranges must be > 0")

    numerator = tx_power_w * gt_lin * gr_lin * (wavelength_m**2) * sigma_sqm
    denom = ((4.0 * math.pi) ** 3) * (r**4) * losses_lin
    return numerator / denom


# ----------------------------
# Detection math (serious stats)
# ----------------------------

def _threshold_noncoherent(*, pfa: float, n_pulses: int) -> float:
    """Threshold for noncoherent integration of N pulses under H0 (chi-square with 2N DOF)."""
    df = 2 * n_pulses
    return float(chi2.isf(pfa, df=df))


def _pd_noncoherent(*, threshold: float, snr_lin: np.ndarray, n_pulses: int) -> np.ndarray:
    """Pd for noncoherent integration under deterministic target model (ncx2 with nc=2*N*SNR)."""
    df = 2 * n_pulses
    nc = 2.0 * n_pulses * np.asarray(snr_lin, dtype=float)
    return ncx2.sf(threshold, df=df, nc=nc)


def _threshold_coherent(*, pfa: float) -> float:
    """Threshold for 2-DOF energy detector under H0 (baseline for coherent-like model)."""
    return float(chi2.isf(pfa, df=2))


def _pd_coherent(*, threshold: float, snr_lin: np.ndarray, n_pulses: int) -> np.ndarray:
    """Pd for coherent-like baseline (df=2 ncx2 with boosted SNR_total=N*SNR)."""
    snr_total = float(n_pulses) * np.asarray(snr_lin, dtype=float)
    nc = 2.0 * snr_total
    return ncx2.sf(threshold, df=2, nc=nc)


# ----------------------------
# Config helpers
# ----------------------------

def _require_section(cfg: Dict[str, Any], key: str) -> Dict[str, Any]:
    """Require cfg[key] to exist and be a dict."""
    sec = cfg.get(key, None)
    if not isinstance(sec, dict):
        raise ValueError(f"Missing or invalid section cfg['{key}'] (must be a dict)")
    return sec


def _require_float(section: Dict[str, Any], key: str, *, default: Optional[float] = None) -> float:
    """Require a numeric field or return default if missing."""
    if key not in section:
        if default is None:
            raise ValueError(f"Missing required field '{key}'")
        return float(default)
    val = section[key]
    try:
        v = float(val)
    except Exception as exc:
        raise ValueError(f"Field '{key}' must be numeric, got {val}") from exc
    if not math.isfinite(v):
        raise ValueError(f"Field '{key}' must be finite, got {val}")
    return v


def _require_pos_float(section: Dict[str, Any], key: str, *, default: Optional[float] = None) -> float:
    """Require a strictly positive float."""
    v = _require_float(section, key, default=default)
    if v <= 0.0:
        raise ValueError(f"Field '{key}' must be > 0, got {v}")
    return v


def _require_nonneg_float(section: Dict[str, Any], key: str, *, default: Optional[float] = None) -> float:
    """Require a nonnegative float."""
    v = _require_float(section, key, default=default)
    if v < 0.0:
        raise ValueError(f"Field '{key}' must be >= 0, got {v}")
    return v


def _get_ranges_m(cfg: Dict[str, Any]) -> np.ndarray:
    """
    Extract ranges as a numpy array.

    Priority:
    1) cfg["metrics"]["ranges_m"] if present (list of floats)
    2) cfg["scenario"]["range_m"] if present (single float)
    3) default: 10 km single point
    """
    metrics = cfg.get("metrics", {}) if isinstance(cfg.get("metrics", {}), dict) else {}
    if "ranges_m" in metrics:
        ranges = metrics["ranges_m"]
        if not isinstance(ranges, list) or len(ranges) == 0:
            raise ValueError("metrics.ranges_m must be a non-empty list")
        arr = np.array([float(x) for x in ranges], dtype=float)
        if np.any(~np.isfinite(arr)) or np.any(arr <= 0.0):
            raise ValueError("All metrics.ranges_m values must be finite and > 0")
        return arr

    scenario = cfg.get("scenario", {}) if isinstance(cfg.get("scenario", {}), dict) else {}
    if "range_m" in scenario:
        r = float(scenario["range_m"])
        if not math.isfinite(r) or r <= 0.0:
            raise ValueError("scenario.range_m must be finite and > 0")
        return np.array([r], dtype=float)

    return np.array([10_000.0], dtype=float)