"""
core/simulation/model_based.py

Fast physical-statistical engine (model-based) for radar performance trade studies.

Purpose
-------
Provide a deterministic, high-speed performance engine suitable for:
- parameter sweeps / trade studies (run-time dominated by pure math, not simulation),
- contract-driven validation against Monte Carlo experiments,
- generating report-ready metrics that are traceable and reproducible.

What this module computes (v1 baseline)
---------------------------------------
1) Power budget (monostatic radar equation)
   - Received power vs range:
       Pr(R) = Pt * Gt * Gr * (lambda^2 * sigma) / ( (4*pi)^3 * R^4 * L )

2) Receiver noise power (thermal)
   - Noise power:
       N = k * T * B * F
     where:
       k : Boltzmann constant
       T : receiver noise temperature [K]
       B : receiver bandwidth [Hz]
       F : noise factor (linear), from NF [dB]

3) SNR and SINR (noise-limited + optional interference)
   - SNR(R)  = Pr(R) / N
   - SINR(R) = Pr(R) / (N + I)
     where I is an optional additive interference power at receiver input.

4) Detection performance for an energy detector (square-law), using exact statistics
   - Noncoherent integration of N pulses:
       T = sum_{i=1..N} |z_i|^2
       Under H0: T ~ chi2(df=2N)
       Under H1: T ~ ncx2(df=2N, nc = 2*N*SNR_per_pulse)
   - Coherent-like baseline (intentional simplification for fast sweeps):
       df = 2 energy detector with boosted SNR_total = N * SNR_per_pulse

Detector Contract (for model-vs-experiment comparisons)
-------------------------------------------------------
When detection.pfa is provided, this engine MUST emit an explicit detector contract under:
    metrics["detection"]["contract"]

The contract is a publishable, replication-ready specification:
- test statistic definition and normalization
- H0 / H1 distribution family and parameters (df, noncentrality expression)
- threshold used
- integration mode and pulse count
- the resulting Pd array

This explicit contract is required so external experiments (Monte Carlo / lab / notebook)
can replicate the exact same detector and compare apples-to-apples.

Validity Contract (repo-wide)
-----------------------------
All runnable engines must emit a top-level:
    metrics["validity"]

This is a compact declaration of:
- statistical model assumptions,
- clutter/interference assumptions,
- known modeling limits.

The intent is traceability and professional honesty, not "marketing".

Interference model (v1)
-----------------------
Optional config block:

    interference:
      model: noise_like_jammer
      jnr_db: <number>

Interpretation:
- "noise-like jammer" modeled as additive white interference at receiver input.
- JNR defined at receiver input as:
      JNR = I / N
  so:
      I = N * JNR_linear
- v1: constant vs range (input-referred).

Important:
- This is not waveform-level jamming. It is a power-domain impairment hook.

Dependencies
------------
- NumPy
- SciPy (chi2 and ncx2 survival functions)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional
import math
import inspect

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

from core.geometry.dwell import make_cpi, make_dwell, cpis_per_second
from core.geometry.counts import make_rd_grid, make_scan_geometry
from core.detection.far_conversion import FARInputs, convert_pfa_to_far

from core.contracts.validity import validity_for_model_based


# ----------------------------
# Internal data containers
# ----------------------------

@dataclass(frozen=True)
class BudgetTerms:
    """
    Holds intermediate deterministic terms for traceability.

    Notes
    -----
    - Keeping these terms explicit makes reports audit-friendly.
    - These terms are purely deterministic functions of config.
    """
    wavelength_m: float
    gt_lin: float
    gr_lin: float
    losses_lin: float
    noise_power_w: float


# ----------------------------
# Small internal helper
# ----------------------------

def _call_with_supported_kwargs(fn: Any, **kwargs: Any) -> Any:
    """
    Call a function using only the kwargs that it actually accepts.

    Why this exists
    ---------------
    `validity_for_model_based(...)` is a contract helper that may evolve.
    We want the engine to stay robust across minor signature changes, while still
    passing meaningful context whenever possible.

    Behavior
    --------
    - Filters kwargs by the function signature.
    - Calls fn(**filtered_kwargs).
    - Lets exceptions propagate (a broken validity contract should fail loudly).
    """
    sig = inspect.signature(fn)
    filtered = {k: v for k, v in kwargs.items() if k in sig.parameters}
    return fn(**filtered)


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
        Reserved for future stochastic features. Not used in v1 (engine is deterministic).

    Returns
    -------
    dict
        Metrics dictionary suitable to be written as metrics.json.
    """
    _ = seed  # deterministic v1

    # --- Required sections ---
    radar = _require_section(cfg, "radar")
    antenna = _require_section(cfg, "antenna")
    receiver = _require_section(cfg, "receiver")
    target = _require_section(cfg, "target")

    fc_hz = _require_pos_float(radar, "fc_hz")
    tx_power_w = _require_pos_float(radar, "tx_power_w")

    # PRF is used for CPI timing + FAR conversion.
    prf_hz = _require_pos_float(radar, "prf_hz", default=1_000.0)

    gain_tx_db = _require_float(antenna, "gain_tx_db", default=0.0)
    gain_rx_db = _require_float(antenna, "gain_rx_db", default=gain_tx_db)

    bw_hz = _require_pos_float(receiver, "bw_hz")
    nf_db = _require_nonneg_float(receiver, "nf_db", default=0.0)
    temperature_k = _require_pos_float(receiver, "temperature_k", default=290.0)

    sigma_sqm = _require_pos_float(target, "rcs_sqm")

    env = cfg.get("environment", {}) if isinstance(cfg.get("environment", {}), dict) else {}
    system_losses_db = _require_nonneg_float(env, "system_losses_db", default=0.0)

    # Ranges: metrics.ranges_m preferred.
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
    # Geometry + counts (explicit counts enable system-level FAR metrics)
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

    # -------------------------
    # Deterministic budget math
    # -------------------------
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

    # Baseline SNR (noise-only denominator)
    snr_lin = pr_w / terms.noise_power_w
    snr_db = lin_to_db_power(np.maximum(snr_lin, np.finfo(float).tiny))

    # -------------------------
    # Optional interference hook
    # -------------------------
    interference = cfg.get("interference", None) if isinstance(cfg.get("interference", None), dict) else None

    i_power_w = 0.0
    interference_model = None
    jnr_db = None

    if interference is not None:
        interference_model = str(interference.get("model", "")).strip()
        if interference_model == "noise_like_jammer":
            jnr_db = _require_float(interference, "jnr_db")
            jnr_lin = db_to_lin_power(float(jnr_db))
            if not (math.isfinite(jnr_lin) and jnr_lin >= 0.0):
                raise ValueError(f"interference.jnr_db produced invalid linear JNR: {jnr_lin}")
            # Input-referred additive interference power:
            i_power_w = float(terms.noise_power_w * jnr_lin)
        else:
            raise ValueError(f"Unsupported interference.model: {interference_model!r}")

    # Always emit SINR arrays:
    denom_w = float(terms.noise_power_w + i_power_w)
    sinr_lin = pr_w / denom_w
    sinr_db = lin_to_db_power(np.maximum(sinr_lin, np.finfo(float).tiny))

    # Effective SNR used by the detector:
    eff_snr_lin = sinr_lin if i_power_w > 0.0 else snr_lin

    # -------------------------------------------------------------
    # Detection performance (optional): uses effective SNR definition
    # -------------------------------------------------------------
    pd = None
    threshold = None
    far = None
    detection_contract: Optional[Dict[str, Any]] = None

    if pfa is not None:
        pfa_f = float(pfa)
        if not (0.0 < pfa_f < 1.0):
            raise ValueError(f"detection.pfa must be in (0, 1), got {pfa_f}")

        detection_contract = {
            "pfa_target": pfa_f,
            "integration": integration,
            "n_pulses": int(n_pulses),
            "snr_definition": "SINR_per_pulse if interference present else SNR_per_pulse",
            "statistic": {
                "name": "energy_sum",
                "definition": "T = sum_{i=1..N} |z_i|^2 (power/energy domain)",
                "normalization": "unit-variance convention for chi-square family (SciPy chi2/ncx2)",
            },
            "h0": {},
            "h1": {},
        }

        if integration == "noncoherent":
            threshold = _threshold_noncoherent(pfa=pfa_f, n_pulses=n_pulses)
            pd = _pd_noncoherent(threshold=threshold, snr_lin=eff_snr_lin, n_pulses=n_pulses)
            detection_contract["h0"] = {"family": "chi2", "df": int(2 * n_pulses)}
            detection_contract["h1"] = {
                "family": "ncx2",
                "df": int(2 * n_pulses),
                "noncentrality": "nc = 2*N*SNR_per_pulse (SNR interpreted per snr_definition)",
            }
        else:
            threshold = _threshold_coherent(pfa=pfa_f)
            pd = _pd_coherent(threshold=threshold, snr_lin=eff_snr_lin, n_pulses=n_pulses)
            detection_contract["h0"] = {"family": "chi2", "df": 2}
            detection_contract["h1"] = {
                "family": "ncx2",
                "df": 2,
                "noncentrality": "nc = 2*SNR_total, SNR_total = N*SNR_per_pulse (per snr_definition)",
            }

        detection_contract["threshold"] = float(threshold)

        far_inputs = FARInputs(
            pfa=pfa_f,
            domain="rd_cell",
            cells_per_cpi=rd_grid.cells_per_cpi,
            cpis_per_second=cpis_rate_hz,
            beams_per_scan=scan_geom.beams_per_scan,
            scans_per_second=scan_geom.scans_per_second,
        )
        far = convert_pfa_to_far(far_inputs)

    # -------------------------
    # Metrics packaging
    # -------------------------
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
            "interference_model": interference_model,
            "jnr_db": float(jnr_db) if jnr_db is not None else None,
        },
        "ranges_m": ranges_m.tolist(),
        "budget": {
            "wavelength_m": terms.wavelength_m,
            "gt_lin": terms.gt_lin,
            "gr_lin": terms.gr_lin,
            "losses_lin": terms.losses_lin,
            "noise_power_w": terms.noise_power_w,
            "noise_power_dbw": lin_to_db_power(terms.noise_power_w),
            "interference_power_w": float(i_power_w),
            "interference_power_dbw": lin_to_db_power(max(float(i_power_w), np.finfo(float).tiny)),
            "noise_plus_interference_w": float(terms.noise_power_w + i_power_w),
        },
        "received_power_w": pr_w.tolist(),
        "received_power_dbw": lin_to_db_power(np.maximum(pr_w, np.finfo(float).tiny)).tolist(),
        "snr_lin": snr_lin.tolist(),
        "snr_db": snr_db.tolist(),
        "sinr_lin": sinr_lin.tolist(),
        "sinr_db": sinr_db.tolist(),
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

    # -------------------------
    # Validity contract (top-level)
    # -------------------------
    # We pass a rich context dictionary; the helper will accept what it wants.
    metrics["validity"] = _call_with_supported_kwargs(
        validity_for_model_based,
        cfg=cfg,
        engine="model_based",
        integration=integration,
        n_pulses=int(n_pulses),
        has_interference=bool(i_power_w > 0.0),
        interference_model=interference_model,
        jnr_db=float(jnr_db) if jnr_db is not None else None,
        snr_kind=("sinr" if i_power_w > 0.0 else "snr"),
    )

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
# Detection math (exact stats)
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