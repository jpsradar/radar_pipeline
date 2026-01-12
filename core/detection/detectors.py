"""
core/detection/detectors.py

Detection decision logic (fixed threshold and CFAR).

This module is intentionally "dumb" in the best sense:
- It does NOT compute SNR budgets or waveform processing.
- It does NOT estimate Pd (that's model-based analytics in core/simulation).
- It ONLY applies decision rules given a test statistic and a threshold model.

Why this exists
---------------
In a professional pipeline, "threshold math" and "decision logic" must be separated:
- thresholds.py defines how thresholds are derived (analytical) or scaled (CFAR alpha)
- cfar.py defines how adaptive noise estimates are formed
- detectors.py defines how to apply the above to produce detections consistently

Inputs (typical)
----------------
- test_stat: ndarray of nonnegative values (power/energy statistic)
- threshold: scalar or ndarray (same broadcast shape as test_stat)
- For CFAR sliding: a 1D vector of powers + (n_train, n_guard, pfa)

Outputs
-------
- detections: boolean ndarray
- threshold_used: ndarray (same shape as test_stat)
- metadata dict (alpha, config, etc.)

CLI usage
---------
This is a library module; it is called by DSP chains or by validation scripts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union, Literal
import math

import numpy as np

from core.detection.thresholds import threshold_energy, threshold_scale_ca_cfar
from core.detection.cfar import ca_cfar_detect, ca_cfar_detect_1d_sliding


class DetectorError(ValueError):
    """Raised when detector inputs are invalid."""


DetectorType = Literal["fixed_energy", "ca_cfar_1d_sliding"]


@dataclass(frozen=True)
class DetectionResult:
    """
    Standard detection output container.

    Attributes
    ----------
    detections : np.ndarray
        Boolean array, True where detection declared.
    threshold : np.ndarray
        Threshold array used for the decision, same shape as detections/test_stat.
    meta : dict
        Useful metadata (alpha, mode, dof, pfa, window sizes, etc.).
    """
    detections: np.ndarray
    threshold: np.ndarray
    meta: Dict[str, Any]


# ---------------------------------------------------------------------
# Fixed-threshold detectors (analytical threshold)
# ---------------------------------------------------------------------

def detect_fixed_energy(
    *,
    test_stat: np.ndarray,
    pfa: float,
    n_pulses: int = 1,
    integration: Literal["noncoherent", "coherent"] = "noncoherent",
) -> DetectionResult:
    """
    Apply a fixed energy threshold to a test statistic.

    Parameters
    ----------
    test_stat : np.ndarray
        Nonnegative test statistic values (e.g., energy per cell).
    pfa : float
        Desired false alarm probability in (0,1) for the chosen statistical model.
    n_pulses : int
        Number of pulses integrated (used by noncoherent threshold model).
    integration : {"noncoherent", "coherent"}
        Selects threshold model.

    Returns
    -------
    DetectionResult
        Includes threshold array and metadata.
    """
    x = _as_nonnegative_finite_array(test_stat, name="test_stat")

    thr_res = threshold_energy(pfa=pfa, n_pulses=n_pulses, mode=integration)
    thr = np.full_like(x, fill_value=float(thr_res.threshold), dtype=float)

    det = x > thr
    meta = {
        "detector": "fixed_energy",
        "pfa": float(pfa),
        "integration": str(integration),
        "n_pulses": int(n_pulses),
        "dof": int(thr_res.dof),
        "threshold_scalar": float(thr_res.threshold),
        "notes": thr_res.notes,
    }
    return DetectionResult(detections=det, threshold=thr, meta=meta)


# ---------------------------------------------------------------------
# CFAR detectors
# ---------------------------------------------------------------------

def detect_ca_cfar_1d_sliding(
    *,
    x_power: np.ndarray,
    pfa: float,
    n_train: int,
    n_guard: int = 0,
) -> DetectionResult:
    """
    Apply 1D sliding CA-CFAR to a vector of power values.

    Use cases
    ---------
    - Range line CFAR (per Doppler bin)
    - Doppler line CFAR (per range bin)
    - Any 1D CFAR pre/post-processing step

    Parameters
    ----------
    x_power : np.ndarray
        1D array of nonnegative power values.
    pfa : float
        Desired false alarm probability per CUT.
    n_train : int
        Number of training cells on each side (total Nref = 2*n_train).
    n_guard : int
        Number of guard cells on each side.

    Returns
    -------
    DetectionResult
        detections, threshold array, and metadata (alpha, window sizes).
    """
    x = _as_nonnegative_finite_array(x_power, name="x_power")
    if x.ndim != 1:
        raise DetectorError("detect_ca_cfar_1d_sliding requires a 1D array")

    det, alpha, thr = ca_cfar_detect_1d_sliding(x=x, pfa=pfa, n_train=n_train, n_guard=n_guard)

    meta = {
        "detector": "ca_cfar_1d_sliding",
        "pfa": float(pfa),
        "n_train": int(n_train),
        "n_guard": int(n_guard),
        "n_ref_total": int(2 * n_train),
        "alpha": float(alpha),
        "edge_policy": "no-detection (False) and threshold=NaN where window incomplete",
    }
    return DetectionResult(detections=det, threshold=thr, meta=meta)


# ---------------------------------------------------------------------
# Generic dispatcher (optional convenience)
# ---------------------------------------------------------------------

def run_detector(
    detector: DetectorType,
    *,
    test_stat: np.ndarray,
    pfa: float,
    n_pulses: int = 1,
    integration: Literal["noncoherent", "coherent"] = "noncoherent",
    n_train: int = 16,
    n_guard: int = 0,
) -> DetectionResult:
    """
    Convenience dispatcher for detector selection.

    This is handy for configs that specify detector type.
    """
    det_name = str(detector).lower().strip()

    if det_name == "fixed_energy":
        return detect_fixed_energy(test_stat=test_stat, pfa=pfa, n_pulses=n_pulses, integration=integration)

    if det_name == "ca_cfar_1d_sliding":
        return detect_ca_cfar_1d_sliding(x_power=test_stat, pfa=pfa, n_train=n_train, n_guard=n_guard)

    raise DetectorError(f"Unknown detector: {detector}")


# ---------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------

def _as_nonnegative_finite_array(arr: np.ndarray, *, name: str) -> np.ndarray:
    """Convert input to float ndarray and validate it's finite and nonnegative."""
    x = np.asarray(arr, dtype=float)
    if np.any(~np.isfinite(x)):
        raise DetectorError(f"{name} must be finite")
    if np.any(x < 0.0):
        raise DetectorError(f"{name} must be nonnegative")
    return x