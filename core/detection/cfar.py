"""
core/detection/cfar.py

CFAR (Constant False Alarm Rate) detectors for homogeneous clutter/noise.

This module provides minimal, correct CA-CFAR building blocks for:
- Computing the CA-CFAR threshold scaling factor alpha given Pfa and N reference cells
- Applying CA-CFAR to 1D CUT sequences (for range OR Doppler lines)
- Providing deterministic behavior suitable for golden tests

Assumptions (v1)
----------------
- Reference cell powers are i.i.d. exponential (i.e., complex Gaussian noise -> |z|^2).
- CA-CFAR uses the mean of N reference cells (training cells).
- No guard cells are modeled inside apply_* yet (you can handle them outside by slicing).

Definitions
-----------
- CUT power: x_cut
- Reference powers: x_ref[0..N-1]
- Noise estimate: z = mean(x_ref)
- Threshold: T = alpha * z
- Declare detection if x_cut > T

Alpha for CA-CFAR (exponential assumption)
------------------------------------------
For N reference cells:
    Pfa = (1 + alpha / N)^(-N)
=>  alpha = N * (Pfa^(-1/N) - 1)

Inputs
------
- pfa: desired false alarm probability per CUT
- n_ref: number of reference cells used in the average
- x_cut: scalar or array of CUT powers (nonnegative)
- x_ref: array of reference cell powers

Outputs
-------
- alpha: threshold scaling factor
- detections: boolean array where True indicates x_cut > alpha * mean(x_ref)

CLI usage
---------
This is a library module; it is called by detection pipelines and tests.
"""

from __future__ import annotations

from typing import Optional, Tuple, Union
import math

import numpy as np


class CFARConfigError(ValueError):
    """Raised when CFAR configuration inputs are invalid."""


def ca_cfar_alpha(*, pfa: float, n_ref: int) -> float:
    """
    Compute CA-CFAR scaling factor alpha for exponential clutter/noise.

    Parameters
    ----------
    pfa : float
        Desired probability of false alarm in (0, 1).
    n_ref : int
        Number of reference (training) cells used in the average, must be >= 1.

    Returns
    -------
    float
        Alpha scaling factor.
    """
    if not isinstance(n_ref, int) or n_ref < 1:
        raise CFARConfigError(f"n_ref must be an integer >= 1, got {n_ref}")
    if not (isinstance(pfa, (int, float)) and math.isfinite(float(pfa))):
        raise CFARConfigError(f"pfa must be a finite number in (0,1), got {pfa}")
    pfa_f = float(pfa)
    if not (0.0 < pfa_f < 1.0):
        raise CFARConfigError(f"pfa must be in (0,1), got {pfa_f}")

    # CA-CFAR alpha under exponential assumption:
    # Pfa = (1 + alpha/N)^(-N) => alpha = N*(Pfa^(-1/N) - 1)
    return float(n_ref * (pfa_f ** (-1.0 / n_ref) - 1.0))


def ca_cfar_detect(
    *,
    x_cut: Union[float, np.ndarray],
    x_ref: np.ndarray,
    pfa: float,
    axis: Optional[int] = None,
) -> Tuple[np.ndarray, float, np.ndarray]:
    """
    Apply CA-CFAR detection given CUT power(s) and reference window powers.

    Parameters
    ----------
    x_cut
        CUT power(s). Scalar or array.
    x_ref
        Reference window powers. Must be nonnegative. If x_cut is array, x_ref must
        be broadcastable against x_cut when averaging along 'axis'.
    pfa
        Desired Pfa in (0,1).
    axis
        Axis along which to average reference cells. If None, x_ref is flattened.

    Returns
    -------
    detections : np.ndarray
        Boolean array indicating detections (x_cut > threshold).
    alpha : float
        CA-CFAR scaling factor.
    threshold : np.ndarray
        Threshold array (alpha * noise_estimate), same shape as x_cut.
    """
    x_ref = np.asarray(x_ref, dtype=float)
    if np.any(x_ref < 0.0) or np.any(~np.isfinite(x_ref)):
        raise CFARConfigError("x_ref must be finite and nonnegative")

    # Determine N reference cells
    n_ref = x_ref.shape[axis] if axis is not None else x_ref.size
    alpha = ca_cfar_alpha(pfa=pfa, n_ref=int(n_ref))

    # Noise estimate
    z = np.mean(x_ref, axis=axis) if axis is not None else float(np.mean(x_ref))

    x_cut_arr = np.asarray(x_cut, dtype=float)
    if np.any(x_cut_arr < 0.0) or np.any(~np.isfinite(x_cut_arr)):
        raise CFARConfigError("x_cut must be finite and nonnegative")

    threshold = alpha * z
    detections = x_cut_arr > threshold
    return detections, alpha, np.asarray(threshold, dtype=float)


def ca_cfar_detect_1d_sliding(
    *,
    x: np.ndarray,
    pfa: float,
    n_train: int,
    n_guard: int = 0,
) -> Tuple[np.ndarray, float, np.ndarray]:
    """
    1D sliding CA-CFAR over a power vector.

    For each CUT index i, reference cells are taken from:
      left:  [i - n_guard - n_train, ..., i - n_guard - 1]
      right: [i + n_guard + 1, ..., i + n_guard + n_train]

    Edge handling
    -------------
    - Indices without a full reference window are marked as False (no detection)
      and their threshold is NaN.

    Parameters
    ----------
    x : np.ndarray
        1D array of power values (nonnegative).
    pfa : float
        Desired false alarm probability per CUT.
    n_train : int
        Number of training cells on EACH side (total Nref = 2*n_train).
    n_guard : int
        Number of guard cells on EACH side.

    Returns
    -------
    detections : np.ndarray
        Boolean detections (same length as x).
    alpha : float
        Scaling factor computed for Nref = 2*n_train.
    threshold : np.ndarray
        Threshold array (same length as x), NaN where undefined.
    """
    x = np.asarray(x, dtype=float)
    if x.ndim != 1:
        raise CFARConfigError("x must be a 1D array")
    if np.any(x < 0.0) or np.any(~np.isfinite(x)):
        raise CFARConfigError("x must be finite and nonnegative")
    if not isinstance(n_train, int) or n_train < 1:
        raise CFARConfigError(f"n_train must be integer >= 1, got {n_train}")
    if not isinstance(n_guard, int) or n_guard < 0:
        raise CFARConfigError(f"n_guard must be integer >= 0, got {n_guard}")

    n_ref_total = 2 * n_train
    alpha = ca_cfar_alpha(pfa=pfa, n_ref=n_ref_total)

    det = np.zeros_like(x, dtype=bool)
    thr = np.full_like(x, fill_value=np.nan, dtype=float)

    for i in range(len(x)):
        left_start = i - n_guard - n_train
        left_end = i - n_guard
        right_start = i + n_guard + 1
        right_end = i + n_guard + n_train + 1

        if left_start < 0 or right_end > len(x):
            continue  # Not enough reference cells

        ref_left = x[left_start:left_end]
        ref_right = x[right_start:right_end]
        ref = np.concatenate([ref_left, ref_right])

        z = float(np.mean(ref))
        t = alpha * z
        thr[i] = t
        det[i] = x[i] > t

    return det, alpha, thr