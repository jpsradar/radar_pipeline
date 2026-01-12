"""
core/detection/integration.py

Pulse integration primitives (coherent and noncoherent) for the radar pipeline.

Purpose
-------
Centralize integration math used by detection/performance layers:
- Coherent integration: sum complex samples across pulses (phase-aligned assumption).
- Noncoherent integration: sum detected energy across pulses (sum(|x|^2)).

This module is a pure-math dependency intended to be reused by:
- core/detection/detectors.py (Pd/Pfa models, decision statistics)
- core/simulation/* engines (when generating or validating integrated statistics)
- future DSP chains (RD-map processing and post-detection integration)

Scope (v1)
----------
- Deterministic NumPy operations only (no RNG, no I/O).
- Array-first APIs with explicit axis handling.
- No thresholding or Pd/Pfa computation (belongs in core/detection/thresholds.py).
- No waveform/ambiguity processing (belongs in core/dsp/*).

Interfaces (Inputs / Outputs)
-----------------------------
Inputs:
- x: np.ndarray (real or complex), pulses located along `axis`.
- mode: {"coherent", "noncoherent"} (wrapper API).
Outputs:
- np.ndarray integrated along the selected axis:
    * coherent_integrate -> complex sum
    * noncoherent_integrate_power -> real power sum
    * noncoherent_integrate_magnitude -> real magnitude sum

Contracts and Invariants
------------------------
- The input must be non-empty.
- Axis must be valid for the input dimensionality.
- Operations are vectorized and deterministic.

Exceptions
----------
- IntegrationError: raised for invalid shapes, empty inputs, or invalid mode/axis.

Dependencies
------------
- NumPy (required)

How to run (developer)
----------------------
This module is not a CLI. Validate via test suite:
    python -m pytest -q
"""


from __future__ import annotations

from typing import Literal, Tuple
import math

import numpy as np


class IntegrationError(ValueError):
    """Raised when integration inputs are invalid."""


IntegrationMode = Literal["coherent", "noncoherent"]


def coherent_integrate(x: np.ndarray, *, axis: int = 0, keepdims: bool = False) -> np.ndarray:
    """
    Coherently integrate complex samples across pulses.

    Parameters
    ----------
    x : np.ndarray
        Complex (preferred) or real samples. Pulses are along `axis`.
    axis : int
        Axis representing pulses.
    keepdims : bool
        If True, keep the integrated axis as length-1.

    Returns
    -------
    np.ndarray
        Complex sum along `axis`.

    Notes
    -----
    Coherent integration assumes phase alignment. If phase is random across pulses,
    coherent integration will not provide the expected gain.
    """
    x = np.asarray(x)
    if x.size == 0:
        raise IntegrationError("Input array is empty")
    return np.sum(x, axis=axis, keepdims=keepdims)


def noncoherent_integrate_power(x: np.ndarray, *, axis: int = 0, keepdims: bool = False) -> np.ndarray:
    """
    Noncoherently integrate power across pulses: sum(|x|^2).

    Parameters
    ----------
    x : np.ndarray
        Complex or real samples. Pulses are along `axis`.
    axis : int
        Axis representing pulses.
    keepdims : bool
        If True, keep the integrated axis as length-1.

    Returns
    -------
    np.ndarray
        Summed power along `axis`.
    """
    x = np.asarray(x)
    if x.size == 0:
        raise IntegrationError("Input array is empty")
    p = np.abs(x) ** 2
    return np.sum(p, axis=axis, keepdims=keepdims)


def noncoherent_integrate_magnitude(x: np.ndarray, *, axis: int = 0, keepdims: bool = False) -> np.ndarray:
    """
    Noncoherently integrate magnitude across pulses: sum(|x|).

    This is less common than power integration, but sometimes used in legacy chains.

    Returns
    -------
    np.ndarray
    """
    x = np.asarray(x)
    if x.size == 0:
        raise IntegrationError("Input array is empty")
    m = np.abs(x)
    return np.sum(m, axis=axis, keepdims=keepdims)


def integrate(
    x: np.ndarray,
    *,
    mode: IntegrationMode,
    axis: int = 0,
    keepdims: bool = False,
) -> np.ndarray:
    """
    Unified integration wrapper.

    Parameters
    ----------
    x : np.ndarray
        Input samples.
    mode : {"coherent","noncoherent"}
        Integration mode. For "noncoherent", we integrate power (sum(|x|^2)).
    axis : int
        Pulse axis.
    keepdims : bool
        Keep integrated axis.

    Returns
    -------
    np.ndarray
    """
    m = str(mode).lower().strip()
    if m == "coherent":
        return coherent_integrate(x, axis=axis, keepdims=keepdims)
    if m == "noncoherent":
        return noncoherent_integrate_power(x, axis=axis, keepdims=keepdims)
    raise IntegrationError(f"Unknown integration mode: {mode!r}")


def coherent_snr_gain_db(n_pulses: int) -> float:
    """
    Ideal coherent integration SNR gain in dB: 10*log10(N)

    Parameters
    ----------
    n_pulses : int
        Number of coherently integrated pulses (N >= 1).

    Returns
    -------
    float
        Gain in dB.
    """
    if not isinstance(n_pulses, int) or n_pulses < 1:
        raise IntegrationError(f"n_pulses must be an integer >= 1, got {n_pulses}")
    return float(10.0 * math.log10(float(n_pulses)))


def noncoherent_snr_gain_db(n_pulses: int) -> float:
    """
    Ideal noncoherent *energy* integration gain in dB: 10*log10(N)

    Notes
    -----
    For noncoherent integration under certain detection models, the effective Pd gain
    differs from pure 10*log10(N). This helper is intentionally "budget level".
    """
    return coherent_snr_gain_db(n_pulses)


def infer_pulse_axis(x: np.ndarray, *, prefer_axis: int = 0) -> int:
    """
    Convenience helper: infer a pulse axis for common shapes.

    This is intentionally conservative:
    - If array is 1D, return axis=0.
    - Otherwise return prefer_axis if valid, else 0.

    Returns
    -------
    int
    """
    x = np.asarray(x)
    if x.ndim <= 1:
        return 0
    if -x.ndim <= prefer_axis < x.ndim:
        return prefer_axis
    return 0