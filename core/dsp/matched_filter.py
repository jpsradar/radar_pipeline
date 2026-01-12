"""
core/dsp/matched_filter.py

Generic matched filtering primitives (reference-driven correlation).

Purpose
-------
Provide a waveform-agnostic matched filter building block:
- Construct matched filter taps from a reference signal (time-reversed conjugate).
- Apply matched filtering along a selected axis using:
    * FFT convolution (default; efficient)
    * Direct convolution (small signals / debugging)

Scope (v1)
----------
- NumPy-only (no SciPy).
- Waveform schema is not assumed; caller supplies reference samples explicitly.
- Output is "same-length" with respect to the filtered axis for predictable downstream use.

Interfaces (Inputs / Outputs)
-----------------------------
Inputs:
- reference: 1D array (real or complex)
- x: ND array, filter applied along `axis`
- mode: {"fft","direct"}
Outputs:
- make_matched_filter -> 1D complex taps
- apply_matched_filter -> ND complex output with same shape as x

Contracts and Invariants
------------------------
- reference must be 1D and non-empty.
- x must be non-empty; axis must be valid.
- No mutation of inputs; deterministic output for given inputs.

Exceptions
----------
- MatchedFilterError: invalid reference shape, empty input, invalid axis/mode.

Dependencies
------------
- NumPy (required)

How to run (developer)
----------------------
This module is not a CLI. Validate via:
    python -m pytest -q
"""

from __future__ import annotations

from typing import Literal
import numpy as np


class MatchedFilterError(ValueError):
    """Raised when matched filter inputs are invalid."""


Mode = Literal["fft", "direct"]


def make_matched_filter(reference: np.ndarray) -> np.ndarray:
    """
    Construct matched filter taps from a reference signal.

    Parameters
    ----------
    reference : np.ndarray
        Reference waveform samples (1D).

    Returns
    -------
    np.ndarray
        Matched filter taps (1D), complex.
    """
    r = np.asarray(reference)
    if r.ndim != 1:
        raise MatchedFilterError("reference must be 1D")
    if r.size == 0:
        raise MatchedFilterError("reference is empty")
    return np.conjugate(r[::-1])


def apply_matched_filter(
    x: np.ndarray,
    reference: np.ndarray,
    *,
    axis: int = -1,
    mode: Mode = "fft",
) -> np.ndarray:
    """
    Apply a matched filter along a specified axis.

    Parameters
    ----------
    x : np.ndarray
        Input data (ND).
    reference : np.ndarray
        Reference waveform (1D).
    axis : int
        Axis along which to filter.
    mode : {"fft","direct"}
        Convolution method.

    Returns
    -------
    np.ndarray
        Filtered array, same shape as x (uses "same" output length).
    """
    x = np.asarray(x)
    if x.size == 0:
        raise MatchedFilterError("Input array is empty")
    if not (-x.ndim <= axis < x.ndim):
        raise MatchedFilterError(f"axis out of bounds: axis={axis} for x.ndim={x.ndim}")

    h = make_matched_filter(reference).astype(complex, copy=False)
    n_h = int(h.size)

    # Move axis to the end for simpler implementation.
    x_m = np.moveaxis(x, axis, -1)
    n = int(x_m.shape[-1])

    if n == 0:
        raise MatchedFilterError("Input length along axis is 0")

    if str(mode).lower().strip() == "direct":
        # Direct convolution per vector (OK for small signals)
        y = np.empty_like(x_m, dtype=complex)
        it = np.nditer(np.zeros(x_m.shape[:-1], dtype=np.uint8), flags=["multi_index"])
        while not it.finished:
            idx = it.multi_index
            v = x_m[idx]
            full = np.convolve(v, h, mode="full")
            start = (n_h - 1) // 2
            y[idx] = full[start : start + n]
            it.iternext()
        return np.moveaxis(y, -1, axis)

    if str(mode).lower().strip() != "fft":
        raise MatchedFilterError(f"Unknown mode: {mode!r}")

    # FFT convolution (vectorized along leading dims)
    n_fft = int(2 ** int(np.ceil(np.log2(n + n_h - 1))))
    H = np.fft.fft(h, n=n_fft)

    X = np.fft.fft(x_m, n=n_fft, axis=-1)
    Y = X * H
    y_full = np.fft.ifft(Y, axis=-1)

    # "same" slicing: center the full convolution to length n
    start = (n_h - 1) // 2
    y_same = y_full[..., start : start + n]
    return np.moveaxis(y_same, -1, axis)