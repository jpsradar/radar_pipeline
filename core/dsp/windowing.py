"""
core/dsp/windowing.py

Window generation and application utilities (NumPy-only).

Purpose
-------
Provide standard DSP windows and a safe application helper to support:
- Doppler FFT windowing (slow-time)
- Range FFT windowing (fast-time)
- Future matched-filter / spectral analysis steps

Scope (v1)
----------
- NumPy-only implementations (no SciPy dependency).
- Deterministic window formulas with explicit handling of edge cases (n==1).
- Broadcasting-safe application along an arbitrary axis.

Interfaces (Inputs / Outputs)
-----------------------------
Inputs:
- make_window(n, kind): n >= 1, kind in {"rect","hann","hamming","blackman"}.
- apply_window(x, axis, window|kind): x is ND array; window length must match x.shape[axis].
Outputs:
- make_window -> np.ndarray shape (n,), dtype float
- apply_window -> np.ndarray same shape as x (window applied along axis)

Contracts and Invariants
------------------------
- No mutation of inputs.
- Window application must preserve shape.
- Invalid axis/window length raises an explicit exception.

Exceptions
----------
- WindowingError: invalid n, kind, axis, empty input, or mismatched window length.

Dependencies
------------
- NumPy (required)

How to run (developer)
----------------------
This module is not a CLI. Validate via:
    python -m pytest -q
"""

from __future__ import annotations

from typing import Literal, Optional
import math

import numpy as np


class WindowingError(ValueError):
    """Raised when window inputs are invalid."""


WindowType = Literal["rect", "hann", "hamming", "blackman"]


def make_window(n: int, *, kind: WindowType = "hann") -> np.ndarray:
    """
    Create a DSP window.

    Parameters
    ----------
    n : int
        Window length (n >= 1).
    kind : {"rect","hann","hamming","blackman"}
        Window type.

    Returns
    -------
    np.ndarray
        Window vector of shape (n,), dtype=float64.
    """
    if not isinstance(n, int) or n < 1:
        raise WindowingError(f"n must be an integer >= 1, got {n}")

    k = str(kind).lower().strip()
    if k == "rect":
        return np.ones((n,), dtype=float)

    # For the standard definitions below, handle the n==1 edge cleanly.
    if n == 1:
        return np.ones((1,), dtype=float)

    i = np.arange(n, dtype=float)
    denom = float(n - 1)

    if k == "hann":
        return 0.5 - 0.5 * np.cos(2.0 * math.pi * i / denom)

    if k == "hamming":
        return 0.54 - 0.46 * np.cos(2.0 * math.pi * i / denom)

    if k == "blackman":
        return (
            0.42
            - 0.5 * np.cos(2.0 * math.pi * i / denom)
            + 0.08 * np.cos(4.0 * math.pi * i / denom)
        )

    raise WindowingError(f"Unknown window kind: {kind!r}")


def apply_window(
    x: np.ndarray,
    *,
    axis: int = -1,
    window: Optional[np.ndarray] = None,
    kind: WindowType = "hann",
) -> np.ndarray:
    """
    Apply a window along a given axis.

    Parameters
    ----------
    x : np.ndarray
        Input array (real or complex).
    axis : int
        Axis along which to apply the window.
    window : np.ndarray | None
        If provided, must have length equal to x.shape[axis].
    kind : WindowType
        Used only if window is None.

    Returns
    -------
    np.ndarray
        Windowed array (same shape as x).
    """
    x = np.asarray(x)
    if x.size == 0:
        raise WindowingError("Input array is empty")
    if x.ndim == 0:
        return x

    if not (-x.ndim <= axis < x.ndim):
        raise WindowingError(f"axis out of bounds: axis={axis} for x.ndim={x.ndim}")

    n = int(x.shape[axis])
    w = np.asarray(window, dtype=float) if window is not None else make_window(n, kind=kind)
    if w.shape != (n,):
        raise WindowingError(f"window must have shape ({n},), got {w.shape}")

    # Reshape for broadcasting
    shape = [1] * x.ndim
    shape[axis] = n
    w_b = w.reshape(shape)

    return x * w_b