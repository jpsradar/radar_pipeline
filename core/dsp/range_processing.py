"""
core/dsp/range_processing.py

Minimal range-domain processing primitive: window + FFT (+ optional fftshift).

Purpose
-------
Provide a conservative FFT block used in multiple radar modalities:
- FMCW / stretch processing (FFT over fast-time beat samples)
- General spectral/range-bin transforms in synthetic pipelines

This module intentionally avoids committing to a waveform schema or a specific
physical range calibration. It provides the transform step only.

Scope (v1)
----------
- NumPy-only, deterministic.
- Windowing is supported for spectral control.
- Optional fftshift is provided for completeness (usually False for range).

Interfaces (Inputs / Outputs)
-----------------------------
Inputs:
- x: np.ndarray (real or complex), range/fast-time axis specified by config.
- cfg: RangeFFTConfig {axis, window, fftshift, n_fft}.
Outputs:
- range_fft -> complex np.ndarray
- range_power -> float np.ndarray |FFT|^2

Contracts and Invariants
------------------------
- Input must be non-empty; axis must be valid.
- If n_fft is specified, it must be an integer >= 1.
- Output shape matches input except for axis length if n_fft is used.

Exceptions
----------
- RangeProcessingError: invalid axis, invalid n_fft, empty input.

Dependencies
------------
- NumPy (required)
- core/dsp/windowing.py (internal)

How to run (developer)
----------------------
This module is not a CLI. Validate via:
    python -m pytest -q
"""


from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import numpy as np

from core.dsp.windowing import apply_window, WindowType


class RangeProcessingError(ValueError):
    """Raised when range processing inputs are invalid."""


@dataclass(frozen=True)
class RangeFFTConfig:
    """
    Configuration for range FFT.

    Attributes
    ----------
    axis : int
        Axis to transform (fast-time).
    window : WindowType
        Window kind applied before FFT.
    fftshift : bool
        If True, apply fftshift along the range axis (rare for range, but sometimes used).
    n_fft : int | None
        If provided, zero-pad or truncate to this FFT length.
    """
    axis: int = -1
    window: WindowType = "hann"
    fftshift: bool = False
    n_fft: Optional[int] = None


def range_fft(x: np.ndarray, *, cfg: RangeFFTConfig = RangeFFTConfig()) -> np.ndarray:
    """
    Compute a range FFT along cfg.axis.

    Parameters
    ----------
    x : np.ndarray
        Input samples.
    cfg : RangeFFTConfig
        Range FFT configuration.

    Returns
    -------
    np.ndarray
        Complex FFT output.
    """
    x = np.asarray(x)
    if x.size == 0:
        raise RangeProcessingError("Input array is empty")
    if not (-x.ndim <= cfg.axis < x.ndim):
        raise RangeProcessingError(f"axis out of bounds: axis={cfg.axis} for x.ndim={x.ndim}")

    xw = apply_window(x, axis=cfg.axis, kind=cfg.window)

    n_fft = cfg.n_fft
    if n_fft is not None:
        if not isinstance(n_fft, int) or n_fft < 1:
            raise RangeProcessingError(f"n_fft must be an integer >= 1, got {n_fft}")

    y = np.fft.fft(xw, n=n_fft, axis=cfg.axis)

    if cfg.fftshift:
        y = np.fft.fftshift(y, axes=(cfg.axis,))

    return y


def range_power(x: np.ndarray, *, cfg: RangeFFTConfig = RangeFFTConfig()) -> np.ndarray:
    """
    Convenience: compute range power spectrum |FFT|^2.
    """
    y = range_fft(x, cfg=cfg)
    return np.abs(y) ** 2