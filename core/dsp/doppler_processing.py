"""
core/dsp/doppler_processing.py

Minimal Doppler processing stage: window + FFT (+ optional fftshift).

Purpose
-------
Provide a reusable Doppler FFT primitive suitable for RD-map construction and
spot-validation workflows:
- Apply a selectable window along the Doppler axis (slow-time).
- Compute FFT with optional zero-padding/truncation.
- Optionally apply fftshift to center zero Doppler.

Scope (v1)
----------
- NumPy-only FFT path, deterministic and side-effect free.
- This is a building block (not a full RD pipeline orchestrator).
- No CFAR here; detection belongs in core/detection/*.

Interfaces (Inputs / Outputs)
-----------------------------
Inputs:
- x: np.ndarray (typically complex), Doppler axis specified by config.
- cfg: DopplerFFTConfig {axis, window, fftshift, n_fft}.
Outputs:
- doppler_fft -> complex np.ndarray spectrum
- doppler_power -> float np.ndarray |spectrum|^2

Contracts and Invariants
------------------------
- Input must be non-empty; axis must be valid.
- If n_fft is specified, it must be an integer >= 1.
- Windowing is applied deterministically before FFT.

Exceptions
----------
- DopplerProcessingError: invalid axis, invalid n_fft, empty input.

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
from typing import Any, Dict, Optional
import numpy as np

from core.dsp.windowing import apply_window, WindowType


class DopplerProcessingError(ValueError):
    """Raised when Doppler processing inputs are invalid."""


@dataclass(frozen=True)
class DopplerFFTConfig:
    """
    Configuration for Doppler FFT.

    Attributes
    ----------
    axis : int
        Axis to transform (slow-time / pulses).
    window : WindowType
        Window kind applied before FFT.
    fftshift : bool
        If True, apply fftshift along the Doppler axis.
    n_fft : int | None
        If provided, zero-pad or truncate to this FFT length.
    """
    axis: int = -1
    window: WindowType = "hann"
    fftshift: bool = True
    n_fft: Optional[int] = None


def doppler_fft(x: np.ndarray, *, cfg: DopplerFFTConfig = DopplerFFTConfig()) -> np.ndarray:
    """
    Compute Doppler FFT along cfg.axis.

    Parameters
    ----------
    x : np.ndarray
        Input data (typically complex), where cfg.axis is the pulse dimension.
    cfg : DopplerFFTConfig
        Doppler FFT configuration.

    Returns
    -------
    np.ndarray
        Complex Doppler spectrum (same shape as input except axis length may be n_fft).
    """
    x = np.asarray(x)
    if x.size == 0:
        raise DopplerProcessingError("Input array is empty")
    if not (-x.ndim <= cfg.axis < x.ndim):
        raise DopplerProcessingError(f"axis out of bounds: axis={cfg.axis} for x.ndim={x.ndim}")

    # Windowing (deterministic)
    xw = apply_window(x, axis=cfg.axis, kind=cfg.window)

    n_fft = cfg.n_fft
    if n_fft is not None:
        if not isinstance(n_fft, int) or n_fft < 1:
            raise DopplerProcessingError(f"n_fft must be an integer >= 1, got {n_fft}")

    y = np.fft.fft(xw, n=n_fft, axis=cfg.axis)

    if cfg.fftshift:
        y = np.fft.fftshift(y, axes=(cfg.axis,))

    return y


def doppler_power(x: np.ndarray, *, cfg: DopplerFFTConfig = DopplerFFTConfig()) -> np.ndarray:
    """
    Convenience: compute Doppler power spectrum |FFT|^2.

    Returns
    -------
    np.ndarray (float)
    """
    y = doppler_fft(x, cfg=cfg)
    return np.abs(y) ** 2