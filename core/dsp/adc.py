"""
core/dsp/adc.py

Simple ADC modeling helpers (clipping and uniform quantization).

Purpose
-------
Provide deterministic ADC primitives useful for controlled DSP experiments:
- Hard clipping to a specified full-scale.
- Uniform mid-tread quantization for real or complex signals.

These utilities enable "what-if" studies (dynamic range, saturation, quantization noise)
without introducing stochastic or hardware-specific complexity.

Scope (v1)
----------
- Deterministic operations only (no RNG, no I/O).
- Works on real or complex arrays (complex handled per component).
- Not wired into performance engines by default; intended as a DSP building block.

Interfaces (Inputs / Outputs)
-----------------------------
Inputs:
- clip(x, full_scale): full_scale > 0
- quantize_uniform(x, n_bits, full_scale): n_bits >= 1, full_scale > 0
- adc_apply(x, cfg): convenience wrapper (clip + quantize)
Outputs:
- Arrays with same shape as input; dtype float/complex float

Contracts and Invariants
------------------------
- Inputs are not mutated.
- Shape is preserved.
- Invalid configuration raises explicit exceptions.

Exceptions
----------
- ADCError: invalid n_bits/full_scale, invalid inputs.

Dependencies
------------
- NumPy (required)

How to run (developer)
----------------------
This module is not a CLI. Validate via:
    python -m pytest -q
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


class ADCError(ValueError):
    """Raised when ADC modeling inputs are invalid."""


@dataclass(frozen=True)
class ADCConfig:
    """
    ADC configuration.

    Attributes
    ----------
    n_bits : int
        Number of quantization bits (>= 1).
    full_scale : float
        Full-scale magnitude for clipping (must be > 0). For complex, applies per component.
    """
    n_bits: int = 12
    full_scale: float = 1.0


def clip(x: np.ndarray, *, full_scale: float) -> np.ndarray:
    """
    Hard clip a real or complex signal to [-full_scale, +full_scale] (per component).

    Parameters
    ----------
    x : np.ndarray
        Input.
    full_scale : float
        Clip value (> 0).

    Returns
    -------
    np.ndarray
    """
    x = np.asarray(x)
    if not np.isfinite(full_scale) or float(full_scale) <= 0.0:
        raise ADCError(f"full_scale must be finite and > 0, got {full_scale}")

    fs = float(full_scale)

    if np.iscomplexobj(x):
        re = np.clip(np.real(x), -fs, fs)
        im = np.clip(np.imag(x), -fs, fs)
        return re + 1j * im

    return np.clip(x, -fs, fs)


def quantize_uniform(x: np.ndarray, *, n_bits: int, full_scale: float) -> np.ndarray:
    """
    Uniform mid-tread quantizer with clipping.

    Parameters
    ----------
    x : np.ndarray
        Real or complex input.
    n_bits : int
        Bits (>= 1).
    full_scale : float
        Full scale (> 0).

    Returns
    -------
    np.ndarray
        Quantized signal (float/complex float).
    """
    x = np.asarray(x)

    if not isinstance(n_bits, int) or n_bits < 1:
        raise ADCError(f"n_bits must be an integer >= 1, got {n_bits}")
    if not np.isfinite(full_scale) or float(full_scale) <= 0.0:
        raise ADCError(f"full_scale must be finite and > 0, got {full_scale}")

    fs = float(full_scale)
    levels = int(2**n_bits)

    # Step size across [-fs, fs]
    # mid-tread quantizer: includes 0 as a reconstruction level
    delta = (2.0 * fs) / float(levels)

    def q_real(u: np.ndarray) -> np.ndarray:
        u = np.clip(u, -fs, fs)
        return delta * np.round(u / delta)

    if np.iscomplexobj(x):
        re = q_real(np.real(x))
        im = q_real(np.imag(x))
        return re + 1j * im

    return q_real(x)


def adc_apply(x: np.ndarray, *, cfg: ADCConfig = ADCConfig()) -> np.ndarray:
    """
    Apply simple ADC model: clip + quantize.

    Returns
    -------
    np.ndarray
    """
    y = clip(x, full_scale=cfg.full_scale)
    return quantize_uniform(y, n_bits=cfg.n_bits, full_scale=cfg.full_scale)