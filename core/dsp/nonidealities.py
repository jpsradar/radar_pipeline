"""
core/dsp/nonidealities.py

Deterministic baseband non-idealities (DC offset, phase rotation, IQ imbalance).

Purpose
-------
Provide small, explicit signal transformations used in DSP validation:
- Add complex DC offset.
- Apply constant phase rotation.
- Apply a simple IQ imbalance model (gain/phase skew) for controlled sensitivity checks.

These are "knobs" for deterministic experiments and regression testing, not a full RF impairment suite.

Scope (v1)
----------
- Deterministic transforms only (no random phase noise, no oscillator models).
- No I/O, no plotting.
- Intended as reusable building blocks for future DSP chains and spot-validation.

Interfaces (Inputs / Outputs)
-----------------------------
Inputs:
- add_dc_offset(x, dc)
- rotate_phase(x, phase_rad)
- apply_iq_imbalance(x, cfg=IQImbalance(gain_imbalance_db, phase_imbalance_deg))
Outputs:
- Arrays with same shape as input; complex dtype for complex operations.

Contracts and Invariants
------------------------
- Inputs are not mutated.
- Shape is preserved.
- Parameters must be finite and validated.

Exceptions
----------
- NonidealityError: invalid parameters (non-finite values).

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
import math
import numpy as np


class NonidealityError(ValueError):
    """Raised when non-ideality configuration is invalid."""


def add_dc_offset(x: np.ndarray, *, dc: complex) -> np.ndarray:
    """
    Add a complex DC offset to a signal.

    Parameters
    ----------
    x : np.ndarray
        Complex input.
    dc : complex
        DC offset.

    Returns
    -------
    np.ndarray
    """
    x = np.asarray(x)
    return x.astype(complex, copy=False) + complex(dc)


def rotate_phase(x: np.ndarray, *, phase_rad: float) -> np.ndarray:
    """
    Apply a constant phase rotation e^{j*phase}.

    Parameters
    ----------
    x : np.ndarray
        Complex input.
    phase_rad : float
        Phase in radians.

    Returns
    -------
    np.ndarray
    """
    if not math.isfinite(float(phase_rad)):
        raise NonidealityError(f"phase_rad must be finite, got {phase_rad}")
    x = np.asarray(x).astype(complex, copy=False)
    ph = float(phase_rad)
    return x * (math.cos(ph) + 1j * math.sin(ph))


@dataclass(frozen=True)
class IQImbalance:
    """
    Simple IQ imbalance model.

    Attributes
    ----------
    gain_imbalance_db : float
        Gain imbalance between I and Q in dB (power-ish; we apply as amplitude ratio).
    phase_imbalance_deg : float
        Quadrature phase error in degrees (0 means perfect 90 degrees).
    """
    gain_imbalance_db: float = 0.0
    phase_imbalance_deg: float = 0.0


def apply_iq_imbalance(x: np.ndarray, *, cfg: IQImbalance = IQImbalance()) -> np.ndarray:
    """
    Apply a simple IQ imbalance model.

    Model (approximate)
    -------------------
    - Apply amplitude scaling between I and Q by an amplitude ratio derived from dB.
    - Apply a small phase skew between I and Q.

    This is a pedagogical model intended for controlled experiments, not a hardware-accurate one.
    """
    x = np.asarray(x).astype(complex, copy=False)

    g_db = float(cfg.gain_imbalance_db)
    ph_deg = float(cfg.phase_imbalance_deg)

    if not (math.isfinite(g_db) and math.isfinite(ph_deg)):
        raise NonidealityError("IQImbalance parameters must be finite")

    # Convert dB to amplitude ratio (20*log10 for amplitude)
    g_amp = 10.0 ** (g_db / 20.0)
    ph = math.radians(ph_deg)

    i = np.real(x) * g_amp
    q = np.imag(x)

    # Apply phase skew by mixing I into Q (small-angle approximation not assumed; apply exact rotation)
    # This is a simple way to introduce non-orthogonality.
    q2 = q * math.cos(ph) + i * math.sin(ph)

    return i + 1j * q2