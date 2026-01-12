"""
core/budgets/snr_sinr.py

SNR/SINR budgeting helpers (noise and interference power).

What this module does
---------------------
- Compute thermal noise power: N = k*T*B*F
- Compute SNR: SNR = S / N
- Compute SINR: SINR = S / (N + I)

Scope (v1)
----------
This module is intentionally minimal and scalar:
- No integration losses, no detector specifics (belongs in detection module)
- No waveform processing or range/Doppler processing (belongs in DSP module)
"""

from __future__ import annotations

import math

from core.config.units import db_to_lin_power, k_boltzmann, lin_to_db_power


class BudgetError(ValueError):
    """Raised when budget inputs are invalid."""


def noise_power_w(*, temperature_k: float, bw_hz: float, nf_db: float) -> float:
    """
    Compute receiver input-referred noise power in Watts.

    N = k * T * B * F

    Parameters
    ----------
    temperature_k : float
        System noise temperature in Kelvin (> 0).
    bw_hz : float
        Receiver bandwidth in Hz (> 0).
    nf_db : float
        Noise figure in dB (power factor, >= 0 is typical).

    Returns
    -------
    float
        Noise power in Watts.
    """
    for name, v, pos in [
        ("temperature_k", temperature_k, True),
        ("bw_hz", bw_hz, True),
        ("nf_db", nf_db, False),
    ]:
        if not isinstance(v, (int, float)) or isinstance(v, bool):
            raise BudgetError(f"{name} must be numeric, got {type(v).__name__}")
        fv = float(v)
        if not math.isfinite(fv):
            raise BudgetError(f"{name} must be finite, got {v}")
        if pos and fv <= 0.0:
            raise BudgetError(f"{name} must be > 0, got {v}")
        if (name == "nf_db") and fv < 0.0:
            raise BudgetError(f"nf_db must be >= 0, got {v}")

    F = float(db_to_lin_power(float(nf_db)))
    n = k_boltzmann() * float(temperature_k) * float(bw_hz) * F
    if not math.isfinite(n) or n <= 0.0:
        raise BudgetError(f"Computed noise power must be finite and > 0, got {n}")
    return float(n)


def snr_linear(*, signal_power_w: float, noise_power_w: float) -> float:
    """
    Compute linear SNR = S / N.

    Returns 0 if signal is 0. Raises on invalid noise.
    """
    if not isinstance(signal_power_w, (int, float)) or isinstance(signal_power_w, bool):
        raise BudgetError(f"signal_power_w must be numeric, got {type(signal_power_w).__name__}")
    if not isinstance(noise_power_w, (int, float)) or isinstance(noise_power_w, bool):
        raise BudgetError(f"noise_power_w must be numeric, got {type(noise_power_w).__name__}")

    s = float(signal_power_w)
    n = float(noise_power_w)

    if not math.isfinite(s) or s < 0.0:
        raise BudgetError(f"signal_power_w must be finite and >= 0, got {signal_power_w}")
    if not math.isfinite(n) or n <= 0.0:
        raise BudgetError(f"noise_power_w must be finite and > 0, got {noise_power_w}")

    return float(s / n) if s > 0.0 else 0.0


def sinr_linear(*, signal_power_w: float, noise_power_w: float, interference_power_w: float = 0.0) -> float:
    """
    Compute linear SINR = S / (N + I).

    Parameters
    ----------
    signal_power_w : float
        Signal power (>= 0).
    noise_power_w : float
        Noise power (> 0).
    interference_power_w : float
        Interference power (>= 0).

    Returns
    -------
    float
        SINR (linear).
    """
    if not isinstance(interference_power_w, (int, float)) or isinstance(interference_power_w, bool):
        raise BudgetError(f"interference_power_w must be numeric, got {type(interference_power_w).__name__}")

    i = float(interference_power_w)
    if not math.isfinite(i) or i < 0.0:
        raise BudgetError(f"interference_power_w must be finite and >= 0, got {interference_power_w}")

    denom = float(noise_power_w) + i
    if denom <= 0.0 or not math.isfinite(denom):
        raise BudgetError(f"N+I must be finite and > 0, got {denom}")

    s = float(signal_power_w)
    if s <= 0.0:
        return 0.0

    return float(s / denom)


def ratio_to_db(ratio_lin: float) -> float:
    """
    Convenience: convert a linear power ratio to dB safely.

    Returns a finite number; clamps extremely small ratios to tiny positive.
    """
    if not isinstance(ratio_lin, (int, float)) or isinstance(ratio_lin, bool):
        raise BudgetError(f"ratio_lin must be numeric, got {type(ratio_lin).__name__}")
    x = float(ratio_lin)
    if not math.isfinite(x) or x < 0.0:
        raise BudgetError(f"ratio_lin must be finite and >= 0, got {ratio_lin}")
    tiny = 1e-300
    return float(lin_to_db_power(max(x, tiny)))