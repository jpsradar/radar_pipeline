"""
core/budgets/radar_equation.py

Radar equation primitives for link budgets.

What this module does
---------------------
Provides explicit, reusable scalar formulas for:
- Wavelength computation
- Monostatic received power (basic radar equation)

Scope (v1)
----------
- Monostatic radar equation (power form)
- No atmospheric absorption models here (belongs in environment module)
- No ambiguity functions or waveform processing here (belongs in DSP modules)

Conventions
-----------
- Gt/Gr are POWER gains (linear) derived from dB via 10^(dB/10).
- Loss terms L are POWER losses (linear multipliers) where L >= 1.
- sigma is RCS in m^2.
"""

from __future__ import annotations

import math

from core.config.units import db_to_lin_power


class RadarEquationError(ValueError):
    """Raised when radar equation inputs are invalid."""


_C_MPS = 299_792_458.0


def wavelength_m(fc_hz: float) -> float:
    """
    Compute wavelength in meters from carrier frequency.

    Parameters
    ----------
    fc_hz : float
        Carrier frequency in Hz (must be > 0).

    Returns
    -------
    float
        Wavelength in meters.
    """
    if not isinstance(fc_hz, (int, float)) or isinstance(fc_hz, bool):
        raise RadarEquationError(f"fc_hz must be numeric, got {type(fc_hz).__name__}")
    f = float(fc_hz)
    if not math.isfinite(f) or f <= 0.0:
        raise RadarEquationError(f"fc_hz must be finite and > 0, got {fc_hz}")
    return float(_C_MPS / f)


def received_power_monostatic_w(
    *,
    pt_w: float,
    fc_hz: float,
    gt_db: float,
    gr_db: float,
    sigma_sqm: float,
    range_m: float,
    system_losses_db: float = 0.0,
) -> float:
    """
    Monostatic radar equation received power (scalar).

    Formula
    -------
    Pr = Pt * Gt * Gr * (lambda^2 * sigma) / ( (4*pi)^3 * R^4 * L )

    Parameters
    ----------
    pt_w : float
        Transmit power in Watts (> 0).
    fc_hz : float
        Carrier frequency in Hz (> 0).
    gt_db : float
        Transmit antenna gain in dB (power gain).
    gr_db : float
        Receive antenna gain in dB (power gain).
    sigma_sqm : float
        Target RCS in m^2 (> 0).
    range_m : float
        Range in meters (> 0).
    system_losses_db : float
        System losses in dB (power loss, typically >= 0).

    Returns
    -------
    float
        Received power in Watts.
    """
    for name, v in [("pt_w", pt_w), ("sigma_sqm", sigma_sqm), ("range_m", range_m)]:
        if not isinstance(v, (int, float)) or isinstance(v, bool):
            raise RadarEquationError(f"{name} must be numeric, got {type(v).__name__}")
        fv = float(v)
        if not math.isfinite(fv) or fv <= 0.0:
            raise RadarEquationError(f"{name} must be finite and > 0, got {v}")

    lam = wavelength_m(fc_hz)
    gt = float(db_to_lin_power(float(gt_db)))
    gr = float(db_to_lin_power(float(gr_db)))
    L = float(db_to_lin_power(float(system_losses_db)))

    numerator = float(pt_w) * gt * gr * (lam ** 2) * float(sigma_sqm)
    denom = ((4.0 * math.pi) ** 3) * (float(range_m) ** 4) * L

    pr = numerator / denom
    # Guard against negative due to numerical weirdness (should not happen for valid inputs)
    return float(max(pr, 0.0))