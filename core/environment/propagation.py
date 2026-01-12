"""
core/environment/propagation.py

Propagation and path-loss utilities for the radar pipeline.

Purpose
-------
Centralize propagation-related math (free-space path loss, two-way loss bookkeeping,
and simple atmospheric attenuation hooks) so that radar equation / interference
modules do not duplicate or diverge in basic RF propagation logic.

This module is deliberately conservative:
- It provides physically correct free-space loss (Friis) in dB.
- It provides explicit two-way (monostatic) loss helpers.
- It provides a simple atmospheric attenuation interface that is *parameter-driven*
  (i.e., you pass the specific attenuation in dB/km rather than embedding a complex
  model here). Weather-driven models live in core/environment/weather.py.

Scope (v1)
----------
Included:
- Speed of light constant and wavelength conversion.
- One-way and two-way free-space path loss (FSPL) in dB.
- Range-based attenuation with user-supplied specific attenuation [dB/km].
- Total loss composition helpers (system losses + FSPL + atmospheric).

Not included (by design in v1):
- Terrain masking / diffraction.
- Refractivity and ducting.
- Multipath / ground bounce.
- Frequency-selective fading.

Inputs / Outputs
----------------
Inputs are numeric scalars (float-like) in SI units unless stated otherwise.
Outputs are floats in either linear units (W) or logarithmic units (dB) as named.

Public API
----------
- SPEED_OF_LIGHT_M_S
- wavelength_m(fc_hz) -> float
- fspl_db(range_m, fc_hz) -> float
- fspl_two_way_db(range_m, fc_hz) -> float
- atmospheric_loss_db(range_m, specific_atten_db_per_km) -> float
- total_two_way_loss_db(range_m, fc_hz, *, system_losses_db=0.0, specific_atten_db_per_km=0.0) -> float
- db_to_lin(db) -> float
- lin_to_db(lin) -> float

Dependencies
------------
- Python standard library only (math)

Execution
---------
This module is not intended to be executed as a script.
Import and use from simulation / interference modules.

Design notes
------------
- Validation is strict: invalid inputs raise ValueError.
- The functions are deterministic and side-effect free.
"""

from __future__ import annotations

import math


SPEED_OF_LIGHT_M_S: float = 299_792_458.0


# -----------------------------------------------------------------------------
# Small dB helpers (kept here to avoid circular imports)
# -----------------------------------------------------------------------------

def db_to_lin(db: float) -> float:
    """Convert dB (power ratio) to linear power ratio."""
    _require_finite_scalar(db, name="db")
    return 10.0 ** (float(db) / 10.0)


def lin_to_db(lin: float) -> float:
    """Convert linear power ratio to dB. Requires lin > 0."""
    _require_finite_scalar(lin, name="lin")
    if float(lin) <= 0.0:
        raise ValueError(f"lin must be > 0, got {lin}")
    return 10.0 * math.log10(float(lin))


# -----------------------------------------------------------------------------
# Core propagation
# -----------------------------------------------------------------------------

def wavelength_m(fc_hz: float) -> float:
    """Return wavelength [m] for carrier frequency fc_hz [Hz]."""
    _require_positive_scalar(fc_hz, name="fc_hz")
    return SPEED_OF_LIGHT_M_S / float(fc_hz)


def fspl_db(range_m: float, fc_hz: float) -> float:
    """
    One-way free-space path loss (FSPL) in dB.

    Formula
    -------
    FSPL = (4*pi*R / lambda)^2  (linear power ratio)
    FSPL_dB = 20*log10(4*pi*R / lambda)

    Parameters
    ----------
    range_m : float
        Propagation range [m]. Must be > 0.
    fc_hz : float
        Carrier frequency [Hz]. Must be > 0.

    Returns
    -------
    float
        One-way FSPL in dB (>= 0).
    """
    _require_positive_scalar(range_m, name="range_m")
    lam = wavelength_m(fc_hz)
    x = (4.0 * math.pi * float(range_m)) / lam
    # x must be > 0 given validations.
    return 20.0 * math.log10(x)


def fspl_two_way_db(range_m: float, fc_hz: float) -> float:
    """
    Two-way (monostatic) free-space path loss in dB.

    Two-way FSPL is simply 2 * one-way FSPL (in dB), representing out-and-back propagation.
    """
    return 2.0 * fspl_db(range_m, fc_hz)


def atmospheric_loss_db(range_m: float, specific_atten_db_per_km: float) -> float:
    """
    One-way atmospheric attenuation in dB, parameterized by specific attenuation.

    Parameters
    ----------
    range_m : float
        Path length [m]. Must be > 0.
    specific_atten_db_per_km : float
        Specific attenuation [dB/km]. Must be >= 0.

    Returns
    -------
    float
        One-way atmospheric loss in dB (>= 0).

    Notes
    -----
    This function is intentionally parameter-driven; compute specific attenuation
    using core/environment/weather.py if desired.
    """
    _require_positive_scalar(range_m, name="range_m")
    _require_nonnegative_scalar(specific_atten_db_per_km, name="specific_atten_db_per_km")
    km = float(range_m) / 1000.0
    return float(specific_atten_db_per_km) * km


def total_two_way_loss_db(
    range_m: float,
    fc_hz: float,
    *,
    system_losses_db: float = 0.0,
    specific_atten_db_per_km: float = 0.0,
) -> float:
    """
    Compose total two-way loss (monostatic) in dB.

    Components (all in dB)
    ----------------------
    - Two-way FSPL:        2 * fspl_db(range_m, fc_hz)
    - Two-way atmosphere:  2 * atmospheric_loss_db(range_m, specific_atten_db_per_km)
    - System losses:       system_losses_db (user-provided lumped losses, >= 0)

    Returns
    -------
    float
        Total two-way loss in dB (>= 0).
    """
    _require_positive_scalar(range_m, name="range_m")
    _require_positive_scalar(fc_hz, name="fc_hz")
    _require_nonnegative_scalar(system_losses_db, name="system_losses_db")
    _require_nonnegative_scalar(specific_atten_db_per_km, name="specific_atten_db_per_km")

    loss_fspl = fspl_two_way_db(range_m, fc_hz)
    loss_atm = 2.0 * atmospheric_loss_db(range_m, specific_atten_db_per_km)
    return float(loss_fspl + loss_atm + float(system_losses_db))


# -----------------------------------------------------------------------------
# Validation helpers
# -----------------------------------------------------------------------------

def _require_finite_scalar(x: float, *, name: str) -> None:
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        raise ValueError(f"{name} must be numeric, got {type(x).__name__}")
    if not math.isfinite(float(x)):
        raise ValueError(f"{name} must be finite, got {x}")


def _require_positive_scalar(x: float, *, name: str) -> None:
    _require_finite_scalar(x, name=name)
    if float(x) <= 0.0:
        raise ValueError(f"{name} must be > 0, got {x}")


def _require_nonnegative_scalar(x: float, *, name: str) -> None:
    _require_finite_scalar(x, name=name)
    if float(x) < 0.0:
        raise ValueError(f"{name} must be >= 0, got {x}")