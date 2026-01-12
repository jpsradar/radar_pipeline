"""
core/environment/weather.py

Weather parameterization and simple RF attenuation helpers.

Purpose
-------
Provide a small, explicit representation of "weather state" and a conservative,
parameter-driven way to translate that state into a *specific attenuation* in dB/km,
which can be consumed by core/environment/propagation.py.

This module is intentionally split from propagation.py:
- propagation.py handles geometry + Friis (physically exact FSPL)
- weather.py handles *environmental* attenuation estimates and user knobs

Scope (v1)
----------
Included:
- WeatherProfile dataclass (temperature, humidity, rain rate, fog liquid water, etc.)
- A conservative, explicitly-labeled "simple" attenuation model:
    * If the user provides a specific_atten_db_per_km override, use it.
    * Otherwise estimate from rain rate and/or fog liquid water density using
      a lightweight heuristic model intended for pipeline-level sensitivity studies.

Not included (by design in v1):
- Full ITU-R recommendations (requires extensive tables and careful validation).
- Frequency-dependent gas absorption (oxygen/water vapor lines).
- Spatially varying weather fields.

Inputs / Outputs
----------------
Inputs: carrier frequency (Hz) and a WeatherProfile.
Outputs: specific attenuation in dB/km (float, >= 0).

Public API
----------
- WeatherProfile
- specific_attenuation_db_per_km(fc_hz, profile) -> float
- describe_weather(profile) -> dict (for manifests/metrics)

Dependencies
------------
- Python standard library only (dataclasses, math, typing)

Execution
---------
Not intended to be executed as a script.
Import from engines or scenario tooling.

Design notes
------------
- This is an explicit *v1* approximation layer.
- For serious engineering claims, replace the heuristic with a vetted model and
  keep the same function signature to avoid rippling changes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional
import math


@dataclass(frozen=True)
class WeatherProfile:
    """
    Compact weather state for RF attenuation estimates.

    Fields
    ------
    temperature_k : float
        Ambient temperature [K].
    relative_humidity : float
        Relative humidity [0..1]. Used only for reporting in v1.
    rain_rate_mm_hr : float
        Rain rate [mm/hr]. If > 0, contributes to attenuation.
    fog_liquid_water_g_m3 : float
        Fog/cloud liquid water density [g/m^3]. If > 0, contributes to attenuation.
    specific_atten_db_per_km : float | None
        Optional explicit override [dB/km]. If set, it is used directly.
    """
    temperature_k: float = 290.0
    relative_humidity: float = 0.5
    rain_rate_mm_hr: float = 0.0
    fog_liquid_water_g_m3: float = 0.0
    specific_atten_db_per_km: Optional[float] = None


def specific_attenuation_db_per_km(fc_hz: float, profile: WeatherProfile) -> float:
    """
    Estimate specific attenuation [dB/km] due to weather.

    Priority
    --------
    1) If profile.specific_atten_db_per_km is provided, return it (validated).
    2) Else compute a simple heuristic estimate from rain + fog components.

    Parameters
    ----------
    fc_hz : float
        Carrier frequency [Hz], must be > 0.
    profile : WeatherProfile
        Weather state.

    Returns
    -------
    float
        Specific attenuation [dB/km], >= 0.

    Heuristic model (v1)
    --------------------
    - Rain attenuation: modeled as k * R^alpha, where R is rain rate [mm/hr].
      k and alpha are crude band-dependent heuristics chosen for stable behavior
      in sensitivity studies (NOT a substitute for ITU-R).
    - Fog attenuation: modeled as beta * M, where M is liquid water density [g/m^3],
      and beta is a frequency-scaled coefficient.

    Notes
    -----
    This function is intentionally explicit and conservative:
    - It will not produce negative attenuation.
    - It will not silently accept invalid inputs (raises ValueError).
    """
    _require_positive(fc_hz, name="fc_hz")

    if profile.specific_atten_db_per_km is not None:
        _require_nonnegative(profile.specific_atten_db_per_km, name="specific_atten_db_per_km")
        return float(profile.specific_atten_db_per_km)

    f_ghz = float(fc_hz) / 1e9

    # --- Rain component: k * R^alpha (heuristic, smooth across bands) ---
    R = float(profile.rain_rate_mm_hr)
    _require_nonnegative(R, name="rain_rate_mm_hr")

    if R <= 0.0:
        gamma_rain = 0.0
    else:
        # Coefficients loosely shaped by typical behavior: higher freq => higher k, alpha ~ 0.8..1.0
        # This is a *pipeline heuristic*, not an ITU-R implementation.
        if f_ghz < 3.0:
            k, alpha = 0.0002, 0.9
        elif f_ghz < 10.0:
            k, alpha = 0.0020, 0.9
        elif f_ghz < 20.0:
            k, alpha = 0.0100, 0.9
        elif f_ghz < 40.0:
            k, alpha = 0.0300, 0.9
        else:
            k, alpha = 0.0800, 0.9
        gamma_rain = k * (R ** alpha)

    # --- Fog component: beta(f) * M (very conservative scaling) ---
    M = float(profile.fog_liquid_water_g_m3)
    _require_nonnegative(M, name="fog_liquid_water_g_m3")

    if M <= 0.0:
        gamma_fog = 0.0
    else:
        # Fog tends to matter more at higher microwave/mmWave frequencies.
        # Use a low, smooth coefficient that increases with frequency.
        beta = 0.005 * (f_ghz / 10.0)  # heuristic slope; 10 GHz => 0.005 dB/km per g/m^3
        gamma_fog = beta * M

    gamma = float(gamma_rain + gamma_fog)

    if not math.isfinite(gamma):
        raise ValueError("Computed specific attenuation is not finite (check inputs).")

    return max(gamma, 0.0)


def describe_weather(profile: WeatherProfile) -> Dict[str, Any]:
    """
    Serialize WeatherProfile into a JSON-friendly dict for manifests/metrics.
    """
    return {
        "temperature_k": float(profile.temperature_k),
        "relative_humidity": float(profile.relative_humidity),
        "rain_rate_mm_hr": float(profile.rain_rate_mm_hr),
        "fog_liquid_water_g_m3": float(profile.fog_liquid_water_g_m3),
        "specific_atten_db_per_km": None if profile.specific_atten_db_per_km is None else float(profile.specific_atten_db_per_km),
    }


# -----------------------------------------------------------------------------
# Validation helpers
# -----------------------------------------------------------------------------

def _require_positive(x: float, *, name: str) -> None:
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        raise ValueError(f"{name} must be numeric, got {type(x).__name__}")
    xf = float(x)
    if not math.isfinite(xf) or xf <= 0.0:
        raise ValueError(f"{name} must be finite and > 0, got {x}")


def _require_nonnegative(x: float, *, name: str) -> None:
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        raise ValueError(f"{name} must be numeric, got {type(x).__name__}")
    xf = float(x)
    if not math.isfinite(xf) or xf < 0.0:
        raise ValueError(f"{name} must be finite and >= 0, got {x}")