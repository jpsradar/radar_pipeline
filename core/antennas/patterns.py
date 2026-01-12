"""
core/antennas/patterns.py

Antenna pattern models (v1) for the radar pipeline.

Purpose
-------
Provide small, deterministic antenna pattern utilities used by higher-level components
(link budgets, scan loss bookkeeping, and future scan scheduling / signal-level checks).

This module is intentionally conservative:
- Pure functions
- No file I/O
- No hidden global state
- Explicit assumptions and conventions

Scope (v1)
----------
- Canonical analytical patterns (dimensionless gain vs off-boresight angle):
    * isotropic (unity)
    * cosine-power mainlobe
    * sinc-squared mainlobe approximation (band-limited style)
- Normalization helpers:
    * peak-normalize to unity (linear power gain)

Non-goals (v1)
--------------
- No measured pattern ingestion (no CSV/NPY loading).
- No polarization, mutual coupling, or full EM modeling.
- No full 2D/3D az/el pattern surfaces (kept 1D angle for v1).

Conventions
-----------
- Angle input: theta_rad (radians), off-boresight. theta=0 => boresight.
- Output: linear POWER gain (not dB). Gain is dimensionless.
- For loss bookkeeping, scan_loss.py defines loss = 1/gain(theta).

Public API (stable)
-------------------
- pattern_isotropic(theta_rad) -> gain_lin
- pattern_cosine(theta_rad, n) -> gain_lin
- pattern_sinc_sq(theta_rad, beamwidth_3db_rad) -> gain_lin
- normalize_peak_to_unity(gain_lin) -> gain_lin

Dependencies
------------
- Python stdlib: math
- NumPy (optional): if available, vectorized evaluation is supported.
  If NumPy is not available, scalar inputs still work.

How to use
----------
Typical usage is via scan loss:
    from core.antennas.patterns import pattern_cosine
    from core.antennas.scan_loss import scan_loss_db
    loss_db = scan_loss_db(theta_rad=0.2, pattern_fn=pattern_cosine, n=12)

Outputs
-------
All functions return floats or numpy arrays (matching input type) in linear units.

Quality properties
------------------
- Deterministic
- Unit-testable (pure functions)
- Defensive input validation
"""

from __future__ import annotations

from typing import Any, Union
import math

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None  # type: ignore


NumberOrArray = Union[float, "np.ndarray"]  # type: ignore[name-defined]


def pattern_isotropic(theta_rad: NumberOrArray) -> NumberOrArray:
    """
    Isotropic pattern: unity gain for all angles.

    Parameters
    ----------
    theta_rad : float or np.ndarray
        Off-boresight angle(s) in radians.

    Returns
    -------
    float or np.ndarray
        Linear power gain (always 1).
    """
    if _is_ndarray(theta_rad):
        return np.ones_like(theta_rad, dtype=float)  # type: ignore[union-attr]
    _require_finite_scalar(theta_rad, name="theta_rad")
    return 1.0


def pattern_cosine(theta_rad: NumberOrArray, *, n: float) -> NumberOrArray:
    """
    Cosine-power mainlobe pattern: gain(theta) = cos(theta)^n for |theta| <= pi/2, else 0.

    Notes
    -----
    This is a simple, auditable approximation:
    - Peak is 1 at boresight.
    - n controls beam sharpness; larger n -> narrower mainlobe.

    Parameters
    ----------
    theta_rad : float or np.ndarray
        Off-boresight angle(s) in radians.
    n : float
        Exponent controlling mainlobe shape. Must be > 0.

    Returns
    -------
    float or np.ndarray
        Linear power gain in [0, 1].
    """
    _require_finite_positive_scalar(n, name="n")

    if _is_ndarray(theta_rad):
        t = np.asarray(theta_rad, dtype=float)  # type: ignore[union-attr]
        _require_finite_array(t, name="theta_rad")
        # mainlobe only: clamp outside +/- pi/2 to zero
        c = np.cos(t)
        out = np.where(np.abs(t) <= (math.pi / 2.0), np.maximum(c, 0.0) ** float(n), 0.0)
        return out

    _require_finite_scalar(theta_rad, name="theta_rad")
    t = float(theta_rad)
    if abs(t) > (math.pi / 2.0):
        return 0.0
    c = math.cos(t)
    if c <= 0.0:
        return 0.0
    return float(c ** float(n))


def pattern_sinc_sq(theta_rad: NumberOrArray, *, beamwidth_3db_rad: float) -> NumberOrArray:
    """
    Sinc-squared mainlobe approximation.

    Model
    -----
    gain(theta) = sinc(k * theta)^2

    where k is chosen so that gain(theta_3db) = 0.5 at theta_3db = beamwidth_3db_rad / 2.

    This is a pragmatic "mainlobe-only-ish" pattern useful for scan loss bookkeeping.

    Parameters
    ----------
    theta_rad : float or np.ndarray
        Off-boresight angle(s) in radians.
    beamwidth_3db_rad : float
        Full 3 dB beamwidth in radians. Must be > 0.

    Returns
    -------
    float or np.ndarray
        Linear power gain in (0, 1] with peak 1 at boresight.
    """
    _require_finite_positive_scalar(beamwidth_3db_rad, name="beamwidth_3db_rad")
    theta_3db = float(beamwidth_3db_rad) / 2.0
    if theta_3db <= 0.0:
        raise ValueError("beamwidth_3db_rad must be > 0")

    # Solve for k: sinc(k*theta_3db)^2 = 0.5 => sinc(k*theta_3db) = 1/sqrt(2)
    # We solve numerically with a small bracket. This is deterministic and done once per call.
    k = _solve_k_for_sinc_sq_half_power(theta_3db)

    if _is_ndarray(theta_rad):
        t = np.asarray(theta_rad, dtype=float)  # type: ignore[union-attr]
        _require_finite_array(t, name="theta_rad")
        x = k * t
        return _sinc_np(x) ** 2  # type: ignore[operator]

    _require_finite_scalar(theta_rad, name="theta_rad")
    x = k * float(theta_rad)
    return float(_sinc_scalar(x) ** 2)


def normalize_peak_to_unity(gain_lin: NumberOrArray) -> NumberOrArray:
    """
    Normalize a linear gain array so that its maximum is 1.

    Parameters
    ----------
    gain_lin : float or np.ndarray
        Linear gains (must be finite and >= 0).

    Returns
    -------
    float or np.ndarray
        Normalized gains. If max == 0, returns input unchanged.
    """
    if _is_ndarray(gain_lin):
        g = np.asarray(gain_lin, dtype=float)  # type: ignore[union-attr]
        _require_finite_array(g, name="gain_lin")
        if np.any(g < 0.0):  # type: ignore[union-attr]
            raise ValueError("gain_lin must be >= 0 everywhere")
        m = float(np.max(g))  # type: ignore[union-attr]
        if m <= 0.0:
            return g
        return g / m  # type: ignore[operator]

    _require_finite_scalar(gain_lin, name="gain_lin")
    g = float(gain_lin)
    if g < 0.0:
        raise ValueError("gain_lin must be >= 0")
    # scalar peak-normalization is identity
    return g


# -----------------------------------------------------------------------------
# Internal helpers
# -----------------------------------------------------------------------------

def _is_ndarray(x: Any) -> bool:
    if np is None:
        return False
    return isinstance(x, np.ndarray)


def _require_finite_scalar(x: Any, *, name: str) -> None:
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        raise TypeError(f"{name} must be numeric, got {type(x).__name__}")
    if not math.isfinite(float(x)):
        raise ValueError(f"{name} must be finite, got {x}")


def _require_finite_positive_scalar(x: Any, *, name: str) -> None:
    _require_finite_scalar(x, name=name)
    if float(x) <= 0.0:
        raise ValueError(f"{name} must be > 0, got {x}")


def _require_finite_array(x: Any, *, name: str) -> None:
    if np is None:  # pragma: no cover
        raise ImportError("NumPy is required for array evaluation")
    if not isinstance(x, np.ndarray):
        raise TypeError(f"{name} must be a numpy.ndarray, got {type(x).__name__}")
    if not np.all(np.isfinite(x)):
        raise ValueError(f"{name} must contain only finite values")


def _sinc_scalar(x: float) -> float:
    # normalized sinc: sinc(x) = sin(x)/x with sinc(0)=1
    if x == 0.0:
        return 1.0
    return math.sin(x) / x


def _sinc_np(x: "np.ndarray") -> "np.ndarray":  # type: ignore[name-defined]
    if np is None:  # pragma: no cover
        raise ImportError("NumPy is required for array sinc")
    out = np.ones_like(x, dtype=float)
    nz = (x != 0.0)
    out[nz] = np.sin(x[nz]) / x[nz]
    return out


def _solve_k_for_sinc_sq_half_power(theta_3db: float) -> float:
    """
    Deterministically solve for k where sinc(k*theta_3db)^2 = 0.5.

    We use a small bisection on y = k*theta_3db in (0, pi):
    - sinc(0)=1
    - sinc(pi)=0
    so there is a unique solution for sinc(y)=1/sqrt(2) in that interval.

    Returns
    -------
    k : float
    """
    target = 1.0 / math.sqrt(2.0)

    # solve for y in (0, pi) then return k = y/theta_3db
    lo = 0.0
    hi = math.pi
    for _ in range(80):  # tight enough
        mid = 0.5 * (lo + hi)
        val = _sinc_scalar(mid)
        if val > target:
            lo = mid
        else:
            hi = mid
    y = 0.5 * (lo + hi)
    return y / theta_3db