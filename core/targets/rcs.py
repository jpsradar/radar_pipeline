"""
core/targets/rcs.py

Radar Cross Section (RCS) models for the radar pipeline (v1).

Purpose
-------
Provide deterministic and stochastic RCS model primitives that can be used by:
- model-based performance calculations (expected/mean RCS),
- Monte Carlo simulations (random RCS draws per CPI / dwell),
- signal-level spot injections (consistent RCS reference).

This module is scoped for v1 credibility:
- Clear unit conventions (square meters).
- Explicit model naming.
- Strict validation and deterministic RNG usage.

Scope (v1)
----------
Included:
- ConstantRCS: fixed sigma [m^2]
- LognormalRCS: sigma drawn lognormally around a median
- AspectCosineRCS: simple deterministic aspect dependence (cos^p) with numerical clamping
- Helper: rcs_to_dbsm / dbsm_to_rcs

Not included (by design in v1):
- Full electromagnetic CAD/RCS libraries
- Polarization effects
- Complex aspect-angle databases

Inputs / Outputs
----------------
- RCS sigma in square meters.
- Angles in radians.
- RNG passed explicitly as numpy.random.Generator for determinism.

Public API
----------
- rcs_to_dbsm(sigma_sqm) -> float
- dbsm_to_rcs(dbsm) -> float
- ConstantRCS.sample(...)
- LognormalRCS.sample(...)
- AspectCosineRCS.sigma(...)

Dependencies
------------
- NumPy
- Python standard library (dataclasses, math, typing)

Execution
---------
Not intended to be executed as a script.

Design notes
------------
- Stochastic models require an explicit RNG to avoid hidden global state.
- Numerical robustness: aspect models clamp near-zero cosine values to 0 to avoid
  floating-point artifacts (e.g., cos(pi/2) ≈ 6e-17).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import math

import numpy as np


def rcs_to_dbsm(sigma_sqm: float) -> float:
    """Convert RCS [m^2] to dBsm. Requires sigma_sqm > 0."""
    _require_positive(sigma_sqm, name="sigma_sqm")
    return 10.0 * math.log10(float(sigma_sqm))


def dbsm_to_rcs(dbsm: float) -> float:
    """Convert dBsm to RCS [m^2]."""
    _require_finite(dbsm, name="dbsm")
    return 10.0 ** (float(dbsm) / 10.0)


@dataclass(frozen=True)
class ConstantRCS:
    """
    Constant RCS model.

    Parameters
    ----------
    sigma_sqm : float
        Fixed RCS [m^2], must be > 0.
    """
    sigma_sqm: float

    def __post_init__(self) -> None:
        _require_positive(self.sigma_sqm, name="sigma_sqm")

    def sample(self, *, rng: Optional[np.random.Generator] = None, size: Optional[int] = None) -> np.ndarray:
        """
        Return constant sigma values.

        Parameters
        ----------
        rng : ignored (present for API consistency)
        size : int | None
            Number of samples.

        Returns
        -------
        np.ndarray
            Array of shape (size,) if size provided else shape (1,).
        """
        n = 1 if size is None else int(size)
        if n <= 0:
            raise ValueError("size must be positive")
        return np.full((n,), float(self.sigma_sqm), dtype=float)


@dataclass(frozen=True)
class LognormalRCS:
    """
    Lognormal RCS model.

    The distribution is parameterized by:
    - median_sigma_sqm: median of sigma in linear domain [m^2]
    - sigma_db: lognormal spread expressed as standard deviation in dB

    Notes
    -----
    If X_dB ~ N(0, sigma_db^2), then sigma = median * 10^(X_dB/10).

    Parameters
    ----------
    median_sigma_sqm : float
        Median RCS [m^2], must be > 0.
    sigma_db : float
        Standard deviation in dB, must be >= 0.
    """
    median_sigma_sqm: float
    sigma_db: float

    def __post_init__(self) -> None:
        _require_positive(self.median_sigma_sqm, name="median_sigma_sqm")
        _require_nonnegative(self.sigma_db, name="sigma_db")

    def sample(self, *, rng: np.random.Generator, size: int) -> np.ndarray:
        """
        Draw lognormal RCS samples.

        Parameters
        ----------
        rng : np.random.Generator
            RNG for determinism.
        size : int
            Number of samples, must be >= 1.

        Returns
        -------
        np.ndarray
            Samples [m^2], shape (size,).
        """
        if not isinstance(rng, np.random.Generator):
            raise ValueError("rng must be a numpy.random.Generator")
        n = int(size)
        if n <= 0:
            raise ValueError("size must be >= 1")

        x_db = rng.normal(loc=0.0, scale=float(self.sigma_db), size=n)
        sigma = float(self.median_sigma_sqm) * (10.0 ** (x_db / 10.0))
        sigma = np.asarray(sigma, dtype=float)
        sigma = np.maximum(sigma, np.finfo(float).tiny)
        return sigma


@dataclass(frozen=True)
class AspectCosineRCS:
    """
    Simple deterministic aspect-dependent RCS model using cosine law.

    sigma(theta) = sigma_boresight * max(cos(theta), 0)^p

    Numerical robustness
    --------------------
    Floating-point evaluation of cos(pi/2) is not exactly zero. To avoid tiny
    positive RCS artifacts at broadside, we clamp very small cos(theta) values
    to 0 with an explicit tolerance.

    Parameters
    ----------
    sigma_boresight_sqm : float
        RCS at boresight (theta = 0) [m^2], must be > 0.
    p : float
        Shape exponent (>= 0). Higher p narrows the lobe.
    zero_cos_tol : float
        Clamp threshold for cos(theta) values close to zero (>= 0).
    """
    sigma_boresight_sqm: float
    p: float = 2.0
    zero_cos_tol: float = 1e-12

    def __post_init__(self) -> None:
        _require_positive(self.sigma_boresight_sqm, name="sigma_boresight_sqm")
        _require_nonnegative(self.p, name="p")
        _require_nonnegative(self.zero_cos_tol, name="zero_cos_tol")

    def sigma(self, theta_rad: float) -> float:
        """
        Compute aspect-dependent sigma for a single angle.

        Parameters
        ----------
        theta_rad : float
            Aspect angle [rad]. Must be finite.

        Returns
        -------
        float
            RCS [m^2], >= 0.
        """
        _require_finite(theta_rad, name="theta_rad")

        c = math.cos(float(theta_rad))
        # Physical backscatter model: no negative contribution.
        if c <= 0.0:
            return 0.0

        # Numerical clamp near zero (e.g., cos(pi/2) ≈ 6e-17).
        if c < float(self.zero_cos_tol):
            return 0.0

        return float(self.sigma_boresight_sqm) * (c ** float(self.p))


# -----------------------------------------------------------------------------
# Validation helpers
# -----------------------------------------------------------------------------

def _require_finite(x: float, *, name: str) -> None:
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        raise ValueError(f"{name} must be numeric, got {type(x).__name__}")
    if not math.isfinite(float(x)):
        raise ValueError(f"{name} must be finite, got {x}")


def _require_positive(x: float, *, name: str) -> None:
    _require_finite(x, name=name)
    if float(x) <= 0.0:
        raise ValueError(f"{name} must be > 0, got {x}")


def _require_nonnegative(x: float, *, name: str) -> None:
    _require_finite(x, name=name)
    if float(x) < 0.0:
        raise ValueError(f"{name} must be >= 0, got {x}")