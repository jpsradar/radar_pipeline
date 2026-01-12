"""
core/antennas/arrays.py

Simple phased-array utilities (v1) for beamforming gain patterns.

Purpose
-------
Provide lightweight, deterministic array-factor computations for canonical array geometries.
This supports scan-loss bookkeeping and future scheduler/beam models without committing to
a full EM toolchain.

Scope (v1)
----------
- Uniform Linear Array (ULA) array factor (power pattern).
- Optional composition with an element pattern (multiplicative in linear power).
- Vectorized evaluation (NumPy).

Non-goals (v1)
--------------
- No mutual coupling, calibration errors, taper synthesis, or adaptive beamforming.
- No planar arrays (UPA), no 2D steering (az/el), no polarization.

Conventions
-----------
- theta_rad is off-broadside angle for a ULA (0 = broadside).
- Element spacing is expressed as d_over_lambda (default 0.5).
- Output is linear POWER gain, peak normalized to 1 for the array factor.

Public API (stable)
-------------------
- ula_array_factor(theta_rad, n_elements, d_over_lambda=0.5, steer_theta_rad=0.0) -> gain_lin
- ula_beam_pattern(theta_rad, n_elements, ..., element_pattern_fn=None, element_kwargs=None) -> gain_lin

Dependencies
------------
- NumPy (required)
- Python stdlib: math

Usage
-----
    import numpy as np
    from core.antennas.arrays import ula_array_factor

    theta = np.linspace(-0.5, 0.5, 200)
    g = ula_array_factor(theta, n_elements=16, d_over_lambda=0.5, steer_theta_rad=0.1)

Outputs
-------
NumPy array (or float) of linear power gain, normalized so that max gain is ~1.

Determinism / quality
---------------------
- Pure functions
- No I/O
- Defensive input validation
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Union
import math

import numpy as np

NumberOrArray = Union[float, np.ndarray]
PatternFn = Callable[..., NumberOrArray]


def ula_array_factor(
    theta_rad: NumberOrArray,
    *,
    n_elements: int,
    d_over_lambda: float = 0.5,
    steer_theta_rad: float = 0.0,
) -> NumberOrArray:
    """
    Compute ULA array factor power pattern (normalized).

    Model
    -----
    AF(θ) = (1/N) * sum_{m=0..N-1} exp(j*m*ψ(θ))
    ψ(θ) = 2π * d/λ * (sin(θ) - sin(θ0))

    Return value is |AF|^2 (power), normalized by 1/N so peak ~ 1.

    Parameters
    ----------
    theta_rad : float or np.ndarray
        Off-broadside angle(s) in radians.
    n_elements : int
        Number of array elements (N). Must be >= 1.
    d_over_lambda : float
        Element spacing divided by wavelength (d/λ). Must be > 0.
    steer_theta_rad : float
        Steering angle θ0 in radians (broadside reference). Finite.

    Returns
    -------
    float or np.ndarray
        Linear power gain (normalized).
    """
    if not isinstance(n_elements, int) or n_elements < 1:
        raise ValueError(f"n_elements must be an integer >= 1, got {n_elements}")
    _require_finite_positive(d_over_lambda, "d_over_lambda")
    _require_finite(steer_theta_rad, "steer_theta_rad")

    th = np.asarray(theta_rad, dtype=float)
    if not np.all(np.isfinite(th)):
        raise ValueError("theta_rad must contain only finite values")

    psi = 2.0 * math.pi * float(d_over_lambda) * (np.sin(th) - math.sin(float(steer_theta_rad)))
    m = np.arange(n_elements, dtype=float)

    # Compute AF efficiently: sum exp(j*m*psi)
    # shape: (N, ...) then sum over axis 0
    af = np.sum(np.exp(1j * (m[:, None] * psi.reshape(1, -1))), axis=0) / float(n_elements)
    g = np.abs(af) ** 2

    g = g.reshape(th.shape)

    # Return scalar if input scalar-like
    if np.isscalar(theta_rad):
        return float(g.item())
    return g


def ula_beam_pattern(
    theta_rad: NumberOrArray,
    *,
    n_elements: int,
    d_over_lambda: float = 0.5,
    steer_theta_rad: float = 0.0,
    element_pattern_fn: Optional[PatternFn] = None,
    element_kwargs: Optional[Dict[str, Any]] = None,
) -> NumberOrArray:
    """
    Compose a ULA array factor with an optional element pattern.

    Pattern composition (power domain):
        G_total(θ) = G_element(θ) * G_array(θ)

    Parameters
    ----------
    theta_rad : float or np.ndarray
        Angle(s) in radians.
    n_elements : int
        ULA element count.
    d_over_lambda : float
        Spacing / wavelength.
    steer_theta_rad : float
        Steering direction.
    element_pattern_fn : callable or None
        Function returning linear power gain vs theta. If None, assumed isotropic (1).
    element_kwargs : dict or None
        Extra keyword args passed to element_pattern_fn.

    Returns
    -------
    float or np.ndarray
        Linear power gain.
    """
    g_array = ula_array_factor(
        theta_rad,
        n_elements=n_elements,
        d_over_lambda=d_over_lambda,
        steer_theta_rad=steer_theta_rad,
    )

    if element_pattern_fn is None:
        return g_array

    kwargs = element_kwargs or {}
    g_el = element_pattern_fn(theta_rad, **kwargs)  # type: ignore[misc]

    # Multiply in power domain, preserving scalar vs array
    if np.isscalar(g_array) and np.isscalar(g_el):
        return float(g_array) * float(g_el)

    return np.asarray(g_array, dtype=float) * np.asarray(g_el, dtype=float)


# -----------------------------------------------------------------------------
# Internal helpers
# -----------------------------------------------------------------------------

def _require_finite(x: Any, name: str) -> None:
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        raise TypeError(f"{name} must be numeric, got {type(x).__name__}")
    if not math.isfinite(float(x)):
        raise ValueError(f"{name} must be finite, got {x}")


def _require_finite_positive(x: Any, name: str) -> None:
    _require_finite(x, name)
    if float(x) <= 0.0:
        raise ValueError(f"{name} must be > 0, got {x}")