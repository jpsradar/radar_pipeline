"""
core/targets/swerling.py

Swerling target fluctuation models (I-IV) for the radar pipeline (v1).

Purpose
-------
Implement canonical Swerling RCS fluctuation models as random multipliers applied
to a mean/nominal RCS level. These are widely used in radar detection theory and
Monte Carlo validation.

This module provides:
- Deterministic, seed-controlled sampling via numpy.random.Generator
- Clear separation between "mean RCS level" and "fluctuation model"
- Utilities to generate amplitude or power multipliers consistent with Swerling cases

Scope (v1)
----------
Included:
- Enum-like constants for Swerling cases: "swerling0", "swerling1", "swerling2", "swerling3", "swerling4"
- power_multiplier_samples(case, rng, n, *, looks=1) -> np.ndarray
- rcs_samples_sqm(mean_sigma_sqm, case, rng, n, *, looks=1) -> np.ndarray

Model definitions (practical v1 interpretation)
-----------------------------------------------
We implement the standard textbook mapping in terms of target power (RCS) fluctuations:

- Swerling 0:
    * Nonfluctuating target: multiplier = 1

- Swerling 1:
    * Fluctuating target, constant over a scan (slow fluctuation).
    * Power multiplier ~ Exponential(mean=1) per "scan".
      In v1, we treat each call as one scan realization unless you request multiple looks.

- Swerling 2:
    * Fluctuating target, changes pulse-to-pulse (fast fluctuation).
    * Power multiplier ~ Exponential(mean=1) per sample.

- Swerling 3:
    * Chi-square with 4 degrees of freedom, slow fluctuation.
    * Power multiplier ~ Gamma(k=2, theta=1/2), mean=1, per scan.

- Swerling 4:
    * Chi-square with 4 degrees of freedom, fast fluctuation.
    * Power multiplier ~ Gamma(k=2, theta=1/2), mean=1, per sample.

"looks" parameter
-----------------
The optional "looks" parameter models incoherent averaging of independent looks:
- We generate 'looks' independent power multipliers and average them.
- This reduces variance while keeping mean=1.

Inputs / Outputs
----------------
- mean_sigma_sqm: mean/nominal RCS [m^2], must be > 0
- Outputs are arrays of RCS samples [m^2] (positive)

Public API
----------
- VALID_CASES
- power_multiplier_samples(case, rng, n, looks=1)
- rcs_samples_sqm(mean_sigma_sqm, case, rng, n, looks=1)

Dependencies
------------
- NumPy
- Python standard library (math, typing)

Execution
---------
Not intended to be executed as a script.

Design notes
------------
- All randomness is explicit (no module-level RNG).
- The implementation is stable and testable with deterministic seeds.
"""

from __future__ import annotations

from typing import Set
import math

import numpy as np


VALID_CASES: Set[str] = {
    "swerling0",
    "swerling1",
    "swerling2",
    "swerling3",
    "swerling4",
}


def power_multiplier_samples(case: str, rng: np.random.Generator, n: int, *, looks: int = 1) -> np.ndarray:
    """
    Draw power (RCS) multipliers for the requested Swerling case.

    Parameters
    ----------
    case : str
        One of VALID_CASES.
    rng : np.random.Generator
        RNG for determinism.
    n : int
        Number of samples (>= 1).
    looks : int
        Number of independent looks averaged incoherently (>= 1).

    Returns
    -------
    np.ndarray
        Array shape (n,) with mean approximately 1 and strictly positive entries.
    """
    if case not in VALID_CASES:
        raise ValueError(f"Unknown Swerling case '{case}'. Valid: {sorted(VALID_CASES)}")
    if not isinstance(rng, np.random.Generator):
        raise ValueError("rng must be a numpy.random.Generator")
    nn = int(n)
    if nn <= 0:
        raise ValueError("n must be >= 1")
    ll = int(looks)
    if ll <= 0:
        raise ValueError("looks must be >= 1")

    # Helper: incoherent averaging of independent looks while preserving mean=1
    def avg_looks(draw_fn) -> np.ndarray:
        x = np.zeros((ll, nn), dtype=float)
        for i in range(ll):
            x[i, :] = draw_fn(nn)
        y = np.mean(x, axis=0)
        y = np.maximum(y, np.finfo(float).tiny)
        return y

    case_l = str(case).lower().strip()

    if case_l == "swerling0":
        return np.ones((nn,), dtype=float)

    if case_l in {"swerling1", "swerling2"}:
        # Exponential(mean=1) = Gamma(k=1, theta=1)
        if case_l == "swerling1":
            # Slow fluctuation: one draw per scan replicated across n
            x0 = float(rng.exponential(scale=1.0))
            x0 = max(x0, np.finfo(float).tiny)
            base = np.full((nn,), x0, dtype=float)
            if ll == 1:
                return base
            # For looks>1, treat each look as another independent "scan" draw
            return avg_looks(lambda m: np.full((m,), float(rng.exponential(scale=1.0)), dtype=float))

        # Swerling2: fast fluctuation per sample
        if ll == 1:
            x = rng.exponential(scale=1.0, size=nn)
            return np.maximum(np.asarray(x, dtype=float), np.finfo(float).tiny)
        return avg_looks(lambda m: rng.exponential(scale=1.0, size=m))

    if case_l in {"swerling3", "swerling4"}:
        # Gamma(k=2, theta=1/2) has mean k*theta = 1
        k = 2.0
        theta = 0.5

        if case_l == "swerling3":
            # Slow fluctuation: one draw per scan
            x0 = float(rng.gamma(shape=k, scale=theta))
            x0 = max(x0, np.finfo(float).tiny)
            base = np.full((nn,), x0, dtype=float)
            if ll == 1:
                return base
            return avg_looks(lambda m: np.full((m,), float(rng.gamma(shape=k, scale=theta)), dtype=float))

        # Swerling4: fast fluctuation per sample
        if ll == 1:
            x = rng.gamma(shape=k, scale=theta, size=nn)
            return np.maximum(np.asarray(x, dtype=float), np.finfo(float).tiny)
        return avg_looks(lambda m: rng.gamma(shape=k, scale=theta, size=m))

    # Should be unreachable due to VALID_CASES
    raise RuntimeError(f"Unhandled Swerling case: {case_l}")


def rcs_samples_sqm(
    mean_sigma_sqm: float,
    case: str,
    rng: np.random.Generator,
    n: int,
    *,
    looks: int = 1,
) -> np.ndarray:
    """
    Draw RCS samples [m^2] given a mean/nominal sigma and a Swerling case.

    Parameters
    ----------
    mean_sigma_sqm : float
        Mean/nominal RCS [m^2], must be > 0.
    case : str
        Swerling case in VALID_CASES.
    rng : np.random.Generator
        RNG for determinism.
    n : int
        Number of samples.
    looks : int
        Number of incoherent looks averaged.

    Returns
    -------
    np.ndarray
        RCS samples [m^2], shape (n,), positive.
    """
    _require_positive(mean_sigma_sqm, name="mean_sigma_sqm")
    mult = power_multiplier_samples(case, rng, n, looks=looks)
    sigma = float(mean_sigma_sqm) * np.asarray(mult, dtype=float)
    sigma = np.maximum(sigma, np.finfo(float).tiny)
    return sigma


# -----------------------------------------------------------------------------
# Validation helpers
# -----------------------------------------------------------------------------

def _require_positive(x: float, *, name: str) -> None:
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        raise ValueError(f"{name} must be numeric, got {type(x).__name__}")
    xf = float(x)
    if not math.isfinite(xf) or xf <= 0.0:
        raise ValueError(f"{name} must be finite and > 0, got {x}")