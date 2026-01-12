"""
core/detection/thresholds.py

Threshold utilities for radar detection (fixed thresholds + CFAR scaling).

Purpose
-------
This module centralizes **threshold math** so detection logic (detectors, Pd/Pfa models,
CFAR wrappers, and validation harnesses) does not duplicate statistical details.

It provides two categories of threshold utilities:

1) Fixed (non-adaptive) thresholds
   - Analytical thresholds derived from the distribution of the test statistic under H0.
   - Used by: core/detection/detectors.py and validation/monte_carlo tooling.

2) CFAR scaling factors (adaptive multipliers)
   - For CA-CFAR under exponential background, returns alpha such that:
         threshold = alpha * mean(reference_cells)
   - Used by: core/detection/cfar.py and any CFAR-based detector wrapper.

Key API guarantees
------------------
- Deterministic: given the same inputs, outputs are deterministic.
- Safe: validates input domains and raises ThresholdError on misuse.
- Dependency strategy:
  - If SciPy is available, use chi-square inverse survival for exact thresholds.
  - If SciPy is NOT available, fall back to an internal Gamma-tail inversion
    (robust bisection + regularized incomplete gamma continued fraction).
  - This keeps the pipeline runnable in minimal environments while remaining correct.

Important conventions (v1)
--------------------------
Noncoherent energy detector:
- Statistic (normalized):
      E = sum_{i=1..N} Xi
  where under H0 each Xi is exponential with mean 1 (noise power per pulse normalized).

- Distribution under H0:
      E ~ Gamma(shape=N, scale=1)

- Threshold definition:
      P(E > T) = Pfa

This is the threshold used by the Monte Carlo Pd detector wrapper:
    energy_threshold_noncoherent(pfa, n_pulses) -> float

Coherent-like baseline threshold:
- For some simplified models, we use a single-sample 2-DOF energy statistic.
- Under H0 this is equivalent to Gamma(shape=1, scale=1) in normalized units.

Outputs
-------
- energy_threshold_noncoherent(pfa, n_pulses) -> float (normalized energy threshold)
- energy_threshold_coherent(pfa) -> float (normalized baseline threshold)
- threshold_noncoherent_energy(...) -> ThresholdResult (adds metadata)
- threshold_coherent_energy(...) -> ThresholdResult (adds metadata)
- threshold_scale_ca_cfar(pfa, n_ref) -> float

CLI usage
---------
This is a library module. Minimal sanity check:

    python -c "from core.detection.thresholds import energy_threshold_noncoherent; \
              print(energy_threshold_noncoherent(pfa=1e-6, n_pulses=16))"

Notes
-----
- The normalization used here matches the Monte Carlo wrappers in this repo:
  noise-only per-pulse energy has mean 1. If you change normalization elsewhere,
  keep the contract consistent at the boundary.

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal
import math

try:
    # Optional dependency: if present we use exact chi-square thresholds.
    from scipy.stats import chi2  # type: ignore
except Exception:  # pragma: no cover
    chi2 = None  # type: ignore

from core.detection.cfar import ca_cfar_alpha


class ThresholdError(ValueError):
    """Raised when threshold inputs are invalid or a threshold cannot be computed."""


IntegrationMode = Literal["noncoherent", "coherent"]


@dataclass(frozen=True)
class ThresholdResult:
    """
    Standard return container for threshold computations.

    Attributes
    ----------
    threshold : float
        Absolute threshold in the domain of the test statistic.
        In this repo's normalization, this is an energy threshold where
        noise-only per-pulse energy has mean 1.
    dof : int
        Degrees of freedom used by the chi-square interpretation (when applicable).
        For Gamma(shape=N, scale=1), the equivalent chi-square DoF is 2N.
    notes : str
        Human-readable explanation of the model and normalization.
    """

    threshold: float
    dof: int
    notes: str


# ---------------------------------------------------------------------
# Fixed thresholds (energy detectors)
# ---------------------------------------------------------------------

def energy_threshold_noncoherent(pfa: float, n_pulses: int) -> float:
    """
    Noncoherent energy threshold in normalized units.

    This is the **wrapper-facing** API required by:
        validation.monte_carlo.mc_pd_detector

    Parameters
    ----------
    pfa : float
        Desired probability of false alarm in (0,1).
    n_pulses : int
        Number of noncoherently integrated pulses (N >= 1).

    Returns
    -------
    float
        Threshold T such that P(E > T) = pfa for E ~ Gamma(shape=N, scale=1).

    Notes
    -----
    - If SciPy is available, we compute via the equivalent chi-square:
          2E ~ Chi-square(df=2N)
      so:
          T = 0.5 * chi2.isf(pfa, df=2N)
    - Without SciPy, we invert the Gamma survival directly (robust bisection).
    """
    _validate_pfa(pfa)
    _validate_positive_int(n_pulses, name="n_pulses")
    n = int(n_pulses)

    if chi2 is not None:  # exact path
        thr_chi2 = float(chi2.isf(float(pfa), df=2 * n))
        # Chi-square(df=2N) corresponds to 2 * Gamma(N,1)
        return 0.5 * thr_chi2

    # Minimal-deps fallback
    return _gamma_threshold_isf(pfa=float(pfa), shape=float(n), scale=1.0)


def energy_threshold_coherent(pfa: float) -> float:
    """
    Coherent-like baseline threshold in normalized units.

    In this repo's simplified coherent baseline, the statistic corresponds to
    one exponential-with-mean-1 energy sample under H0:
        E ~ Gamma(shape=1, scale=1)

    Parameters
    ----------
    pfa : float
        Desired probability of false alarm in (0,1).

    Returns
    -------
    float
        Threshold T such that P(E > T) = pfa for E ~ Gamma(1,1).

    Notes
    -----
    - If SciPy is available:
          2E ~ Chi-square(df=2) => T = 0.5 * chi2.isf(pfa, df=2)
    - Without SciPy: invert Gamma survival for (shape=1, scale=1), which is just:
          P(E > T) = exp(-T) => T = -ln(pfa)
      We use the general fallback for consistency.
    """
    _validate_pfa(pfa)

    if chi2 is not None:
        thr_chi2 = float(chi2.isf(float(pfa), df=2))
        return 0.5 * thr_chi2

    # For shape=1, scale=1, this is exactly -ln(pfa), but keep the common path.
    return _gamma_threshold_isf(pfa=float(pfa), shape=1.0, scale=1.0)


def threshold_noncoherent_energy(*, pfa: float, n_pulses: int) -> ThresholdResult:
    """
    Fixed threshold for a noncoherent energy detector integrating N pulses.

    Statistic (normalized)
    ----------------------
    E = sum_{i=1..N} Xi,  Xi ~ Exp(mean=1) under H0

    Distribution under H0:
    ----------------------
    E ~ Gamma(shape=N, scale=1)

    Threshold:
    ----------
    Choose T such that P(E > T) = Pfa.

    Returns
    -------
    ThresholdResult
        threshold is returned in normalized energy units (mean noise per pulse = 1).
        dof is reported as 2N (chi-square equivalence) for traceability.
    """
    thr = energy_threshold_noncoherent(pfa=float(pfa), n_pulses=int(n_pulses))
    return ThresholdResult(
        threshold=float(thr),
        dof=int(2 * int(n_pulses)),
        notes="Noncoherent energy threshold under H0: E~Gamma(N,1), ISF(Pfa).",
    )


def threshold_coherent_energy(*, pfa: float) -> ThresholdResult:
    """
    Fixed threshold for a coherent-like baseline 2-DOF energy detector.

    Under the normalized convention:
    E ~ Gamma(1,1) under H0

    Returns
    -------
    ThresholdResult
        threshold is in normalized energy units, dof reported as 2.
    """
    thr = energy_threshold_coherent(pfa=float(pfa))
    return ThresholdResult(
        threshold=float(thr),
        dof=2,
        notes="Coherent-like baseline threshold under H0: E~Gamma(1,1), ISF(Pfa).",
    )


def threshold_energy(
    *,
    pfa: float,
    n_pulses: int = 1,
    mode: IntegrationMode = "noncoherent",
) -> ThresholdResult:
    """
    Convenience wrapper selecting the energy-threshold model.

    Parameters
    ----------
    pfa : float
        Desired probability of false alarm in (0,1).
    n_pulses : int
        Number of pulses (used for noncoherent).
    mode : {"noncoherent", "coherent"}
        Threshold model selector.

    Returns
    -------
    ThresholdResult
    """
    mode_n = str(mode).lower().strip()
    if mode_n == "noncoherent":
        return threshold_noncoherent_energy(pfa=pfa, n_pulses=n_pulses)
    if mode_n == "coherent":
        return threshold_coherent_energy(pfa=pfa)
    raise ThresholdError("mode must be 'noncoherent' or 'coherent'")


# ---------------------------------------------------------------------
# CFAR scalars (adaptive)
# ---------------------------------------------------------------------

def threshold_scale_ca_cfar(*, pfa: float, n_ref: int) -> float:
    """
    Return CA-CFAR scaling factor alpha for exponential background.

    Parameters
    ----------
    pfa : float
        Desired false alarm probability in (0,1).
    n_ref : int
        Number of reference cells used to estimate noise level (>=1).

    Returns
    -------
    float
        alpha such that threshold = alpha * mean(reference_cells)
    """
    _validate_pfa(pfa)
    _validate_positive_int(n_ref, name="n_ref")
    return float(ca_cfar_alpha(pfa=float(pfa), n_ref=int(n_ref)))


# ---------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------

def _validate_pfa(pfa: float) -> None:
    """Validate that Pfa is a finite probability in (0,1)."""
    if not isinstance(pfa, (int, float)) or not math.isfinite(float(pfa)):
        raise ThresholdError(f"pfa must be a finite float in (0,1), got {pfa}")
    p = float(pfa)
    if not (0.0 < p < 1.0):
        raise ThresholdError(f"pfa must be in (0,1), got {p}")


def _validate_positive_int(x: int, *, name: str) -> None:
    """Validate that x is an integer >= 1."""
    if not isinstance(x, int) or x < 1:
        raise ThresholdError(f"{name} must be an integer >= 1, got {x}")


# ---------------------------------------------------------------------
# Minimal-deps fallback: Gamma ISF via regularized upper incomplete gamma
# ---------------------------------------------------------------------

def _gamma_threshold_isf(*, pfa: float, shape: float, scale: float) -> float:
    """
    Invert the Gamma survival function for Gamma(shape, scale).

    Finds T such that:
        P(X > T) = pfa,   X ~ Gamma(shape, scale)

    Strategy
    --------
    - Compute in the unit-scale domain: Y = T/scale, Y ~ Gamma(shape, 1)
    - Invert Q(shape, y) = pfa via monotone bisection.

    This is intentionally "boring and robust" for V1.
    """
    if not (0.0 < pfa < 1.0) or not math.isfinite(pfa):
        raise ThresholdError(f"pfa must be finite in (0,1), got {pfa}")
    if not (shape > 0.0) or not math.isfinite(shape):
        raise ThresholdError(f"shape must be finite and > 0, got {shape}")
    if not (scale > 0.0) or not math.isfinite(scale):
        raise ThresholdError(f"scale must be finite and > 0, got {scale}")

    # Work in y = T/scale
    target = float(pfa)

    lo = 0.0
    hi = max(1.0, float(shape))  # a reasonable starting point

    # Expand hi until sf(hi) <= target
    for _ in range(200):
        if _gamma_sf_unit(shape, hi) <= target:
            break
        hi *= 2.0
        if hi > 1e12:
            raise ThresholdError("Failed to bracket Gamma ISF (hi exploded).")
    else:
        raise ThresholdError("Failed to bracket Gamma ISF (max iterations).")

    # Bisection
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        sf_mid = _gamma_sf_unit(shape, mid)
        if sf_mid > target:
            lo = mid
        else:
            hi = mid
        if hi > 0.0 and (hi - lo) / hi < 1e-12:
            break

    return float(hi * scale)


def _gamma_sf_unit(a: float, x: float) -> float:
    """
    Survival function for Gamma(a, 1): sf(x) = Q(a, x).
    """
    if x <= 0.0:
        return 1.0
    if not math.isfinite(x):
        return 0.0
    return _gammainc_upper_reg(a, x)


def _gammainc_upper_reg(a: float, x: float) -> float:
    """
    Regularized upper incomplete gamma Q(a, x) = Γ(a, x) / Γ(a).

    Implementation notes
    --------------------
    - Uses a continued fraction (Lentz method) for Q when x >= a.
    - For x < a, computes P via series and returns Q = 1 - P.
    """
    if a <= 0.0 or not math.isfinite(a):
        raise ThresholdError(f"Invalid a for gammainc: {a}")
    if x < 0.0 or not math.isfinite(x):
        raise ThresholdError(f"Invalid x for gammainc: {x}")

    if x < a + 1.0:
        # Series for P(a,x)
        p = _gammainc_lower_reg_series(a, x)
        q = 1.0 - p
        # Numerical guard
        if q < 0.0:
            q = 0.0
        if q > 1.0:
            q = 1.0
        return q

    # Continued fraction for Q(a,x)
    return _gammainc_upper_reg_cf(a, x)


def _gammainc_lower_reg_series(a: float, x: float) -> float:
    """
    Regularized lower incomplete gamma P(a,x) via series expansion.

    P(a,x) = exp(-x) * x^a / Gamma(a) * sum_{n=0..inf} x^n / (a*(a+1)*...*(a+n))
    """
    if x == 0.0:
        return 0.0

    gln = math.lgamma(a)
    ap = a
    summ = 1.0 / a
    delt = summ

    for _ in range(1, 10_000):
        ap += 1.0
        delt *= x / ap
        summ += delt
        if abs(delt) < abs(summ) * 1e-14:
            break

    pref = math.exp(-x + a * math.log(x) - gln)
    p = pref * summ

    # Clamp to [0,1]
    if p < 0.0:
        return 0.0
    if p > 1.0:
        return 1.0
    return p


def _gammainc_upper_reg_cf(a: float, x: float) -> float:
    """
    Regularized upper incomplete gamma Q(a,x) via continued fraction (Lentz method).

    This is stable for x >= a+1.
    """
    gln = math.lgamma(a)
    # Tiny value to prevent division by zero
    tiny = 1e-300

    b0 = x + 1.0 - a
    c = 1.0 / tiny
    d = 1.0 / max(b0, tiny)
    h = d

    for i in range(1, 10_000):
        an = -float(i) * (float(i) - a)
        b = b0 + 2.0 * float(i)

        d = an * d + b
        if abs(d) < tiny:
            d = tiny
        c = b + an / c
        if abs(c) < tiny:
            c = tiny

        d = 1.0 / d
        delta = d * c
        h *= delta

        if abs(delta - 1.0) < 1e-14:
            break

    pref = math.exp(-x + a * math.log(x) - gln)
    q = pref * h

    # Clamp to [0,1]
    if q < 0.0:
        return 0.0
    if q > 1.0:
        return 1.0
    return q