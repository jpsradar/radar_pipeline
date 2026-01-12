"""
core/environment/clutter_models.py

Clutter/noise power statistical models for radar performance simulation.

This module provides a professional, extensible library of clutter POWER models:
- Exponential (Rayleigh amplitude) power model (baseline for thermal noise)
- Weibull power model
- Lognormal power model
- K-distribution power model (spiky sea/ground clutter)

What this module does
---------------------
- Defines distribution classes with a consistent API:
    - sample(rng, size) -> np.ndarray of nonnegative powers
    - mean_power() / var_power()
    - cdf(x) / sf(x) / ppf(q) where closed-form is available; otherwise numerical
- Provides utilities for:
    - Heterogeneous clutter (range/azimuth dependent mean power scaling)
    - Converting between mean power and distribution parameters (where applicable)
- Enforces reproducibility by requiring an explicit numpy Generator

Inputs
------
- RNG: numpy.random.Generator (required for sampling)
- Model parameters per distribution (documented below)
- Optionally, a spatial scaling map (e.g., per-cell mean multiplier)

Outputs
-------
- Samples: arrays of clutter powers (float64, >=0)
- Distribution utilities: CDF/SF/PPF, moments (mean/var)

Intended usage
--------------
- For CFAR validation (homogeneous and heterogeneous backgrounds)
- For fast Monte Carlo FAR/Pfa studies
- For feeding signal-level engines (power maps or amplitude synthesis later)

CLI usage (example)
-------------------
    python -c "import numpy as np; \
               from core.environment.clutter_models import ExponentialPower; \
               rng=np.random.default_rng(0); m=ExponentialPower(mean_power=1.0); \
               x=m.sample(rng, size=5); print(x)"

Notes on conventions
--------------------
- We model POWER directly (e.g., |z|^2) since detectors and CFAR typically operate on power.
- If later you need complex I/Q synthesis, build it on top (amplitude + random phase).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol, Tuple, Union
import math

import numpy as np

try:
    from scipy import special  # type: ignore
    from scipy.stats import gamma, lognorm, weibull_min  # type: ignore
except Exception as exc:  # pragma: no cover
    raise ImportError(
        "SciPy is required for clutter models (gamma/lognorm/weibull/special). "
        "Install with: pip install scipy"
    ) from exc


# ----------------------------
# Exceptions
# ----------------------------

class ClutterModelError(ValueError):
    """Raised when clutter model parameters are invalid."""


# ----------------------------
# Protocol (interface contract)
# ----------------------------

class PowerDistribution(Protocol):
    """
    Interface contract for clutter/noise POWER distributions.

    Implementations must:
    - return nonnegative finite samples
    - provide mean/var (closed-form when possible)
    - provide at least cdf/sf OR document limitations
    """

    def sample(self, rng: np.random.Generator, size: Union[int, Tuple[int, ...]]) -> np.ndarray:
        ...

    def mean_power(self) -> float:
        ...

    def var_power(self) -> float:
        ...

    def cdf(self, x: np.ndarray) -> np.ndarray:
        ...

    def sf(self, x: np.ndarray) -> np.ndarray:
        ...

    def ppf(self, q: np.ndarray) -> np.ndarray:
        ...


# ----------------------------
# Helpers
# ----------------------------

def apply_mean_scaling(power_samples: np.ndarray, mean_multiplier: np.ndarray) -> np.ndarray:
    """
    Apply spatial mean scaling to power samples.

    This is a common way to create heterogeneous clutter while preserving the
    "shape" distribution: power_scaled = power * mean_multiplier.

    Parameters
    ----------
    power_samples : np.ndarray
        Nonnegative power samples.
    mean_multiplier : np.ndarray
        Nonnegative scaling factors broadcastable to power_samples.

    Returns
    -------
    np.ndarray
        Scaled power samples.
    """
    x = np.asarray(power_samples, dtype=float)
    m = np.asarray(mean_multiplier, dtype=float)

    if np.any(~np.isfinite(x)) or np.any(x < 0.0):
        raise ClutterModelError("power_samples must be finite and nonnegative")
    if np.any(~np.isfinite(m)) or np.any(m < 0.0):
        raise ClutterModelError("mean_multiplier must be finite and nonnegative")

    return x * m


def _require_pos_float(x: float, name: str) -> float:
    if not isinstance(x, (int, float)) or not math.isfinite(float(x)):
        raise ClutterModelError(f"{name} must be a finite number, got {x}")
    xf = float(x)
    if xf <= 0.0:
        raise ClutterModelError(f"{name} must be > 0, got {x}")
    return xf


def _require_nonneg_float(x: float, name: str) -> float:
    if not isinstance(x, (int, float)) or not math.isfinite(float(x)):
        raise ClutterModelError(f"{name} must be a finite number, got {x}")
    xf = float(x)
    if xf < 0.0:
        raise ClutterModelError(f"{name} must be >= 0, got {x}")
    return xf


def _as_1d(x: np.ndarray) -> np.ndarray:
    return np.asarray(x, dtype=float).reshape(-1)


# ----------------------------
# Exponential power (Rayleigh amplitude)
# ----------------------------

@dataclass(frozen=True)
class ExponentialPower:
    """
    Exponential POWER distribution (baseline for thermal noise / Rayleigh amplitude clutter).

    Power X ~ Exp(scale=mu) where mu = E[X] is the mean power.

    PDF: f(x) = (1/mu) * exp(-x/mu), x>=0
    CDF: F(x) = 1 - exp(-x/mu)
    """
    mean_power: float

    def __post_init__(self) -> None:
        _require_pos_float(self.mean_power, "mean_power")

    def sample(self, rng: np.random.Generator, size: Union[int, Tuple[int, ...]]) -> np.ndarray:
        x = rng.exponential(scale=self.mean_power, size=size)
        return np.asarray(x, dtype=float)

    def mean_power(self) -> float:  # type: ignore[override]
        return float(self.mean_power)

    def var_power(self) -> float:
        mu = float(self.mean_power)
        return mu * mu

    def cdf(self, x: np.ndarray) -> np.ndarray:
        mu = float(self.mean_power)
        xx = np.asarray(x, dtype=float)
        return np.where(xx <= 0.0, 0.0, 1.0 - np.exp(-xx / mu))

    def sf(self, x: np.ndarray) -> np.ndarray:
        mu = float(self.mean_power)
        xx = np.asarray(x, dtype=float)
        return np.where(xx <= 0.0, 1.0, np.exp(-xx / mu))

    def ppf(self, q: np.ndarray) -> np.ndarray:
        qq = np.asarray(q, dtype=float)
        if np.any(~np.isfinite(qq)) or np.any(qq < 0.0) or np.any(qq > 1.0):
            raise ClutterModelError("q must be in [0,1]")
        mu = float(self.mean_power)
        # Inverse CDF: x = -mu * ln(1-q)
        return -mu * np.log(np.maximum(1.0 - qq, np.finfo(float).tiny))


# ----------------------------
# Weibull power
# ----------------------------

@dataclass(frozen=True)
class WeibullPower:
    """
    Weibull POWER distribution.

    Parameterization uses SciPy weibull_min(c=k, scale=lam).

    - shape k > 0
    - scale lam > 0

    Mean: lam * Gamma(1 + 1/k)
    Var : lam^2 * [Gamma(1 + 2/k) - Gamma(1 + 1/k)^2]
    """
    shape_k: float
    scale_lam: float

    def __post_init__(self) -> None:
        _require_pos_float(self.shape_k, "shape_k")
        _require_pos_float(self.scale_lam, "scale_lam")

    def sample(self, rng: np.random.Generator, size: Union[int, Tuple[int, ...]]) -> np.ndarray:
        # SciPy supports random_state=Generator
        x = weibull_min.rvs(c=self.shape_k, scale=self.scale_lam, size=size, random_state=rng)
        return np.asarray(x, dtype=float)

    def mean_power(self) -> float:
        k = float(self.shape_k)
        lam = float(self.scale_lam)
        return lam * float(special.gamma(1.0 + 1.0 / k))

    def var_power(self) -> float:
        k = float(self.shape_k)
        lam = float(self.scale_lam)
        g1 = float(special.gamma(1.0 + 1.0 / k))
        g2 = float(special.gamma(1.0 + 2.0 / k))
        return (lam * lam) * (g2 - g1 * g1)

    def cdf(self, x: np.ndarray) -> np.ndarray:
        return weibull_min.cdf(np.asarray(x, dtype=float), c=self.shape_k, scale=self.scale_lam)

    def sf(self, x: np.ndarray) -> np.ndarray:
        return weibull_min.sf(np.asarray(x, dtype=float), c=self.shape_k, scale=self.scale_lam)

    def ppf(self, q: np.ndarray) -> np.ndarray:
        qq = np.asarray(q, dtype=float)
        if np.any(qq < 0.0) or np.any(qq > 1.0):
            raise ClutterModelError("q must be in [0,1]")
        return weibull_min.ppf(qq, c=self.shape_k, scale=self.scale_lam)


# ----------------------------
# Lognormal power
# ----------------------------

@dataclass(frozen=True)
class LognormalPower:
    """
    Lognormal POWER distribution.

    Parameterization:
    - If Y ~ Normal(mu, sigma), then X = exp(Y) is lognormal.
    - SciPy lognorm uses shape=sigma and scale=exp(mu).

    Mean: exp(mu + 0.5*sigma^2)
    Var : (exp(sigma^2)-1) * exp(2*mu + sigma^2)
    """
    mu_ln: float
    sigma_ln: float

    def __post_init__(self) -> None:
        _require_pos_float(self.sigma_ln, "sigma_ln")  # sigma must be > 0
        if not isinstance(self.mu_ln, (int, float)) or not math.isfinite(float(self.mu_ln)):
            raise ClutterModelError("mu_ln must be a finite number")

    def sample(self, rng: np.random.Generator, size: Union[int, Tuple[int, ...]]) -> np.ndarray:
        x = lognorm.rvs(s=self.sigma_ln, scale=math.exp(self.mu_ln), size=size, random_state=rng)
        return np.asarray(x, dtype=float)

    def mean_power(self) -> float:
        mu = float(self.mu_ln)
        s = float(self.sigma_ln)
        return math.exp(mu + 0.5 * s * s)

    def var_power(self) -> float:
        mu = float(self.mu_ln)
        s = float(self.sigma_ln)
        return (math.exp(s * s) - 1.0) * math.exp(2.0 * mu + s * s)

    def cdf(self, x: np.ndarray) -> np.ndarray:
        return lognorm.cdf(np.asarray(x, dtype=float), s=self.sigma_ln, scale=math.exp(self.mu_ln))

    def sf(self, x: np.ndarray) -> np.ndarray:
        return lognorm.sf(np.asarray(x, dtype=float), s=self.sigma_ln, scale=math.exp(self.mu_ln))

    def ppf(self, q: np.ndarray) -> np.ndarray:
        qq = np.asarray(q, dtype=float)
        if np.any(qq < 0.0) or np.any(qq > 1.0):
            raise ClutterModelError("q must be in [0,1]")
        return lognorm.ppf(qq, s=self.sigma_ln, scale=math.exp(self.mu_ln))


# ----------------------------
# K-distribution power (spiky clutter)
# ----------------------------

@dataclass(frozen=True)
class KPower:
    """
    K-distribution POWER model via a Gamma mixture of exponentials.

    A common constructive definition:
    - Conditioned on texture G ~ Gamma(shape=v, scale=theta),
      the speckle power is exponential with mean = G.
    - Marginally, power follows a K-like heavy-tailed distribution.

    Sampling approach (robust and fast)
    -----------------------------------
    1) Draw texture g ~ Gamma(v, theta)
    2) Draw power x ~ Exponential(scale=g)

    Parameters
    ----------
    shape_v : float
        Texture gamma shape parameter v > 0 (smaller => heavier tails/spikier).
    scale_theta : float
        Texture gamma scale theta > 0.
    """
    shape_v: float
    scale_theta: float

    def __post_init__(self) -> None:
        _require_pos_float(self.shape_v, "shape_v")
        _require_pos_float(self.scale_theta, "scale_theta")

    def sample(self, rng: np.random.Generator, size: Union[int, Tuple[int, ...]]) -> np.ndarray:
        # texture g ~ Gamma(k=v, theta=scale_theta)
        g = rng.gamma(shape=self.shape_v, scale=self.scale_theta, size=size)
        x = rng.exponential(scale=g, size=size)
        return np.asarray(x, dtype=float)

    def mean_power(self) -> float:
        # E[X] = E[E[X|G]] = E[G] = v*theta
        v = float(self.shape_v)
        th = float(self.scale_theta)
        return v * th

    def var_power(self) -> float:
        # Var[X] = E[Var(X|G)] + Var(E[X|G])
        # For Exp(mean=G): Var(X|G)=G^2, E[X|G]=G
        # => Var[X] = E[G^2] + Var[G]
        # Gamma moments: E[G]=v*th, Var[G]=v*th^2, E[G^2]=Var+Mean^2 = v*th^2 + (v*th)^2
        v = float(self.shape_v)
        th = float(self.scale_theta)
        var_g = v * th * th
        mean_g = v * th
        e_g2 = var_g + mean_g * mean_g
        return e_g2 + var_g

    def cdf(self, x: np.ndarray) -> np.ndarray:
        """
        CDF is non-trivial in closed form for this constructive mixture.
        We provide a numerical approximation via the mixture CDF:

            F_X(x) = E_G[ 1 - exp(-x/G) ],  G>0

        This integral has special-function forms for K-distribution, but to keep this
        module robust and dependency-light, we provide a numerical expectation.

        For performance-critical usage, prefer Monte Carlo sampling or add the
        closed-form with Bessel K functions later.

        Parameters
        ----------
        x : np.ndarray
            Nonnegative powers.

        Returns
        -------
        np.ndarray
            Approximate CDF values in [0,1].
        """
        xx = np.asarray(x, dtype=float)
        if np.any(xx < 0.0) or np.any(~np.isfinite(xx)):
            raise ClutterModelError("x must be finite and >= 0")

        # Numerical expectation using Gauss-Laguerre quadrature would be better,
        # but we keep a deterministic Monte Carlo approximation here.
        # Users needing exact CDF can add closed-form later.
        rng = np.random.default_rng(0)
        g = rng.gamma(shape=self.shape_v, scale=self.scale_theta, size=200_000)
        # Avoid division by tiny g
        g = np.maximum(g, np.finfo(float).tiny)
        # Broadcast
        xx2 = xx.reshape(-1, 1)
        vals = 1.0 - np.exp(-xx2 / g.reshape(1, -1))
        return np.mean(vals, axis=1).reshape(xx.shape)

    def sf(self, x: np.ndarray) -> np.ndarray:
        return 1.0 - self.cdf(x)

    def ppf(self, q: np.ndarray) -> np.ndarray:
        """
        Quantile function computed by numerical inversion using sampling.

        This is intended for diagnostics, not for tight inner loops.
        """
        qq = np.asarray(q, dtype=float)
        if np.any(qq < 0.0) or np.any(qq > 1.0) or np.any(~np.isfinite(qq)):
            raise ClutterModelError("q must be finite in [0,1]")

        rng = np.random.default_rng(1)
        samples = self.sample(rng, size=2_000_000)
        samples.sort()
        idx = np.clip((qq * (len(samples) - 1)).astype(int), 0, len(samples) - 1)
        return samples[idx]


# ----------------------------
# Parameter helpers (optional but practical)
# ----------------------------

def exponential_from_mean(mean_power: float) -> ExponentialPower:
    """Factory helper for exponential power model from mean power."""
    return ExponentialPower(mean_power=_require_pos_float(mean_power, "mean_power"))


def k_from_mean_and_shape(*, mean_power: float, shape_v: float) -> KPower:
    """
    Build a KPower model given mean power and shape v.

    Since mean = v*theta => theta = mean/v
    """
    mp = _require_pos_float(mean_power, "mean_power")
    v = _require_pos_float(shape_v, "shape_v")
    theta = mp / v
    return KPower(shape_v=v, scale_theta=theta)