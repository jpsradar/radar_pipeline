"""
core/antennas/scan_loss.py

Scan loss and pointing loss utilities (v1).

Purpose
-------
Provide explicit, auditable scan/pointing loss terms suitable for:
- link budgets (effective gain reductions off-boresight),
- system-level performance summaries,
- traceable reporting (loss in linear and dB).

Definitions
-----------
Given a normalized antenna pattern G(θ) in linear power (peak=1 at boresight):
- Pointing/scan loss (linear) is defined as:
      L(θ) = 1 / max(G(θ), eps)
  so L >= 1 and L_dB >= 0.

Scope (v1)
----------
- Compute loss from a supplied pattern function.
- Support scalar and NumPy array evaluation.
- Provide safe clamps to avoid division by zero.

Non-goals (v1)
--------------
- No scheduler integration (that belongs in geometry/scan_scheduler.py).
- No time-varying scan strategies; this is purely instantaneous loss evaluation.

Public API (stable)
-------------------
- pointing_loss_lin(theta_rad, pattern_fn, **pattern_kwargs) -> loss_lin
- scan_loss_db(theta_rad, pattern_fn, **pattern_kwargs) -> loss_db

Dependencies
------------
- Python stdlib: math
- NumPy (optional)

Usage
-----
    from core.antennas.patterns import pattern_cosine
    from core.antennas.scan_loss import scan_loss_db

    loss_db = scan_loss_db(theta_rad=0.25, pattern_fn=pattern_cosine, n=10)

Outputs
-------
- loss_lin: float or np.ndarray (>= 1)
- loss_db: float or np.ndarray (>= 0)

Quality properties
------------------
- Deterministic, pure functions
- No I/O
- Defensive validation of finiteness and non-negativity
"""

from __future__ import annotations

from typing import Any, Callable, Union
import math

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None  # type: ignore


NumberOrArray = Union[float, "np.ndarray"]  # type: ignore[name-defined]
PatternFn = Callable[..., NumberOrArray]


def pointing_loss_lin(
    theta_rad: NumberOrArray,
    *,
    pattern_fn: PatternFn,
    eps: float = 1e-30,
    **pattern_kwargs: Any,
) -> NumberOrArray:
    """
    Compute pointing/scan loss in linear units from a normalized pattern.

    Parameters
    ----------
    theta_rad : float or np.ndarray
        Off-boresight angle(s) in radians.
    pattern_fn : callable
        Pattern function returning linear power gain (ideally peak-normalized to 1 at boresight).
    eps : float
        Small floor to avoid division by zero; must be > 0.
    **pattern_kwargs
        Passed through to pattern_fn.

    Returns
    -------
    float or np.ndarray
        Linear loss L = 1/max(G, eps), so L >= 1.
    """
    _require_finite_positive_scalar(eps, name="eps")
    g = pattern_fn(theta_rad, **pattern_kwargs)

    if _is_ndarray(g):
        gg = np.asarray(g, dtype=float)  # type: ignore[union-attr]
        _require_finite_array(gg, name="gain")
        if np.any(gg < 0.0):  # type: ignore[union-attr]
            raise ValueError("pattern_fn returned negative gain values (invalid for power gain)")
        out = 1.0 / np.maximum(gg, float(eps))  # type: ignore[union-attr]
        return out

    _require_finite_scalar(g, name="gain")
    gg = float(g)
    if gg < 0.0:
        raise ValueError("pattern_fn returned negative gain (invalid for power gain)")
    return 1.0 / max(gg, float(eps))


def scan_loss_db(
    theta_rad: NumberOrArray,
    *,
    pattern_fn: PatternFn,
    eps: float = 1e-30,
    **pattern_kwargs: Any,
) -> NumberOrArray:
    """
    Compute scan/pointing loss in dB.

    Parameters
    ----------
    theta_rad : float or np.ndarray
        Off-boresight angle(s) in radians.
    pattern_fn : callable
        Pattern function returning linear power gain.
    eps : float
        Gain floor to avoid log(0); must be > 0.
    **pattern_kwargs
        Passed through to pattern_fn.

    Returns
    -------
    float or np.ndarray
        Loss in dB (>= 0).
    """
    loss_lin = pointing_loss_lin(theta_rad, pattern_fn=pattern_fn, eps=eps, **pattern_kwargs)

    if _is_ndarray(loss_lin):
        ll = np.asarray(loss_lin, dtype=float)  # type: ignore[union-attr]
        _require_finite_array(ll, name="loss_lin")
        if np.any(ll < 1.0 - 1e-15):  # allow tiny numeric noise
            raise ValueError("loss_lin must be >= 1 for scan loss definition")
        return 10.0 * np.log10(ll)  # type: ignore[union-attr]

    _require_finite_scalar(loss_lin, name="loss_lin")
    ll = float(loss_lin)
    if ll < 1.0 - 1e-15:
        raise ValueError("loss_lin must be >= 1 for scan loss definition")
    return 10.0 * math.log10(ll)


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