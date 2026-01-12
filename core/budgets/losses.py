"""
core/budgets/losses.py

Loss / gain budgeting helpers.

What this module does
---------------------
Provides small, explicit helpers for combining losses and gains consistently:
- Convert between dB and linear power ratios (delegated to core.config.units)
- Combine multiple losses (in dB) safely
- Validate common constraints (e.g., non-negative system losses)

Scope (v1)
----------
This module is intentionally minimal and does NOT attempt to model specific
hardware chains. It provides primitives used by budget calculations.

Important conventions
---------------------
- dB values represent POWER ratios unless explicitly stated otherwise.
- Losses in dB are non-negative numbers (0 means no loss).
- Gains in dB can be positive or negative.
"""

from __future__ import annotations

from typing import Iterable, Optional
import math

from core.config.units import db_to_lin_power, lin_to_db_power


class LossBudgetError(ValueError):
    """Raised when loss budgeting inputs are invalid."""


def sum_losses_db(losses_db: Iterable[float], *, allow_none: bool = True) -> float:
    """
    Sum a collection of loss terms expressed in dB.

    Parameters
    ----------
    losses_db : iterable[float]
        Individual loss terms in dB (typically >= 0).
    allow_none : bool
        If True, ignores None values.

    Returns
    -------
    float
        Total loss in dB.
    """
    total = 0.0
    for x in losses_db:
        if x is None and allow_none:  # type: ignore[comparison-overlap]
            continue
        if x is None:
            raise LossBudgetError("Loss term is None but allow_none=False")
        if not isinstance(x, (int, float)) or isinstance(x, bool):
            raise LossBudgetError(f"Loss term must be numeric, got {type(x).__name__}")
        if not math.isfinite(float(x)):
            raise LossBudgetError(f"Loss term must be finite, got {x}")
        # We do not enforce non-negativity here because some users store "gain" terms in the same list.
        total += float(x)
    return float(total)


def loss_db_to_linear(loss_db: float) -> float:
    """
    Convert a loss in dB (power ratio) to a linear multiplier L >= 1.

    Example
    -------
    loss_db=3 dB => L ~= 1.995 (approx 2x loss)
    """
    if not isinstance(loss_db, (int, float)) or isinstance(loss_db, bool):
        raise LossBudgetError(f"loss_db must be numeric, got {type(loss_db).__name__}")
    if not math.isfinite(float(loss_db)):
        raise LossBudgetError(f"loss_db must be finite, got {loss_db}")
    return float(db_to_lin_power(float(loss_db)))


def linear_to_loss_db(loss_linear: float) -> float:
    """
    Convert a linear loss multiplier L >= 1 to a dB loss value.

    Example
    -------
    L=2 => loss_db ~= 3.0103 dB
    """
    if not isinstance(loss_linear, (int, float)) or isinstance(loss_linear, bool):
        raise LossBudgetError(f"loss_linear must be numeric, got {type(loss_linear).__name__}")
    if not math.isfinite(float(loss_linear)):
        raise LossBudgetError(f"loss_linear must be finite, got {loss_linear}")
    if float(loss_linear) <= 0.0:
        raise LossBudgetError(f"loss_linear must be > 0, got {loss_linear}")
    return float(lin_to_db_power(float(loss_linear)))


def validate_nonnegative_db(x_db: Optional[float], *, name: str) -> float:
    """
    Validate that a dB value is finite and >= 0, returning a float.

    This is useful for inputs such as "system_losses_db" that are conceptually losses.
    """
    if x_db is None:
        return 0.0
    if not isinstance(x_db, (int, float)) or isinstance(x_db, bool):
        raise LossBudgetError(f"{name} must be numeric, got {type(x_db).__name__}")
    v = float(x_db)
    if not math.isfinite(v):
        raise LossBudgetError(f"{name} must be finite, got {x_db}")
    if v < 0.0:
        raise LossBudgetError(f"{name} must be >= 0 dB, got {v}")
    return v