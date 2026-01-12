"""
sweeps/pareto.py

Pareto front extraction for multi-objective radar trade studies.

What this module does
---------------------
Given a set of points (metrics), identifies the non-dominated subset
under user-defined objective directions.

This is used by reporting to:
- summarize trade-offs
- show "best achievable" frontier for competing objectives

Definitions
-----------
A point A dominates point B if A is at least as good as B in all objectives
and strictly better in at least one objective, under the chosen directions.

Inputs
------
- values: dict[str, list[float]]  (one list per objective, same length)
- directions: dict[str, "min"|"max"]

Outputs
-------
- pareto_idx: list[int] indices of non-dominated points
"""

from __future__ import annotations

from typing import Dict, List, Literal
import numpy as np


Direction = Literal["min", "max"]


class ParetoError(ValueError):
    """Raised when Pareto inputs are invalid."""


def pareto_front_indices(
    *,
    values: Dict[str, List[float]],
    directions: Dict[str, Direction],
) -> List[int]:
    """
    Compute indices of the Pareto front.

    Parameters
    ----------
    values : dict[str, list[float]]
        Objective arrays (same length).
    directions : dict[str, "min"|"max"]
        Objective direction per key.

    Returns
    -------
    list[int]
        Indices of non-dominated points.
    """
    if not values:
        raise ParetoError("values must be non-empty")

    keys = list(values.keys())
    n = len(values[keys[0]])
    if n == 0:
        raise ParetoError("objective lists must be non-empty")

    for k in keys:
        if len(values[k]) != n:
            raise ParetoError("all objective lists must have the same length")
        if k not in directions:
            raise ParetoError(f"missing direction for objective '{k}'")
        if directions[k] not in ("min", "max"):
            raise ParetoError(f"invalid direction for '{k}': {directions[k]}")

    # Convert to matrix with all objectives transformed to minimization.
    mat = []
    for k in keys:
        arr = np.asarray(values[k], dtype=float)
        if directions[k] == "max":
            arr = -arr
        mat.append(arr)
    X = np.stack(mat, axis=1)  # shape: (n_points, n_obj)

    # Non-dominated check (O(n^2) is fine for moderate sweep sizes).
    dominated = np.zeros(n, dtype=bool)

    for i in range(n):
        if dominated[i]:
            continue
        xi = X[i]
        for j in range(n):
            if i == j or dominated[i]:
                continue
            xj = X[j]
            # j dominates i if xj <= xi in all and < in at least one (minimization space)
            if np.all(xj <= xi) and np.any(xj < xi):
                dominated[i] = True

    return [i for i in range(n) if not dominated[i]]