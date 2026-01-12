"""
core/tracker/metrics.py

Tracker metrics and diagnostics for the radar pipeline (v1).

Purpose
-------
Provide small, deterministic utilities to summarize tracker behavior for:
- metrics.json outputs
- regressions / sanity checks
- reporting (tables/plots in reports/)

This module does not implement tracking. It only computes metrics from Track objects
(see core/tracker/logic.py).

Metrics provided (v1)
---------------------
- Track counts: total / confirmed / tentative
- Track ages and hit/miss rates
- Innovation (residual) statistics if provided by caller
- Basic consistency checks on covariance (PSD-ish and finite)

Inputs
------
- tracks: list[core.tracker.logic.Track]
- optional per-track innovations / residuals supplied as arrays

Outputs
-------
- dict[str, Any] safe to serialize to JSON

Dependencies
------------
- numpy
- core/tracker/logic.Track (runtime import only; no circular import issues in normal use)

Usage
-----
This module is typically called by:
- CLI run_case pipeline when tracker is integrated in v2+
- validation harness for tracker smoke tests

Stability / Compatibility
-------------------------
The output schema is stable in v1: keys documented below will not be renamed.
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence

import numpy as np


def summarize_tracks(tracks: Sequence[Any]) -> Dict[str, Any]:
    """
    Summarize a list of Track-like objects.

    Requirements on each element
    ----------------------------
    - has attributes: status (str), age_steps (int), history.consecutive_misses (int)
    - has state mean x (array-like len 6) and covariance P (6x6)

    Returns
    -------
    dict
        JSON-serializable summary.
    """
    total = int(len(tracks))
    confirmed = 0
    tentative = 0
    ages: List[int] = []
    misses: List[int] = []
    cov_ok = 0
    cov_bad = 0

    for trk in tracks:
        st = str(getattr(trk, "status", ""))
        if st == "confirmed":
            confirmed += 1
        elif st == "tentative":
            tentative += 1

        ages.append(int(getattr(trk, "age_steps", 0)))

        hist = getattr(trk, "history", None)
        cm = int(getattr(hist, "consecutive_misses", 0)) if hist is not None else 0
        misses.append(cm)

        P = np.asarray(getattr(trk, "P", np.zeros((6, 6))), dtype=float)
        if _covariance_sane(P):
            cov_ok += 1
        else:
            cov_bad += 1

    out: Dict[str, Any] = {
        "counts": {
            "total": total,
            "confirmed": int(confirmed),
            "tentative": int(tentative),
        },
        "ages_steps": {
            "min": int(min(ages)) if ages else 0,
            "median": float(np.median(np.asarray(ages, dtype=float))) if ages else 0.0,
            "max": int(max(ages)) if ages else 0,
        },
        "consecutive_misses": {
            "min": int(min(misses)) if misses else 0,
            "median": float(np.median(np.asarray(misses, dtype=float))) if misses else 0.0,
            "max": int(max(misses)) if misses else 0,
        },
        "covariance_health": {
            "ok": int(cov_ok),
            "bad": int(cov_bad),
        },
    }
    return out


def innovation_stats(innovations: Sequence[np.ndarray]) -> Dict[str, Any]:
    """
    Compute summary stats for a sequence of innovation vectors (residuals).

    Parameters
    ----------
    innovations : sequence of np.ndarray
        Each element should be shape (k,), typically (3,) for position residuals.

    Returns
    -------
    dict
        JSON-serializable stats.
    """
    if len(innovations) == 0:
        return {"count": 0}

    mags: List[float] = []
    for v in innovations:
        a = np.asarray(v, dtype=float).reshape(-1)
        if a.size == 0 or not np.all(np.isfinite(a)):
            continue
        mags.append(float(np.linalg.norm(a)))

    if len(mags) == 0:
        return {"count": 0}

    x = np.asarray(mags, dtype=float)
    return {
        "count": int(x.size),
        "mean_norm": float(np.mean(x)),
        "median_norm": float(np.median(x)),
        "p90_norm": float(np.percentile(x, 90.0)),
        "max_norm": float(np.max(x)),
    }


def _covariance_sane(P: np.ndarray) -> bool:
    """
    Conservative covariance sanity check.

    We do NOT do a full PSD proof; we check:
    - correct shape
    - finite entries
    - symmetry (within tolerance)
    - non-negative diagonal
    """
    if P.shape != (6, 6):
        return False
    if not np.all(np.isfinite(P)):
        return False
    if not np.allclose(P, P.T, atol=1e-9, rtol=0.0):
        return False
    d = np.diag(P)
    if np.any(d < -1e-12):
        return False
    return True