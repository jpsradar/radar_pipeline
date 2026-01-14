"""
tests/test_metric_queries.py

Deterministic tests for metric query helpers.

Purpose
-------
This module validates small, query-oriented helpers that extract and interpret
information from a metrics.json payload produced by the radar pipeline.

These helpers answer common engineering questions such as:
- At what maximum range does Pd remain above a given threshold?
- What is Pd at a specific range (with interpolation)?
- Are range and Pd vectors well-formed and consistent?

The intent is to make these queries:
- explicit
- deterministic
- stable under refactoring

Scope
-----
- Model-based style metrics payloads with Pd defined as a function of range.
- Pure functions only (no I/O, no plotting, no randomness).

These tests are not concerned with how Pd is computed, only with how reported
metrics are interpreted.
"""

from __future__ import annotations

from typing import Any, Dict

import pytest

from core.metrics.performance import (
    extract_pd_curve,
    extract_ranges_m,
    pd_at_range,
    range_at_pd,
)


def _metrics_fixture() -> Dict[str, Any]:
    """
    Minimal, well-formed metrics payload for query testing.

    The structure matches the conventions used by model_based engines:
    - ranges_m defines the evaluation grid
    - detection.pd defines Pd at each range

    Values are chosen to be monotonic and easy to reason about.
    """
    return {
        "engine": "model_based",
        "ranges_m": [10_000.0, 20_000.0, 30_000.0],
        "detection": {
            "pd": [0.95, 0.80, 0.40],
        },
    }


def test_extract_ranges_and_pd_curve() -> None:
    """
    Range and Pd extractors must return the expected vectors
    when the metrics payload follows the standard convention.
    """
    m = _metrics_fixture()
    assert extract_ranges_m(m) == [10_000.0, 20_000.0, 30_000.0]
    assert extract_pd_curve(m) == [0.95, 0.80, 0.40]


def test_range_at_pd_thresholds() -> None:
    """
    range_at_pd must return the maximum range at which Pd >= threshold.
    """
    m = _metrics_fixture()

    assert range_at_pd(m, pd_min=0.90) == 10_000.0
    assert range_at_pd(m, pd_min=0.80) == 20_000.0
    assert range_at_pd(m, pd_min=0.50) == 20_000.0
    assert range_at_pd(m, pd_min=0.10) == 30_000.0


def test_range_at_pd_returns_none_if_threshold_not_met() -> None:
    """
    If Pd never reaches the requested threshold, the query must return None.
    """
    m = _metrics_fixture()
    assert range_at_pd(m, pd_min=0.99) is None


def test_pd_at_range_exact_and_interpolated() -> None:
    """
    pd_at_range must:
    - return exact values at grid points
    - linearly interpolate between adjacent ranges
    """
    m = _metrics_fixture()

    # Exact grid points
    assert pd_at_range(m, range_m=10_000.0) == 0.95
    assert pd_at_range(m, range_m=30_000.0) == 0.40

    # Linear interpolation between 10 km (0.95) and 20 km (0.80)
    pd_mid = pd_at_range(m, range_m=15_000.0)
    assert pd_mid is not None
    assert abs(pd_mid - 0.875) < 1e-12


def test_pd_at_range_outside_grid_returns_none() -> None:
    """
    pd_at_range must not extrapolate outside the provided range grid.
    """
    m = _metrics_fixture()
    assert pd_at_range(m, range_m=5_000.0) is None
    assert pd_at_range(m, range_m=40_000.0) is None


def test_invalid_pd_threshold_is_rejected() -> None:
    """
    Invalid Pd thresholds must be rejected explicitly.
    """
    m = _metrics_fixture()

    with pytest.raises(ValueError):
        range_at_pd(m, pd_min=0.0)

    with pytest.raises(ValueError):
        range_at_pd(m, pd_min=1.1)