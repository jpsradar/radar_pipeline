"""
validation/golden_tests/test_radar_equation.py

Golden test: monostatic radar equation sanity + invariants.

Goal
----
Lock down the core physics relationship so future refactors (losses, gains, units)
do not silently break the fundamental scaling laws.

What we verify (v1)
-------------------
1) Received power scales as R^-4 for monostatic radar equation.
2) SNR scales the same way when noise power is constant (fixed k*T*B*F).
3) Budget terms are present and reasonable.

How to run
----------
From project root:
    pytest -q

Notes
-----
- This test intentionally does NOT validate exact absolute power numbers.
  Absolute values depend on many conventions (peak vs avg power, processing gains, etc).
  The scaling law is the non-negotiable invariant.
"""

from __future__ import annotations

import math
from typing import Any, Dict

import numpy as np

from core.simulation.model_based import run_model_based_case


def _minimal_cfg_for_budget(ranges_m: list[float]) -> Dict[str, Any]:
    """
    Build a minimal configuration dict that exercises the model-based engine.

    We keep numbers plausible but the test asserts ratios (scale laws), not absolutes.
    """
    return {
        "radar": {
            "fc_hz": 9.6e9,
            "tx_power_w": 1500.0,
        },
        "antenna": {
            "gain_tx_db": 33.0,
            "gain_rx_db": 33.0,
        },
        "receiver": {
            "bw_hz": 3.0e6,
            "nf_db": 4.5,
            "temperature_k": 290.0,
        },
        "target": {
            "rcs_sqm": 1.0,
        },
        "environment": {
            "system_losses_db": 6.0,
        },
        "metrics": {
            "ranges_m": ranges_m,
        },
        # Detection is not needed for this golden test.
    }


def test_received_power_scales_with_r_to_minus_4() -> None:
    """
    For monostatic radar equation: Pr ∝ 1 / R^4

    We test the ratio:
        Pr(R1) / Pr(R2) == (R2/R1)^4
    """
    r1 = 10_000.0
    r2 = 20_000.0
    cfg = _minimal_cfg_for_budget([r1, r2])

    metrics = run_model_based_case(cfg, seed=0)

    pr = np.array(metrics["received_power_w"], dtype=float)
    assert pr.shape == (2,)
    assert pr[0] > 0.0 and pr[1] > 0.0

    observed_ratio = pr[0] / pr[1]
    expected_ratio = (r2 / r1) ** 4  # (20000/10000)^4 = 16

    # We allow small numerical error.
    assert math.isfinite(observed_ratio)
    assert math.isclose(observed_ratio, expected_ratio, rel_tol=1e-10, abs_tol=0.0)


def test_snr_scales_with_r_to_minus_4_when_noise_fixed() -> None:
    """
    With fixed receiver noise power (k*T*B*F constant), SNR ∝ Pr ∝ 1/R^4.
    """
    r1 = 12_000.0
    r2 = 24_000.0
    cfg = _minimal_cfg_for_budget([r1, r2])

    metrics = run_model_based_case(cfg, seed=0)

    snr = np.array(metrics["snr_lin"], dtype=float)
    assert snr.shape == (2,)
    assert snr[0] > 0.0 and snr[1] > 0.0

    observed_ratio = snr[0] / snr[1]
    expected_ratio = (r2 / r1) ** 4  # (24000/12000)^4 = 16

    assert math.isfinite(observed_ratio)
    assert math.isclose(observed_ratio, expected_ratio, rel_tol=1e-10, abs_tol=0.0)


def test_budget_terms_present_and_reasonable() -> None:
    """
    Ensure key budget fields exist and have sane ranges (non-negotiable traceability).
    """
    cfg = _minimal_cfg_for_budget([10_000.0])
    metrics = run_model_based_case(cfg, seed=0)

    budget = metrics.get("budget", {})
    assert isinstance(budget, dict)

    # Wavelength for ~9.6 GHz should be around 0.031 m (order-of-magnitude check).
    wavelength_m = float(budget["wavelength_m"])
    assert 0.005 < wavelength_m < 0.10

    # Gains and losses are linear factors >= 0.
    gt_lin = float(budget["gt_lin"])
    gr_lin = float(budget["gr_lin"])
    losses_lin = float(budget["losses_lin"])
    assert gt_lin > 0.0
    assert gr_lin > 0.0
    assert losses_lin >= 1.0  # losses are modeled as a factor >= 1

    # Noise power must be strictly positive.
    noise_power_w = float(budget["noise_power_w"])
    assert noise_power_w > 0.0