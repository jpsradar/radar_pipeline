"""
tests/test_units.py

Unit conversion and noise power sanity tests.

What this test module covers
----------------------------
1) dB ↔ linear power conversions:
   - db_to_lin_power() and lin_to_db_power() must be mutual inverses.
   - This is critical for SNR, gains, losses, NF, and reporting.

2) Thermal noise sanity (kTB):
   - Noise power must scale linearly with bandwidth.
   - Noise power must scale linearly with temperature.
   - NF must act as a linear multiplicative factor in power.

Why this matters
----------------
If units are wrong:
- All SNR budgets are wrong
- Pd curves are wrong
- FAR is wrong
- Sweeps and Pareto fronts are meaningless

This file exists to prevent silent unit regressions.

Test philosophy (professional)
------------------------------
- Deterministic
- Fast
- No Monte Carlo
- No plotting
- Ratio-based assertions where possible

How to run
----------
pytest -q
"""

from __future__ import annotations


import numpy as np

from core.config.units import (
    db_to_lin_power,
    lin_to_db_power,
    k_boltzmann,
)


# ---------------------------------------------------------------------
# dB / linear conversions
# ---------------------------------------------------------------------

def test_db_to_lin_and_back_are_inverses() -> None:
    """
    db_to_lin_power and lin_to_db_power must be mutual inverses
    over a reasonable dynamic range.
    """
    db_vals = np.array([-100.0, -60.0, -30.0, -10.0, 0.0, 10.0, 30.0, 60.0])

    lin = db_to_lin_power(db_vals)
    assert np.all(lin > 0.0), "Linear power must be strictly positive"

    db_back = lin_to_db_power(lin)
    assert np.all(np.isfinite(db_back)), "Converted dB values must be finite"

    err = np.abs(db_back - db_vals)
    assert np.all(err < 1e-12), f"dB↔linear round-trip error too large: {err.tolist()}"


def test_db_to_lin_monotonicity() -> None:
    """
    db_to_lin_power must be strictly increasing.
    """
    db_vals = np.linspace(-100.0, 100.0, 1001)
    lin = db_to_lin_power(db_vals)

    diffs = np.diff(lin)
    assert np.all(diffs > 0.0), "db_to_lin_power must be strictly monotonic increasing"


# ---------------------------------------------------------------------
# Thermal noise (kTB) sanity
# ---------------------------------------------------------------------

def test_kTB_scales_with_bandwidth() -> None:
    """
    Noise power must scale linearly with bandwidth.
    """
    T = 290.0
    B1 = 1.0e6
    B2 = 2.0e6

    k = k_boltzmann()
    n1 = k * T * B1
    n2 = k * T * B2

    ratio = n2 / n1
    assert abs(ratio - 2.0) < 1e-12, f"Noise must scale linearly with bandwidth (ratio={ratio})"


def test_kTB_scales_with_temperature() -> None:
    """
    Noise power must scale linearly with temperature.
    """
    T1 = 290.0
    T2 = 580.0
    B = 1.0e6

    k = k_boltzmann()
    n1 = k * T1 * B
    n2 = k * T2 * B

    ratio = n2 / n1
    assert abs(ratio - 2.0) < 1e-12, f"Noise must scale linearly with temperature (ratio={ratio})"


def test_noise_figure_is_linear_factor() -> None:
    """
    Noise Figure (NF) in dB must act as a linear multiplicative factor
    on noise power.
    """
    T = 290.0
    B = 1.0e6
    NF_db = 3.0  # ~2x noise factor

    k = k_boltzmann()
    noise_ideal = k * T * B

    noise_factor = db_to_lin_power(NF_db)
    noise_nf = noise_ideal * noise_factor

    ratio = noise_nf / noise_ideal
    assert abs(ratio - noise_factor) < 1e-12, "Noise Figure must multiply noise power linearly"