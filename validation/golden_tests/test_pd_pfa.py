"""
validation/golden_tests/test_pd_pfa.py

Golden test: detection statistics integrity (Pd/Pfa) for the model-based engine.

Goal
----
Ensure the probability-of-detection (Pd) calculation remains correct and stable
as the pipeline evolves (CFAR, clutter models, additional losses, refactors).

What we verify (v1)
-------------------
1) Monotonicity: Pd must increase as SNR increases (for fixed Pfa and N pulses).
2) Integration gain: For fixed SNR per pulse and Pfa, increasing N pulses must
   not reduce Pd (noncoherent integration).
3) Basic bounds: Pd in [0, 1], sane behavior at low/high SNR.

Notes
-----
- We do not attempt to empirically measure Pfa by Monte Carlo here (that's a separate test).
- We instead validate deterministic properties and sanity constraints
  based on the closed-form chi-square / noncentral chi-square implementation.

How to run
----------
From project root:
    pytest -q
"""

from __future__ import annotations

from typing import Any, Dict, List
import numpy as np

from core.simulation.model_based import run_model_based_case


def _cfg_for_detection(*, range_m: float, rcs_sqm: float, pfa: float, n_pulses: int) -> Dict[str, Any]:
    """
    Build a minimal config that will produce Pd at a single range.

    We vary RCS to control SNR without changing range or receiver settings.
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
            "rcs_sqm": rcs_sqm,
        },
        "environment": {
            "system_losses_db": 6.0,
        },
        "metrics": {
            "ranges_m": [range_m],
        },
        "detection": {
            "pfa": pfa,
            "n_pulses": n_pulses,
            "integration": "noncoherent",
        },
    }


def _extract_single_pd(metrics: Dict[str, Any]) -> float:
    det = metrics.get("detection", None)
    assert isinstance(det, dict)
    pd = det.get("pd", None)
    assert isinstance(pd, list) and len(pd) == 1
    return float(pd[0])


def _extract_single_snr(metrics: Dict[str, Any]) -> float:
    snr = metrics.get("snr_lin", None)
    assert isinstance(snr, list) and len(snr) == 1
    return float(snr[0])


def test_pd_monotonic_in_snr() -> None:
    """
    Pd must strictly increase (or at least not decrease) as SNR increases
    for fixed Pfa and N pulses.
    """
    pfa = 1e-6
    n_pulses = 32
    r = 20_000.0

    # Sweep RCS over orders of magnitude to induce SNR increases.
    rcs_values = [0.01, 0.1, 1.0, 10.0, 100.0]

    pds: List[float] = []
    snrs: List[float] = []

    for sigma in rcs_values:
        cfg = _cfg_for_detection(range_m=r, rcs_sqm=sigma, pfa=pfa, n_pulses=n_pulses)
        m = run_model_based_case(cfg, seed=0)
        pds.append(_extract_single_pd(m))
        snrs.append(_extract_single_snr(m))

    # Sanity: SNR should increase with sigma (since Pr ∝ sigma).
    assert all(snrs[i] < snrs[i + 1] for i in range(len(snrs) - 1))

    # Pd must be non-decreasing with SNR.
    assert all(pds[i] <= pds[i + 1] + 1e-15 for i in range(len(pds) - 1))

    # Bounds.
    assert all(0.0 <= pd <= 1.0 for pd in pds)

    # Optional extra sanity: very high SNR should yield high Pd.
    assert pds[-1] > 0.9


def test_noncoherent_integration_improves_pd() -> None:
    """
    For fixed per-pulse SNR and fixed Pfa, increasing the number of integrated pulses
    should not reduce Pd for the noncoherent energy detector.
    """
    pfa = 1e-6
    r = 25_000.0

    # Fix sigma to hold per-pulse SNR constant.
    sigma = 1.0

    n_list = [1, 4, 16, 64]
    pds: List[float] = []

    for n in n_list:
        cfg = _cfg_for_detection(range_m=r, rcs_sqm=sigma, pfa=pfa, n_pulses=n)
        m = run_model_based_case(cfg, seed=0)
        pds.append(_extract_single_pd(m))

    assert all(0.0 <= pd <= 1.0 for pd in pds)

    # Pd should be non-decreasing with N pulses.
    assert all(pds[i] <= pds[i + 1] + 1e-15 for i in range(len(pds) - 1))

    # And we should see a meaningful improvement at higher N.
    assert pds[-1] > pds[0]


def test_pd_sane_at_low_and_high_snr() -> None:
    """
    Low SNR => Pd near Pfa-ish regime (not necessarily equal, but very low).
    High SNR => Pd near 1.
    """
    pfa = 1e-6
    n_pulses = 16
    r = 30_000.0

    # Very small RCS -> very low SNR
    cfg_low = _cfg_for_detection(range_m=r, rcs_sqm=1e-6, pfa=pfa, n_pulses=n_pulses)
    m_low = run_model_based_case(cfg_low, seed=0)
    pd_low = _extract_single_pd(m_low)

    # Very large RCS -> very high SNR
    cfg_high = _cfg_for_detection(range_m=r, rcs_sqm=1e3, pfa=pfa, n_pulses=n_pulses)
    m_high = run_model_based_case(cfg_high, seed=0)
    pd_high = _extract_single_pd(m_high)

    assert 0.0 <= pd_low <= 1.0
    assert 0.0 <= pd_high <= 1.0

    # Loose but meaningful expectations.
    assert pd_low < 0.05
    assert pd_high > 0.95