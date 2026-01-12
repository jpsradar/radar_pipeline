"""
tests/test_monotonicity.py

Monotonicity sanity tests for detection performance.

What this test module covers
----------------------------
1) Pd monotonicity vs SNR:
   - For fixed Pfa and integration settings, Pd must be non-decreasing as SNR increases.
   - We enforce SNR increase via decreasing range (closer target => higher received power => higher SNR).

Why this matters
----------------
This test catches broken wiring in:
- threshold computation
- chi-square / noncentral chi-square usage
- integration parameter handling (n_pulses, coherent/noncoherent)

If Pd is not monotonic, your ROC/Pd(R) curves are untrustworthy.

Test philosophy (professional)
------------------------------
- Deterministic
- Fast (no Monte Carlo)
- No I/O, no plotting
- Strong assertions with actionable error messages

How to run
----------
pytest -q
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from core.simulation.model_based import run_model_based_case


def _base_cfg() -> Dict[str, Any]:
    """
    Minimal deterministic config for model_based monotonicity tests.

    Notes
    -----
    Values are physically plausible but not tied to a specific radar.
    """
    return {
        "radar": {
            "fc_hz": 10.0e9,
            "tx_power_w": 1.0e3,
            "prf_hz": 1.0e3,
        },
        "antenna": {"gain_tx_db": 30.0, "gain_rx_db": 30.0},
        "receiver": {"bw_hz": 5.0e6, "nf_db": 5.0, "temperature_k": 290.0},
        "target": {"rcs_sqm": 1.0},
        "environment": {"system_losses_db": 0.0},
        "geometry": {"n_range_bins": 256, "n_doppler_bins": 16, "beams_per_scan": 1, "n_cpi_per_dwell": 1},
    }


def test_pd_noncoherent_is_monotonic_with_snr() -> None:
    """
    Pd must be non-decreasing as SNR increases (noncoherent integration).

    We pick decreasing ranges so SNR increases monotonically.
    """
    cfg = _base_cfg()
    cfg["detection"] = {"pfa": 1e-6, "n_pulses": 16, "integration": "noncoherent"}
    cfg["metrics"] = {"ranges_m": [40_000.0, 30_000.0, 20_000.0, 15_000.0, 10_000.0]}

    m = run_model_based_case(cfg, seed=None)

    det = m.get("detection", None)
    assert isinstance(det, dict) and "pd" in det, "metrics must include detection.pd when detection.pfa is provided"

    pd = det["pd"]
    assert isinstance(pd, list) and len(pd) >= 2, "detection.pd must be a list with at least 2 entries"

    pd_arr = np.asarray(pd, dtype=float)
    assert np.all(np.isfinite(pd_arr)), "Pd must not contain NaN/Inf"

    # Allow extremely small numerical noise.
    diffs = np.diff(pd_arr)
    assert np.all(diffs >= -1e-12), f"Pd is not monotonic non-decreasing (noncoherent): Pd={pd_arr.tolist()}"


def test_pd_coherent_is_monotonic_with_snr() -> None:
    """
    Pd must be non-decreasing as SNR increases (coherent-like baseline in model_based).

    This validates that coherent-mode mapping does not break monotonicity.
    """
    cfg = _base_cfg()
    cfg["detection"] = {"pfa": 1e-6, "n_pulses": 16, "integration": "coherent"}
    cfg["metrics"] = {"ranges_m": [40_000.0, 30_000.0, 20_000.0, 15_000.0, 10_000.0]}

    m = run_model_based_case(cfg, seed=None)

    det = m.get("detection", None)
    assert isinstance(det, dict) and "pd" in det, "metrics must include detection.pd when detection.pfa is provided"

    pd = det["pd"]
    assert isinstance(pd, list) and len(pd) >= 2, "detection.pd must be a list with at least 2 entries"

    pd_arr = np.asarray(pd, dtype=float)
    assert np.all(np.isfinite(pd_arr)), "Pd must not contain NaN/Inf"

    diffs = np.diff(pd_arr)
    assert np.all(diffs >= -1e-12), f"Pd is not monotonic non-decreasing (coherent): Pd={pd_arr.tolist()}"