"""
tests/test_energy.py

Energy / power-domain sanity tests for the radar pipeline.

What this test module covers
----------------------------
1) Monostatic radar equation scaling:
   - Received power must scale as 1/R^4.
   - We validate Pr(R1)/Pr(R2) against (R2/R1)^4.

Why this matters
----------------
If this breaks, your entire performance engine is garbage:
- SNR vs range will be wrong
- Pd vs range will be wrong
- any sweep/report built on top will be misleading

Test philosophy (professional)
------------------------------
- Deterministic
- Fast (no Monte Carlo)
- No I/O, no plotting
- Tight numerical tolerances (this should be exact to floating precision)

How to run
----------
pytest -q
"""

from __future__ import annotations

from typing import Any, Dict

from core.simulation.model_based import run_model_based_case


def _base_cfg() -> Dict[str, Any]:
    """
    Minimal deterministic config for model_based engine tests.

    Notes
    -----
    Values are physically plausible but not tied to any specific radar program.
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


def test_received_power_scales_as_inverse_r4() -> None:
    """
    Validate Pr ~ 1/R^4 by comparing two ranges.

    Acceptance
    ----------
    Relative error must be extremely small because:
    - both values are computed by the same formula
    - the ratio cancels most constants
    """
    cfg = _base_cfg()
    cfg["metrics"] = {"ranges_m": [10_000.0, 20_000.0]}  # 10 km, 20 km

    m = run_model_based_case(cfg, seed=None)

    pr = m.get("received_power_w", None)
    assert isinstance(pr, list) and len(pr) == 2, (
        "metrics must include received_power_w list of length 2 "
        "(expected when metrics.ranges_m has 2 entries)"
    )

    pr1 = float(pr[0])
    pr2 = float(pr[1])
    assert pr1 > 0.0 and pr2 > 0.0, "received power must be positive at both ranges"

    ratio_emp = pr1 / pr2
    ratio_theory = (20_000.0 / 10_000.0) ** 4  # (R2/R1)^4

    rel_err = abs(ratio_emp - ratio_theory) / ratio_theory
    assert rel_err < 1e-10, (
        f"Radar equation scaling mismatch: empirical={ratio_emp:.6g}, "
        f"theory={ratio_theory:.6g}, rel_err={rel_err:.3e}"
    )