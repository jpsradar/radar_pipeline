"""
tests/test_physical_monotonicity.py

Physical monotonicity invariants for the radar performance engine (model_based).

What this module covers
-----------------------
These are "engineering sanity" invariants that must hold for a noise-limited,
deterministic point-target baseline under a fixed detection rule.

1) Received power monotonicity vs range:
   - Monostatic free-space radar equation implies Pr must strictly decrease as range increases.

2) SNR monotonicity vs range:
   - If noise power is constant and Pr decreases, SNR(dB) must strictly decrease with range.

3) Pd monotonicity vs range (under fixed Pfa and fixed integration):
   - Under a fixed detector threshold (fixed Pfa) and fixed integration settings,
     Pd must be non-increasing with range for a deterministic target in noise.

4) Pd monotonicity vs transmit power (under identical geometry/detector):
   - Increasing transmit power must not reduce Pd at any range.
   - It should strictly improve Pd somewhere away from saturation (not all-1 or all-0).

Why this matters
----------------
If any of these break, the repo stops "thinking like a radar engineer":
- Range scaling becomes untrustworthy
- Pd-vs-range plots can lie
- Trade-off sweeps become meaningless

Test philosophy (professional)
------------------------------
- Deterministic (no Monte Carlo)
- Fast (single model_based call per scenario)
- No file I/O, no plotting
- Monotonic assertions with tiny numerical tolerance

How to run
----------
pytest -q
"""

from __future__ import annotations

from typing import Any, Dict, List

from core.simulation.model_based import run_model_based_case


def _base_cfg(ranges_m: List[float]) -> Dict[str, Any]:
    """
    Minimal, deterministic baseline config for model_based monotonicity tests.

    Notes
    -----
    - Noise-limited (no clutter, no interference)
    - Deterministic point target (no Swerling) unless your engine explicitly adds it.
    - Fixed detection settings to make Pd comparisons meaningful.
    """
    return {
        "radar": {
            "fc_hz": 9.5e9,
            "tx_power_w": 2.0e3,
            "prf_hz": 2.0e3,
        },
        "antenna": {"gain_tx_db": 32.0, "gain_rx_db": 32.0},
        "receiver": {"bw_hz": 2.0e6, "nf_db": 4.5, "temperature_k": 290.0},
        "target": {"rcs_sqm": 1.0},
        "environment": {"system_losses_db": 6.0},
        "geometry": {
            "n_range_bins": 128,
            "n_doppler_bins": 64,
            "beams_per_scan": 8,
            "n_cpi_per_dwell": 1,
        },
        # Keep detector fixed for Pd monotonicity claims
        "detection": {
            "pfa": 1e-6,
            "integration": "noncoherent",
            "n_pulses": 16,
        },
        "metrics": {"ranges_m": ranges_m},
    }


def test_received_power_and_snr_decrease_with_range() -> None:
    """
    Pr must strictly decrease with range and SNR(dB) must strictly decrease with range.
    """
    ranges = [5_000.0, 10_000.0, 15_000.0, 20_000.0, 25_000.0, 30_000.0, 35_000.0, 40_000.0]
    cfg = _base_cfg(ranges)

    m = run_model_based_case(cfg, seed=None)

    pr = m.get("received_power_w", None)
    snr_db = m.get("snr_db", None)

    assert isinstance(pr, list) and len(pr) == len(ranges), "expected received_power_w list aligned to ranges"
    assert isinstance(snr_db, list) and len(snr_db) == len(ranges), "expected snr_db list aligned to ranges"

    pr_f = [float(x) for x in pr]
    snr_db_f = [float(x) for x in snr_db]

    # Strictly decreasing Pr with range
    for i in range(1, len(pr_f)):
        assert pr_f[i] < pr_f[i - 1], (
            f"received power must strictly decrease with range: "
            f"Pr[{i-1}]={pr_f[i-1]:.6g}, Pr[{i}]={pr_f[i]:.6g}"
        )

    # Strictly decreasing SNR(dB) with range
    for i in range(1, len(snr_db_f)):
        assert snr_db_f[i] < snr_db_f[i - 1], (
            f"SNR(dB) must strictly decrease with range: "
            f"SNR[{i-1}]={snr_db_f[i-1]:.6g}, SNR[{i}]={snr_db_f[i]:.6g}"
        )


def test_pd_is_nonincreasing_with_range_for_fixed_detector() -> None:
    """
    With fixed Pfa + fixed integration, Pd must be non-increasing with range
    in a noise-limited deterministic-target baseline.
    """
    ranges = [5_000.0, 10_000.0, 15_000.0, 20_000.0, 25_000.0, 30_000.0, 35_000.0, 40_000.0]
    cfg = _base_cfg(ranges)

    m = run_model_based_case(cfg, seed=None)

    det = m.get("detection", {}) or {}
    pd = det.get("pd", None)

    assert isinstance(pd, list) and len(pd) == len(ranges), "expected detection.pd list aligned to ranges"

    pd_f = [float(x) for x in pd]

    # Non-increasing with a tiny tolerance for numeric noise
    eps = 1e-12
    for i in range(1, len(pd_f)):
        assert pd_f[i] <= pd_f[i - 1] + eps, (
            f"Pd must be non-increasing with range: "
            f"Pd[{i-1}]={pd_f[i-1]:.6g}, Pd[{i}]={pd_f[i]:.6g}"
        )


def test_pd_does_not_decrease_when_tx_power_increases() -> None:
    """
    Increasing transmit power must not reduce Pd at any range (same detector/geometry).
    """
    ranges = [5_000.0, 10_000.0, 15_000.0, 20_000.0, 25_000.0, 30_000.0, 35_000.0, 40_000.0]

    cfg0 = _base_cfg(ranges)
    m0 = run_model_based_case(cfg0, seed=None)
    pd0 = (m0.get("detection", {}) or {}).get("pd", None)
    assert isinstance(pd0, list) and len(pd0) == len(ranges)
    pd0_f = [float(x) for x in pd0]

    # +6 dB TX power => x4 in linear
    cfg1 = _base_cfg(ranges)
    cfg1["radar"]["tx_power_w"] = float(cfg0["radar"]["tx_power_w"]) * 4.0
    m1 = run_model_based_case(cfg1, seed=None)
    pd1 = (m1.get("detection", {}) or {}).get("pd", None)
    assert isinstance(pd1, list) and len(pd1) == len(ranges)
    pd1_f = [float(x) for x in pd1]

    # Must not be worse anywhere
    eps = 1e-12
    for a, b in zip(pd0_f, pd1_f):
        assert b + eps >= a, f"Pd must not decrease when tx_power increases: base={a:.6g}, boosted={b:.6g}"

    # Should improve somewhere away from saturation; require at least one meaningful increase.
    # (Avoid fragile expectations: just require one delta > 1e-6.)
    improved = any((b - a) > 1e-6 for a, b in zip(pd0_f, pd1_f))
    assert improved, "Expected Pd to improve at least at one range when tx_power increases by +6 dB"