"""
tests/test_counts.py

Count / geometry sanity tests for system-level rates (cells, CPIs, scans, FAR).

What this test module covers
----------------------------
1) RD grid counts:
   - cells_per_cpi must equal n_range_bins * n_doppler_bins

2) Scan geometry consistency:
   - scan_time_s must equal beams_per_scan * dwell_time_s
   - scans_per_second must equal 1 / scan_time_s

3) FAR conversion consistency:
   - FAR must scale linearly with:
       * Pfa
       * cells_per_cpi
       * scans_per_second
   - FAR per scan must be consistent with FAR per second and scans_per_second:
       far_per_scan == far_per_second / scans_per_second

Why this matters
----------------
If these break, you cannot claim "real FAR" or system-level load estimates.
Everything downstream (reports, tracker false tracks, extraction load) becomes untrustworthy.

Test philosophy (professional)
------------------------------
- Deterministic
- Fast (no Monte Carlo)
- No I/O, no plotting
- Prefer ratio/scale checks over fragile absolute expectations

How to run
----------
pytest -q
"""

from __future__ import annotations

import math

from core.geometry.counts import make_rd_grid, make_scan_geometry
from core.detection.far_conversion import pfa_to_far, FARBreakdown


def test_rd_grid_cells_per_cpi_is_product() -> None:
    """
    cells_per_cpi must be the Cartesian product of range and Doppler bins.
    """
    n_r = 256
    n_d = 64
    grid = make_rd_grid(n_range_bins=n_r, n_doppler_bins=n_d)

    assert grid.n_range_bins == n_r
    assert grid.n_doppler_bins == n_d
    assert grid.cells_per_cpi == n_r * n_d, "cells_per_cpi must equal n_range_bins * n_doppler_bins"


def test_scan_geometry_time_and_rate_are_consistent() -> None:
    """
    Scan time and scan rate must be internally consistent.
    """
    beams = 10
    dwell_s = 0.2

    geom = make_scan_geometry(beams_per_scan=beams, dwell_time_s=dwell_s)

    assert geom.beams_per_scan == beams
    assert math.isfinite(geom.dwell_time_s) and geom.dwell_time_s > 0.0
    assert math.isfinite(geom.scan_time_s) and geom.scan_time_s > 0.0
    assert math.isfinite(geom.scans_per_second) and geom.scans_per_second > 0.0

    assert abs(geom.scan_time_s - beams * dwell_s) < 1e-12, "scan_time_s must equal beams_per_scan * dwell_time_s"
    assert abs(geom.scans_per_second - (1.0 / geom.scan_time_s)) < 1e-12, "scans_per_second must equal 1/scan_time_s"


def test_far_scales_linearly_and_is_scan_consistent() -> None:
    """
    FAR conversion must:
    - scale linearly with pfa, cells_per_cpi, scans_per_second
    - be consistent between per-second and per-scan rates
    """
    # Base geometry
    grid0 = make_rd_grid(n_range_bins=256, n_doppler_bins=64)

    # 10 beams, 0.2 s dwell => scan_time=2.0 s => scans_per_second=0.5
    geom0 = make_scan_geometry(beams_per_scan=10, dwell_time_s=0.2)

    pfa0 = 1e-3
    out0 = pfa_to_far(pfa=pfa0, rd_grid=grid0, scan_geom=geom0, return_breakdown=True)
    assert isinstance(out0, FARBreakdown)

    far0 = out0.far_hz
    assert math.isfinite(far0) and far0 >= 0.0, "FAR must be finite and non-negative"

    # Consistency identity:
    # far_per_scan == far_per_second / scans_per_second
    # (If scans_per_second is the scan rate used in FAR computation, this should match tightly.)
    assert geom0.scans_per_second > 0.0
    far_per_scan_0 = far0 / geom0.scans_per_second

    # --- Scaling checks (ratio-based, robust) ---

    # Scale Pfa by 2x -> FAR must 2x
    out_pfa = pfa_to_far(pfa=pfa0 * 2.0, rd_grid=grid0, scan_geom=geom0, return_breakdown=True)
    assert isinstance(out_pfa, FARBreakdown)
    _assert_ratio_close(out_pfa.far_hz / far0, 2.0, "FAR must scale linearly with pfa")

    # Scale cells_per_cpi by 2x -> FAR must 2x
    grid_cells = make_rd_grid(n_range_bins=512, n_doppler_bins=64)  # doubles n_range_bins => doubles cells
    out_cells = pfa_to_far(pfa=pfa0, rd_grid=grid_cells, scan_geom=geom0, return_breakdown=True)
    assert isinstance(out_cells, FARBreakdown)
    _assert_ratio_close(out_cells.far_hz / far0, 2.0, "FAR must scale linearly with cells_per_cpi")

    # Scale scans_per_second by 2x -> FAR must 2x
    # Keep beams fixed; halve dwell -> half scan_time -> double scans_per_second.
    geom_fast = make_scan_geometry(beams_per_scan=10, dwell_time_s=0.1)
    out_scans = pfa_to_far(pfa=pfa0, rd_grid=grid0, scan_geom=geom_fast, return_breakdown=True)
    assert isinstance(out_scans, FARBreakdown)
    _assert_ratio_close(out_scans.far_hz / far0, 2.0, "FAR must scale linearly with scans_per_second")

    # Per-scan consistency should remain true under the modified geometry
    far_per_scan_fast = out_scans.far_hz / geom_fast.scans_per_second
    _assert_ratio_close(
        far_per_scan_fast / far_per_scan_0,
        1.0,
        "FAR per scan must remain invariant when only scan rate changes",
    )


def _assert_ratio_close(ratio: float, expected: float, msg: str) -> None:
    """
    Helper for scale checks.

    Uses relative error with a tight tolerance because these are pure multiplicative relationships.
    """
    assert math.isfinite(ratio), f"{msg}: ratio must be finite"
    rel_err = abs(ratio - expected) / max(abs(expected), 1e-30)
    assert rel_err < 1e-12, f"{msg}: expected ratio={expected}, got {ratio} (rel_err={rel_err:.3e})"