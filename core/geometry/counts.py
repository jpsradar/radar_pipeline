"""
core/geometry/counts.py

Detection cell, beam, and scan counting utilities.

What this module does
---------------------
Provides explicit, auditable counts and timing for:
- Range-Doppler (RD) grid size => independent detection trials per CPI
- Scan geometry (beams per scan, dwell time) => scans/second
- These quantities are REQUIRED to:
    * convert Pfa -> FAR
    * size downstream processing (extractor / tracker)
    * perform trade-offs between resolution, dwell, and update rate

Design contract (important)
---------------------------
- No hidden assumptions: all counts and rates are computed from explicit inputs.
- Outputs are immutable dataclasses to make traceability and debugging easier.
- This module is purely geometric/counting-based:
    * No signal processing
    * No statistics

Typical usage
-------------
    from core.geometry.counts import make_rd_grid, make_scan_geometry

    grid = make_rd_grid(n_range_bins=256, n_doppler_bins=64)
    geom = make_scan_geometry(beams_per_scan=10, dwell_time_s=0.2)
"""

from __future__ import annotations

from dataclasses import dataclass
import math


# ---------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------

class CountError(ValueError):
    """Raised when a counting/geometry configuration is invalid."""


# ---------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class RDGrid:
    """
    Range-Doppler grid definition.

    Attributes
    ----------
    n_range_bins : int
        Number of independent range bins.
    n_doppler_bins : int
        Number of independent Doppler bins.
    cells_per_cpi : int
        Number of independent RD cells evaluated per CPI:
            cells_per_cpi = n_range_bins * n_doppler_bins
    """
    n_range_bins: int
    n_doppler_bins: int
    cells_per_cpi: int


@dataclass(frozen=True)
class ScanGeometry:
    """
    Scan geometry definition.

    Attributes
    ----------
    beams_per_scan : int
        Number of beams (pointings) in a full scan.
    dwell_time_s : float
        Dwell time per beam [s].
    scan_time_s : float
        Scan time [s], computed as:
            scan_time_s = beams_per_scan * dwell_time_s
    scans_per_second : float
        Scan repetition rate [Hz], computed as:
            scans_per_second = 1 / scan_time_s
    """
    beams_per_scan: int
    dwell_time_s: float
    scan_time_s: float
    scans_per_second: float


# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------

def make_rd_grid(*, n_range_bins: int, n_doppler_bins: int) -> RDGrid:
    """
    Construct RD grid counts.

    Parameters
    ----------
    n_range_bins : int
        Number of independent range bins.
    n_doppler_bins : int
        Number of independent Doppler bins.

    Returns
    -------
    RDGrid
        RD grid container with cells_per_cpi populated.
    """
    try:
        n_r = int(n_range_bins)
        n_d = int(n_doppler_bins)
    except Exception as exc:
        raise CountError("n_range_bins and n_doppler_bins must be integers") from exc

    if n_r < 1:
        raise CountError("n_range_bins must be >= 1")
    if n_d < 1:
        raise CountError("n_doppler_bins must be >= 1")

    cells = n_r * n_d
    return RDGrid(n_range_bins=n_r, n_doppler_bins=n_d, cells_per_cpi=cells)


def make_scan_geometry(*, beams_per_scan: int, dwell_time_s: float) -> ScanGeometry:
    """
    Construct scan geometry and derived timing quantities.

    Parameters
    ----------
    beams_per_scan : int
        Number of beams in a full scan.
    dwell_time_s : float
        Dwell time per beam [s]. Must be > 0.

    Returns
    -------
    ScanGeometry
        Fully specified scan geometry.
    """
    try:
        beams = int(beams_per_scan)
    except Exception as exc:
        raise CountError("beams_per_scan must be an integer") from exc

    try:
        dwell = float(dwell_time_s)
    except Exception as exc:
        raise CountError("dwell_time_s must be numeric") from exc

    if beams < 1:
        raise CountError("beams_per_scan must be >= 1")
    if not math.isfinite(dwell) or dwell <= 0.0:
        raise CountError("dwell_time_s must be finite and > 0")

    scan_time = float(beams) * dwell
    scans_per_second = 1.0 / scan_time

    return ScanGeometry(
        beams_per_scan=beams,
        dwell_time_s=dwell,
        scan_time_s=scan_time,
        scans_per_second=scans_per_second,
    )