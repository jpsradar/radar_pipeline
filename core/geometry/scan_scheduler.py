"""
core/geometry/scan_scheduler.py

Deterministic scan scheduling primitives (v1).

Purpose
-------
Provide a minimal, explicit scan scheduling layer that can generate a reproducible
beam-pointing sequence and compute basic scan-time bookkeeping.

This module is intentionally small and "boring":
- deterministic
- no I/O
- no plotting
- explicit assumptions

Why it exists in v1
-------------------
Even a simplified pipeline needs a clean place to represent:
- what is a "scan" (a sequence of beams),
- how beams relate to dwell time,
- how to compute scan time and scan rate from a plan.

This complements:
- core.geometry.dwell  (dwell time concepts)
- core.geometry.counts (cells per CPI, scan geometry consistency tests)

Scope (v1)
----------
- Represent a scan as a list of BeamPointing objects:
    * azimuth angle [rad] (required)
    * elevation angle [rad] (optional, default 0)
    * dwell_time_s [s] (required, can be constant across plan)
- Provide constructors for common toy scans:
    * raster scan over azimuth (uniform grid)
    * single-beam "stare" scan
- Provide summary bookkeeping:
    * beams_per_scan
    * scan_time_s
    * scans_per_second

Non-goals (v1)
--------------
- No target-prioritized scheduling, adaptive revisit, or resource management.
- No antenna dynamics (slew limits) — can be layered later.
- No waveform coupling (PRI/PRF constraints) — handled elsewhere.

Public API (stable)
-------------------
Data models:
- BeamPointing
- ScanPlan

Constructors:
- make_stare_scan(dwell_time_s, az_rad=0.0, el_rad=0.0) -> ScanPlan
- make_raster_scan(az_min_rad, az_max_rad, n_beams, dwell_time_s, el_rad=0.0) -> ScanPlan

Bookkeeping:
- scan_time_s(plan) -> float
- scans_per_second(plan) -> float

Dependencies
------------
- Python stdlib: dataclasses, math

Usage
-----
    from core.geometry.scan_scheduler import make_raster_scan, scan_time_s

    plan = make_raster_scan(-0.5, 0.5, n_beams=10, dwell_time_s=0.2)
    t_scan = scan_time_s(plan)

Outputs
-------
This module returns in-memory dataclasses and floats (no file output).

Determinism / quality
---------------------
- Pure, deterministic constructors
- Defensive validation for finiteness and positivity
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence
import math


@dataclass(frozen=True)
class BeamPointing:
    """
    Single beam pointing command.

    Attributes
    ----------
    az_rad : float
        Azimuth pointing angle in radians.
    el_rad : float
        Elevation pointing angle in radians (default 0).
    dwell_time_s : float
        Time spent on this beam in seconds (must be > 0).
    """
    az_rad: float
    dwell_time_s: float
    el_rad: float = 0.0

    def __post_init__(self) -> None:
        _require_finite(self.az_rad, "az_rad")
        _require_finite(self.el_rad, "el_rad")
        _require_finite_positive(self.dwell_time_s, "dwell_time_s")


@dataclass(frozen=True)
class ScanPlan:
    """
    Scan plan: an ordered sequence of beam pointings.

    Attributes
    ----------
    beams : list[BeamPointing]
        Ordered list of beam commands. Must be non-empty.

    Notes
    -----
    This is the minimal v1 representation. More advanced scheduling can be layered
    by generating different beam lists.
    """
    beams: Sequence[BeamPointing]

    def __post_init__(self) -> None:
        if not isinstance(self.beams, Sequence) or len(self.beams) == 0:
            raise ValueError("ScanPlan.beams must be a non-empty sequence")
        for b in self.beams:
            if not isinstance(b, BeamPointing):
                raise TypeError("ScanPlan.beams must contain only BeamPointing objects")


def make_stare_scan(*, dwell_time_s: float, az_rad: float = 0.0, el_rad: float = 0.0) -> ScanPlan:
    """
    Create a single-beam "stare" scan.

    Parameters
    ----------
    dwell_time_s : float
        Dwell time in seconds (> 0).
    az_rad : float
        Azimuth angle in radians.
    el_rad : float
        Elevation angle in radians.

    Returns
    -------
    ScanPlan
        Plan containing exactly one BeamPointing.
    """
    beam = BeamPointing(az_rad=float(az_rad), el_rad=float(el_rad), dwell_time_s=float(dwell_time_s))
    return ScanPlan(beams=[beam])


def make_raster_scan(
    az_min_rad: float,
    az_max_rad: float,
    *,
    n_beams: int,
    dwell_time_s: float,
    el_rad: float = 0.0,
) -> ScanPlan:
    """
    Create a simple azimuth raster scan with uniformly spaced beam centers.

    Parameters
    ----------
    az_min_rad : float
        Minimum azimuth angle (radians).
    az_max_rad : float
        Maximum azimuth angle (radians).
    n_beams : int
        Number of beams in the scan (>= 1).
    dwell_time_s : float
        Dwell time per beam in seconds (> 0).
    el_rad : float
        Elevation angle for all beams (radians).

    Returns
    -------
    ScanPlan
        Plan with n_beams beam pointings.

    Notes
    -----
    If az_min_rad == az_max_rad, all beams point to the same azimuth (degenerate raster).
    """
    _require_finite(az_min_rad, "az_min_rad")
    _require_finite(az_max_rad, "az_max_rad")
    if not isinstance(n_beams, int) or n_beams < 1:
        raise ValueError(f"n_beams must be an integer >= 1, got {n_beams}")
    _require_finite_positive(dwell_time_s, "dwell_time_s")
    _require_finite(el_rad, "el_rad")

    if n_beams == 1:
        az_list = [0.5 * (float(az_min_rad) + float(az_max_rad))]
    else:
        step = (float(az_max_rad) - float(az_min_rad)) / float(n_beams - 1)
        az_list = [float(az_min_rad) + i * step for i in range(n_beams)]

    beams: List[BeamPointing] = [
        BeamPointing(az_rad=az, el_rad=float(el_rad), dwell_time_s=float(dwell_time_s))
        for az in az_list
    ]
    return ScanPlan(beams=beams)


def beams_per_scan(plan: ScanPlan) -> int:
    """Return number of beams in a scan plan."""
    return int(len(plan.beams))


def scan_time_s(plan: ScanPlan) -> float:
    """
    Compute total scan time.

    Returns
    -------
    float
        Sum of dwell times across all beams.
    """
    t = 0.0
    for b in plan.beams:
        t += float(b.dwell_time_s)
    if not math.isfinite(t) or t <= 0.0:
        raise ValueError("Computed scan_time_s must be finite and > 0")
    return t


def scans_per_second(plan: ScanPlan) -> float:
    """
    Compute scan rate (scans per second).

    Returns
    -------
    float
        1 / scan_time_s(plan).
    """
    t = scan_time_s(plan)
    return 1.0 / t


# -----------------------------------------------------------------------------
# Internal helpers
# -----------------------------------------------------------------------------

def _require_finite(x: float, name: str) -> None:
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        raise TypeError(f"{name} must be numeric, got {type(x).__name__}")
    if not math.isfinite(float(x)):
        raise ValueError(f"{name} must be finite, got {x}")


def _require_finite_positive(x: float, name: str) -> None:
    _require_finite(x, name)
    if float(x) <= 0.0:
        raise ValueError(f"{name} must be > 0, got {x}")