"""
core/geometry/dwell.py

Dwell, CPI, and scan timing geometry utilities.

What this module does
---------------------
Defines deterministic relationships between:
- PRF
- number of pulses
- CPI duration
- dwell time
- scan rate

This module is purely kinematic/timing-based:
- No signal processing
- No detection logic
- No statistics

It exists to ensure that all timing quantities used for:
- Pd computation
- FAR conversion
- update rate / latency analysis

are CONSISTENT and TRACEABLE.

Key principle
-------------
Every time-related quantity must be derivable from:
- PRF
- pulses per CPI
- beam dwell definition
- scan strategy
"""

from __future__ import annotations

from dataclasses import dataclass
import math


class GeometryError(ValueError):
    """Raised when geometry inputs are invalid."""


# ---------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class CPI:
    """
    Coherent Processing Interval (CPI) definition.

    Attributes
    ----------
    prf_hz : float
        Pulse repetition frequency [Hz].
    n_pulses : int
        Number of pulses in one CPI.
    duration_s : float
        CPI duration [s].
    """
    prf_hz: float
    n_pulses: int
    duration_s: float


@dataclass(frozen=True)
class Dwell:
    """
    Beam dwell definition.

    Attributes
    ----------
    cpi : CPI
        CPI definition.
    n_cpi : int
        Number of CPIs per dwell.
    dwell_time_s : float
        Total dwell time [s].
    """
    cpi: CPI
    n_cpi: int
    dwell_time_s: float


# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------

def make_cpi(*, prf_hz: float, n_pulses: int) -> CPI:
    """
    Construct a CPI from PRF and pulse count.

    Parameters
    ----------
    prf_hz : float
        Pulse repetition frequency [Hz].
    n_pulses : int
        Number of pulses per CPI.

    Returns
    -------
    CPI
        Fully specified CPI with duration.
    """
    if prf_hz <= 0.0:
        raise GeometryError("prf_hz must be > 0")
    if n_pulses < 1:
        raise GeometryError("n_pulses must be >= 1")

    duration = n_pulses / prf_hz
    return CPI(prf_hz=prf_hz, n_pulses=n_pulses, duration_s=duration)


def make_dwell(*, cpi: CPI, n_cpi: int) -> Dwell:
    """
    Construct a dwell definition from CPI repetition.

    Parameters
    ----------
    cpi : CPI
        CPI definition.
    n_cpi : int
        Number of CPIs per dwell.

    Returns
    -------
    Dwell
        Fully specified dwell timing.
    """
    if n_cpi < 1:
        raise GeometryError("n_cpi must be >= 1")

    dwell_time = n_cpi * cpi.duration_s
    return Dwell(cpi=cpi, n_cpi=n_cpi, dwell_time_s=dwell_time)


def cpis_per_second(cpi: CPI) -> float:
    """
    Compute number of CPIs processed per second.

    Returns
    -------
    float
        CPIs per second.
    """
    if cpi.duration_s <= 0.0:
        raise GeometryError("Invalid CPI duration")
    return 1.0 / cpi.duration_s