"""
core/detection/far_conversion.py

False Alarm Rate (FAR) conversion utilities for the radar pipeline.

What this module does
---------------------
Provides explicit, auditable conversion from:
- Probability of false alarm (Pfa) defined per detection trial
to:
- System-level false alarm rates (per second, per scan)

This module intentionally exposes BOTH APIs:
1) New API (simple, geometry-driven):
   - pfa_to_far(pfa, rd_grid, scan_geom, return_breakdown)

2) Legacy/compat API (used by existing modules/tests in this repo snapshot):
   - FARInputs dataclass
   - convert_pfa_to_far(inputs) -> FARResult

Design contract
---------------
- No hidden assumptions.
- All rates are derived from explicit counts and timing.
- Domain definition is explicit in FARInputs.domain.

Domains (legacy API)
--------------------
- "rd_cell": Pfa is per RD cell per CPI, then:
    FAR = Pfa * cells_per_cpi * cpis_per_second * beams_per_scan
- "beam": Pfa is per beam-decision, then:
    FAR = Pfa * beams_per_second
- "scan": Pfa is per scan-decision, then:
    FAR = Pfa * scans_per_second

NOTE
----
The new API intentionally models:
    FAR [Hz] = Pfa * cells_per_cpi * scans_per_second
because its inputs are RDGrid + ScanGeometry.
If you need CPI-level detail (cpis_per_second, n_cpi_per_dwell), use the legacy API.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Literal, Union
import math


# ---------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------

class FARConversionError(ValueError):
    """Raised when FAR conversion inputs are invalid."""


# ---------------------------------------------------------------------
# New API containers
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class FARBreakdown:
    """
    Detailed breakdown for the new API.

    Attributes
    ----------
    pfa : float
        Probability of false alarm per detection cell.
    cells_per_cpi : int
        Number of independent detection cells evaluated per CPI.
    scans_per_second : float
        Number of scans per second.
    far_hz : float
        Resulting false alarm rate [Hz].
    """
    pfa: float
    cells_per_cpi: int
    scans_per_second: float
    far_hz: float


def pfa_to_far(
    *,
    pfa: float,
    rd_grid: Any,
    scan_geom: Any,
    return_breakdown: bool = False,
) -> Union[float, FARBreakdown]:
    """
    Convert cell-level Pfa to system-level FAR using RDGrid + ScanGeometry.

    Formula
    -------
    FAR [Hz] = Pfa * cells_per_cpi * scans_per_second

    Parameters
    ----------
    pfa : float
        Probability of false alarm per RD cell (0 < pfa < 1).
    rd_grid : RDGrid-like
        Must provide .cells_per_cpi.
    scan_geom : ScanGeometry-like
        Must provide .scans_per_second.
    return_breakdown : bool
        If True, returns FARBreakdown.

    Returns
    -------
    float | FARBreakdown
    """
    _validate_pfa(pfa)
    cells = _require_int_attr(rd_grid, "cells_per_cpi", min_value=1)
    scans_per_sec = _require_float_attr(scan_geom, "scans_per_second", min_value=0.0, strictly_positive=True)

    far = float(pfa) * float(cells) * float(scans_per_sec)

    if return_breakdown:
        return FARBreakdown(
            pfa=float(pfa),
            cells_per_cpi=int(cells),
            scans_per_second=float(scans_per_sec),
            far_hz=float(far),
        )
    return float(far)


# ---------------------------------------------------------------------
# Legacy/compat API (kept stable for existing imports)
# ---------------------------------------------------------------------

FARDomain = Literal["rd_cell", "beam", "scan"]


@dataclass(frozen=True)
class FARInputs:
    """
    Legacy FAR input contract (used by existing modules/tests).

    Attributes
    ----------
    pfa : float
        Probability of false alarm (interpretation depends on domain).
    domain : {"rd_cell", "beam", "scan"}
        Defines what "trial" means.
    cells_per_cpi : int
        Number of RD cells per CPI (required for domain="rd_cell").
    cpis_per_second : float
        CPI rate per second (required for domain="rd_cell").
    beams_per_scan : int
        Beams per scan (required for domain="rd_cell" and "beam" when beams_per_second is derived).
    scans_per_second : float
        Scan rate (required for domain="scan"; also used to derive beams_per_second).
    """
    pfa: float
    domain: FARDomain

    cells_per_cpi: int
    cpis_per_second: float
    beams_per_scan: int
    scans_per_second: float


@dataclass(frozen=True)
class FARResult:
    """
    Legacy FAR output (used by existing modules/tests).

    Attributes
    ----------
    far_per_second : float
        FAR in events/sec.
    far_per_scan : float
        FAR in events/scan.
    breakdown : dict
        Traceability dictionary (inputs + intermediate rates).
    """
    far_per_second: float
    far_per_scan: float
    breakdown: Dict[str, Any]


def convert_pfa_to_far(inputs: FARInputs) -> FARResult:
    """
    Convert Pfa to FAR using the legacy contract.

    Domain behavior
    ---------------
    - rd_cell:
        FAR/sec = Pfa * cells_per_cpi * cpis_per_second * beams_per_scan
      (explicitly includes CPI rate and beams/scan as separate multipliers)

    - beam:
        beams/sec = beams_per_scan * scans_per_second
        FAR/sec = Pfa * beams/sec

    - scan:
        FAR/sec = Pfa * scans_per_second

    Notes
    -----
    - This is deterministic and purely multiplicative.
    - It is the correct home for "counting" logic that must be auditable.
    """
    _validate_pfa(inputs.pfa)

    dom = str(inputs.domain).strip().lower()
    if dom not in {"rd_cell", "beam", "scan"}:
        raise FARConversionError(f"Unsupported domain: {inputs.domain}")

    beams_per_scan = _require_int(inputs.beams_per_scan, "beams_per_scan", min_value=1)
    scans_per_second = _require_float(inputs.scans_per_second, "scans_per_second", min_value=0.0, strictly_positive=True)
    beams_per_second = float(beams_per_scan) * float(scans_per_second)

    if dom == "scan":
        far_per_second = float(inputs.pfa) * float(scans_per_second)

    elif dom == "beam":
        far_per_second = float(inputs.pfa) * float(beams_per_second)

    else:  # "rd_cell"
        cells_per_cpi = _require_int(inputs.cells_per_cpi, "cells_per_cpi", min_value=1)
        cpis_per_second = _require_float(inputs.cpis_per_second, "cpis_per_second", min_value=0.0, strictly_positive=True)

        far_per_second = (
            float(inputs.pfa)
            * float(cells_per_cpi)
            * float(cpis_per_second)
            * float(beams_per_scan)
        )

    far_per_scan = float(far_per_second) / float(scans_per_second)

    breakdown = {
        "pfa": float(inputs.pfa),
        "domain": dom,
        "cells_per_cpi": int(inputs.cells_per_cpi),
        "cpis_per_second": float(inputs.cpis_per_second),
        "beams_per_scan": int(beams_per_scan),
        "scans_per_second": float(scans_per_second),
        "beams_per_second": float(beams_per_second),
    }

    return FARResult(
        far_per_second=float(far_per_second),
        far_per_scan=float(far_per_scan),
        breakdown=breakdown,
    )


# ---------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------

def _validate_pfa(pfa: float) -> None:
    try:
        p = float(pfa)
    except Exception as exc:
        raise FARConversionError(f"pfa must be numeric, got {pfa}") from exc
    if not math.isfinite(p) or not (0.0 < p < 1.0):
        raise FARConversionError(f"pfa must be in (0,1), got {pfa}")


def _require_int(x: Any, name: str, *, min_value: int) -> int:
    try:
        v = int(x)
    except Exception as exc:
        raise FARConversionError(f"{name} must be integer-like, got {x}") from exc
    if v < min_value:
        raise FARConversionError(f"{name} must be >= {min_value}, got {v}")
    return v


def _require_float(x: Any, name: str, *, min_value: float, strictly_positive: bool) -> float:
    try:
        v = float(x)
    except Exception as exc:
        raise FARConversionError(f"{name} must be numeric, got {x}") from exc
    if not math.isfinite(v):
        raise FARConversionError(f"{name} must be finite, got {x}")
    if strictly_positive and v <= min_value:
        raise FARConversionError(f"{name} must be > {min_value}, got {v}")
    if (not strictly_positive) and v < min_value:
        raise FARConversionError(f"{name} must be >= {min_value}, got {v}")
    return v


def _require_int_attr(obj: Any, attr: str, *, min_value: int) -> int:
    if not hasattr(obj, attr):
        raise FARConversionError(f"Object must provide attribute '{attr}'")
    return _require_int(getattr(obj, attr), attr, min_value=min_value)


def _require_float_attr(obj: Any, attr: str, *, min_value: float, strictly_positive: bool) -> float:
    if not hasattr(obj, attr):
        raise FARConversionError(f"Object must provide attribute '{attr}'")
    return _require_float(getattr(obj, attr), attr, min_value=min_value, strictly_positive=strictly_positive)