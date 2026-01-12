"""
core/interference/interferers.py

Interference source models and aggregation utilities.

Purpose
-------
Provide a small set of reusable, deterministic primitives for modeling external
interference at the radar receiver input. This module is intentionally general:
it can represent unintentional emitters (other radars), co-channel interferers,
and serves as the base for explicit jammer models in core/interference/jammers.py.

Scope (v1)
----------
Included:
- Interferer dataclass with RF parameters and geometry.
- Friis-based received interference power at the radar receiver input.
- Bandwidth-coupled interference power into a victim receiver bandwidth.
- Aggregation helpers for multiple interferers.

Not included (by design in v1):
- Detailed spectral masks / modulation.
- Time-varying scanning schedules for interferers.
- Polarization mismatch and detailed coupling models.

Inputs / Outputs
----------------
- Frequencies in Hz, powers in W, bandwidths in Hz, ranges in m.
- Outputs are powers in W and (optionally) ratios in dB.

Public API
----------
- Interferer
- received_interference_power_w(interferer, *, victim_fc_hz, victim_bw_hz, coupling_loss_db=0.0) -> float
- aggregate_interference_power_w(interferers, *, victim_fc_hz, victim_bw_hz, coupling_loss_db=0.0) -> float
- w_to_dbw(w) -> float
- dbw_to_w(dbw) -> float

Dependencies
------------
- Python standard library (dataclasses, math, typing)

Execution
---------
Not intended to be executed as a script.

Design notes
------------
- This is an *input-referred* model: powers are at the receiver input pre-detection.
- The goal is stable wiring + sensitivity analysis, not a full EW simulator.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable
import math

from core.environment.propagation import fspl_db, db_to_lin


@dataclass(frozen=True)
class Interferer:
    """
    External emitter / interferer.

    Fields
    ------
    name : str
        Human-readable identifier.
    fc_hz : float
        Interferer center frequency [Hz].
    tx_power_w : float
        Transmit power [W].
    tx_gain_db : float
        Transmit antenna gain toward victim [dB].
    rx_gain_db : float
        Optional victim-side gain toward interferer [dB]. Use 0 if unknown.
    bandwidth_hz : float
        Occupied bandwidth [Hz]. Used for bandwidth coupling into victim BW.
    range_m : float
        Separation distance [m] from interferer to victim receiver. Must be > 0.
    """
    name: str
    fc_hz: float
    tx_power_w: float
    tx_gain_db: float = 0.0
    rx_gain_db: float = 0.0
    bandwidth_hz: float = 1.0
    range_m: float = 1.0


def dbw_to_w(dbw: float) -> float:
    """Convert dBW to W."""
    _require_finite(dbw, name="dbw")
    return 10.0 ** (float(dbw) / 10.0)


def w_to_dbw(w: float) -> float:
    """Convert W to dBW. Requires w > 0."""
    _require_finite(w, name="w")
    if float(w) <= 0.0:
        raise ValueError(f"w must be > 0, got {w}")
    return 10.0 * math.log10(float(w))


def received_interference_power_w(
    interferer: Interferer,
    *,
    victim_fc_hz: float,
    victim_bw_hz: float,
    coupling_loss_db: float = 0.0,
) -> float:
    """
    Compute received interference power at victim receiver input (W).

    Model (v1)
    ----------
    1) Friis one-way free-space loss at victim carrier frequency:
       Pr = Pt * Gt * Gr / FSPL
    2) Bandwidth coupling:
       - Assume interferer power is flat across its occupied bandwidth.
       - Coupled power into victim bandwidth is scaled by overlap fraction:
           frac = min(1, victim_bw / interferer_bw)
       This is intentionally conservative and avoids spectral mask complexity.
    3) Optional lumped coupling_loss_db to represent additional isolation
       (filter rejection, polarization mismatch, side-lobe coupling, etc.)

    Parameters
    ----------
    interferer : Interferer
    victim_fc_hz : float
        Victim receiver center frequency [Hz].
    victim_bw_hz : float
        Victim receiver noise-equivalent bandwidth [Hz].
    coupling_loss_db : float
        Additional non-negative isolation loss [dB].

    Returns
    -------
    float
        Interference power at victim input [W], >= 0.
    """
    _require_positive(victim_fc_hz, name="victim_fc_hz")
    _require_positive(victim_bw_hz, name="victim_bw_hz")
    _require_nonnegative(coupling_loss_db, name="coupling_loss_db")

    _validate_interferer(interferer)

    # Use victim frequency for FSPL so the "victim" reference frame is consistent.
    loss_db = fspl_db(range_m=interferer.range_m, fc_hz=victim_fc_hz)

    gt = db_to_lin(float(interferer.tx_gain_db))
    gr = db_to_lin(float(interferer.rx_gain_db))
    iso = db_to_lin(float(coupling_loss_db))

    pr_total = float(interferer.tx_power_w) * gt * gr / (db_to_lin(loss_db) * iso)

    # Bandwidth coupling (flat PSD assumption)
    bw_i = float(interferer.bandwidth_hz)
    frac = min(1.0, float(victim_bw_hz) / bw_i) if bw_i > 0.0 else 0.0
    pr_coupled = pr_total * frac

    if not math.isfinite(pr_coupled) or pr_coupled < 0.0:
        raise ValueError("Computed interference power is invalid (non-finite or negative).")

    return pr_coupled


def aggregate_interference_power_w(
    interferers: Iterable[Interferer],
    *,
    victim_fc_hz: float,
    victim_bw_hz: float,
    coupling_loss_db: float = 0.0,
) -> float:
    """
    Sum interference powers from multiple interferers (W).
    """
    total = 0.0
    for itf in interferers:
        total += received_interference_power_w(
            itf,
            victim_fc_hz=victim_fc_hz,
            victim_bw_hz=victim_bw_hz,
            coupling_loss_db=coupling_loss_db,
        )
    return float(total)


# -----------------------------------------------------------------------------
# Validation helpers
# -----------------------------------------------------------------------------

def _validate_interferer(i: Interferer) -> None:
    if not isinstance(i.name, str) or not i.name:
        raise ValueError("Interferer.name must be a non-empty string")
    _require_positive(i.fc_hz, name=f"{i.name}.fc_hz")
    _require_positive(i.tx_power_w, name=f"{i.name}.tx_power_w")
    _require_finite(i.tx_gain_db, name=f"{i.name}.tx_gain_db")
    _require_finite(i.rx_gain_db, name=f"{i.name}.rx_gain_db")
    _require_positive(i.bandwidth_hz, name=f"{i.name}.bandwidth_hz")
    _require_positive(i.range_m, name=f"{i.name}.range_m")


def _require_finite(x: float, *, name: str) -> None:
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        raise ValueError(f"{name} must be numeric, got {type(x).__name__}")
    if not math.isfinite(float(x)):
        raise ValueError(f"{name} must be finite, got {x}")


def _require_positive(x: float, *, name: str) -> None:
    _require_finite(x, name=name)
    if float(x) <= 0.0:
        raise ValueError(f"{name} must be > 0, got {x}")


def _require_nonnegative(x: float, *, name: str) -> None:
    _require_finite(x, name=name)
    if float(x) < 0.0:
        raise ValueError(f"{name} must be >= 0, got {x}")