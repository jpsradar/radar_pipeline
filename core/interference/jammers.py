"""
core/interference/jammers.py

Explicit jammer models built on top of the interferer primitives.

Purpose
-------
Represent common jammer configurations (spot, barrage/noise) and provide
simple, deterministic calculations for:
- jammer power coupled into the radar receiver bandwidth,
- J/N and J/S ratios at receiver input (pre-detection),
- basic burn-through range estimates under a monostatic radar equation.

This module is intentionally scoped for v1 pipeline credibility:
- It is explicit about assumptions and avoids hidden "magic" parameters.
- It keeps interfaces stable so higher-fidelity models can replace internals later.

Scope (v1)
----------
Included:
- Jammer dataclass (extends Interferer semantics with jammer mode fields)
- Jamming power at receiver input (W)
- J/N in linear and dB
- Optional burn-through range calculation (solve for range where target Pr equals jammer J)

Not included (by design in v1):
- Time/frequency agility and reactive techniques.
- Smart jammer waveform effects on CFAR / matched filter sidelobes.
- ECCM logic.

Inputs / Outputs
----------------
All quantities are SI units unless otherwise stated.

Public API
----------
- Jammer
- jammer_power_at_receiver_w(jammer, *, victim_fc_hz, victim_bw_hz, coupling_loss_db=0.0) -> float
- j_to_n_lin(j_w, noise_w) -> float
- j_to_n_db(j_w, noise_w) -> float
- burnthrough_range_m_monostatic(...)-> float  (numeric solve, deterministic)

Dependencies
------------
- Python standard library (dataclasses, math, typing)
- core/interference/interferers.py
- core/budgets/radar_equation.py (if available) is NOT required; we keep a local scalar equation.

Execution
---------
Not intended to be executed as a script.

Design notes
------------
- Receiver-referred powers are the currency in this pipeline layer.
- Burn-through uses a scalar monostatic radar equation consistent with the simulation engine.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

from core.environment.propagation import db_to_lin
from core.interference.interferers import Interferer, received_interference_power_w


@dataclass(frozen=True)
class Jammer:
    """
    Jammer emitter definition.

    Fields
    ------
    name : str
        Identifier.
    fc_hz : float
        Jammer center frequency [Hz].
    tx_power_w : float
        Total jammer transmit power [W] (across occupied_bw_hz).
    tx_gain_db : float
        Jammer gain toward victim [dB].
    rx_gain_db : float
        Victim-side gain toward jammer [dB].
    occupied_bw_hz : float
        Jammer occupied bandwidth [Hz]. For a noise/barrage jammer, this is the spread.
        For a spot jammer, this may be narrow.
    range_m : float
        Jammer-to-victim range [m].
    mode : str
        One of:
        - "spot": jammer energy concentrated in a narrow bandwidth (occupied_bw_hz).
        - "barrage": jammer energy spread; coupling into victim is bandwidth-limited.
    """
    name: str
    fc_hz: float
    tx_power_w: float
    tx_gain_db: float = 0.0
    rx_gain_db: float = 0.0
    occupied_bw_hz: float = 1.0
    range_m: float = 1.0
    mode: str = "barrage"


def jammer_power_at_receiver_w(
    jammer: Jammer,
    *,
    victim_fc_hz: float,
    victim_bw_hz: float,
    coupling_loss_db: float = 0.0,
) -> float:
    """
    Compute jammer power coupled into the victim receiver bandwidth (W).

    Implementation
    --------------
    We map Jammer -> Interferer and reuse the shared received_interference_power_w() logic.

    Notes on mode
    -------------
    - "spot": treated the same as "barrage" in v1 except via occupied_bw_hz;
      if occupied_bw_hz << victim_bw_hz, coupling fraction saturates at 1.
    - "barrage": if occupied_bw_hz >> victim_bw_hz, coupling fraction is victim_bw / occupied_bw.

    This is a clean v1 abstraction: later you can implement true spectral overlap.
    """
    i = Interferer(
        name=jammer.name,
        fc_hz=float(jammer.fc_hz),
        tx_power_w=float(jammer.tx_power_w),
        tx_gain_db=float(jammer.tx_gain_db),
        rx_gain_db=float(jammer.rx_gain_db),
        bandwidth_hz=float(jammer.occupied_bw_hz),
        range_m=float(jammer.range_m),
    )
    return received_interference_power_w(
        i,
        victim_fc_hz=victim_fc_hz,
        victim_bw_hz=victim_bw_hz,
        coupling_loss_db=coupling_loss_db,
    )


def j_to_n_lin(j_w: float, noise_w: float) -> float:
    """Return J/N (linear)."""
    _require_finite_positive(noise_w, name="noise_w")
    _require_finite_nonnegative(j_w, name="j_w")
    return float(j_w) / float(noise_w)


def j_to_n_db(j_w: float, noise_w: float) -> float:
    """Return J/N (dB)."""
    jn = j_to_n_lin(j_w, noise_w)
    return 10.0 * math.log10(max(jn, float("1e-300")))


def burnthrough_range_m_monostatic(
    *,
    fc_hz: float,
    tx_power_w: float,
    gain_tx_db: float,
    gain_rx_db: float,
    rcs_sqm: float,
    system_losses_db: float,
    jammer_power_at_rx_w: float,
    min_range_m: float = 10.0,
    max_range_m: float = 500_000.0,
) -> float:
    """
    Estimate burn-through range for a monostatic radar vs jammer, using a scalar equation.

    Definition (v1)
    ---------------
    Burn-through is the range R where target received power Pr(R) equals jammer power J
    at the receiver input (pre-detection). This is a crude but useful benchmark.

    Pr(R) = Pt * Gt * Gr * (lambda^2 * sigma) / ((4*pi)^3 * R^4 * L)

    Solve Pr(R) = J for R.

    Parameters
    ----------
    fc_hz, tx_power_w, gain_tx_db, gain_rx_db, rcs_sqm, system_losses_db
        Radar parameters (SI units, dB where noted).
    jammer_power_at_rx_w : float
        Jammer power already coupled to receiver input (W). Must be > 0.
    min_range_m, max_range_m : float
        Clamp range domain for numeric robustness.

    Returns
    -------
    float
        Burn-through range [m].

    Notes
    -----
    - This ignores processing gain, waveform effects, and CFAR behavior.
    - It is deterministic and suitable for pipeline-level trade studies.
    """
    _require_finite_positive(fc_hz, name="fc_hz")
    _require_finite_positive(tx_power_w, name="tx_power_w")
    _require_finite_positive(rcs_sqm, name="rcs_sqm")
    _require_finite_nonnegative(system_losses_db, name="system_losses_db")
    _require_finite_positive(jammer_power_at_rx_w, name="jammer_power_at_rx_w")
    _require_finite_positive(min_range_m, name="min_range_m")
    _require_finite_positive(max_range_m, name="max_range_m")
    if max_range_m <= min_range_m:
        raise ValueError("max_range_m must be > min_range_m")

    lam = 299_792_458.0 / float(fc_hz)
    gt = db_to_lin(float(gain_tx_db))
    gr = db_to_lin(float(gain_rx_db))
    L = db_to_lin(float(system_losses_db))

    # Solve R = (K / J)^(1/4)
    K = float(tx_power_w) * gt * gr * (lam ** 2) * float(rcs_sqm) / (((4.0 * math.pi) ** 3) * L)
    if not math.isfinite(K) or K <= 0.0:
        raise ValueError("Derived radar constant K is invalid (non-finite or non-positive).")

    R = (K / float(jammer_power_at_rx_w)) ** 0.25
    if not math.isfinite(R):
        raise ValueError("Burn-through solution is non-finite.")

    return float(min(max(R, float(min_range_m)), float(max_range_m)))


# -----------------------------------------------------------------------------
# Validation helpers
# -----------------------------------------------------------------------------

def _require_finite_positive(x: float, *, name: str) -> None:
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        raise ValueError(f"{name} must be numeric, got {type(x).__name__}")
    xf = float(x)
    if not math.isfinite(xf) or xf <= 0.0:
        raise ValueError(f"{name} must be finite and > 0, got {x}")


def _require_finite_nonnegative(x: float, *, name: str) -> None:
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        raise ValueError(f"{name} must be numeric, got {type(x).__name__}")
    xf = float(x)
    if not math.isfinite(xf) or xf < 0.0:
        raise ValueError(f"{name} must be finite and >= 0, got {x}")