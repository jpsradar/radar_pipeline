"""
core/config/units.py

Unit handling and normalization utilities for the radar pipeline.

What this module does
---------------------
- Provides unit conversion helpers used across the pipeline.
- Centralizes dB <-> linear conversions (power and amplitude conventions).
- Supports BOTH scalar and NumPy array inputs for conversion functions.
- Implements a safe normalization hook:
      normalize_case_config(cfg: dict) -> dict

Design contract (IMPORTANT)
---------------------------
- This module MUST NEVER mutate the input dict in-place.
- This module MUST ALWAYS return a dict.
- This module MUST NOT add top-level sections that were not present in the input.
  (e.g., a Monte Carlo case must remain a Monte Carlo case, without injecting
  empty "radar"/"receiver" sections that would violate strict schemas.)

Normalization scope (v1)
------------------------
Performance cases:
- radar:
    - tx_power_dbm -> tx_power_w
    - fc_ghz -> fc_hz
- receiver:
    - nf_lin -> nf_db
    - bw_mhz -> bw_hz
- Adds a non-breaking "_normalization" report section (fields_converted, warnings)

Monte Carlo cases:
- No unit conversion required by default (values are already dimensionless / canonical).
- Still attaches "_normalization" report, but does not invent any sections.
"""

from __future__ import annotations

from typing import Any, Dict, List, Union
import math
import copy

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None  # type: ignore


# ---------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------

class UnitError(ValueError):
    """Raised when unit normalization or conversion fails."""


# ---------------------------------------------------------------------
# Physical constants / helpers
# ---------------------------------------------------------------------

def k_boltzmann() -> float:
    """Return Boltzmann constant [J/K]."""
    return 1.380649e-23


# ---------------------------------------------------------------------
# dB / linear conversions (scalar + NumPy array support)
# ---------------------------------------------------------------------

NumberOrArray = Union[float, "np.ndarray"]  # type: ignore[name-defined]


def db_to_lin_power(db: NumberOrArray) -> NumberOrArray:
    """
    Convert dB to linear POWER ratio.

    Notes
    -----
    - Use for power gains/losses, noise figure (power factor), etc.
    - For amplitude ratios, use db_to_lin_amplitude().
    - Supports scalar or NumPy array inputs.
    """
    if _is_ndarray(db):
        _require_finite_array(db, name="db")
        return 10.0 ** (db / 10.0)
    _require_finite_scalar(db, name="db")
    return 10.0 ** (float(db) / 10.0)


def lin_to_db_power(lin: NumberOrArray) -> NumberOrArray:
    """
    Convert linear POWER ratio to dB.

    Notes
    -----
    - Input must be > 0.
    - Supports scalar or NumPy array inputs.
    """
    if _is_ndarray(lin):
        _require_finite_array(lin, name="lin")
        if np is None:  # pragma: no cover
            raise UnitError("NumPy is required for array conversions")
        if np.any(lin <= 0.0):
            raise UnitError("lin_to_db_power expects all entries > 0 for array input")
        return 10.0 * np.log10(lin)
    _require_finite_scalar(lin, name="lin")
    if float(lin) <= 0.0:
        raise UnitError(f"lin_to_db_power expects lin > 0, got {lin}")
    return 10.0 * math.log10(float(lin))


def db_to_lin_amplitude(db: NumberOrArray) -> NumberOrArray:
    """
    Convert dB to linear AMPLITUDE ratio.

    Notes
    -----
    - Use for voltage/field amplitude ratios.
    - For power ratios, use db_to_lin_power().
    - Supports scalar or NumPy array inputs.
    """
    if _is_ndarray(db):
        _require_finite_array(db, name="db")
        return 10.0 ** (db / 20.0)
    _require_finite_scalar(db, name="db")
    return 10.0 ** (float(db) / 20.0)


def lin_to_db_amplitude(lin: NumberOrArray) -> NumberOrArray:
    """
    Convert linear AMPLITUDE ratio to dB.

    Notes
    -----
    - Input must be > 0.
    - Supports scalar or NumPy array inputs.
    """
    if _is_ndarray(lin):
        _require_finite_array(lin, name="lin")
        if np is None:  # pragma: no cover
            raise UnitError("NumPy is required for array conversions")
        if np.any(lin <= 0.0):
            raise UnitError("lin_to_db_amplitude expects all entries > 0 for array input")
        return 20.0 * np.log10(lin)
    _require_finite_scalar(lin, name="lin")
    if float(lin) <= 0.0:
        raise UnitError(f"lin_to_db_amplitude expects lin > 0, got {lin}")
    return 20.0 * math.log10(float(lin))


def dbm_to_w(dbm: float) -> float:
    """
    Convert dBm to Watts.

    Definition
    ----------
    0 dBm = 1 mW.
    """
    _require_finite_scalar(dbm, name="dbm")
    return 1e-3 * (10.0 ** (float(dbm) / 10.0))


def w_to_dbm(w: float) -> float:
    """
    Convert Watts to dBm.

    Notes
    -----
    - Input must be > 0.
    """
    _require_finite_scalar(w, name="w")
    if float(w) <= 0.0:
        raise UnitError(f"w_to_dbm expects w > 0, got {w}")
    return 10.0 * math.log10(float(w) / 1e-3)


# ---------------------------------------------------------------------
# Normalization hook (called by loaders)
# ---------------------------------------------------------------------

def normalize_case_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize a case configuration dict into canonical fields.

    IMPORTANT
    ---------
    - Must always return a dict.
    - Must not inject top-level sections that were not present.
    - Must be safe for both performance cases and Monte Carlo cases.
    """
    if not isinstance(cfg, dict):
        raise UnitError("normalize_case_config expects cfg to be a dict")

    out: Dict[str, Any] = _deep_copy_dict(cfg)

    converted: List[str] = []
    warnings: List[str] = []

    # Detect case type by presence of top-level keys.
    has_mc = isinstance(out.get("monte_carlo", None), dict)
    has_perf = any(isinstance(out.get(k, None), dict) for k in ("radar", "antenna", "receiver", "target"))

    # --- Performance-case normalization (only if those sections exist) ---
    if has_perf:
        if isinstance(out.get("radar", None), dict):
            radar = out["radar"]

            # TX power normalization
            if "tx_power_w" not in radar and "tx_power_dbm" in radar:
                radar["tx_power_w"] = dbm_to_w(float(radar["tx_power_dbm"]))
                converted.append("radar.tx_power_dbm -> radar.tx_power_w")

            # Carrier frequency normalization
            if "fc_hz" not in radar and "fc_ghz" in radar:
                radar["fc_hz"] = float(radar["fc_ghz"]) * 1e9
                converted.append("radar.fc_ghz -> radar.fc_hz")

            _sanity_check_positive_optional(radar, "fc_hz", warnings)
            _sanity_check_positive_optional(radar, "tx_power_w", warnings)

        if isinstance(out.get("receiver", None), dict):
            receiver = out["receiver"]

            # Noise figure normalization (linear -> dB power ratio)
            if "nf_db" not in receiver and "nf_lin" in receiver:
                receiver["nf_db"] = float(lin_to_db_power(float(receiver["nf_lin"])))  # scalar by contract
                converted.append("receiver.nf_lin -> receiver.nf_db")

            # Bandwidth normalization
            if "bw_hz" not in receiver and "bw_mhz" in receiver:
                receiver["bw_hz"] = float(receiver["bw_mhz"]) * 1e6
                converted.append("receiver.bw_mhz -> receiver.bw_hz")

            _sanity_check_positive_optional(receiver, "bw_hz", warnings)
            _sanity_check_nonnegative_optional(receiver, "nf_db", warnings)

        # Temperature sanity (if present)
        if isinstance(out.get("receiver", None), dict) and "temperature_k" in out["receiver"]:
            _sanity_check_positive_optional(out["receiver"], "temperature_k", warnings)

    # --- Monte Carlo case: do not invent/modify top-level structure ---
    if has_mc and not has_perf:
        mc = out["monte_carlo"]
        bg = mc.get("background", None)
        if isinstance(bg, dict) and "mean_power" in bg:
            try:
                mp = float(bg["mean_power"])
                if not math.isfinite(mp) or mp <= 0.0:
                    warnings.append("monte_carlo.background.mean_power is non-positive or non-finite")
            except Exception:
                warnings.append("monte_carlo.background.mean_power is not numeric")

    # Attach a non-breaking normalization report for traceability.
    out.setdefault("_normalization", {})
    if isinstance(out["_normalization"], dict):
        out["_normalization"]["fields_converted"] = converted
        out["_normalization"]["warnings"] = warnings

    return out


# ---------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------

def _deep_copy_dict(d: Dict[str, Any]) -> Dict[str, Any]:
    """Deep copy a dict safely (inputs are small; correctness over micro-optimizations)."""
    return copy.deepcopy(d)


def _is_ndarray(x: Any) -> bool:
    """Return True if x is a NumPy ndarray (without hard-failing if NumPy missing)."""
    if np is None:
        return False
    return isinstance(x, np.ndarray)


def _require_finite_scalar(x: Any, *, name: str) -> None:
    """Ensure x is a finite scalar number."""
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        raise UnitError(f"{name} must be numeric, got {type(x).__name__}")
    if not math.isfinite(float(x)):
        raise UnitError(f"{name} must be finite, got {x}")


def _require_finite_array(x: Any, *, name: str) -> None:
    """Ensure x is a finite NumPy array."""
    if np is None:  # pragma: no cover
        raise UnitError(f"{name} array input requires NumPy")
    if not isinstance(x, np.ndarray):
        raise UnitError(f"{name} must be a NumPy ndarray, got {type(x).__name__}")
    if not np.all(np.isfinite(x)):
        raise UnitError(f"{name} must contain only finite values")


def _sanity_check_positive_optional(section: Dict[str, Any], key: str, warnings: List[str]) -> None:
    """Warn (non-fatal) if a numeric field exists but is not positive."""
    if key not in section:
        return
    try:
        v = float(section[key])
        if not math.isfinite(v) or v <= 0.0:
            warnings.append(f"{key} is not positive/finite (value={section[key]})")
    except Exception:
        warnings.append(f"{key} is not numeric (value={section[key]})")


def _sanity_check_nonnegative_optional(section: Dict[str, Any], key: str, warnings: List[str]) -> None:
    """Warn (non-fatal) if a numeric field exists but is negative."""
    if key not in section:
        return
    try:
        v = float(section[key])
        if not math.isfinite(v) or v < 0.0:
            warnings.append(f"{key} is negative/non-finite (value={section[key]})")
    except Exception:
        warnings.append(f"{key} is not numeric (value={section[key]})")