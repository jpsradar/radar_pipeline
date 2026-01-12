"""
core/targets/kinematics.py

Target kinematics primitives for the radar pipeline (v1).

Purpose
-------
Provide deterministic, reusable target motion primitives that can be consumed by:
- simulation engines (model_based / signal_level / future DSP),
- scenario generation tools (sweeps, Monte Carlo cases),
- reporting/metrics modules.

This module defines a minimal but serious kinematics layer for v1:
- 2D/3D position/velocity state representation
- Constant-velocity (CV) propagation
- Relative geometry helpers (range, range-rate, azimuth/elevation)

Scope (v1)
----------
Included:
- TargetState dataclass (position/velocity in meters and m/s)
- propagate_cv(state, dt_s) -> TargetState
- range_and_rangerate(sensor_pos_m, sensor_vel_mps, target_state) -> (range_m, range_rate_mps)
- az_el(sensor_pos_m, target_pos_m) -> (az_rad, el_rad)

Not included (by design in v1):
- Maneuver models (CA/CT)
- Coordinated turns
- Earth curvature / geodetic frames
- Multi-target interaction

Inputs / Outputs
----------------
- All vectors are NumPy arrays of shape (3,) in SI units.
- Range is meters; range-rate is m/s.
- Angles are radians.

Public API
----------
- TargetState
- propagate_cv(state, dt_s)
- range_and_rangerate(sensor_pos_m, sensor_vel_mps, target_state)
- az_el(sensor_pos_m, target_pos_m)
- unit_vector(v)

Dependencies
------------
- NumPy
- Python standard library (dataclasses, math, typing)

Execution
---------
Not intended to be executed as a script.

Design notes
------------
- Strict input validation (raises ValueError on invalid shapes/non-finite values).
- Deterministic and side-effect free (no hidden state).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple
import math

import numpy as np


@dataclass(frozen=True)
class TargetState:
    """
    Target kinematic state in an inertial Cartesian frame.

    Attributes
    ----------
    pos_m : np.ndarray
        Position vector [m], shape (3,).
    vel_mps : np.ndarray
        Velocity vector [m/s], shape (3,).
    """
    pos_m: np.ndarray
    vel_mps: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "pos_m", _as_vec3(self.pos_m, name="pos_m"))
        object.__setattr__(self, "vel_mps", _as_vec3(self.vel_mps, name="vel_mps"))


def propagate_cv(state: TargetState, dt_s: float) -> TargetState:
    """
    Propagate a target state with a constant-velocity (CV) model.

    Parameters
    ----------
    state : TargetState
        Current state.
    dt_s : float
        Time step [s]. Can be 0; must be finite.

    Returns
    -------
    TargetState
        Propagated state: pos(t+dt) = pos + vel*dt, vel unchanged.
    """
    _require_finite_scalar(dt_s, name="dt_s")
    pos2 = state.pos_m + state.vel_mps * float(dt_s)
    return TargetState(pos_m=pos2, vel_mps=state.vel_mps)


def range_and_rangerate(
    sensor_pos_m: np.ndarray,
    sensor_vel_mps: np.ndarray,
    target: TargetState,
) -> Tuple[float, float]:
    """
    Compute range and range-rate between sensor and target.

    Range-rate sign convention
    --------------------------
    Positive range-rate means the target is receding (range increasing).

    Parameters
    ----------
    sensor_pos_m : np.ndarray
        Sensor position [m], shape (3,).
    sensor_vel_mps : np.ndarray
        Sensor velocity [m/s], shape (3,).
    target : TargetState
        Target state.

    Returns
    -------
    (range_m, range_rate_mps)
    """
    sp = _as_vec3(sensor_pos_m, name="sensor_pos_m")
    sv = _as_vec3(sensor_vel_mps, name="sensor_vel_mps")

    rel_pos = target.pos_m - sp
    rel_vel = target.vel_mps - sv

    r = float(np.linalg.norm(rel_pos))
    if r <= 0.0:
        # Degenerate co-located case; return zero range and zero rangerate.
        return 0.0, 0.0

    u = rel_pos / r
    rr = float(np.dot(rel_vel, u))
    return r, rr


def az_el(sensor_pos_m: np.ndarray, target_pos_m: np.ndarray) -> Tuple[float, float]:
    """
    Compute azimuth and elevation angles from sensor to target.

    Coordinate convention (v1)
    --------------------------
    We assume:
    - x axis: forward
    - y axis: left
    - z axis: up

    Azimuth is atan2(y, x) in radians.
    Elevation is atan2(z, sqrt(x^2 + y^2)) in radians.

    Parameters
    ----------
    sensor_pos_m : np.ndarray
        Sensor position [m], shape (3,).
    target_pos_m : np.ndarray
        Target position [m], shape (3,).

    Returns
    -------
    (az_rad, el_rad)
    """
    sp = _as_vec3(sensor_pos_m, name="sensor_pos_m")
    tp = _as_vec3(target_pos_m, name="target_pos_m")

    v = tp - sp
    x, y, z = float(v[0]), float(v[1]), float(v[2])

    az = math.atan2(y, x)
    xy = math.hypot(x, y)
    el = math.atan2(z, xy)
    return float(az), float(el)


def unit_vector(v: np.ndarray) -> np.ndarray:
    """
    Return the unit vector in the direction of v.

    Returns zeros if ||v|| == 0.
    """
    vv = _as_vec3(v, name="v")
    n = float(np.linalg.norm(vv))
    if n <= 0.0:
        return np.zeros(3, dtype=float)
    return vv / n


# -----------------------------------------------------------------------------
# Validation helpers
# -----------------------------------------------------------------------------

def _as_vec3(x: np.ndarray, *, name: str) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    if arr.shape != (3,):
        raise ValueError(f"{name} must have shape (3,), got {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values")
    return arr


def _require_finite_scalar(x: float, *, name: str) -> None:
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        raise ValueError(f"{name} must be numeric, got {type(x).__name__}")
    if not math.isfinite(float(x)):
        raise ValueError(f"{name} must be finite, got {x}")