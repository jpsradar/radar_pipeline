"""
core/tracker/logic.py

Multi-target tracker logic for the radar pipeline (v1).

Purpose
-------
Implement a deterministic, reviewable tracking loop that converts a stream of detections
into a set of tracks with stable identities and sensible lifecycle rules.

This v1 tracker is intended to be:
- credible and testable (unit-test friendly)
- simple enough to audit
- good enough for demo pipelines and sanity validation

It is NOT a production-grade tracker (no IMM, no JPDAF, no MHT).
But it is not a toy: it includes gating, association, filtering, and lifecycle management.

State / Measurement Model (v1)
------------------------------
- State is Cartesian position and velocity: x = [px, py, pz, vx, vy, vz]^T.
- Motion model: constant velocity (CV).
- Measurement: Cartesian position z = [px, py, pz]^T.
- Filter: classic linear Kalman filter with configurable process noise.

Inputs
------
Detections: list[dict] where each dict contains:
- "pos_m": np.ndarray shape (3,) (required)
- "t_s": float                 (required)
Optional:
- "cov_m2": np.ndarray shape (3,3) measurement covariance (defaults to diag)
- "snr_db": float
- "id": any

Tracker step expects detections for a single timestamp (or very close).
If multiple timestamps are mixed, results will be deterministic but semantically wrong.

Outputs
-------
- Track objects containing:
  - id, status ("tentative" | "confirmed"), age, last_update
  - state mean (6,), covariance (6,6)
  - history bookkeeping (hits/misses)

Public API (v1)
---------------
- TrackerParams: configuration
- Track: track dataclass
- Tracker: class managing a list of Track objects
    - step(detections, t_s) -> list[Track]   (updates internal tracks and returns snapshot)
    - tracks property for current list

Dependencies
------------
- numpy
- core/tracker/initiation.py

How to validate (example)
-------------------------
You can run a small smoke test in-console:
    python - <<'PY'
    import numpy as np
    from core.tracker.logic import Tracker, TrackerParams
    trk = Tracker(TrackerParams())
    t = 0.0
    det = [{"pos_m": np.array([0.0,0.0,0.0]), "t_s": t}]
    trk.step(det, t)
    t = 1.0
    det = [{"pos_m": np.array([1.0,0.0,0.0]), "t_s": t}]
    trk.step(det, t)
    print([ (x.track_id, x.status) for x in trk.tracks ])
    PY

Stability / Compatibility
-------------------------
- This module defines v1 tracker semantics. Future versions can extend Track fields
  but should keep the public API stable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from core.tracker.initiation import (
    InitiationParams,
    TentativeHistory,
    should_confirm,
    should_delete,
    validate_detection_pos,
)


@dataclass(frozen=True)
class TrackerParams:
    """
    Tracker configuration parameters.

    Attributes
    ----------
    initiation : InitiationParams
        Confirmation/deletion thresholds.
    meas_pos_sigma_m : float
        Default 1-sigma measurement noise on position (meters) if cov_m2 not provided.
    process_accel_sigma_mps2 : float
        1-sigma acceleration process noise used to build Q for CV model.
    gate_chi2 : float
        Squared Mahalanobis gate threshold for association (position space).
        Typical values: ~9.21 for 95% in 3D, ~11.34 for 99% in 3D (chi-square).
    max_assoc_per_track : int
        v1 uses 1-to-1 association; kept for future extension but enforced as 1 here.
    """
    initiation: InitiationParams = InitiationParams()
    meas_pos_sigma_m: float = 25.0
    process_accel_sigma_mps2: float = 5.0
    gate_chi2: float = 11.34  # ~99% in 3 DOF
    max_assoc_per_track: int = 1

    def __post_init__(self) -> None:
        if self.meas_pos_sigma_m <= 0.0 or not np.isfinite(self.meas_pos_sigma_m):
            raise ValueError("meas_pos_sigma_m must be finite and > 0")
        if self.process_accel_sigma_mps2 <= 0.0 or not np.isfinite(self.process_accel_sigma_mps2):
            raise ValueError("process_accel_sigma_mps2 must be finite and > 0")
        if self.gate_chi2 <= 0.0 or not np.isfinite(self.gate_chi2):
            raise ValueError("gate_chi2 must be finite and > 0")
        if self.max_assoc_per_track != 1:
            raise ValueError("v1 tracker enforces 1-to-1 association (max_assoc_per_track must be 1)")


@dataclass
class Track:
    """
    Tracker state container (mutable).

    Fields
    ------
    track_id : int
        Stable unique ID assigned by Tracker.
    status : str
        "tentative" or "confirmed".
    x : np.ndarray
        State mean, shape (6,): [px, py, pz, vx, vy, vz]
    P : np.ndarray
        State covariance, shape (6,6)
    history : TentativeHistory
        Hit/miss bookkeeping (used for confirm/delete logic).
    age_steps : int
        Number of tracker steps since creation.
    last_t_s : float
        Last update timestamp in seconds.
    """
    track_id: int
    status: str
    x: np.ndarray
    P: np.ndarray
    history: TentativeHistory
    age_steps: int = 0
    last_t_s: float = 0.0
    meta: Dict[str, Any] = field(default_factory=dict)

    def pos(self) -> np.ndarray:
        return self.x[0:3].copy()

    def vel(self) -> np.ndarray:
        return self.x[3:6].copy()


class Tracker:
    """
    Deterministic multi-target tracker (v1).

    Notes
    -----
    - Association is greedy by minimum Mahalanobis distance with gating.
    - 1-to-1 mapping: each detection assigned to at most one track and vice versa.
    - Unassigned detections spawn tentative tracks.
    - Tracks without detections get a miss.
    """

    def __init__(self, params: TrackerParams) -> None:
        self._params = params
        self._tracks: List[Track] = []
        self._next_id = 1

    @property
    def tracks(self) -> List[Track]:
        # Return a shallow copy for safety.
        return list(self._tracks)

    def step(self, detections: List[Dict[str, Any]], t_s: float) -> List[Track]:
        """
        Advance the tracker by one time step.

        Parameters
        ----------
        detections : list[dict]
            Detections for this time step (see module header).
        t_s : float
            Current timestamp in seconds.

        Returns
        -------
        list[Track]
            Snapshot of current tracks after update.
        """
        t_s = float(t_s)
        if not np.isfinite(t_s):
            raise ValueError("t_s must be finite")

        # 1) Predict existing tracks to current time
        for trk in self._tracks:
            dt = max(0.0, t_s - float(trk.last_t_s))
            self._predict(trk, dt)

        # 2) Build association cost matrix (Mahalanobis distance^2 in position space)
        det_pos, det_R = self._prepare_detections(detections)
        assignments = self._associate(det_pos, det_R)

        # 3) Apply updates for assigned pairs
        assigned_tracks = set()
        assigned_dets = set()

        for track_idx, det_idx in assignments:
            trk = self._tracks[track_idx]
            z = det_pos[det_idx]
            R = det_R[det_idx]
            self._update(trk, z, R)

            trk.history.add_hit(t_s=t_s)
            trk.last_t_s = t_s
            trk.age_steps += 1

            if trk.status == "tentative" and should_confirm(trk.history, self._params.initiation):
                trk.status = "confirmed"

            assigned_tracks.add(track_idx)
            assigned_dets.add(det_idx)

        # 4) Miss updates for unassigned tracks
        for i, trk in enumerate(self._tracks):
            if i in assigned_tracks:
                continue
            trk.history.add_miss(t_s=t_s)
            trk.last_t_s = t_s
            trk.age_steps += 1

        # 5) Spawn tracks from unassigned detections
        for j in range(len(detections)):
            if j in assigned_dets:
                continue
            self._spawn_from_detection(det_pos[j], det_R[j], t_s)

        # 6) Delete tracks that exceed miss threshold
        kept: List[Track] = []
        for trk in self._tracks:
            if should_delete(trk.history, self._params.initiation):
                continue
            kept.append(trk)
        self._tracks = kept

        return self.tracks

    # -------------------------
    # Internals
    # -------------------------

    def _prepare_detections(self, detections: List[Dict[str, Any]]) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        det_pos: List[np.ndarray] = []
        det_R: List[np.ndarray] = []

        default_var = float(self._params.meas_pos_sigma_m) ** 2
        R_default = np.diag([default_var, default_var, default_var]).astype(float)

        for d in detections:
            if not isinstance(d, dict):
                raise ValueError("Each detection must be a dict")
            if "pos_m" not in d or "t_s" not in d:
                raise ValueError("Detection must contain keys: 'pos_m' and 't_s'")

            z = validate_detection_pos(d["pos_m"])
            det_pos.append(z)

            R = d.get("cov_m2", None)
            if R is None:
                det_R.append(R_default.copy())
            else:
                Rn = np.asarray(R, dtype=float)
                if Rn.shape != (3, 3):
                    raise ValueError(f"Detection cov_m2 must be shape (3,3), got {Rn.shape}")
                if not np.all(np.isfinite(Rn)):
                    raise ValueError("Detection cov_m2 must be finite")
                # Ensure symmetric positive-ish
                Rn = 0.5 * (Rn + Rn.T)
                det_R.append(Rn)

        return det_pos, det_R

    def _associate(self, det_pos: List[np.ndarray], det_R: List[np.ndarray]) -> List[Tuple[int, int]]:
        # No tracks or no detections => trivial
        if len(self._tracks) == 0 or len(det_pos) == 0:
            return []

        # Compute all gated candidate pairs with costs
        candidates: List[Tuple[float, int, int]] = []
        for i, trk in enumerate(self._tracks):
            for j, z in enumerate(det_pos):
                d2 = self._mahalanobis_pos2(trk, z, det_R[j])
                if d2 <= float(self._params.gate_chi2):
                    candidates.append((float(d2), i, j))

        # Greedy: sort by smallest distance
        candidates.sort(key=lambda x: x[0])

        assigned_tracks = set()
        assigned_dets = set()
        out: List[Tuple[int, int]] = []

        for _, i, j in candidates:
            if i in assigned_tracks or j in assigned_dets:
                continue
            out.append((i, j))
            assigned_tracks.add(i)
            assigned_dets.add(j)

        return out

    def _spawn_from_detection(self, z: np.ndarray, R: np.ndarray, t_s: float) -> None:
        # Initialize state with zero velocity (conservative)
        x0 = np.zeros((6,), dtype=float)
        x0[0:3] = z

        P0 = np.zeros((6, 6), dtype=float)
        P0[0:3, 0:3] = R
        # Velocity uncertainty large by default (unobserved)
        vel_var = (50.0 ** 2)
        P0[3:6, 3:6] = np.diag([vel_var, vel_var, vel_var])

        hist = TentativeHistory.new(self._params.initiation.confirm_m)
        hist.add_hit(t_s=t_s)

        trk = Track(
            track_id=int(self._next_id),
            status="tentative",
            x=x0,
            P=P0,
            history=hist,
            age_steps=1,
            last_t_s=float(t_s),
        )
        self._next_id += 1
        self._tracks.append(trk)

    def _predict(self, trk: Track, dt: float) -> None:
        dt = float(dt)
        if dt <= 0.0:
            return

        # CV transition
        F = np.eye(6, dtype=float)
        F[0, 3] = dt
        F[1, 4] = dt
        F[2, 5] = dt

        # Process noise for acceleration white noise model
        q = float(self._params.process_accel_sigma_mps2) ** 2
        Q = _cv_process_noise_Q(dt, q)

        trk.x = F @ trk.x
        trk.P = F @ trk.P @ F.T + Q

    def _update(self, trk: Track, z: np.ndarray, R: np.ndarray) -> None:
        # Measurement model: position only
        H = np.zeros((3, 6), dtype=float)
        H[0, 0] = 1.0
        H[1, 1] = 1.0
        H[2, 2] = 1.0

        y = z - (H @ trk.x)
        S = H @ trk.P @ H.T + R
        # Robust inversion for 3x3
        S_inv = np.linalg.inv(S)
        K = trk.P @ H.T @ S_inv

        trk.x = trk.x + (K @ y)
        I = np.eye(6, dtype=float)
        trk.P = (I - K @ H) @ trk.P

    def _mahalanobis_pos2(self, trk: Track, z: np.ndarray, R: np.ndarray) -> float:
        H = np.zeros((3, 6), dtype=float)
        H[0, 0] = 1.0
        H[1, 1] = 1.0
        H[2, 2] = 1.0

        y = z - (H @ trk.x)
        S = H @ trk.P @ H.T + R
        S_inv = np.linalg.inv(S)
        d2 = float(y.T @ S_inv @ y)
        return d2


def _cv_process_noise_Q(dt: float, q: float) -> np.ndarray:
    """
    Continuous white acceleration noise discretized for CV model in 3D.

    Parameters
    ----------
    dt : float
        Time step (s)
    q : float
        Acceleration noise spectral density proxy (sigma_a^2), units (m/s^2)^2

    Returns
    -------
    np.ndarray
        6x6 process noise covariance.
    """
    dt2 = dt * dt
    dt3 = dt2 * dt
    dt4 = dt2 * dt2

    # 1D block for [pos, vel] with white accel
    Q1 = np.array(
        [[dt4 / 4.0, dt3 / 2.0],
         [dt3 / 2.0, dt2]],
        dtype=float,
    ) * q

    Q = np.zeros((6, 6), dtype=float)
    # x
    Q[np.ix_([0, 3], [0, 3])] = Q1
    # y
    Q[np.ix_([1, 4], [1, 4])] = Q1
    # z
    Q[np.ix_([2, 5], [2, 5])] = Q1
    return Q