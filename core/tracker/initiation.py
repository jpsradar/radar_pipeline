"""
core/tracker/initiation.py

Track initiation utilities for the radar pipeline (v1).

Purpose
-------
Provide deterministic, testable track initiation logic that can be used by:
- future multi-target simulations / detections-to-tracks glue
- report generation and algorithm sanity checks
- tracker unit tests (gating, confirmation, pruning)

Design goals (v1)
-----------------
- Conservative and explicit: no hidden heuristics.
- Deterministic given (detections, timestamps, parameters).
- Minimal external coupling: this module does not depend on DSP or target motion modules.
- Works with Cartesian detections (x,y,z) and optional measurement covariance.

Concepts
--------
We use a standard "tentative -> confirmed" initiation scheme:
- A new tentative track is created from a detection if it cannot be associated to an existing track.
- A tentative track is promoted to confirmed after N hits in a sliding window of M steps
  (classic M-of-N logic).
- Tracks (tentative or confirmed) are deleted after a configured number of consecutive misses.

Inputs
------
Detections are expected as a list of dicts, each containing:
- "pos_m": np.ndarray shape (3,)  (required)
- "t_s": float                    (required)
Optional:
- "cov_m2": np.ndarray shape (3,3) measurement covariance in meters^2
- "snr_db": float
- "id": any stable identifier

Outputs
-------
This module provides helper functions and dataclasses used by core/tracker/logic.py:
- InitiationParams: configuration for confirmation/deletion
- TentativeHistory: rolling hit/miss bookkeeping
- should_confirm(history, params) -> bool
- should_delete(history, params) -> bool

Dependencies
------------
- numpy

How to use
----------
This module is typically used by the main tracker loop in core/tracker/logic.py.
You generally won't call it directly from CLI.

Stability / Compatibility
-------------------------
- v1 guarantees: input keys and dataclass fields documented above.
- Future versions may extend detection fields, but will keep these stable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Deque, Optional
from collections import deque

import numpy as np


@dataclass(frozen=True)
class InitiationParams:
    """
    Parameters controlling track confirmation and deletion.

    Attributes
    ----------
    confirm_m : int
        Window length for M-of-N confirmation (number of most recent steps tracked).
    confirm_n : int
        Minimum number of hits within the last M steps required to confirm.
    delete_after_misses : int
        Number of consecutive misses after which a track is deleted.
    """
    confirm_m: int = 5
    confirm_n: int = 3
    delete_after_misses: int = 5

    def __post_init__(self) -> None:
        if self.confirm_m <= 0:
            raise ValueError("confirm_m must be >= 1")
        if self.confirm_n <= 0:
            raise ValueError("confirm_n must be >= 1")
        if self.confirm_n > self.confirm_m:
            raise ValueError("confirm_n must be <= confirm_m")
        if self.delete_after_misses <= 0:
            raise ValueError("delete_after_misses must be >= 1")


@dataclass
class TentativeHistory:
    """
    Rolling hit/miss bookkeeping for a track.

    Notes
    -----
    - 'hits' is a rolling window of booleans with maxlen=confirm_m.
    - 'consecutive_misses' is tracked explicitly for deletion.
    """
    hits: Deque[bool]
    consecutive_misses: int = 0
    last_update_t_s: Optional[float] = None

    @staticmethod
    def new(confirm_m: int) -> "TentativeHistory":
        if confirm_m <= 0:
            raise ValueError("confirm_m must be >= 1")
        return TentativeHistory(hits=deque(maxlen=confirm_m))

    def add_hit(self, *, t_s: float) -> None:
        self.hits.append(True)
        self.consecutive_misses = 0
        self.last_update_t_s = float(t_s)

    def add_miss(self, *, t_s: float) -> None:
        self.hits.append(False)
        self.consecutive_misses += 1
        self.last_update_t_s = float(t_s)


def should_confirm(history: TentativeHistory, params: InitiationParams) -> bool:
    """
    Return True if a track should be promoted to confirmed.

    Confirmation rule (M-of-N)
    --------------------------
    Confirm if number of True values within the rolling window >= confirm_n.
    """
    # If we haven't accumulated enough steps yet, we still can confirm early if enough hits exist.
    hit_count = int(sum(1 for x in history.hits if x))
    return hit_count >= int(params.confirm_n)


def should_delete(history: TentativeHistory, params: InitiationParams) -> bool:
    """
    Return True if a track should be deleted.

    Deletion rule
    -------------
    Delete if consecutive_misses >= delete_after_misses.
    """
    return int(history.consecutive_misses) >= int(params.delete_after_misses)


def validate_detection_pos(pos_m: np.ndarray) -> np.ndarray:
    """
    Validate and normalize detection position vector.

    Returns
    -------
    np.ndarray
        float64 array of shape (3,)

    Raises
    ------
    ValueError if invalid.
    """
    v = np.asarray(pos_m, dtype=float).reshape(-1)
    if v.shape != (3,):
        raise ValueError(f"detection pos_m must be shape (3,), got {v.shape}")
    if not np.all(np.isfinite(v)):
        raise ValueError("detection pos_m must be finite")
    return v