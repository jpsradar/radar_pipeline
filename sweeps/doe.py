"""
sweeps/doe.py

Design-of-Experiments (DoE) utilities for the radar pipeline.

Purpose
-------
This module generates deterministic experiment designs (parameter sets) to drive
sweeps in a controlled and reproducible way.

It supports common DoE styles used in engineering pipelines:
- Full factorial grids (exact coverage, can explode combinatorially)
- Random uniform sampling (baseline stochastic exploration)
- Latin Hypercube Sampling (LHS) (space-filling, efficient for moderate dimensions)

This module is intentionally decoupled from radar engines:
- It only produces *parameter dictionaries*.
- Execution (running cases/engines) is handled elsewhere (e.g., cli/run_sweep.py).

Inputs
------
Parameter spaces are defined with ParamSpec objects:
- name: a dotpath key (e.g., "radar.tx_power_w", "receiver.nf_db")
- kind: "continuous" | "discrete"
- bounds/values: numeric bounds for continuous, explicit values for discrete
- transform: optional mapping for continuous variables (e.g., "log10")

Outputs
-------
Primary output is a list[dict[str, float|int|str]] where each dict is one experiment.

These dicts are designed to be:
- JSON-serializable
- directly applicable as overrides to a base config (via dotpaths)

Determinism / Reproducibility
-----------------------------
All stochastic designs accept a seed and use numpy.random.Generator.
Given the same (specs, n, seed), the output is byte-for-byte stable.

Dependencies
------------
- numpy
- standard library only otherwise

Usage (examples)
----------------
1) Full factorial:
    specs = [
        ParamSpec.discrete("receiver.nf_db", [3.0, 5.0, 7.0]),
        ParamSpec.discrete("radar.tx_power_w", [500.0, 1000.0]),
    ]
    design = full_factorial(specs)

2) LHS:
    specs = [
        ParamSpec.continuous("radar.tx_power_w", 200.0, 2000.0, transform="log10"),
        ParamSpec.continuous("receiver.nf_db", 2.0, 10.0),
    ]
    design = latin_hypercube(specs, n=50, seed=123)

3) Random:
    design = random_uniform(specs, n=50, seed=123)

Notes
-----
- This module does not read or write files.
- It does not mutate inputs.
- It avoids pandas to keep the dependency surface small.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple
import itertools
import math

import numpy as np


@dataclass(frozen=True)
class ParamSpec:
    """
    Parameter specification for DoE generation.

    Attributes
    ----------
    name : str
        Dotpath key for the parameter (e.g., "radar.tx_power_w").
    kind : str
        "continuous" or "discrete".
    low : float | None
        Lower bound for continuous parameters (in *natural* units).
    high : float | None
        Upper bound for continuous parameters (in *natural* units).
    values : list[Any] | None
        Explicit values for discrete parameters.
    transform : str | None
        Optional transform for continuous parameters:
        - None: sample uniformly in [low, high]
        - "log10": sample uniformly in log10-space, then map back (positive only)
    """
    name: str
    kind: str
    low: Optional[float] = None
    high: Optional[float] = None
    values: Optional[Tuple[Any, ...]] = None
    transform: Optional[str] = None

    @staticmethod
    def continuous(name: str, low: float, high: float, *, transform: Optional[str] = None) -> "ParamSpec":
        return ParamSpec(name=name, kind="continuous", low=float(low), high=float(high), values=None, transform=transform)

    @staticmethod
    def discrete(name: str, values: Sequence[Any]) -> "ParamSpec":
        return ParamSpec(name=name, kind="discrete", low=None, high=None, values=tuple(values), transform=None)

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("ParamSpec.name must be a non-empty string")
        if self.kind not in ("continuous", "discrete"):
            raise ValueError("ParamSpec.kind must be 'continuous' or 'discrete'")

        if self.kind == "continuous":
            if self.low is None or self.high is None:
                raise ValueError("Continuous ParamSpec requires low/high")
            if not (math.isfinite(float(self.low)) and math.isfinite(float(self.high))):
                raise ValueError("Continuous ParamSpec low/high must be finite")
            if float(self.high) <= float(self.low):
                raise ValueError("Continuous ParamSpec requires high > low")
            if self.transform not in (None, "log10"):
                raise ValueError("transform must be None or 'log10'")
            if self.transform == "log10":
                if float(self.low) <= 0.0 or float(self.high) <= 0.0:
                    raise ValueError("log10 transform requires low/high > 0")
        else:
            if self.values is None or len(self.values) == 0:
                raise ValueError("Discrete ParamSpec requires a non-empty values list")


def full_factorial(specs: Sequence[ParamSpec]) -> List[Dict[str, Any]]:
    """
    Generate a full factorial design over discrete parameters.

    Requirements
    ------------
    - All specs must be discrete.

    Returns
    -------
    list[dict]
        Each dict maps spec.name -> chosen value.

    Notes
    -----
    - This can explode combinatorially; caller is responsible for size control.
    - Order is deterministic and stable: lexical order of specs, then cartesian product order.
    """
    specs = list(specs)
    for s in specs:
        if s.kind != "discrete":
            raise ValueError("full_factorial supports only discrete ParamSpec")

    axes = [list(s.values or ()) for s in specs]
    out: List[Dict[str, Any]] = []
    for combo in itertools.product(*axes):
        row = {specs[i].name: combo[i] for i in range(len(specs))}
        out.append(row)
    return out


def random_uniform(specs: Sequence[ParamSpec], *, n: int, seed: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Generate a random uniform design.

    Behavior
    --------
    - continuous: uniform in [low, high] (or uniform in log10-space if transform="log10")
    - discrete: uniform random choice among values

    Parameters
    ----------
    specs : sequence of ParamSpec
    n : int
        Number of samples.
    seed : int | None
        RNG seed for deterministic output.

    Returns
    -------
    list[dict]
    """
    if n <= 0:
        raise ValueError("n must be >= 1")

    rng = np.random.default_rng(None if seed is None else int(seed))
    out: List[Dict[str, Any]] = []

    for _ in range(int(n)):
        row: Dict[str, Any] = {}
        for s in specs:
            if s.kind == "discrete":
                vals = list(s.values or ())
                idx = int(rng.integers(0, len(vals)))
                row[s.name] = vals[idx]
            else:
                row[s.name] = _sample_continuous(rng, s)
        out.append(row)

    return out


def latin_hypercube(specs: Sequence[ParamSpec], *, n: int, seed: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Generate a Latin Hypercube Sampling (LHS) design.

    Scope (v1)
    ----------
    LHS is applied to continuous dimensions. Discrete dimensions are sampled independently
    (uniform choice per sample) and included alongside continuous LHS coordinates.

    Parameters
    ----------
    specs : sequence of ParamSpec
    n : int
        Number of samples (rows).
    seed : int | None
        RNG seed for deterministic output.

    Returns
    -------
    list[dict]
        Each dict maps spec.name -> sampled value.

    Implementation notes
    --------------------
    - For each continuous dimension, we split [0,1] into n strata and pick one point
      uniformly within each stratum, then permute strata per dimension.
    - This is the classic, credible LHS construction.
    """
    if n <= 0:
        raise ValueError("n must be >= 1")

    rng = np.random.default_rng(None if seed is None else int(seed))
    specs = list(specs)

    cont = [s for s in specs if s.kind == "continuous"]
    disc = [s for s in specs if s.kind == "discrete"]

    # Build LHS unit hypercube for continuous specs: shape (n, d)
    d = len(cont)
    if d > 0:
        u = np.zeros((n, d), dtype=float)
        for j in range(d):
            # strata boundaries
            strata = (np.arange(n, dtype=float) + rng.random(n)) / float(n)
            rng.shuffle(strata)
            u[:, j] = strata
    else:
        u = np.zeros((n, 0), dtype=float)

    out: List[Dict[str, Any]] = []
    for i in range(n):
        row: Dict[str, Any] = {}

        # continuous dims: map u[i,j] -> [low,high] (or log transform)
        for j, s in enumerate(cont):
            row[s.name] = _map_unit_to_param(float(u[i, j]), s)

        # discrete dims: uniform choice per sample
        for s in disc:
            vals = list(s.values or ())
            idx = int(rng.integers(0, len(vals)))
            row[s.name] = vals[idx]

        out.append(row)

    return out


# ---------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------

def _sample_continuous(rng: np.random.Generator, s: ParamSpec) -> float:
    u = float(rng.random())
    return _map_unit_to_param(u, s)


def _map_unit_to_param(u: float, s: ParamSpec) -> float:
    if s.kind != "continuous":
        raise ValueError("_map_unit_to_param expects continuous ParamSpec")
    lo = float(s.low)  # type: ignore[arg-type]
    hi = float(s.high)  # type: ignore[arg-type]
    if not (0.0 <= u <= 1.0) or not math.isfinite(u):
        raise ValueError("u must be finite in [0,1]")

    if s.transform is None:
        return lo + (hi - lo) * u

    if s.transform == "log10":
        lo_l = math.log10(lo)
        hi_l = math.log10(hi)
        x_l = lo_l + (hi_l - lo_l) * u
        return 10.0 ** x_l

    raise ValueError(f"Unsupported transform: {s.transform}")