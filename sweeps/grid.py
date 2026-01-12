"""
sweeps/grid.py

Deterministic grid sweep engine for radar performance studies.

What this module does
---------------------
Executes a Cartesian (grid) sweep over selected configuration parameters,
running a full radar case for each grid point and collecting results.

This is NOT an optimizer.
This is NOT a heuristic search.
This is a deterministic Design Space Exploration (DSE) engine.

Primary purpose
---------------
- Trade studies (range vs Pd vs FAR vs dwell vs PRF, etc.)
- Sensitivity analysis (local effects of parameter changes)
- Generating structured datasets for plots and Pareto analysis

Design principles
-----------------
- Every sweep point is a fully reproducible radar case
- No mutation of input configs in-place
- No hidden defaults
- No randomization (unless the underlying engine uses it explicitly)

Inputs
------
- base_cfg : dict
    Fully validated case configuration (single-case).
- sweep_spec : dict
    Definition of sweep parameters and values.
- runner : callable
    Function that executes a single case (e.g., run_model_based_case).

Sweep specification format
--------------------------
sweep:
  parameters:
    - path: "radar.prf_hz"
      values: [500, 1000, 2000]
    - path: "detection.n_pulses"
      values: [8, 16, 32]

Cartesian product is implied.

Outputs
-------
- List of result dicts, one per grid point, each containing:
    - sweep_point: concrete parameter values
    - metrics: output from the engine
"""

from __future__ import annotations

from typing import Any, Dict, List, Callable
import copy
import itertools


class SweepError(ValueError):
    """Raised when sweep specification is invalid."""


# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------

def run_grid_sweep(
    *,
    base_cfg: Dict[str, Any],
    sweep_spec: Dict[str, Any],
    runner: Callable[[Dict[str, Any]], Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Execute a Cartesian grid sweep.

    Parameters
    ----------
    base_cfg : dict
        Base case configuration (will not be modified).
    sweep_spec : dict
        Sweep specification dictionary.
    runner : callable
        Function that runs a single case and returns metrics.

    Returns
    -------
    list of dict
        One entry per grid point with sweep_point + metrics.
    """
    params = _parse_sweep_parameters(sweep_spec)

    keys = [p["path"] for p in params]
    values = [p["values"] for p in params]

    results: List[Dict[str, Any]] = []

    for combo in itertools.product(*values):
        cfg_i = copy.deepcopy(base_cfg)
        sweep_point = {}

        for path, val in zip(keys, combo):
            _set_by_path(cfg_i, path, val)
            sweep_point[path] = val

        metrics = runner(cfg_i)

        results.append(
            {
                "sweep_point": sweep_point,
                "metrics": metrics,
            }
        )

    return results


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _parse_sweep_parameters(sweep_spec: Dict[str, Any]) -> List[Dict[str, Any]]:
    sweep = sweep_spec.get("sweep", None)
    if not isinstance(sweep, dict):
        raise SweepError("Missing 'sweep' section")

    params = sweep.get("parameters", None)
    if not isinstance(params, list) or len(params) == 0:
        raise SweepError("sweep.parameters must be a non-empty list")

    out = []
    for p in params:
        path = p.get("path", None)
        values = p.get("values", None)

        if not isinstance(path, str):
            raise SweepError("Each sweep parameter requires a string 'path'")
        if not isinstance(values, list) or len(values) == 0:
            raise SweepError(f"sweep parameter '{path}' must have non-empty values list")

        out.append({"path": path, "values": values})

    return out


def _set_by_path(cfg: Dict[str, Any], path: str, value: Any) -> None:
    """
    Set cfg["a"]["b"]["c"] = value given path "a.b.c".

    This is intentionally strict:
    - Intermediate dicts must already exist
    - No silent creation of structure
    """
    keys = path.split(".")
    cur = cfg

    for k in keys[:-1]:
        if k not in cur or not isinstance(cur[k], dict):
            raise SweepError(f"Invalid sweep path '{path}' (missing '{k}')")
        cur = cur[k]

    cur[keys[-1]] = value