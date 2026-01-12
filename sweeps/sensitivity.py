"""
sweeps/sensitivity.py

Sensitivity analysis utilities for the radar pipeline.

Purpose
-------
Provide serious, deterministic sensitivity analysis primitives that can be used
to assess how outputs (metrics) respond to parameter perturbations.

This module is intentionally engine-agnostic:
- It does not import or call radar engines directly.
- Instead, the user supplies an evaluation callable: eval_fn(cfg) -> metrics.

This keeps sensitivity analysis reusable across:
- model_based engine
- monte_carlo engine (with fixed seeds for determinism)
- signal_level engine
- future DSP-heavy engines

What is included (v1)
---------------------
1) One-At-a-Time (OAT) sensitivity:
   - Vary one parameter at a time around a baseline config.
   - Uses either relative perturbation (multiplicative) or absolute perturbation (additive).
   - Produces finite-difference gradients and normalized sensitivities.

2) Dotpath manipulation utilities:
   - get/set nested dict values using "a.b.c" paths.
   - Non-mutating: input configs are not modified in-place.

Inputs
------
- base_cfg: dict (baseline case configuration)
- params: list of dotpaths to perturb
- eval_fn: callable returning metrics (dict)
- metric_path: dotpath into metrics to extract scalar response (e.g., "detection.pd[0]" is NOT supported;
  for v1 we support dict dotpaths to scalar values only).
- step: perturbation size (relative or absolute)
- mode: "relative" (x*(1±step)) or "absolute" (x±step)

Outputs
-------
A JSON-serializable dict with:
- baseline value
- per-parameter plus/minus evaluations
- central difference gradient estimate
- normalized sensitivity (if baseline parameter is non-zero)

Determinism / Reproducibility
-----------------------------
Determinism is guaranteed if eval_fn is deterministic for given configs.
For Monte Carlo style eval_fn, pass a fixed seed into the config or closure.

Dependencies
------------
- numpy
- standard library only otherwise

Usage (example)
---------------
    from sweeps.sensitivity import oat_sensitivity
    from core.simulation.model_based import run_model_based_case

    def eval_fn(cfg):
        return run_model_based_case(cfg, seed=123)

    out = oat_sensitivity(
        base_cfg=cfg,
        params=["radar.tx_power_w", "receiver.nf_db"],
        eval_fn=eval_fn,
        metric_path="detection.pd_mean",
        step=0.05,
        mode="relative",
    )

Notes
-----
- v1 focuses on scalar metric extraction.
- If you need vector metrics (Pd vs range), compute a scalar summary in eval_fn.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Sequence, Tuple
import copy
import math



MetricFn = Callable[[Dict[str, Any]], Dict[str, Any]]


def oat_sensitivity(
    *,
    base_cfg: Dict[str, Any],
    params: Sequence[str],
    eval_fn: MetricFn,
    metric_path: str,
    step: float,
    mode: str = "relative",
) -> Dict[str, Any]:
    """
    One-at-a-time (OAT) sensitivity with central differences.

    Parameters
    ----------
    base_cfg : dict
        Baseline configuration (not mutated).
    params : sequence[str]
        Dotpaths to perturb (must refer to scalar numeric fields in base_cfg).
    eval_fn : callable
        Function that takes a config dict and returns a metrics dict.
    metric_path : str
        Dotpath into metrics dict to extract a scalar numeric response.
    step : float
        Perturbation step size.
        - relative: x*(1±step)
        - absolute: x±step
    mode : str
        "relative" or "absolute"

    Returns
    -------
    dict
        JSON-serializable sensitivity report.

    Raises
    ------
    ValueError for invalid inputs or non-numeric base values.
    """
    if mode not in ("relative", "absolute"):
        raise ValueError("mode must be 'relative' or 'absolute'")
    if not (math.isfinite(float(step)) and float(step) > 0.0):
        raise ValueError("step must be finite and > 0")
    if not isinstance(metric_path, str) or not metric_path.strip():
        raise ValueError("metric_path must be a non-empty string")

    base_cfg_c = deep_copy_cfg(base_cfg)
    base_metrics = eval_fn(base_cfg_c)
    y0 = float(extract_scalar_metric(base_metrics, metric_path))

    out: Dict[str, Any] = {
        "method": "oat_central_difference_v1",
        "metric_path": metric_path,
        "mode": mode,
        "step": float(step),
        "baseline": {
            "metric": float(y0),
        },
        "parameters": [],
    }

    for p in params:
        x0 = get_dotpath(base_cfg_c, p)
        if isinstance(x0, bool) or not isinstance(x0, (int, float)):
            raise ValueError(f"OAT requires numeric scalar parameter at '{p}', got {type(x0).__name__}")
        x0f = float(x0)
        if not math.isfinite(x0f):
            raise ValueError(f"Parameter '{p}' must be finite, got {x0}")

        x_minus, x_plus = _perturb(x0f, step=float(step), mode=mode)

        cfg_minus = deep_copy_cfg(base_cfg_c)
        cfg_plus = deep_copy_cfg(base_cfg_c)
        set_dotpath(cfg_minus, p, x_minus)
        set_dotpath(cfg_plus, p, x_plus)

        m_minus = eval_fn(cfg_minus)
        m_plus = eval_fn(cfg_plus)

        y_minus = float(extract_scalar_metric(m_minus, metric_path))
        y_plus = float(extract_scalar_metric(m_plus, metric_path))

        # Central difference gradient dy/dx
        denom = (x_plus - x_minus)
        if denom == 0.0:
            raise ValueError(f"Degenerate perturbation for '{p}' (x_plus == x_minus)")
        grad = (y_plus - y_minus) / denom

        # Normalized sensitivity: (x0/y0) * dy/dx (if y0 != 0)
        if abs(y0) > 0.0 and math.isfinite(y0):
            sens_norm = (x0f / y0) * grad
        else:
            sens_norm = None

        out["parameters"].append(
            {
                "path": p,
                "x0": float(x0f),
                "x_minus": float(x_minus),
                "x_plus": float(x_plus),
                "y_minus": float(y_minus),
                "y_plus": float(y_plus),
                "gradient_dy_dx": float(grad),
                "normalized_sensitivity": float(sens_norm) if sens_norm is not None and math.isfinite(float(sens_norm)) else None,
            }
        )

    return out


# ---------------------------------------------------------------------
# Dotpath utilities (non-mutating by default)
# ---------------------------------------------------------------------

def deep_copy_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Deep copy a configuration dict (correctness over micro-optimizations)."""
    if not isinstance(cfg, dict):
        raise ValueError("cfg must be a dict")
    return copy.deepcopy(cfg)


def get_dotpath(d: Dict[str, Any], path: str) -> Any:
    """
    Get a nested value from dict using dotpath "a.b.c".

    Raises KeyError if path does not exist.
    """
    if not isinstance(path, str) or not path.strip():
        raise ValueError("path must be non-empty string")
    cur: Any = d
    for key in path.split("."):
        if not isinstance(cur, dict):
            raise KeyError(f"Dotpath '{path}' traversed non-dict at '{key}'")
        if key not in cur:
            raise KeyError(f"Missing key '{key}' in dotpath '{path}'")
        cur = cur[key]
    return cur


def set_dotpath(d: Dict[str, Any], path: str, value: Any) -> None:
    """
    Set a nested value in dict using dotpath "a.b.c".

    This mutates 'd' (caller typically passes a copied dict).

    Raises KeyError if path does not exist (strict by design).
    """
    if not isinstance(path, str) or not path.strip():
        raise ValueError("path must be non-empty string")
    keys = path.split(".")
    cur: Any = d
    for key in keys[:-1]:
        if not isinstance(cur, dict):
            raise KeyError(f"Dotpath '{path}' traversed non-dict at '{key}'")
        if key not in cur:
            raise KeyError(f"Missing key '{key}' in dotpath '{path}'")
        cur = cur[key]
    last = keys[-1]
    if not isinstance(cur, dict) or last not in cur:
        raise KeyError(f"Missing key '{last}' in dotpath '{path}'")
    cur[last] = value


def extract_scalar_metric(metrics: Dict[str, Any], metric_path: str) -> float:
    """
    Extract a scalar numeric metric from a metrics dict via dotpath.

    v1 limitation:
    - Only supports dot-separated dict keys, no indexing (e.g., 'pd[0]' not supported).
    """
    val = get_dotpath(metrics, metric_path)
    if isinstance(val, bool) or not isinstance(val, (int, float)):
        raise ValueError(f"Metric at '{metric_path}' must be numeric scalar, got {type(val).__name__}")
    v = float(val)
    if not math.isfinite(v):
        raise ValueError(f"Metric at '{metric_path}' must be finite, got {val}")
    return v


# ---------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------

def _perturb(x0: float, *, step: float, mode: str) -> Tuple[float, float]:
    if mode == "relative":
        # x*(1±step). If x0==0, relative is degenerate; we still perturb symmetrically around 0.
        return (x0 * (1.0 - step), x0 * (1.0 + step))
    if mode == "absolute":
        return (x0 - step, x0 + step)
    raise ValueError("mode must be 'relative' or 'absolute'")