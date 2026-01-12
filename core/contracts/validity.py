# core/contracts/validity.py
"""
core/contracts/validity.py

Explicit "validity contract" helpers.

Why this exists
---------------
This repository's goal is NOT to pretend the physics/statistics are universal.
It should *declare* what each engine run assumes so:
- reports are honest
- comparisons (model vs MC) are meaningful
- trade-offs and failure modes are visible

Contract shape (stable)
-----------------------
All engines that produce metrics.json should include:

metrics["validity"] = {
  "stat_model": str,
  "clutter": "none" | "homogeneous" | "heterogeneous",
  "interference": "none" | "noise_like_jammer" | ...,
  "limits": [str, ...]
}

Notes
-----
- Keep this lightweight: no heavy imports, no schema dependencies.
- It should be safe to call even if optional blocks are missing.
"""

from __future__ import annotations

from typing import Any, Dict, List


def _get_interference_label(cfg: Dict[str, Any]) -> str:
    """
    Map cfg["interference"] into a stable, report-friendly label.

    Current v1 supported case:
    - model: "noise_like_jammer"  -> additive noise-like interference at receiver input

    If the block is absent or invalid -> "none".
    """
    inter = cfg.get("interference", None)
    if not isinstance(inter, dict):
        return "none"

    model = str(inter.get("model", "")).strip().lower()
    if model == "noise_like_jammer":
        return "noise_like_jammer"

    # Unknown/unsupported model still counts as "some interference" in the config,
    # but to keep reporting stable and avoid lying, we classify it generically.
    # If you later add more models, extend this mapping explicitly.
    return "unknown"


def validity_for_model_based(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validity contract for model-based (analytic) engine runs.

    This engine assumes:
    - deterministic point target (unless later extended with Swerling models)
    - thermal noise modeled as Gaussian (receiver input), energy detector statistic uses chi-square family
    - clutter is not modeled (v1)
    - interference may be present if cfg["interference"] exists
    """
    interference = _get_interference_label(cfg)

    limits: List[str] = [
        "Monostatic free-space radar equation (no multipath/ducting).",
        "Point target RCS is deterministic unless explicitly extended (no Swerling in v1).",
        "Receiver noise modeled as Gaussian; detection statistic uses chi2/ncx2 family.",
        "No clutter model in this engine (noise-limited baseline).",
    ]

    if interference == "noise_like_jammer":
        limits.append("Interference modeled as additive noise-like power at receiver input (constant vs range in v1).")
    elif interference == "unknown":
        limits.append("Interference block present but model is not recognized by validity mapping (treat with caution).")

    return {
        "stat_model": "Gaussian noise + deterministic target (energy detector via chi2/ncx2)",
        "clutter": "none",
        "interference": "none" if interference == "none" else interference,
        "limits": limits,
    }


def validity_for_monte_carlo(cfg: Dict[str, Any], *, detector: str) -> Dict[str, Any]:
    """
    Validity contract for Monte Carlo Pfa experiments.

    This engine assumes:
    - Background samples in POWER domain drawn from the configured distribution.
    - For ca_cfar_independent: trials are independent by construction.
    - For ca_cfar_1d_sliding: samples are independent draws per trial, but window overlap
      creates correlation in threshold/detections along the line.
    - Heterogeneity (if enabled) is represented via multiplicative mean scaling.
    """
    mc = cfg.get("monte_carlo", {}) if isinstance(cfg.get("monte_carlo", {}), dict) else {}
    bg = mc.get("background", {}) if isinstance(mc.get("background", {}), dict) else {}
    model = str(bg.get("model", "exponential")).strip().lower()

    hetero = bg.get("hetero", None)
    clutter = "heterogeneous" if (isinstance(hetero, dict) and bool(hetero.get("enabled", False))) else "homogeneous"

    limits: List[str] = [
        "Background is modeled in POWER domain using the configured distribution.",
        "This experiment estimates Pfa (false alarm probability), not Pd (detection probability).",
        "No waveform / matched-filter / range-Doppler physics (statistical detector-level validation).",
    ]

    det = str(detector).strip().lower()
    if det == "ca_cfar_independent":
        limits.append("Independent-trial CA-CFAR: CUT and references are i.i.d. draws per trial (best for math validation).")
    elif det == "ca_cfar_1d_sliding":
        limits.append("Sliding CA-CFAR: window overlap induces correlated thresholds/detections along the line.")
    else:
        limits.append(f"Detector '{detector}' not recognized in validity mapping (treat with caution).")

    if clutter == "heterogeneous":
        limits.append("Heterogeneous background implemented via multiplicative mean scaling (piecewise or scalar).")

    return {
        "stat_model": f"Background power ~ {model} (Monte Carlo sampling)",
        "clutter": clutter,
        "interference": "none",
        "limits": limits,
    }