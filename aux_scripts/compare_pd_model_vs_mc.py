#!/usr/bin/env python3
"""
aux_scripts/compare_pd_model_vs_mc.py

Pd vs SNR comparison: analytic model vs Monte Carlo (with confidence intervals)

Purpose
-------
Provide a reproducible, quantitative comparison between:
  - analytic Pd predictions derived from closed-form statistical models, and
  - empirical Pd estimates obtained via the repository's Monte Carlo engine,

as a function of SNR.

This is a falsification artifact:
it makes model validity explicit, shows where assumptions hold/break,
and whether discrepancies are statistically significant.

Scope (TOP-NOTCH baseline)
--------------------------
- Detector:
    * CA-CFAR (independent trials)
- Integration:
    * Power-domain noncoherent integration (sum of per-pulse powers)
- Background:
    * Exponential power (thermal noise / homogeneous clutter)
- Target fluctuation:
    * Swerling 0/1/2/3/4 (power-domain multipliers, consistent with core/simulation/monte_carlo.py)

Analytic model assumptions
--------------------------
- Background power per pulse in the CUT is exponential with mean = mean_power.
- After noncoherent integration of N pulses:
      X_bg ~ Erlang/Gamma(k=N, theta=mean_power)   [integer shape]
- CA-CFAR threshold is modeled with *expected* reference mean (not the random sample mean):
      thr = alpha(pfa, n_ref) * E[mean(ref_integrated)]
      E[mean(ref_integrated)] = N * mean_power
  This is intentionally "model-side smoothing": it tests how much of Pd behavior
  is driven by CUT statistics vs threshold randomness.

Signal + Swerling models in the power domain
--------------------------------------------
Let mean_sig_per_pulse = snr_lin * mean_power.

- Swerling-0 (nonfluctuating):
    signal_integrated = N * mean_sig_per_pulse (deterministic)

- Swerling-1 (slow, exponential multiplier per CPI):
    V ~ Erlang(k=1, theta=1) (Exp with mean 1)
    signal_integrated = N * mean_sig_per_pulse * V

- Swerling-2 (fast, exponential multiplier per pulse):
    Vsum = sum_{i=1..N} Vi, Vi~Exp(mean=1) iid
    Vsum ~ Erlang(k=N, theta=1)
    signal_integrated = mean_sig_per_pulse * Vsum

- Swerling-3 (slow, gamma(k=2) multiplier per CPI):
    V ~ Erlang(k=2, theta=0.5) (mean 1)
    signal_integrated = N * mean_sig_per_pulse * V

- Swerling-4 (fast, gamma(k=2) per pulse):
    Vi ~ Erlang(k=2, theta=0.5) iid
    Vsum ~ Erlang(k=2N, theta=0.5)
    signal_integrated = mean_sig_per_pulse * Vsum

Given:
    CUT = X_bg + signal_integrated
    X_bg ~ Erlang(k=N, theta=mean_power)

Pd model is computed as:
    Pd = P(CUT > thr)
For Swerling-0: closed-form Erlang tail.
For Swerling-1/2/3/4: 1D integral over Erlang-distributed V (or Vsum),
computed deterministically (adaptive Simpson), without SciPy.

Monte Carlo reference
---------------------
- Uses core/simulation/monte_carlo.run_monte_carlo as empirical reference.
- For each SNR point:
    * runs independent trials,
    * estimates Pd empirically,
    * uses the engine-provided Wilson 95% CI when available (single source of truth),
      falling back to local Wilson computation if absent.

Contract enforcement (fail loud)
--------------------------------
This script is only valid if the case matches the modeled regime:
  - monte_carlo.task == "pd"
  - monte_carlo.detector == "ca_cfar_independent"
  - monte_carlo.background.model == "exponential"
  - monte_carlo.background.hetero.enabled == false (heterogeneity is out-of-model on purpose)
  - monte_carlo.pd.integration == "noncoherent"
  - monte_carlo.pd.swerling in {swerling0..swerling4}

Outputs
-------
For a given case and seed, the script writes under:
  results/comparisons/<tag>__seed<SEED>/

Artifacts:
  - comparison.csv   : numeric comparison (model vs MC + CI)
  - comparison.json  : full metadata + summary diagnostics
  - comparison.png   : Pd vs SNR plot (model curve + MC points + CI band)
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from core.detection.cfar import ca_cfar_alpha
from core.simulation.monte_carlo import run_monte_carlo


# -----------------------------
# Binomial CI (Wilson) [fallback]
# -----------------------------


def wilson_ci(p: float, n: int, z: float = 1.96) -> Tuple[float, float]:
    if n <= 0:
        return (0.0, 1.0)
    p = float(p)
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p + z2 / (2.0 * n)) / denom
    half = (z / denom) * math.sqrt((p * (1.0 - p) / n) + (z2 / (4.0 * n * n)))
    return (max(0.0, center - half), min(1.0, center + half))


def _extract_pd_metrics(m: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract the Pd metrics dict from run_monte_carlo output.

    Supported shapes:
      - task=pd:     {"pd_empirical": ..., "confidence_intervals": ...}
      - task=pfa_pd: {"pd": {... "pd_empirical": ...}}
    """
    if not isinstance(m, dict):
        raise SystemExit(f"[ERROR] run_monte_carlo returned non-dict: {type(m)}")

    # Direct Pd task
    if "pd_empirical" in m:
        return m

    # Envelope
    pd = m.get("pd", None)
    if isinstance(pd, dict) and "pd_empirical" in pd:
        return pd

    raise SystemExit(
        "[ERROR] Could not locate pd metrics in run_monte_carlo output. "
        f"Top-level keys={sorted(list(m.keys()))}"
    )


def _extract_wilson_ci_from_metrics(pd_metrics: Dict[str, Any]) -> Optional[Tuple[float, float]]:
    """
    Prefer engine-provided Wilson CI if available:
      confidence_intervals.wilson_95.{low,high}
    """
    ci = pd_metrics.get("confidence_intervals", None)
    if not isinstance(ci, dict):
        return None
    w = ci.get("wilson_95", None)
    if not isinstance(w, dict):
        return None
    lo = w.get("low", None)
    hi = w.get("high", None)
    if lo is None or hi is None:
        return None
    try:
        return (float(lo), float(hi))
    except Exception:
        return None


# -----------------------------
# Erlang/Gamma helpers (integer shape)
# -----------------------------


def erlang_tail_prob(x: float, *, k: int, theta: float) -> float:
    """
    Tail probability P(X > x) for Erlang/Gamma with integer shape k and scale theta.
    If X ~ Gamma(k, theta), k integer >= 1:
      P(X > x) = exp(-x/theta) * sum_{m=0}^{k-1} (x/theta)^m / m!
    """
    if k < 1:
        raise ValueError("k must be >= 1")
    if theta <= 0.0:
        raise ValueError("theta must be > 0")
    if x <= 0.0:
        return 1.0

    t = x / theta
    s = 0.0
    term = 1.0  # m=0
    for m in range(0, k):
        if m == 0:
            term = 1.0
        else:
            term *= t / m
        s += term
    return float(math.exp(-t) * s)


def erlang_pdf(x: float, *, k: int, theta: float) -> float:
    """
    PDF for Erlang/Gamma with integer k>=1, scale theta>0:
      f(x)= x^{k-1} * exp(-x/theta) / (theta^k * (k-1)!)
    """
    if x < 0.0:
        return 0.0
    if k < 1:
        raise ValueError("k must be >= 1")
    if theta <= 0.0:
        raise ValueError("theta must be > 0")

    if x == 0.0:
        return (1.0 / theta) if k == 1 else 0.0

    return float((x ** (k - 1)) * math.exp(-x / theta) / ((theta ** k) * math.factorial(k - 1)))


# -----------------------------
# Deterministic 1D integration (adaptive Simpson)
# -----------------------------


def _simpson(f, a: float, b: float) -> float:
    c = 0.5 * (a + b)
    return (b - a) / 6.0 * (f(a) + 4.0 * f(c) + f(b))


def adaptive_simpson(f, a: float, b: float, *, eps: float = 1e-8, max_depth: int = 20) -> float:
    """
    Adaptive Simpson integration of f over [a,b].
    Deterministic, no external deps.
    """
    whole = _simpson(f, a, b)

    def rec(a0: float, b0: float, whole0: float, depth: int) -> float:
        c0 = 0.5 * (a0 + b0)
        left = _simpson(f, a0, c0)
        right = _simpson(f, c0, b0)
        if depth <= 0:
            return left + right
        if abs(left + right - whole0) <= 15.0 * eps:
            return left + right + (left + right - whole0) / 15.0
        return rec(a0, c0, left, depth - 1) + rec(c0, b0, right, depth - 1)

    return float(rec(a, b, whole, max_depth))


# -----------------------------
# Pd model (Swerling 0–4)
# -----------------------------


def _swerling_signal_v_dist(*, swerling: str, n_pulses: int) -> Tuple[int, float, bool]:
    """
    Returns (k_v, theta_v, is_sum_model) where:
      - If is_sum_model=False: signal_integrated = N * mean_sig * V, V~Erlang(k_v, theta_v)
      - If is_sum_model=True : signal_integrated = mean_sig * Vsum, Vsum~Erlang(k_v, theta_v)
    """
    sw = swerling.lower().strip()
    if sw == "swerling1":
        return (1, 1.0, False)
    if sw == "swerling2":
        return (n_pulses, 1.0, True)
    if sw == "swerling3":
        return (2, 0.5, False)
    if sw == "swerling4":
        return (2 * n_pulses, 0.5, True)
    raise ValueError(f"Unsupported swerling for V-dist: {swerling}")


def pd_model(
    *,
    swerling: str,
    snr_db: float,
    mean_power: float,
    n_pulses: int,
    pfa: float,
    n_ref: int,
) -> float:
    """
    Analytic Pd model using expected reference mean for the CA-CFAR threshold.

    thr = alpha(pfa,n_ref) * (N*mean_power)

    CUT = X_bg + signal_integrated
    X_bg ~ Erlang(k=N, theta=mean_power)

    signal_integrated depends on Swerling (see module docstring).
    """
    sw = swerling.lower().strip()
    alpha = ca_cfar_alpha(pfa=pfa, n_ref=n_ref)
    thr = float(alpha) * (n_pulses * mean_power)

    snr_lin = 10.0 ** (float(snr_db) / 10.0)
    mean_sig = float(snr_lin) * float(mean_power)  # per pulse

    if sw == "swerling0":
        x_needed = thr - (n_pulses * mean_sig)
        return erlang_tail_prob(x_needed, k=n_pulses, theta=mean_power)

    k_v, theta_v, is_sum = _swerling_signal_v_dist(swerling=sw, n_pulses=n_pulses)
    b = (n_pulses * mean_sig) if (not is_sum) else mean_sig

    if b <= 0.0:
        return erlang_tail_prob(thr, k=n_pulses, theta=mean_power)

    v0 = thr / b
    p_v_ge = erlang_tail_prob(v0, k=k_v, theta=theta_v)

    def integrand(v: float) -> float:
        y = thr - b * v
        return erlang_tail_prob(y, k=n_pulses, theta=mean_power) * erlang_pdf(v, k=k_v, theta=theta_v)

    if v0 <= 0.0:
        return 1.0

    integ = adaptive_simpson(integrand, 0.0, float(v0), eps=1e-8, max_depth=22)
    pd = p_v_ge + integ
    return float(min(1.0, max(0.0, pd)))


# -----------------------------
# YAML IO (PyYAML)
# -----------------------------


def load_yaml(path: Path) -> Dict[str, Any]:
    try:
        import yaml  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise SystemExit("[ERROR] PyYAML not available. Install pyyaml or use repo loader.") from exc
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def dump_json(path: Path, obj: Dict[str, Any]) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


# -----------------------------
# Contract checks
# -----------------------------


def _fail(msg: str) -> None:
    raise SystemExit(f"[ERROR] {msg}")


def _require_bool(d: Dict[str, Any], path: str, default: bool = False) -> bool:
    # shallow helper for hetero.enabled etc
    if path not in d:
        return bool(default)
    v = d[path]
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)) and v in (0, 1):
        return bool(v)
    _fail(f"{path} must be boolean, got {v!r}")
    return False  # unreachable


def _check_contract(cfg: Dict[str, Any]) -> None:
    mc = cfg.get("monte_carlo", {}) or {}
    bg = mc.get("background", {}) or {}
    pd_cfg = mc.get("pd", {}) or {}

    if str(mc.get("task", "")).lower().strip() != "pd":
        _fail("This comparison script requires monte_carlo.task == 'pd'")

    if str(mc.get("detector", "")).lower().strip() != "ca_cfar_independent":
        _fail("This comparison script currently models only detector=ca_cfar_independent")

    if str(bg.get("model", "")).lower().strip() != "exponential":
        _fail("This comparison script currently models only background.model == 'exponential'")

    hetero = bg.get("hetero", {}) or {}
    if isinstance(hetero, dict) and _require_bool(hetero, "enabled", default=False):
        _fail("This comparison script requires background.hetero.enabled == false (hetero is out-of-model)")

    if str(pd_cfg.get("integration", "")).lower().strip() != "noncoherent":
        _fail("This comparison script requires pd.integration == 'noncoherent' (power-domain)")

    sw = str(pd_cfg.get("swerling", "")).lower().strip()
    if sw not in {"swerling0", "swerling1", "swerling2", "swerling3", "swerling4"}:
        _fail(f"pd.swerling must be one of swerling0..swerling4, got {sw!r}")


# -----------------------------
# Main
# -----------------------------


@dataclass(frozen=True)
class SweepPoint:
    snr_db: float
    pd_mc: float
    ci_low: float
    ci_high: float
    pd_model: float
    abs_err: float
    inside_ci: bool


def main() -> int:
    ap = argparse.ArgumentParser(description="Compare Pd model vs Monte Carlo over SNR grid (plot+CSV+JSON).")
    ap.add_argument("--case", required=True, help="Path to case YAML (contains monte_carlo + sweep.snr_db_grid).")
    ap.add_argument("--seed", type=int, default=int(os.environ.get("SEED", "123")))
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--out-tag", default=None, help="Optional override for output comparison tag directory.")
    args = ap.parse_args()

    case_path = Path(args.case)
    cfg = load_yaml(case_path)

    _check_contract(cfg)

    mc = cfg.get("monte_carlo", {}) or {}
    bg = mc.get("background", {}) or {}
    pd_cfg = mc.get("pd", {}) or {}
    sweep = cfg.get("sweep", {}) or {}

    snr_grid = sweep.get("snr_db_grid", None)
    if not isinstance(snr_grid, list) or not snr_grid:
        _fail("case must define sweep.snr_db_grid as a non-empty list")

    # Required MC fields
    pfa = float(mc["pfa"])
    n_trials = int(mc["n_trials"])
    n_ref = int(mc.get("n_ref", 32))
    mean_power = float(bg["mean_power"])
    n_pulses = int(pd_cfg["n_pulses"])
    swerling = str(pd_cfg.get("swerling", "")).lower().strip()

    # Output directory
    tag = args.out_tag or f"demo_pd_model_vs_mc_{swerling}"
    out_dir = Path("results/comparisons") / f"{tag}__seed{args.seed}"
    if out_dir.exists() and not args.overwrite:
        _fail(f"Output dir exists (use --overwrite): {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    points: List[SweepPoint] = []

    for snr_db in snr_grid:
        snr_db_f = float(snr_db)

        # Deep copy config and override only pd.snr_db + seed
        cfg_point = json.loads(json.dumps(cfg))
        cfg_point["monte_carlo"]["seed"] = int(args.seed)
        cfg_point["monte_carlo"]["pd"]["snr_db"] = float(snr_db_f)

        raw = run_monte_carlo(cfg_point, seed=int(args.seed))
        pd_metrics = _extract_pd_metrics(raw)

        pd_mc = float(pd_metrics.get("pd_empirical"))
        n_tr = int(pd_metrics.get("n_trials"))

        ci_engine = _extract_wilson_ci_from_metrics(pd_metrics)
        if ci_engine is not None:
            ci_low, ci_high = ci_engine
        else:
            ci_low, ci_high = wilson_ci(pd_mc, n_tr)

        pd_mod = pd_model(
            swerling=swerling,
            snr_db=snr_db_f,
            mean_power=mean_power,
            n_pulses=n_pulses,
            pfa=pfa,
            n_ref=n_ref,
        )

        abs_err = abs(pd_mod - pd_mc)
        inside = (ci_low <= pd_mod <= ci_high)

        points.append(
            SweepPoint(
                snr_db=snr_db_f,
                pd_mc=pd_mc,
                ci_low=ci_low,
                ci_high=ci_high,
                pd_model=pd_mod,
                abs_err=abs_err,
                inside_ci=bool(inside),
            )
        )

        print(
            f"[OK] {swerling}  SNR={snr_db_f:>6.1f} dB  Pd_mc={pd_mc:.6g}  "
            f"CI=[{ci_low:.6g},{ci_high:.6g}]  Pd_model={pd_mod:.6g}  |Δ|={abs_err:.3g}  inCI={inside}"
        )

    # CSV
    csv_path = out_dir / "comparison.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["snr_db", "pd_model", "pd_mc", "ci_wilson_low", "ci_wilson_high", "abs_error", "model_inside_ci"])
        for p in points:
            w.writerow([p.snr_db, p.pd_model, p.pd_mc, p.ci_low, p.ci_high, p.abs_err, int(p.inside_ci)])

    # JSON summary
    max_err = max(p.abs_err for p in points) if points else None
    frac_inside = float(sum(1 for p in points if p.inside_ci) / max(len(points), 1))

    payload = {
        "case": str(case_path),
        "seed": int(args.seed),
        "tag": tag,
        "swerling": swerling,
        "pfa": pfa,
        "n_ref": n_ref,
        "n_trials_per_point": n_trials,
        "n_pulses": n_pulses,
        "background": {"model": str(bg.get("model")), "mean_power": mean_power, "hetero": bg.get("hetero")},
        "summary": {
            "max_abs_error": max_err,
            "fraction_model_inside_wilson_ci": frac_inside,
        },
        "rows": [
            {
                "snr_db": p.snr_db,
                "pd_model": p.pd_model,
                "pd_mc": p.pd_mc,
                "ci_wilson_low": p.ci_low,
                "ci_wilson_high": p.ci_high,
                "abs_error": p.abs_err,
                "model_inside_ci": p.inside_ci,
            }
            for p in points
        ],
    }
    dump_json(out_dir / "comparison.json", payload)

    x = np.array([p.snr_db for p in points], dtype=float)
    y_mc = np.array([p.pd_mc for p in points], dtype=float)
    y_lo = np.array([p.ci_low for p in points], dtype=float)
    y_hi = np.array([p.ci_high for p in points], dtype=float)
    y_mod = np.array([p.pd_model for p in points], dtype=float)
    y_err = np.abs(y_mod - y_mc)

    fig, (ax_pd, ax_err) = plt.subplots(
        2,
        1,
        sharex=True,
        figsize=(9.5, 7.0),
        gridspec_kw={"height_ratios": [2.1, 1.0]},
    )

    ax_pd.plot(x, y_mod, marker="o", linewidth=2.0, label="Analytic Pd model")
    ax_pd.plot(x, y_mc, marker="s", linestyle="--", linewidth=2.0, label="Monte Carlo Pd")
    ax_pd.fill_between(x, y_lo, y_hi, alpha=0.18, label="Wilson 95% CI (MC)")

    y_max = float(np.nanmax([np.nanmax(y_mod), np.nanmax(y_mc), np.nanmax(y_hi)]))
    ax_pd.set_ylim(-0.02, min(1.0, max(0.12, 1.25 * y_max)))
    ax_pd.set_ylabel("Pd")
    ax_pd.set_title(f"Pd Model Validation — {swerling} (seed={args.seed})")
    ax_pd.grid(True, linestyle="--", alpha=0.35)
    ax_pd.legend(loc="best")

    ax_err.plot(x, np.maximum(y_err, 1e-12), marker="o", linewidth=2.0)
    ax_err.set_yscale("log")
    ax_err.set_xlabel("SNR [dB]")
    ax_err.set_ylabel("|ΔPd|")
    ax_err.set_title("Absolute model-vs-Monte-Carlo residual")
    ax_err.grid(True, which="both", linestyle="--", alpha=0.35)

    fig.tight_layout()

    png_path = out_dir / f"pd_model_vs_mc_validation_{swerling}.png"
    fig.savefig(png_path, dpi=200)
    plt.close(fig)

    pretty = Path("${PROJECT_ROOT}") / out_dir
    print("[OK] Wrote artifacts:")
    print(f"  {pretty / 'comparison.csv'}")
    print(f"  {pretty / 'comparison.json'}")
    print(f"  {pretty / png_path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())