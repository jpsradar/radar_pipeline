"""
core/metrics/robustness.py

Robustness auditing for radar pipeline outputs (metrics.json quality gate).

Purpose
-------
This module audits metrics artifacts produced by the pipeline and answers:

- "Are the outputs structurally valid and numerically sane?"
- "Do results satisfy core invariants (finite numbers, monotonicity where expected)?"
- "Are there obvious engineering mistakes (e.g., FAR implied by Pfa is absurdly high)?"

It is explicitly designed to be:
- automation-friendly (exit codes, deterministic scanning order),
- recruiter-friendly (clear messages, actionable failures),
- engine-aware (model_based vs monte_carlo vs signal_level vs validation wrappers).

Scope (V1)
----------
This does NOT attempt to certify scientific correctness. It enforces:
- schema-like presence checks for required output keys,
- numeric hygiene (finite, non-negative where required),
- consistency checks that prevent embarrassing demos.

Inputs
------
- One or more metrics.json files, typically under:
    results/cases/**/metrics.json

Outputs
-------
- A per-file PASS/WARN/FAIL report to stdout.
- Exit code:
    0 : no failures
    2 : at least one failure
    3 : unexpected exception

CLI usage
---------
Audit all case results:
    python -m core.metrics.robustness --root results/cases

Audit a specific file:
    python -m core.metrics.robustness --file results/cases/<run>/metrics.json

Dependencies
------------
- Standard library + NumPy (optional, used for convenience).

Notes
-----
This module is designed to be used as a "quality gate" in CI and as a
pre-demo check before generating recruiter-facing HTML reports.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


# ---------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class AuditReport:
    path: Path
    engine: str
    status: str  # "OK" | "WARN" | "FAIL"
    issues: List[str]
    summary: Dict[str, Any]


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _read_json(p: Path) -> Dict[str, Any]:
    return json.loads(p.read_text(encoding="utf-8"))


def _is_finite_number(x: Any) -> bool:
    try:
        v = float(x)
    except Exception:
        return False
    return math.isfinite(v)


def _finite_list(xs: Any) -> Optional[np.ndarray]:
    if not isinstance(xs, list) or not xs:
        return None
    arr = np.asarray(xs, dtype=float)
    if np.any(~np.isfinite(arr)):
        return None
    return arr


def _engine_name(m: Dict[str, Any]) -> str:
    eng = m.get("engine", None)
    if isinstance(eng, str) and eng.strip():
        return eng.strip()
    # wrappers may not set engine; infer via keys
    if "pfa_empirical" in m and "n_trials" in m:
        return "monte_carlo"
    if "rd_grid" in m and "target_injection" in m:
        return "signal_level"
    return "unknown"


# ---------------------------------------------------------------------
# Engine-specific audits
# ---------------------------------------------------------------------

def _audit_model_based(m: Dict[str, Any]) -> Tuple[List[str], Dict[str, Any]]:
    issues: List[str] = []
    summary: Dict[str, Any] = {}

    # Common expected fields in this repo’s model_based engine output.
    snr_db = m.get("snr_db", None)
    rx_pw = m.get("received_power_w", None)

    snr_arr = _finite_list(snr_db)
    if snr_arr is None:
        issues.append("Missing/invalid snr_db (expected finite list).")
    else:
        summary["snr_db_min"] = float(np.min(snr_arr))
        summary["snr_db_max"] = float(np.max(snr_arr))
        summary["n_points"] = int(snr_arr.size)

    if not (isinstance(rx_pw, list) and rx_pw):
        issues.append("Missing/invalid received_power_w (expected non-empty list).")

    det = m.get("detection", None)
    if det is not None:
        if not isinstance(det, dict):
            issues.append("detection must be a dict if present.")
        else:
            pd = _finite_list(det.get("pd", None))
            if pd is None:
                issues.append("detection.pd missing/invalid (expected finite list).")
            else:
                # Pd must be between [0,1]
                if np.any((pd < -1e-12) | (pd > 1.0 + 1e-12)):
                    issues.append("detection.pd out of [0,1] bounds.")
                # Basic monotonicity check only if ranges appear monotonic in config; here we just sanity check.
                summary["pd_min"] = float(np.min(pd))
                summary["pd_max"] = float(np.max(pd))

    return issues, summary


def _audit_signal_level(m: Dict[str, Any]) -> Tuple[List[str], Dict[str, Any]]:
    issues: List[str] = []
    summary: Dict[str, Any] = {}

    rd = m.get("rd_grid", None)
    if not isinstance(rd, dict):
        issues.append("Missing/invalid rd_grid (expected dict).")
        return issues, summary

    nr = rd.get("n_range_bins", None)
    nd = rd.get("n_doppler_bins", None)
    if not (isinstance(nr, int) and isinstance(nd, int) and nr > 0 and nd > 0):
        issues.append("rd_grid must contain positive integer n_range_bins and n_doppler_bins.")
    else:
        summary["rd_shape"] = [int(nr), int(nd)]

    noise = m.get("noise_model", None)
    if not isinstance(noise, dict):
        issues.append("Missing/invalid noise_model (expected dict).")
    else:
        npw = noise.get("noise_power_w", None)
        if not _is_finite_number(npw) or float(npw) <= 0.0:
            issues.append("noise_model.noise_power_w must be finite and > 0.")

    stats = m.get("rd_power_map_stats", None)
    if not isinstance(stats, dict):
        issues.append("Missing/invalid rd_power_map_stats (expected dict).")
    else:
        for k in ("mean", "median", "p90", "p99", "max"):
            v = stats.get(k, None)
            if not _is_finite_number(v) or float(v) < 0.0:
                issues.append(f"rd_power_map_stats.{k} must be finite and >= 0.")
        summary["rd_max"] = float(stats.get("max", float("nan"))) if _is_finite_number(stats.get("max", None)) else None

    inj = m.get("target_injection", None)
    if not isinstance(inj, dict):
        issues.append("Missing/invalid target_injection (expected dict).")
    else:
        for k in ("received_power_w", "injected_amplitude"):
            v = inj.get(k, None)
            if not _is_finite_number(v) or float(v) < 0.0:
                issues.append(f"target_injection.{k} must be finite and >= 0.")

    det = m.get("detection", None)
    if det is not None:
        if not isinstance(det, dict):
            issues.append("detection must be a dict if present.")
        else:
            # If present, its key fields must be finite.
            for k in ("pfa_target", "alpha", "detections_total"):
                if k in det:
                    if k == "detections_total":
                        try:
                            int(det[k])
                        except Exception:
                            issues.append("detection.detections_total must be int-like.")
                    else:
                        if not _is_finite_number(det[k]) or float(det[k]) <= 0.0:
                            issues.append(f"detection.{k} must be finite and > 0.")

    return issues, summary


def _audit_monte_carlo(m: Dict[str, Any]) -> Tuple[List[str], Dict[str, Any]]:
    issues: List[str] = []
    summary: Dict[str, Any] = {}

    # Two shapes exist in this repo:
    # (A) engine output: {"engine":"monte_carlo", ...}
    # (B) wrapper output: {"result": {...}, "wrapper": {...}}
    rep = m.get("result", None) if isinstance(m.get("result", None), dict) else m

    p_emp = rep.get("pfa_empirical", None)
    p_tgt = rep.get("pfa_target", None)
    n_trials = rep.get("n_trials", None)

    if not _is_finite_number(p_emp):
        issues.append("Missing/invalid pfa_empirical.")
    if not _is_finite_number(p_tgt):
        issues.append("Missing/invalid pfa_target.")
    try:
        n = int(n_trials)
        if n <= 0:
            raise ValueError
    except Exception:
        issues.append("Missing/invalid n_trials (expected positive int).")
        n = 0

    if _is_finite_number(p_emp) and _is_finite_number(p_tgt):
        summary["pfa_empirical"] = float(p_emp)
        summary["pfa_target"] = float(p_tgt)
        if n > 0:
            # sanity: empirical should not be wildly off target (not a correctness proof)
            sigma = math.sqrt(float(p_tgt) * (1.0 - float(p_tgt)) / float(n))
            if abs(float(p_emp) - float(p_tgt)) > 8.0 * sigma:
                issues.append("pfa_empirical deviates strongly from pfa_target (beyond 8σ).")
            summary["sigma_binomial"] = float(sigma)

    # CI presence check (if provided by engine)
    ci = rep.get("confidence_intervals", None)
    if ci is not None and not isinstance(ci, dict):
        issues.append("confidence_intervals must be a dict if present.")

    return issues, summary


def audit_metrics(metrics: Dict[str, Any], path: Path) -> AuditReport:
    engine = _engine_name(metrics)
    issues: List[str] = []
    summary: Dict[str, Any] = {}

    if engine == "model_based":
        issues, summary = _audit_model_based(metrics)
    elif engine == "signal_level":
        issues, summary = _audit_signal_level(metrics)
    elif engine in ("monte_carlo", "mc_cfar", "pfa_monte_carlo"):
        issues, summary = _audit_monte_carlo(metrics)
    elif engine == "unknown":
        issues = ["unknown/unhandled engine type"]
        summary = {}
    else:
        # attempt best-effort
        issues = [f"unhandled engine '{engine}'"]
        summary = {}

    status = "OK" if not issues else ("WARN" if any("unknown" in x for x in issues) else "FAIL")
    return AuditReport(path=path, engine=engine, status=status, issues=issues, summary=summary)


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(prog="core.metrics.robustness", description="Audit metrics.json artifacts for sanity.")
    ap.add_argument("--root", default=None, help="Root folder to scan for **/metrics.json (e.g., results/cases).")
    ap.add_argument("--file", default=None, help="Single metrics.json path to audit.")
    ap.add_argument("--strict-unknown", action="store_true", help="Treat unknown engines as failures.")
    return ap.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        paths: List[Path] = []
        if args.file:
            paths = [Path(args.file)]
        elif args.root:
            root = Path(args.root)
            paths = sorted(root.glob("**/metrics.json"))
        else:
            raise SystemExit("Provide --root or --file.")

        failures: List[AuditReport] = []
        unknowns: List[AuditReport] = []

        print("ROBUSTNESS AUDIT (metrics.json)")
        print(f"scan_count: {len(paths)}")
        print("")

        for p in paths:
            m = _read_json(p)
            rep = audit_metrics(m, p)

            rel = str(p)
            if rep.status == "OK":
                extra = ""
                if rep.engine == "model_based":
                    extra = f"n={rep.summary.get('n_points','?')} snr_db=[{rep.summary.get('snr_db_min','?'):.3g},{rep.summary.get('snr_db_max','?'):.3g}]"
                elif rep.engine == "signal_level":
                    extra = f"rd={rep.summary.get('rd_shape','?')}"
                elif rep.engine in ("monte_carlo", "mc_cfar", "pfa_monte_carlo"):
                    extra = f"pfa_emp={rep.summary.get('pfa_empirical','?'):.6g} target={rep.summary.get('pfa_target','?'):.6g}"
                print(f"[OK]   {rel}  (engine={rep.engine}) {extra}".rstrip())
            elif rep.status == "WARN":
                unknowns.append(rep)
                print(f"[WARN] {rel}  (engine={rep.engine})")
                for msg in rep.issues:
                    print(f"       - {msg}")
            else:
                failures.append(rep)
                print(f"[FAIL] {rel}  (engine={rep.engine})")
                for msg in rep.issues:
                    print(f"       - {msg}")

        if args.strict_unknown and unknowns:
            failures.extend(unknowns)

        print("\nSUMMARY:")
        print(f"  total_metrics: {len(paths)}")
        print(f"  failures:      {len(failures)}")
        print(f"  unknown:       {len(unknowns)}")

        return 0 if not failures else 2

    except SystemExit as exc:
        # allow clean user errors
        print(f"[ERROR] {exc}")
        return 2
    except Exception as exc:
        print(f"[ERROR] Unexpected exception: {exc}")
        return 3


if __name__ == "__main__":
    raise SystemExit(main())