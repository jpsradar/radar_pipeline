"""
validation/monte_carlo/mc_pd_detector.py

Model-vs-MonteCarlo validation for the pipeline's energy detector contract.

Intent
------
Provide a deterministic, engineering-grade validation that compares:
- Closed-form model_based detection (Pd vs range/SNR) using chi-square family stats
vs
- Empirical Monte Carlo estimates of Pd (and optionally Pfa) using the exact same
  detector contract (df/noncentrality/threshold).

Why this exists
---------------
A serious radar repo must show a clear separation between:
1) A closed-form / analytic model (fast, assumptions explicit)
2) An empirical validation path (Monte Carlo) that replicates the same detector contract
3) Quantified discrepancy + confidence intervals (showing where the model is trustworthy)

Two operating modes
-------------------
A) Contract-driven mode (recommended; used by the repo):
   - Provide --case <yaml/json>
   - The script loads the case, runs core.simulation.model_based.run_model_based_case(),
     then samples the corresponding noncentral chi-square distribution using NumPy
     to estimate Pd_empirical per range, plus Wilson 95% CI.

B) Legacy/manual mode (kept for standalone testing):
   - Provide --pfa, --n-pulses, --snr-db, etc.
   - Uses the repo threshold API if present, otherwise fails loudly.

Statistical model (contract-driven)
-----------------------------------
The detector is defined by the model_based "detection.contract" output:
- H0: chi-square with df degrees of freedom
- H1: noncentral chi-square with same df and a noncentrality rule:
    * noncoherent: nc = 2*N*SNR_per_pulse
    * coherent-like: nc = 2*SNR_total, SNR_total = N*SNR_per_pulse
- Threshold is the exact numeric threshold emitted by model_based.

Monte Carlo then samples:
  T ~ noncentral_chisquare(df, nc)
and computes Pd_emp = P(T > threshold).

Outputs
-------
Writes (or prints) a JSON wrapper:
- wrapper: metadata (timestamp, args)
- result:
    - model_based snapshot: ranges, snr_lin, pd_model, threshold, contract
    - mc estimates: pfa_h0 (optional), pd_empirical per range, Wilson 95% CI
    - discrepancy: max/mean absolute error between pd_model and pd_empirical

Exit codes
----------
0 : success
2 : invalid arguments / runtime error
3 : failed to write JSON
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import math
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from core.config.loaders import ConfigError, LoadOptions, load_case
from core.simulation.model_based import run_model_based_case


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="mc_pd_detector",
        description=(
            "Monte Carlo validation of Pd vs range/SNR using the model_based detector contract. "
            "Use --case for the repo-integrated mode."
        ),
    )

    mode = parser.add_mutually_exclusive_group(required=True)

    # Recommended mode: driven by the model_based contract from a case file.
    mode.add_argument(
        "--case",
        type=str,
        default=None,
        help="Case YAML/JSON to load (recommended mode). Example: configs/cases/demo_pd_noise.yaml",
    )

    # Legacy/manual mode: kept for standalone experiments.
    mode.add_argument(
        "--manual",
        action="store_true",
        help="Enable legacy/manual mode (requires --pfa, --n-pulses, --snr-db).",
    )

    # Shared controls
    parser.add_argument("--n-trials", type=int, default=200_000, help="Trials per point (default: 200000).")
    parser.add_argument("--seed", type=int, default=123, help="Deterministic RNG seed (default: 123).")
    parser.add_argument("--ci", action="store_true", help="If set, compute Wilson 95% confidence intervals.")
    parser.add_argument(
        "--out",
        type=str,
        default="results/validation/model_vs_mc_pd.json",
        help="Output JSON path (default: results/validation/model_vs_mc_pd.json). Use '-' to print to stdout.",
    )

    # Manual-mode parameters
    parser.add_argument("--pfa", type=float, default=None, help="[manual] Target Pfa in (0,1).")
    parser.add_argument("--n-pulses", type=int, default=None, help="[manual] Number of integrated pulses (>=1).")
    parser.add_argument(
        "--snr-db",
        type=str,
        default=None,
        help='[manual] Comma-separated SNR points in dB. Example: "--snr-db=-10,-5,0,5,10".',
    )

    args = parser.parse_args()

    if args.n_trials <= 0:
        parser.error("--n-trials must be >= 1")
    if args.case is not None:
        return args

    # Manual mode validation
    if not bool(args.manual):
        parser.error("Internal error: expected --manual when --case is not provided")

    if args.pfa is None or not (0.0 < float(args.pfa) < 1.0):
        parser.error("--pfa is required in manual mode and must be in (0,1)")
    if args.n_pulses is None or int(args.n_pulses) < 1:
        parser.error("--n-pulses is required in manual mode and must be >= 1")
    if not args.snr_db:
        parser.error("--snr-db is required in manual mode")

    # Parse SNR list
    try:
        snr_vals = [float(x.strip()) for x in str(args.snr_db).split(",") if x.strip() != ""]
    except Exception:
        parser.error(f"--snr-db must be comma-separated floats, got: {args.snr_db}")

    if len(snr_vals) == 0 or not all(math.isfinite(x) for x in snr_vals):
        parser.error(f"--snr-db contains invalid values: {snr_vals}")

    args.snr_db_list = snr_vals
    return args


# ---------------------------------------------------------------------
# Confidence interval (Wilson)
# ---------------------------------------------------------------------

def wilson_ci_95(k: int, n: int) -> Dict[str, float]:
    """Wilson score interval (95% two-sided) for a binomial proportion."""
    if n <= 0:
        raise ValueError("n must be positive for Wilson CI")
    if k < 0 or k > n:
        raise ValueError("k must satisfy 0 <= k <= n for Wilson CI")

    z = 1.959963984540054
    phat = k / n
    z2 = z * z

    denom = 1.0 + z2 / n
    center = (phat + z2 / (2.0 * n)) / denom
    half = (z / denom) * math.sqrt((phat * (1.0 - phat) / n) + (z2 / (4.0 * n * n)))
    return {"low": float(max(0.0, center - half)), "high": float(min(1.0, center + half))}


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------

def _utc_now_str() -> str:
    return _dt.datetime.now(tz=_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _pretty_path(path: Path, project_root: Path) -> str:
    """Avoid absolute paths in outputs by rendering ${PROJECT_ROOT}/... when possible."""
    try:
        rel = path.resolve().relative_to(project_root.resolve())
        return str(Path("${PROJECT_ROOT}") / rel)
    except Exception:
        return path.name


def _write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False)
    path.write_text(text + "\n", encoding="utf-8")


def _require_dict(d: Any, name: str) -> Dict[str, Any]:
    if not isinstance(d, dict):
        raise RuntimeError(f"Expected '{name}' to be a dict, got {type(d).__name__}")
    return d


def _get_contract_from_model_based(metrics: Dict[str, Any]) -> Dict[str, Any]:
    det = _require_dict(metrics.get("detection"), "metrics.detection")
    contract = _require_dict(det.get("contract"), "metrics.detection.contract")
    h0 = _require_dict(contract.get("h0"), "contract.h0")
    h1 = _require_dict(contract.get("h1"), "contract.h1")

    if str(h0.get("family")) != "chi2":
        raise RuntimeError(f"Unsupported H0 family in contract: {h0.get('family')}")
    if str(h1.get("family")) != "ncx2":
        raise RuntimeError(f"Unsupported H1 family in contract: {h1.get('family')}")

    df = int(h0.get("df"))
    if df <= 0:
        raise RuntimeError(f"Invalid contract df: {df}")

    thr = float(contract.get("threshold"))
    if not (math.isfinite(thr) and thr > 0.0):
        raise RuntimeError(f"Invalid contract threshold: {thr}")

    integration = str(contract.get("integration")).lower().strip()
    n_pulses = int(contract.get("n_pulses"))
    pfa_target = float(contract.get("pfa_target"))

    if integration not in {"noncoherent", "coherent"}:
        raise RuntimeError(f"Unsupported integration in contract: {integration}")
    if n_pulses < 1:
        raise RuntimeError(f"Invalid n_pulses in contract: {n_pulses}")
    if not (0.0 < pfa_target < 1.0):
        raise RuntimeError(f"Invalid pfa_target in contract: {pfa_target}")

    return {
        "df": df,
        "threshold": thr,
        "integration": integration,
        "n_pulses": n_pulses,
        "pfa_target": pfa_target,
        "raw": contract,
    }


# ---------------------------------------------------------------------
# Contract-driven experiment (repo mode)
# ---------------------------------------------------------------------

def run_model_vs_mc_from_case(
    *,
    case_path: Path,
    n_trials: int,
    seed: int,
    include_ci: bool,
    project_root: Path,
) -> Dict[str, Any]:
    """
    Load a case, run model_based, then run Monte Carlo that replicates the detector contract.

    Monte Carlo sampling uses NumPy's noncentral_chisquare which matches the (df, nonc)
    parameterization used by SciPy's ncx2 in model_based.
    """
    cfg = load_case(
        case_path,
        schema_dir="configs/schemas",
        schema_name="case.schema.json",
        options=LoadOptions(strict=True, resolve_paths=True, normalize_units=True),
    )

    model = run_model_based_case(cfg=cfg, seed=seed)
    contract = _get_contract_from_model_based(model)

    ranges_m = np.asarray(model.get("ranges_m", []), dtype=float)
    snr_lin = np.asarray(model.get("snr_lin", []), dtype=float)

    if ranges_m.size == 0 or snr_lin.size == 0 or ranges_m.size != snr_lin.size:
        raise RuntimeError("model_based output must include aligned ranges_m and snr_lin arrays")

    det = _require_dict(model.get("detection"), "metrics.detection")
    pd_model = np.asarray(det.get("pd", []), dtype=float)
    if pd_model.size != snr_lin.size:
        raise RuntimeError("model_based detection.pd must be aligned to snr_lin/ranges_m")

    rng = np.random.default_rng(int(seed))

    df = int(contract["df"])
    thr = float(contract["threshold"])
    statistic = contract["raw"].get("statistic", {})
    threshold_for_ncx2 = 2.0 * thr if statistic.get("domain") == "normalized_energy" else thr

    integration = str(contract["integration"])
    n_pulses = int(contract["n_pulses"])
    pfa_target = float(contract["pfa_target"])

    # Optional: empirical Pfa under H0 using the exact df and threshold.
    # Under H0, nc=0.
    t0 = rng.noncentral_chisquare(df=df, nonc=0.0, size=int(n_trials))
    k0 = int(np.sum(t0 > threshold_for_ncx2))
    pfa_emp = float(k0 / n_trials)

    # Pd under H1 per range point: nc depends on integration mode.
    pd_emp_list: List[float] = []
    ci_list: List[Dict[str, float]] = []

    for snr in snr_lin.tolist():
        snr_f = float(snr)
        if not (math.isfinite(snr_f) and snr_f >= 0.0):
            raise RuntimeError(f"Non-finite or negative snr_lin encountered: {snr_f}")

        if integration == "noncoherent":
            # Contract: nc = 2*N*SNR_per_pulse
            nonc = 2.0 * float(n_pulses) * snr_f
        else:
            # Contract: df=2, nc = 2*SNR_total, SNR_total = N*SNR_per_pulse
            nonc = 2.0 * float(n_pulses) * snr_f

        t1 = rng.noncentral_chisquare(df=df, nonc=nonc, size=int(n_trials))
        k1 = int(np.sum(t1 > threshold_for_ncx2))
        pd_emp = float(k1 / n_trials)
        pd_emp_list.append(pd_emp)

        if include_ci:
            ci_list.append(wilson_ci_95(k1, n_trials))

    pd_emp_arr = np.asarray(pd_emp_list, dtype=float)
    err_abs = np.abs(pd_emp_arr - pd_model)

    result: Dict[str, Any] = {
        "engine": "model_vs_mc_pd",
        "case": {
            "path": _pretty_path(case_path, project_root),
        },
        "model_based": {
            "ranges_m": [float(x) for x in ranges_m.tolist()],
            "snr_lin": [float(x) for x in snr_lin.tolist()],
            "snr_db": [float(10.0 * math.log10(max(float(x), 1e-300))) for x in snr_lin.tolist()],
            "pd_model": [float(x) for x in pd_model.tolist()],
            "detection": {
                "threshold": float(thr),
                "pfa_target": float(pfa_target),
                "df": int(df),
                "integration": integration,
                "n_pulses": int(n_pulses),
                "contract": contract["raw"],
            },
        },
        "monte_carlo": {
            "n_trials": int(n_trials),
            "seed": int(seed),
            "pfa_h0": {
                "exceedances": int(k0),
                "pfa_empirical": float(pfa_emp),
            },
            "pd_h1": {
                "pd_empirical": [float(x) for x in pd_emp_arr.tolist()],
            },
        },
        "discrepancy": {
            "pd_abs_err_max": float(np.max(err_abs)) if err_abs.size else 0.0,
            "pd_abs_err_mean": float(np.mean(err_abs)) if err_abs.size else 0.0,
        },
    }

    if include_ci:
        result["monte_carlo"]["pfa_h0"]["wilson_95"] = wilson_ci_95(k0, n_trials)
        result["monte_carlo"]["pd_h1"]["wilson_95"] = ci_list

    return result


# ---------------------------------------------------------------------
# Legacy/manual mode (kept for standalone testing)
# ---------------------------------------------------------------------

def run_mc_pd_detector_manual(
    *,
    pfa: float,
    n_pulses: int,
    n_trials: int,
    snr_db: List[float],
    seed: int,
    include_ci: bool,
) -> Dict[str, Any]:
    """
    Legacy manual-mode Monte Carlo using complex Gaussian samples and the repo threshold API.

    NOTE:
    This path is intentionally strict and is not the preferred repo integration path.
    """
    # Import locally to avoid coupling for the contract-driven mode.
    from core.detection import thresholds as _thresholds  # type: ignore

    fn = getattr(_thresholds, "energy_threshold_noncoherent", None)
    if fn is None or not callable(fn):
        raise RuntimeError(
            "Missing threshold API: core.detection.thresholds.energy_threshold_noncoherent(pfa, n_pulses) is required "
            "for manual mode."
        )

    if not (0.0 < pfa < 1.0):
        raise ValueError(f"pfa must be in (0,1), got {pfa}")
    if n_pulses < 1:
        raise ValueError(f"n_pulses must be >= 1, got {n_pulses}")
    if n_trials < 1:
        raise ValueError(f"n_trials must be >= 1, got {n_trials}")
    if len(snr_db) == 0:
        raise ValueError("snr_db list must be non-empty")

    thr = float(fn(float(pfa), int(n_pulses)))
    if not (math.isfinite(thr) and thr > 0.0):
        raise RuntimeError(f"Computed threshold must be finite and > 0, got {thr}")

    rng = np.random.default_rng(int(seed))
    noise_power = 1.0  # dimensionless

    # H0: estimate Pfa empirical (complex CN noise, energy sum)
    z0 = (rng.normal(0.0, math.sqrt(noise_power / 2.0), size=(n_trials, n_pulses))
          + 1j * rng.normal(0.0, math.sqrt(noise_power / 2.0), size=(n_trials, n_pulses)))
    t0 = np.sum(np.abs(z0) ** 2, axis=1)
    k0 = int(np.sum(t0 > thr))
    pfa_emp = float(k0 / n_trials)

    out: Dict[str, Any] = {
        "engine": "mc_pd_detector_manual",
        "threshold": {
            "method": "energy_threshold_noncoherent",
            "pfa_target": float(pfa),
            "n_pulses": int(n_pulses),
            "threshold": float(thr),
            "noise_power_per_pulse": float(noise_power),
        },
        "pfa_h0": {
            "n_trials": int(n_trials),
            "exceedances": int(k0),
            "pfa_empirical": float(pfa_emp),
        },
        "pd_h1": {
            "n_trials": int(n_trials),
            "snr_db": [float(x) for x in snr_db],
            "pd_empirical": [],
        },
    }

    if include_ci:
        out["pfa_h0"]["wilson_95"] = wilson_ci_95(k0, n_trials)

    pd_list: List[float] = []
    ci_list: List[Dict[str, float]] = []

    for snr_db_i in snr_db:
        snr_lin = 10.0 ** (float(snr_db_i) / 10.0)
        a = math.sqrt(max(0.0, snr_lin * noise_power))

        # Random phase per trial to avoid phase-lock artifacts.
        phi = rng.uniform(0.0, 2.0 * math.pi, size=(n_trials, 1))
        s = a * (np.cos(phi) + 1j * np.sin(phi))  # (n_trials,1)

        z1 = (rng.normal(0.0, math.sqrt(noise_power / 2.0), size=(n_trials, n_pulses))
              + 1j * rng.normal(0.0, math.sqrt(noise_power / 2.0), size=(n_trials, n_pulses)))
        y = z1 + s
        t1 = np.sum(np.abs(y) ** 2, axis=1)

        k1 = int(np.sum(t1 > thr))
        pd_emp = float(k1 / n_trials)

        pd_list.append(pd_emp)
        if include_ci:
            ci_list.append(wilson_ci_95(k1, n_trials))

    out["pd_h1"]["pd_empirical"] = [float(x) for x in pd_list]
    if include_ci:
        out["pd_h1"]["wilson_95"] = ci_list

    return out


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> int:
    args = _parse_args()
    project_root = Path.cwd().resolve()

    try:
        if args.case is not None:
            payload = run_model_vs_mc_from_case(
                case_path=Path(str(args.case)),
                n_trials=int(args.n_trials),
                seed=int(args.seed),
                include_ci=bool(args.ci),
                project_root=project_root,
            )
        else:
            # Manual mode
            snr_vals = [float(x.strip()) for x in str(args.snr_db).split(",") if x.strip() != ""]
            payload = run_mc_pd_detector_manual(
                pfa=float(args.pfa),
                n_pulses=int(args.n_pulses),
                n_trials=int(args.n_trials),
                snr_db=snr_vals,
                seed=int(args.seed),
                include_ci=bool(args.ci),
            )
    except (ConfigError, ValueError, RuntimeError) as exc:
        print(f"[ERROR] {exc}")
        return 2
    except Exception as exc:
        print(f"[ERROR] Unexpected exception: {exc}")
        return 2

    wrapper = {
        "wrapper": {
            "tool": "validation.monte_carlo.mc_pd_detector",
            "timestamp_utc": _utc_now_str(),
            "args": vars(args),
        },
        "result": payload,
    }

    if str(args.out).strip() == "-":
        print(json.dumps(wrapper, indent=2, sort_keys=True, ensure_ascii=False))
        return 0

    out_path = Path(str(args.out))
    try:
        _write_json(out_path, wrapper)
        print(f"[OK] Wrote: {_pretty_path(out_path, project_root)}")
    except Exception as exc:
        print(f"[ERROR] Failed to write JSON: {exc}")
        return 3

    return 0


if __name__ == "__main__":
    raise SystemExit(main())