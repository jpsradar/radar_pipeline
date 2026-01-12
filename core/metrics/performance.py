"""
core/metrics/performance.py

Performance reporting utilities for the radar pipeline (system-engineering grade).

Purpose
-------
This module converts raw engine outputs (e.g., model_based / monte_carlo / signal_level)
into human-friendly, recruiter-ready artifacts:

- A structured SNR budget breakdown (linear and dB) at one or more ranges.
- Detection performance summaries: Pd(R) for multiple detector requirements.
- Explicit false-alarm rates (FAR) derived from Pfa and trial counts (cells × Doppler × beams × dwells/s).
- Report-ready tables and narrative bullets that explain "if you change X, Y happens, because Z".

Design principles
-----------------
- Deterministic and diff-friendly output (stable ordering, explicit units).
- Clear separation between:
  (1) physical budget terms (Pt/G/L/NF/B/temperature),
  (2) detector terms (Pfa/Pd/integration),
  (3) trial counting terms (what constitutes an "attempt").
- No hardcoded absolute paths. All file outputs use user-provided paths or are returned as strings.

Scope (V1)
----------
- Works with the existing V1 schema sections:
  radar, antenna, receiver, target, environment, geometry, detection, metrics.
- Computes a simplified monostatic radar equation SNR budget:
    SNR(R) = Pr(R) / (k*T*B*F)   (losses included)
  where Pr(R) uses Pt, Gt, Gr, λ, σ, R^4, and system losses.
- FAR is computed from the count model in core/geometry/counts.py if available,
  otherwise falls back to a conservative explicit approximation.

Inputs
------
- cfg (dict): validated case config.
- metrics (dict): metrics.json produced by an engine (optional for budget-only use).

Outputs
-------
Primary programmatic outputs:
- build_performance_summary(cfg, metrics) -> dict
- render_html_summary(summary_dict) -> str (self-contained HTML fragment)

CLI usage
---------
Budget-only (no metrics.json):
    python -m core.metrics.performance --case configs/cases/demo_pd_noise.yaml --budget --ranges 10000,20000

Summarize a run directory:
    python -m core.metrics.performance --case configs/cases/demo_pd_noise.yaml \
        --metrics results/cases/<run>/metrics.json --html out/report_fragment.html

Dependencies
------------
- Standard library + NumPy.
- Optionally uses:
    core.budgets.radar_equation (if present)
    core.geometry.counts       (if present)
  but includes internal fallbacks so the file remains self-contained.

Notes for "engineer-adult" reporting
-----------------------------------
This module is intentionally verbose in its returned summary: it is meant to feed
a recruiter-facing HTML report generator, not to be a minimal math helper.

"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

# Physical constant (kept local to make this file self-contained).
_K_BOLTZMANN = 1.380649e-23  # J/K


# ---------------------------------------------------------------------
# Utilities: dB helpers (self-contained)
# ---------------------------------------------------------------------

def lin_to_db_power(x: float) -> float:
    """Convert linear power ratio to dB (10*log10)."""
    x = float(x)
    if x <= 0.0 or not math.isfinite(x):
        return float("-inf")
    return 10.0 * math.log10(x)


def db_to_lin_power(db: float) -> float:
    """Convert dB to linear power ratio."""
    return 10.0 ** (float(db) / 10.0)


def _as_float(x: Any, name: str) -> float:
    try:
        v = float(x)
    except Exception as exc:
        raise ValueError(f"{name} must be numeric, got {x!r}") from exc
    if not math.isfinite(v):
        raise ValueError(f"{name} must be finite, got {v}")
    return v


def _get_section(cfg: Dict[str, Any], key: str) -> Dict[str, Any]:
    v = cfg.get(key, None)
    if not isinstance(v, dict):
        raise ValueError(f"cfg['{key}'] must be a dict (missing or invalid)")
    return v


# ---------------------------------------------------------------------
# Budget model (monostatic)
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class BudgetTerm:
    name: str
    kind: str  # "gain" | "loss" | "noise" | "signal"
    linear: float
    db: float
    units: str
    notes: str


def _wavelength_m(fc_hz: float) -> float:
    c = 299_792_458.0
    return c / float(fc_hz)


def received_power_w_monostatic(
    *,
    fc_hz: float,
    tx_power_w: float,
    gain_tx_db: float,
    gain_rx_db: float,
    rcs_sqm: float,
    range_m: float,
    system_losses_db: float = 0.0,
) -> float:
    """
    Monostatic radar equation (received power):

        Pr = Pt * Gt * Gr * (λ^2 * σ) / ( (4π)^3 * R^4 * L )

    where L is the (linear) system loss factor.
    """
    if range_m <= 0.0 or not math.isfinite(range_m):
        raise ValueError(f"range_m must be finite and > 0, got {range_m}")

    lam = _wavelength_m(fc_hz)
    gt = db_to_lin_power(gain_tx_db)
    gr = db_to_lin_power(gain_rx_db)
    losses = db_to_lin_power(system_losses_db)

    num = float(tx_power_w) * gt * gr * (lam ** 2) * float(rcs_sqm)
    den = ((4.0 * math.pi) ** 3) * (float(range_m) ** 4) * float(losses)
    return float(num / den)


def noise_power_w(
    *,
    bw_hz: float,
    nf_db: float,
    temperature_k: float = 290.0,
) -> float:
    """
    Receiver input noise power (kTB * F).
    """
    if bw_hz <= 0.0 or not math.isfinite(bw_hz):
        raise ValueError(f"bw_hz must be finite and > 0, got {bw_hz}")
    if temperature_k <= 0.0 or not math.isfinite(temperature_k):
        raise ValueError(f"temperature_k must be finite and > 0, got {temperature_k}")

    f = db_to_lin_power(nf_db)
    return float(_K_BOLTZMANN * float(temperature_k) * float(bw_hz) * float(f))


def snr_linear_from_budget(pr_w: float, n_w: float) -> float:
    if n_w <= 0.0 or not math.isfinite(n_w):
        raise ValueError(f"noise power must be finite and > 0, got {n_w}")
    if pr_w < 0.0 or not math.isfinite(pr_w):
        raise ValueError(f"received power must be finite and >= 0, got {pr_w}")
    return float(pr_w / n_w)


def build_snr_budget(
    cfg: Dict[str, Any],
    *,
    range_m: float,
) -> Dict[str, Any]:
    """
    Build a traceable SNR budget at a single range.

    Returns a dict with:
    - terms: list[BudgetTerm as dict]
    - received_power_w / dbw
    - noise_power_w / dbw
    - snr_lin / snr_db
    """
    radar = _get_section(cfg, "radar")
    ant = _get_section(cfg, "antenna")
    rx = _get_section(cfg, "receiver")
    tgt = _get_section(cfg, "target")
    env = cfg.get("environment", {}) if isinstance(cfg.get("environment", {}), dict) else {}

    fc_hz = _as_float(radar.get("fc_hz"), "radar.fc_hz")
    pt_w = _as_float(radar.get("tx_power_w"), "radar.tx_power_w")

    gt_db = _as_float(ant.get("gain_tx_db", 0.0), "antenna.gain_tx_db")
    gr_db = _as_float(ant.get("gain_rx_db", gt_db), "antenna.gain_rx_db")

    bw_hz = _as_float(rx.get("bw_hz"), "receiver.bw_hz")
    nf_db = _as_float(rx.get("nf_db", 0.0), "receiver.nf_db")
    t_k = _as_float(rx.get("temperature_k", 290.0), "receiver.temperature_k")

    sigma = _as_float(tgt.get("rcs_sqm"), "target.rcs_sqm")
    sys_loss_db = _as_float(env.get("system_losses_db", 0.0), "environment.system_losses_db")

    pr = received_power_w_monostatic(
        fc_hz=fc_hz,
        tx_power_w=pt_w,
        gain_tx_db=gt_db,
        gain_rx_db=gr_db,
        rcs_sqm=sigma,
        range_m=float(range_m),
        system_losses_db=sys_loss_db,
    )
    n = noise_power_w(bw_hz=bw_hz, nf_db=nf_db, temperature_k=t_k)
    snr_lin = snr_linear_from_budget(pr, n)

    terms: List[BudgetTerm] = [
        BudgetTerm("Pt", "signal", pt_w, lin_to_db_power(pt_w), "W", "Transmit peak/average power as modeled."),
        BudgetTerm("Gt", "gain", db_to_lin_power(gt_db), float(gt_db), "dB", "Transmit antenna gain."),
        BudgetTerm("Gr", "gain", db_to_lin_power(gr_db), float(gr_db), "dB", "Receive antenna gain."),
        BudgetTerm("sigma", "signal", sigma, lin_to_db_power(sigma), "m^2", "Target RCS (deterministic in V1)."),
        BudgetTerm("lambda^2", "signal", _wavelength_m(fc_hz) ** 2, lin_to_db_power(_wavelength_m(fc_hz) ** 2), "m^2", "Wavelength term."),
        BudgetTerm("R^4", "loss", (float(range_m) ** 4), lin_to_db_power(float(range_m) ** 4), "m^4", "Geometric spreading (monostatic)."),
        BudgetTerm("system_losses", "loss", db_to_lin_power(sys_loss_db), float(sys_loss_db), "dB", "Aggregate system losses factor L."),
        BudgetTerm("kTB", "noise", _K_BOLTZMANN * t_k * bw_hz, lin_to_db_power(_K_BOLTZMANN * t_k * bw_hz), "W", "Thermal noise floor."),
        BudgetTerm("NF", "noise", db_to_lin_power(nf_db), float(nf_db), "dB", "Receiver noise figure."),
    ]

    return {
        "range_m": float(range_m),
        "received_power_w": float(pr),
        "received_power_dbw": float(lin_to_db_power(max(pr, np.finfo(float).tiny))),
        "noise_power_w": float(n),
        "noise_power_dbw": float(lin_to_db_power(max(n, np.finfo(float).tiny))),
        "snr_lin": float(snr_lin),
        "snr_db": float(lin_to_db_power(max(snr_lin, np.finfo(float).tiny))),
        "terms": [t.__dict__ for t in terms],
        "assumptions": {
            "monostatic_radar_equation": True,
            "snr_defined_as_pr_over_ktbf": True,
            "rcs_is_deterministic_in_v1": True,
        },
    }


# ---------------------------------------------------------------------
# FAR counting (attempt model)
# ---------------------------------------------------------------------

def estimate_trials_per_second(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Estimate how many detection "attempts" occur per second.

    Preferred path:
    - Use core.geometry.counts if present (single source of truth).

    Fallback approximation (explicit):
    - attempts/s = n_range_bins * n_doppler_bins * beams_per_scan * scans_per_second
      where scans_per_second ~= 1 / (beams_per_scan * dwell_time_s) if available.

    Returns a dict with counts and notes.
    """
    geom = cfg.get("geometry", {}) if isinstance(cfg.get("geometry", {}), dict) else {}
    radar = cfg.get("radar", {}) if isinstance(cfg.get("radar", {}), dict) else {}
    det = cfg.get("detection", {}) if isinstance(cfg.get("detection", {}), dict) else {}

    # Try to import canonical count model
    try:
        from core.geometry.counts import estimate_attempts_per_second  # type: ignore

        out = estimate_attempts_per_second(cfg)  # expected to return dict
        out = out if isinstance(out, dict) else {"attempts_per_second": float(out)}
        out.setdefault("source", "core.geometry.counts.estimate_attempts_per_second")
        return out
    except Exception:
        pass

    n_r = int(geom.get("n_range_bins", 256) or 256)
    n_d = int(geom.get("n_doppler_bins", 64) or 64)
    beams = int(geom.get("beams_per_scan", 1) or 1)

    # Very conservative scan rate estimate:
    # - If dwell_time_s exists, scans/s ~ 1/(beams*dwell)
    # - else assume 1 scan/s (explicit and conservative)
    dwell_s = None
    if "dwell_time_s" in geom:
        try:
            dwell_s = float(geom["dwell_time_s"])
        except Exception:
            dwell_s = None

    if dwell_s is not None and math.isfinite(dwell_s) and dwell_s > 0.0:
        scans_s = 1.0 / (float(beams) * float(dwell_s))
        scan_note = "Derived from geometry.dwell_time_s and beams_per_scan."
    else:
        scans_s = 1.0
        scan_note = "Fallback assumption: 1 scan/s (no dwell_time_s provided)."

    attempts_s = float(n_r) * float(n_d) * float(beams) * float(scans_s)

    return {
        "attempts_per_second": attempts_s,
        "n_range_bins": int(n_r),
        "n_doppler_bins": int(n_d),
        "beams_per_scan": int(beams),
        "scans_per_second": float(scans_s),
        "notes": [
            "Fallback attempt model used (core.geometry.counts not available or failed import).",
            scan_note,
            "Assumption: one detection trial per RD cell per beam per scan.",
        ],
        "source": "performance.py:fallback_attempt_model",
        "detector_context": {
            "pfa": det.get("pfa", None),
            "integration": det.get("integration", None),
            "n_pulses": det.get("n_pulses", None),
            "prf_hz": radar.get("prf_hz", None),
        },
    }


def far_from_pfa(cfg: Dict[str, Any], *, pfa: float) -> Dict[str, Any]:
    """
    FAR = Pfa * (attempts per second).

    Returns a dict with:
    - far_per_second
    - attempts_per_second
    - attempt model explanation
    """
    counts = estimate_trials_per_second(cfg)
    attempts_s = float(counts.get("attempts_per_second", float("nan")))
    if not math.isfinite(attempts_s) or attempts_s < 0.0:
        raise ValueError(f"attempts_per_second must be finite and >= 0, got {attempts_s}")

    p = float(pfa)
    if not (0.0 < p < 1.0):
        raise ValueError(f"pfa must be in (0,1), got {pfa}")

    return {
        "pfa": float(p),
        "attempts_per_second": float(attempts_s),
        "far_per_second": float(p * attempts_s),
        "attempt_model": counts,
    }


# ---------------------------------------------------------------------
# Summary builders (report-facing)
# ---------------------------------------------------------------------

def build_performance_summary(cfg: Dict[str, Any], metrics: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Build a recruiter-facing performance summary.

    This function returns a structured dict suitable for:
    - HTML rendering
    - JSON serialization
    - downstream tradeoff dashboards
    """
    ranges: List[float] = []

    # Prefer explicit ranges from cfg.metrics.ranges_m, else fall back to scenario.range_m if present.
    cfg_metrics = cfg.get("metrics", {}) if isinstance(cfg.get("metrics", {}), dict) else {}
    if isinstance(cfg_metrics.get("ranges_m", None), list) and cfg_metrics["ranges_m"]:
        ranges = [float(x) for x in cfg_metrics["ranges_m"]]
    else:
        scenario = cfg.get("scenario", {}) if isinstance(cfg.get("scenario", {}), dict) else {}
        if "range_m" in scenario:
            ranges = [float(scenario["range_m"])]

    if not ranges:
        ranges = [10_000.0]  # explicit default

    budgets = [build_snr_budget(cfg, range_m=r) for r in ranges]

    det = cfg.get("detection", {}) if isinstance(cfg.get("detection", {}), dict) else {}
    pfa = det.get("pfa", None)

    far = None
    if pfa is not None:
        try:
            far = far_from_pfa(cfg, pfa=float(pfa))
        except Exception as exc:
            far = {"error": str(exc), "pfa": pfa}

    engine = None
    if isinstance(metrics, dict):
        engine = metrics.get("engine", None)

    # Extract Pd(R) if present (model_based output convention in this repo).
    pd_block = None
    if isinstance(metrics, dict):
        d = metrics.get("detection", None)
        if isinstance(d, dict) and isinstance(d.get("pd", None), list):
            pd_block = {
                "pd": [float(x) for x in d["pd"]],
                "pfa": d.get("pfa", det.get("pfa", None)),
                "integration": d.get("integration", det.get("integration", None)),
                "n_pulses": d.get("n_pulses", det.get("n_pulses", None)),
            }

    # Narrative bullets: the “adult engineer” layer.
    bullets: List[str] = []
    bullets.append("SNR budget is computed explicitly from the monostatic radar equation and k·T·B·F noise model.")
    if pfa is not None and far is not None and "far_per_second" in far:
        bullets.append(
            f"False-alarm rate is not a vibe: FAR = Pfa × attempts/s ≈ {far['far_per_second']:.3g}/s "
            f"for this scan/cell model."
        )
    bullets.append(
        "Ranges are evaluated at user-provided metrics.ranges_m (preferred) or scenario.range_m; "
        "default is 10 km if neither is provided."
    )
    bullets.append(
        "If recruiters read one thing: show what constitutes a trial (RD cell × Doppler bin × beam × scan rate) "
        "and how that drives FAR."
    )

    return {
        "engine": engine,
        "ranges_m": ranges,
        "snr_budgets": budgets,
        "far": far,
        "pd_block": pd_block,
        "narrative": {
            "bullets": bullets,
            "assumptions": {
                "trial_definition": "one detection decision per RD cell per beam per scan",
                "range_to_bin_mapping": "not modeled in V1 budgets (budget uses physical range directly)",
            },
        },
    }


def render_html_summary(summary: Dict[str, Any], *, title: str = "Radar Performance Summary (V1)") -> str:
    """
    Render a clean, self-contained HTML fragment (no external CSS/JS).
    This is intended to be embedded into a larger report.html.
    """
    def esc(s: Any) -> str:
        return (
            str(s)
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
        )

    snr_rows = []
    for b in summary.get("snr_budgets", []):
        snr_rows.append(
            f"<tr><td>{esc(b.get('range_m'))}</td>"
            f"<td>{esc(round(float(b.get('received_power_dbw', 0.0)), 2))}</td>"
            f"<td>{esc(round(float(b.get('noise_power_dbw', 0.0)), 2))}</td>"
            f"<td><b>{esc(round(float(b.get('snr_db', 0.0)), 2))}</b></td></tr>"
        )
    snr_table = (
        "<table>"
        "<thead><tr><th>Range (m)</th><th>Pr (dBW)</th><th>N (dBW)</th><th>SNR (dB)</th></tr></thead>"
        f"<tbody>{''.join(snr_rows)}</tbody></table>"
    )

    far = summary.get("far", None)
    far_html = "<p><i>No FAR computed (detection.pfa not provided).</i></p>"
    if isinstance(far, dict) and "far_per_second" in far:
        far_html = (
            "<table><thead><tr><th>Pfa</th><th>Attempts/s</th><th>FAR (/s)</th></tr></thead><tbody>"
            f"<tr><td>{esc(far.get('pfa'))}</td>"
            f"<td>{esc(round(float(far.get('attempts_per_second', 0.0)), 3))}</td>"
            f"<td><b>{esc(round(float(far.get('far_per_second', 0.0)), 6))}</b></td></tr>"
            "</tbody></table>"
        )
    elif isinstance(far, dict) and "error" in far:
        far_html = f"<p><b>FAR error:</b> {esc(far.get('error'))}</p>"

    bullets = summary.get("narrative", {}).get("bullets", [])
    bullets_html = "<ul>" + "".join(f"<li>{esc(x)}</li>" for x in bullets) + "</ul>"

    # Minimal CSS: readable, recruiter-friendly.
    css = """
    <style>
      body { font-family: -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, Helvetica, Arial, sans-serif; line-height: 1.35; }
      h2 { margin: 0.2rem 0 0.6rem 0; }
      h3 { margin: 1.2rem 0 0.4rem 0; }
      table { border-collapse: collapse; width: 100%; margin: 0.4rem 0 0.8rem 0; }
      th, td { border: 1px solid #ddd; padding: 8px; text-align: left; font-size: 0.95rem; }
      th { background: #f6f7f9; }
      code, pre { background: #f6f7f9; padding: 2px 4px; border-radius: 4px; }
      .muted { color: #666; }
      .card { border: 1px solid #e3e5e8; border-radius: 8px; padding: 12px 14px; margin: 10px 0; }
    </style>
    """

    html = f"""
    {css}
    <div class="card">
      <h2>{esc(title)}</h2>
      <div class="muted">Engine: {esc(summary.get('engine'))}</div>
      <h3>SNR budget at requested ranges</h3>
      {snr_table}
      <h3>False Alarm Rate (FAR)</h3>
      {far_html}
      <h3>Engineering takeaways</h3>
      {bullets_html}
    </div>
    """
    return html.strip()


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_case_yaml_or_json(path: Path) -> Dict[str, Any]:
    # self-contained loader: JSON is native; YAML requires PyYAML.
    if path.suffix.lower() in (".json",):
        return _read_json(path)
    if path.suffix.lower() in (".yaml", ".yml"):
        try:
            import yaml  # type: ignore
        except Exception as exc:
            raise RuntimeError("YAML case file requires PyYAML. Install with: pip install pyyaml") from exc
        return yaml.safe_load(path.read_text(encoding="utf-8"))
    raise ValueError(f"Unsupported case file extension: {path.suffix}")


def _parse_ranges(s: str) -> List[float]:
    parts = [p.strip() for p in str(s).split(",") if p.strip()]
    out: List[float] = []
    for p in parts:
        v = float(p)
        if not math.isfinite(v) or v <= 0.0:
            raise ValueError(f"Invalid range value: {p!r}")
        out.append(v)
    return out


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(prog="core.metrics.performance", description="Build performance summaries (budget/Pd/FAR).")
    ap.add_argument("--case", required=True, help="Path to YAML/JSON case config.")
    ap.add_argument("--metrics", default=None, help="Optional metrics.json to include detection summaries.")
    ap.add_argument("--ranges", default=None, help="Override ranges (comma-separated), e.g. 10000,20000.")
    ap.add_argument("--budget", action="store_true", help="Budget-only mode (ignore metrics).")
    ap.add_argument("--json", dest="json_out", default=None, help="Write summary JSON to this file.")
    ap.add_argument("--html", dest="html_out", default=None, help="Write self-contained HTML fragment to this file.")
    return ap.parse_args()


def main() -> int:
    args = _parse_args()
    case_path = Path(args.case)
    cfg = _read_case_yaml_or_json(case_path)

    if args.ranges:
        cfg = dict(cfg)
        cfg["metrics"] = dict(cfg.get("metrics", {}) if isinstance(cfg.get("metrics", {}), dict) else {})
        cfg["metrics"]["ranges_m"] = _parse_ranges(args.ranges)

    metrics = None
    if args.metrics and not args.budget:
        metrics = _read_json(Path(args.metrics))

    summary = build_performance_summary(cfg, metrics)

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    if args.html_out:
        html = render_html_summary(summary)
        Path(args.html_out).write_text(html + "\n", encoding="utf-8")

    # Always print something useful for interactive use.
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())