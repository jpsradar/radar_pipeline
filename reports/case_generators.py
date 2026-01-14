"""
reports/case_generators.py

Standalone HTML case report generator for radar_pipeline runs.

Intent
------
Convert a single run directory (results/cases/<run_id>/) into a portable, reader-friendly
artifact:
- report.html (self-contained; plots embedded as base64)
- plots/*.png (also persisted for convenience)

The report is *pure post-processing*:
- It does not run simulations.
- It only reads pipeline outputs (metrics.json, optionally case_manifest.json).

Design principles
-----------------
- Deterministic output given the same inputs (stable plots + stable ordering).
- Fail-soft behavior: missing fields become explicit warnings in the report.
- Minimal dependencies: standard library + NumPy + Matplotlib.

Publish-safety policy
---------------------
The report must be safe to share publicly:
- No absolute filesystem paths are rendered in HTML.
- Paths are displayed as "${PROJECT_ROOT}/..." when resolvable, otherwise reduced to basenames.

Supported output "shapes" (by engine family)
--------------------------------------------
- model_based: range-dependent curves (e.g., SNR, Pd, received power)
- signal_level: RD summary statistics and detection sanity plots
- monte_carlo / pfa_monte_carlo: empirical Pfa vs target with confidence intervals
- mc_pd_detector: Pd vs SNR with confidence interval band (when available)

Non-goals
---------
- This module does not validate physical correctness of metrics.
- It does not attempt to infer missing data or synthesize unavailable arrays.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import base64
import json
import math
import textwrap
import datetime as _dt

import matplotlib.pyplot as plt
import numpy as np


# ----------------------------
# Public API
# ----------------------------

@dataclass(frozen=True)
class CaseReportPaths:
    """Resolved output paths for a case report."""
    out_dir: Path
    plots_dir: Path
    report_html: Path


def generate_case_report_html(
    *,
    case_dir: Path,
    metrics_path: Optional[Path] = None,
    manifest_path: Optional[Path] = None,
    out_dir: Optional[Path] = None,
    title: Optional[str] = None,
) -> CaseReportPaths:
    """
    Generate an HTML report for a single case run.

    Parameters
    ----------
    case_dir : Path
        Case output directory containing metrics.json (e.g., results/cases/<run>/).
    metrics_path : Path | None
        Explicit path to metrics.json. Defaults to case_dir/metrics.json.
    manifest_path : Path | None
        Optional path to case_manifest.json. Defaults to case_dir/case_manifest.json if present.
    out_dir : Path | None
        Output directory (defaults to case_dir).
    title : str | None
        Optional title override.

    Returns
    -------
    CaseReportPaths
        Paths to the generated artifacts.
    """
    case_dir = Path(case_dir).resolve()
    out_dir = Path(out_dir).resolve() if out_dir is not None else case_dir
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    mp = (Path(metrics_path).resolve() if metrics_path is not None else (case_dir / "metrics.json").resolve())
    if not mp.exists():
        raise FileNotFoundError(f"metrics.json not found: {mp}")

    metrics = _read_json(mp)

    man = None
    if manifest_path is not None:
        man_p = Path(manifest_path).resolve()
        if man_p.exists():
            man = _read_json(man_p)
    else:
        man_p = (case_dir / "case_manifest.json").resolve()
        if man_p.exists():
            man = _read_json(man_p)

    engine = _infer_engine(metrics)
    report_title = title or f"Radar Pipeline Case Report — {engine}"

    warnings: List[str] = []
    plots: List[Tuple[str, str]] = []

    # Optional cross-run validation (model vs MC)
    validation_obj = _try_load_model_vs_mc_validation(case_dir)

    summary_html = _render_summary_table(metrics=metrics, manifest=man, engine=engine, warnings=warnings)

    if engine == "model_based":
        plots.extend(_plots_model_based(metrics=metrics, plots_dir=plots_dir, warnings=warnings))
    elif engine == "signal_level":
        plots.extend(_plots_signal_level(metrics=metrics, plots_dir=plots_dir, warnings=warnings))
    elif engine in ("monte_carlo", "pfa_monte_carlo", "mc_cfar"):
        plots.extend(_plots_monte_carlo_pfa(metrics=metrics, plots_dir=plots_dir, warnings=warnings))
    elif engine == "mc_pd_detector":
        plots.extend(_plots_mc_pd_detector(metrics=metrics, plots_dir=plots_dir, warnings=warnings))
    else:
        warnings.append(f"Engine '{engine}' is not explicitly supported by the case report generator (V1).")

    validation_html = _render_validation_card(validation_obj, warnings=warnings)

    html = _render_full_html(
        title=report_title,
        engine=engine,
        summary_html=summary_html,
        validation_html=validation_html,
        plots=plots,
        warnings=warnings,
        metrics_path=mp,
        case_dir=case_dir,
        generated_utc=_dt.datetime.now(tz=_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    )

    report_html = out_dir / "report.html"
    report_html.write_text(html, encoding="utf-8")

    return CaseReportPaths(out_dir=out_dir, plots_dir=plots_dir, report_html=report_html)
    

# ----------------------------
# Engine inference
# ----------------------------

def _infer_engine(metrics: Dict[str, Any]) -> str:
    """
    Infer engine name from metrics.

    Convention:
    - Most pipeline metrics include "engine".
    - Some wrappers may nest under "result" (validation wrappers).
    """
    if isinstance(metrics.get("engine"), str):
        return str(metrics["engine"]).strip()
    res = metrics.get("result")
    if isinstance(res, dict) and isinstance(res.get("engine"), str):
        return str(res["engine"]).strip()
    # Legacy naming from monte_carlo output in your pipeline:
    if isinstance(res, dict) and isinstance(res.get("task"), str):
        t = str(res["task"]).strip()
        if t == "pfa_monte_carlo":
            return "monte_carlo"
    return "unknown"


# ----------------------------
# Plot helpers
# ----------------------------

def _plots_model_based(*, metrics: Dict[str, Any], plots_dir: Path, warnings: List[str]) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []

    ranges = _get_list(metrics, "ranges_m", nested=("metrics", "ranges_m"))
    snr_db = _get_list(metrics, "snr_db", nested=("snr_db",))
    pr_w = _get_list(metrics, "received_power_w", nested=("received_power_w",))

    if ranges is None:
        warnings.append("model_based: missing metrics.ranges_m (cannot plot vs range).")
        return out

    r = np.asarray(ranges, dtype=float)

    # SNR vs range
    if snr_db is not None:
        y = np.asarray(snr_db, dtype=float)
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(r, y, marker="o")
        ax.set_xlabel("Range (m)")
        ax.set_ylabel("SNR (dB)")
        ax.set_title("Model-Based: SNR vs Range")
        png = plots_dir / "snr_vs_range.png"
        _save_fig(fig, png)
        out.append(("SNR vs Range", _png_to_data_uri(png)))
    else:
        warnings.append("model_based: missing snr_db (skipping SNR plot).")

    # Received power vs range
    if pr_w is not None:
        y = np.asarray(pr_w, dtype=float)
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(r, y, marker="o")
        ax.set_xlabel("Range (m)")
        ax.set_ylabel("Received Power (W)")
        ax.set_title("Model-Based: Received Power vs Range")
        ax.set_yscale("log")
        png = plots_dir / "received_power_vs_range.png"
        _save_fig(fig, png)
        out.append(("Received Power vs Range (log scale)", _png_to_data_uri(png)))
    else:
        warnings.append("model_based: missing received_power_w (skipping power plot).")

    # Pd vs range (optional)
    det = metrics.get("detection")
    if isinstance(det, dict) and isinstance(det.get("pd"), list):
        pd = np.asarray(det["pd"], dtype=float)
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(r, pd, marker="o")
        ax.set_xlabel("Range (m)")
        ax.set_ylabel("Pd")
        ax.set_title("Model-Based: Pd vs Range")
        ax.set_ylim(0.0, 1.0)
        png = plots_dir / "pd_vs_range.png"
        _save_fig(fig, png)
        out.append(("Pd vs Range", _png_to_data_uri(png)))
    else:
        warnings.append("model_based: missing detection.pd (skipping Pd plot).")

    return out


def _plots_signal_level(*, metrics: Dict[str, Any], plots_dir: Path, warnings: List[str]) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []

    stats = metrics.get("rd_power_map_stats")
    if not isinstance(stats, dict):
        warnings.append("signal_level: missing rd_power_map_stats (skipping RD plots).")
        return out

    keys = ["median", "p90", "p99", "max"]
    vals = []
    for k in keys:
        v = stats.get(k)
        if v is None:
            warnings.append(f"signal_level: missing rd_power_map_stats.{k} (skipping percentile chart).")
            return out
        vals.append(float(v))

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.bar(keys, vals)
    ax.set_ylabel("RD Power (arb. units)")
    ax.set_title("Signal-Level: RD Power Summary (percentiles/max)")
    ax.set_yscale("log")
    png = plots_dir / "rd_power_summary.png"
    _save_fig(fig, png)
    out.append(("RD Power Summary (log scale)", _png_to_data_uri(png)))

    det = metrics.get("detection")
    if isinstance(det, dict):
        tpow = det.get("target_cell_power")
        tthr = det.get("target_threshold")
        if tpow is not None and tthr is not None and _finite(tpow) and _finite(tthr):
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
            ax.bar(["Target Cell Power", "CFAR Threshold"], [float(tpow), float(tthr)])
            ax.set_title("Signal-Level: Target Cell Power vs CFAR Threshold")
            ax.set_yscale("log")
            png = plots_dir / "target_vs_threshold.png"
            _save_fig(fig, png)
            out.append(("Target Cell Power vs CFAR Threshold (log scale)", _png_to_data_uri(png)))
        else:
            warnings.append("signal_level: detection present but missing target_cell_power/target_threshold (skipping CFAR plot).")
    else:
        warnings.append("signal_level: detection not present (skipping CFAR plot).")

    return out


def _plots_monte_carlo_pfa(*, metrics: Dict[str, Any], plots_dir: Path, warnings: List[str]) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []

    res = metrics.get("result") if isinstance(metrics.get("result"), dict) else metrics
    if not isinstance(res, dict):
        warnings.append("monte_carlo: unexpected metrics structure (missing dict result).")
        return out

    pfa_emp = res.get("pfa_empirical")
    pfa_tgt = res.get("pfa_target")
    ci = res.get("confidence_intervals", {}).get("wilson_95") if isinstance(res.get("confidence_intervals"), dict) else None

    if not (_finite(pfa_emp) and _finite(pfa_tgt)):
        warnings.append("monte_carlo: missing pfa_empirical/pfa_target (skipping plot).")
        return out

    low = None
    high = None
    if isinstance(ci, dict):
        low = ci.get("low")
        high = ci.get("high")

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.bar(["Target", "Empirical"], [float(pfa_tgt), float(pfa_emp)])
    ax.set_title("Monte Carlo: Pfa Target vs Empirical")
    ax.set_ylabel("Pfa")
    ax.set_yscale("log")

    if _finite(low) and _finite(high):
        y = float(pfa_emp)
        yerr = np.array([[y - float(low)], [float(high) - y]])
        ax.errorbar([1], [y], yerr=yerr, fmt="none", capsize=6)
    else:
        warnings.append("monte_carlo: missing Wilson CI (plot will not show interval).")

    png = plots_dir / "pfa_target_vs_empirical.png"
    _save_fig(fig, png)
    out.append(("Pfa Target vs Empirical (log scale)", _png_to_data_uri(png)))
    return out


def _plots_mc_pd_detector(*, metrics: Dict[str, Any], plots_dir: Path, warnings: List[str]) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    res = metrics.get("result") if isinstance(metrics.get("result"), dict) else metrics
    if not isinstance(res, dict):
        warnings.append("mc_pd_detector: unexpected metrics structure.")
        return out

    pd_h1 = res.get("pd_h1")
    if not isinstance(pd_h1, dict):
        warnings.append("mc_pd_detector: missing pd_h1 (skipping Pd plot).")
        return out

    snr_db = pd_h1.get("snr_db")
    pd_emp = pd_h1.get("pd_empirical")
    if not (isinstance(snr_db, list) and isinstance(pd_emp, list) and len(snr_db) == len(pd_emp) and len(snr_db) > 0):
        warnings.append("mc_pd_detector: invalid snr_db/pd_empirical arrays.")
        return out

    x = np.asarray(snr_db, dtype=float)
    y = np.asarray(pd_emp, dtype=float)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(x, y, marker="o")
    ax.set_title("MC Pd Detector: Pd vs SNR")
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("Pd")
    ax.set_ylim(0.0, 1.0)

    wil = pd_h1.get("wilson_95")
    if isinstance(wil, list) and len(wil) == len(x):
        low = np.asarray([float(w.get("low", np.nan)) for w in wil], dtype=float)
        high = np.asarray([float(w.get("high", np.nan)) for w in wil], dtype=float)
        if np.all(np.isfinite(low)) and np.all(np.isfinite(high)):
            ax.fill_between(x, low, high, alpha=0.2)
        else:
            warnings.append("mc_pd_detector: wilson_95 contains non-finite values (skipping CI band).")
    else:
        warnings.append("mc_pd_detector: missing wilson_95 (skipping CI band).")

    png = plots_dir / "pd_vs_snr.png"
    _save_fig(fig, png)
    out.append(("Pd vs SNR (with CI band if available)", _png_to_data_uri(png)))
    return out


def _save_fig(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _png_to_data_uri(png_path: Path) -> str:
    b = png_path.read_bytes()
    enc = base64.b64encode(b).decode("ascii")
    return f"data:image/png;base64,{enc}"


# ----------------------------
# HTML rendering
# ----------------------------

def _render_full_html(
    *,
    title: str,
    engine: str,
    summary_html: str,
    validation_html: str,
    plots: List[Tuple[str, str]],
    warnings: List[str],
    metrics_path: Path,
    case_dir: Path,
    generated_utc: str,
) -> str:
    warn_html = ""
    if warnings:
        items = "\n".join(f"<li>{_escape(w)}</li>" for w in warnings)
        warn_html = f"""
        <section class="card warn">
          <h2>Warnings / Missing Fields</h2>
          <ul>{items}</ul>
        </section>
        """

    plots_html = ""
    if plots:
        blocks = []
        for caption, uri in plots:
            blocks.append(
                f"""
                <figure class="plot">
                  <img src="{uri}" alt="{_escape(caption)}" />
                  <figcaption>{_escape(caption)}</figcaption>
                </figure>
                """
            )
        plots_html = f"""
        <section class="card">
          <h2>Plots</h2>
          <div class="plot-grid">
            {''.join(blocks)}
          </div>
        </section>
        """
    else:
        plots_html = f"""
        <section class="card">
          <h2>Plots</h2>
          <p>No plots were generated for engine <code>{_escape(engine)}</code>. See warnings above.</p>
        </section>
        """

    css = _default_css()
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>{_escape(title)}</title>
  <style>{css}</style>
</head>
<body>
  <header class="header">
    <div>
      <h1>{_escape(title)}</h1>
      <div class="meta">
        <span><b>Engine:</b> { _escape(engine) }</span>
        <span><b>Generated (UTC):</b> { _escape(generated_utc) }</span>
      </div>
    </div>
  </header>

  <main class="container">
    <section class="card">
      <h2>Run Summary</h2>
      {summary_html}
    </section>

    {validation_html}
    {warn_html}
    {plots_html}

    <section class="card">
      <h2>Raw Metrics (excerpt)</h2>
      <p class="hint">This is a compact preview. The source of truth is <code>metrics.json</code>.</p>
      <pre class="code">{_escape(_pretty_json_excerpt(metrics_path, max_chars=12000))}</pre>
    </section>
  </main>

  <footer class="footer">
    <p>Generated by <code>reports.case_generators</code> (V1)</p>
  </footer>
</body>
</html>
"""


def _render_summary_table(*, metrics: Dict[str, Any], manifest: Optional[Dict[str, Any]], engine: str, warnings: List[str]) -> str:
    rows: List[Tuple[str, str]] = []

    rows.append(("engine", engine))

    if isinstance(manifest, dict):
        rows.append(("run_name", str(manifest.get("run_name", "")) or "(missing)"))
        rows.append(("git", str(manifest.get("git", {}).get("hash", "")) if isinstance(manifest.get("git"), dict) else "(missing)"))
        rows.append(("seed", str(manifest.get("seed", "")) if "seed" in manifest else "(missing)"))
        extras = manifest.get("extras")
        if isinstance(extras, dict):
            rows.append(("engine_requested", str(extras.get("engine_requested", ""))))
            rows.append(("engine_selected", str(extras.get("engine_selected", ""))))
    else:
        warnings.append("case_manifest.json not found (summary will omit run metadata).")

    if engine == "model_based":
        ranges = _get_list(metrics, "ranges_m", nested=("metrics", "ranges_m"))
        snr_db = _get_list(metrics, "snr_db", nested=("snr_db",))

        if ranges is not None:
            rows.append(("n_ranges", str(len(ranges))))
        if snr_db is not None and len(snr_db) > 0:
            rows.append(("snr_db_min", f"{min(map(float, snr_db)):.3g}"))
            rows.append(("snr_db_max", f"{max(map(float, snr_db)):.3g}"))

        det = metrics.get("detection")
        if isinstance(det, dict):
            if isinstance(det.get("pfa_target"), (int, float)):
                rows.append(("pfa_target", f"{float(det['pfa_target']):.3g}"))
            if isinstance(det.get("threshold"), (int, float)):
                rows.append(("threshold", f"{float(det['threshold']):.6g}"))
            if isinstance(det.get("n_pulses"), int):
                rows.append(("n_pulses", str(det["n_pulses"])))
            if isinstance(det.get("integration"), str):
                rows.append(("integration", str(det["integration"])))
        else:
            warnings.append("model_based: missing detection block (cannot summarize Pfa/threshold/integration).")

    if engine == "signal_level":
        rd = metrics.get("rd_grid")
        if isinstance(rd, dict):
            rows.append(("rd_grid", f"{rd.get('n_range_bins','?')} x {rd.get('n_doppler_bins','?')}"))
        nm = metrics.get("noise_model")
        if isinstance(nm, dict) and _finite(nm.get("noise_power_w")):
            rows.append(("noise_power_w", f"{float(nm['noise_power_w']):.3e}"))
        inj = metrics.get("target_injection")
        if isinstance(inj, dict) and _finite(inj.get("snr_injected_db")):
            rows.append(("snr_injected_db", f"{float(inj['snr_injected_db']):.3g} dB"))

    if engine in ("monte_carlo", "pfa_monte_carlo", "mc_cfar"):
        res = metrics.get("result") if isinstance(metrics.get("result"), dict) else metrics
        if isinstance(res, dict):
            if _finite(res.get("pfa_target")):
                rows.append(("pfa_target", f"{float(res['pfa_target']):.3g}"))
            if _finite(res.get("pfa_empirical")):
                rows.append(("pfa_empirical", f"{float(res['pfa_empirical']):.3g}"))

    if engine == "mc_pd_detector":
        res = metrics.get("result") if isinstance(metrics.get("result"), dict) else metrics
        if isinstance(res, dict) and isinstance(res.get("threshold"), dict):
            thr = res["threshold"]
            if _finite(thr.get("threshold")):
                rows.append(("threshold", f"{float(thr['threshold']):.6g}"))
            if _finite(thr.get("pfa_target")):
                rows.append(("pfa_target", f"{float(thr['pfa_target']):.3g}"))
            if isinstance(thr.get("n_pulses"), int):
                rows.append(("n_pulses", str(thr["n_pulses"])))

    trs = "\n".join(
        f"<tr><th>{_escape(k)}</th><td>{_escape(v)}</td></tr>"
        for k, v in rows
    )
    return f"""
    <table class="table">
      <tbody>
        {trs}
      </tbody>
    </table>
    """
    
    
def _render_validation_card(validation_obj: Optional[Dict[str, Any]], *, warnings: List[str]) -> str:
    """
    Render a compact "model vs Monte Carlo" sanity card, if the artifact exists.

    This is intentionally minimal:
    - It should be stable even if the validation JSON evolves.
    - If fields are missing, we warn and still render what we can.
    """
    if validation_obj is None:
        return ""

    if not isinstance(validation_obj, dict):
        warnings.append("Validation artifact exists but is not a JSON object (skipping validation card).")
        return ""

    res = validation_obj.get("result")
    if not isinstance(res, dict):
        warnings.append("Validation artifact missing 'result' object (skipping validation card).")
        return ""

    disc = res.get("discrepancy", {})
    if not isinstance(disc, dict):
        disc = {}

    mc = res.get("monte_carlo", {})
    if not isinstance(mc, dict):
        mc = {}

    n_trials = mc.get("n_trials", None)
    pd_abs_err_max = disc.get("pd_abs_err_max", None)
    pd_abs_err_mean = disc.get("pd_abs_err_mean", None)

    def fmt(x: Any) -> str:
        try:
            xf = float(x)
            if not math.isfinite(xf):
                return "(non-finite)"
            return f"{xf:.6g}"
        except Exception:
            return "(missing)"

    rows = [
        ("artifact", "results/validation/model_vs_mc_pd.json"),
        ("n_trials_per_point", str(int(n_trials)) if isinstance(n_trials, int) else "(missing)"),
        ("pd_abs_err_max", fmt(pd_abs_err_max)),
        ("pd_abs_err_mean", fmt(pd_abs_err_mean)),
    ]

    trs = "\n".join(f"<tr><th>{_escape(k)}</th><td>{_escape(v)}</td></tr>" for k, v in rows)

    return f"""
    <section class="card">
      <h2>Model vs Monte Carlo (sanity)</h2>
      <p class="hint">
        This check validates that the closed-form detector model matches an empirical Monte Carlo estimate
        under the exact same detector contract (df / threshold / noncentrality).
      </p>
      <table class="table">
        <tbody>
          {trs}
        </tbody>
      </table>
    </section>
    """


def _default_css() -> str:
    return textwrap.dedent("""
    :root {
      --bg: #0b0d10;
      --card: #12161c;
      --text: #e8eef6;
      --muted: #a9b4c0;
      --warn: #2a1a12;
      --accent: #5fb3ff;
      --border: #263241;
      --mono: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
      --sans: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: var(--sans);
      background: var(--bg);
      color: var(--text);
    }
    .header {
      padding: 28px 20px;
      border-bottom: 1px solid var(--border);
      background: linear-gradient(180deg, #0b0d10, #0b0d10 40%, #0a0c0f);
    }
    .header h1 { margin: 0 0 10px 0; font-size: 22px; }
    .meta {
      display: flex;
      flex-wrap: wrap;
      gap: 10px 18px;
      color: var(--muted);
      font-size: 13px;
    }
    .container {
      max-width: 1100px;
      margin: 0 auto;
      padding: 18px 14px 40px 14px;
    }
    .card {
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 16px;
      margin: 14px 0;
    }
    .card h2 {
      margin: 0 0 12px 0;
      font-size: 16px;
      letter-spacing: 0.2px;
    }
    .warn { background: var(--warn); border-color: #5a2b19; }
    .table {
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
    }
    .table th, .table td {
      padding: 10px 10px;
      border-bottom: 1px solid var(--border);
      vertical-align: top;
      text-align: left;
    }
    .table th { width: 220px; color: var(--muted); font-weight: 600; }
    code { font-family: var(--mono); color: #d7e8ff; }
    .hint { color: var(--muted); font-size: 13px; }
    .code {
      font-family: var(--mono);
      font-size: 12px;
      line-height: 1.45;
      color: #dbe7f7;
      background: #0a0c10;
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 12px;
      overflow-x: auto;
      white-space: pre-wrap;
      word-break: break-word;
    }
    .plot-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(330px, 1fr));
      gap: 14px;
    }
    figure.plot {
      margin: 0;
      padding: 10px;
      border: 1px solid var(--border);
      border-radius: 12px;
      background: #0a0c10;
    }
    figure.plot img {
      width: 100%;
      height: auto;
      border-radius: 8px;
      display: block;
      border: 1px solid #1b2532;
    }
    figure.plot figcaption {
      margin-top: 8px;
      color: var(--muted);
      font-size: 12px;
    }
    .footer {
      padding: 18px 14px;
      color: var(--muted);
      border-top: 1px solid var(--border);
      text-align: center;
      font-size: 12px;
    }
    """)


# ----------------------------
# Path display helpers (publish-safe)
# ----------------------------

def _pretty_path(path: Path, project_root: Path) -> str:
    """
    Render a path without leaking absolute filesystem locations.

    - If `path` is inside `project_root`: "${PROJECT_ROOT}/<relative>"
    - Otherwise: "<basename>"
    """
    try:
        rel = path.resolve().relative_to(project_root.resolve())
        return str(Path("${PROJECT_ROOT}") / rel)
    except Exception:
        return path.name


# ----------------------------
# JSON helpers / safety
# ----------------------------

def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _pretty_json_excerpt(path: Path, *, max_chars: int) -> str:
    try:
        obj = _read_json(path)
        txt = json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False)
    except Exception as exc:
        return f"<failed to render json excerpt: {exc}>"
    if len(txt) <= max_chars:
        return txt
    return txt[:max_chars] + "\n... (truncated) ...\n"


def _pretty_json_excerpt_from_display(display_path: str, *, max_chars: int) -> str:
    """
    The HTML wants to show a metrics excerpt, but the header uses a redacted path string.

    We keep the excerpt behavior unchanged by attempting to locate metrics.json relative
    to the current working directory when possible; otherwise we return a placeholder.

    This function is intentionally conservative: it should never leak absolute paths.
    """
    try:
        # If display is "${PROJECT_ROOT}/something", strip it and use cwd as the root.
        prefix = "${PROJECT_ROOT}/"
        if display_path.startswith(prefix):
            rel = display_path[len(prefix):]
            p = (Path.cwd() / rel).resolve()
            if p.exists() and p.name == "metrics.json":
                return _pretty_json_excerpt(p, max_chars=max_chars)
        # Otherwise, do not attempt to resolve arbitrary paths.
        return "<metrics excerpt unavailable (path redacted)>"
    except Exception:
        return "<metrics excerpt unavailable (path redacted)>"


def _escape(s: str) -> str:
    return (
        str(s)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#039;")
    )


def _finite(x: Any) -> bool:
    try:
        return math.isfinite(float(x))
    except Exception:
        return False


def _get_list(metrics: Dict[str, Any], key: str, *, nested: Tuple[str, ...]) -> Optional[List[Any]]:
    v = metrics.get(key)
    if isinstance(v, list):
        return v
    cur: Any = metrics
    for k in nested:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(k)
    if isinstance(cur, list):
        return cur
    return None
    

def _find_project_root(start: Path) -> Path:
    """
    Heuristically find the repository root to locate cross-run artifacts (e.g., results/validation/*).

    Strategy
    --------
    Walk upwards until we find a pyproject.toml (preferred) or .git (fallback).
    If not found, return the resolved start directory.
    """
    cur = Path(start).resolve()
    for _ in range(30):
        if (cur / "pyproject.toml").exists():
            return cur
        if (cur / ".git").exists():
            return cur
        if cur.parent == cur:
            break
        cur = cur.parent
    return Path(start).resolve()


def _try_load_model_vs_mc_validation(case_dir: Path) -> Optional[Dict[str, Any]]:
    """
    Load the optional model-vs-MC validation artifact if present.

    Contract
    --------
    Expected location:
        <repo_root>/results/validation/model_vs_mc_pd.json

    Returns
    -------
    dict | None
        Parsed JSON if present and valid, else None.
    """
    repo_root = _find_project_root(case_dir)
    p = repo_root / "results" / "validation" / "model_vs_mc_pd.json"
    if not p.exists():
        return None

    try:
        obj = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None

    if not isinstance(obj, dict):
        return None
    return obj