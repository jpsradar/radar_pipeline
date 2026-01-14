"""
reports/html_case_report.py

Self-contained HTML report generator for a single case run.

Purpose
-------
Generate a single-file HTML report (report.html) for a run directory. The report is
intended for engineering review and traceability:

- Summarize what was executed (engine, seed, case source when available).
- Present key outputs (SNR span, detection summary, FAR/trials when present).
- Surface operationally relevant warnings derived strictly from available fields.
- Embed plots (optional) and include reproducibility appendix (metrics + manifest).

Design constraints
------------------
- Standalone output: no external CSS/JS dependencies and no relative image paths.
- Path hygiene: never emit absolute paths; show repo-relative or tail-only text.
- Deterministic formatting: stable ordering and stable rounding to minimize diffs.

Inputs
------
- metrics: dict loaded from <run_dir>/metrics.json (required)
- manifest: dict loaded from <run_dir>/case_manifest.json (optional)
- plots: mapping {name: png_bytes} to embed charts as base64 (optional)

Outputs
-------
- HTML string suitable to write to <run_dir>/report.html

Dependencies
------------
- Python standard library only (base64, json, math, datetime)

Usage
-----
from pathlib import Path
import json
from reports.html_case_report import render_case_report_html, read_optional_json

run_dir = Path("results/cases/<run_id>")
metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
manifest = read_optional_json(run_dir / "case_manifest.json")
html = render_case_report_html(metrics=metrics, manifest=manifest, plots={}, title="Radar Case Report")
(run_dir / "report.html").write_text(html, encoding="utf-8")
"""

from __future__ import annotations

import base64
import datetime as _dt
import json
import math
from typing import Any, Dict, Mapping, Optional


# ---------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------


def read_optional_json(path) -> Optional[Dict[str, Any]]:
    """
    Best-effort JSON reader.

    Returns None if:
    - file does not exist
    - JSON is invalid
    - read fails for any reason
    """
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


# ---------------------------------------------------------------------
# Formatting / safety utilities
# ---------------------------------------------------------------------


def _isfinite(x: Any) -> bool:
    try:
        return math.isfinite(float(x))
    except Exception:
        return False


def _fmt(x: Any, *, ndp: int = 3) -> str:
    """
    Stable numeric formatting for HTML.

    - ints are shown without decimals
    - floats use fixed-point unless very small/large, then scientific notation
    - None is rendered as an em dash
    """
    if x is None:
        return "—"
    if isinstance(x, bool):
        return "true" if x else "false"
    if isinstance(x, int):
        return str(x)
    if _isfinite(x):
        v = float(x)
        if v != 0.0 and (abs(v) < 1e-3 or abs(v) >= 1e6):
            return f"{v:.{ndp}e}"
        return f"{v:.{ndp}f}".rstrip("0").rstrip(".")
    return str(x)


def _to_data_uri_png(png_bytes: bytes) -> str:
    b64 = base64.b64encode(png_bytes).decode("ascii")
    return f"data:image/png;base64,{b64}"


def _json_pretty(obj: Any) -> str:
    return json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False)


def _safe_relpath_text(path_str: Optional[str]) -> str:
    """
    Render a path without leaking absolute locations.

    Policy
    ------
    - If the string looks like it contains user-home prefixes, keep only a short tail.
    - If already relative-ish, keep it.
    - If absolute and unknown, keep only a short tail.

    This is an intentionally conservative sanitizer: it prefers omission over leakage.
    """
    if not path_str:
        return "—"
    s = str(path_str).replace("\\", "/")

    # Common home-like prefixes.
    for marker in ("/Users/", "/home/", "C:/Users/"):
        if marker in s:
            parts = [p for p in s.split("/") if p]
            if len(parts) >= 3:
                return "/".join(parts[-3:])
            return parts[-1] if parts else "—"

    # Relative-ish path: safe to show.
    if not s.startswith("/"):
        return s

    # Absolute but unknown: show short tail only.
    parts = [p for p in s.split("/") if p]
    return "/".join(parts[-3:]) if len(parts) >= 3 else (parts[-1] if parts else "—")


# ---------------------------------------------------------------------
# Metric extraction (stdlib-only, conservative)
# ---------------------------------------------------------------------


def _engine(metrics: Dict[str, Any]) -> str:
    e = metrics.get("engine")
    return str(e) if isinstance(e, str) else "unknown"


def _extract_ranges_m(metrics: Dict[str, Any]) -> Optional[list[float]]:
    raw = metrics.get("ranges_m")
    if not isinstance(raw, list) or not raw:
        return None
    out: list[float] = []
    for x in raw:
        try:
            v = float(x)
        except Exception:
            return None
        if not math.isfinite(v):
            return None
        out.append(v)
    return out


def _extract_pd_curve(metrics: Dict[str, Any]) -> Optional[list[float]]:
    det = metrics.get("detection")
    if not isinstance(det, dict):
        return None
    raw = det.get("pd")
    if not isinstance(raw, list) or not raw:
        return None
    out: list[float] = []
    for x in raw:
        try:
            v = float(x)
        except Exception:
            return None
        if not math.isfinite(v):
            return None
        out.append(v)
    return out


def _range_at_pd(metrics: Dict[str, Any], *, pd_min: float) -> Optional[float]:
    """
    Maximum range where Pd >= pd_min, using the provided (ranges_m, detection.pd).

    - Returns None if the curve is missing, malformed, length-mismatched,
      or the threshold is never reached.
    - Does not interpolate or extrapolate; this is a grid-based query.
    """
    thr = float(pd_min)
    if not math.isfinite(thr) or thr <= 0.0 or thr > 1.0:
        raise ValueError(f"pd_min must be in (0,1], got {pd_min!r}")

    r = _extract_ranges_m(metrics)
    p = _extract_pd_curve(metrics)
    if not r or not p or len(r) != len(p):
        return None

    ok = [rv for rv, pv in zip(r, p) if pv >= thr]
    return max(ok) if ok else None


# ---------------------------------------------------------------------
# Summary + warnings (no guessing; fields must exist)
# ---------------------------------------------------------------------


def _extract_case_summary(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract a compact run summary strictly from available fields.
    """
    eng = _engine(metrics)

    ranges_m = metrics.get("ranges_m") if isinstance(metrics.get("ranges_m"), list) else None
    snr_db = metrics.get("snr_db") if isinstance(metrics.get("snr_db"), list) else None
    det = metrics.get("detection") if isinstance(metrics.get("detection"), dict) else None
    far = metrics.get("far") if isinstance(metrics.get("far"), dict) else None

    out: Dict[str, Any] = {"engine": eng}

    if ranges_m and snr_db and len(ranges_m) == len(snr_db) and len(ranges_m) > 0:
        out["range_span_m"] = (float(min(ranges_m)), float(max(ranges_m)))
        out["snr_span_db"] = (float(min(snr_db)), float(max(snr_db)))

    if det and isinstance(det.get("pfa"), (int, float)):
        out["pfa"] = float(det["pfa"])
    if det and isinstance(det.get("pd"), list) and det["pd"]:
        out["pd_at_first_range"] = float(det["pd"][0])

    # FASE 5-style queries if a Pd curve is present
    r90 = _range_at_pd(metrics, pd_min=0.90)
    r80 = _range_at_pd(metrics, pd_min=0.80)
    r50 = _range_at_pd(metrics, pd_min=0.50)
    if r90 is not None or r80 is not None or r50 is not None:
        out["range_at_pd_90_m"] = r90
        out["range_at_pd_80_m"] = r80
        out["range_at_pd_50_m"] = r50

    if far:
        out["far_per_second"] = far.get("per_second")
        out["far_per_scan"] = far.get("per_scan")

    if eng == "signal_level":
        rd = metrics.get("rd_grid") if isinstance(metrics.get("rd_grid"), dict) else None
        if rd:
            out["rd_grid"] = [rd.get("n_range_bins"), rd.get("n_doppler_bins")]

    return out


def _engineering_warnings(metrics: Dict[str, Any]) -> list[str]:
    """
    Emit warnings derived from explicit numeric fields.

    This function does not infer hidden context or apply domain-specific heuristics
    beyond simple thresholds on already-reported outputs.
    """
    warns: list[str] = []

    # FAR: operational load risk if per-second rate is high.
    far = metrics.get("far") if isinstance(metrics.get("far"), dict) else None
    if far:
        ps = far.get("per_second")
        if _isfinite(ps):
            v = float(ps)
            if v >= 1.0:
                warns.append(f"High false-alarm load: FAR ≈ {_fmt(v, ndp=3)} / s (verify Pfa vs trials/sec).")
            elif v > 0.1:
                warns.append(f"Non-trivial false-alarm load: FAR ≈ {_fmt(v, ndp=3)} / s (may stress downstream processing).")

    # Extremely strict Pfa: expected detection penalty unless compensated by processing gain.
    det = metrics.get("detection") if isinstance(metrics.get("detection"), dict) else None
    if det and _isfinite(det.get("pfa")):
        pfa = float(det["pfa"])
        if pfa < 1e-9:
            warns.append("Very strict Pfa (< 1e-9): expect detection loss unless processing gain is available.")

    return warns


# ---------------------------------------------------------------------
# Report rendering
# ---------------------------------------------------------------------


def render_case_report_html(
    *,
    metrics: Dict[str, Any],
    manifest: Optional[Dict[str, Any]],
    plots: Mapping[str, bytes],
    title: str = "Radar Case Report",
) -> str:
    """
    Render a complete self-contained HTML report for one run.
    """
    now = _dt.datetime.now(tz=_dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")
    summary = _extract_case_summary(metrics)
    warns = _engineering_warnings(metrics)

    # Best-effort provenance fields from manifest.
    case_src = None
    seed = None
    engine_selected = None
    if isinstance(manifest, dict):
        case_src = manifest.get("case_path") or manifest.get("source_case")
        seed = manifest.get("seed")
        extras = manifest.get("extras") if isinstance(manifest.get("extras"), dict) else None
        if extras and isinstance(extras.get("engine_selected"), str):
            engine_selected = extras["engine_selected"]

    # Assumptions section: show explicit engine assumptions when present.
    assumptions = metrics.get("assumptions") if isinstance(metrics.get("assumptions"), dict) else {}

    # Optional plots.
    plot_cards: list[str] = []
    for name, png in plots.items():
        uri = _to_data_uri_png(png)
        plot_cards.append(
            f"""
          <div class="card">
            <h2>{name}</h2>
            <div class="imgwrap"><img src="{uri}" alt="{name}"/></div>
          </div>
        """.rstrip()
        )

    # Appendix JSON (collapsible).
    metrics_json = _json_pretty(metrics)
    manifest_json = _json_pretty(manifest) if isinstance(manifest, dict) else None

    # SNR budget table (only if present; no guessing).
    snr_budget = metrics.get("snr_budget") if isinstance(metrics.get("snr_budget"), dict) else None

    def snr_budget_table_html() -> str:
        if not isinstance(snr_budget, dict):
            return "<div class='muted'>No SNR budget table found in metrics for this engine.</div>"
        rows = []
        for k in sorted(snr_budget.keys()):
            blk = snr_budget.get(k)
            if not isinstance(blk, dict):
                continue
            v_lin = blk.get("value_lin")
            v_db = blk.get("value_db")
            notes = blk.get("notes", "")
            rows.append(
                f"<tr><td class='mono'>{k}</td><td>{_fmt(v_lin)}</td><td>{_fmt(v_db)}</td><td>{notes}</td></tr>"
            )
        if not rows:
            return "<div class='muted'>SNR budget present but empty.</div>"
        return f"""
          <table>
            <thead><tr><th>Block</th><th>Value (lin)</th><th>Value (dB)</th><th>Notes</th></tr></thead>
            <tbody>
              {''.join(rows)}
            </tbody>
          </table>
        """.rstrip()

    def detection_section_html() -> str:
        det = metrics.get("detection") if isinstance(metrics.get("detection"), dict) else None
        if not det:
            return "<div class='muted'>No detection model output found for this engine.</div>"

        lines: list[str] = []

        if _isfinite(det.get("pfa")):
            lines.append(f"<li><b>Pfa:</b> {_fmt(det.get('pfa'), ndp=6)}</li>")

        # If a Pd curve exists, expose both a point and threshold-range queries.
        if isinstance(det.get("pd"), list) and det["pd"]:
            pd0 = det["pd"][0]
            lines.append(f"<li><b>Pd@first range:</b> {_fmt(pd0, ndp=6)}</li>")

            r90 = _range_at_pd(metrics, pd_min=0.90)
            r80 = _range_at_pd(metrics, pd_min=0.80)
            r50 = _range_at_pd(metrics, pd_min=0.50)

            if r90 is not None:
                lines.append(f"<li><b>Range @ Pd≥0.90:</b> {_fmt(r90, ndp=0)} m</li>")
            if r80 is not None:
                lines.append(f"<li><b>Range @ Pd≥0.80:</b> {_fmt(r80, ndp=0)} m</li>")
            if r50 is not None:
                lines.append(f"<li><b>Range @ Pd≥0.50:</b> {_fmt(r50, ndp=0)} m</li>")

            if r90 is None and r80 is None and r50 is None:
                lines.append(
                    "<li class='muted'>Pd curve present but does not reach 0.50/0.80/0.90 on the provided range grid.</li>"
                )

        if _isfinite(det.get("snr_threshold_db")):
            lines.append(f"<li><b>Equivalent threshold (SNR_dt):</b> {_fmt(det.get('snr_threshold_db'))} dB</li>")

        return f"<ul>{''.join(lines) if lines else '<li class=muted>Detection fields present but incomplete.</li>'}</ul>"

    def far_section_html() -> str:
        far = metrics.get("far") if isinstance(metrics.get("far"), dict) else None
        trials = metrics.get("trials") if isinstance(metrics.get("trials"), dict) else None
        parts: list[str] = []

        if trials:
            cs = trials.get("cells_per_second")
            cscan = trials.get("cells_per_scan")
            if _isfinite(cs) or _isfinite(cscan):
                parts.append("<h3>Trials</h3><ul>")
                if _isfinite(cs):
                    parts.append(f"<li><b>cells/sec:</b> {_fmt(cs)}</li>")
                if _isfinite(cscan):
                    parts.append(f"<li><b>cells/scan:</b> {_fmt(cscan)}</li>")
                parts.append("</ul>")

        if far:
            ps = far.get("per_second")
            pscan = far.get("per_scan")
            parts.append("<h3>FAR</h3><ul>")
            if _isfinite(ps):
                parts.append(f"<li><b>FAR/sec:</b> {_fmt(ps)}</li>")
            if _isfinite(pscan):
                parts.append(f"<li><b>FAR/scan:</b> {_fmt(pscan)}</li>")
            parts.append("</ul>")

        if not parts:
            return "<div class='muted'>No FAR/trials outputs found in metrics for this engine.</div>"
        return "".join(parts)

    def assumptions_html() -> str:
        if not isinstance(assumptions, dict) or not assumptions:
            return "<div class='muted'>No explicit assumptions block found.</div>"
        items = []
        for k in sorted(assumptions.keys()):
            items.append(f"<li><span class='mono'>{k}</span>: <b>{_fmt(assumptions[k])}</b></li>")
        return f"<ul>{''.join(items)}</ul>"

    def top_drivers_html() -> str:
        """
        Present driver notes only when supported by structured fields.

        This section is intentionally conservative: it avoids claiming dominance
        without explicit attribution in the metrics payload.
        """
        drivers: list[str] = []

        nm = metrics.get("noise_model") if isinstance(metrics.get("noise_model"), dict) else None
        if nm and _isfinite(nm.get("noise_power_dbw")):
            drivers.append("Noise floor is explicitly reported; BW and NF are primary sensitivity knobs in noise-limited regimes.")

        env = metrics.get("environment") if isinstance(metrics.get("environment"), dict) else None
        if env and _isfinite(env.get("system_losses_db")):
            drivers.append("System losses are applied; losses reduce SNR and therefore reduce achievable range.")

        if not drivers:
            return "<div class='muted'>Not enough structured fields to attribute drivers (add snr_budget / trials blocks to metrics).</div>"
        return "<ul>" + "".join([f"<li>{d}</li>" for d in drivers]) + "</ul>"

    src_txt = _safe_relpath_text(str(case_src) if case_src else None)

    warn_block = ""
    if warns:
        warn_items = "".join([f"<li>{w}</li>" for w in warns])
        warn_block = f"<div class='card warn'><h2>Engineering Warnings</h2><ul>{warn_items}</ul></div>"

    return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{title}</title>
  <style>
    :root {{
      --fg:#111; --muted:#666; --bg:#fff; --card:#fff; --border:#e3e5e8; --head:#f6f7f9;
      --warnbg:#fff6f6; --warnbd:#ffd2d2;
    }}
    body {{ margin: 18px; font-family: -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
           color:var(--fg); background:var(--bg); line-height:1.42; }}
    .topbar {{ display:flex; align-items:flex-start; justify-content:space-between; gap:12px; margin-bottom:14px; max-width: 1120px; }}
    .title h1 {{ margin:0; font-size:1.35rem; }}
    .muted {{ color:var(--muted); font-size:0.95rem; }}
    .mono {{ font-family: ui-monospace,SFMono-Regular,Menlo,Monaco,Consolas,"Liberation Mono","Courier New",monospace; }}
    .btn {{ border:1px solid var(--border); border-radius:10px; padding:8px 10px; background:#fff; cursor:pointer; font-weight:650; }}
    .btn:hover {{ background: var(--head); }}
    .grid {{ display:grid; grid-template-columns: 1fr; gap: 12px; max-width: 1120px; }}
    .card {{ border:1px solid var(--border); border-radius:12px; padding: 12px 14px; background: var(--card); }}
    h2 {{ margin:0 0 10px 0; font-size:1.15rem; }}
    h3 {{ margin: 12px 0 6px 0; font-size:1.02rem; }}
    ul {{ margin: 0.35rem 0 0 1.2rem; }}
    table {{ border-collapse: collapse; width: 100%; margin-top: 8px; }}
    th, td {{ border:1px solid #ddd; padding: 8px; text-align:left; font-size: 0.95rem; vertical-align: top; }}
    th {{ background: var(--head); }}
    .imgwrap {{ overflow-x:auto; }}
    img {{ max-width: 100%; height: auto; border:1px solid var(--border); border-radius: 10px; }}
    .warn {{ border:1px solid var(--warnbd); background: var(--warnbg); }}
    details > summary {{ cursor: pointer; user-select: none; font-weight: 650; }}
    pre {{ white-space: pre-wrap; word-break: break-word; border:1px solid var(--border); border-radius:10px; padding:10px; background:#fafbfc; }}
  </style>
</head>
<body>
  <div class="topbar">
    <div class="title">
      <h1>{title}</h1>
      <div class="muted">
        Generated: <span class="mono">{now}</span>
        &nbsp;•&nbsp; Engine: <span class="mono">{_fmt(engine_selected or summary.get("engine"))}</span>
        &nbsp;•&nbsp; Seed: <span class="mono">{_fmt(seed)}</span>
      </div>
      <div class="muted">Case: <span class="mono">{src_txt}</span></div>
    </div>
    <div>
      <button class="btn" onclick="downloadHTML('report.html')">Download HTML</button>
    </div>
  </div>

  <div class="grid">

    <div class="card">
      <h2>Executive Summary</h2>
      <ul>
        <li><b>Execution:</b> engine <span class="mono">{_fmt(summary.get("engine"))}</span> using the current case configuration.</li>
        <li><b>Key outputs (when present):</b>
          <span class="mono">SNR span</span> {_fmt(summary.get("snr_span_db")[0] if isinstance(summary.get("snr_span_db"), tuple) else None)}…{_fmt(summary.get("snr_span_db")[1] if isinstance(summary.get("snr_span_db"), tuple) else None)} dB,
          <span class="mono">FAR/sec</span> {_fmt(summary.get("far_per_second"), ndp=6)},
          <span class="mono">Pfa</span> {_fmt(summary.get("pfa"), ndp=6)}.
        </li>
      </ul>

      <h3>Driver notes (from structured fields)</h3>
      {top_drivers_html()}
    </div>

    {warn_block}

    <div class="card">
      <h2>Assumptions & Definitions</h2>
      {assumptions_html()}
    </div>

    <div class="card">
      <h2>SNR Budget Table</h2>
      {snr_budget_table_html()}
    </div>

    <div class="card">
      <h2>Detection Performance</h2>
      {detection_section_html()}
    </div>

    <div class="card">
      <h2>False Alarms: Pfa → FAR</h2>
      {far_section_html()}
    </div>

    {"".join(plot_cards)}

    <div class="card">
      <h2>Appendix</h2>
      <details>
        <summary>metrics.json (reproducibility)</summary>
        <pre class="mono">{metrics_json}</pre>
      </details>
      <details style="margin-top:10px;">
        <summary>case_manifest.json (reproducibility)</summary>
        <pre class="mono">{manifest_json if manifest_json is not None else "—"}</pre>
      </details>
    </div>

  </div>

  <script>
    function downloadHTML(filename) {{
      const html = "<!doctype html>\\n" + document.documentElement.outerHTML;
      const blob = new Blob([html], {{type: "text/html;charset=utf-8"}});
      const a = document.createElement("a");
      a.href = URL.createObjectURL(blob);
      a.download = filename;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(a.href);
    }}
  </script>
</body>
</html>
"""