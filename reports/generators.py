"""
reports/generators.py

Report generation for sweep results.

Purpose
-------
Generate deterministic, file-based reports from sweep outputs produced by `cli/run_sweep.py`.

This module turns a sweep JSON into:
- `report.json`: structured summary + KPIs per sweep point + objective values + Pareto indices
- `plots/`: series plots and Pareto scatter (when applicable)
- `report.html`: a human-readable report with:
    * executive summary (what varied, what moved, key extremes)
    * KPI tables (sortable, highlighted Pareto points)
    * embedded plots (relative paths)
    * objective definitions and consistency notes

Input format (expected)
-----------------------
Sweep JSON is written by `cli/run_sweep.py` as:
{
  "n_points": <int>,
  "results": [
    {
      "sweep_point": {"radar.prf_hz": 1000, ...},
      "metrics": {...}   # metrics.json-like dict from engine
    },
    ...
  ]
}

Outputs
-------
<out_dir>/
  - report.json
  - report.html
  - plots/
      - far_vs_param.png (optional)
      - snr0_vs_param.png (optional)
      - pd0_vs_param.png (optional)
      - pareto.png (optional)

Determinism
-----------
- Stable ordering of rows and table columns.
- No timestamps inside report artifacts.
- Relative paths are used in HTML (portable across machines).

Usage
-----
from pathlib import Path
from reports.generators import generate_sweep_report

generate_sweep_report(
    sweep_json_path=Path("results/sweeps/my_sweep.json"),
    out_dir=Path("results/reports/my_sweep_report"),
    objectives={"far.per_second": "min", "snr_db@10000": "max"},
)

CLI integration
---------------
This module is intended to be called by higher-level CLI entrypoints. It does not define a CLI.

Dependencies
------------
- NumPy
- reports.plots for plotting
- sweeps.pareto for Pareto front indices
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import json
import math

import numpy as np

from reports.plots import plot_xy, plot_pareto_scatter
from sweeps.pareto import pareto_front_indices, Direction


class ReportError(ValueError):
    """Raised when report generation fails due to invalid inputs."""


# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------

def generate_sweep_report(
    *,
    sweep_json_path: Path,
    out_dir: Path,
    objectives: Optional[Dict[str, Direction]] = None,
    write_html: bool = True,
) -> Dict[str, Any]:
    """
    Generate a sweep report from a sweep results JSON.

    Parameters
    ----------
    sweep_json_path : Path
        Path to sweep results JSON (from cli/run_sweep.py).
    out_dir : Path
        Output directory for report.json, report.html and plots/.
    objectives : dict[str, "min"|"max"] | None
        If provided, compute Pareto front using these objectives.
    write_html : bool
        If True, write report.html in addition to report.json.

    Returns
    -------
    dict
        The report object (also written to report.json).
    """
    sweep = _load_json(sweep_json_path)
    results = sweep.get("results", None)
    if not isinstance(results, list) or len(results) == 0:
        raise ReportError("sweep JSON missing non-empty 'results' list")

    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    for idx, entry in enumerate(results):
        sp = entry.get("sweep_point", {})
        met = entry.get("metrics", {})
        if not isinstance(sp, dict) or not isinstance(met, dict):
            raise ReportError("each result must contain dict sweep_point and dict metrics")
        rows.append({"index": idx, "sweep_point": sp, "metrics": met})

    sweep_param_keys = _infer_sweep_param_keys(rows)

    kpis = [_extract_kpis(r["metrics"]) for r in rows]

    report: Dict[str, Any] = {
        "source": {"sweep_json": str(sweep_json_path)},
        "n_points": len(rows),
        "sweep_param_keys": sweep_param_keys,
        "kpis": kpis,
        "notes": [
            "KPIs are extracted strictly; missing fields remain null.",
            "Objective evaluation is strict; missing/non-finite values raise an error.",
        ],
    }

    # Series plots (single sweep parameter only, deterministic and readable)
    plot_paths: Dict[str, str] = {}
    sweep_param = sweep_param_keys[0] if len(sweep_param_keys) == 1 else None
    if sweep_param is not None:
        x = [r["sweep_point"].get(sweep_param, None) for r in rows]
        if all(v is not None for v in x):
            far_s = [k.get("far_per_second", None) for k in kpis]
            if all(v is not None for v in far_s):
                out_path = plots_dir / "far_vs_param.png"
                plot_xy(
                    x=x,
                    y=far_s,
                    xlabel=sweep_param,
                    ylabel="FAR [false alarms / s]",
                    title="FAR vs sweep parameter",
                    out_path=out_path,
                    ylog=True,
                )
                plot_paths["far_vs_param"] = _relpath(out_path, out_dir)

            snr0 = [k.get("snr_db_at_first_range", None) for k in kpis]
            if all(v is not None for v in snr0):
                out_path = plots_dir / "snr0_vs_param.png"
                plot_xy(
                    x=x,
                    y=snr0,
                    xlabel=sweep_param,
                    ylabel="SNR [dB]",
                    title="SNR (first range) vs sweep parameter",
                    out_path=out_path,
                    ylog=False,
                )
                plot_paths["snr0_vs_param"] = _relpath(out_path, out_dir)

            pd0 = [k.get("pd_at_first_range", None) for k in kpis]
            if all(v is not None for v in pd0):
                out_path = plots_dir / "pd0_vs_param.png"
                plot_xy(
                    x=x,
                    y=pd0,
                    xlabel=sweep_param,
                    ylabel="Pd [-]",
                    title="Pd (first range) vs sweep parameter",
                    out_path=out_path,
                    ylog=False,
                )
                plot_paths["pd0_vs_param"] = _relpath(out_path, out_dir)

    # Pareto analysis (if objectives provided)
    pareto_idx: Optional[List[int]] = None
    obj_values: Optional[Dict[str, List[float]]] = None
    if objectives is not None:
        obj_values = _collect_objectives(rows, objectives)
        pareto_idx = pareto_front_indices(values=obj_values, directions=objectives)
        report["pareto"] = {
            "objectives": objectives,
            "pareto_indices": pareto_idx,
        }
        report["objectives"] = obj_values

        # Plot Pareto only for 2 objectives.
        if len(objectives) == 2:
            k1, k2 = list(objectives.keys())
            x = obj_values[k1]
            y = obj_values[k2]
            out_path = plots_dir / "pareto.png"
            plot_pareto_scatter(
                x=x,
                y=y,
                xlabel=k1,
                ylabel=k2,
                title="Pareto trade-off scatter",
                out_path=out_path,
                highlight_idx=pareto_idx,
            )
            plot_paths["pareto"] = _relpath(out_path, out_dir)

    report["plots"] = plot_paths
    report["summary"] = _build_executive_summary(
        rows=rows,
        kpis=kpis,
        sweep_param=sweep_param,
        objectives=objectives,
        objective_values=obj_values,
        pareto_indices=pareto_idx,
    )

    _write_json(out_dir / "report.json", report)

    if write_html:
        html = render_sweep_report_html(
            report=report,
            rows=rows,
            kpis=kpis,
            out_dir=out_dir,
        )
        (out_dir / "report.html").write_text(html + "\n", encoding="utf-8")

    return report


# ---------------------------------------------------------------------
# KPI extraction (strict, stable)
# ---------------------------------------------------------------------

def _extract_kpis(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract stable KPIs from engine metrics.

    Policy
    ------
    - No guessing.
    - If a field doesn't exist or is not finite, KPI becomes None.

    KPIs (v1)
    ---------
    - far_per_second / far_per_scan        : from metrics["far"]
    - snr_db_at_first_range                : from metrics["snr_db"][0]
    - pd_at_first_range                    : from metrics["detection"]["pd"][0]
    - engine                               : from metrics["engine"]
    - ranges_m (for display)               : from metrics["ranges_m"]
    """
    out: Dict[str, Any] = {"engine": metrics.get("engine", None)}

    far = metrics.get("far", None)
    if isinstance(far, dict):
        out["far_per_second"] = _as_float_or_none(far.get("per_second", None))
        out["far_per_scan"] = _as_float_or_none(far.get("per_scan", None))
    else:
        out["far_per_second"] = None
        out["far_per_scan"] = None

    snr_db = metrics.get("snr_db", None)
    if isinstance(snr_db, list) and len(snr_db) > 0:
        out["snr_db_at_first_range"] = _as_float_or_none(snr_db[0])
    else:
        out["snr_db_at_first_range"] = None

    det = metrics.get("detection", None)
    if isinstance(det, dict):
        pd = det.get("pd", None)
        if isinstance(pd, list) and len(pd) > 0:
            out["pd_at_first_range"] = _as_float_or_none(pd[0])
        else:
            out["pd_at_first_range"] = None
    else:
        out["pd_at_first_range"] = None

    ranges = metrics.get("ranges_m", None)
    if isinstance(ranges, list) and ranges:
        # keep as-is; HTML renderer can format
        out["ranges_m"] = ranges
    else:
        out["ranges_m"] = None

    return out


# ---------------------------------------------------------------------
# Objectives
# ---------------------------------------------------------------------

def _collect_objectives(
    rows: List[Dict[str, Any]],
    objectives: Dict[str, Direction],
) -> Dict[str, List[float]]:
    """
    Collect objective series from rows using a strict key syntax.

    Supported objective keys (v1)
    -----------------------------
    1) "far.per_second" / "far.per_scan"
    2) "snr_db@<range_m>" or "pd@<range_m>" where <range_m> is numeric in meters,
       matched to the nearest provided range in metrics["ranges_m"].

    Raises
    ------
    ReportError if any objective is missing or non-finite in any sweep point.
    """
    out: Dict[str, List[float]] = {k: [] for k in objectives.keys()}

    for r in rows:
        met = r["metrics"]
        for k in objectives.keys():
            v = _eval_objective_key(met, k)
            if v is None or not math.isfinite(v):
                raise ReportError(f"Objective '{k}' is missing or non-finite for some points")
            out[k].append(float(v))

    return out


def _eval_objective_key(metrics: Dict[str, Any], key: str) -> Optional[float]:
    if key == "far.per_second":
        far = metrics.get("far", None)
        if isinstance(far, dict):
            return _as_float_or_none(far.get("per_second", None))
        return None

    if key == "far.per_scan":
        far = metrics.get("far", None)
        if isinstance(far, dict):
            return _as_float_or_none(far.get("per_scan", None))
        return None

    if "@" in key:
        name, r_txt = key.split("@", 1)
        try:
            r_req = float(r_txt)
        except Exception:
            return None

        ranges = metrics.get("ranges_m", None)
        if not isinstance(ranges, list) or len(ranges) == 0:
            return None
        r_arr = np.asarray(ranges, dtype=float)
        if np.any(~np.isfinite(r_arr)):
            return None
        idx = int(np.argmin(np.abs(r_arr - r_req)))

        if name == "snr_db":
            snr_db = metrics.get("snr_db", None)
            if isinstance(snr_db, list) and len(snr_db) == len(ranges):
                return _as_float_or_none(snr_db[idx])
            return None

        if name == "pd":
            det = metrics.get("detection", None)
            if isinstance(det, dict):
                pd = det.get("pd", None)
                if isinstance(pd, list) and len(pd) == len(ranges):
                    return _as_float_or_none(pd[idx])
            return None

    return None


# ---------------------------------------------------------------------
# Executive summary (deterministic, data-driven)
# ---------------------------------------------------------------------

def _build_executive_summary(
    *,
    rows: List[Dict[str, Any]],
    kpis: List[Dict[str, Any]],
    sweep_param: Optional[str],
    objectives: Optional[Dict[str, Direction]],
    objective_values: Optional[Dict[str, List[float]]],
    pareto_indices: Optional[List[int]],
) -> Dict[str, Any]:
    """
    Construct a compact summary derived from available KPIs/objectives.

    Summary fields are best-effort:
    - sweep_param and its min/max
    - best/worst FAR, SNR0, Pd0 indices (if available)
    - Pareto count (if available)
    """
    out: Dict[str, Any] = {}

    out["sweep_param"] = sweep_param
    if sweep_param is not None:
        vals = []
        for r in rows:
            v = r["sweep_point"].get(sweep_param, None)
            if v is not None:
                try:
                    vals.append(float(v))
                except Exception:
                    vals = []
                    break
        if vals and all(math.isfinite(x) for x in vals):
            out["sweep_param_min"] = float(min(vals))
            out["sweep_param_max"] = float(max(vals))

    def _best_idx(key: str, *, mode: str) -> Optional[int]:
        xs: List[Tuple[int, float]] = []
        for i, k in enumerate(kpis):
            v = k.get(key, None)
            if v is None:
                continue
            try:
                fv = float(v)
            except Exception:
                continue
            if math.isfinite(fv):
                xs.append((i, fv))
        if not xs:
            return None
        if mode == "min":
            return int(min(xs, key=lambda t: t[1])[0])
        if mode == "max":
            return int(max(xs, key=lambda t: t[1])[0])
        return None

    out["best_far_per_second_idx"] = _best_idx("far_per_second", mode="min")
    out["best_snr0_idx"] = _best_idx("snr_db_at_first_range", mode="max")
    out["best_pd0_idx"] = _best_idx("pd_at_first_range", mode="max")

    if pareto_indices is not None:
        out["pareto_count"] = int(len(pareto_indices))
        out["pareto_indices"] = [int(i) for i in pareto_indices]

    if objectives is not None:
        out["objectives"] = objectives
        if objective_values is not None:
            # include min/max per objective for quick reading
            obj_ext: Dict[str, Any] = {}
            for k, series in objective_values.items():
                arr = np.asarray(series, dtype=float)
                if np.all(np.isfinite(arr)) and arr.size:
                    obj_ext[k] = {"min": float(np.min(arr)), "max": float(np.max(arr))}
            out["objective_extrema"] = obj_ext

    return out


# ---------------------------------------------------------------------
# HTML rendering
# ---------------------------------------------------------------------

def render_sweep_report_html(
    *,
    report: Dict[str, Any],
    rows: List[Dict[str, Any]],
    kpis: List[Dict[str, Any]],
    out_dir: Path,
) -> str:
    """
    Render a self-contained HTML report (inline CSS/JS; no external deps).

    - Uses relative paths for images (portable).
    - Tables are readable and sortable.
    - Pareto points are highlighted when available.
    """
    plots = report.get("plots", {}) if isinstance(report.get("plots", {}), dict) else {}
    pareto = set()
    if isinstance(report.get("pareto", None), dict):
        idx = report["pareto"].get("pareto_indices", [])
        if isinstance(idx, list):
            pareto = {int(i) for i in idx}

    sweep_param_keys = report.get("sweep_param_keys", [])
    sweep_param_keys = sweep_param_keys if isinstance(sweep_param_keys, list) else []

    # Build stable column set for sweep_point table:
    # - include all sweep_param_keys
    # - plus KPI columns
    kpi_cols = [
        ("engine", "Engine"),
        ("far_per_second", "FAR / s"),
        ("far_per_scan", "FAR / scan"),
        ("snr_db_at_first_range", "SNR@R0 (dB)"),
        ("pd_at_first_range", "Pd@R0"),
    ]
    sp_cols = [(str(k), str(k)) for k in sweep_param_keys]

    def esc(s: Any) -> str:
        return (
            str(s)
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
        )

    def fmt(x: Any) -> str:
        if x is None:
            return ""
        if isinstance(x, (int, np.integer)):
            return str(int(x))
        try:
            v = float(x)
        except Exception:
            return esc(x)
        if not math.isfinite(v):
            return ""
        # compact numeric formatting:
        av = abs(v)
        if (av != 0.0 and av < 1e-3) or av >= 1e6:
            return f"{v:.3g}"
        return f"{v:.6g}"

    # Executive summary block
    summary = report.get("summary", {}) if isinstance(report.get("summary", {}), dict) else {}
    sum_lines: List[str] = []
    if summary.get("sweep_param", None):
        sp = summary.get("sweep_param")
        mn = summary.get("sweep_param_min", None)
        mx = summary.get("sweep_param_max", None)
        if mn is not None and mx is not None:
            sum_lines.append(f"<li><b>Sweep variable:</b> {esc(sp)} (min={esc(fmt(mn))}, max={esc(fmt(mx))})</li>")
        else:
            sum_lines.append(f"<li><b>Sweep variable:</b> {esc(sp)}</li>")
    else:
        sum_lines.append("<li><b>Sweep variable:</b> (multiple parameters)</li>")

    if "pareto_count" in summary:
        sum_lines.append(f"<li><b>Pareto-efficient points:</b> {esc(summary.get('pareto_count'))}</li>")

    def _link_to_row(i: Optional[int], label: str) -> str:
        if i is None:
            return f"<li><b>{esc(label)}:</b> (not available)</li>"
        return f"<li><b>{esc(label)}:</b> point #{int(i)}</li>"

    sum_lines.append(_link_to_row(summary.get("best_far_per_second_idx", None), "Lowest FAR/s"))
    sum_lines.append(_link_to_row(summary.get("best_snr0_idx", None), "Highest SNR@R0"))
    sum_lines.append(_link_to_row(summary.get("best_pd0_idx", None), "Highest Pd@R0"))

    # Objectives block
    obj_html = ""
    if isinstance(report.get("pareto", None), dict):
        objectives = report["pareto"].get("objectives", None)
        if isinstance(objectives, dict):
            items = "".join(f"<li><code>{esc(k)}</code> → <b>{esc(v)}</b></li>" for k, v in objectives.items())
            obj_html = f"""
              <div class="card">
                <h2>Objectives</h2>
                <ul>{items}</ul>
              </div>
            """

    # Plots block (show only those that exist in report)
    plot_cards: List[str] = []
    for key, title in [
        ("far_vs_param", "FAR vs sweep parameter"),
        ("snr0_vs_param", "SNR@R0 vs sweep parameter"),
        ("pd0_vs_param", "Pd@R0 vs sweep parameter"),
        ("pareto", "Pareto scatter"),
    ]:
        if key in plots:
            rel = str(plots[key])
            plot_cards.append(
                f"""
                <div class="card">
                  <h2>{esc(title)}</h2>
                  <div class="imgwrap">
                    <img src="{esc(rel)}" alt="{esc(title)}" />
                  </div>
                  <div class="muted">{esc(rel)}</div>
                </div>
                """
            )
    plots_html = "\n".join(plot_cards) if plot_cards else "<div class='card'><h2>Plots</h2><div class='muted'>(no plots available)</div></div>"

    # Main table rows
    header_cells = ["<th>#</th>"] + [f"<th>{esc(lbl)}</th>" for _, lbl in sp_cols] + [f"<th>{esc(lbl)}</th>" for _, lbl in kpi_cols]
    body_rows: List[str] = []
    for i, r in enumerate(rows):
        cls = "pareto" if i in pareto else ""
        sp = r["sweep_point"]
        k = kpis[i] if i < len(kpis) else {}

        cells = [f"<td class='mono'>{i}</td>"]
        for key, _lbl in sp_cols:
            cells.append(f"<td>{esc(fmt(sp.get(key, None)))}</td>")
        for key, _lbl in kpi_cols:
            cells.append(f"<td>{esc(fmt(k.get(key, None)))}</td>")

        body_rows.append(f"<tr class='{cls}'>" + "".join(cells) + "</tr>")

    # Small helper: export HTML (download current DOM)
    js = r"""
    <script>
      function downloadHTML(filename) {
        const html = "<!doctype html>\n" + document.documentElement.outerHTML;
        const blob = new Blob([html], {type: "text/html;charset=utf-8"});
        const a = document.createElement("a");
        a.href = URL.createObjectURL(blob);
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(a.href);
      }

      function sortTable(tableId, colIndex, numeric) {
        const table = document.getElementById(tableId);
        const tbody = table.tBodies[0];
        const rows = Array.from(tbody.rows);

        const dirAttr = table.getAttribute("data-sortdir-" + colIndex);
        const dir = (dirAttr === "asc") ? "desc" : "asc";

        rows.sort((a,b) => {
          const ax = a.cells[colIndex].innerText.trim();
          const bx = b.cells[colIndex].innerText.trim();
          if (numeric) {
            const av = parseFloat(ax); const bv = parseFloat(bx);
            const aok = Number.isFinite(av); const bok = Number.isFinite(bv);
            if (!aok && !bok) return 0;
            if (!aok) return (dir === "asc") ? 1 : -1;
            if (!bok) return (dir === "asc") ? -1 : 1;
            return (dir === "asc") ? (av - bv) : (bv - av);
          } else {
            return (dir === "asc") ? ax.localeCompare(bx) : bx.localeCompare(ax);
          }
        });

        for (const r of rows) tbody.appendChild(r);
        table.setAttribute("data-sortdir-" + colIndex, dir);
      }
    </script>
    """

    css = """
    <style>
      :root {
        --fg: #111;
        --muted: #666;
        --bg: #fff;
        --card: #fff;
        --border: #e3e5e8;
        --head: #f6f7f9;
        --hl: #fcfcff;
      }
      body {
        margin: 20px;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        color: var(--fg);
        background: var(--bg);
        line-height: 1.35;
      }
      .topbar {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 12px;
        margin-bottom: 14px;
      }
      .title h1 { margin: 0; font-size: 1.35rem; }
      .title .muted { margin-top: 4px; }
      .btn {
        border: 1px solid var(--border);
        border-radius: 8px;
        padding: 8px 10px;
        background: #fff;
        cursor: pointer;
        font-weight: 600;
      }
      .btn:hover { background: var(--head); }
      .grid {
        display: grid;
        grid-template-columns: 1fr;
        gap: 12px;
        max-width: 1100px;
      }
      .card {
        border: 1px solid var(--border);
        border-radius: 10px;
        padding: 12px 14px;
        background: var(--card);
      }
      h2 { margin: 0 0 10px 0; font-size: 1.15rem; }
      ul { margin: 0.4rem 0 0 1.2rem; }
      code, .mono { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace; }
      .muted { color: var(--muted); font-size: 0.95rem; }
      table { border-collapse: collapse; width: 100%; margin-top: 6px; }
      th, td { border: 1px solid #ddd; padding: 8px; text-align: left; font-size: 0.95rem; }
      th { background: var(--head); user-select: none; cursor: pointer; }
      tr.pareto td { background: var(--hl); font-weight: 700; }
      .imgwrap { overflow-x: auto; }
      img { max-width: 100%; height: auto; border: 1px solid var(--border); border-radius: 8px; }
      .foot { margin-top: 10px; font-size: 0.9rem; color: var(--muted); }
    </style>
    """

    # Sort heuristics: numeric columns (all except sweep_point strings). We mark some as numeric explicitly.
    # Column indices: 0 is '#', then sweep_point cols, then KPI cols.
    numeric_cols = set([0])
    for j in range(len(sp_cols)):
        numeric_cols.add(1 + j)
    for j in range(len(kpi_cols)):
        numeric_cols.add(1 + len(sp_cols) + j)

    # Build header with sort callbacks
    ths: List[str] = []
    for col_i, th in enumerate(header_cells):
        numeric = "true" if col_i in numeric_cols else "false"
        label = th.replace("<th>", "").replace("</th>", "")
        ths.append(f"<th onclick='sortTable(\"kpi_table\", {col_i}, {numeric})'>{label}</th>")

    # Source path display (portable): show as provided, without assuming a filesystem.
    src = report.get("source", {}).get("sweep_json", "")
    src_disp = esc(src)

    html = f"""
    <!doctype html>
    <html>
      <head>
        <meta charset="utf-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <title>Sweep Report</title>
        {css}
      </head>
      <body>
        <div class="topbar">
          <div class="title">
            <h1>Sweep Report</h1>
            <div class="muted">Source: <span class="mono">{src_disp}</span></div>
          </div>
          <div>
            <button class="btn" onclick="downloadHTML('report.html')">Export HTML</button>
          </div>
        </div>

        <div class="grid">
          <div class="card">
            <h2>Summary</h2>
            <ul>
              {''.join(sum_lines)}
            </ul>
            <div class="foot">
              Points highlighted in the table correspond to the Pareto set when objectives are defined.
            </div>
          </div>

          {obj_html}

          {plots_html}

          <div class="card">
            <h2>All points</h2>
            <div class="muted">Click any column header to sort.</div>
            <table id="kpi_table">
              <thead>
                <tr>
                  {''.join(ths)}
                </tr>
              </thead>
              <tbody>
                {''.join(body_rows)}
              </tbody>
            </table>
          </div>
        </div>

        {js}
      </body>
    </html>
    """
    return html.strip()


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _infer_sweep_param_keys(rows: List[Dict[str, Any]]) -> List[str]:
    keys = set()
    for r in rows:
        sp = r["sweep_point"]
        for k in sp.keys():
            keys.add(k)
    return sorted(keys)


def _as_float_or_none(x: Any) -> Optional[float]:
    try:
        v = float(x)
    except Exception:
        return None
    if not math.isfinite(v):
        return None
    return v


def _relpath(path: Path, start: Path) -> str:
    """
    Compute a portable relative path for HTML embedding.
    Falls back to filename if relpath cannot be computed.
    """
    try:
        return str(path.relative_to(start))
    except Exception:
        return path.name


def _load_json(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ReportError(f"Failed to load JSON: {path}") from exc


def _write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")