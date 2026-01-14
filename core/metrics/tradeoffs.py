"""
core/metrics/tradeoffs.py

Trade-off analysis utilities for radar system design (sweep → insight → narrative).

Purpose
-------
A radar "pipeline" that impresses is not just math or a simulator.
It must demonstrate systems thinking: constraints, trade-offs, and what changes matter.

This module is built to take:
- a set of run outputs (metrics.json + manifest/config metadata),
- produce comparable derived scalars (range at Pd target, FAR, SNR margin),
- and rank/visualize trade-offs (sensitivity and Pareto front).

V1 deliverables
---------------
- Sweep ingestion (folder with many run subfolders).
- Normalize heterogeneous engine outputs into a common record schema.
- Provide:
    (1) Pareto front computation (e.g., maximize range, minimize FAR)
    (2) Sensitivity estimates (finite-difference where possible)
    (3) Report-ready tables (HTML fragment) to embed into reader-facing output

Inputs
------
- A directory containing multiple case runs, each with:
    - metrics.json
    - case_manifest.json (optional but strongly recommended)

Outputs
-------
Programmatic:
- load_runs(root) -> list[dict]
- build_tradeoff_table(records, objectives=...) -> dict
- pareto_front(points, maximize=..., minimize=...) -> indices

CLI:
    python -m core.metrics.tradeoffs --runs results/cases --html out/tradeoffs_fragment.html

Dependencies
------------
- Standard library + NumPy.
- No pandas required (kept lean and portable).

Notes
-----
This is not an optimizer. It is an explanation engine:
it turns data into "if you change X, Y happens, because Z".

"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np


# ---------------------------------------------------------------------
# IO / ingestion
# ---------------------------------------------------------------------

def _read_json(p: Path) -> Dict[str, Any]:
    return json.loads(p.read_text(encoding="utf-8"))


def find_run_dirs(root: Path) -> List[Path]:
    """
    Discover run directories under a root (expects metrics.json inside).
    """
    out: List[Path] = []
    for m in sorted(root.glob("**/metrics.json")):
        out.append(m.parent)
    # stable unique
    uniq = []
    seen = set()
    for d in out:
        if d not in seen:
            seen.add(d)
            uniq.append(d)
    return uniq


def load_runs(root: Path) -> List[Dict[str, Any]]:
    """
    Load run records from a root folder.

    Each record includes:
    - run_dir
    - engine
    - metrics (dict)
    - manifest (dict|None)
    - derived (common comparable scalars)
    """
    records: List[Dict[str, Any]] = []
    for run_dir in find_run_dirs(root):
        metrics_path = run_dir / "metrics.json"
        manifest_path = run_dir / "case_manifest.json"

        metrics = _read_json(metrics_path)
        manifest = _read_json(manifest_path) if manifest_path.exists() else None

        engine = str(metrics.get("engine", "unknown"))
        derived = derive_common_scalars(metrics)

        records.append(
            {
                "run_dir": str(run_dir),
                "engine": engine,
                "metrics": metrics,
                "manifest": manifest,
                "derived": derived,
            }
        )
    return records


# ---------------------------------------------------------------------
# Normalization: derived scalars
# ---------------------------------------------------------------------

def _finite(x: Any) -> Optional[float]:
    try:
        v = float(x)
    except Exception:
        return None
    if not math.isfinite(v):
        return None
    return v


def _range_at_pd_target(ranges_m: Any, pd: Any, *, pd_target: float) -> Optional[float]:
    """
    Estimate the farthest range such that Pd >= pd_target.

    Requires:
    - ranges_m: list of ranges (assumed sorted descending or ascending; we handle both)
    - pd: list of Pd values aligned with ranges
    """
    if not isinstance(ranges_m, list) or not isinstance(pd, list):
        return None
    if len(ranges_m) != len(pd) or len(pd) < 2:
        return None

    r = np.asarray([float(x) for x in ranges_m], dtype=float)
    p = np.asarray([float(x) for x in pd], dtype=float)
    if np.any(~np.isfinite(r)) or np.any(~np.isfinite(p)):
        return None

    # Ensure we interpret "farthest" correctly: larger range is farther.
    order = np.argsort(r)
    r_sorted = r[order]
    p_sorted = p[order]

    # Find the max range where Pd >= target.
    mask = p_sorted >= float(pd_target)
    if not np.any(mask):
        return None
    return float(np.max(r_sorted[mask]))


def derive_common_scalars(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert engine-specific metrics into a common scalar set.

    Common derived fields (best-effort):
    - snr_db_min / snr_db_max
    - range_at_pd_0p9_m (if Pd curve present)
    - pfa_empirical / pfa_target (if MC)
    - rd_shape (if signal_level)
    """
    engine = str(metrics.get("engine", "unknown"))
    out: Dict[str, Any] = {"engine": engine}

    # model_based: expected snr_db list and detection.pd list
    if engine == "model_based":
        snr_db = metrics.get("snr_db", None)
        if isinstance(snr_db, list) and snr_db:
            arr = np.asarray(snr_db, dtype=float)
            if np.all(np.isfinite(arr)):
                out["snr_db_min"] = float(np.min(arr))
                out["snr_db_max"] = float(np.max(arr))

        det = metrics.get("detection", None)
        ranges_m = metrics.get("ranges_m", None) or metrics.get("ranges", None)
        if isinstance(det, dict) and isinstance(det.get("pd", None), list):
            out["range_at_pd_0p9_m"] = _range_at_pd_target(ranges_m, det["pd"], pd_target=0.9)
            out["range_at_pd_0p5_m"] = _range_at_pd_target(ranges_m, det["pd"], pd_target=0.5)

    # monte_carlo: pfa empirical
    if engine in ("monte_carlo", "mc_cfar", "pfa_monte_carlo"):
        p_emp = _finite(metrics.get("pfa_empirical", None))
        p_tgt = _finite(metrics.get("pfa_target", None))
        if p_emp is not None:
            out["pfa_empirical"] = p_emp
        if p_tgt is not None:
            out["pfa_target"] = p_tgt

    # wrapper-style
    if "result" in metrics and isinstance(metrics["result"], dict):
        rep = metrics["result"]
        p_emp = _finite(rep.get("pfa_empirical", None))
        p_tgt = _finite(rep.get("pfa_target", None))
        if p_emp is not None:
            out["pfa_empirical"] = p_emp
        if p_tgt is not None:
            out["pfa_target"] = p_tgt

    # signal_level
    if engine == "signal_level":
        rd = metrics.get("rd_grid", None)
        if isinstance(rd, dict):
            nr = rd.get("n_range_bins", None)
            nd = rd.get("n_doppler_bins", None)
            if isinstance(nr, int) and isinstance(nd, int):
                out["rd_shape"] = [nr, nd]

    return out


# ---------------------------------------------------------------------
# Pareto and sensitivity
# ---------------------------------------------------------------------

def pareto_front(
    points: np.ndarray,
    *,
    maximize: Sequence[int] = (),
    minimize: Sequence[int] = (),
) -> List[int]:
    """
    Compute Pareto front indices for multi-objective points.

    Parameters
    ----------
    points : np.ndarray, shape (N, D)
        Objective vectors.
    maximize : indices to maximize
    minimize : indices to minimize

    Returns
    -------
    list[int]
        Indices of Pareto-efficient points.
    """
    if points.ndim != 2:
        raise ValueError("points must be 2D (N,D)")
    n, d = points.shape
    max_set = set(maximize)
    min_set = set(minimize)
    if max_set & min_set:
        raise ValueError("an objective index cannot be both maximize and minimize")

    # Convert to "all maximize" by negating minimization axes.
    P = points.copy().astype(float)
    for j in min_set:
        P[:, j] = -P[:, j]

    efficient = []
    for i in range(n):
        dominated = False
        for k in range(n):
            if k == i:
                continue
            # k dominates i if k >= i in all dims and > in at least one dim (maximize convention)
            if np.all(P[k] >= P[i]) and np.any(P[k] > P[i]):
                dominated = True
                break
        if not dominated:
            efficient.append(i)
    return efficient


def build_tradeoff_table(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Build a simple tradeoff table from run records.

    V1 focuses on:
    - range_at_pd_0p9_m (maximize)
    - pfa_empirical (minimize) when present
    """
    rows: List[Dict[str, Any]] = []
    for r in records:
        d = r.get("derived", {})
        rows.append(
            {
                "run_dir": r.get("run_dir"),
                "engine": d.get("engine"),
                "range_at_pd_0p9_m": d.get("range_at_pd_0p9_m"),
                "range_at_pd_0p5_m": d.get("range_at_pd_0p5_m"),
                "snr_db_min": d.get("snr_db_min"),
                "snr_db_max": d.get("snr_db_max"),
                "pfa_empirical": d.get("pfa_empirical"),
                "pfa_target": d.get("pfa_target"),
            }
        )

    # Pareto: maximize range_at_pd_0p9, minimize pfa_empirical (if both exist)
    # Build objective matrix with NaN-safe filtering.
    ranges = np.array([np.nan if x["range_at_pd_0p9_m"] is None else float(x["range_at_pd_0p9_m"]) for x in rows], dtype=float)
    pfa = np.array([np.nan if x["pfa_empirical"] is None else float(x["pfa_empirical"]) for x in rows], dtype=float)

    valid = np.isfinite(ranges) & np.isfinite(pfa)
    pareto_idx: List[int] = []
    if int(np.sum(valid)) >= 2:
        P = np.stack([ranges[valid], pfa[valid]], axis=1)
        # maximize range (col 0), minimize pfa (col 1)
        idx_local = pareto_front(P, maximize=[0], minimize=[1])
        valid_idx = np.where(valid)[0]
        pareto_idx = [int(valid_idx[i]) for i in idx_local]

    return {
        "rows": rows,
        "pareto_indices": pareto_idx,
        "objectives": [
            {"name": "range_at_pd_0p9_m", "direction": "maximize"},
            {"name": "pfa_empirical", "direction": "minimize"},
        ],
        "notes": [
            "Pareto front is computed only for rows that have both range_at_pd_0p9_m and pfa_empirical.",
            "In V1, tradeoff objectives are intentionally minimal; expand per project goals (latency/cost/MTI/etc.).",
        ],
    }


# ---------------------------------------------------------------------
# HTML rendering (self-contained fragment)
# ---------------------------------------------------------------------

def render_html_tradeoffs(table: Dict[str, Any], *, title: str = "Trade-off Summary (V1)") -> str:
    def esc(s: Any) -> str:
        return (
            str(s)
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
        )

    pareto = set(int(i) for i in table.get("pareto_indices", []))

    rows_html = []
    for i, r in enumerate(table.get("rows", [])):
        cls = "pareto" if i in pareto else ""
        rows_html.append(
            "<tr class='{cls}'>"
            "<td>{i}</td><td>{run}</td><td>{eng}</td>"
            "<td>{r90}</td><td>{pfa}</td><td>{snr}</td>"
            "</tr>".format(
                cls=cls,
                i=i,
                run=esc(r.get("run_dir")),
                eng=esc(r.get("engine")),
                r90=esc(r.get("range_at_pd_0p9_m")),
                pfa=esc(r.get("pfa_empirical")),
                snr=esc(f"{r.get('snr_db_min')} .. {r.get('snr_db_max')}"),
            )
        )

    css = """
    <style>
      .card { border: 1px solid #e3e5e8; border-radius: 8px; padding: 12px 14px; margin: 10px 0; }
      table { border-collapse: collapse; width: 100%; margin: 0.4rem 0 0.8rem 0; }
      th, td { border: 1px solid #ddd; padding: 8px; text-align: left; font-size: 0.95rem; }
      th { background: #f6f7f9; }
      tr.pareto td { font-weight: 700; background: #fcfcff; }
      .muted { color: #666; }
    </style>
    """

    notes = table.get("notes", [])
    notes_html = "<ul>" + "".join(f"<li>{esc(n)}</li>" for n in notes) + "</ul>"

    html = f"""
    {css}
    <div class="card">
      <h2>{esc(title)}</h2>
      <div class="muted">Pareto-highlighted rows are bold (maximize range, minimize Pfa).</div>
      <table>
        <thead>
          <tr>
            <th>#</th><th>Run dir</th><th>Engine</th>
            <th>Range@Pd≥0.9 (m)</th><th>Pfa empirical</th><th>SNR dB (min..max)</th>
          </tr>
        </thead>
        <tbody>
          {''.join(rows_html)}
        </tbody>
      </table>
      <h3>Interpretation notes</h3>
      {notes_html}
    </div>
    """
    return html.strip()


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(prog="core.metrics.tradeoffs", description="Trade-off analysis over many runs.")
    ap.add_argument("--runs", required=True, help="Root folder containing multiple run dirs (metrics.json inside).")
    ap.add_argument("--json", dest="json_out", default=None, help="Write normalized tradeoff table JSON.")
    ap.add_argument("--html", dest="html_out", default=None, help="Write self-contained HTML fragment.")
    return ap.parse_args()


def main() -> int:
    args = _parse_args()
    root = Path(args.runs)

    records = load_runs(root)
    table = build_tradeoff_table(records)

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(table, indent=2, sort_keys=True), encoding="utf-8")

    if args.html_out:
        Path(args.html_out).write_text(render_html_tradeoffs(table) + "\n", encoding="utf-8")

    print(json.dumps(table, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())