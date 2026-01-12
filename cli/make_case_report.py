"""
cli/make_case_report.py

Standalone HTML report generator for a *single* case output directory.

Why this exists
---------------
- `cli.run_case --report` is the main path (best for day-to-day runs).
- This module provides a post-processing entry point when you already have a run directory
  and only want to regenerate the report (e.g., after changing report styling).

Inputs
------
--in   : Path to a case directory OR to a metrics.json file inside it.
--out  : Optional output directory (defaults to the case directory).
--title: Optional HTML title override.

Outputs
-------
- report.html (standalone, plots embedded as base64 PNG)
- plots/*.png (also written to disk)

Usage
-----
    python -m cli.make_case_report --in results/cases/<run_dir>
    python -m cli.make_case_report --in results/cases/<run_dir>/metrics.json

Exit codes
----------
0 : success
2 : input missing/invalid
3 : unexpected exception

Notes
-----
This is intentionally orchestration only; rendering lives in `reports.case_generators`.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

from reports.case_generators import generate_case_report_html


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="make_case_report",
        description="Generate an HTML report for a single radar pipeline case run.",
    )
    parser.add_argument(
        "--in",
        dest="inp",
        required=True,
        help="Case directory or metrics.json path (e.g., results/cases/<run>/ or .../metrics.json).",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output directory (default: case directory).",
    )
    parser.add_argument(
        "--title",
        default=None,
        help="Optional report title override.",
    )
    return parser.parse_args()


def _resolve_inputs(inp: Path) -> tuple[Path, Path, Optional[Path]]:
    """
    Resolve case_dir, metrics_path, manifest_path from a user-provided input path.

    Rules
    -----
    - If inp is a directory:
        metrics = inp/metrics.json
        manifest = inp/case_manifest.json if present
    - If inp is a file named metrics.json:
        case_dir = inp.parent
        manifest = case_dir/case_manifest.json if present
    """
    inp = inp.resolve()

    if inp.is_dir():
        case_dir = inp
        metrics = case_dir / "metrics.json"
        if not metrics.exists():
            raise FileNotFoundError(f"metrics.json not found in case dir: {case_dir}")
        manifest = case_dir / "case_manifest.json"
        return case_dir, metrics, (manifest if manifest.exists() else None)

    if inp.is_file():
        if inp.name != "metrics.json":
            raise ValueError("If --in is a file, it must be metrics.json.")
        case_dir = inp.parent
        manifest = case_dir / "case_manifest.json"
        return case_dir, inp, (manifest if manifest.exists() else None)

    raise FileNotFoundError(f"Input path not found: {inp}")


def main() -> int:
    args = _parse_args()
    try:
        inp = Path(args.inp)
        case_dir, metrics_path, manifest_path = _resolve_inputs(inp)

        out_dir = Path(args.out).resolve() if args.out else case_dir
        paths = generate_case_report_html(
            case_dir=case_dir,
            metrics_path=metrics_path,
            manifest_path=manifest_path,
            out_dir=out_dir,
            title=args.title,
        )
        print(f"[OK] Wrote report: {paths.report_html}")
        print(f"[OK] Wrote plots:  {paths.plots_dir}")
        return 0

    except (FileNotFoundError, ValueError) as exc:
        print(f"[ERROR] {exc}")
        return 2
    except Exception as exc:
        print(f"[ERROR] Unexpected exception: {exc}")
        return 3


if __name__ == "__main__":
    raise SystemExit(main())