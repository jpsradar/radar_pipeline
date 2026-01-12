"""
cli/make_case_report.py

Generate a polished HTML report for a single case output directory.

Why this exists
---------------
- `cli.run_case` produces machine-readable artifacts (metrics.json, case_manifest.json).
- Recruiters and stakeholders need a *human-facing* artifact: `report.html` + plots.

Inputs
------
- --in: path to a case directory or metrics.json
- --out: optional output directory (default: same directory as metrics.json)
- --title: optional HTML title

Outputs
-------
- report.html (standalone, plots embedded as base64)
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
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

from reports.case_generators import generate_case_report_html


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="make_case_report",
        description="Generate an HTML report for a single radar pipeline case run.",
    )
    p.add_argument(
        "--in",
        dest="inp",
        required=True,
        help="Case directory or metrics.json path (e.g., results/cases/<run>/ or .../metrics.json).",
    )
    p.add_argument(
        "--out",
        default=None,
        help="Output directory (default: case directory).",
    )
    p.add_argument(
        "--title",
        default=None,
        help="Optional report title override.",
    )
    return p.parse_args()


def _resolve_inputs(inp: Path) -> tuple[Path, Optional[Path], Optional[Path]]:
    """
    Resolve case_dir, metrics_path, manifest_path from a user-provided input path.

    Rules:
    - If inp is a directory: metrics=inp/metrics.json, manifest=inp/case_manifest.json if exists
    - If inp is a file named metrics.json: case_dir=parent, metrics=inp, manifest=parent/case_manifest.json if exists
    """
    inp = inp.resolve()
    if inp.is_dir():
        case_dir = inp
        metrics = case_dir / "metrics.json"
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