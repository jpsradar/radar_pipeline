"""
cli/make_report.py

CLI entrypoint to generate reports from sweep results.

What this script does
---------------------
- Loads a sweep results JSON produced by cli/run_sweep.py
- Generates a deterministic report directory:
    - report.json (summary + KPIs + Pareto indices)
    - plots/ (PNG figures)
- Does NOT re-run simulations; it only post-processes sweep outputs.

Inputs (CLI)
------------
- --sweep-json: path to sweep results JSON (required)
- --out-dir: output directory for the report (required)
- --objectives: optional multi-objective definition, repeatable:
    --objective "<key>:<min|max>"

Objective key syntax (supported)
-------------------------------
- far.per_second
- far.per_scan
- snr_db@<range_m>
- pd@<range_m>

Examples
--------
1) Basic report (no Pareto):
    python -m cli.make_report --sweep-json results/sweeps/sweep.json --out-dir results/reports/sweep_report

2) Pareto report (2 objectives):
    python -m cli.make_report \
      --sweep-json results/sweeps/sweep.json \
      --out-dir results/reports/sweep_report \
      --objective "far.per_second:min" \
      --objective "snr_db@10000:max"
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional

from reports.generators import generate_sweep_report
from sweeps.pareto import Direction


class CLIError(ValueError):
    """Raised when CLI arguments are invalid."""


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="make_report",
        description="Generate report artifacts from sweep results JSON.",
    )

    parser.add_argument(
        "--sweep-json",
        required=True,
        help="Path to sweep results JSON produced by cli/run_sweep.py.",
    )
    parser.add_argument(
        "--out-dir",
        required=True,
        help="Output directory for report.json and plots/.",
    )
    parser.add_argument(
        "--objective",
        action="append",
        default=[],
        help=(
            "Objective definition in the form '<key>:<min|max>'. "
            "Repeatable, e.g. --objective 'far.per_second:min' --objective 'snr_db@10000:max'."
        ),
    )

    return parser.parse_args()


def _parse_objectives(items: List[str]) -> Optional[Dict[str, Direction]]:
    """
    Parse repeatable --objective arguments into a directions dict.

    Parameters
    ----------
    items : list[str]
        Each item must be '<key>:<min|max>'.

    Returns
    -------
    dict[str, Direction] | None
        Parsed objectives, or None if no objectives were provided.
    """
    if not items:
        return None

    out: Dict[str, Direction] = {}
    for raw in items:
        if ":" not in raw:
            raise CLIError(f"Invalid --objective '{raw}'. Expected '<key>:<min|max>'.")
        key, direction = raw.split(":", 1)
        key = key.strip()
        direction = direction.strip().lower()

        if not key:
            raise CLIError(f"Invalid --objective '{raw}': empty key.")
        if direction not in ("min", "max"):
            raise CLIError(f"Invalid --objective '{raw}': direction must be 'min' or 'max'.")

        out[key] = direction  # type: ignore[assignment]

    if len(out) < 1:
        return None
    return out


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> int:
    args = _parse_args()

    sweep_json_path = Path(args.sweep_json)
    out_dir = Path(args.out_dir)

    if not sweep_json_path.exists():
        print(f"[ERROR] sweep JSON not found: {sweep_json_path}")
        return 2

    try:
        objectives = _parse_objectives(list(args.objective))
    except CLIError as exc:
        print(f"[ERROR] {exc}")
        return 3

    try:
        generate_sweep_report(
            sweep_json_path=sweep_json_path,
            out_dir=out_dir,
            objectives=objectives,
        )
    except Exception as exc:
        print(f"[ERROR] Report generation failed: {exc}")
        return 4

    print(f"[OK] Report complete: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())