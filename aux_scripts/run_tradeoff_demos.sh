#!/usr/bin/env bash
set -euo pipefail

###############################################################################
# aux_scripts/run_tradeoff_demos.sh
#
# Reproducible engineering trade-off demo runner for the radar pipeline.
#
# Intent
# ------
# Run a curated set of *system-level trade-off* demonstrations using only:
#   - committed case configs (YAML)
#   - committed engines (model_based / monte_carlo / signal_level)
#   - deterministic seeds
#
# The goal is to make it obvious (to an external radar engineer) that:
#   - performance is probabilistic (Pd/Pfa/FAR), not point-valued
#   - the radar is a coupled system: improving one knob often degrades another
#   - model vs experiment are separated and compared, not conflated
#
# What this script runs
# ---------------------
# 1) Integration vs FAR trade-off (model_based sweep)
# 2) Antenna gain vs coverage / revisit trade-off (model_based cases)
# 3) CFAR robustness trade-off (monte_carlo cases)
# 4) Detector vs environment trade-off (CA vs OS)
# 5) Pd under noise with target fluctuations (Swerling 0–4)
#    - Runs any committed configs matching:
#        configs/cases/demo_pd_noise_swerling*.yaml
#    - Writes comparison artifacts (CSV+JSON) under results/comparisons/
#
# Reproducibility + safety
# ------------------------
# - All runs are seeded.
# - Outputs are written under results/ only.
# - Safe cleanup only touches matching directories under results/.
#
# Usage
# -----
#   ./aux_scripts/run_tradeoff_demos.sh
#
# Optional environment variables
# ------------------------------
#   SEED=123        Deterministic seed tag (default: 123)
#   OVERWRITE=1     Allow overwriting existing result directories
#
###############################################################################

SEED="${SEED:-123}"
export SEED

OVERWRITE_FLAG=""
if [[ "${OVERWRITE:-0}" == "1" ]]; then
  OVERWRITE_FLAG="--overwrite"
fi

echo "============================================================"
echo " Radar pipeline – trade-off demo suite"
echo " Seed       : ${SEED}"
echo " Overwrite  : ${OVERWRITE:-0}"
echo "============================================================"
echo

# -----------------------------------------------------------------------------
# Helper: safe cleanup of results subdirs without zsh glob failures
# -----------------------------------------------------------------------------
cleanup_dirs() {
  # Usage: cleanup_dirs <base_dir> <name_glob>
  local base_dir="$1"
  local name_glob="$2"

  if [[ -d "${base_dir}" ]]; then
    find "${base_dir}" -maxdepth 1 -mindepth 1 -type d -name "${name_glob}" -exec rm -rf {} + 2>/dev/null || true
  fi
}

# -----------------------------------------------------------------------------
# Helper: run a case only if the config exists (keeps the runner forward-compatible)
# -----------------------------------------------------------------------------
run_case_if_exists() {
  # Usage: run_case_if_exists <yaml_path>
  local case_path="$1"
  if [[ -f "${case_path}" ]]; then
    python -m cli.run_case \
      --case "${case_path}" \
      --engine auto \
      --seed "${SEED}" \
      ${OVERWRITE_FLAG} \
      --strict
  else
    echo "[SKIP] Missing config: ${case_path}"
  fi
}

###############################################################################
# 1) Integration (N pulses) vs FAR / Pd
###############################################################################
echo "[1/5] Integration vs FAR trade-off (sweep)"

cleanup_dirs "results/sweeps" "demo_pd_noise__integration_vs_far__seed${SEED}__cfg*"

python aux_scripts/sweep_integration_vs_far.py \
  --case configs/cases/demo_pd_noise.yaml \
  --seed "${SEED}" \
  ${OVERWRITE_FLAG} \
  --strict

echo

###############################################################################
# 2) Antenna gain vs coverage / revisit trade-off
###############################################################################
echo "[2/5] Antenna gain vs coverage trade-off"

cleanup_dirs "results/cases" "demo_antenna_vs_coverage_*__seed${SEED}__cfg*"

run_case_if_exists "configs/cases/demo_antenna_vs_coverage_low_gain.yaml"
run_case_if_exists "configs/cases/demo_antenna_vs_coverage_high_gain.yaml"

echo

###############################################################################
# 3) CFAR robustness under homogeneous vs non-homogeneous clutter
###############################################################################
echo "[3/5] CFAR robustness trade-off (Monte Carlo)"

cleanup_dirs "results/cases" "demo_clutter_cfar*__seed${SEED}__cfg*"

run_case_if_exists "configs/cases/demo_clutter_cfar.yaml"
run_case_if_exists "configs/cases/demo_clutter_cfar_hetero.yaml"

echo

###############################################################################
# 4) Detector vs environment: CA-CFAR vs OS-CFAR under non-homogeneous background
###############################################################################
echo "[4/5] Detector vs environment trade-off (CA vs OS)"

cleanup_dirs "results/cases" "demo_detector_vs_environment_*__seed${SEED}__cfg*"
cleanup_dirs "results/comparisons" "demo_detector_vs_environment__seed${SEED}"

run_case_if_exists "configs/cases/demo_detector_vs_environment_ca.yaml"
run_case_if_exists "configs/cases/demo_detector_vs_environment_os.yaml"

# Compare newest CA vs OS runs (by seed) and write CSV+JSON artifacts under results/comparisons/
python - <<'PY'
from __future__ import annotations

import csv
import json
import os
from pathlib import Path

ROOT = Path.cwd()
SEED = int(os.environ.get("SEED", "123"))

def newest_metrics(case_prefix: str) -> Path:
    base = ROOT / "results/cases"
    pat = f"{case_prefix}__monte_carlo__seed{SEED}__cfg*/metrics.json"
    candidates = sorted(base.glob(pat))
    if not candidates:
        raise SystemExit(f"[ERROR] No metrics.json found for prefix={case_prefix} (pattern={pat})")
    return candidates[-1]

def load(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8"))

def pick(m: dict) -> dict:
    pfa_t = float(m.get("pfa_target"))
    pfa_e = float(m.get("pfa_empirical"))
    det = str(m.get("detector"))
    ci = (m.get("confidence_intervals", {}) or {}).get("wilson_95", {}) or {}
    lo = float(ci.get("low"))
    hi = float(ci.get("high"))
    return {
        "detector": det,
        "pfa_target": pfa_t,
        "pfa_empirical": pfa_e,
        "abs_error": abs(pfa_e - pfa_t),
        "wilson95_low": lo,
        "wilson95_high": hi,
    }

run_ca = newest_metrics("demo_detector_vs_environment_ca")
run_os = newest_metrics("demo_detector_vs_environment_os")

m_ca = pick(load(run_ca))
m_os = pick(load(run_os))
ok = m_os["abs_error"] <= m_ca["abs_error"]

out_dir = ROOT / "results/comparisons" / f"demo_detector_vs_environment__seed{SEED}"
out_dir.mkdir(parents=True, exist_ok=True)

rows = [
    {"case": "CA", **m_ca},
    {"case": "OS", **m_os},
]

csv_path = out_dir / "comparison.csv"
with csv_path.open("w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)

json_path = out_dir / "comparison.json"
json_path.write_text(
    json.dumps({"rows": rows, "expectation_passed": ok}, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)

pretty = Path("${PROJECT_ROOT}") / out_dir.relative_to(ROOT)

print("[OK] Detector-vs-environment comparison written:")
print(f"  {pretty / 'comparison.csv'}")
print(f"  {pretty / 'comparison.json'}")
print("[RESULT] abs_error(CA) =", m_ca["abs_error"], "abs_error(OS) =", m_os["abs_error"])
print("[RESULT] expectation OS closer than CA:", ok)

if not ok:
    print("[WARN] OS was NOT closer than CA in this run. Consider adjusting rank_frac or hetero strength.")
PY

echo

###############################################################################
# 5) Pd under noise with target fluctuations (Swerling 0–4)
###############################################################################
echo "[5/5] Pd under noise with target fluctuations (Swerling 0–4)"

cleanup_dirs "results/cases" "demo_pd_noise_swerling*__seed${SEED}__cfg*"
cleanup_dirs "results/comparisons" "demo_pd_noise_swerling__seed${SEED}"

# Run every committed Pd/Swerling case matching the pattern (sorted for determinism).
mapfile -t SWERLING_CASES < <(
  find "configs/cases" -maxdepth 1 -type f -name "demo_pd_noise_swerling*.yaml" 2>/dev/null | sort
)

if [[ "${#SWERLING_CASES[@]}" -eq 0 ]]; then
  echo "[WARN] No Swerling Pd configs found (expected pattern: configs/cases/demo_pd_noise_swerling*.yaml)"
else
  for f in "${SWERLING_CASES[@]}"; do
    run_case_if_exists "${f}"
  done

  # Build a comparison table from the newest metrics of each swerling case.
  python - <<'PY'
from __future__ import annotations

import csv
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path.cwd()
SEED = int(os.environ.get("SEED", "123"))

CASE_DIR = ROOT / "results/cases"
OUT_DIR = ROOT / "results/comparisons" / f"demo_pd_noise_swerling__seed{SEED}"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RX_SW = re.compile(r"(swerling[0-4])", re.IGNORECASE)

def load_json(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8"))

def newest_metrics_for_case_stem(case_stem: str) -> Optional[Path]:
    pat = f"{case_stem}__model_based__seed{SEED}__cfg*/metrics.json"
    c = sorted(CASE_DIR.glob(pat))
    return c[-1] if c else None

def extract_pd(metrics: Dict[str, Any]) -> Tuple[List[float], List[float]]:
    det = metrics.get("detection", {}) or {}
    pd = det.get("pd", None)
    ranges = (metrics.get("metrics", {}) or {}).get("ranges_m", None)

    # In this repo, ranges are usually in metrics.ranges_m, but we stay defensive.
    if ranges is None:
        ranges = metrics.get("ranges_m", None)

    if not isinstance(pd, list) or not pd:
        return ([], [])
    if not isinstance(ranges, list) or len(ranges) != len(pd):
        # If ranges missing/mismatched, still return pd; ranges empty => no range_at_pd.
        return ([float(x) for x in pd], [])
    return ([float(x) for x in pd], [float(r) for r in ranges])

def range_at_pd(ranges_m: List[float], pd: List[float], thr: float) -> Optional[float]:
    if not ranges_m or not pd or len(ranges_m) != len(pd):
        return None
    # Define "range_at_pd" as the *max* range where Pd >= thr (engineering-friendly).
    ok = [r for r, p in zip(ranges_m, pd) if p >= thr]
    return max(ok) if ok else None

# Discover case stems from committed configs (stable intent)
cfgs = sorted((ROOT / "configs/cases").glob("demo_pd_noise_swerling*.yaml"))
case_stems = [p.stem for p in cfgs]

rows: List[Dict[str, Any]] = []
missing: List[str] = []

for stem in case_stems:
    mp = newest_metrics_for_case_stem(stem)
    if mp is None:
        missing.append(stem)
        continue

    m = load_json(mp)
    pd, ranges = extract_pd(m)

    # Infer fluctuation from case stem (demo_pd_noise_swerlingX)
    m_sw = RX_SW.search(stem)
    fluct = (m_sw.group(1).lower() if m_sw else "unknown")

    pd0 = (pd[0] if pd else None)
    row = {
        "case": stem,
        "fluctuation": fluct,
        "pd_at_first_range": pd0,
        "min_pd": (min(pd) if pd else None),
        "max_pd": (max(pd) if pd else None),
        "range_at_pd_50_m": range_at_pd(ranges, pd, 0.50),
        "range_at_pd_80_m": range_at_pd(ranges, pd, 0.80),
        "range_at_pd_90_m": range_at_pd(ranges, pd, 0.90),
        "run_metrics": str(mp.parent),
    }
    rows.append(row)

# Write artifacts
csv_path = OUT_DIR / "comparison.csv"
json_path = OUT_DIR / "comparison.json"

if rows:
    # Stable column order
    fieldnames = [
        "case",
        "fluctuation",
        "pd_at_first_range",
        "min_pd",
        "max_pd",
        "range_at_pd_50_m",
        "range_at_pd_80_m",
        "range_at_pd_90_m",
        "run_metrics",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

payload = {
    "seed": SEED,
    "rows": rows,
    "missing_cases": missing,
}
json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

pretty = Path("${PROJECT_ROOT}") / OUT_DIR.relative_to(ROOT)
print("[OK] Swerling Pd comparison written:")
print(f"  {pretty / 'comparison.csv'}")
print(f"  {pretty / 'comparison.json'}")
if missing:
    print("[WARN] Missing runs for:", ", ".join(missing))
PY
fi

echo
echo "============================================================"
echo "[OK] Trade-off demo suite complete."
echo " Artifacts available under: results/"
echo "============================================================"