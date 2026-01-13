#!/usr/bin/env bash
set -euo pipefail

###############################################################################
# aux_scripts/run_tradeoff_demos.sh
#
# Reproducible trade-off demo runner for the radar pipeline.
#
# Purpose
# -------
# This script executes a curated set of *engineering trade-off demonstrations*
# using only committed configs and engines. It is intended for:
#
# - Manual technical review
# - Pre-report artifact generation
# - Sanity-level regression checking (non-gating)
#
# This script is NOT a CI gate and does NOT encode internal development phases.
# It simply runs representative demos that expose fundamental radar trade-offs.
#
# What this script runs
# ---------------------
# 1) Integration vs FAR trade-off (model_based sweep)
#    - Increasing N improves Pd but reduces FAR per second (longer CPI).
#
# 2) Antenna gain vs coverage/revisit trade-off (model_based cases)
#    - Higher gain improves SNR/Pd but worsens scan time / revisit rate.
#
# 3) CFAR robustness trade-off (monte_carlo cases)
#    - Homogeneous background: CA-CFAR holds target Pfa.
#    - Non-homogeneous background (CUT-only scaling): CFAR breaks.
#
# Artifacts
# ---------
# All outputs are written under results/ and are safe to delete.
# No files outside results/ are modified.
#
# Usage
# -----
#   ./aux_scripts/run_tradeoff_demos.sh
#
# Optional environment variables
# ------------------------------
#   SEED=123          Deterministic seed tag (default: 123)
#   OVERWRITE=1       Allow overwriting results directories
#
###############################################################################

SEED="${SEED:-123}"
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

###############################################################################
# 1) Integration (N pulses) vs FAR / Pd
###############################################################################

echo "[1/3] Integration vs FAR trade-off (sweep)"

python aux_scripts/sweep_integration_vs_far.py \
  --case configs/cases/demo_pd_noise.yaml \
  --seed "${SEED}" \
  ${OVERWRITE_FLAG} \
  --strict

echo

###############################################################################
# 2) Antenna gain vs coverage / revisit trade-off
###############################################################################

echo "[2/3] Antenna gain vs coverage trade-off"

rm -rf results/cases/demo_antenna_vs_coverage_* || true

python -m cli.run_case \
  --case configs/cases/demo_antenna_vs_coverage_low_gain.yaml \
  --engine auto \
  --seed "${SEED}" \
  ${OVERWRITE_FLAG} \
  --strict

python -m cli.run_case \
  --case configs/cases/demo_antenna_vs_coverage_high_gain.yaml \
  --engine auto \
  --seed "${SEED}" \
  ${OVERWRITE_FLAG} \
  --strict

echo

###############################################################################
# 3) CFAR robustness under homogeneous vs non-homogeneous clutter
###############################################################################

echo "[3/3] CFAR robustness trade-off (Monte Carlo)"

rm -rf results/cases/demo_clutter_cfar* || true

python -m cli.run_case \
  --case configs/cases/demo_clutter_cfar.yaml \
  --engine auto \
  --seed "${SEED}" \
  ${OVERWRITE_FLAG} \
  --strict

python -m cli.run_case \
  --case configs/cases/demo_clutter_cfar_hetero.yaml \
  --engine auto \
  --seed "${SEED}" \
  ${OVERWRITE_FLAG} \
  --strict

echo
echo "============================================================"
echo "[OK] Trade-off demo suite complete."
echo " Artifacts available under: results/"
echo "============================================================"