#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash aux_scripts/sanitize_case_outputs.sh results/cases/_smoke_case
#
# What it does:
# - Replaces absolute repo root occurrences with ${PROJECT_ROOT}
# - Redacts user/hostname (inside environment.user/environment.hostname if present)
# - Redacts any remaining absolute path strings under known prefixes

CASE_DIR="${1:-}"
if [[ -z "${CASE_DIR}" ]]; then
  echo "ERROR: Missing case directory argument."
  echo "Usage: $0 results/cases/_smoke_case"
  exit 2
fi

MANIFEST="${CASE_DIR}/case_manifest.json"
REPORT="${CASE_DIR}/report.html"

if [[ ! -f "${MANIFEST}" ]]; then
  echo "ERROR: Manifest not found: ${MANIFEST}"
  exit 2
fi
if [[ ! -f "${REPORT}" ]]; then
  echo "ERROR: Report not found: ${REPORT}"
  exit 2
fi

# Infer PROJECT_ROOT from standard layout: <repo>/results/cases/<case>
PROJECT_ROOT="${PROJECT_ROOT:-}"
if [[ -z "${PROJECT_ROOT}" ]]; then
  PROJECT_ROOT="$(cd "${CASE_DIR}/../../.." && pwd)"
fi

echo "[INFO] Sanitizing case artifacts..."
echo "[INFO]   CASE_DIR     = ${CASE_DIR}"
echo "[INFO]   PROJECT_ROOT = ${PROJECT_ROOT}"

# Pick python from the active environment (prefer python, fallback python3)
PY_BIN="${PY_BIN:-}"
if [[ -z "${PY_BIN}" ]]; then
  if command -v python >/dev/null 2>&1; then
    PY_BIN="python"
  elif command -v python3 >/dev/null 2>&1; then
    PY_BIN="python3"
  else
    echo "ERROR: Neither 'python' nor 'python3' found in PATH."
    exit 3
  fi
fi

CASE_DIR="${CASE_DIR}" PROJECT_ROOT="${PROJECT_ROOT}" "${PY_BIN}" - <<'PY'
import json
import os
import re
from pathlib import Path

case_dir = Path(os.environ["CASE_DIR"])
project_root = os.environ["PROJECT_ROOT"]

manifest_path = case_dir / "case_manifest.json"
report_path = case_dir / "report.html"

ABS_PREFIXES = (
    "/Users/",
    "/home/",
    "C:\\Users\\",
    "C:\\\\Users\\\\",
)

def replace_root(s: str) -> str:
    return s.replace(project_root, "${PROJECT_ROOT}")

def redact_abs_paths(s: str) -> str:
    # Conservative: replace any obvious absolute prefix that remains
    for pref in ABS_PREFIXES:
        if pref in s:
            # Keep only basename after last slash/backslash for minimal usefulness
            base = re.split(r"[\\/]", s)[-1] or "path"
            return f"<ABSOLUTE_PATH_REDACTED>:{base}"
    return s

# --- report.html
report = report_path.read_text(encoding="utf-8", errors="replace")
report2 = redact_abs_paths(replace_root(report))
if report2 != report:
    report_path.write_text(report2, encoding="utf-8")
print(f"[OK] Sanitized: {report_path}")

# --- manifest json
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

def walk(obj):
    if isinstance(obj, dict):
        for k, v in list(obj.items()):
            if isinstance(v, (dict, list)):
                walk(v)
            elif isinstance(v, str):
                obj[k] = redact_abs_paths(replace_root(v))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            if isinstance(v, (dict, list)):
                walk(v)
            elif isinstance(v, str):
                obj[i] = redact_abs_paths(replace_root(v))

walk(manifest)

# Real redactions live here in your manifest:
env = manifest.get("environment", {})
if isinstance(env, dict):
    if "user" in env:
        env["user"] = "REDACTED"
    if "hostname" in env:
        env["hostname"] = "REDACTED"

manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
print(f"[OK] Sanitized: {manifest_path}")

# Quick verification
text_report = report_path.read_text(encoding="utf-8", errors="replace")
text_manifest = manifest_path.read_text(encoding="utf-8", errors="replace")
hits = []
for pref in ABS_PREFIXES:
    if pref in text_report or pref in text_manifest:
        hits.append(pref)

if hits:
    print("[WARN] Found potential leaks for prefixes:", hits)
else:
    print("[OK] No obvious absolute-path leaks")
PY

echo "[OK] Done."