#!/usr/bin/env bash
set -euo pipefail

echo "[1/3] ruff"
python -m ruff check .

echo "[2/3] pytest"
pytest -q

echo "[3/3] sanity checks"
python -m validation.sanity_checks

echo "[OK] Validation complete"
