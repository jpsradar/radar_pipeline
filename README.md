# Radar Performance Pipeline

Model-based radar performance and simulation pipeline for quantitative trade-off analysis
(Pd/Pfa ↔ FAR load, integration, CFAR behavior, and system-level constraints).

## Quickstart

```bash
git clone <repo-url>
cd radar_pipeline

python -m venv .venv
source .venv/bin/activate

pip install -U pip
pip install -r requirements.txt
pip install -e ".[dev]"
```

Run a demo case:
```bash
radar-run-case --case configs/cases/demo_pd_noise.yaml --report
```

Run a sweep:
```bash
radar-run-sweep \
  --case configs/cases/demo_pd_noise.yaml \
  --sweep configs/sweeps/demo_pfa_sweep.yaml \
  --out results/sweeps/demo \
  --report
```

Outputs are written to:
- `results/cases/<run_name>/`
- `results/sweeps/<sweep_name>/report/`

## Repository Structure

- `core/` – physics-based models (budgets, detection, CFAR, targets, environment)
- `cli/` – executable entrypoints (cases, sweeps)
- `validation/` – golden tests + Monte Carlo sanity checks
- `reports/` – HTML reports and plots
- `configs/` – YAML case and sweep definitions
- `sweeps/` – DOE / Pareto / sensitivity machinery

## What to Look At

- `demo_pd_noise.yaml` – detection performance vs range
- CFAR Monte Carlo validation in `validation/`
- HTML reports generated with `--report`
