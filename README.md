# Radar Performance Pipeline

A reproducible radar performance analysis pipeline for system-level trade-off studies  
under explicit statistical assumptions and operational constraints.

This repository provides an engineering framework to analyze how radar design  
choices—antenna, waveform, integration, detection strategy, and environment—  
affect probability of detection (Pd), false alarms, false alarm rate (FAR),  
and coverage.

The emphasis is on engineering trade-offs, statistical validity, and limits—  
not on isolated formulas, opaque simulators, or black-box performance claims.

---

## Project positioning

This repository implements system-level radar performance modeling with:

- explicit coupling between Pd, Pfa, and FAR  
- analytic and Monte Carlo evaluation modes  
- reproducible trade-off experiments  
- configuration-driven execution  

It is intended as a technical reference implementation for radar performance  
analysis, detection theory validation, and system-level reasoning.

---

## Scope and intent

This pipeline is designed to support questions such as:

- How does increased integration improve Pd while simultaneously reshaping system-level FAR?
- How does a fixed per-decision Pfa translate into real false-alarm load once geometry is considered?
- When do analytic detection models remain valid, and when do they diverge from Monte Carlo?
- Which impairment dominates first: noise, clutter heterogeneity, geometry, or detector choice?

The repository is intended for architecture-level reasoning and comparative analysis.  
It is not a real-time signal processor, tracker, or mission-level simulator.

---

## Modeling philosophy

### System-level coupling

Radar performance is treated as a coupled system problem.

Changes to any component—antenna gain, integration length, detector choice,  
scan geometry, or environment—necessarily affect multiple performance metrics.  
There is no notion of a universally “best” configuration—only explicit trade-offs.

---

### Explicit statistical treatment

All reported performance metrics are probabilistic by construction:

- Pd, Pfa, and FAR are modeled explicitly  
- Noise, clutter, and target fluctuations are treated statistically  
- Monte Carlo simulation is used for validation, not as a replacement for models  

Results are expected to vary with assumptions and environment.

---

### Separation of model and experiment

Each run declares its execution mode explicitly:

- model_based — analytic or closed-form evaluation  
- monte_carlo — empirical simulation  
- signal_level — waveform-level simulation (where applicable)  

Analytic predictions and Monte Carlo results are compared directly.  
Discrepancies are preserved as first-class outputs, not smoothed away.

---

### Declared validity and limits

Every execution records a validity contract describing:

- statistical assumptions  
- clutter regime  
- operational and modeling limits  

If assumptions are violated, results are not silently trusted.

---

## Reproducibility and traceability

Each execution produces a self-contained, auditable result bundle:

results/cases/<run_id>/
├── metrics.json
├── manifest.json
├── case_manifest.json

The run identifier encodes:

- case name  
- execution engine  
- random seed  
- configuration hash  

All results are reproducible given the same inputs.

Generated artifacts (results/, reports/, etc.) are not tracked by git  
and are intended to be regenerated.

---

## Configuration-driven architecture

Radar definitions are externalized into YAML configurations:

- waveform and PRF  
- antenna and scan geometry  
- detection and CFAR parameters  
- environment and target models  

The codebase avoids hidden constants.  
Changing a configuration is expected to change performance.

JSON schemas enforce structure and prevent ambiguous or underspecified cases.

---

## Testing and validation

The repository includes deterministic tests that assert physical and statistical invariants:

- radar equation scaling  
- unit consistency  
- geometry and count consistency  
- FAR scaling behavior  
- physical monotonicity (e.g., Pd vs SNR, range, integration length)  

Golden tests and Monte Carlo validation are used to ensure regression-free evolution.

---

## What is intentionally not modeled

The following are outside the scope of this repository:

- multipath, ducting, and complex propagation effects  
- advanced tracking and track-before-detect  
- hardware-specific non-idealities beyond first-order models  
- real-time or embedded execution constraints  

This pipeline is intended as an upstream analysis and reasoning tool.

---

## Where this pipeline fails (by design)

This repository intentionally exposes regimes where common radar assumptions break.  
These are not implementation bugs—they are the primary signals used for engineering judgment.

Documented failure modes include:

- CFAR under heterogeneous clutter  
  CA-CFAR does not maintain the requested Pfa in heterogeneous environments.

- Analytic Pd vs Monte Carlo divergence  
  Closed-form detection models diverge at low SNR, under strong fluctuations, and near threshold.

- Pfa is not an operational metric by itself  
  FAR depends on the full trial rate (range × Doppler × beams × scan rate).

- Model validity is conditional  
  Results are valid only within declared statistical and operational assumptions.

---

## Repository structure

- core/ — physics-based models  
- cli/ — deterministic orchestration  
- configs/ — YAML definitions and schemas  
- tests/ — invariants and consistency checks  
- validation/ — Monte Carlo validation  
- notebook/ — reproducible walkthrough  
- reports/ — generated artifacts  

---

## Quick start (verified from clean clone)

git clone <repo_url>
cd radar_pipeline

python -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
pip install jupyter nbconvert nbformat ipykernel

pytest -q
ruff check .

jupyter nbconvert \
  --to notebook \
  --execute notebook/radar_pipeline_walkthrough.ipynb \
  --output radar_pipeline_walkthrough.executed.ipynb \
  --output-dir /tmp
  
  
Expected:

- All tests pass  
- Notebook executes without errors  
- Output generated under:

results/notebook_cases/.../

---

## Reproducible walkthrough

A complete demonstration is provided in:

notebook/radar_pipeline_walkthrough.ipynb

Running the notebook end-to-end produces:

- integration vs FAR trade-offs  
- Pd scaling with integration  
- empirical vs analytic comparisons  

All outputs are written to results/ and can be regenerated.

---

## Typical workflow

1. Define a radar scenario via YAML configuration  
2. Run model-based evaluation  
3. Validate with Monte Carlo simulation  
4. Inspect discrepancies and limits  
5. Modify parameters and observe trade-offs  

The pipeline is structured to make these steps explicit, repeatable, and auditable.

---

## What this project demonstrates

- Translation of radar detection theory into executable models  
- Explicit handling of statistical detection limits  
- Reproducible experiment design  
- Separation of assumptions, models, and validation  
- Visibility of real-world failure modes in radar systems