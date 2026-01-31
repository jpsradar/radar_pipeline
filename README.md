# Radar Performance Pipeline

A reproducible radar performance analysis pipeline for **system-level trade-off studies**
under explicit statistical assumptions and operational constraints.

This repository provides an engineering framework to analyze how radar design
choices—antenna, waveform, integration, detection strategy, and environment—
affect **probability of detection (Pd)**, **false alarms**, **false alarm rate (FAR)**,
and **coverage**.

The emphasis is on **engineering trade-offs, statistical validity, and limits**—
not on isolated formulas, opaque simulators, or black-box performance claims.

---

## Scope and intent

This pipeline is designed to support questions such as:

- How does increased integration improve Pd while simultaneously reshaping system-level FAR?
- How does a fixed per-decision Pfa translate into real false-alarm load once geometry is considered?
- When do analytic detection models remain valid, and when do they diverge from Monte Carlo?
- Which impairment dominates first: noise, clutter heterogeneity, geometry, or detector choice?

The repository is intended for **architecture-level reasoning and comparative analysis**.
It is *not* a real-time signal processor, tracker, or mission-level simulator.

---

## Modeling philosophy

### System-level coupling

Radar performance is treated as a **coupled system problem**.

Changes to any component—antenna gain, integration length, detector choice,
scan geometry, or environment—necessarily affect multiple performance metrics.
There is no notion of a universally “best” configuration—only explicit trade-offs.

---

### Explicit statistical treatment

All reported performance metrics are probabilistic by construction:

- Pd, Pfa, and FAR are modeled explicitly
- Noise, clutter, and target fluctuations are treated statistically
- Monte Carlo simulation is used for **validation**, not as a replacement for models

Results are expected to vary with assumptions and environment.

---

### Separation of model and experiment

Each run declares its execution mode explicitly:

- `model_based` — analytic or closed-form evaluation
- `monte_carlo` — empirical simulation
- `signal_level` — waveform-level simulation (where applicable)

Analytic predictions and Monte Carlo results are compared directly.
Discrepancies are preserved as first-class outputs, not smoothed away.

---

### Declared validity and limits

Every execution records a **validity contract** describing:

- statistical assumptions
- clutter regime
- operational and modeling limits

If assumptions are violated, results are not silently trusted.

---

## Reproducibility and traceability

Each execution produces a self-contained, auditable result bundle:

results/cases/<run_id>/
├── metrics.json          # performance results + execution metadata
├── manifest.json         # execution context and validity contract
├── case_manifest.json    # normalized configuration and provenance


The run identifier encodes:
- case name
- execution engine
- random seed
- configuration hash

All results are reproducible given the same inputs.

Generated artifacts (`results/`, `reports/`, etc.) are **not tracked by git** and are
intended to be regenerated.

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

This pipeline is intended as an **upstream analysis and reasoning tool**.

---

## Where this pipeline fails (by design)

This repository intentionally exposes regimes where common radar assumptions break.
These are not implementation bugs—they are the primary signals used for engineering judgment.

Documented failure modes include:

- **CFAR under heterogeneous clutter**  
  CA-CFAR does not maintain the requested Pfa in heterogeneous environments.
  This behavior is preserved and made visible via Monte Carlo validation and
  comparison with alternative detectors (e.g., OS-CFAR).

- **Analytic Pd vs Monte Carlo divergence**  
  Closed-form detection models diverge from empirical results at low SNR,
  under strong target fluctuations (Swerling cases), and near threshold.
  These discrepancies are measured, reported, and gated rather than hidden.

- **Pfa is not an operational metric by itself**  
  A fixed Pfa does not determine system false-alarm load without explicit accounting
  of trials per second (range cells × Doppler bins × beams × scan rate).
  FAR inflation is demonstrated explicitly in integration and geometry trade-offs.

- **Model validity is conditional**  
  Results are meaningful only within the declared statistical and operational contracts
  (noise model, clutter regime, stationarity, far-field assumptions).
  Outside these limits, outputs are intentionally not guaranteed to be correct.

The pipeline is designed to make these failures **visible, reproducible, and auditable**.

---

## Repository structure

- `core/` — physics-based models (budgets, detection, geometry, environment)
- `cli/` — deterministic orchestration (cases and sweeps)
- `configs/` — YAML case definitions and schemas
- `tests/` — physical invariants and consistency checks
- `validation/` — golden tests and Monte Carlo validation
- `notebook/` — reproducible walkthrough and trade-off analysis
- `reports/` — generated analysis artifacts (not tracked)

---

## Reproducible walkthrough

A complete, end-to-end demonstration is provided in:

notebook/radar_pipeline_walkthrough.ipynb

To reproduce all figures and results from a clean clone:

1. Create and activate a virtual environment
2. Install dependencies
3. Launch Jupyter
4. Open the notebook
5. **Run all cells** (Kernel → Restart & Run All)

All results are generated under `results/` and can be safely deleted and regenerated.

---

## Typical workflow

1. Define a radar and scenario via a YAML case configuration
2. Run a model-based evaluation
3. Validate with Monte Carlo simulation
4. Inspect discrepancies and limits of validity
5. Modify one design parameter and observe the resulting trade-offs

The pipeline is structured to make these steps **explicit, repeatable, and auditable**.