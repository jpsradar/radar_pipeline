# Radar Performance Pipeline

Radar performance analysis pipeline for system-level trade-off studies under
explicit statistical assumptions and operational constraints.

This repository provides a reproducible framework to analyze how radar design
choices (antenna, waveform, integration, detection, and environment) affect
probability of detection, false alarms, false alarm rate, and coverage.

The emphasis is on **engineering trade-offs, statistical validity, and limits**,
not on isolated formulas or black-box simulation.

---

## Scope and intent

The pipeline is designed to support questions such as:

- How does increased integration improve Pd while inflating system-level FAR?
- How does a fixed Pfa translate into real false alarm load once geometry is considered?
- When does an analytic Pd model remain valid, and when does it diverge from Monte Carlo?
- Which impairment dominates first: noise, clutter heterogeneity, geometry, or detector choice?

The repository is intended for **architecture-level reasoning** and comparative
analysis, not for real-time signal processing or mission-level simulation.

---

## Modeling philosophy

### System-level coupling

Radar performance is treated as a coupled system problem.
Changes to any component (antenna gain, integration length, detector, environment)
necessarily affect other performance metrics.

There is no notion of a “best” configuration—only explicit trade-offs.

---

### Explicit statistical treatment

All performance metrics are probabilistic:

- Pd, Pfa, and FAR are modeled explicitly
- Noise, clutter, and target fluctuations are treated statistically
- Monte Carlo simulation is used as validation, not as a substitute for models

Results are expected to vary with environment and assumptions.

---

### Separation of model and experiment

Each run declares its execution mode:

- `model_based` — analytic or closed-form evaluation
- `monte_carlo` — empirical simulation
- `signal_level` — waveform-level simulation (where applicable)

Analytic predictions and Monte Carlo results are compared explicitly, and
discrepancies are preserved as part of the output.

---

### Declared validity and limits

Every run records a **validity contract** describing:

- statistical model assumptions
- clutter regime
- operational limits

If assumptions are violated, results are not silently trusted.

---

## Reproducibility and traceability

Each execution produces a self-contained, auditable record:

results/cases/<run_id>/
├── config.normalized.json   # normalized configuration actually executed
├── manifest.json            # execution + validity contract
├── case_manifest.json       # detailed provenance (repo-native)
└── metrics.json             # results with execution and validity metadata

The run identifier encodes:
- case name
- execution engine
- random seed
- configuration hash

All results are reproducible given the same inputs.

---

## Configuration-driven architecture

Radar definitions are externalized into YAML files:

- waveform and PRF
- antenna and scan geometry
- detection and CFAR parameters
- environment and target models

The code does not embed hidden constants; changing a configuration is expected
to change performance.

Schemas enforce structure and prevent ambiguous configurations.

---

## Testing and validation

The repository includes deterministic tests that assert physical and statistical
invariants:

- radar equation scaling
- unit consistency
- geometry and count consistency
- FAR scaling behavior
- physical monotonicity (e.g., Pd vs SNR, range, integration)

Golden tests and Monte Carlo checks are used to ensure regression-free evolution.

---

## What is intentionally not modeled

The following are outside the scope of this repository:

- multipath, ducting, and complex propagation effects
- advanced tracking and track-before-detect
- hardware-specific non-idealities beyond first-order models
- real-time or embedded constraints

The pipeline is intended as an upstream analysis and reasoning tool.

---

## Where this pipeline fails (by design)

This repository intentionally exposes regimes where common radar assumptions break.
These are not implementation bugs; they are the primary signals used for engineering judgment.

Known and documented failure modes include:

- **CFAR under heterogeneous clutter**  
  CA-CFAR does not maintain the requested Pfa in heterogeneous environments.
  This behavior is preserved and made visible through Monte Carlo validation and
  comparison against alternative detectors (e.g., OS-CFAR).

- **Analytic Pd vs Monte Carlo divergence**  
  Closed-form detection models diverge from empirical results at low SNR,
  high target fluctuation regimes (Swerling cases), and near threshold.
  These discrepancies are measured, reported, and gated in CI rather than hidden.

- **Pfa is not an operational metric by itself**  
  A fixed Pfa does not determine system false-alarm load without explicit accounting
  of trials per second (range cells × Doppler bins × beams × scan rate).
  FAR inflation is demonstrated explicitly in integration and geometry trade-offs.

- **Model validity is conditional**  
  Results are only meaningful within the declared statistical and operational contracts
  (noise model, clutter regime, stationarity, far-field assumptions).
  Outside these limits, outputs are intentionally not guaranteed to be correct.

The pipeline is designed to make these failures visible and reproducible,
so that system-level decisions are driven by limits and trade-offs rather than by
single-number performance claims.

---

## Repository structure

- `core/` — physics-based models (budgets, detection, geometry, environment)
- `cli/` — deterministic orchestration (cases and sweeps)
- `configs/` — YAML case definitions with schemas
- `tests/` — physical invariants and consistency checks
- `validation/` — golden tests and Monte Carlo validation
- `reports/` — analysis artifacts and plots

---

## Typical workflow

1. Define a radar and scenario in a case configuration.
2. Run a model-based evaluation.
3. Validate with Monte Carlo simulation.
4. Inspect discrepancies and limits of validity.
5. Modify one design parameter and observe the resulting trade-offs.

The pipeline is structured to make these steps explicit and repeatable.