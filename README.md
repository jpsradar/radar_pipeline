## What this is (quickly)

A radar detection analysis pipeline that:

- Models probability of detection (Pd) analytically  
- Validates results against Monte Carlo simulations (Swerling targets)  
- Maps detection performance into real system constraints via false-alarm rate (FAR)  
- Exposes where common radar assumptions break  

This is not a simulator demo — it is a tool for **engineering trade-off decisions**.


# Radar Performance Pipeline

A reproducible radar performance analysis pipeline for **system-level trade-off studies** under explicit statistical assumptions and operational constraints.

This repository provides an engineering framework to analyze how radar design choices—antenna, waveform, integration, detection strategy, and environment—
affect **probability of detection (Pd)**, **false alarms**, **false alarm rate (FAR)**, and **coverage**.

The emphasis is on **engineering trade-offs, statistical validity, and limits**— not on isolated formulas, opaque simulators, or black-box performance claims.

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

Changes to any component—antenna gain, integration length, detector choice, scan geometry, or environment—necessarily affect multiple performance metrics.
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

## Minimal technical example (noncoherent integration)

For a nonfluctuating target under AWGN, the noncoherent energy detector (after integrating N pulses) follows a central/non-central chi-square model.

- Under H0:
  T ~ χ²(2N)

- Under H1:
  T ~ χ²(2N, 2N·SNR)

The probability of detection is:

Pd = Q_N(√(2N·SNR), √(2γ))

where:

- Q_N is the generalized Marcum Q-function  
- γ is the detection threshold set by Pfa  

This pipeline implements both:

- analytic evaluation of Pd and Pfa  
- Monte Carlo validation of the same regime  

and exposes their divergence under:

- low SNR  
- finite sample effects  
- threshold proximity  

This is one of the core trade-off axes explored in the repository.

---

## Operational interpretation of FAR

Per-decision Pfa is not directly actionable.

The operational false-alarm rate is:

FAR = Pfa × N_cells × scan_rate

where:

- N_cells includes:
  - range bins
  - Doppler bins
  - beams / azimuth sectors
- scan_rate depends on revisit strategy

This repository makes that mapping explicit and measurable. A configuration that appears acceptable in terms of Pfa can become
operationally unusable once translated into FAR. This effect is demonstrated in the integration vs FAR sweep.

---

## Design questions supported

This repository enables quantitative answers to questions such as:

- What integration length maximizes Pd under a fixed FAR constraint?
- When does increasing N stop being beneficial due to FAR explosion?
- How robust is CA-CFAR under non-homogeneous clutter?
- At what SNR do analytic Pd models become unreliable?
- What is the sensitivity of Pd to geometry vs detector choice?

All answers are reproducible and tied to explicit assumptions.

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

This repository intentionally exposes regimes where common radar assumptions break. These are not implementation bugs—they are the primary signals used for engineering judgment.

Documented failure modes include:

- **CFAR under heterogeneous clutter**  
  CA-CFAR does not maintain the requested Pfa in heterogeneous environments.

- **Analytic Pd vs Monte Carlo divergence**  
  Closed-form detection models diverge under low SNR and near-threshold regimes.

- **Pfa is not an operational metric by itself**  
  FAR inflation emerges once system geometry is accounted for.

- **Model validity is conditional**  
  Results are only meaningful within declared assumptions.

Failures are made **visible, reproducible, and auditable**.

---

## Repository structure

- `core/` — physics-based models  
- `cli/` — deterministic orchestration  
- `configs/` — YAML definitions  
- `tests/` — invariants and consistency checks  
- `validation/` — Monte Carlo validation  
- `notebook/` — reproducible walkthrough  
- `reports/` — generated outputs (not tracked)  

---

## Reproducible walkthrough

A complete demonstration is provided in:

notebook/radar_pipeline_walkthrough.ipynb

To reproduce from a clean clone:

1. Create and activate a virtual environment  
2. Install dependencies  
3. Launch Jupyter  
4. Open the notebook  
5. Run all cells  

All outputs are generated under `results/`.

---

## Typical workflow

1. Define a radar scenario via YAML  
2. Run model-based evaluation  
3. Validate with Monte Carlo  
4. Inspect discrepancies  
5. Adjust design parameters  

All steps are **explicit, reproducible, and auditable**.

---

## What this is NOT

This repository does not provide:

- a real-time radar processing chain  
- a tracking system  
- hardware-level RF modeling  
- mission-level simulation  

It is focused on **first-principles performance analysis**.

---

## Summary

This project is a reference implementation of:

- detection theory at system level  
- reproducible radar trade-off analysis  
- explicit modeling of performance limits  

It is designed to make assumptions visible, results auditable,
and failure modes unavoidable.

