# Makefile
#
# Minimal developer ergonomics for reproducible radar-pipeline runs.
# Keep this file intentionally small and dependency-free.

PY ?= python3

# Default demo case (override: make case CASE=configs/cases/demo_pd_noise.yaml)
CASE ?= configs/cases/demo_pd_noise.yaml
OUT  ?= results/cases
SEED ?=
ENGINE ?= auto
REPORT ?= 1

SWEEP ?= configs/cases/sweep_example.yaml
SWEEP_OUT ?= results/sweeps

.PHONY: help case sweep report clean

help:
	@echo "Targets:"
	@echo "  make case   CASE=... OUT=... SEED=... ENGINE=auto|model_based|monte_carlo|signal_level REPORT=1|0"
	@echo "  make sweep  CASE=... SWEEP=... OUT=... SEED=... ENGINE=model_based|monte_carlo|signal_level REPORT=1|0"
	@echo "  make clean  (remove results/*)"
	@echo ""
	@echo "Examples:"
	@echo "  make case CASE=configs/cases/demo_pd_noise.yaml ENGINE=model_based"
	@echo "  make case CASE=configs/cases/demo_clutter_cfar_hetero.yaml ENGINE=monte_carlo SEED=123"
	@echo "  make sweep CASE=configs/cases/demo_pd_noise.yaml SWEEP=configs/cases/sweep_example.yaml ENGINE=model_based"

case:
	@args="--case $(CASE) --out $(OUT) --engine $(ENGINE)"; \
	if [ -n "$(SEED)" ]; then args="$$args --seed $(SEED)"; fi; \
	if [ "$(REPORT)" = "1" ]; then args="$$args --report"; fi; \
	$(PY) -m cli.run_case $$args

sweep:
	@args="--case $(CASE) --sweep $(SWEEP) --out $(SWEEP_OUT) --engine $(ENGINE)"; \
	if [ -n "$(SEED)" ]; then args="$$args --seed $(SEED)"; fi; \
	if [ "$(REPORT)" = "1" ]; then args="$$args --report"; fi; \
	$(PY) -m cli.run_sweep $$args

report:
	@echo "Reports are generated via --report in make case/sweep."

clean:
	rm -rf results