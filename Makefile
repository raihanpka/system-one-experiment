UV ?= uv
PY := $(UV) run python
RUN := $(UV) run columnmap

.DEFAULT_GOAL := help
.PHONY: help test lint run sample full pairs report data open clean

help: ## Show this list
	@grep -E '^[a-z]+:.*##' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  make %-8s %s\n", $$1, $$2}'

test: ## Unit tests, no model needed
	$(UV) run pytest -q

lint: ## Unused imports and names
	uvx pyflakes src tests

run: ## Run one sample or all: make run SAMPLES=c (asks when SAMPLES is empty; all = every sample)
	@s="$(SAMPLES)"; if [ -z "$$s" ]; then read -p "Sampel (a, b, c, d, atau all): " s; fi; \
	if [ "$$s" = "all" ]; then $(RUN) --all; else $(RUN) --samples "$$s"; fi

sample: ## Run samples a,b,c,d into results/<samples>_<timestamp>/
	$(RUN) --samples abcd

full: ## Run all four samples
	$(RUN) --all

pairs: ## Slow calibrated mode (one yes/no per column and field) on a,b,c,d
	$(RUN) --samples abcd --mode pairs

report: ## Rebuild files, figures and tables of the newest run, no model
	$(RUN) --report-only

data: ## Write the template and sample CSV files under data/ (deterministic)
	$(PY) -m columnmap.dataset.export

open: ## Open the summary of the newest run
	open "$$(ls -dt results/*/ | head -1)report/summary.md"

clean: ## Delete results
	rm -rf results
