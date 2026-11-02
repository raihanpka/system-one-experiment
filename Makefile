UV ?= uv
PY := $(UV) run python

.DEFAULT_GOAL := help
.PHONY: help setup test sample full pairs permute report data bench open status stop clean

help: ## Show this list
	@grep -E '^[a-z]+:.*##' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  make %-8s %s\n", $$1, $$2}'

setup: ## One time: Python deps, model download, build the llama.cpp scorer
	sh scripts/setup_jev.sh

test: ## Unit tests, no model needed
	$(UV) run pytest -q

sample: ## Run samples a,b,c,d and write results/latest (result.json, report, figures)
	$(PY) src --samples abcd

full: ## Run all seven samples
	$(PY) src --all

pairs: ## Slow calibrated mode (one yes/no per column and field) on a,b,c,d
	$(PY) src --samples abcd --mode pairs --out results/pairs

permute: ## Also test option-order stability on a,b,c,d
	$(PY) src --samples abcd --permute

report: ## Rebuild figures and tables from results/latest/result.json, no model
	$(PY) src --report-only

data: ## Write the template and sample CSV files under data/
	$(PY) -m src.dataset.export

bench: ## Speed and accuracy of the two scoring modes (about 5 minutes)
	$(PY) scripts/bench_modes.py

open: ## Open the report
	open results/latest/report/summary.md

status: ## Is the model process running?
	@pgrep -fl jev-score-v2 || echo "model not running"

stop: ## Stop any running model process
	-pkill -f jev-score-v2

clean: ## Delete results
	rm -rf results
