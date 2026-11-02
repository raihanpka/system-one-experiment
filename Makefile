UV ?= uv
PY := $(UV) run python
RUN := $(UV) run columnmap

.DEFAULT_GOAL := help
.PHONY: help setup test lint sample full pairs report data bench open status stop clean

help: ## Show this list
	@grep -E '^[a-z]+:.*##' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  make %-8s %s\n", $$1, $$2}'

setup: ## Python deps, model download, build the scorer against brew llama.cpp (rerun after brew upgrade)
	sh scripts/setup_jev.sh

test: ## Unit tests, no model needed
	$(UV) run pytest -q

lint: ## Unused imports and names
	uvx pyflakes src tests scripts

sample: ## Run samples a,b,c,d into results/<samples>_<timestamp>/ (about 6 minutes)
	$(RUN) --samples abcd

full: ## Run all seven samples
	$(RUN) --all

pairs: ## Slow calibrated mode (one yes/no per column and field) on a,b,c,d
	$(RUN) --samples abcd --mode pairs

report: ## Rebuild files, figures and tables of the newest run, no model
	$(RUN) --report-only

data: ## Write the template and sample CSV files under data/ (deterministic)
	$(PY) -m columnmap.dataset.export

bench: ## Speed and accuracy of the two scoring modes (about 5 minutes)
	$(PY) scripts/bench_modes.py

open: ## Open the summary of the newest run
	open "$$(ls -dt results/*/ | head -1)report/summary.md"

status: ## Is the model process running?
	@pgrep -fl jev-score-v2 || echo "model not running"

stop: ## Stop any running model process
	-pkill -f jev-score-v2

clean: ## Delete results
	rm -rf results
