UV ?= uv
PY := $(UV) run python
RUN := $(UV) run columnmap
TABLE ?=
KNOWLEDGE ?= data/knowledge
OUT ?= converted.csv
REVIEW ?= empty

.DEFAULT_GOAL := help
.PHONY: help test lint check run sample full report data map graph schema open clean

help: ## Show this list
	@grep -E '^[a-z]+:.*##' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  make %-8s %s\n", $$1, $$2}'

test: ## Unit tests, no model needed
	$(UV) run pytest -q

lint: ## Unused imports and names
	uvx pyflakes src tests

check: ## Format check, lint, and tests
	uvx ruff@0.15.0 format --check src tests
	$(MAKE) lint
	$(MAKE) test

run: ## Run one sample or all: make run SAMPLES=c (asks when SAMPLES is empty; all = every sample)
	@s="$(SAMPLES)"; if [ -z "$$s" ]; then read -p "Sampel (a, b, c, d, atau all): " s; fi; \
	if [ "$$s" = "all" ]; then $(RUN) --all; else $(RUN) --samples "$$s"; fi

sample: ## Run samples a,b,c,d into results/<samples>_<timestamp>/
	$(RUN) --samples abcd

full: ## Run all four samples
	$(RUN) --all

report: ## Rebuild files and summary of the newest run, no model
	$(RUN) --report-only

data: ## Write the template, samples, and the source-controlled knowledge folder under data/
	$(PY) -m columnmap.bench.export

map: ## Map a partner table: make map TABLE=in.csv KNOWLEDGE=data/knowledge OUT=converted.csv
	$(RUN) --table "$(TABLE)" --knowledge "$(KNOWLEDGE)" --output "$(OUT)"

graph: ## The same through the LangGraph harness: make graph TABLE=in.csv REVIEW=empty
	$(RUN) --table "$(TABLE)" --knowledge "$(KNOWLEDGE)" --output "$(OUT)" --graph --review "$(REVIEW)"

schema: ## Print the extracted input schema: make schema TABLE=in.csv
	$(RUN) --table "$(TABLE)" --schema

open: ## Open the summary of the newest run
	open "$$(ls -dt results/*/ | head -1)report/summary.md"

clean: ## Delete results
	rm -rf results
