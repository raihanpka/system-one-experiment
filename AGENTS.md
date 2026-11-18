# columnmap

Maps messy input columns to a fixed insurance/finance output template. Each output field's description is read as named roles, each role is bound to one input column, and code builds the expression. A field is filled when its source is present (`p_present` reaches the fill score and `p_none` stays below the ceiling); otherwise it stays empty. The goal is a converted file equal to the expected file. Generic working rules come from the dotagent, the facts below are this repo's.

## Commands

- Build: `uv build`
- Run: `make run SAMPLES=a` (calls OpenRouter, run only when the user orders it)
- Map a partner table: `make map TABLE=in.csv KNOWLEDGE=my_knowledge/ OUT=converted.csv`
- Map through the graph: `make graph TABLE=in.csv REVIEW=empty` (needs `uv sync --extra graph`)
- Extract the input schema: `make schema TABLE=in.csv`
- Calibrate the dataset policy from a labeled run: `make calibrate RUN_DIR=results/<run> BUDGET=0.02`
- Format: `uvx ruff@0.15.0 format src tests` (check only: `uvx ruff@0.15.0 format --check src tests`)
- Lint: `make lint`
- Test: `make test` (one file: `uv run pytest -q tests/test_mapper.py`)
- Full gate: `make check` (format check, lint, test)
- DeepEval check of a finished run, no model call: `uv run columnmap --deepeval results/<run folder>`
- Rebuild a report with other fill scores, no model call: `uv run columnmap --report-only --fill-score 0.8 --copy-fill-score 0.5`

## Architecture

- Style: one small package whose core has no third-party dependency. Core files at `src/columnmap/`: `knowledge.py`, `policy.py`, `decider.py`, `prompts.py`, `registry.py`, `registry_mlflow.py`, `schema.py`, `table.py`, `expressions.py`, `roles.py`, `pipeline.py`, `mapper.py`, `config.py`. Evaluation only in `src/columnmap/bench/` (synthetic data, ground truth, report, cases, calibration, DeepEval, CLI). The optional graph harness in `src/columnmap/harness/` needs the `graph` extra (`uv sync --extra graph`).
- Optional extras: `graph` (LangGraph harness), `excel` (openpyxl input), `registry` (MLflow Prompt Registry). The core never imports them.
- Coupling: core files never import `bench` or `harness`; both import the core. Details in `docs/ARCHITECTURE.md`.

## Rules

- Branch from `main`. One task is one branch and one PR.
- Commits are one line, `type(scope): summary`, no trailer. Scopes: mapper, knowledge, decider, prompts, registry, schema, table, expressions, roles, pipeline, config, bench, cli, deps, docs, test, results, gitignore, format.
- Names say what changed. Never a plan label such as `milestone-1`, `task-2`, `slice 3`, or "finishing task", and never `(#N)`, in a commit message, PR title, or branch.
- Never edit protected paths: `.env`, `appsettings.local.json`, `data/real/`. A hook refuses it.
- Never edit generated files: `data/` (regenerate with `make data`) and `results/` (run output). `data/knowledge/` is a generated example; copy it into the project to edit prompts for a real dataset.
- One dataset is one knowledge folder: `outputs.json`, `reference/*.csv`, `prompts.json`, `policy.json`. Commit it with the code; MLflow Prompt Registry is the versioned alternative.
- The policy is calibrated, not hand-tuned: `make calibrate RUN_DIR=... BUDGET=0.02` writes `policy.json` and records the point estimate and the 95% upper bound. Do not change floors per batch or per file. `appsettings.json` is the fallback when a dataset has no policy.
- Never call the model (`make run`, `make sample`, `make full`) without the user's explicit order.
- Seeds: `Run:Seed` 3407 is the default sample seed for `make run`/`make data`; the split lists are separate. 11 for dev tuning. 12 and 13 (holdout) were used to choose the double-claim rule and the copy fill score, so they are no longer clean. 14 and 15 checked the frozen pipeline once. A new claim needs a seed nobody has looked at. 21 and 22 were already used for the one test run; 31 and 32 only with the user's permission.
- Report every mapping result as `x / y` (fields right out of the fields counted), per sample and context, and say which seed.
- The model gets no ground truth: no rule, kind, or source id of the synthetic template. Reference tables and field descriptions are task input.
- The API key goes in `.env` as `OPENROUTER_API_KEY` only. Never print it, log it, or commit it.
- A PR stays reviewable in one sitting, a few hundred changed lines is typical.
