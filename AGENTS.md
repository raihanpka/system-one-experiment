# columnmap

Maps messy input columns to a fixed insurance/finance output template. Each output field is filled with the best expression over the input columns when its score reaches the fill score; otherwise it stays empty. The goal is a converted file equal to the expected file. Generic working rules come from the dotagent, the facts below are this repo's.

## Commands

- Build: `uv build`
- Run: `make run SAMPLES=a` (calls OpenRouter, run only when the user orders it)
- Format: `uvx ruff@0.15.0 format src tests` (check only: `uvx ruff@0.15.0 format --check src tests`)
- Lint: `make lint`
- Test: `make test` (one file: `uv run pytest -q tests/test_mapper.py`)
- DeepEval check of a finished run, no model call: `uv run columnmap --deepeval results/<run folder>`
- Rebuild a report with other fill scores, no model call: `uv run columnmap --report-only --fill-score 0.8 --copy-fill-score 0.5`

## Architecture

- Style: one small package, no third-party dependency. Library files at `src/columnmap/`: `knowledge.py`, `decider.py`, `prompts.py`, `expressions.py`, `mapper.py`, `config.py`. Evaluation only in `src/columnmap/bench/` (synthetic data, ground truth, report, DeepEval, CLI).
- Coupling: library files never import `bench`. Details in `docs/ARCHITECTURE.md`.

## Rules

- Branch from `main`. One task is one branch and one PR.
- Commits are one line, `type(scope): summary`, no trailer. Scopes: mapper, knowledge, decider, prompts, expressions, config, bench, cli, deps, docs, test, results, gitignore, format.
- Names say what changed. Never a plan label such as `milestone-1`, `task-2`, `slice 3`, or "finishing task", and never `(#N)`, in a commit message, PR title, or branch.
- Never edit protected paths: `.env`, `appsettings.local.json`, `data/real/`. A hook refuses it.
- Never edit generated files: `data/` (regenerate with `make data`) and `results/` (run output).
- Never call the model (`make run`, `make sample`, `make full`) without the user's explicit order.
- Seeds: 11 for dev tuning. 12 and 13 (holdout) and 14 and 15 were used once to check a frozen pipeline and then to pick the double-claim rule, so a new claim needs a seed nobody has looked at. 21 and 22 were already used for the one test run; 31 and 32 only with the user's permission.
- Report every mapping result as `x / y` (fields right out of the fields counted), per sample and context, and say which seed.
- The model gets no ground truth: no rule, kind, or source id of the synthetic template. Reference tables and field descriptions are task input.
- The API key goes in `.env` as `OPENROUTER_API_KEY` only. Never print it, log it, or commit it.
- A PR stays reviewable in one sitting, a few hundred changed lines is typical.
