# columnmap

Maps messy input columns to a fixed insurance/finance output template. A field is filled automatically only when calibrated confidence and precision justify it, otherwise it goes to human review. Generic working rules come from the dotagent, the facts below are this repo's.

## Commands

- Build: `uv build`
- Run: `make run SAMPLES=a` (calls OpenRouter, run only when the user orders it)
- Format: `uvx ruff@0.15.0 format src tests` (check only: `uvx ruff@0.15.0 format --check src tests`)
- Lint: `make lint`
- Test: `make test` (one file: `uv run pytest -q tests/core/test_selective.py`)
- One module: not used, the repo is one package.

## Architecture

- Style: layered pipeline in one Python package, because one command runs every stage. Structure: horizontal layers by stage.
- Coupling: `core` and `decisions` import no stage. A stage imports `core`, `decisions`, and earlier stages only, in this order: `dataset`, `mapping`, `honest`, `convert`, `report`. Enforced by none yet, see `docs/DOTAGENT-SETUP.md`.

## Modules

- `src/columnmap/core/`: shared helpers (settings, paths, scoring, calibration, selective cutoff, validation, prompts, template). Import the module, there is no single public file.
- `src/columnmap/decisions/`: decider interface and question and answer types. Imports nothing from the repo.
- `src/columnmap/backends/`: OpenRouter adapter, the only backend. May depend on `core` and `decisions`.
- `src/columnmap/dataset/`: writes the template and sample CSV files (`make data`).
- `src/columnmap/mapping/`: scores each input column against each output field.
- `src/columnmap/honest/`: the honest run, which picks the fields from the scores.
- `src/columnmap/convert/`: fills the output file with the calibrated cutoff.
- `src/columnmap/report/`: builds files, figures, and tables from all stages.
- `src/columnmap/main.py`, `calibrate.py`, `experiment.py`: entry points, may use any stage.

## Rules

- Branch from `main`. One task is one branch and one PR.
- Commits are one line, `type(scope): summary`, no trailer. Scopes: calibration, convert, core, dataset, decisions, honest, mapping, report, settings, ontology, results, gitignore, deps, docs, test, format.
- Names say what changed. Never a plan label such as `milestone-1`, `task-2`, `slice 3`, or "finishing task", and never `(#N)`, in a commit message, PR title, or branch.
- Never edit protected paths: `.env`, `appsettings.local.json`, `data/real/`. A hook refuses it.
- Never edit generated files: `data/` (regenerate with `make data`), `results/` (run output), `calibration.json` (refit with `uv run columnmap --fit-calibration <run folders>`).
- Never call the model (`make run`, `make sample`, `make full`, `make pairs`) without the user's explicit order.
- Seeds: 11 for dev tuning only; 12 and 13 for calibration; 21 and 22 were already used for the one test run; 31 and 32 only with the user's permission.
- The API key goes in `.env` as `OPENROUTER_API_KEY` only. Never print it, log it, or commit it.
- A PR stays reviewable in one sitting, a few hundred changed lines is typical.
