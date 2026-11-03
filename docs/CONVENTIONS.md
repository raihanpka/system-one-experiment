# Conventions

Only rules that differ from the language defaults or that the code cannot tell you.

## Code

- Format with `ruff format` pinned to 0.15.0 (`uvx ruff@0.15.0 format`). Lint is `pyflakes` through `make lint`, no other linter.
- Results are numbers, not claims: a cutoff is chosen only from data it was not tuned on, and a precision target is met with the one-sided lower bound at 95 percent, not the point estimate.
- The API key is read from the environment only (`OPENROUTER_API_KEY`, loaded from `.env`). Never in a tracked file.

## Tests

- Tests mirror `src/` under `tests/`. Fake deciders live in `tests/fakes.py`, so the suite needs no model and no network.
- Run one file with `uv run pytest -q tests/core/test_selective.py`.

## Generated files

- `data/`: regenerate with `make data`, never edit by hand.
- `results/`: run output of `make run` and the other run targets. Not edited by hand.
- `calibration.json`: refit with `uv run columnmap --fit-calibration <run folders>`, never edit by hand.
