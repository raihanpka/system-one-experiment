# Conventions

Only rules that differ from the language defaults or that the code cannot tell you.

## Code

- Format with `ruff format` pinned to 0.15.0 (`uvx ruff@0.15.0 format`). Lint is `pyflakes` through `make lint`, no other linter.
- The core library has no third-party dependency. DeepEval is a dev dependency and is imported only by `bench/deepeval.py`; LangGraph is the optional `graph` extra and is imported only by `harness/`.
- The ground truth never reaches the model or the expression catalog. The model sees column names and values, field names, types and descriptions, and reference tables. Never a rule, a kind, or a source id.
- Report every result as `x / y` with the denominator named. Tune only on the dev split.
- The API key is read from the environment only (`OPENROUTER_API_KEY`, loaded from `.env`). Never in a tracked file.

## Tests

- Tests mirror `src/columnmap/`: library tests in `tests/`, bench tests in `tests/bench/`. Fake deciders live in `tests/bench/fakes.py`, so the suite needs no model and no network.
- Run one file with `uv run pytest -q tests/test_mapper.py`.

## Generated files

- `data/`: regenerate with `make data`, never edit by hand. It includes `data/knowledge/`, the folder format another project fills in.
- `results/`: run output of `make run` and the other run targets. Not edited by hand.
