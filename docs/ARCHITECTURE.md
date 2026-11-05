# Architecture

## Style

- One small Python package, `columnmap`, that other projects import. It has no third-party dependency (standard library only).
- Two parts: the library (the mapping itself) and `columnmap.bench` (synthetic data, ground truth, reports, DeepEval). A project that only maps tables imports the library and never touches `bench`.
- Coupling rule: the library files never import `bench`. `bench` imports the library.

## Library

| File | Owns |
|---|---|
| `knowledge.py` | `Knowledge`: the output fields (name, type, description) and the reference tables (code lists), loaded from a folder. Switching data means pointing to another folder. |
| `decider.py` | The Decisions interface: `Question`, `Answer`, `Response`, the `Decider` protocol, and `OpenRouterDecider`. |
| `prompts.py` | All prompt text. The model sees column names and sample values, field names and descriptions, and candidate expressions with what they compute. |
| `expressions.py` | The expression AST (evaluate, render) and the catalog of candidate expressions, including lookups from reference tables. Deterministic, no model. |
| `mapper.py` | `score_columns` (stage 1), `choose_expressions` (stage 2), `convert`, and `Mapper`, which runs both. |
| `config.py` | Paths and the `appsettings.json` / `.env` settings used by the CLI. The library itself takes plain arguments. |

```python
knowledge = Knowledge.load("my_knowledge/")        # outputs.json + reference/*.csv
mapping = Mapper(OpenRouterDecider(key, model, url), knowledge).map(rows)
header, converted = mapping.convert(rows)          # fields below the fill score stay empty
```

## Bench (evaluation only)

`bench/` holds the synthetic template and generator (`template.py`, `generate.py`, `samples.py`), the CSV export (`export.py`, which also writes `data/knowledge/`), the resumable run store (`store.py`, `run.py`, `runs.py`, `usage.py`), grading against the ground truth (`metrics.py`), the report (`summary.py`, `report.py`), DeepEval (`deepeval.py`), and the CLI (`cli.py`).

## Current violations

None. Check: `grep -rn "bench" src/columnmap/*.py` prints only `__init__.py` and `__main__.py` comments or imports of `bench.cli`.

## Flow

Stage 1 asks the model one question per input column: which output fields does it provide data for, given the field names and descriptions. Stage 2 takes the four best columns of each field, builds candidate expressions from every subset (copy, year, date difference, arithmetic, concat, lookup in a reference table), lets the model pick a shortlist and check each candidate against the field description, and keeps the simplest expression whose score is close to the best. A field is filled when that score reaches the fill score; converting is deterministic.
