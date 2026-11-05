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

1. **Kind (once per knowledge).** The model reads each output field's description and says what kind of value it is: copy, join, arithmetic, date difference, year, or lookup. The question is asked with the options in three shuffled orders and the answers are averaged, so a field does not flip between kinds from run to run.
2. **Stage 1 (per file).** One question per input column: which output fields does it provide data for, given the field names and descriptions. The raw scores rank the columns for each field.
3. **Stage 2 (per field).** The field's kind fixes how many columns it needs (one for copy, year and lookup; two for a date difference; two or three for arithmetic and join). The model chooses the column set among all subsets of the five best columns, with an explicit "none" option, while seeing every input column name. Then it chooses the expression that builds the field from exactly that set (deterministic candidates: copy, year, date difference, arithmetic, concat, lookup in a reference table). A kind without candidates falls back to a plain copy.
4. **Fill.** The probability of the chosen set decides: at least `Fill.derived` (0.7) for derived fields, at least `Fill.copy` (0.4) for copied fields. A column claimed by two copied fields stays with the higher score; the other gets score 0. Converting is deterministic.
