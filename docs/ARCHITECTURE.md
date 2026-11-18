# Architecture

## Style

- One small Python package, `columnmap`, that other projects import. The core has no third-party dependency (standard library only).
- Three parts: the core library (the mapping itself), `columnmap.bench` (synthetic data, ground truth, reports, DeepEval), and the optional `columnmap.harness` (a LangGraph around the same mapping).
- The graph harness is an optional extra: `uv sync --extra graph`. The core never imports it, and a project that only maps tables never installs it.
- Coupling rule: the core files never import `bench` or `harness`. `bench` and `harness` import the core.

## Library

| File | Owns |
|---|---|
| `knowledge.py` | `Knowledge`: the output fields (name, type, description) and the reference tables (code lists), loaded from a folder. Switching data means pointing to another folder. |
| `decider.py` | The Decisions interface: `Question`, `Answer`, `Response`, the `Decider` protocol, and `OpenRouterDecider`. |
| `prompts.py` | All prompt text. The model sees column names and sample values, field names and descriptions, role questions, and candidate expressions with what they compute. |
| `expressions.py` | The expression AST (evaluate, render), the catalog of candidate expressions, and the deterministic unit inference (`amount`, `rate`, `count`, `date`, `text`, `id`). |
| `roles.py` | Reads a field description as a shape: the operator and its named operands, with the unit each operand expects. It never reads the rule, the kind, or a source id. |
| `mapper.py` | `score_columns` (stage 1), `choose_expressions` (stage 2: role binding, expression, fallback), `convert`, `is_filled`, and `Mapper`, which runs both stages. |
| `config.py` | Paths and the `appsettings.json` / `.env` settings used by the CLI. The library itself takes plain arguments. |

```python
knowledge = Knowledge.load("my_knowledge/")        # outputs.json + reference/*.csv
mapping = Mapper(OpenRouterDecider(key, model, url), knowledge).map(rows)
header, converted = mapping.convert(rows)          # a field without a present source stays empty
```

## Bench (evaluation only)

`bench/` holds the synthetic template and generator (`template.py`, `generate.py`, `samples.py`), the CSV export (`export.py`, which also writes `data/knowledge/`), the resumable run store (`store.py`, `run.py`, `runs.py`, `usage.py`), grading against the ground truth (`metrics.py`), the report (`summary.py`, `report.py`), DeepEval (`deepeval.py`), and the CLI (`cli.py`).

## Harness (optional)

`harness/` turns the same stages into a LangGraph with durable state and human review. It needs the graph extra; the core never imports it.

```python
from columnmap.harness import run_graph

run = run_graph(rows, decider, knowledge, thread_id="batch-1")
header, converted = run.header, run.converted
if run.paused:                       # a field was close to the fill policy
    run.resume({"total_charge": "accept"})   # or "empty"
```

The graph is `kinds -> score -> decide -> verify -> gate`, then `review` (an `interrupt`) or `commit`. `verify` sends a field to review when its expression fails on the sample rows or when it sits within 0.15 of the fill floor. The resume value is a dict of field to `accept` or `empty`. `Retrying` retries a batch that returns a refusal once; a refusal after the retry fails closed and leaves the field empty.

## Current violations

None in the core. Check: `grep -rn "bench\|harness" src/columnmap/*.py` prints only comments and the CLI import of `bench.cli`. The harness imports LangGraph by design.

## Flow

1. **Kind (once per knowledge).** The model reads each output field's description and says what kind of value it is: copy, join, arithmetic, date difference, year, or lookup. The question is asked with the options in three shuffled orders and the answers are averaged, so a field does not flip between kinds from run to run.
2. **Stage 1 (per file).** One question per input column: which output fields does it provide data for, given the field names and descriptions. The raw scores rank the columns for each field.
3. **Stage 2 (per field).** A readable description becomes a shape with named roles: `total_charge` is "transaction amount plus fee", `coverage_days` is "days between coverage start and coverage end". Each role is one choice question over the compatible input columns plus an explicit `none`; the questions for a group of fields share one request. Code builds the expression from the chosen columns, so the model never picks a formula. A rate column is not offered for an amount role, and a plain code is not offered for a date role. A description the parser cannot read falls back to the old column-set question plus a separate presence question.
4. **Fill.** A field is filled when its source is present: `p_present` (the weakest role, or the presence answer in the fallback) reaches `Fill.copy` (0.4) for a copied field or `Fill.derived` (0.7) for a derived one, and `p_none` stays at or below `Fill.none_ceiling` (0.5). A column claimed by two copied fields stays with the higher score; the other gets score 0 and presence 0. Converting is deterministic.
