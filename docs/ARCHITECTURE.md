# Architecture

## Style

- One small Python package, `columnmap`, that other projects import. The core has no third-party dependency (standard library only).
- Four parts: the core library (schema, prompts, mapping), `columnmap.bench` (synthetic data, ground truth, reports, DeepEval), the optional `columnmap.harness` (a LangGraph agent around the same mapping), and the optional input/prompt backends (`excel`, `registry`).
- Optional extras: `uv sync --extra graph` (LangGraph), `--extra excel` (openpyxl), `--extra registry` (MLflow Prompt Registry).
- Coupling rule: the core files never import `bench` or `harness`. `bench` and `harness` import the core.

## Library

| File | Owns |
|---|---|
| `knowledge.py` | `Knowledge`: the output fields, reference tables, prompt overrides, and policy, loaded from one source-controlled folder. `registry(base)` layers the folder over the chosen prompt backend. |
| `policy.py` | `Policy`: the derived/copy/none ceilings, the model they were calibrated for, and the calibration facts. `knowledge/policy.json` is the source; `appsettings.json` is the fallback. |
| `decider.py` | The Decisions interface: `Question`, `Answer`, `Response`, the `Decider` protocol, and `OpenRouterDecider`. |
| `prompts.py` | The built-in prompt book (`DEFAULT_TEMPLATES`) and the question builders. Every text is a `{{variable}}` template; a knowledge folder or MLflow can override any name. |
| `registry.py` | `Prompt`, `FileRegistry`, `ChainRegistry`: the source-controlled prompt book with double-brace variables. |
| `registry_mlflow.py` | `MlflowRegistry`: loads `prompts:/<prefix><name>/<version>` or `@alias` through `mlflow.genai.load_prompt`. MLflow is imported only when this backend is used. |
| `schema.py` | Extracts the input schema ourselves: type, unit, samples, null rate, and distinct count per column. Partners never supply a schema. |
| `table.py` | `read_table`: CSV/TSV with the standard library, XLSX through the `excel` extra. |
| `expressions.py` | The expression AST, the candidate catalog, and the deterministic unit inference (`amount`, `rate`, `count`, `date`, `text`, `id`). |
| `roles.py` | Reads a field description as a shape: the operator and its named operands, with the unit each expects. It never reads the rule, the kind, or a source id. |
| `pipeline.py` | The explicit stage machine: `kinds -> score -> decide`. `Mapper`, the bench, and the graph all use it; there is no hidden state beyond the kind cache. |
| `mapper.py` | The stages themselves, `convert`, `is_filled`, and `Mapper`, a thin wrapper around `Pipeline`. |
| `config.py` | Paths and the `appsettings.json` / `.env` settings used by the CLI. The library itself takes plain arguments. |

```python
from columnmap import Knowledge, Mapper, OpenRouterDecider, read_table

rows = read_table("partner.xlsx")                  # CSV/TSV always; XLSX with --extra excel
knowledge = Knowledge.load("my_knowledge/")        # outputs, reference, prompts, policy
mapping = Mapper(OpenRouterDecider(key, model, url), knowledge).map(rows)
header, converted = mapping.convert(rows)          # uses the dataset policy; empty when not present
```

## Knowledge folder

One folder per dataset, committed next to the code (or registered in MLflow):

```
my_knowledge/
  outputs.json          the target template: name, type, description (task input)
  reference/*.csv       code -> label tables for lookups (task input)
  prompts.json          editable overrides of any prompt in the built-in book
  policy.json           fill floors calibrated for this dataset and model
```

- `outputs.json` and `reference/` describe the target. They may be complex (`x + y`, a date
  difference, a code lookup); the role parser reads the description, never the rule.
- `prompts.json` lets the user change the wording per dataset without touching Python. A full
  default book is written by `make data` into `data/knowledge/prompts.json` as a starting point.
- MLflow is the versioned alternative: `MlflowRegistry(prefix="columnmap/", version="production")`.
  The folder overrides win, then the MLflow registry, then the built-in book.
- `policy.json` is chosen by `columnmap --calibrate <labeled run> --budget 0.02`, not by hand. The
  command sweeps derived and copy floors, maximizes field accuracy under the false-fill budget, and
  records the point estimate plus a Wilson 95% upper bound. If the upper bound is above the budget,
  the label set is too small to certify it; collect more fields. `appsettings.json` is the fallback
  when a dataset has no policy.

```python
from columnmap import MlflowRegistry, Mapper

registry = MlflowRegistry(prefix="columnmap/", version="production")
mapper = Mapper(decider, knowledge, registry=registry)   # folder overrides -> MLflow -> built-in
```

## Bench (evaluation only)

`bench/` holds the synthetic template and generator (`template.py`, `generate.py`, `samples.py`), the CSV export (`export.py`, which also writes `data/knowledge/`), the resumable run store (`store.py`, `run.py`, `runs.py`, `usage.py`), grading against the ground truth (`metrics.py`), the report (`summary.py`, `report.py`), the run reader shared by evaluation and calibration (`cases.py`), threshold calibration (`calibrate.py`), DeepEval (`deepeval.py`), and the CLI (`cli.py`).

## Harness (optional)

`harness/` turns the same pipeline into a LangGraph agent with durable state, bounded retries, a
deterministic fallback, and human review. It needs the graph extra; the core never imports it.

```python
from columnmap.harness import run_graph

run = run_graph(rows, decider, knowledge, thread_id="batch-1")
header, converted = run.header, run.converted
if run.paused:                                  # a field was close to the fill policy
    run.resume({"total_charge": "accept"})      # or "empty"
if run.degraded:                                # the model failed after retries
    print(run.errors)                           # only exact name matches were filled
```

Reliability layers, in the order they act:

1. **Retry.** `RetryPolicy` on `kinds`, `score`, and `decide`: transient `RuntimeError`/`OSError`
   retry with backoff; programming errors do not.
2. **Refusal retry.** `Retrying` re-asks a batch that returns a refusal once; a refusal after that
   fails closed and leaves the field empty.
3. **Degrade.** A LangGraph `error_handler` runs after retries are exhausted. It routes to
   `degrade`, which only fills a copied field that has an exact column-name match; everything else
   stays empty, and the run records the error.
4. **Repair.** `verify` detects an expression that fails on the sample rows. `repair` tries the next
   candidate expression, at most once, then re-verifies. No alternative means the field stays empty.
5. **Review.** A field within 0.15 of the fill floor, or with `p_none` close to the ceiling, is the
   only ambiguous band. `review="empty"` resolves it to no (empty), `review="accept"` to yes, both
   without a pause; `review="ask"` pauses on `interrupt()` and resumes with a dict of field to
   `accept` or `empty`. The product output is binary in every mode; the score and the method stay in
   the decisions CSV for audit.

The graph is `kinds -> score -> decide -> verify`, then `repair` (loop back to `verify`), `review`
(an `interrupt` only when `review="ask"`), `degrade`, or `commit`. A checkpointer plus `thread_id`
makes every pause resumable; nodes are idempotent and side effects happen only in `commit`.

## Current violations

None in the core. Check: `grep -rn "bench\|harness" src/columnmap/*.py` prints only comments and the CLI import of `bench.cli`. The harness imports LangGraph by design; MLflow and openpyxl are imported lazily by their backends.

## Flow

1. **Schema (per file).** `extract(rows)` derives type, unit, samples, null rate, and distinct count
   for every input column. The model sees these facts instead of raw rows.
2. **Kind (once per knowledge).** The model reads each output field's description and says what kind
   of value it is: copy, join, arithmetic, date difference, year, or lookup. The question is asked
   with the options in three shuffled orders and the answers are averaged.
3. **Stage 1 (per file).** One question per input column: which output fields does it provide data
   for. The raw scores rank the columns for each field.
4. **Stage 2 (per field).** A readable description becomes a shape with named roles: `total_charge`
   is "transaction amount plus fee", `coverage_days` is "days between coverage start and coverage
   end". Each role is one choice over the compatible input columns plus an explicit `none`; the
   questions for a group of fields share one request. Code builds the expression from the chosen
   columns, so the model never picks a formula. A rate column is not offered for an amount role, and
   a plain code is not offered for a date role. A description the parser cannot read falls back to
   the column-set question plus a separate presence question.
5. **Fill.** The dataset `policy.json` (or the `appsettings.json` fallback) sets the floors. A field
   is filled when its source is present: `p_present` (the weakest role, or the presence answer in
   the fallback) reaches the copy or derived floor, and `p_none` stays at or below the ceiling. A
   column claimed by two copied fields stays with the higher score; the other gets score 0 and
   presence 0. Converting is deterministic.
