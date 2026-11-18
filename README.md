<div align="center">

# Column Mapper

[Architecture](./docs/ARCHITECTURE.md) &nbsp;&middot;&nbsp; [Model](https://openrouter.ai/microsoft/microsoft-decision-1.1) &nbsp;&middot;&nbsp; [OpenRouter](https://openrouter.ai)

</div>

### About

Maps the columns of a partner table onto the output fields of a target template and gives a converted table. The Decisions model (`microsoft/microsoft-decision-1.1` through OpenRouter) reads the kind of each output field from its description, scores which columns feed which field, and binds each named role of the description to one column (or none). Code builds the expression from those columns (copy, year, date difference, arithmetic, concat, reference-table lookup), and a field is filled only when its source is present: `p_present` reaches the fill score and `p_none` stays below the ceiling.

The input schema is extracted from the partner data itself (type, unit, samples, null rate, distinct count); a partner never has to supply one. All of that is deterministic and standard-library only. The same package ships an optional LangGraph agent with retries, a deterministic fallback, a repair loop, and human review.

### Quickstart

```python
from columnmap import Knowledge, Mapper, OpenRouterDecider, read_table

rows = read_table("partner.xlsx")              # CSV/TSV always; XLSX with --extra excel
knowledge = Knowledge.load("my_knowledge/")    # outputs.json + reference/*.csv + prompts.json
decider = OpenRouterDecider(api_key, "microsoft/microsoft-decision-1.1", "https://openrouter.ai/api/alpha/decisions")
mapping = Mapper(decider, knowledge).map(rows)
header, converted = mapping.convert(rows)       # a field without a present source stays empty
```

### Knowledge folder

One folder per dataset, committed with the code or registered in MLflow:

```
my_knowledge/
  outputs.json          target template: name, type, description
  reference/*.csv       code -> label tables for lookups
  prompts.json          editable overrides of any prompt in the built-in book
```

`make data` writes a complete example to `data/knowledge/`, including the default prompt book. Copy that folder into your project before editing; `make data` regenerates the example.

The prompt book is a registry, not an f-string. The folder wins, then the backend, then the built-in defaults:

```python
from columnmap import MlflowRegistry, Mapper

registry = MlflowRegistry(prefix="columnmap/", version="production")  # needs --extra registry
mapper = Mapper(decider, knowledge, registry=registry)
```

MLflow uses immutable versions and aliases (`prompts:/columnmap/bind@production`); a numeric version loads `prompts:/columnmap/bind/7`.

### Graph harness

The optional agent needs the `graph` extra and adds no dependency to the core:

```bash
uv sync --extra graph
```

```python
from columnmap.harness import run_graph

run = run_graph(rows, decider, knowledge, thread_id="batch-1", review="empty")
header, converted = run.header, run.converted   # always yes or no, never "maybe"
if run.degraded:                                # the model failed after retries
    print(run.errors)                           # only exact name matches were filled
```

`review="empty"` is the product mode: a field close to the policy is a no (empty), with no pause.
`review="accept"` makes it a yes. `review="ask"` (the library default) pauses on an `interrupt` so a
human can choose; the CLI default is `empty`. The output is binary in every mode.

The graph retries transient failures with `RetryPolicy`, degrades to exact name matches when retries are exhausted, repairs a failed expression once with the next candidate, and pauses only when the review policy is `ask`. A checkpointer plus `thread_id` makes every pause resumable.

---

### Getting Started

```bash
cp .env.example .env   # then put the OpenRouter key in .env
make check             # format check, lint, and tests, no model call
make help              # all commands
```

Extras: `uv sync --extra graph` (LangGraph), `--extra excel` (XLSX input), `--extra registry` (MLflow prompts). Settings live in `appsettings.json` (every option is listed in `appsettings.json.examples`) and `appsettings.local.json` (gitignored). Environment variables `COLUMNMAP__Section__Key` override both.

### Map a partner table

```bash
uv run columnmap --table in.csv --knowledge my_knowledge/ --output converted.csv
uv run columnmap --table in.csv --schema --json          # the schema we extracted, no model
uv run columnmap --table in.csv --graph --review empty   # through the agent, fail closed
make map TABLE=in.csv KNOWLEDGE=my_knowledge/ OUT=converted.csv
make schema TABLE=in.csv
```

`--json` prints one machine-readable object (`rows`, `fields`, `filled`, `empty`, `degraded`, `errors`, `reviewed`). `--review ask` prompts per field; `accept` keeps the expression; `empty` leaves the field empty.

### Running the bench

```bash
uv run columnmap                 # samples a, b, c, d
uv run columnmap --samples ac    # only a and c
uv run columnmap --split holdout # seeds 12 and 13, never tuned on
uv run columnmap --report-only --fill-score 0.8 --copy-fill-score 0.5   # rebuild the newest run, no model
uv run columnmap --deepeval results/<run>          # DeepEval metrics, no model
```

Samples: `a` identical names, `b` synonyms, `c` unusual technical terms, `d` mixed styles with extra columns. Every run writes `results/<run>/report/summary.md` with each count as x / y against the ground truth, fill precision, and per sample the converted file, the decisions (including `p_present`/`p_none`), and the column scores.

<div align="center">
<sub>Copyright &copy; 2026 Raihan Putra Kirana. All rights reserved.</sub>
</div>
