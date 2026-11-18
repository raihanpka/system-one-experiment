<div align="center">

# Column Mapper

[Architecture](./docs/ARCHITECTURE.md) &nbsp;&middot;&nbsp; [Model](https://openrouter.ai/microsoft/microsoft-decision-1.1) &nbsp;&middot;&nbsp; [OpenRouter](https://openrouter.ai)

</div>

### About

Maps the columns of an input table onto the output fields of a target and gives a converted table. The Decisions model (`microsoft/microsoft-decision-1.1` through OpenRouter) reads the kind of each output field from its description, scores which columns feed which field, and binds each named role of the description to one column (or none). Code builds the expression from those columns (copy, year, date difference, arithmetic, concat, reference-table lookup), and a field is filled only when its source is present: `p_present` reaches the fill score and `p_none` stays below the ceiling. Everything else is deterministic.

The core library is one small package with no third-party dependency. To use it in another project, give it your own knowledge folder (output fields and reference tables):

```python
from columnmap import Knowledge, Mapper, OpenRouterDecider

knowledge = Knowledge.load("my_knowledge/")   # outputs.json + reference/*.csv, see data/knowledge/ for the format
decider = OpenRouterDecider(api_key, "microsoft/microsoft-decision-1.1", "https://openrouter.ai/api/alpha/decisions")
mapping = Mapper(decider, knowledge).map(rows)        # rows: list of dicts
header, converted = mapping.convert(rows)             # a field without a present source stays empty
```

`columnmap.bench` evaluates the mapper on synthetic samples with a ground truth; a project that only maps tables never imports it.

The optional graph harness needs the `graph` extra and adds no dependency to the core:

```bash
uv sync --extra graph
```

```python
from columnmap.harness import run_graph

run = run_graph(rows, decider, knowledge, thread_id="batch-1")
header, converted = run.header, run.converted
if run.paused:                                  # a field was close to the fill policy
    run.resume({"total_charge": "accept"})      # or "empty"
```

---

### Getting Started

```bash
cp .env.example .env   # then put the OpenRouter key in .env
make test              # unit tests, no model call
make run               # asks: a, b, c, d, or all (or make run SAMPLES=c)
make help              # all commands
```

Settings live in `appsettings.json` (every option is listed in `appsettings.json.examples`) and `appsettings.local.json` (your overrides, gitignored). Environment variables `COLUMNMAP__Section__Key` override both.

### Running the bench

```bash
uv run columnmap                 # samples a, b, c, d
uv run columnmap --samples ac    # only a and c
uv run columnmap --split holdout # seeds 12 and 13, never tuned on
uv run columnmap --report-only --fill-score 0.8 --copy-fill-score 0.5   # rebuild the newest run, no model
uv run columnmap --deepeval results/<run>          # DeepEval metrics, no model
```

Samples: `a` identical names, `b` synonyms, `c` unusual technical terms, `d` mixed styles with extra columns. Every run writes `results/<run>/report/summary.md` with each count as x / y against the ground truth, and per sample the converted file, the decisions, and the column scores.

<div align="center">
<sub>Copyright &copy; 2026 Raihan Putra Kirana. All rights reserved.</sub>
</div>
