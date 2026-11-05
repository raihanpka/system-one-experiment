<div align="center">

# Column Mapper

[Architecture](./docs/ARCHITECTURE.md) &nbsp;&middot;&nbsp; [Model](https://openrouter.ai/liquid/d1-ultra) &nbsp;&middot;&nbsp; [OpenRouter](https://openrouter.ai)

</div>

### About

Maps the columns of an input table onto the output fields of a target and gives a converted table. The Decisions model (`~liquid/d1-ultra-latest` through OpenRouter) reads the kind of each output field from its description, scores which columns feed which field, chooses the set of columns a field needs (or none), and chooses the expression (copy, year, date difference, arithmetic, concat, reference-table lookup). Everything else is deterministic. A field is filled when the probability of its column set reaches the fill score.

The library is one small package with no third-party dependency. To use it in another project, give it your own knowledge folder (output fields and reference tables):

```python
from columnmap import Knowledge, Mapper, OpenRouterDecider

knowledge = Knowledge.load("my_knowledge/")   # outputs.json + reference/*.csv, see data/knowledge/ for the format
decider = OpenRouterDecider(api_key, "~liquid/d1-ultra-latest", "https://openrouter.ai/api/alpha/decisions")
mapping = Mapper(decider, knowledge).map(rows)        # rows: list of dicts
header, converted = mapping.convert(rows)             # fields below the fill score stay empty
```

`columnmap.bench` evaluates the mapper on synthetic samples with a ground truth; a project that only maps tables never imports it.

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
