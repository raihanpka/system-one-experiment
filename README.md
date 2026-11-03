<div align="center">

# Decision Mapper

[Notes](./NOTES.md) &nbsp;&middot;&nbsp; [Model](https://openrouter.ai/liquid/d1-ultra) &nbsp;&middot;&nbsp; [OpenRouter](https://openrouter.ai)

![python](https://img.shields.io/static/v1?logo=python&label=&message=python&color=36465D&logoColor=AAA&style=flat-square)
![uv](https://img.shields.io/static/v1?logo=uv&label=&message=uv&color=36465D&logoColor=AAA&style=flat-square)
![openrouter](https://img.shields.io/static/v1?label=&message=OpenRouter&color=36465D&style=flat-square)

</div>

### About This Repo

Maps messy input columns onto a fixed output template with a decision model. Each output field gets a probability per input column. Confident fields are filled, the rest go to a review list with three candidates.

- **Decision Model**: `~liquid/d1-ultra-latest` through the OpenRouter Decisions API, hosted only.
- **Calibrated Output**: Platt scaling fitted on calibration data, stored per setup.
- **Review First**: Fields below the precision target stay empty and go to a person.

---

### Getting Started

```bash
cp .env.example .env   # then put the OpenRouter key in .env
make test              # unit tests, no model call
make run               # asks: a, b, c, d, or all (or make run SAMPLES=c)
make help              # all commands
```

Settings live in `appsettings.json` (defaults; every option is listed in `appsettings.json.examples`) and `appsettings.local.json` (your overrides, gitignored). Environment variables `COLUMNMAP__Section__Key` override both. The OpenRouter key goes in `.env`; `.env` is loaded automatically at start.

---

### Running

```bash
uv run columnmap                 # samples a, b, c, d
uv run columnmap --samples ac    # only a and c
uv run columnmap --all           # all four samples
uv run columnmap --report-only   # rebuild the newest run, no model
uv run columnmap --deepeval results/<run>   # DeepEval metrics on a finished run, no model
uv run columnmap --help          # all options
```

Samples: `a` identical names, `b` synonyms, `c` unusual technical terms, `d` mixed styles with extra columns.

---

### Output

```text
results/<samples>_<timestamp>/
  result.json       raw model output
  <sample>/         converted file, decisions, review list, probabilities
  report/           summary.md and figures
```

---

### Layout

```text
src/columnmap/
  core/        settings, calibration (Platt), scoring, store, paths
  decisions/   question types and the Decider protocol
  backends/    OpenRouterDecider, the hosted adapter
  honest/      stage without ground truth: candidate expressions ranked by the model
  mapping/     column to field scoring (options and pairs modes), reject rules
  convert/     converts files from decisions; metrics against ground truth (eval only)
  dataset/     synthetic samples and expected files (ground truth, eval only)
  report/      summary, figures, tables
  evaluate/    DeepEval metrics on a finished run against ground truth (eval only)
  calibrate.py fits Platt parameters per setup; the key holds model, quant, mode, context, prompt, cutoff, and policy version
```

To change the model, change `OpenRouter:Model` in the settings. Calibration keys refuse fits made for another model, quantization, or policy version.

### Tech Stack

- **Language**: [Python](https://www.python.org) 3.12, [uv](https://docs.astral.sh/uv)
- **Model**: [Liquid D1 Ultra](https://openrouter.ai/liquid/d1-ultra) through [OpenRouter](https://openrouter.ai)
- **Data and Figures**: [pandas](https://pandas.pydata.org), [seaborn](https://seaborn.pydata.org), [Matplotlib](https://matplotlib.org)

<div align="center">
<sub>Copyright &copy; 2026 Raihan Putra Kirana. All rights reserved.</sub>
</div>
