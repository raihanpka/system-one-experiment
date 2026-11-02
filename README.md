<div align="center">

# Decision Mapper

[Notes](./NOTES.md) &nbsp;&middot;&nbsp; [Model](https://huggingface.co/chaoliangUNSW/Jev-Style-2B-Decision-v3-GGUF) &nbsp;&middot;&nbsp; [llama.cpp](https://github.com/ggml-org/llama.cpp)

![python](https://img.shields.io/static/v1?logo=python&label=&message=python&color=36465D&logoColor=AAA&style=flat-square)
![uv](https://img.shields.io/static/v1?logo=uv&label=&message=uv&color=36465D&logoColor=AAA&style=flat-square)
![llama.cpp](https://img.shields.io/static/v1?label=&message=llama.cpp&color=36465D&style=flat-square)
![huggingface](https://img.shields.io/static/v1?logo=huggingface&label=&message=huggingface&color=36465D&logoColor=AAA&style=flat-square)

</div>

### About This Repo

Maps messy input columns onto a fixed output template with a decision model. Each output field gets a probability per input column. Confident fields are filled, the rest go to a review list with three candidates.

- **Decision Model**: Jev-Style-2B-Decision-v3 (Q8_0) through llama.cpp, local only.
- **Calibrated Output**: Platt scaling fitted on calibration data, stored per setup.
- **Review First**: Fields below the precision target stay empty and go to a person.

---

### Getting Started

```bash
make setup     # once: Python dependencies, model download, scorer build
make run       # asks: a, b, c, d, or all (or make run SAMPLES=c)
make sample    # run samples a, b, c, d
make help      # all commands
```

Settings live in `config.toml` and are checked when the run starts. `[backend] type` is `local_gguf` (the only one). `[device] mode` is `auto` (Metal on Apple Silicon, else CPU), `metal`, or `cpu`.

---

### Running

```bash
uv run columnmap                 # samples a, b, c, d
uv run columnmap --samples ac    # only a and c
uv run columnmap --all           # all four samples
uv run columnmap --report-only   # rebuild the newest run, no model
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
  core/        config, device, calibration (Platt, temperature), scoring, store, paths
  decisions/   question types and the Decider protocol; JevDecider is the local adapter
  honest/      stage without ground truth: candidate expressions ranked by the model
  mapping/     column to field scoring (options and pairs modes), reject rules
  convert/     converts files from decisions; metrics against ground truth (eval only)
  dataset/     synthetic samples and expected files (ground truth, eval only)
  report/      summary, figures, tables
  calibrate.py fits Platt parameters per setup; the key holds model, quant, mode, context, prompt, cutoff, and policy version
```

To change the model, add an adapter in `decisions/` that implements `decide`, then choose it in `main.py`. Calibration keys refuse fits made for another model, quantization, or policy version.

### Tech Stack

- **Language**: [Python](https://www.python.org) 3.12, [uv](https://docs.astral.sh/uv)
- **Model**: [Jev-Style-2B-Decision-v3](https://huggingface.co/chaoliangUNSW/Jev-Style-2B-Decision-v3-GGUF), [llama.cpp](https://github.com/ggml-org/llama.cpp)
- **Data and Figures**: [pandas](https://pandas.pydata.org), [seaborn](https://seaborn.pydata.org), [Matplotlib](https://matplotlib.org)

<div align="center">
<sub>Copyright &copy; 2026 Raihan Putra Kirana. All rights reserved.</sub>
</div>
