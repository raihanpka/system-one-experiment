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

- **Decision Model**: Jev-Style-2B-Decision-v3 (Q8_0) through llama.cpp, local, or a hosted backend (planned).
- **Calibrated Output**: Platt scaling fitted on calibration data, stored per setup.
- **Review First**: Fields below the precision target stay empty and go to a person.

---

### Getting Started

```bash
make setup     # once: Python dependencies, model download, scorer build
make sample    # run samples a, b, c, d
make help      # all commands
```

Settings live in `config.toml`. Pick the backend with `[backend] type`, and the device with `[device] mode` (`auto` uses Metal, CUDA, or CPU). API keys go in environment variables, never in the file.

---

### Running

```bash
uv run columnmap                 # samples a, b, c, d
uv run columnmap --samples ac    # only a and c
uv run columnmap --all           # all five samples
uv run columnmap --report-only   # rebuild the newest run, no model
uv run columnmap --help          # all options
```

Samples: `a` identical names, `b` synonyms, `c` system codes, `d` mixed styles with extra columns, `e` Indonesian.

---

### Output

```text
results/<samples>_<timestamp>/
  result.json       raw model output
  <sample>/         converted file, decisions, review list, probabilities
  report/           summary.md and figures
```

---

### Tech Stack

- **Language**: [Python](https://www.python.org) 3.12, [uv](https://docs.astral.sh/uv)
- **Model**: [Jev-Style-2B-Decision-v3](https://huggingface.co/chaoliangUNSW/Jev-Style-2B-Decision-v3-GGUF), [llama.cpp](https://github.com/ggml-org/llama.cpp)
- **Data and Figures**: [pandas](https://pandas.pydata.org), [seaborn](https://seaborn.pydata.org), [Matplotlib](https://matplotlib.org)

<div align="center">
<sub>Copyright &copy; 2026 Raihan Putra Kirana. All rights reserved.</sub>
</div>
