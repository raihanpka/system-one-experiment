<div align="center">

# Column Mapping With a Local Decision Model

[Notes](./NOTES.md) &nbsp;&middot;&nbsp; [Model](https://huggingface.co/chaoliangUNSW/Jev-Style-2B-Decision-v3-GGUF) &nbsp;&middot;&nbsp; [llama.cpp](https://github.com/ggml-org/llama.cpp)

![python](https://img.shields.io/static/v1?logo=python&label=&message=python&color=36465D&logoColor=AAA&style=flat-square)
![uv](https://img.shields.io/static/v1?logo=uv&label=&message=uv&color=36465D&logoColor=AAA&style=flat-square)
![llama.cpp](https://img.shields.io/static/v1?label=&message=llama.cpp&color=36465D&style=flat-square)
![huggingface](https://img.shields.io/static/v1?logo=huggingface&label=&message=huggingface&color=36465D&logoColor=AAA&style=flat-square)
![apple silicon](https://img.shields.io/static/v1?logo=apple&label=&message=apple%20silicon&color=36465D&logoColor=AAA&style=flat-square)

</div>

### About This Repo

An experiment that maps messy input columns, with any naming, onto a canonical template using [Jev-Style-2B-Decision-v3](https://huggingface.co/chaoliangUNSW/Jev-Style-2B-Decision-v3-GGUF) running locally through llama.cpp on Apple Silicon. The model returns a probability for each output field and never generates text. Each run writes the template filled from the model's decisions, the raw model output, and a report with figures.

- **Local Decision Model**: Jev-Style-2B-Decision-v3 (Q8_0) scored through llama.cpp on the Mac GPU, with no server and no API key.
- **Synthetic Samples**: Five samples, easiest to hardest, whose column names drift from identical headers to system codes such as `OBJ_INFO_ADDR_C`. Each file has 15 to 30 rows and 1 to 2 anomalies, such as a zero, an empty cell, or a sentinel date.
- **Real Mapping Problems**: Some output fields have no source column and must stay empty, some need a transformation (concat, calculation, lookup), and some input columns are decoys.
- **Two Scoring Modes**: `options` asks one question per input column and is about 7 times faster. `pairs` asks one yes/no question per column and field and gives calibrated probabilities.
- **Converted Files**: The template filled from the model's decisions, saved as CSV and scored against the expected file.
- **Transparent Results**: A `result.json` per run, plus per-field decisions and the full probability matrix as CSV.
- **Automatic Report**: A `summary.md` per run with tables and SVG figures in Times New Roman.
- **Clean Structure**: Vertical slices with low coupling, and tests that use fake models so they run without the model.

---

### Getting Started

This project needs a Mac with Apple Silicon, [uv](https://docs.astral.sh/uv), and [Homebrew](https://brew.sh).

```bash
# once: Python dependencies, model download (2 GB), and the scorer built against Homebrew llama.cpp
make setup

# run samples a, b, c, and d (about 6 minutes on an M3)
make sample

# open the summary of the newest run
make open

# list every make command
make help
```

Model settings (quantization, GPU layers, flash attention, and threads) live in `config.toml`, the single place for them.

---

### Running the Experiment

`make sample` runs `uv run columnmap`. Call it directly to choose the samples and options.

```bash
# run samples a, b, c, and d (the default)
uv run columnmap

# run only samples a and c
uv run columnmap --samples ac

# run all five samples
uv run columnmap --all

# show the model only the column names, without sample values
uv run columnmap --conds name

# use the slower mode with calibrated probabilities
uv run columnmap --mode pairs

# continue (or extend) an existing run folder
uv run columnmap --out results/easy_identical-medium_synonym-hard_codes-hardest_noisy_20261102-153012

# rebuild the files and figures of the newest run without the model
uv run columnmap --report-only

# build the samples from another seed (not written to data/)
uv run columnmap --seed 5

# run every seed of the dev split (for tuning), or the calibration split
uv run columnmap --split dev
uv run columnmap --split calibration

# fit calibration.json from finished dev or calibration runs (never from a test run)
uv run columnmap --fit-calibration results/RUN_A results/RUN_B

# report a run with a given calibration file
uv run columnmap --report-only --out results/RUN --calibration calibration.json

# the test split is looked at once, at the very end
uv run columnmap --split test --final

# show all options
uv run columnmap --help
```

Options:

- **`--samples abcd`**: Chooses which samples to run, by letter. The default is `abcd`.
- **`--all`**: Runs all five samples.
- **`--conds name,samples`**: Sets what the model sees about each input column. `name` is the header text only, and `samples` is the header plus 5 values taken from the rows. `desc` and `both` also add a column description. They are off by default because the real template is a header only.
- **`--mode options`**: The default. Asks one question per input column with all output fields as options and reads the score of each option. Probabilities are not calibrated.
- **`--mode pairs`**: Asks one yes/no question per input column and output field. Slower, but calibrated.
- **`--report-only`**: Rebuilds the files, figures, and tables from `result.json` without running the model.
- **`--root DIR`**: The folder that holds all runs. The default is `results`.
- **`--data DIR`**: The folder with the template and samples. The default is `data`.
- **`--out DIR`**: Uses this run folder, and resumes it, instead of creating a new one.

---

### Samples

- **a, `a_easy_identical`**: Input columns are named exactly like the template.
- **b, `b_medium_synonym`**: Input columns use synonyms.
- **c, `c_hard_codes`**: Input columns are system codes, for example `OBJ_INFO_ADDR_C`.
- **d, `d_hardest_noisy`**: A mix of all name styles, plus extra columns that match nothing in the template (distractors) and some missing sources.
- **e, `e_indonesian`**: Input columns are in Indonesian.

A few terms used in the reports:

- **Field**: one column of the output template.
- **Source**: the input column that should fill a field. Some fields have no source in a given file and must stay empty.
- **Distractor** (called decoy in the code): an input column that looks related but matches nothing in the template, for example `policy_version` next to `policy_number`.
- **Expression**: the rule that makes a field from input columns: copied as is, joined, calculated, reformatted, or looked up. The model picks one among candidates built from the columns it chose; it never writes a formula.

---

### Splits, Calibration, and Auto-Fill

Data comes from three seed splits in `config.toml`: `dev` for tuning, `calibration` for fitting, and `test` for the final look. Only the baseline seed is written to `data/`; other seeds are rebuilt from their number.

The model's scores are turned into probabilities with Platt scaling (two numbers, a slope and a bias). They are fitted on calibration runs and stored in `calibration.json`, keyed by model, quantization, mode, context, prompt version, and column cutoff. A key that does not match the current setup is refused with a message, so a change in the setup needs a new fit.

A field is filled automatically only when its calibrated confidence reaches the cutoff that gives the target precision (`target_precision` in `config.toml`, 99% by default). Every other field stays empty in the converted file and is listed in `review_*.csv` with its three best candidates. If no cutoff reaches the target with enough fields behind it, nothing is filled automatically.

---

### Where Files Go

```text
data/                                tracked, the single source of truth for inputs
  template.json                      template definition
  template_header.csv                the empty target template
  template_columns.csv               template columns with type, kind, and rule
  samples/c_hard_codes/
    input_hard_codes.csv                  the messy input file
    expected_hard_codes.csv               what a perfect converter writes
    ground_truth_hard_codes.csv           true input column and expression per output field

results/<samples>_<timestamp>/       one folder per run, for example easy_identical-medium_synonym_20261102-153012
  result.json                        raw model output of this run, for transparency
  c_hard_codes/
    converted_hard_codes_name.csv         the template filled from the model's decisions
    decisions_hard_codes_name.csv         per output field: chosen input column, confidence, calibrated confidence, method, expression
    review_hard_codes_name.csv            fields left for a person, with three candidates and their probabilities
    probabilities_hard_codes_name.csv     probability of every (input column, output field) pair
  report/
    summary.md                       tables and figures of this run
    figures/*.svg                    accuracy, outcomes, drift regression, calibration, reliability, precision against coverage, heatmaps
```

Files ending in `_name` or `_samples` show what the model saw for that context. The `results/` folder is not tracked by git.

---

### llama.cpp and the Model

llama.cpp comes from Homebrew only. After `brew upgrade llama.cpp`, run `make setup` to rebuild the scorer. The folder `vendor/jev-v3-2b-gguf/` is the model's own download: the GGUF file, its Python runtime, the scorer source, and the tokenizer. The scorer (`jev-score-v2`) is that source compiled against the Homebrew llama.cpp. The stock `llama-server` cannot serve this model because it generates text and cannot return the decision scores.

---

### Code Layout

```text
src/columnmap/
  main.py, experiment.py          entry point (the columnmap command) and the run over the samples
  core/                           template, expression language, prompts, scoring math, result store, run folders, paths
  decisions/                      question types (predicate, choice, score) and the Jev adapter
  dataset/                        synthetic samples and their CSV files
  mapping/ transform/ stability/  one slice each: run, metrics, figures, tables
  convert/                        builds the converted file from the model's decisions and scores it
  report/                         assembles files, figures, and summary.md
tests/                            mirrors src/columnmap and uses fake deciders
```

```bash
# unit tests, no model needed
make test

# unused imports and names
make lint

# speed and accuracy of the two modes on the real model (about 5 minutes)
make bench
```

---

### Tech Stack

- **Language & Tooling**: [Python](https://www.python.org) 3.12, [uv](https://docs.astral.sh/uv)
- **Model & Runtime**: [Jev-Style-2B-Decision-v3](https://huggingface.co/chaoliangUNSW/Jev-Style-2B-Decision-v3-GGUF), [llama.cpp](https://github.com/ggml-org/llama.cpp)
- **Data & Figures**: [NumPy](https://numpy.org), [pandas](https://pandas.pydata.org), [seaborn](https://seaborn.pydata.org), [Matplotlib](https://matplotlib.org)
- **Testing & Linting**: [pytest](https://pytest.org), [pyflakes](https://github.com/PyCQA/pyflakes)

