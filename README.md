# Column mapping with a local decision model

Experiment: map messy input columns (any naming) onto a canonical template with
Jev-Style-2B-Decision-v3, running locally through llama.cpp on Apple Silicon.
The model returns a probability for each output field. It does not generate text.

## Quick start

```bash
make setup     # once: Python deps (uv), model download (2 GB), build the scorer against brew llama.cpp
make sample    # samples a, b, c, d (about 6 minutes on an M3)
make open      # open the summary of the newest run
make help      # every command
```

## Run directly

```bash
uv run python src                                  # samples a,b,c,d, fast mode
uv run python src --samples ac                     # only a (identical) and c (coded)
uv run python src --all                            # all seven samples
uv run python src --conds name                     # the model sees column names only
uv run python src --mode pairs                     # slow, calibrated probabilities
uv run python src --permute                        # also test option-order stability
uv run python src --out results/<run folder>       # continue or extend an existing run
uv run python src --report-only                    # rebuild files and figures of the newest run, no model
uv run python src --help
```

| option | meaning |
|---|---|
| `--samples abcd` | which samples to run, by letter (table below). Default `abcd`. |
| `--all` | all seven samples. |
| `--conds name,samples` | what the model sees about each input column. `name`: the header text only. `samples`: the header plus 5 values taken from the rows. `desc` and `both` add a column description; they are not used by default because the real template is a header only. |
| `--mode options` | default. One question per input column with all output fields as options; the score of each option is read. About 7 times faster. Probabilities are not calibrated. |
| `--mode pairs` | one yes/no question per (input column, output field). Slower, calibrated. |
| `--permute` | also re-ask each choice question with shuffled option order to see how stable the answers are. |
| `--report-only` | no model; rebuild the files, figures and tables from `result.json`. |
| `--root DIR` | folder that holds all runs. Default `results`. |
| `--data DIR` | folder with the template and samples. Default `data`. |
| `--out DIR` | use this run folder (resumes it) instead of creating a new one. |

Model settings (quantisation, GPU layers, flash attention, threads) are in `config.toml`, the single place for them.

## llama.cpp and the model

llama.cpp comes from Homebrew only (`brew upgrade llama.cpp`, then `make setup` to rebuild the scorer).
`vendor/jev-v3-2b-gguf/` is the model's own download: the GGUF file, its Python runtime, the scorer source and
the tokenizer. The scorer (`jev-score-v2`) is that source compiled against brew's llama.cpp. Stock `llama-cli` and
`llama-server` cannot be used for this model: they generate text and cannot return the decision scores.

## Samples

| id | folder | input columns are named |
|---|---|---|
| a | `a_identical` | exactly like the template |
| b | `b_synonym` | with synonyms |
| c | `c_coded` | as system codes, for example `OBJ_INFO_ADDR_C` |
| d | `d_mixed` | a mix of all styles, plus decoy columns |
| e | `e_style` | same words, other style (camelCase, UPPER, kebab) |
| f | `f_abbreviation` | abbreviations |
| g | `g_indonesian` | Indonesian |

Each input file has 15 to 30 rows and 1 to 2 anomalies (a zero, an empty cell, a sentinel date).
Some output fields have no source column and must stay empty. Some need a transformation
(concat, calculation, lookup).

## Where files go

```text
data/                                tracked; the single source of truth for inputs
  template.json                      template definition
  template_header.csv                the empty target template
  template_columns.csv               template columns with type, kind and rule
  samples/c_coded/
    input_coded.csv                 the messy input file
    expected_coded.csv              what a perfect converter writes
    ground_truth_coded.csv          true input column and expression per output field

results/<samples>_<timestamp>/       one folder per run, for example identical-synonym_20261102-153012
  result.json                        raw model output of this run, for transparency
  c_coded/
    converted_coded_name.csv        the template filled from Jev's decisions
    decisions_coded_name.csv        per output field: chosen input column, probability, kind, expression
    probabilities_coded_name.csv    probability of every (input column, output field) pair
  report/
    summary.md                       tables and figures of this run
    figures/*.svg                    accuracy, outcomes, drift regression, calibration, kinds, heatmaps

results/latest                       link to the newest run
```

Files ending in `_name` or `_samples` say what the model saw for that context.
`results/` is not tracked by git.

## Code layout

```text
src/
  cli.py, experiment.py        command line, and running the model over the samples
  core/                        template, expression language, prompts, scoring math, result store, run folders
  decisions/                   contract shaped like OpenAI's Decisions API, and the Jev adapter
  dataset/                     synthetic samples and their CSV files
  mapping/ transform/ stability/   one slice each: run, metrics, figures, tables
  convert/                     builds the converted file from Jev's decisions and scores it
  report/                      assembles files, figures and summary.md
tests/                         mirrors src/, uses fake deciders (no model needed)
```

## Tests

```bash
make test      # unit tests, no model
make lint      # unused imports and names
make bench     # speed and accuracy of the two modes on the real model, about 5 minutes
```

Notes, findings and the task plan are in `NOTES.md` and `tasks/`.
