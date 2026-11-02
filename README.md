# Column mapping with a local decision model

Experiment: map messy input columns (any naming) to a canonical template, with
Jev-Style-2B-Decision-v3 running locally through llama.cpp on Apple Silicon.
The model returns a probability for each output field. No text is generated.

## Run

```
make setup     # once: Python deps (uv), model download (2 GB), build the scorer
make sample    # samples a,b,c,d, about 5 minutes
make report    # rebuild figures and tables without the model
make help      # all commands
```

Same thing without make: `uv run python src --samples abcd`.
Options: `--all`, `--mode pairs`, `--permute`, `--conds name,samples`, `--out DIR`.

## Output

`results/latest/`

- `result.json`: all raw results, one readable file
- `report/summary.md`: tables and figures
- `report/figures/*.svg`: accuracy, outcome breakdown, drift regression, calibration, transformations, one heatmap per sample
- `report/predictions_<sample>_<context>.csv`: predicted input column and probability for every output field
- `samples/<case>/`: `input.csv`, `expected_output.csv`, `mapping.csv` (ground truth)

## Samples

| id | naming of input columns |
|---|---|
| a | identical to the template |
| b | synonyms |
| c | legacy codes such as OBJ_INFO_ADDR_C |
| d | mixed, with decoy columns |
| e, f, g | style only, abbreviations, Indonesian |

Each input file has 15 to 30 rows and 1 to 2 anomalies (a zero, an empty cell, a sentinel date).
Some output fields have no source column and must stay empty. Some need a transformation
(concat, calculation, lookup).

## What the model sees

The template is a header only. For each input column the model gets the column name and 5 sample
values, and the output fields by name. Column descriptions are not given (`--conds desc` exists
as an upper bound).

Modes: `options` (default) asks one question per input column and reads a score per output field.
`pairs` asks one yes/no question per column and field. It is about 7 times slower and calibrated.

## Tests

`make test` runs the unit tests with a fake model. `make bench` measures the two modes on the real model.
Notes, findings and the task plan are in `NOTES.md` and `tasks/`.
