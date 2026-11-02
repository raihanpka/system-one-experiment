# Results

Model: Jev-Style-2B-Decision-v3 (llama.cpp, local). Output fields are always given by name only. For each input column the model sees: **name only** = the header text; **name + values** = the header and 5 values taken from the rows (file keys: `name`, `samples`).

| sample | input, expected, ground truth | Jev outputs (this run) | input naming |
|---|---|---|---|
| c | data/samples/c_coded/ | c_coded/ | system codes |

Per sample and context: the converted file, the decisions and the probability matrix. Raw model output for everything: `result.json`.

## At a Glance

- **name only, options mode**: top-1 61.9%, exact 61.9%. Fields without a source left empty: 100.0% with the fixed rule, 100.0% with a cutoff (fitted on another run). Converted cells correct: 28.6% fixed, 28.6% with the cutoff; wrongly filled cells: 0 and 0. 14 s per file.
- **name only, pairs mode**: top-1 52.4%, exact 42.9%. Fields without a source left empty: 100.0% with the fixed rule, 100.0% with a cutoff (fitted on another run). Converted cells correct: 14.3% fixed, 9.5% with the cutoff; wrongly filled cells: 0 and 0. 70 s per file.
- **name + values, options mode**: top-1 71.4%, exact 66.7%. Fields without a source left empty: 100.0% with the fixed rule, 100.0% with a cutoff (fitted on another run). Converted cells correct: 47.6% fixed, 47.6% with the cutoff; wrongly filled cells: 0 and 0. 18 s per file.
- **name + values, pairs mode**: top-1 52.4%, exact 52.4%. Fields without a source left empty: 100.0% with the fixed rule, 100.0% with a cutoff (fitted on another run). Converted cells correct: 23.8% fixed, 28.6% with the cutoff; wrongly filled cells: 0 and 0. 113 s per file.
- **name + whole table, options mode**: top-1 57.1%, exact 57.1%. Fields without a source left empty: 80.0% with the fixed rule, 100.0% with a cutoff (fitted on another run). Converted cells correct: 33.3% fixed, 14.3% with the cutoff; wrongly filled cells: 30 and 0. 26 s per file.
- **name + whole table, pairs mode**: top-1 33.3%, exact 23.8%. Fields without a source left empty: 20.0% with the fixed rule, 70.0% with a cutoff (fitted on another run). Converted cells correct: 23.8% fixed, 4.8% with the cutoff; wrongly filled cells: 120 and 45. 114 s per file.

![accuracy](figures/accuracy.svg)

![outcomes](figures/outcomes.svg)

![drift](figures/drift.svg)

![calibration](figures/calibration.svg)

![kind](figures/kind.svg)

![transform](figures/transform.svg)

Heatmaps: `figures/heatmap_<sample>_<context>.svg`.

All numbers are percentages unless noted. Levels: 0 identical, 1 style only, 2 abbreviation, 3 synonym, 4 Indonesian, 5 system codes (OBJ_INFO_ADDR), 6 mixed + decoys. Few cases per level: read as indications.

## Mapping: Which Input Column Feeds Each Output Field (options mode)

### Jev sees: name only

| level | cases | top1 | exact | top3 | reject n/a | decoy flagged | pair F1 | ECE |
|---|---|---|---|---|---|---|---|---|
| 5 | 1 | 61.9 | 61.9 | 71.4 | 100.0 | - | 40.0 | 0.018 |
| all | 1 | 61.9 | 61.9 | 71.4 | 100.0 | - | 40.0 | 0.018 |

### Jev sees: name + values

| level | cases | top1 | exact | top3 | reject n/a | decoy flagged | pair F1 | ECE |
|---|---|---|---|---|---|---|---|---|
| 5 | 1 | 71.4 | 66.7 | 85.7 | 100.0 | - | 62.2 | 0.015 |
| all | 1 | 71.4 | 66.7 | 85.7 | 100.0 | - | 62.2 | 0.015 |

### Jev sees: name + whole table

| level | cases | top1 | exact | top3 | reject n/a | decoy flagged | pair F1 | ECE |
|---|---|---|---|---|---|---|---|---|
| 5 | 1 | 57.1 | 57.1 | 85.7 | 80.0 | - | 45.8 | 0.027 |
| all | 1 | 57.1 | 57.1 | 85.7 | 80.0 | - | 45.8 | 0.027 |

### Baseline without a model (string similarity of names only)

| level | top1 | exact | top3 |
|---|---|---|---|
| 5 | 28.6 | 28.6 | 38.1 |

## Temperature and calibration (pooled)

Probability = sigmoid((s_true - s_false) / T). T = 0.828 is the calibrated value. top1 does not depend on T.

Jev sees: name only

| T | ECE | Brier | pair F1 | precision | recall |
|---|---|---|---|---|---|
| 0.500 | 0.025 | 0.032 | 40.0 | 57.1 | 30.8 |
| 0.700 | 0.017 | 0.031 | 40.0 | 57.1 | 30.8 |
| 0.828 | 0.018 | 0.032 | 40.0 | 57.1 | 30.8 |
| 1.000 | 0.036 | 0.032 | 40.0 | 57.1 | 30.8 |
| 1.500 | 0.095 | 0.040 | 40.0 | 57.1 | 30.8 |
| 2.000 | 0.149 | 0.053 | 40.0 | 57.1 | 30.8 |
| 3.000 | 0.230 | 0.083 | 40.0 | 57.1 | 30.8 |

Jev sees: name + values

| T | ECE | Brier | pair F1 | precision | recall |
|---|---|---|---|---|---|
| 0.500 | 0.022 | 0.025 | 62.2 | 73.7 | 53.8 |
| 0.700 | 0.018 | 0.025 | 62.2 | 73.7 | 53.8 |
| 0.828 | 0.015 | 0.025 | 62.2 | 73.7 | 53.8 |
| 1.000 | 0.033 | 0.026 | 62.2 | 73.7 | 53.8 |
| 1.500 | 0.087 | 0.033 | 62.2 | 73.7 | 53.8 |
| 2.000 | 0.141 | 0.045 | 62.2 | 73.7 | 53.8 |
| 3.000 | 0.223 | 0.074 | 62.2 | 73.7 | 53.8 |

Jev sees: name + whole table

| T | ECE | Brier | pair F1 | precision | recall |
|---|---|---|---|---|---|
| 0.500 | 0.029 | 0.035 | 45.8 | 50.0 | 42.3 |
| 0.700 | 0.017 | 0.034 | 45.8 | 50.0 | 42.3 |
| 0.828 | 0.027 | 0.034 | 45.8 | 50.0 | 42.3 |
| 1.000 | 0.047 | 0.035 | 45.8 | 50.0 | 42.3 |
| 1.500 | 0.107 | 0.043 | 45.8 | 50.0 | 42.3 |
| 2.000 | 0.165 | 0.058 | 45.8 | 50.0 | 42.3 |
| 3.000 | 0.240 | 0.089 | 45.8 | 50.0 | 42.3 |

## Transformation kind (given the true source columns)

| Jev sees | level | n outputs | accuracy |
|---|---|---|---|
| name only | 5 | 21 | 71.4 |
| name + values | 5 | 21 | 85.7 |
| name + whole table | 5 | 21 | 85.7 |

## Rejecting Fields Without a Source

A field is kept when its best input column passes the rule. Kept = fields with a source that are kept; Rejected = fields without a source that are left empty; Decoys = kept fields whose best column is a decoy. Cutoffs fitted on this run are optimistic (same data); cutoffs fitted on another run are a fair test.

| Jev sees | rule | kept % | rejected % | balanced % | top-1 when kept % | decoys | converted cells correct % | wrongly filled |
|---|---|---|---|---|---|---|---|---|
| name only | fixed, probability 0.5 | 33.3 | 100.0 | 66.7 | 85.7 | 0 | 28.6 | 0 |
| name only | score cutoff -0.42, fitted on this run, optimistic | 57.1 | 100.0 | 78.6 | 66.7 | 0 | 42.9 | 0 |
| name only | lift cutoff 2.74, fitted on this run, optimistic | 57.1 | 100.0 | 78.6 | 66.7 | 0 | 42.9 | 0 |
| name only | score cutoff 0.22, fitted on other run | 33.3 | 100.0 | 66.7 | 85.7 | 0 | 28.6 | 0 |
| name only | lift cutoff 3.90, fitted on other run | 19.0 | 100.0 | 59.5 | 100.0 | 0 | 19.0 | 0 |
| name + values | fixed, probability 0.5 | 52.4 | 100.0 | 76.2 | 90.9 | 0 | 47.6 | 0 |
| name + values | score cutoff -0.07, fitted on this run, optimistic | 52.4 | 100.0 | 76.2 | 90.9 | 0 | 47.6 | 0 |
| name + values | lift cutoff 2.96, fitted on this run, optimistic | 61.9 | 100.0 | 81.0 | 84.6 | 0 | 52.4 | 0 |
| name + values | score cutoff 0.09, fitted on other run | 52.4 | 100.0 | 76.2 | 90.9 | 0 | 47.6 | 0 |
| name + values | lift cutoff 4.89, fitted on other run | 4.8 | 100.0 | 52.4 | 100.0 | 0 | 4.8 | 0 |
| name + whole table | fixed, probability 0.5 | 57.1 | 80.0 | 68.6 | 75.0 | 0 | 33.3 | 30 |
| name + whole table | score cutoff 0.02, fitted on this run, optimistic | 57.1 | 100.0 | 78.6 | 75.0 | 0 | 33.3 | 0 |
| name + whole table | lift cutoff 3.20, fitted on this run, optimistic | 42.9 | 100.0 | 71.4 | 77.8 | 0 | 23.8 | 0 |
| name + whole table | score cutoff 0.47, fitted on other run | 38.1 | 100.0 | 69.0 | 62.5 | 0 | 14.3 | 0 |
| name + whole table | lift cutoff 3.84, fitted on other run | 28.6 | 100.0 | 64.3 | 83.3 | 0 | 14.3 | 0 |

## Options Mode Versus Pairs Mode

| Jev sees | mode | top1 | exact | reject n/a (fixed 0.5) | pair F1 | ECE | seconds per file |
|---|---|---|---|---|---|---|---|
| name only | options | 61.9 | 61.9 | 100.0 | 40.0 | 0.018 | 14 |
| name only | pairs | 52.4 | 42.9 | 100.0 | 29.4 | 0.036 | 70 |
| name + values | options | 71.4 | 66.7 | 100.0 | 62.2 | 0.015 | 18 |
| name + values | pairs | 52.4 | 52.4 | 100.0 | 28.6 | 0.035 | 113 |
| name + whole table | options | 57.1 | 57.1 | 80.0 | 45.8 | 0.027 | 26 |
| name + whole table | pairs | 33.3 | 23.8 | 20.0 | 15.9 | 0.422 | 114 |

## Converted File Against the Expected File

Cells that should hold a value and match, and cells filled where nothing was expected (rule: probability 0.5). A derived field uses the expression Jev picked among the candidates.

| Jev sees | sample | cells expected | correct % | wrongly filled |
|---|---|---|---|---|
| name only | coded-000 | 315 | 28.6 | 0 |
| name + values | coded-000 | 315 | 47.6 | 0 |
| name + whole table | coded-000 | 315 | 33.3 | 30 |

## Right expression among 4 candidates (chance 25%)

| Jev sees | level | n | choice | yes/no per candidate | choice value ok | yes/no value ok |
|---|---|---|---|---|---|---|
| name only | 5 | 4 | 75.0 | 75.0 | 75.0 | 75.0 |
| name + values | 5 | 4 | 100.0 | 75.0 | 100.0 | 75.0 |
