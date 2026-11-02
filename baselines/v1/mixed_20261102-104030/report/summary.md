# Results

Model: Jev-Style-2B-Decision-v3 (llama.cpp, local). Output fields are always given by name only. For each input column the model sees: **name only** = the header text; **name + values** = the header and 5 values taken from the rows (file keys: `name`, `samples`).

| sample | input, expected, ground truth | Jev outputs (this run) | input naming |
|---|---|---|---|
| d | data/samples/d_mixed/ | d_mixed/ | mixed, decoys |

Per sample and context: the converted file, the decisions and the probability matrix. Raw model output for everything: `result.json`.

## At a Glance

- **name only, options mode**: top-1 81.2%, exact 75.0%. Fields without a source left empty: 80.0% with the fixed rule, 66.7% with a cutoff (fitted on another run). Converted cells correct: 68.8% fixed, 68.8% with the cutoff; wrongly filled cells: 69 and 115. 18 s per file.
- **name only, pairs mode**: top-1 93.8%, exact 81.2%. Fields without a source left empty: 86.7% with the fixed rule, 73.3% with a cutoff (fitted on another run). Converted cells correct: 75.0% fixed, 81.2% with the cutoff; wrongly filled cells: 46 and 92. 105 s per file.
- **name + values, options mode**: top-1 87.5%, exact 81.2%. Fields without a source left empty: 80.0% with the fixed rule, 80.0% with a cutoff (fitted on another run). Converted cells correct: 75.0% fixed, 75.0% with the cutoff; wrongly filled cells: 69 and 69. 26 s per file.
- **name + values, pairs mode**: top-1 93.8%, exact 81.2%. Fields without a source left empty: 93.3% with the fixed rule, 53.3% with a cutoff (fitted on another run). Converted cells correct: 81.2% fixed, 87.5% with the cutoff; wrongly filled cells: 23 and 161. 118 s per file.
- **name + whole table, options mode**: top-1 93.8%, exact 81.2%. Fields without a source left empty: 86.7% with the fixed rule, 93.3% with a cutoff (fitted on another run). Converted cells correct: 68.8% fixed, 68.8% with the cutoff; wrongly filled cells: 46 and 23. 33 s per file.
- **name + whole table, pairs mode**: top-1 93.8%, exact 87.5%. Fields without a source left empty: 26.7% with the fixed rule, 40.0% with a cutoff (fitted on another run). Converted cells correct: 68.8% fixed, 68.8% with the cutoff; wrongly filled cells: 253 and 207. 144 s per file.

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
| 6 | 1 | 81.2 | 75.0 | 93.8 | 80.0 | 50.0 | 63.2 | 0.017 |
| all | 1 | 81.2 | 75.0 | 93.8 | 80.0 | - | 63.2 | 0.017 |

### Jev sees: name + values

| level | cases | top1 | exact | top3 | reject n/a | decoy flagged | pair F1 | ECE |
|---|---|---|---|---|---|---|---|---|
| 6 | 1 | 87.5 | 81.2 | 93.8 | 80.0 | 66.7 | 66.7 | 0.013 |
| all | 1 | 87.5 | 81.2 | 93.8 | 80.0 | - | 66.7 | 0.013 |

### Jev sees: name + whole table

| level | cases | top1 | exact | top3 | reject n/a | decoy flagged | pair F1 | ECE |
|---|---|---|---|---|---|---|---|---|
| 6 | 1 | 93.8 | 81.2 | 100.0 | 86.7 | 16.7 | 70.3 | 0.022 |
| all | 1 | 93.8 | 81.2 | 100.0 | 86.7 | - | 70.3 | 0.022 |

### Baseline without a model (string similarity of names only)

| level | top1 | exact | top3 |
|---|---|---|---|
| 6 | 56.2 | 43.8 | 62.5 |

## Temperature and calibration (pooled)

Probability = sigmoid((s_true - s_false) / T). T = 0.828 is the calibrated value. top1 does not depend on T.

Jev sees: name only

| T | ECE | Brier | pair F1 | precision | recall |
|---|---|---|---|---|---|
| 0.500 | 0.014 | 0.018 | 63.2 | 66.7 | 60.0 |
| 0.700 | 0.010 | 0.018 | 63.2 | 66.7 | 60.0 |
| 0.828 | 0.017 | 0.018 | 63.2 | 66.7 | 60.0 |
| 1.000 | 0.031 | 0.019 | 63.2 | 66.7 | 60.0 |
| 1.500 | 0.085 | 0.026 | 63.2 | 66.7 | 60.0 |
| 2.000 | 0.136 | 0.037 | 63.2 | 66.7 | 60.0 |
| 3.000 | 0.221 | 0.066 | 63.2 | 66.7 | 60.0 |

Jev sees: name + values

| T | ECE | Brier | pair F1 | precision | recall |
|---|---|---|---|---|---|
| 0.500 | 0.013 | 0.016 | 66.7 | 68.4 | 65.0 |
| 0.700 | 0.007 | 0.016 | 66.7 | 68.4 | 65.0 |
| 0.828 | 0.013 | 0.016 | 66.7 | 68.4 | 65.0 |
| 1.000 | 0.027 | 0.017 | 66.7 | 68.4 | 65.0 |
| 1.500 | 0.078 | 0.022 | 66.7 | 68.4 | 65.0 |
| 2.000 | 0.129 | 0.033 | 66.7 | 68.4 | 65.0 |
| 3.000 | 0.214 | 0.062 | 66.7 | 68.4 | 65.0 |

Jev sees: name + whole table

| T | ECE | Brier | pair F1 | precision | recall |
|---|---|---|---|---|---|
| 0.500 | 0.015 | 0.014 | 70.3 | 76.5 | 65.0 |
| 0.700 | 0.011 | 0.015 | 70.3 | 76.5 | 65.0 |
| 0.828 | 0.022 | 0.016 | 70.3 | 76.5 | 65.0 |
| 1.000 | 0.037 | 0.017 | 70.3 | 76.5 | 65.0 |
| 1.500 | 0.097 | 0.025 | 70.3 | 76.5 | 65.0 |
| 2.000 | 0.154 | 0.039 | 70.3 | 76.5 | 65.0 |
| 3.000 | 0.236 | 0.071 | 70.3 | 76.5 | 65.0 |

## Transformation kind (given the true source columns)

| Jev sees | level | n outputs | accuracy |
|---|---|---|---|
| name only | 6 | 16 | 100.0 |
| name + values | 6 | 16 | 93.8 |
| name + whole table | 6 | 16 | 93.8 |

## Rejecting Fields Without a Source

A field is kept when its best input column passes the rule. Kept = fields with a source that are kept; Rejected = fields without a source that are left empty; Decoys = kept fields whose best column is a decoy. Cutoffs fitted on this run are optimistic (same data); cutoffs fitted on another run are a fair test.

| Jev sees | rule | kept % | rejected % | balanced % | top-1 when kept % | decoys | converted cells correct % | wrongly filled |
|---|---|---|---|---|---|---|---|---|
| name only | fixed, probability 0.5 | 75.0 | 80.0 | 77.5 | 91.7 | 2 | 68.8 | 69 |
| name only | score cutoff 0.22, fitted on this run, optimistic | 68.8 | 86.7 | 77.7 | 90.9 | 2 | 62.5 | 46 |
| name only | lift cutoff 3.90, fitted on this run, optimistic | 75.0 | 86.7 | 80.8 | 91.7 | 2 | 68.8 | 46 |
| name only | score cutoff -0.42, fitted on other run | 81.2 | 66.7 | 74.0 | 92.3 | 4 | 68.8 | 115 |
| name only | lift cutoff 2.74, fitted on other run | 81.2 | 60.0 | 70.6 | 92.3 | 4 | 68.8 | 138 |
| name + values | fixed, probability 0.5 | 81.2 | 80.0 | 80.6 | 100.0 | 2 | 75.0 | 69 |
| name + values | score cutoff 0.09, fitted on this run, optimistic | 81.2 | 86.7 | 84.0 | 100.0 | 1 | 75.0 | 46 |
| name + values | lift cutoff 4.89, fitted on this run, optimistic | 62.5 | 100.0 | 81.2 | 100.0 | 0 | 62.5 | 0 |
| name + values | score cutoff -0.07, fitted on other run | 81.2 | 80.0 | 80.6 | 100.0 | 2 | 75.0 | 69 |
| name + values | lift cutoff 2.96, fitted on other run | 81.2 | 73.3 | 77.3 | 100.0 | 2 | 75.0 | 92 |
| name + whole table | fixed, probability 0.5 | 81.2 | 86.7 | 84.0 | 100.0 | 0 | 68.8 | 46 |
| name + whole table | score cutoff 0.47, fitted on this run, optimistic | 81.2 | 93.3 | 87.3 | 100.0 | 0 | 68.8 | 23 |
| name + whole table | lift cutoff 3.84, fitted on this run, optimistic | 81.2 | 93.3 | 87.3 | 100.0 | 0 | 68.8 | 23 |
| name + whole table | score cutoff 0.02, fitted on other run | 81.2 | 93.3 | 87.3 | 100.0 | 0 | 68.8 | 23 |
| name + whole table | lift cutoff 3.20, fitted on other run | 81.2 | 86.7 | 84.0 | 100.0 | 1 | 68.8 | 46 |

## Options Mode Versus Pairs Mode

| Jev sees | mode | top1 | exact | reject n/a (fixed 0.5) | pair F1 | ECE | seconds per file |
|---|---|---|---|---|---|---|---|
| name only | options | 81.2 | 75.0 | 80.0 | 63.2 | 0.017 | 18 |
| name only | pairs | 93.8 | 81.2 | 86.7 | 62.9 | 0.039 | 105 |
| name + values | options | 87.5 | 81.2 | 80.0 | 66.7 | 0.013 | 26 |
| name + values | pairs | 93.8 | 81.2 | 93.3 | 66.7 | 0.033 | 118 |
| name + whole table | options | 93.8 | 81.2 | 86.7 | 70.3 | 0.022 | 33 |
| name + whole table | pairs | 93.8 | 87.5 | 26.7 | 20.3 | 0.306 | 144 |

## Converted File Against the Expected File

Cells that should hold a value and match, and cells filled where nothing was expected (rule: probability 0.5). A derived field uses the expression Jev picked among the candidates.

| Jev sees | sample | cells expected | correct % | wrongly filled |
|---|---|---|---|---|
| name only | mixed-000 | 368 | 68.8 | 69 |
| name + values | mixed-000 | 368 | 75.0 | 69 |
| name + whole table | mixed-000 | 368 | 68.8 | 46 |

## Right expression among 4 candidates (chance 25%)

| Jev sees | level | n | choice | yes/no per candidate | choice value ok | yes/no value ok |
|---|---|---|---|---|---|---|
| name only | 6 | 5 | 80.0 | 60.0 | 80.0 | 60.0 |
| name + values | 6 | 5 | 100.0 | 60.0 | 100.0 | 60.0 |
