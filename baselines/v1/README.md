# Baseline v1

Frozen results of the first real runs, kept so later changes can be compared against them.

### Runs

- `coded_20261102-103424`: sample `coded-000` (later renamed `hard_codes-000`). 26 columns, 21 of 31 output fields have a source.
- `mixed_20261102-104030`: sample `mixed-000` (later renamed `hardest_noisy-000`). 32 columns (6 distractors), 16 of 31 output fields have a source.

Both runs used seed 1234, three contexts (name only, name + values, name + whole table) and both modes (options and pairs). `data_hash.json` holds the hash of each input file so a test can prove the regenerated data is identical.

### Metric Definitions

- **Top-1**: for each output field that has a source, the input column with the highest score is a true source.
- **Fields kept**: fields with a source whose best column passes the rule (fixed rule: probability 0.5).
- **Converted cells correct**: share of cells equal to the expected file. In this baseline the transformation kind and the candidate expressions came from the ground truth, so this number is optimistic. `honest.md` (added later) recomputes it without the ground truth.
