# Baseline v1 Without Ground Truth Help

Same mapping scores as baseline v1 (options mode, fixed rule 0.5). The old column used the true source columns to pick the transformation kind and the candidate expressions; the new column chooses columns from the scores, builds candidates with a program and lets the model rank them. Nothing from the ground truth reaches the model or the conversion.

| Jev sees | sample | old converted cells correct % | honest cells correct % | fields correct | fields lost to wrong columns | fields lost to wrong expression | wrongly filled cells |
|---|---|---|---|---|---|---|---|
| name only | hard_codes-000 | 28.6 | 14.3 | 3 of 21 | 16 | 2 | 0 |
| name only | hardest_noisy-000 | 68.8 | 37.5 | 6 of 16 | 7 | 3 | 69 |
| name + values | hard_codes-000 | 47.6 | 23.8 | 5 of 21 | 12 | 4 | 0 |
| name + values | hardest_noisy-000 | 75.0 | 37.5 | 6 of 16 | 6 | 4 | 69 |
| name + whole table | hard_codes-000 | 33.3 | 23.8 | 5 of 21 | 14 | 2 | 30 |
| name + whole table | hardest_noisy-000 | 68.8 | 43.8 | 7 of 16 | 6 | 3 | 46 |

Fields lost to wrong columns: the stage chose a different set of columns than the true sources (including no column at all). Fields lost to wrong expression: the columns were right but the picked expression, or a code table that no catalog can guess, was not.
