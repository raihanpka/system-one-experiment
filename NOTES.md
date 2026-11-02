# NOTES

## 1. Environment

1. Python: uv-managed CPython 3.12.13. Every Homebrew Python here fails `import pyexpat` (system libexpat too old); `uv` rejects them and `brew reinstall` did not help. `UV_PYTHON=3.12` is set in `~/.zshrc`.
2. llama.cpp: Homebrew only. Installed 0.8.0 (build 11960) with ggml 0.28.0 and nlohmann-json (header only, build dependency). No llama.cpp source is kept.
3. `jev-score-v2` is the model authors' `jev_score_v2.cpp` compiled against brew by `scripts/build_scorer.sh`. It rebuilds only when brew versions change. After `brew upgrade llama.cpp`, run `make setup` again.
4. Parity of that build against the authors' pinned commit 441df11: max score difference 0.0024, same top-1 in 16 of 16 questions, same speed. The model card example gives `ask_user` with p 0.857 on both.
5. `vendor/jev-v3-2b-gguf/` is the model repo download: Q8_0 GGUF (1.9 GB), the authors' Python runtime, the scorer source, tokenizer and readout config. The runtime reads them side by side.
6. `llama-server` and the llama.cpp chat tool cannot serve this model. They generate tokens and cannot return the decision scores.
7. Model settings live only in `config.toml` `[model]`: `quant`, `ngl`, `flash_attn`, `threads` (same meaning as llama.cpp `-ngl`, `-fa`, `-t`).
8. Cleanup done on request: removed brew `python@3.12` and the pip user dirs `~/Library/Python/3.9` and `3.14`. Kept `python@3.14` (poetry venv) and `python@3.15` (nmap, virtualenv, watchman).
9. Tooling quirks: `hf download` ignores `--include` when file names are given. The commit hook wants `type(scope): summary`, one line, no heredoc in the same command.

## 2. Model and speed

1. Jev-Style-2B-Decision-v3 is Qwen3.5-2B fine-tuned to score options. Option score = logit(yes) minus logit(no). Probability = softmax or sigmoid of score / T with T = 0.828. Context limit 25,600 tokens. Apache-2.0; some training data has its own terms.
2. Device: MacBook Air M3, 8-core CPU, 8-core GPU, 16 GB, Metal 4, fanless, so long runs may throttle. Memory bandwidth about 100 GB/s (not checked).
3. The load is prefill bound: about 0.1 s per yes/no question of roughly 60 tokens on Q8_0 with all layers on Metal.
4. `options` mode (one choice question per column, read each option's raw score) is about 7 times faster than `pairs` (one yes/no per column and field). Coded names, name only: 11.7 s vs 83.5 s, top-1 12/26 vs 10/26. Mixed, name + values: 25.3 s vs 159.3 s, 23/26 vs 25/26. Option probabilities are not calibrated; `pairs` is.
5. Third-party sources (vendor claims, not independent) credit the demo speed to one forward pass without a decode loop, short prompts, state reuse and many questions per request.

## 3. Data

1. Template: 26 source concepts, 31 output fields (11 derived: concat, arithmetic, format, lookup), 6 decoy columns. Rules are a small expression tree (`core/dsl.py`).
2. Seven named samples, from identical names to system codes such as `OBJ_INFO_ADDR_C` (kept UPPER_SNAKE). See the README table.
3. Each file keeps 14 to 22 of the 26 source columns (so some fields have no source), 15 to 30 rows, and 1 to 2 anomalies (0, negative, x1000, empty, `N/A`, sentinel date). Where a rule cannot handle a value the expected cell is blank.
4. The real template is a header only, so the model gets the input column name (and 5 sample values) and output names only. Descriptions exist in code (`desc`, `both`) but are not used.
5. `data/` is tracked and is the only copy of inputs, expected files and ground truth. A run writes only its own outputs.
6. CSV only. An `.xlsx` file would need `openpyxl`.

## 4. Decisions

1. `decisions/` copies the shape of OpenAI's Decisions API (predicate, choice, score). The official page failed to load five times (`ECONNRESET`), so field names come from LocalAI, LangChain and TypeSafe docs and are unverified, notably the element keys of `probabilities`.
2. Temperature on Jev only rescales scores (calibration) and never changes the top answer. OpenAI and TypeSafe expose no temperature.
3. Option order matters for `choice` (smoke test, 3 orders): the transform pick stayed the same for 10/11 and 9/11 outputs, the kind pick for 25/31 and 28/31. Average over orders when it matters.
4. Candidate expressions come from the ground truth (1 true, 3 distractors). Jev picks one; it does not write formulas.
5. A field is kept when its best input column passes a rule: fixed 0.5, or a cutoff on the best score or on its lift above the median column, fitted on ground truth (`--calibrate-from RUN` fits it on another run).

## 5. Real Runs on Two Cases

1. Two separate runs, each with 3 contexts and both modes: `coded-000` (26 columns, 21 of 31 fields have a source) in `results/coded_20261102-103424` and `mixed-000` (32 columns with 6 decoys, 16 of 31 fields have a source) in `results/mixed_20261102-104030`. Each folder has its own `report/summary.md`. One file per case, so read the numbers as indications.
2. Correction: an earlier note said fields without a source were never rejected. That was a division by zero: those samples had no such fields. The generator now keeps only 14 to 22 of 26 source columns, and undefined rates print as "-".
3. Top-1 (options / pairs). Coded: name only 61.9 / 52.4, name + values 71.4 / 52.4, name + whole table 57.1 / 33.3. Mixed: name only 81.2 / 93.8, name + values 87.5 / 93.8, name + whole table 93.8 / 93.8.
4. Time per file: options 14 to 33 s, pairs 70 to 144 s.
5. Fixed rule (probability 0.5): on coded it keeps only 33 to 57 percent of the fields that have a source and leaves 100 percent of the no-source fields empty (too cautious on system codes). On mixed it keeps 75 to 81 percent and leaves 80 to 93 percent of the no-source fields empty.
6. Cutoffs fitted on ground truth: on the same run they help a little (coded name only: fields kept 33 to 57 percent, converted cells correct 28.6 to 42.9 percent) but that is optimistic. Fitted on the other run they do not help and are often worse (mixed name only, options: no-source fields left empty 80 to 66.7 percent; mixed name + values, pairs: 93.3 to 53.3). Scores sit on a different scale in each sample, so one cutoff does not transfer; more files are needed to say more.
7. Whole table: helps mixed (top-1 93.8 in options mode), hurts coded (57.1 vs 71.4 with values), and in pairs mode the model accepts almost every field (only 20 to 27 percent of no-source fields left empty at the fixed rule). It costs about 1.5 times the time of name + values.
8. Converted cells correct (fixed rule): coded 14 to 48 percent, mixed 69 to 81 percent. Derived fields lose accuracy through the kind and expression picks.
9. Takeaway: name + values in options mode is the best default for speed and accuracy. Pairs mode is better only on the easier file and is 5 to 6 times slower.

## 6. Structure

1. Package `columnmap` in `src/`, one command: `uv run columnmap`.
2. Layers: `core` and `decisions` depend on nothing above them. `dataset` uses `core`. The slices `mapping`, `transform`, `stability`, `convert` use `core` and `decisions`. `report` assembles the slices. `experiment` runs the model. `main` is the only place that knows the Jev adapter and reads `config.toml`.
3. Each slice owns its run, metrics, figures and tables. `stability` imports only `transform.prompts`, and `convert` uses `mapping.reject` for the keep rules. Only `report` imports figures.
4. Moving to OpenAI or hosted Jev means one new adapter in `decisions/` and one line in `main.py`.
5. Tests mirror `src/` and use fake deciders (`tests/fakes.py`). `make test`, `make lint`.

## 7. Open items

1. More cases per level (5 per level is about 50 minutes for 2 contexts) to get confidence intervals; today it is 1 case per sample.
2. A per-field threshold fitted on the ground truth so fields without a source and decoy columns get rejected; compare with `--mode pairs`.
3. An extra context with whole rows or the whole table (a prompt change only).
4. Check the OpenAI Decisions schema against the official page, and add an OpenAI or hosted Jev adapter if a migration is planned.
5. An `.xlsx` export, only if CSV is not enough.
