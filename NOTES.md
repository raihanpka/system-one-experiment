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

1. `decisions/` defines the question types (predicate, choice, score) and a `Decider` protocol. `JevDecider` is the only adapter.
2. Temperature on Jev only rescales scores (calibration) and never changes the top answer.
3. Option order matters for `choice` (smoke test, 3 orders): the transform pick stayed the same for 10/11 and 9/11 outputs, the kind pick (stage since removed) for 25/31 and 28/31. Average over orders when it matters.
4. Candidate expressions come from the ground truth (1 true, 3 distractors). Jev picks one; it does not write formulas.
5. A field is kept when its best input column passes a rule: fixed 0.5, or a cutoff on the best score or on its lift above the median column, fitted on ground truth (`--calibrate-from RUN` fits it on another run).

## 5. Real Runs

1. Earlier runs (`coded`, `mixed`, one file each, 3 contexts, 2 modes) are frozen in `baselines/v1/`. Their converted cells used the true transformation kind and candidate expressions (stage since removed), so their cell numbers are optimistic. `baselines/v1/honest.md` recomputes them without ground truth help.
2. Honest stage (no ground truth reaches the model or the conversion): the three best columns per field, a program builds candidate expressions from their names and types, the model ranks them with three example results each. Baseline v1 without ground truth help: 3 to 7 of 16 to 21 fields with a source correct, depending on sample and context.
3. Dev split (seed 11, five samples, name plus values): top-1 92,1 percent, fields correct 51 of 101, wrongly filled cells 73. Ground truth enters only as the test of the result, never as input.
4. Calibration split (seeds 12 and 13) fitted Platt parameters. Pair-level calibration is good: ECE 0.006 raw and 0.005 calibrated on the test split. Field-level calibration is not: 76 and 73 points, ECE 0.114 raw and 0.126 calibrated on seed 21, 0.181 raw and 0.093 calibrated on seed 22. Target was 0.03 or less, so this fails. Few points are the likely cause.
5. Test split (seeds 21 and 22, looked at once, after the calibration fit): top-1 90,0 and 90,6 percent. No cutoff reaches the 99 percent target precision with enough fields, so nothing is filled automatically. Every field goes to the review list with three candidates. The highest precision cutoff with 15 or more fields on the calibration split was about 92 percent.
6. Experiments on dev that were tried and rejected (the rule: keep only if top-1 or correct cells rise and precision does not fall). Each was reverted in the code:
   - Whole-table context: top-1 down on coded, up on mixed; not adopted.
   - Profile context (type, range, distinct count): top-1 91,1 vs 92,1, more wrong fills.
   - Prompt wording versions v2 and v3: fields correct 47 vs 51, wrong fills 135 and 141 vs 73.
   - Option order averaging, three orders: top-1 91,1, wrong fills 120, time about 2,2 times.
   - Schema context (all other headers): top-1 90,1, wrong fills 191.
   - Neighbouring columns: top-1 89,1, wrong fills 135.
   - Two-stage (options shortlist, yes/no check): top-1 81,2, fields correct 38, correct cells 37,9; fewer wrong fills (30) but many more misses.
   - Full pairs mode: top-1 79,2, 7,7 minutes and 31k tokens per run.
7. Column cutoff (`--keep`): on dev the best cutoff is 0.0 (probability 0.5); stricter or looser cutoffs lose more than they gain.
8. Cost: options mode about 13 s and 6 to 8k prompt tokens per file on the M3; a dev split of five samples runs in about 1 to 2 minutes per context.
9. Takeaway: the reliable part is the mapping ranking (top-1 around 90 percent on these files). Field-level confidence does not yet separate right from wrong fills well enough to fill automatically at 99 percent, so the review list carries most of the work. More calibration data, or a better field-level score, is the next step; that needs more cases than this experiment has.

## 6. Structure

1. Package `columnmap` in `src/`, one command: `uv run columnmap`.
2. Layers: `core` and `decisions` depend on nothing above them. `dataset` uses `core`. The slices `mapping`, `transform`, `stability`, `convert` use `core` and `decisions`. `report` assembles the slices. `experiment` runs the model. `main` is the only place that knows the Jev adapter and reads `config.toml`.
3. Each slice owns its run, metrics, figures and tables. `stability` imports only `transform.prompts`, and `convert` uses `mapping.reject` for the keep rules. Only `report` imports figures.
4. A new model backend means one adapter in `decisions/` and one line in `main.py`.
5. Tests mirror `src/` and use fake deciders (`tests/fakes.py`). `make test`, `make lint`.

## 7. Open Items

1. None planned. Ignored on request: more cases per level, an `.xlsx` export, a quantization speed comparison, and other models.
