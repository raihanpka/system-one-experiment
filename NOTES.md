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
2. Four named samples, from identical names to unusual technical terms such as `OBJ_INFO_ADDR_C` (kept UPPER_SNAKE). The Indonesian sample was removed on request.
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

## 8. Lift Sweep on Dev (E2 and E3)

1. Sweep on dev, seed 11, four samples, name plus values: 12 points (lift_top 0.5 to 4.0, lift_second 1.0 to 4.0). Table in `results/sweep/summary.md`.
2. Baseline with the fixed rule (keep 0.0, probability 0.5): 47 of 74 fields with a source correct, 73 wrongly filled cells, 63.1 percent cells correct.
3. Criterion: fields correct must rise and wrongly filled cells may rise by at most 10 percent (at most 80 cells).
4. Result: no point qualifies. Points that fill more fields (lift_top 0.5 to 2.0) reach 49 to 57 fields correct but 481 to 867 wrongly filled cells. The strictest point (lift_top 4.0) fills fewer cells (88) but fields correct drops to 29 to 44.
5. Decision: keep the fixed rule (keep 0.0). The lift rule stays in code and config (commented out) for later.

## 9. Outdesc Context (R1)

1. Run: dev seed 11, sample d, options mode. Context `outdesc` = output name + template description.
2. Results (22 fields with source):

| Jev sees | top-1 | fields correct | cells correct % | wrong fills |
|---|---|---|---|---|
| name only | 90.9 | 15 | 68.2 | 52 |
| name + values | 90.9 | 16 | 72.7 | 26 |
| name + outdesc | 86.4 | 15 | 68.2 | 52 |

3. Decision: rejected. Top-1 down, no more fields correct, wrong fills up. Not default. Code kept.
4. Caveat: one file. 86.4 vs 90.9 is one field.

## 10. Calibration Fit and Held-Out Check (F1–F3, C4)

1. Honest + mapping runs: seeds 12 and 13, samples abcd, contexts name and samples, options mode, keep 0.0.
2. Fit: Platt on seed 12 only (`calibration.json`, keys `...|options|name|v1/keep0.0|field` and `|pair`, same for samples).
3. Held-out check on seed 13 (field level, 52 points):

| Jev sees | ECE raw | ECE Platt | target |
|---|---|---|---|
| name | 0.149 | 0.101 | 0.03 |
| name + values | 0.093 | 0.056 | 0.03 |

Pair level is fine: ECE 0.008 and 0.011 after Platt.

4. Result: field ECE fails the target in both contexts. Points are few (about 52 to 57). Not fixed by more seeds here (F4 removed).
5. C4 (Platt vs temperature, held-out seed 13): log-loss and ECE are almost equal. Field name: Platt ll 0.534 ECE 0.101, temperature ll 0.535 ECE 0.086. Field samples: Platt ll 0.421 ECE 0.056, temperature ll 0.422 ECE 0.075. Decision: keep Platt (two parameters, already stored). Temperature code stays for the calibration module.

## 11. Policy Version and Config Check (C3, K1, K4, K6, K7, D1, D2)

1. Calibration key now ends with the policy version `p1` (`calibrate.POLICY_VERSION`). Old keys were dropped; `calibration.json` refit on seed 12 (same numbers as section 10).
2. `config.toml` checked on load (`core/config.py`): backend type, quant, run mode, target precision, splits. `[device] mode` picks GPU layers (cpu = 0).
3. Tests: 133 pass, `pyflakes` clean.
4. README describes the layout and the local-only backend.

## 12. OpenRouter Settings (not yet an adapter)

1. `[backend.openrouter]` in `config.toml`: model `~liquid/d1-ultra-latest`, base URL `https://openrouter.ai/api/alpha/decisions`, key from env `OPENROUTER_API_KEY`. The key is never in the file. `.env.example` and `.env` (gitignored) are set up. `.env` is not loaded automatically.
2. Output is calibrated probabilities (`noul` probability, `choice` and `score` distributions), not logits. The `Backend.score` contract (raw logits) cannot be met as is. Calibration would have to be fitted on probabilities under a separate backend key.
3. Retention: the pasted API page has no privacy text. OpenRouter's listing says successful requests may be used to train Liquid models. Send synthetic data only until that is checked.
4. `main` refuses any backend type other than `local_gguf` until an adapter exists.

## 13. Hosted Backend (OpenRouter d1-ultra) Calibration

1. Hosted run on synthetic data only: seeds 12 and 13, samples abcd, contexts name and samples, options mode. Calibration key `~liquid/d1-ultra-latest|api|...`.
2. Fit on seed 12 only (`calibration.json`). Check on seed 13 (held out, 57 to 62 field points).
3. Pair level: ECE 0.014 to 0.015 raw, 0.006 to 0.007 calibrated. Good.
4. Field level: ECE 0.182 raw, 0.129 calibrated (name); 0.104 raw, 0.096 calibrated (name + values). Target 0.03: fails, same as Jev.
5. Hosted mapping ranking is fine (top-1 on sample d 90.9 to 95.5). Field confidence is still not good enough for auto-fill at 100 percent.
6. Adapter fixes found by a live call: score probabilities are keyed by index with a legend; choice and score raw scores use logit(p) so 0 means 0.5 (log(p) rejected every field).

