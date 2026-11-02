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
6. Not measured yet: F16 vs Q8_0 vs Q4_K_M speed, shorter prompts, a two-stage prune, MLX.

## 3. Data

1. Template: 26 source concepts, 31 output fields (11 derived: concat, arithmetic, format, lookup), 6 decoy columns. Rules are a small expression tree (`core/dsl.py`).
2. Seven named samples, from identical names to system codes such as `OBJ_INFO_ADDR_C` (kept UPPER_SNAKE). See the README table.
3. Each file has 15 to 30 rows and 1 to 2 anomalies (0, negative, x1000, empty, `N/A`, sentinel date). Where a rule cannot handle a value the expected cell is blank.
4. The real template is a header only, so the model gets the input column name (and 5 sample values) and output names only. Descriptions exist in code (`desc`, `both`) but are not used.
5. `data/` is tracked and is the only copy of inputs, expected files and ground truth. A run writes only its own outputs.
6. CSV only. An `.xlsx` file would need `openpyxl`.

## 4. Decisions

1. `decisions/` copies the shape of OpenAI's Decisions API (predicate, choice, score). The official page failed to load five times (`ECONNRESET`), so field names come from LocalAI, LangChain and TypeSafe docs and are unverified, notably the element keys of `probabilities`.
2. Temperature on Jev only rescales scores (calibration) and never changes the top answer. OpenAI and TypeSafe expose no temperature.
3. Option order matters for `choice` (smoke test, 3 orders): the transform pick stayed the same for 10/11 and 9/11 outputs, the kind pick for 25/31 and 28/31. Average over orders when it matters.
4. Candidate expressions come from the ground truth (1 true, 3 distractors). Jev picks one; it does not write formulas.

## 5. First real run (4 samples, 2 contexts, options mode, 6 minutes)

1. Top-1: name only 79.3 percent, name + values 86.2 percent. By sample, identical 95.7 / 95.7, synonym 87.1 / 90.3, coded 48.4 / 71.0, mixed 90.3 / 90.3. A string-similarity baseline gets 82.6 on identical and about 19 on the rest.
2. Pooled ECE is 0.007 at T 0.828 and grows with T (0.21 at T 3). The many negative pairs flatter it.
3. Weak spots: fields with no source are not rejected at synonym, coded and mixed (0 percent at threshold 0.5), 60 percent of decoys reached P of 0.5 or more, and pair recall at 0.5 is only 44 to 50 percent.
4. Transformation kind, given the true sources: 77 to 100 percent.
5. `results/` was cleared on request. Run `make sample` to recreate it.

## 6. Structure

1. Package `columnmap` in `src/`, one command: `uv run columnmap`.
2. Layers: `core` and `decisions` depend on nothing above them. `dataset` uses `core`. The slices `mapping`, `transform`, `stability`, `convert` use `core` and `decisions`. `report` assembles the slices. `experiment` runs the model. `main` is the only place that knows the Jev adapter and reads `config.toml`.
3. Each slice owns its run, metrics, figures and tables. `stability` imports only `transform.prompts`. Only `report` imports figures.
4. Moving to OpenAI or hosted Jev means one new adapter in `decisions/` and one line in `main.py`.
5. Tests mirror `src/` and use fake deciders (`tests/fakes.py`). `make test`, `make lint`.

## 7. Open items

1. See `tasks/TODO.md`, section "Belum dikerjakan".
