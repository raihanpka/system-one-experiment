# NOTES

## T0: Jev v3 GGUF on Apple M3 (Metal), done

- Backend: GGUF Q8_0 + `jev-score-v2` built against llama.cpp `441df11f65ea0b6d0c72965aaf70c8241070ddcb`. Build took about 26 s. No MLX or MPS fallback needed.
- Reproduce: `sh scripts/setup_jev.sh`.
- Model card example (`send` / `ask_user` / `discard`): answer `ask_user`, p = 0.857 on Q8_0 (card: 0.860 on F16). Raw scores: send -0.687, ask_user 0.869, discard -2.671. Global temperature 0.8279.
- `noul` (true/false) works and returns `scores` for `false` and `true`.
- Latency (Q8_0, Metal, short state, persistent scorer process): first call 0.23 s, median 0.10 s per `noul` query. Model load about 20 s.
- Python API: `JevStyleDecisionGGUF(quant="Q8_0")`, `.decide(state, question, options=None, qtype=None, temperature=None)` and `.score_many(state, questions)` (state computed once for many questions). `temperature=1.0` gives uncalibrated probabilities. `InputBudgetError` is raised above 25,600 tokens.

## Environment findings

- All Homebrew Pythons (3.12, 3.14, 3.15) fail `import pyexpat`: `Symbol not found: _XML_SetAllocTrackerActivationThreshold` (system libexpat is older than the bottles expect). `uv` rejects them with "Broken Python installation, platform.mac_ver() returned an empty value". `brew reinstall python@3.12` did not help.
- Workaround, approved by the user: uv-managed CPython 3.12.13 (`uv python install 3.12`), the exact version the Jev authors tested. `UV_PYTHON=3.12` is already set in `~/.zshrc`.
- `cmake` is not installed system-wide. It is in the uv dev group, so `uv sync` provides it in `.venv/bin`.
- Cleanup done at the user's request: `brew uninstall python@3.12` (no formula or venv depended on it; `~/.zshrc` still has a now-dangling PATH entry for it), and removed global pip user-site dirs `~/Library/Python/3.9` and `~/Library/Python/3.14`. Kept `python@3.14` (the pipx `poetry` venv uses it) and `python@3.15` (needed by nmap, virtualenv, watchman).
- `hf download` with explicit filenames ignores `--include`. Name every file you want.
- `vendor/` and `.venv/` are git-ignored.

## T1: contract (`src/decisions/`), done, OpenAI schema still unverified

- `developers.openai.com/api/docs/guides/decisions` and `platform.openai.com/docs/guides/decisions` fail with `read ECONNRESET` (5 attempts). Retry later, or paste the page, then compare against `tests/fixtures/*.json`.
- Shape assumed from LocalAI (`/v1/decisions`, "OpenAI-compatible contract"), LangChain (`langchain-openai>=1.7.0`) and TypeSafe (`/v1/systemone`). Request: `model`, `input`, `questions[]` (`type` predicate|choice|score, `name`, `instructions`, `choices[{value,description}]`, `levels[{label,description}]`). Response: `model`, `answers[]` (`name`, `type`, `probability` | `choice`+`confidence`+`probabilities` | `score`+`confidence`+`probabilities`, or `type: "refusal"`), `usage`.
- Known gaps: the exact element keys of `probabilities` are a guess (`{value, probability}`; score adds `label`). TypeSafe returns `probabilities` as an object keyed by option, OpenAI-compatible routes as an array. `raw_scores` is our own extension and is omitted when absent.
- Differences for a future adapter: TypeSafe uses `state` + `questions` as an object keyed by name, `noul` instead of `predicate`, `criteria` instead of `choices`/`levels`, and a flat `answers` object.
- Limits enforced locally: choice needs 2+ unique values; score needs 2 to 10 levels (Jev and LocalAI limit); question names unique.

## T3-T5: data and runner

- Template: 26 sources, 31 outputs (11 derived: concat, arith, format, lookup), 6 decoys. Level 5 names are legacy-style codes (`OBJ_INFO_ADDR_C`, `MVT_AMT_MAIN`), kept UPPER_SNAKE (or lower snake), never camelCase.
- Files: see `README.md` ("Where files go"). `data/` is tracked and is the single source of truth for inputs; each run writes only its own outputs under `results/<samples>_<timestamp>/`.
- Runner config is `config.toml` (stdlib `tomllib`) instead of YAML, to avoid a dependency.
- Throughput (Q8_0, M3): about 0.1 s per question regardless of state sharing (31 questions = 3.2 s). One case (about 26 input columns x 31 outputs + 31 kind questions) takes about 90 s of scoring plus load time. Default `n_per_level = 3` gives 21 cases, about 32 min per context condition, about 2 h for all four. Raise it in `config.toml` or with `--n`.
- Smoke result, 1 case at level 5, condition `both`: best input column is a true source for 31/31 outputs, transformation kind 31/31. Optimistic, since descriptions and samples are given; `name` is the hard condition.

## T7: context ablation and option-order stability, done

- Can Jev read row contents? Yes. `state` is free text or JSON up to 25,600 tokens, so any part of the table can go in. Today the `samples`/`both` conditions send 5 values per column (evenly spaced rows, `SAMPLE_K` in `src/core/prompts.py`). Sending whole rows (cross-column context) or the full column is a prompt change only; cost grows with state length (the state is computed once per column and reused across the 31 output questions), and a 2B model gets less reliable on long tables, so it is worth testing as an extra condition rather than assuming.
- Data per file: 15 to 30 rows, wider value pools, and 1 to 2 anomalies per file (zero, negative, x1000 outlier, empty string, `N/A`, sentinel date `1900-01-01`). Anomalies are listed in `ground_truth_<word>.csv` as kind `anomaly` (`column@rowN`). Expected outputs stay computed from the row as it is; where a rule cannot handle the value (0 divisor, unknown status code) the expected cell is blank (`safe_evaluate` returns None).
- Order stability smoke (Q8_0, condition `name`, 1 case each at levels 3 and 5, 3 orders): transform choice picks the same expression across orders for 10/11 and 9/11 outputs (mean accuracy over orders 0.88 and 0.82); kind choice is order-consistent for 25/31 and 28/31 (0.84 and 0.78). Picks are not stuck on slot 0 (25 to 34 percent). So option order does change answers sometimes; `choice` results should be averaged over orders.

## Decision: no column descriptions for the model

- The real template is a header row only (no rows, no descriptions), and real inputs are a header plus aligned rows. So the model gets: input column name + 5 sample values, and the output field name only. Default conditions are `name` and `samples` (`config.toml`). `desc` and `both` stay in `src/context.py` as an upper bound for later, but are not run unless requested with `--cond`.
- Input CSVs are checked aligned: every row has the same number of fields as the header (21 of 21 files).
- Sample run first (user request): levels 0, 3, 5, 6, one case each, before any larger run. The full 2 to 3 hour run was stopped and `results/` reset.

## Speed: why the demos feel fast, and what we changed

- Device: MacBook Air M3 (Mac15,12), 8-core CPU (4 performance, 4 efficiency), 8-core GPU, 16 GB unified memory, Metal 4, fanless. No thermal warning was recorded so far, but a fanless chip throttles on long sustained runs. Base M3 memory bandwidth is about 100 GB/s (from memory, not checked against Apple's spec page).
- Our workload is prefill bound (many short prompts, no generated tokens), so GPU compute matters more than memory bandwidth. Measured: about 0.1 s per yes/no question of roughly 60 tokens on Q8_0, all layers on Metal.
- Why demos look fast (third-party sources, vendor claims, not independent): the model reads logits at the last position in one forward pass, so there is no decode loop; prompts are short; the state is reused across questions; and one request can carry many questions. An `any2jev` benchmark (Qwen3-0.6B, RTX 2060 SUPER, fp32) reports 778 ms for JSON generation, 137 ms for label logits with one prefill per question, and 42 ms with all questions packed into one pass. ClassOne reports 52 ms local versus 330 ms for the Jev cloud API (self-reported). Packed scoring cost some accuracy in one project (84.3 versus 87.0 percent).
- Measured here (`make bench`, Q8_0, per input column, top-1 = best output field is a true target):
  - L5-000, context `name`, 26 columns: pairs 83.5 s, 10/26 correct; options 11.7 s, 12/26.
  - L6-000, context `samples`, 32 columns: pairs 159.3 s, 25/26; options 25.3 s, 23/26.
  - So `options` mode (one choice question per column, raw score read per output field) is about 7 times faster with similar top-1. Its probabilities are sigmoid(score / T) per field and are not calibrated; `pairs` stays available for calibrated numbers.
- Already in place: all layers on the GPU, flash attention auto, one long-lived scorer process, state reused across questions of one column.
- Not done yet, worth measuring: F16 versus Q8_0 versus Q4_K_M speed (F16 may prefill faster on Metal since it skips dequantisation; needs a 3.8 GB or 1.3 GB download), shorter question text, a two-stage prune (options mode, then yes/no only on the top candidates), kind questions only for outputs that matter, MLX, and running on a machine with a fan or a discrete GPU.

## Structure (vertical slices, low coupling)

- Dependency direction: `core` and `decisions` depend on nothing above them; `dataset` uses `core`; each slice (`mapping`, `transform`, `stability`, `convert`) uses `core`, `decisions` and, for prompts only, `transform.prompts`; `report` assembles slices; `experiment` runs the model; `cli` is the only place that knows the concrete Jev adapter and reads `config.toml`.
- Each slice owns its run function, metrics, figures and tables. `stability` imports `transform.prompts` (prompt builders only); nothing imports a slice's figures except `report`.
- `decisions` is the port: types shaped like OpenAI's Decisions API and a `Decider` protocol. Swapping to OpenAI or hosted Jev means adding one adapter in `src/decisions/` and one line in `cli.py`.
- Tests mirror `src/` and use fake deciders in `tests/fakes.py`. `make lint` runs pyflakes.

## First real run (4 samples, 2 contexts, options mode, Q8_0, M3, 6 minutes)

- Top-1 (best input column is a true source): name only 79.3 percent, name + values 86.2 percent. By sample (name only / name + values): identical 95.7 / 95.7, synonym 87.1 / 90.3, legacy codes 48.4 / 71.0, mixed 90.3 / 90.3. A string-similarity baseline gets 82.6 percent on identical names and about 19 percent on the others.
- Calibration: pooled ECE 0.007 at T = 0.828 and it worsens for higher T (0.21 at T = 3). The pooled number is flattered by the many negative pairs.
- Weak spots: output fields with no source column are not rejected at levels 3 to 6 (reject rate 0 percent at threshold 0.5), and 60 percent of decoy columns reached P >= 0.5 for some field. Pair recall at 0.5 is low (44 to 50 percent), so the options-mode scores sit below 0.5 for many true pairs. A per-field threshold fitted on the ground truth, or `--mode pairs`, is the next thing to try.
- Transformation kind given the true sources: 77 to 100 percent depending on sample and context.
- These numbers came from the layout before the restructure; `results/` was cleared at the user's request and is regenerated by `make sample`.
