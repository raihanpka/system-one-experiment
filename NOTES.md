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
- Files on disk (per the user's request: light, one file per converter run): `uv run python -m src.export_csv [--n 3]` writes `data/template_header.csv` (empty target template), `data/template_columns.csv` (name, type, kind, description, rule) and, per case, a folder `data/cases/<case_id>/` with `input.csv` (the messy file), `expected_output.csv` (the converted, upload-ready file) and `mapping.csv` (ground truth: output, kind, available, input_columns, expression; decoy inputs listed with kind `decoy`). 21 cases = 63 small files, about 250 KB, git-ignored. There is no JSONL ground-truth file anymore: runners rebuild cases in memory from the seed (`config.toml`, `seed = 1234`; ids are stable across `--n`). An `.xlsx` view needs `openpyxl` (not added); Excel opens the CSVs directly.
- Runner config is `config.toml` (stdlib `tomllib`) instead of YAML, to avoid a dependency.
- Throughput (Q8_0, M3): about 0.1 s per question regardless of state sharing (31 questions = 3.2 s). One case (about 26 input columns x 31 outputs + 31 kind questions) takes about 90 s of scoring plus load time. Default `n_per_level = 3` gives 21 cases, about 32 min per context condition, about 2 h for all four. Raise it in `config.toml` or with `--n`.
- Smoke result, 1 case at level 5, condition `both`: best input column is a true source for 31/31 outputs, transformation kind 31/31. Optimistic, since descriptions and samples are given; `name` is the hard condition.
