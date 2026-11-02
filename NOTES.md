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
