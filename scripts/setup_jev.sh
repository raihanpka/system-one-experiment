#!/bin/sh
# set up the Jev v3 runtime under vendor/ (idempotent, run from the repo root); needs uv and Homebrew
set -eu
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"

uv sync

# Model + runtime scripts. Download the Q8_0 file by name; --include is ignored when filenames are given.
uvx --from huggingface_hub hf download chaoliangUNSW/Jev-Style-2B-Decision-v3-GGUF \
  Jev-Style-2B-Decision-v3-Q8_0.gguf jev_style_decision_gguf.py jev_score_v2.cpp build_jev_score.sh \
  readout_config.json release_config.json manifest.json requirements.txt README.md LICENSE NOTICE \
  tokenizer/tokenizer.json tokenizer/tokenizer_config.json \
  --local-dir vendor/jev-v3-2b-gguf

# The scorer is compiled against the Homebrew llama.cpp (scripts/build_scorer.sh); no llama.cpp source is kept.
sh scripts/build_scorer.sh

# Smoke test: the model card example, expect answer ask_user with p ~ 0.86.
(cd vendor/jev-v3-2b-gguf && "$ROOT/.venv/bin/python" jev_style_decision_gguf.py --quant Q8_0 \
  --state "The user said: 'Write a reply to Bob with the Q3 report attached, but let me check it before it goes out.' The draft is written and the report is attached." \
  --question "What should the assistant do next?" \
  --options '{"send": "press Send now", "ask_user": "ask the user to review the draft", "discard": "delete the draft"}' \
  | grep -E '"answer"|"ask_user"')
