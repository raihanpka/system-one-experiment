#!/bin/sh
# Reproduce the Jev v3 GGUF runtime under vendor/. Idempotent. Run from the repo root.
# Needs: uv. Python comes from uv (brew Python is broken here, see NOTES.md). cmake comes from the uv dev group.
set -eu
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
PIN=441df11f65ea0b6d0c72965aaf70c8241070ddcb

uv sync
PATH="$ROOT/.venv/bin:$PATH"

# Model + runtime scripts. Download the Q8_0 file by name; --include is ignored when filenames are given.
uvx --from huggingface_hub hf download chaoliangUNSW/Jev-Style-2B-Decision-v3-GGUF \
  Jev-Style-2B-Decision-v3-Q8_0.gguf jev_style_decision_gguf.py jev_score_v2.cpp build_jev_score.sh \
  readout_config.json release_config.json manifest.json requirements.txt README.md LICENSE NOTICE \
  tokenizer/tokenizer.json tokenizer/tokenizer_config.json \
  --local-dir vendor/jev-v3-2b-gguf

# llama.cpp at the pinned commit, only to build the scorer.
if [ ! -d vendor/llama.cpp/.git ]; then
  git init -q vendor/llama.cpp
  git -C vendor/llama.cpp fetch -q --depth 1 https://github.com/ggml-org/llama.cpp "$PIN"
  git -C vendor/llama.cpp checkout -q FETCH_HEAD
fi

[ -x vendor/jev-v3-2b-gguf/build/jev-score-v2 ] || (cd vendor/jev-v3-2b-gguf && sh build_jev_score.sh ../llama.cpp)

# Smoke test: the model card example, expect answer ask_user with p ~ 0.86.
(cd vendor/jev-v3-2b-gguf && "$ROOT/.venv/bin/python" jev_style_decision_gguf.py --quant Q8_0 \
  --state "The user said: 'Write a reply to Bob with the Q3 report attached, but let me check it before it goes out.' The draft is written and the report is attached." \
  --question "What should the assistant do next?" \
  --options '{"send": "press Send now", "ask_user": "ask the user to review the draft", "discard": "delete the draft"}' \
  | grep -E '"answer"|"ask_user"')
