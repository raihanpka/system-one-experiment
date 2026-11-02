#!/bin/sh
# Build jev-score-v2 against the Homebrew llama.cpp, the single source of llama.cpp for this project.
# Rebuilds only when the brew versions change. After `brew upgrade llama.cpp`, run `make setup` again.
set -eu
ROOT=$(cd "$(dirname "$0")/.." && pwd)
SRC="$ROOT/vendor/jev-v3-2b-gguf"
command -v brew >/dev/null || { echo "Homebrew is required (https://brew.sh)"; exit 1; }
for f in llama.cpp ggml nlohmann-json; do
  brew list --versions "$f" >/dev/null 2>&1 || brew install "$f"
done
LC=$(brew --prefix llama.cpp); GG=$(brew --prefix ggml); NJ=$(brew --prefix nlohmann-json)
VER=$(brew list --versions llama.cpp ggml | tr '\n' ' ')
OUT="$SRC/build/jev-score-v2"; STAMP="$SRC/build/brew-versions.txt"
if [ -x "$OUT" ] && [ "$(cat "$STAMP" 2>/dev/null)" = "$VER" ]; then
  echo "scorer up to date: $VER"; exit 0
fi
mkdir -p "$SRC/build"
c++ -std=c++17 -O2 -Wall -Wno-unused-function -I"$LC/include" -I"$GG/include" -I"$NJ/include" \
  "$SRC/jev_score_v2.cpp" -o "$OUT" -L"$LC/lib" -L"$GG/lib" -lllama -lggml -lggml-base \
  -Wl,-rpath,"$LC/lib" -Wl,-rpath,"$GG/lib"
echo "$VER" > "$STAMP"
echo "built $OUT against: $VER"
