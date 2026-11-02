# Decision Mapper

Maps messy input columns onto a fixed output template with a decision model. The model scores each input column against each output field. Confident fields are filled, the rest go to review with three candidates.

Status: the local llama.cpp backend works. Hosted backends and device auto-detection are planned.

## Quick Start

```bash
make setup     # once: Python dependencies, model download, scorer build
make sample    # run samples a, b, c, d
make help      # all commands
```

## Configure

All settings live in `config.toml`.

- `[backend] type`: `local_gguf` (now), or `openrouter`, `hf_inference`, `typesafe` (planned).
- `[device] mode`: `auto` picks Metal on Apple Silicon, CUDA on NVIDIA, or CPU. You can force one.
- API keys stay out of the file. Config names an environment variable, for example `OPENROUTER_API_KEY`.

## Run

```bash
uv run columnmap                 # samples a, b, c, d
uv run columnmap --samples ac    # only a and c
uv run columnmap --all           # all five samples
uv run columnmap --report-only   # rebuild the newest run, no model
uv run columnmap --help          # all options
```

Samples: `a` identical names, `b` synonyms, `c` system codes, `d` mixed styles with extra columns, `e` Indonesian.

## Output

```text
results/<samples>_<timestamp>/
  result.json              raw model output
  <sample>/                converted file, decisions, review list, probabilities
  report/summary.md        tables and figures
```

## Layout

```text
src/columnmap/   core, backends, calibration, policy, dataset, report, experiment
tests/           mirrors src, uses fake backends (no model needed)
```

## Tests

```bash
make test
make lint
```
