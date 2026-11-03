# Architecture

## Style

- System: layered pipeline inside one Python package, because one command runs every stage and the stages share one set of helpers.
- Structure: horizontal layers by pipeline stage. Shared packages `core` and `decisions` sit below the stages.
- Coupling rule: `core` and `decisions` import no stage. A stage imports `core`, `decisions`, and earlier stages only, in the order `dataset`, `mapping`, `honest`, `convert`, `report`.

## Shape

One package, `src/columnmap`. `main.py`, `calibrate.py`, and `experiment.py` are entry points that run the stages in order. Backends (`backends/`) sit behind the interface in `decisions/`, so the stages never import a model.

## Modules

Each module owns one thing. Import the module, there is no single public file.

| Module | Owns | May depend on | Public interface |
|---|---|---|---|
| `core/` | settings, paths, scoring, calibration, selective cutoff, validation, prompts, template | none of the stages | the module itself |
| `decisions/` | decider interface, question and answer types, backend factory | `core` (settings), `backends` (imported inside the factory function only) | `base.py`, `factory.py` |
| `backends/` | OpenRouter adapter and local Jev GGUF adapter | `core`, `decisions` | `openrouter.py`, `jev_gguf.py` |
| `dataset/` | template and sample CSV files (`make data`) | `core` | `export.py`, `samples.py` |
| `mapping/` | score of each input column for each output field | `core`, `decisions`, `dataset` | `run.py`, `metrics.py` |
| `honest/` | the honest run that picks fields from the scores | `core`, `decisions` | `run.py` |
| `convert/` | fills the output file with the calibrated cutoff | `core`, `decisions`, `mapping`, `honest` | `auto.py`, `metrics.py` |
| `report/` | files, figures, and tables from all stages | every stage above it | `build.py` |

## Current violations

None. Checked with:

```
grep -rnE "from \.\.?(dataset|mapping|honest|convert|report)" src/columnmap/core src/columnmap/decisions src/columnmap/backends
grep -rnE "from \.\.(honest|convert|report)" src/columnmap/mapping src/columnmap/honest src/columnmap/dataset
grep -rnE "from \.\.report" src/columnmap/convert
```

All three print nothing. Known cycle, not a stage rule: `decisions/factory.py` imports `backends` inside its functions, and `backends` imports `decisions.types` at load time. It loads without error because the import is lazy.

## Boundary check

None yet. The rule holds today by grep, not by a tool. See `docs/DOTAGENT-SETUP.md`, open items.

## Flow

The dataset stage writes the template and the sample inputs. The mapping stage asks the decider to score each input column against each output field and keeps the raw scores. The honest stage picks the fields from those scores. The convert stage fills the output file: a field is filled automatically only when its score passes the calibrated cutoff and its values pass the template type, otherwise it goes to review. The report stage writes the files, figures, and tables from all of these.
