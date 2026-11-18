# Configuration

Three files and the environment, in this order (later wins):

| Layer | Holds | Committed? |
|---|---|---|
| `appsettings.json` | the model endpoint, the default seed, the split seeds | yes |
| `appsettings.local.json` | your personal overrides | no |
| `COLUMN_MAPPING__Section__Key` | CI/container overrides; secrets | no |
| `knowledge/policy.json` | fill floors for one dataset | yes, with that dataset |

The API key never lives in any of these. It is read from the environment variable named by
`OpenRouter:ApiKeyEnv` (default `OPENROUTER_API_KEY`, usually loaded from `.env`).

## What `appsettings.json` should contain

Keep it small. It is the machine-wide default, not a per-dataset tuning file:

```json
{
  "OpenRouter": {
    "Model": "microsoft/microsoft-decision-1.1",
    "BaseUrl": "https://openrouter.ai/api/alpha/decisions",
    "ApiKeyEnv": "OPENROUTER_API_KEY"
  },
  "Run": {
    "Seed": 3407
  },
  "Splits": {
    "Dev": [11],
    "Holdout": [12, 13],
    "Test": [21, 22]
  }
}
```

`Run:FillScore`, `Run:CopyFillScore`, and `Run:NoneCeiling` are **optional** overrides. They are not
in the committed file because the floor belongs to a dataset policy, not to the machine. When a
dataset has no `policy.json`, `Fill()` supplies `0.75 / 0.4 / 0.5`.

## What `knowledge/policy.json` should contain

One policy per dataset, next to `outputs.json` and `prompts.json`:

```json
{
  "model": "microsoft/microsoft-decision-1.1",
  "fill": {"derived": 0.75, "copy": 0.4, "none_ceiling": 0.5},
  "calibration": {
    "run": "all-seed11_...",
    "seed": 11,
    "budget": 0.02,
    "false_fill": 1,
    "no_source": 50,
    "false_fill_rate": 0.02,
    "upper_false_fill": 0.105,
    "budget_met_point": true,
    "budget_met_upper": false
  }
}
```

Write it with `make calibrate RUN_DIR=results/<labeled run> BUDGET=0.02`; do not hand-edit it per
batch. A dataset policy always wins over `appsettings.json`. `--fill-score` and
`--copy-fill-score` override both for a one-off experiment.

## Seeds

- `Run:Seed` (3407) is the default sample seed used by `make run` and `make data` when no
  `--seed` or `--split` is given.
- `Splits:Dev` is `[11]` for tuning.
- `Splits:Holdout` is `[12, 13]`; already semi-used, so not a clean claim.
- `Splits:Test` is `[21, 22]`; already used, needs `--final`.
- A fresh claim needs a seed nobody has looked at; 31 and 32 only with permission.

## Examples

```bash
# map a partner table; the dataset policy in my_knowledge/ wins
uv run column-mapping --table in.xlsx --knowledge my_knowledge/ --output converted.csv

# force a different floor for one experiment
uv run column-mapping --table in.csv --knowledge my_knowledge/ --fill-score 0.8 --copy-fill-score 0.5

# rebuild a report with other floors, no model call
uv run column-mapping --report-only --fill-score 0.8 --copy-fill-score 0.5

# replace the appsettings value from the environment
COLUMN_MAPPING__Run__Seed=11 uv run column-mapping --samples a
```
