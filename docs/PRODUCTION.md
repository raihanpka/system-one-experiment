# Production

This is the operating guide for using Column Mapping with System One Model on partner tables. It separates what is ready now,
what is deliberately a development harness, and what still needs infrastructure or labels.

## What can go to production now

| Piece | Status | Use |
|---|---|---|
| Core `Mapper` | Ready for batch | Deterministic yes/no output: a field is filled or empty. No graph, no pause. |
| Dataset `policy.json` | Ready with labels | Calibrated floors, model version, and calibration facts travel with the dataset. |
| `--audit` JSONL | Ready | One line per field: model, policy, `p_present`, `p_none`, method, columns, expression. No ground truth. |
| `OpenRouterDecider` retries | Ready | 429/5xx and transient network errors retry with exponential backoff; 4xx fails fast. |
| CLI table mode `--json` | Ready | Machine-readable run id, model, policy, fill/empty counts, elapsed time, audit path. |
| Graph harness | Development / guarded | Retry, degrade, repair, and review. Use `review="empty"` or `"accept"` for a binary product; `"ask"` only when a human is in the loop. |
| Bench and synthetic seeds | Evaluation only | Never use the synthetic ground truth in production. |

The current dev run (seed 11, `microsoft/microsoft-decision-1.1`) is **122 / 124 fields right**,
73/74 with a source, 49/50 without a source, and 73/74 fill precision. That is evidence, not a
production SLA. A production SLA is the error budget you agree to after calibrating on your own
labeled partner data.

## Why 100% is not a natural target

The last two dev errors are not engineering gaps:

- `gross_premium`: the model could not bind "base premium" to `CHG_AMT_RAW`. The column is a raw
  charge amount; the target word is "premium". A human with the template's hidden source map knows
  they are the same source; the model is not allowed to see that map.
- `transaction_fee_amount`: the model filled from `CHG_AMT_RAW` (14.9M) while the sample also has
  `MVT_AMT_SVC` (40k). The ground truth says no source, because `MVT_AMT_SVC` belongs to a different
  target source. A human would plausibly call the service amount a fee.

Both are **semantic ambiguity under the allowed inputs**, not a threshold that can be tuned. Forcing
100% would require one of:

1. the template's source ids/ground truth in the prompt (leakage; forbidden by this repo), or
2. a curated input data dictionary/ontology (a legitimate product input, but a different task), or
3. per-seed prompt tweaks (overfitting; fails on a new seed).

A binary system still answers: ambiguous means **no (empty)**, not a guess. That is the manager's
"no doubt": the output is decided and auditable, even when the evidence is incomplete. If the
business needs 100% coverage, the missing piece is a data dictionary, not a lower threshold.

## Calibration

Calibrate on a labeled sample of real partner fields, not on the synthetic dev seed.

1. Collect at least 100–300 fields with a known source/empty answer. For a 2% false-fill budget with
   a 95% Wilson upper bound, the no-source group should be around 150 fields or more; 50 fields
   cannot certify 2% even with zero false fills.
2. Stratify by field kind (copy, arithmetic, date, lookup), partner, locale, and amount/text.
3. Run `make calibrate RUN_DIR=results/<labeled run> BUDGET=0.02`. The command chooses the floors that
   maximize fields right under the budget, records the point estimate and the upper bound, and writes
   `policy.json`.
4. Review the policy with the business owner. The threshold is a risk decision, not a model setting.
5. Freeze the model, prompt book, and policy as one version. A change to any of them is a new
   calibration.

Useful metrics: Brier score and reliability/ECE for probability quality, false-fill and false-empty
rates for the decision, coverage (filled share), and precision/recall by slice. Accuracy alone hides
overconfidence.

## Recalibration

Recalibrate only on a trigger, not per batch:

- the model or model version changes;
- the prompt book or policy changes;
- a new partner, locale, currency, or file format appears;
- input drift: PSI above ~0.2 on key columns, or a KS test with a small p-value;
- audit finds the false-fill rate above budget, or fill precision drops by more than 2 points;
- a scheduled review (for example quarterly), even without a trigger.

Process:

1. Collect a fresh labeled sample from the new distribution (do not reuse the tuning seed).
2. Run `calibrate`; compare the new policy with the old one.
3. Run in shadow mode for 1–2 weeks: write both outputs, act on neither, compare.
4. Canary 5–10% of traffic; watch latency, fallback/degrade rate, fill rate, audit false fills.
5. Promote; keep the previous model/prompt/policy version for rollback.
6. Record `calibrated_at`, the run id, the budget, and the upper bound in `policy.json`.

If the point estimate meets the budget but the upper bound does not, the label set is too small.
Collect more labels; do not lower the floor to make the number look better.

## Operations

- **Binary output.** A field is filled (yes) or empty (no); there is no "maybe" in the converted file.
- **Audit.** Keep `--audit` for every production batch. It contains no ground truth and no secrets.
  Retain it per your data policy; it is what lets you explain a yes/no later.
- **Observability.** Track request success, p50/p95 latency, retry rate, degrade rate, fill rate,
  review rate (if the graph is used), and a sampled audit of false fills. Alert on a change in fill
  rate or degrade rate, not only on 5xx.
- **Limits.** Set a maximum rows/bytes and a request timeout at the job boundary. The model sees only
  a compact sample, but conversion and audit still load the file.
- **Data handling.** Partner values are sent to the configured OpenRouter model. Check the provider's
  retention/training terms, redact PII before sending when required, and keep the API key in the
  environment. Do not log raw state by default.
- **Prompt injection.** Input text is collapsed and capped before it enters a prompt, and answers are
  constrained to the offered choices. Treat partner text as data; never as instructions.
- **Rollback.** Keep the previous `policy.json`, prompt book, and model version. A policy change is a
  deployable artifact, not a config edit in place.

## Enterprise readiness checklist

- [x] CI runs format check, lint, tests, and build (`.github/workflows/ci.yml`).
- [x] Model is pinned to an exact version in `appsettings.json`.
- [x] Prompt book, policy, and dataset are versioned together in a knowledge folder.
- [x] Policy carries model, calibration run, seed, budget, point estimate, upper bound, and time.
- [x] Retries with backoff for 429/5xx/network; 4xx fails fast; timeout is bounded.
- [x] Per-field audit log with scores, method, columns, and expression.
- [x] Binary output; empty is a valid no; no "maybe" in the file.
- [ ] Monitoring/drift dashboard and alerting (needs your observability stack).
- [ ] PII redaction and data-retention policy (needs your data classification).
- [ ] Shadow/canary/rollback automation (process now; automate in your scheduler).
- [ ] Labeled production sample large enough to certify the error budget.

## References

- NIST AI Risk Management Framework: https://www.nist.gov/itl/ai-risk-management-framework
- ISO/IEC 42001 AI management systems: https://www.iso.org/standard/42001
- OpenRouter Decisions API: https://openrouter.ai/docs/guides/features/decisions
- LangGraph fault tolerance: https://docs.langchain.com/oss/python/langgraph/fault-tolerance
- Conformal risk control for structured LLM outputs: https://arxiv.org/html/2606.29054
