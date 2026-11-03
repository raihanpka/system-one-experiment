from __future__ import annotations

from ..core.prompts import context_label as ctx
from ..core.scoring import FIXED, accept_score
from ..core.tables import md_table, pct
from ..mapping.reject import field_scores, fit_cuts, reject_metrics
from .honest import convert_fields
from .metrics import cell_match, field_correct


# one row per file: how much of the template the honest stage filled correctly, without any ground truth in its inputs
def honest_tables(store, cases, conds, only=None) -> list[str]:
    trs = []
    for cond in conds:
        for rec in store.rows("honest", cond, only):
            case = cases[rec["case_id"]]
            _, conv = convert_fields(
                rec["fields"], case["rows"], [o["name"] for o in case["outputs"]]
            )
            s, ok = cell_match(case, conv), field_correct(case, conv)
            with_src = [o["name"] for o in case["outputs"] if o["available"]]
            without = [o["name"] for o in case["outputs"] if not o["available"]]
            trs.append(
                [
                    ctx(cond),
                    rec["case_id"],
                    f"{sum(ok[f] for f in with_src)} of {len(with_src)}",
                    f"{sum(not ok[f] for f in without)} of {len(without)}",
                    s["expected_cells"],
                    pct(s["correct_rate"]),
                    s["wrong_fill"],
                ]
            )
    if not trs:
        return []
    return [
        "## Converted File Without Ground Truth Help",
        "",
        "Columns, transformation, and expression all come from the model's own choices. Fields correct = every cell matches the expected file. "
        "Wrongly filled = fields that have no source but were filled.",
        "",
        md_table(
            [
                "Jev sees",
                "sample",
                "fields with a source correct",
                "fields without a source wrongly filled",
                "cells expected",
                "cells correct %",
                "wrongly filled cells",
            ],
            trs,
        ),
        "",
    ]


# the rules a field can be kept by: fixed 0.5, and cutoffs fitted on ground truth (this run, or another run)
def rules_for(rows, cases, other_rows=None) -> list[tuple[str, object]]:
    rules = [("fixed, probability 0.5", FIXED)]
    for label, data in (("this run, optimistic", rows), ("other run", other_rows)):
        if not data:
            continue
        cuts = fit_cuts(data, cases)
        if cuts["d1"] is not None:
            rules.append(
                (
                    f"score cutoff {cuts['d1']:.2f}, fitted on {label}",
                    accept_score(cuts["d1"]),
                )
            )
    return rules


def rule_tables(
    store, cases, conds, only=None, other=None, section="mapping"
) -> list[str]:
    trs = []
    for cond in conds:
        rows = store.rows(section, cond, only)
        if not rows:
            continue
        other_rows = other.rows(section, cond) if other else None
        feats = field_scores(rows, cases)
        for label, accept in rules_for(rows, cases, other_rows):
            m = reject_metrics(feats, accept)
            trs.append(
                [
                    ctx(cond),
                    label,
                    pct(m["kept"]),
                    pct(m["rejected"]),
                    pct(m["balanced"]),
                    pct(m["top1_when_kept"]),
                    m["decoys_picked"],
                ]
            )
    if not trs:
        return []
    return [
        "## Rejecting Fields Without a Source",
        "",
        "A field is kept when its best input column passes the rule. Kept = fields with a source that are kept; "
        "Rejected = fields without a source that are left empty; Decoys = kept fields whose best column is a distractor. "
        "Cutoffs fitted on this run are optimistic (same data); cutoffs fitted on another run are a fair test.",
        "",
        md_table(
            [
                "Jev sees",
                "rule",
                "kept %",
                "rejected %",
                "balanced %",
                "top-1 when kept %",
                "distractors",
            ],
            trs,
        ),
        "",
    ]
