from __future__ import annotations

from ..core.prompts import context_label as ctx
from ..core.scoring import FIXED, accept_lift, accept_score
from ..core.tables import md_table, pct
from ..mapping.reject import field_scores, fit_cuts, reject_metrics
from .apply import convert_case
from .metrics import cell_match, file_cells


def convert_tables(store, cases, conds, only=None, section="mapping") -> list[str]:
    trs = []
    for cond in conds:
        tr = {r["case_id"]: r for r in store.rows("transform", cond, only)}
        for m in store.rows(section, cond, only):
            case = cases[m["case_id"]]
            s = cell_match(case, convert_case(m, tr.get(m["case_id"]), case)[1])
            trs.append([ctx(cond), m["case_id"], s["expected_cells"], pct(s["correct_rate"]), s["wrong_fill"]])
    if not trs:
        return []
    return ["## Converted File Against the Expected File", "",
            "Cells that should hold a value and match, and cells filled where nothing was expected (rule: probability 0.5). "
            "A derived field uses the expression Jev picked among the candidates.", "",
            md_table(["Jev sees", "sample", "cells expected", "correct %", "wrongly filled"], trs), ""]


# the rules a field can be kept by: fixed 0.5, and cutoffs fitted on ground truth (this run, or another run)
def rules_for(rows, cases, other_rows=None) -> list[tuple[str, object]]:
    rules = [("fixed, probability 0.5", FIXED)]
    for label, data in (("this run, optimistic", rows), ("other run", other_rows)):
        if not data:
            continue
        cuts = fit_cuts(data, cases)
        if cuts["d1"] is not None:
            rules.append((f"score cutoff {cuts['d1']:.2f}, fitted on {label}", accept_score(cuts["d1"])))
        if cuts["lift"] is not None:
            rules.append((f"lift cutoff {cuts['lift']:.2f}, fitted on {label}", accept_lift(cuts["lift"])))
    return rules


def rule_tables(store, cases, conds, only=None, other=None, section="mapping") -> list[str]:
    trs = []
    for cond in conds:
        rows = store.rows(section, cond, only)
        if not rows:
            continue
        other_rows = other.rows(section, cond) if other else None
        tr_rows = store.rows("transform", cond, only)
        feats = field_scores(rows, cases)
        for label, accept in rules_for(rows, cases, other_rows):
            m, c = reject_metrics(feats, accept), file_cells(rows, tr_rows, cases, accept)
            trs.append([ctx(cond), label, pct(m["kept"]), pct(m["rejected"]), pct(m["balanced"]), pct(m["top1_when_kept"]),
                        m["decoys_picked"], pct(c["correct_rate"]), c["wrong_fill"]])
    if not trs:
        return []
    return ["## Rejecting Fields Without a Source", "",
            "A field is kept when its best input column passes the rule. Kept = fields with a source that are kept; "
            "Rejected = fields without a source that are left empty; Decoys = kept fields whose best column is a decoy. "
            "Cutoffs fitted on this run are optimistic (same data); cutoffs fitted on another run are a fair test.", "",
            md_table(["Jev sees", "rule", "kept %", "rejected %", "balanced %", "top-1 when kept %", "decoys", "converted cells correct %",
                      "wrongly filled"], trs), ""]
