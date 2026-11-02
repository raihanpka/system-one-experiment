# a short headline per context and mode: the numbers a reader wants first

from __future__ import annotations

from ..convert.honest import convert_fields
from ..convert.metrics import cell_match, field_correct
from ..convert.tables import rules_for
from ..core.prompts import context_label as ctx
from ..core.scoring import FIXED
from ..core.tables import pct
from ..mapping.metrics import mapping_metrics
from ..mapping.reject import field_scores, reject_metrics

MODES = (("mapping", "options"), ("mapping_pairs", "pairs"))


def _honest(store, cases, cond, mode, only) -> str:
    recs = [r for r in store.rows("honest", cond, only) if r.get("mode", "options") == mode]
    if not recs:
        return ""
    cells = wrong = fields = good = 0
    for rec in recs:
        case = cases[rec["case_id"]]
        _, conv = convert_fields(rec["fields"], case["rows"], [o["name"] for o in case["outputs"]])
        s, ok = cell_match(case, conv), field_correct(case, conv)
        cells += s["expected_cells"]
        wrong += s["wrong_fill"]
        good += sum(ok[o["name"]] for o in case["outputs"] if o["available"])
        fields += sum(o["available"] for o in case["outputs"])
    correct = sum(cell_match(cases[r["case_id"]], convert_fields(r["fields"], cases[r["case_id"]]["rows"], [o["name"] for o in cases[r["case_id"]]["outputs"]])[1])["correct"] for r in recs)
    return f" Without ground truth help: {good} of {fields} fields correct, {pct(correct / cells if cells else None)}% of cells correct, {wrong} cells wrongly filled."


def at_a_glance(store, cases, conds, only=None, other=None) -> list[str]:
    out = ["## At a Glance", ""]
    for cond in conds:
        for section, mode in MODES:
            rows = store.rows(section, cond, only)
            if not rows:
                continue
            other_rows = other.rows(section, cond) if other else None
            rules = rules_for(rows, cases, other_rows)
            fitted = next((r for r in rules if r[0].startswith("score cutoff") and "other run" in r[0]), None)
            m, feats = mapping_metrics(rows, cases), field_scores(rows, cases)
            line = (f"- **{ctx(cond)}, {mode} mode**: top-1 {pct(m['top1'])}%, exact {pct(m['exact'])}%. "
                    f"Fields without a source left empty: {pct(reject_metrics(feats, FIXED)['rejected'])}% with the fixed rule")
            if fitted:
                line += f", {pct(reject_metrics(feats, fitted[1])['rejected'])}% with a cutoff fitted on another run"
            line += "." + _honest(store, cases, cond, mode, only)
            secs = [r["seconds"] for r in rows if "seconds" in r]
            if secs:
                line += f" {sum(secs) / len(secs):.0f} s per file."
            out.append(line)
    return out + [""] if len(out) > 2 else []
