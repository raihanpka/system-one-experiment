# a short headline per context and mode: the numbers a reader wants first

from __future__ import annotations

from ..convert.metrics import file_cells
from ..convert.tables import rules_for
from ..core.prompts import context_label as ctx
from ..core.scoring import FIXED
from ..core.tables import pct
from ..mapping.metrics import mapping_metrics
from ..mapping.reject import field_scores, reject_metrics

MODES = (("mapping", "options"), ("mapping_pairs", "pairs"))


def at_a_glance(store, cases, conds, only=None, other=None) -> list[str]:
    out = ["## At a Glance", ""]
    for cond in conds:
        for section, mode in MODES:
            rows = store.rows(section, cond, only)
            if not rows:
                continue
            other_rows = other.rows(section, cond) if other else None
            rules = rules_for(rows, cases, other_rows)
            fitted = next((r for r in rules if r[0].startswith("score cutoff") and "other run" in r[0]), None) or \
                next((r for r in rules if r[0].startswith("score cutoff")), None)
            m, feats = mapping_metrics(rows, cases), field_scores(rows, cases)
            tr = store.rows("transform", cond, only)
            fixed_rej, fixed_cells = reject_metrics(feats, FIXED)["rejected"], file_cells(rows, tr, cases, FIXED)
            secs = [r["seconds"] for r in rows if "seconds" in r]
            line = (f"- **{ctx(cond)}, {mode} mode**: top-1 {pct(m['top1'])}%, exact {pct(m['exact'])}%. "
                    f"Fields without a source left empty: {pct(fixed_rej)}% with the fixed rule")
            if fitted:
                rej, cells = reject_metrics(feats, fitted[1])["rejected"], file_cells(rows, tr, cases, fitted[1])
                where = "fitted on another run" if "other run" in fitted[0] else "fitted on this run, optimistic"
                line += (f", {pct(rej)}% with a cutoff ({where}). Converted cells correct: {pct(fixed_cells['correct_rate'])}% "
                         f"fixed, {pct(cells['correct_rate'])}% with the cutoff; wrongly filled cells: {fixed_cells['wrong_fill']} "
                         f"and {cells['wrong_fill']}.")
            else:
                line += f". Converted cells correct: {pct(fixed_cells['correct_rate'])}% ({fixed_cells['wrong_fill']} wrongly filled)."
            if secs:
                line += f" {sum(secs) / len(secs):.0f} s per file."
            out.append(line)
    return out + [""] if len(out) > 2 else []
