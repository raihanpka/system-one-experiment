# score the frozen baseline again without ground truth help: runs the honest stage on the stored baseline mapping scores
# usage: uv run python scripts/rescore_baseline.py   (about 3 minutes, writes baselines/v1/honest.md and honest_result.json)

import re

from columnmap.convert.honest import convert_fields
from columnmap.convert.metrics import cell_match, field_causes, field_correct
from columnmap.core.baselines import RUNS, V1, baseline_rows
from columnmap.core.prompts import context_label
from columnmap.core.settings import load_settings, model_options
from columnmap.core.store import Store
from columnmap.core.tables import md_table, pct
from columnmap.core.template import load_template
from columnmap.dataset.generate import generate
from columnmap.decisions.jev import JevDecider
from columnmap.honest.run import run_case

CONDS = ("name", "samples", "table")
cases = {c["case_id"]: c for c in generate(load_template(), 1, 1234)}
out = Store(V1 / "honest_result.json")


def old_cells(run: str, cond: str) -> str:
    text = (V1 / run / "report" / "summary.md").read_text()
    m = re.search(rf"\*\*{re.escape(context_label(cond))}, options mode\*\*.*?Converted cells correct: ([\d.]+)%", text)
    return m.group(1) if m else "-"


with JevDecider(**model_options(load_settings())) as d:      # this baseline was scored by the local model
    for cond in CONDS:
        for row in baseline_rows("mapping", cond):
            if not out.has("honest", cond, row["case_id"]):
                out.add("honest", cond, run_case(d, cases[row["case_id"]], cond, row, "options"))

trs = []
for cond in CONDS:
    for cid, run in RUNS.items():
        rec = next(r for r in out.rows("honest", cond) if r["case_id"] == cid)
        case = cases[cid]
        _, conv = convert_fields(rec["fields"], case["rows"], [o["name"] for o in case["outputs"]])
        s, ok = cell_match(case, conv), field_correct(case, conv)
        c = field_causes(case, rec["fields"], ok)
        n = sum(c.values())
        trs.append([context_label(cond), cid, old_cells(run, cond), pct(s["correct_rate"]), f"{c['correct']} of {n}", c["wrong_columns"],
                    c["wrong_expression"], s["wrong_fill"]])
(V1 / "honest.md").write_text("\n".join([
    "# Baseline v1 Without Ground Truth Help", "",
    "Same mapping scores as baseline v1 (options mode, fixed rule 0.5). The old column used the true source columns to pick the "
    "transformation kind and the candidate expressions; the new column chooses columns from the scores, builds candidates with a program "
    "and lets the model rank them. Nothing from the ground truth reaches the model or the conversion.", "",
    md_table(["Jev sees", "sample", "old converted cells correct %", "honest cells correct %", "fields correct", "fields lost to wrong columns",
              "fields lost to wrong expression", "wrongly filled cells"], trs), "",
    "Fields lost to wrong columns: the stage chose a different set of columns than the true sources (including no column at all). "
    "Fields lost to wrong expression: the columns were right but the picked expression, or a code table that no catalog can guess, was not.", ""]))
print((V1 / "honest.md").read_text())
