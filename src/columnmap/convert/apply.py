# apply Jev's decisions to an input file: the converted rows plus a log of each decision

from __future__ import annotations

import statistics

from ..core.dsl import render, safe_evaluate
from ..core.scoring import FIXED, T0, pair_p


# per output field: the best input column is kept if `accept(best score, lift)` holds; below that the field stays empty
# a derived field runs the expression Jev picked among the candidates, which come from the ground truth
def convert_case(map_row: dict, tr_row: dict | None, case: dict, accept=FIXED):
    cols = [k["name"] for k in case["columns"]]
    name_of = {k["src"]: k["name"] for k in case["columns"] if k["src"]}
    by_id = [{s: row[n] for s, n in name_of.items()} for row in case["rows"]]
    transform = (tr_row or {}).get("transform", {})
    columns, decisions = [], []
    for o in case["outputs"]:
        sc = sorted(((map_row["matrix"][c][o["name"]]["d"], c) for c in cols), reverse=True)
        (d1, c1), (d2, c2) = sc[0], sc[1]
        lift = d1 - statistics.median(d for d, _ in sc)
        keep = accept(d1, lift)
        kind = map_row["kind"].get(o["name"], {}).get("choice", "passthrough")
        expr = ""
        if not keep:
            method, values = "unmapped", [None] * len(case["rows"])
        elif kind != "passthrough" and o["name"] in transform:
            cand = o["candidates"][transform[o["name"]]["choice_idx"]]
            method, expr = "expression", render(cand, name_of.get)
            values = [safe_evaluate(cand, r) for r in by_id]
        else:
            method, values = "copy", [row[c1] for row in case["rows"]]
        columns.append(values)
        correct = c1 in o["sources"] if o["available"] else not keep
        decisions.append({"output": o["name"], "available": o["available"], "true_input": "|".join(o["sources"]),
                          "chosen_input": c1, "p": round(pair_p({"d": d1}, T0), 4), "runner_up": c2,
                          "p_runner_up": round(pair_p({"d": d2}, T0), 4), "correct": correct, "kind_chosen": kind,
                          "method": method, "expression": expr})
    rows = [list(r) for r in zip(*columns)]
    return [o["name"] for o in case["outputs"]], rows, decisions
