"""Apply Jev's decisions to an input file: the converted (template-shaped) rows plus a log of each decision.

Per output field: take the input column with the highest probability; below THRESHOLD the field stays empty.
If Jev named the transformation `passthrough` the column is copied; otherwise the expression Jev picked among
the candidates is executed on the rows. The candidates come from the ground truth, so this measures whether
Jev's choices produce the right file, not whether it can invent a formula.
"""
from __future__ import annotations

from ..core.dsl import render, safe_evaluate
from ..core.scoring import T0, pair_p

THRESHOLD = 0.5


def convert_case(map_row: dict, tr_row: dict | None, case: dict, threshold: float = THRESHOLD):
    """Returns (header, rows, decisions); rows hold python values, None = empty."""
    cols = [k["name"] for k in case["columns"]]
    name_of = {k["src"]: k["name"] for k in case["columns"] if k["src"]}
    by_id = [{s: row[n] for s, n in name_of.items()} for row in case["rows"]]
    transform = (tr_row or {}).get("transform", {})
    columns, decisions = [], []
    for o in case["outputs"]:
        sc = sorted(((pair_p(map_row["matrix"][c][o["name"]], T0), c) for c in cols), reverse=True)
        (p1, c1), (p2, c2) = sc[0], sc[1]
        kind = map_row["kind"].get(o["name"], {}).get("choice", "passthrough")
        expr = ""
        if p1 < threshold:
            method, values = "unmapped", [None] * len(case["rows"])
        elif kind != "passthrough" and o["name"] in transform:
            cand = o["candidates"][transform[o["name"]]["choice_idx"]]
            method, expr = "expression", render(cand, name_of.get)
            values = [safe_evaluate(cand, r) for r in by_id]
        else:
            method, values = "copy", [row[c1] for row in case["rows"]]
        columns.append(values)
        correct = c1 in o["sources"] if o["available"] else p1 < threshold
        decisions.append({"output": o["name"], "available": o["available"], "true_input": "|".join(o["sources"]),
                          "predicted_input": c1, "p": round(p1, 4), "runner_up": c2, "p_runner_up": round(p2, 4),
                          "correct": correct, "kind_predicted": kind, "method": method, "expression": expr})
    rows = [list(r) for r in zip(*columns)]
    return [o["name"] for o in case["outputs"]], rows, decisions
