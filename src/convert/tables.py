from __future__ import annotations

from ..core.tables import md_table, pct
from .apply import convert_case
from .metrics import cell_match


def convert_tables(store, cases, conds, only=None) -> list[str]:
    trs = []
    for cond in conds:
        tr = {r["case_id"]: r for r in store.rows("transform", cond, only)}
        for m in store.rows("mapping", cond, only):
            case = cases[m["case_id"]]
            s = cell_match(case, convert_case(m, tr.get(m["case_id"]), case)[1])
            trs.append([cond, m["case_id"], s["expected_cells"], pct(s["correct_rate"]), s["wrong_fill"]])
    if not trs:
        return []
    return ["## Converted file against the expected file", "",
            "Cells that should hold a value and match, and cells filled where nothing was expected. "
            "Threshold 0.5; a derived field uses the expression Jev picked among the candidates.", "",
            md_table(["context", "sample", "cells expected", "correct %", "wrongly filled"], trs), ""]
