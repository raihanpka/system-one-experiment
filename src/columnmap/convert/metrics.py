from __future__ import annotations

import math

from ..core.scoring import FIXED, safe_div
from .apply import convert_case


def _same(a, b) -> bool:
    if isinstance(a, float) or isinstance(b, float):
        try:
            return math.isclose(float(a), float(b), rel_tol=1e-9)
        except (TypeError, ValueError):
            return False
    return a == b


# Converted file against the expected file: cells right where a value is expected, cells wrongly filled
def cell_match(case: dict, rows: list[list]) -> dict:
    expected_cells = correct = wrong_fill = 0
    for j, o in enumerate(case["outputs"]):
        for i, row in enumerate(rows):
            want = o["expected"][i] if o["available"] else None
            got = row[j]
            if want is None:
                wrong_fill += got is not None
            else:
                expected_cells += 1
                correct += got is not None and _same(got, want)
    return {"expected_cells": expected_cells, "correct": correct, "wrong_fill": wrong_fill,
            "correct_rate": safe_div(correct, expected_cells)}


# converted cells over several files for one acceptance rule
def file_cells(map_rows, tr_rows, cases, accept=FIXED) -> dict:
    tr = {r["case_id"]: r for r in tr_rows}
    total = dict(expected_cells=0, correct=0, wrong_fill=0)
    for m in map_rows:
        case = cases[m["case_id"]]
        s = cell_match(case, convert_case(m, tr.get(m["case_id"]), case, accept)[1])
        for k in total:
            total[k] += s[k]
    total["correct_rate"] = safe_div(total["correct"], total["expected_cells"])
    return total
