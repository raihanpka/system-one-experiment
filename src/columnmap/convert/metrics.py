from __future__ import annotations

import math

from ..core.scoring import safe_div


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


# per field: all cells equal the expected cells (a field without a source is correct when it stays empty)
def field_correct(case: dict, rows: list[list]) -> dict:
    out = {}
    for j, o in enumerate(case["outputs"]):
        col = [r[j] for r in rows]
        if not o["available"]:
            out[o["name"]] = all(v is None for v in col)
        else:
            out[o["name"]] = all(v is not None and _same(v, w) if w is not None else v is None for v, w in zip(col, o["expected"]))
    return out
