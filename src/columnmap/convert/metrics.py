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
