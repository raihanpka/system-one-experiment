# the counts of a run against the ground truth, every result as x / y, and the fields that are still wrong

from __future__ import annotations

from ..mapper import Fill, convert, is_filled
from .metrics import cell_match, field_correct
from .tables import md_table

HEADER = [
    "sample",
    "column top-1 (fields with a source)",
    "fields right (with a source)",
    "has a source, left empty",
    "no source, left empty",
    "filled right (of filled)",
    "all fields right",
    "cells right",
]


# the top-scored column of a field; a tie goes to the name that sorts first
def top_column(matrix: dict, field: str) -> str:
    return min(matrix, key=lambda c: (-matrix[c][field], c))


def sample_counts(case: dict, matrix: dict, rec: dict, fill: Fill) -> dict:
    outs = [o["name"] for o in case["outputs"]]
    _, conv = convert(rec["fields"], case["rows"], outs, fill)
    ok = field_correct(case, conv)
    cells = cell_match(case, conv)
    avail = [o for o in case["outputs"] if o["available"]]
    nosrc = [o for o in case["outputs"] if not o["available"]]
    filled = {f: is_filled(rec["fields"][f], fill) for f in outs}
    errors = []
    for o in case["outputs"]:
        f = o["name"]
        if ok[f]:
            continue
        kind = (
            "no source, but filled"
            if not o["available"]
            else "has a source, left empty"
            if not filled[f]
            else "filled, wrong"
        )
        errors.append(
            {
                "field": f,
                "kind": kind,
                "expression": rec["fields"][f]["best_text"],
                "score": rec["fields"][f]["score"],
                "truth": o["sources"],
            }
        )
    return {
        "top1": sum(top_column(matrix, o["name"]) in set(o["sources"]) for o in avail),
        "with_source": len(avail),
        "right_with_source": sum(ok[o["name"]] for o in avail),
        "left_empty": sum(not filled[o["name"]] for o in avail),
        "no_source": len(nosrc),
        "no_source_empty": sum(ok[o["name"]] for o in nosrc),
        "filled": sum(filled.values()),
        "filled_right": sum(ok[f] and filled[f] for f in outs),
        "right": sum(ok.values()),
        "fields": len(outs),
        "cells_right": cells["correct"],
        "cells": cells["expected_cells"],
        "errors": errors,
    }


def _row(label: str, c: dict) -> list[str]:
    return [
        label,
        f"{c['top1']} / {c['with_source']}",
        f"{c['right_with_source']} / {c['with_source']}",
        f"{c['left_empty']} / {c['with_source']}",
        f"{c['no_source_empty']} / {c['no_source']}",
        f"{c['filled_right']} / {c['filled']}",
        f"{c['right']} / {c['fields']}",
        f"{c['cells_right']} / {c['cells']}",
    ]


def total(rows: list[dict]) -> dict:
    keys = [k for k in rows[0] if k != "errors"]
    return {k: sum(r[k] for r in rows) for k in keys}


def table(per_sample: list[tuple[str, dict]]) -> str:
    rows = [_row(label, c) for label, c in per_sample]
    rows.append(_row("**total**", total([c for _, c in per_sample])))
    return md_table(HEADER, rows)


def error_lines(per_sample: list[tuple[str, dict]]) -> list[str]:
    out = []
    for label, c in per_sample:
        for e in c["errors"]:
            score = "" if e["score"] is None else f", score {e['score']:.2f}"
            out.append(
                f"- {label} / `{e['field']}`: {e['kind']}; chose `{e['expression']}`{score}; "
                f"true sources {e['truth'] or 'none'}"
            )
    return out
