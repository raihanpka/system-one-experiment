# the converted file from the honest stage: each field runs the expression it picked on the columns it chose

from __future__ import annotations

from ..core.dsl import safe_evaluate


def convert_fields(fields: dict, rows: list[dict], order: list[str]) -> tuple[list[str], list[list]]:
    cols = [[safe_evaluate(fields[f]["best"], r) for r in rows] if fields[f]["best"] else [None] * len(rows) for f in order]
    return order, [list(r) for r in zip(*cols)]


# one row per field for decisions_*.csv; the truth_ columns come from the synthetic ground truth and exist only for evaluation
def decisions(fields: dict, case: dict, correct: dict) -> list[dict]:
    out = []
    for o in case["outputs"]:
        r = fields[o["name"]]
        out.append({"output": o["name"], "chosen_input": "|".join(r["cols"]), "confidence": round(r["conf"], 4), "method": r["method"],
                    "expression": r["best_text"] or "", "p_expr": "" if r["p_expr"] is None else round(r["p_expr"], 4),
                    "truth_available": o["available"], "truth_input": "|".join(o["sources"]), "field_correct": correct[o["name"]]})
    return out
