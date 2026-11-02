# prompt text per context: name only, or name plus sample values (descriptions, whole table, and the other contexts were rejected)

from __future__ import annotations

from ..decisions.types import Question

CONDS = ("name", "samples")
SAMPLE_K = 5
CONTEXT_LABEL = {"name": "name only", "samples": "name + values"}


# What the model sees about each input column, in words (file names keep the short key)
def context_label(cond: str) -> str:
    return CONTEXT_LABEL.get(cond, cond)


# k values of one column from evenly spaced rows (so the odd cell is sometimes visible, not always)
def samples(rows, name, k=SAMPLE_K):
    n = len(rows)
    return [rows[i * (n - 1) // (k - 1)][name] for i in range(k)] if n >= k else [r[name] for r in rows]


def _col_text(name, cond, samples):
    s = f"{name}"
    if cond == "samples" and samples:
        s += "\n  Sample values: " + ", ".join(map(str, samples))
    return s


def input_state(col: dict, cond: str, samples=()) -> str:
    lines = [f"Input column: {col['name']}"]
    if cond == "samples" and samples:
        lines.append("Sample values: " + ", ".join(map(str, samples)))
    return "\n".join(lines)


# output side: the field name only; the model picks the input columns that feed it
def mapping_question(name: str) -> Question:
    return Question("predicate", f"This input column provides the data for the output field '{name}'.", name=name)


def kind_state(cols: list[tuple[str, list]], cond: str) -> str:
    return "Input columns:\n" + "\n".join("- " + _col_text(n, cond, s) for n, s in cols)
