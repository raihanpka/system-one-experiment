"""Prompt text per context condition: name only | +description | +sample values | both."""
from __future__ import annotations

from .decisions.types import Choice, Question

CONDS = ("name", "desc", "samples", "both")
SAMPLE_K = 5


def samples(rows, name, k=SAMPLE_K):
    """k values of one column from evenly spaced rows (so the odd cell is sometimes visible, not always)."""
    n = len(rows)
    return [rows[i * (n - 1) // (k - 1)][name] for i in range(k)] if n >= k else [r[name] for r in rows]
KIND_LABEL = {"pass": "passthrough", "concat": "concat", "arith": "arithmetic", "format": "format", "lookup": "lookup"}
KIND_DESC = {
    "passthrough": "the value is copied unchanged from one input column",
    "concat": "text from several input columns is joined together",
    "arithmetic": "the value is calculated from input columns with + - * / or a difference between dates",
    "format": "the value is a reformatted part of one input column, such as the year of a date",
    "lookup": "a code in an input column is translated into a label",
}


def _desc(cond):
    return cond in ("desc", "both")


def _samples(cond):
    return cond in ("samples", "both")


def _col_text(name, cond, desc, samples):
    s = f"{name}"
    if _desc(cond) and desc:
        s += f"\n  Description: {desc}"
    if _samples(cond) and samples:
        s += "\n  Sample values: " + ", ".join(map(str, samples))
    return s


def input_state(col: dict, cond: str, desc: str | None = None, samples=()) -> str:
    lines = [f"Input column: {col['name']}"]
    if _desc(cond) and desc:
        lines.append(f"Description: {desc}")
    if _samples(cond) and samples:
        lines.append("Sample values: " + ", ".join(map(str, samples)))
    return "\n".join(lines)


def _label(name, desc, cond):
    return f"'{name}'" + (f" (meaning: {desc})" if _desc(cond) and desc else "")


def mapping_question(name: str, desc: str | None, cond: str) -> Question:
    return Question("predicate", f"This input column provides the data for the output field {_label(name, desc, cond)}.", name=name)


def kind_state(cols: list[tuple[str, str | None, list]], cond: str) -> str:
    return "Input columns:\n" + "\n".join("- " + _col_text(n, cond, d, s) for n, d, s in cols)


def kind_question(name: str, desc: str | None, cond: str) -> Question:
    return Question("choice", f"How is the output field {_label(name, desc, cond)} produced from these input columns?",
                    name=name, choices=tuple(Choice(k, v) for k, v in KIND_DESC.items()))
