# Prompt text per context condition: name only | +description | +sample values | both

from __future__ import annotations

from ..decisions.types import Choice, Question
from .template import KIND_DESC

CONDS = ("name", "desc", "samples", "both")
SAMPLE_K = 5
CONTEXT_LABEL = {"name": "name only", "samples": "name + values", "desc": "name + description",
                 "both": "name + description + values"}


# What the model sees about each input column, in words (file names keep the short key)
def context_label(cond: str) -> str:
    return CONTEXT_LABEL.get(cond, cond)


# k values of one column from evenly spaced rows (so the odd cell is sometimes visible, not always)
def samples(rows, name, k=SAMPLE_K):
    n = len(rows)
    return [rows[i * (n - 1) // (k - 1)][name] for i in range(k)] if n >= k else [r[name] for r in rows]
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
