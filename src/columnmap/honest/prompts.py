# prompt text for ranking candidate expressions; every candidate is shown with what it computes on real rows

from __future__ import annotations

from ..core.dsl import render, safe_evaluate
from ..core.prompts import kind_state, samples
from ..decisions.types import Choice, Question


def _fmt(v) -> str:
    if v is None:
        return "error"
    return f"{v:.2f}" if isinstance(v, float) else str(v)


# what the expression gives on three evenly spaced rows
def examples(ast: dict, rows: list[dict], k: int = 3) -> list[str]:
    n = len(rows)
    idx = sorted({i * (n - 1) // (k - 1) for i in range(k)}) if n >= k else range(n)
    return [_fmt(safe_evaluate(ast, rows[i])) for i in idx]


def expression_text(ast: dict) -> str:
    return render(ast, lambda s: s)


def option_text(ast: dict, rows: list[dict]) -> str:
    return f"{expression_text(ast)} => " + "; ".join(examples(ast, rows))


def pool_state(names: list[str], rows: list[dict]) -> str:
    return kind_state([(n, None, samples(rows, n)) for n in names], "samples")


def pick_question(field: str, options: list[str]) -> Question:
    return Question("choice", f"Which expression produces the output field '{field}' from these input columns?", name="pick",
                    choices=tuple(Choice(o) for o in options))


def check_question(field: str, ast: dict, rows: list[dict], name: str) -> Question:
    return Question("predicate", f"The output field '{field}' is computed as: {expression_text(ast)} (for example {'; '.join(examples(ast, rows))}).",
                    name=name)
