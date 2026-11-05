# All prompt text. The model sees input column names and sample values, output field names and descriptions,
# and candidate expressions with what they compute on real rows. Never the ground truth.

from __future__ import annotations

from .decider import Choice, Question
from .expressions import render, safe_evaluate
from .knowledge import Output

CONTEXTS = ("name", "samples")
SAMPLE_K = 5
CONTEXT_LABEL = {"name": "name only", "samples": "name + values"}


def context_label(context: str) -> str:
    return CONTEXT_LABEL.get(context, context)


# k values of one column from evenly spaced rows (so the odd cell is sometimes visible, not always)
def samples(rows, name, k=SAMPLE_K):
    n = len(rows)
    return (
        [rows[i * (n - 1) // (k - 1)][name] for i in range(k)]
        if n >= k
        else [r[name] for r in rows]
    )


def input_state(name: str, context: str, values=()) -> str:
    lines = [f"Input column: {name}"]
    if context == "samples" and values:
        lines.append("Sample values: " + ", ".join(map(str, values)))
    return "\n".join(lines)


# one choice question per input column: which output fields does it provide data for
def mapping_question(outputs: tuple[Output, ...]) -> Question:
    return Question(
        "choice",
        "Which output fields does this input column provide data for?",
        name="field",
        choices=tuple(Choice(o.name, o.description) for o in outputs),
    )


def pool_state(names: list[str], rows: list[dict]) -> str:
    return "Input columns:\n" + "\n".join(
        f"- {n}\n  Sample values: " + ", ".join(map(str, samples(rows, n)))
        for n in names
    )


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


def pick_question(field: str, options: list[str]) -> Question:
    return Question(
        "choice",
        f"Which expression produces the output field '{field}' from these input columns?",
        name="pick",
        choices=tuple(Choice(o) for o in options),
    )


# the description of the field is part of the question; the check asks for the complete value, no extra or missing column
def check_question(
    field: str, description: str, ast: dict, rows: list[dict], name: str
) -> Question:
    return Question(
        "predicate",
        f"The output field '{field}' means: {description}. It is computed as: {expression_text(ast)} "
        f"(for example {'; '.join(examples(ast, rows))}). Is this the complete and correct value of the field, "
        "built from exactly the columns that are needed, with no extra column and no missing column?",
        name=name,
    )
