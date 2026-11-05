# All prompt text. The model sees input column names and sample values, output field names and descriptions,
# and candidate expressions with what they compute on real rows. Never the ground truth.

from __future__ import annotations

from .decider import Choice, Question
from .expressions import column_type, render, safe_evaluate
from .knowledge import Output

SAMPLE_K = 5


# k values of one column from evenly spaced rows (so the odd cell is sometimes visible, not always)
def samples(rows, name, k=SAMPLE_K):
    n = len(rows)
    return (
        [rows[i * (n - 1) // (k - 1)][name] for i in range(k)]
        if n >= k
        else [r[name] for r in rows]
    )


# the model always sees the column name and sample values; the name alone proved too weak
def input_state(name: str, values=()) -> str:
    lines = [f"Input column: {name}"]
    if values:
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


def _shape(v) -> str:
    return "".join(
        "9" if c.isdigit() else "A" if c.isupper() else "a" if c.isalpha() else c
        for c in str(v)
    )


# what all the values of a column look like, computed from every row (type, range, distinct values, format)
def profile(values) -> str:
    vals = [v for v in values if str(v).strip() != ""]
    if not vals:
        return "all empty"
    distinct = sorted({str(v) for v in vals})
    parts = [f"{len(vals)} of {len(values)} filled, {len(distinct)} distinct"]
    kind = column_type(vals)
    if kind == "date":
        parts.append(f"dates {distinct[0]} to {distinct[-1]}")
    elif kind == "number":
        nums = []
        for v in vals:
            try:
                nums.append(float(str(v)))
            except ValueError:
                pass
        lo, hi = min(nums), max(nums)
        whole = all(x == int(x) for x in nums)
        parts.append(
            f"{'whole numbers' if whole else 'decimals'} {lo:g} to {hi:g}"
            + (", all between 0 and 1" if 0 <= lo and hi <= 1 else "")
        )
    else:
        shapes = {_shape(v) for v in vals}
        parts.append(
            f"text, length {min(len(str(v)) for v in vals)} to {max(len(str(v)) for v in vals)}"
            + (f", format {next(iter(shapes))}" if len(shapes) == 1 else "")
        )
    if len(distinct) <= 6:
        parts.append("values: " + ", ".join(distinct))
    return "; ".join(parts)


def pool_state(names: list[str], rows: list[dict]) -> str:
    return "Input columns:\n" + "\n".join(
        f"- {n}\n  Sample values: {', '.join(map(str, samples(rows, n)))}\n  Profile: {profile([r[n] for r in rows])}"
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


# what kind of value an output field is, decided from its name and description alone
KINDS = {
    "copy": "the value is copied unchanged from one input column",
    "join": "text from several input columns is joined into one value",
    "arithmetic": "the value is calculated from number columns with + - * /",
    "date_difference": "the value is the number of days between two date columns",
    "year": "the value is the year taken from one date column",
    "lookup": "a code in one input column is translated into a label with a code table",
}
NONE = "none: the field cannot be built from the input columns"


def kind_question() -> Question:
    return Question(
        "choice",
        "What kind of value is this output field? Decide from its name and description only.",
        name="kind",
        choices=tuple(Choice(k, v) for k, v in KINDS.items()),
    )


def kind_state(out: Output) -> str:
    return f"Output field: {out.name}\nDescription: {out.description}\nType: {out.type}"


# the state for the column-set and expression questions: the field, every input column name, and the best columns with values
def field_state(
    out: Output, all_columns: list[str], names: list[str], rows: list[dict]
) -> str:
    return (
        kind_state(out)
        + f"\nAll input columns: {', '.join(all_columns)}\n"
        + pool_state(names, rows)
    )


def set_label(columns) -> str:
    return " & ".join(columns)


def set_question(out: Output, labels: list[str]) -> Question:
    return Question(
        "choice",
        f"Which input columns does the output field '{out.name}' need? Choose the set that has exactly the inputs its "
        f"description names or implies, no more and no fewer. Choose '{NONE}' if they are not all among the input columns.",
        name="set",
        choices=tuple(Choice(x) for x in [*labels, NONE]),
    )


def expression_question(out: Output, options: list[str]) -> Question:
    return Question(
        "choice",
        f"Which expression gives the value of the output field '{out.name}' ({out.description})?",
        name="expr",
        choices=tuple(Choice(o) for o in options),
    )
