# All prompt text, in one editable registry. A knowledge folder can override any name in prompts.json;
# MLflow can serve the same names as versioned prompts. The model sees column names, the extracted input
# schema, output field names and descriptions, and candidate expressions with what they compute. Never
# the ground truth.

from __future__ import annotations

from .decider import Choice, Question
from .expressions import render, safe_evaluate
from .knowledge import Output
from .registry import FileRegistry

SAMPLE_K = 5
NONE = "none: the field cannot be built from the input columns"

KIND_KEYS = ("copy", "join", "arithmetic", "date_difference", "year", "lookup")

DEFAULT_TEMPLATES = {
    "mapping": "Which output fields does this input column provide data for?",
    "kind_question": "What kind of value is this output field? Decide from its name and description only.",
    "kind_copy": "the value is copied unchanged from one input column",
    "kind_join": "text from several input columns is joined into one value",
    "kind_arithmetic": "the value is calculated from number columns with + - * /",
    "kind_date_difference": "the value is the number of days between two date columns",
    "kind_year": "the value is the year taken from one date column",
    "kind_lookup": "a code in one input column is translated into a label with a code table",
    "kind_state": "Output field: {{field}}\nDescription: {{description}}\nType: {{type}}",
    "input_header": "Input column: {{name}}",
    "input_type": "Type: {{kind}}",
    "input_unit": "Unit: {{unit}}",
    "input_null_rate": "Null rate: {{null_rate}}",
    "input_samples": "Sample values: {{values}}",
    "file_header": "Input columns:",
    "file_row": "- {{name}} (type={{kind}}, unit={{unit}})\n  Sample values: {{values}}",
    "pool_row": "- {{name}}\n  Sample values: {{values}}",
    "all_columns": "All input columns: {{columns}}",
    "set": "Which input columns does the output field '{{field}}' need? Choose the set that has exactly the inputs its description names or implies, no more and no fewer. Choose '{{none}}' if they are not all among the input columns.",
    "expression": "Which expression gives the value of the output field '{{field}}' ({{description}})?",
    "bind": "Output field: {{field}} ({{description}}). Which input column holds the {{role}}? Choose the column that holds this component, even when its wording differs. Choose none only if no input column holds it; a merely similar column that holds a different thing is not a match.",
    "presence": "Does the input contain every component that the output field '{{field}}' is built from? Field description: {{description}}. Answer yes only if each component named or implied by the description is present as an input column. A merely similar column does not replace a missing one.",
    "presence_true": "Every component the field is defined from is present in the input.",
    "presence_false": "At least one component is missing; a similar column does not count.",
}
DEFAULT = FileRegistry(DEFAULT_TEMPLATES)


# k values of one column from evenly spaced rows (so the odd cell is sometimes visible, not always)
def samples(rows, name, k=SAMPLE_K):
    n = len(rows)
    return (
        [rows[i * (n - 1) // (k - 1)][name] for i in range(k)]
        if n >= k
        else [r[name] for r in rows]
    )


# the model always sees the column name and sample values; the extracted type and unit are added when known
def input_state(
    name: str,
    values=(),
    kind: str | None = None,
    unit: str | None = None,
    null_rate: float | None = None,
    registry=DEFAULT,
) -> str:
    lines = [registry.get("input_header").format(name=name)]
    if kind:
        lines.append(registry.get("input_type").format(kind=kind))
    if unit:
        lines.append(registry.get("input_unit").format(unit=unit))
    if null_rate is not None:
        lines.append(registry.get("input_null_rate").format(null_rate=null_rate))
    if values:
        lines.append(
            registry.get("input_samples").format(values=", ".join(map(str, values)))
        )
    return "\n".join(lines)


# one choice question per input column: which output fields does it provide data for
def mapping_question(outputs: tuple[Output, ...], registry=DEFAULT) -> Question:
    return Question(
        "choice",
        registry.get("mapping").template,
        name="field",
        choices=tuple(Choice(o.name, o.description) for o in outputs),
    )


def pool_state(names: list[str], rows: list[dict], registry=DEFAULT) -> str:
    lines = [registry.get("file_header").template]
    for name in names:
        lines.append(
            registry.get("pool_row").format(
                name=name, values=", ".join(map(str, samples(rows, name)))
            )
        )
    return "\n".join(lines)


# every input column once, with the schema facts we extracted, so one request can answer every field
def file_state(source, registry=DEFAULT) -> str:
    columns = getattr(source, "columns", None)
    if (
        columns is None
    ):  # rows: keep the old behavior for callers that do not build a Schema
        return pool_state(list(source[0]), source, registry)
    lines = [registry.get("file_header").template]
    for c in columns:
        lines.append(
            registry.get("file_row").format(
                name=c.name,
                kind=c.type,
                unit=c.unit,
                values=", ".join(c.samples),
            )
        )
    return "\n".join(lines)


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


def kind_question(registry=DEFAULT) -> Question:
    return Question(
        "choice",
        registry.get("kind_question").template,
        name="kind",
        choices=tuple(Choice(k, registry.get(f"kind_{k}").template) for k in KIND_KEYS),
    )


def kind_state(out: Output, registry=DEFAULT) -> str:
    return registry.get("kind_state").format(
        field=out.name, description=out.description, type=out.type
    )


# the state for the fallback column-set and expression questions: the field, every input column name,
# and the best columns with values
def field_state(
    out: Output,
    all_columns: list[str],
    names: list[str],
    rows: list[dict],
    registry=DEFAULT,
) -> str:
    return (
        kind_state(out, registry)
        + "\n"
        + registry.get("all_columns").format(columns=", ".join(all_columns))
        + "\n"
        + pool_state(names, rows, registry)
    )


def set_label(columns) -> str:
    return " & ".join(columns)


def set_question(out: Output, labels: list[str], registry=DEFAULT) -> Question:
    return Question(
        "choice",
        registry.get("set").format(
            field=out.name, description=out.description, none=NONE
        ),
        name="set",
        choices=tuple(Choice(x) for x in [*labels, NONE]),
    )


def expression_question(out: Output, options: list[str], registry=DEFAULT) -> Question:
    return Question(
        "choice",
        registry.get("expression").format(field=out.name, description=out.description),
        name="expr",
        choices=tuple(Choice(o) for o in options),
    )


# one role of a field, bound to exactly one input column (or none). The state is the whole file, so
# every field can ask its roles in the same request.
def bind_question(
    out: Output, role, columns: list[str], name: str, registry=DEFAULT, hint=None
) -> Question:
    text = registry.get("bind").format(
        field=out.name, description=out.description, role=role.phrase
    )
    if hint:
        text = f"{text} {hint}"
    return Question(
        "choice",
        text,
        name=name,
        choices=tuple(Choice(c) for c in columns) + (Choice(NONE),),
    )


# the fallback for a description the role parser cannot read: ask whether the source exists at all,
# separately from which columns build it. A similar column does not count.
def presence_question(
    out: Output, name: str = "presence", registry=DEFAULT
) -> Question:
    return Question(
        "predicate",
        registry.get("presence").format(field=out.name, description=out.description),
        name=name,
        true=registry.get("presence_true").template,
        false=registry.get("presence_false").template,
    )
