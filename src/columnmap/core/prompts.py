# prompt text per context: name only, plus description, plus sample values, both, or the whole table

from __future__ import annotations

import csv
import io

from ..decisions.types import Choice, Question
from .candidates import column_type
from .template import KIND_DESC

CONDS = ("name", "desc", "samples", "both", "table", "profile")
SAMPLE_K = 5
CONTEXT_LABEL = {"name": "name only", "samples": "name + values", "desc": "name + description",
                 "both": "name + description + values", "table": "name + whole table", "profile": "name + values + profile"}


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
    return cond in ("samples", "both", "profile")


def _col_text(name, cond, desc, samples):
    s = f"{name}"
    if _desc(cond) and desc:
        s += f"\n  Description: {desc}"
    if _samples(cond) and samples:
        s += "\n  Sample values: " + ", ".join(map(str, samples))
    return s


def input_state(col: dict, cond: str, desc: str | None = None, samples=(), profile: str | None = None) -> str:
    lines = [f"Input column: {col['name']}"]
    if _desc(cond) and desc:
        lines.append(f"Description: {desc}")
    if cond == "profile" and profile:
        lines.append(profile)
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


# the table is the same text for every column, so the scorer computes it once per file and reuses it
def table_state(case: dict) -> str:
    names = [c["name"] for c in case["columns"]]
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(names)
    w.writerows([[r[n] for n in names] for r in case["rows"]])
    return "Input table (CSV):\n" + buf.getvalue().rstrip("\n")


def table_options_question(col: str, outs: list[str]) -> Question:
    return Question("choice", f"Which output fields does the input column '{col}' provide data for?", name="field",
                    choices=tuple(Choice(o) for o in outs))


def table_pair_question(col: str, out: str) -> Question:
    return Question("predicate", f"The input column '{col}' provides the data for the output field '{out}'.", name=out)


# kind and transformation questions describe the columns the usual way when the context is the whole table
def cond_base(cond: str) -> str:
    return "samples" if cond == "table" else cond


def _num(v):
    try:
        return float(str(v))
    except ValueError:
        return None


# k distinct values spread over the sorted values (numbers by size, text by spelling), so unusual cells and formats show up
def diverse_values(values, k: int = SAMPLE_K) -> list:
    distinct = list(dict.fromkeys(v for v in values if str(v).strip() != ""))
    if len(distinct) <= k:
        return sorted(distinct, key=lambda v: (_num(v) is None, _num(v) if _num(v) is not None else 0, str(v)))
    nums = [_num(v) for v in distinct]
    ordered = sorted(distinct, key=lambda v: _num(v)) if all(n is not None for n in nums) else sorted(distinct, key=str)
    return [ordered[i * (len(ordered) - 1) // (k - 1)] for i in range(k)]


def column_profile(values) -> dict:
    full = [v for v in values if str(v).strip() != ""]
    kind = column_type(values)
    out = {"type": kind, "n": len(values), "empty": len(values) - len(full), "distinct": len(set(map(str, full)))}
    if kind == "number":
        nums = [n for n in map(_num, full) if n is not None]
        out["lo"], out["hi"] = (int(min(nums)) if float(min(nums)).is_integer() else min(nums)), (int(max(nums)) if float(max(nums)).is_integer() else max(nums))
    elif kind == "date":
        dates = sorted(str(v) for v in full)
        out["lo"], out["hi"] = dates[0], dates[-1]
    elif full:
        out["lo"], out["hi"] = min(len(str(v)) for v in full), max(len(str(v)) for v in full)
    return out


# one line a person could write: what kind of values the column holds and how varied it is
def profile_text(values) -> str:
    p = column_profile(values)
    shape = {"number": f"range {p.get('lo')} to {p.get('hi')}", "date": f"range {p.get('lo')} to {p.get('hi')}"}.get(p["type"], f"{p.get('lo', 0)} to {p.get('hi', 0)} characters")
    return f"Type: {p['type']}, {shape}, {p['distinct']} distinct of {p['n']} values" + (f", {p['empty']} empty" if p["empty"] else "")
