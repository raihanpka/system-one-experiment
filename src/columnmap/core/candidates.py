# expression candidates for the chosen input columns; only names and values are used, never ground truth

from __future__ import annotations

from datetime import date
from itertools import permutations

from .dsl import render

LIMIT = 24
SEPS = (" ", ", ", "-", "/", "")


def _is_number(v) -> bool:
    try:
        float(str(v))
        return True
    except ValueError:
        return False


def _is_date(v) -> bool:
    s = str(v)
    if "-" not in s:
        return False
    try:
        date.fromisoformat(s)
        return True
    except ValueError:
        return False


# a few odd cells (empty, N/A, a zero) do not change what the column is
def column_type(values) -> str:
    vals = [v for v in values if str(v).strip() != ""]
    if not vals:
        return "text"
    if sum(_is_date(v) for v in vals) >= 0.7 * len(vals):
        return "date"
    if sum(_is_number(v) for v in vals) >= 0.7 * len(vals):
        return "number"
    return "text"


def _size(node: dict) -> int:
    return 1 + sum(_size(a) for a in node.get("args", ()))


def _concat(sep: str, *args) -> dict:
    return {"op": "concat", "sep": sep, "args": list(args)}


def _op(op: str, *args) -> dict:
    return {"op": op, "args": list(args)}


def _pair(a: dict, b: dict, ta: str, tb: str) -> list[dict]:
    out = []
    if ta == tb == "number":
        one = {"lit": 1}
        out += [
            _op("add", a, b),
            _op("sub", a, b),
            _op("sub", b, a),
            _op("mul", a, b),
            _op("div", a, b),
            _op("div", b, a),
            _op("mul", a, _op("add", one, b)),
            _op("mul", a, _op("sub", one, b)),
            _op("mul", b, _op("add", one, a)),
            _op("mul", b, _op("sub", one, a)),
        ]
    elif ta == tb == "date":
        out += [_op("days_between", a, b), _op("days_between", b, a)]
    if "date" not in (
        ta,
        tb,
    ):  # identifiers such as postal codes look numeric but join like text
        out += [_concat(sep, x, y) for x, y in ((a, b), (b, a)) for sep in SEPS]
    return out


def _triple(cs: list[dict], ts: list[str]) -> list[dict]:
    out = []
    if "date" not in ts:
        for order in (cs, cs[::-1]):
            out += [_concat(sep, *order) for sep in (" ", ", ", "-")]
        out += [_concat(" ", _concat(", ", x, y), z) for x, y, z in permutations(cs)]
    if all(t == "number" for t in ts):
        out.append(_op("add", _op("add", cs[0], cs[1]), cs[2]))
    return out


# simplest first, at most `limit`; fields that need a rule outside this catalog (for example a code table) get none
# the type a candidate produces: a copy keeps its column's type, year and arithmetic give a number, concat and lookup give text
def result_type(node: dict, types: dict) -> str:
    if "src" in node:
        return types[node["src"]]
    if node.get("op") in ("year", "days_between", "add", "sub", "mul", "div"):
        return "number"
    return "string"


def candidates(cols, limit: int = LIMIT, out_type: str | None = None) -> list[dict]:
    srcs = [{"src": name} for name, _ in cols]
    types = [column_type(v) for _, v in cols]
    if len(cols) == 1:
        out = []
    elif len(cols) == 2:
        out = _pair(srcs[0], srcs[1], types[0], types[1])
    elif len(cols) == 3:
        out = _triple(srcs, types)
    else:
        out = []
    # every chosen column also gives its plain copy (and year for dates), so a passthrough field is never missing from the list
    single = [srcs[i] for i in range(len(cols))]
    years = [_op("year", srcs[i]) for i in range(len(cols)) if types[i] == "date"]
    out = single + years + out
    seen, unique = set(), []
    for c in out:
        key = render(c, lambda s: s)
        if key not in seen:
            seen.add(key)
            unique.append((key, c))
    if out_type in (
        "date",
        "number",
    ):  # a date or number field must receive a date or number
        types_by_name = {name: t for (name, _), t in zip(cols, types)}
        unique = [
            (k, c) for k, c in unique if result_type(c, types_by_name) == out_type
        ]
    unique.sort(key=lambda kc: (_size(kc[1]), kc[0]))
    return [c for _, c in unique[:limit]]
