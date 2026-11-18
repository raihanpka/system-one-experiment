# Expressions: a tiny AST (evaluate, render) and the catalog of candidate expressions for chosen columns.
# Deterministic and model-free; only column names and values are used, never ground truth.

from __future__ import annotations

import json
import re
from datetime import date
from itertools import permutations

_ARITH = {"add": "+", "sub": "-", "mul": "*", "div": "/"}
_PREC = {"add": 1, "sub": 1, "mul": 2, "div": 2}


def _day(x):
    return date.fromisoformat(str(x))


# arithmetic takes numbers only: a text value must not repeat ("abc" * 5) or join ("a" + "b").
# CSV cells arrive as text, so a numeric string is coerced; only non-numeric text is rejected.
def _num(x):
    if isinstance(x, bool):
        raise TypeError(f"not a number: {x!r}")
    if isinstance(x, (int, float)):
        return x
    if isinstance(x, str):
        text = x.strip()
        try:
            return int(text) if re.fullmatch(r"[+-]?\d+", text) else float(text)
        except ValueError:
            raise TypeError(f"not a number: {x!r}") from None
    raise TypeError(f"not a number: {x!r}")


def evaluate(node: dict, row: dict):
    if "src" in node:
        return row[node["src"]]
    if "lit" in node:
        return node["lit"]
    op, args = node.get("op"), [evaluate(a, row) for a in node.get("args", ())]
    if op == "concat":
        return node["sep"].join(str(a) for a in args)
    if op == "add":
        return _num(args[0]) + _num(args[1])
    if op == "sub":
        return _num(args[0]) - _num(args[1])
    if op == "mul":
        return _num(args[0]) * _num(args[1])
    if op == "div":
        return _num(args[0]) / _num(args[1])
    if op == "year":
        return _day(args[0]).year
    if op == "days_between":
        return (_day(args[1]) - _day(args[0])).days
    if op == "lookup":
        return node["map"][str(args[0])]
    raise ValueError(f"unknown op {op!r}")


# The input columns an expression reads, in first-use order
def src_names(node: dict) -> list[str]:
    if "src" in node:
        return [node["src"]]
    seen: list[str] = []
    for a in node.get("args", ()):
        for n in src_names(a):
            if n not in seen:
                seen.append(n)
    return seen


# Readable one-line form; `name_of(src_id)` gives the column name to show
def render(node: dict, name_of) -> str:
    if "src" in node:
        return name_of(node["src"])
    if "lit" in node:
        return json.dumps(node["lit"], ensure_ascii=False)
    op, args = node["op"], node.get("args", ())
    if op in _ARITH:

        def side(a, right):
            s = render(a, name_of)
            if a.get("op") in _PREC and (
                _PREC[a["op"]] < _PREC[op] or (right and _PREC[a["op"]] == _PREC[op])
            ):
                return f"({s})"
            return s

        return f"{side(args[0], False)} {_ARITH[op]} {side(args[1], True)}"
    if op == "concat":
        sep = json.dumps(node["sep"], ensure_ascii=False)
        return "concat(" + f", {sep}, ".join(render(a, name_of) for a in args) + ")"
    if op == "lookup":
        return f"lookup({render(args[0], name_of)}, {json.dumps(node['map'], ensure_ascii=False)})"
    return f"{op}(" + ", ".join(render(a, name_of) for a in args) + ")"


# evaluate(), but a value the rule cannot handle (0 divisor, unknown code, bad date) gives None
def safe_evaluate(node: dict, row: dict):
    try:
        return evaluate(node, row)
    except (ZeroDivisionError, KeyError, ValueError, TypeError):
        return None


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


# words in a column or field name that say which unit it holds
RATE_HINTS = (
    "rate",
    "pct",
    "percent",
    "percentage",
    "tax",
    "levy",
    "discount",
    "ratio",
    "share",
)
AMOUNT_HINTS = (
    "amount",
    "amt",
    "charge",
    "premium",
    "payable",
    "payout",
    "sum",
    "fee",
    "cost",
    "price",
    "balance",
    "value",
    "total",
    "limit",
    "lmt",
)
COUNT_HINTS = ("days", "count", "qty", "quantity", "age", "year")
IDENTIFIER_HINTS = (
    "id",
    "key",
    "code",
    "ref",
    "number",
    "no",
    "nomor",
    "kode",
    "uniq",
    "case",
    "zip",
    "postal",
    "postcode",
    "plz",
    "cd",
)
DATE_HINTS = (
    "date",
    "dt",
    "day",
    "birth",
    "incident",
    "start",
    "end",
    "expiry",
    "effective",
    "posting",
    "receipt",
    "from",
    "to",
)


def _tokens(text: str) -> list[str]:
    spaced = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", str(text))
    return [t for t in re.split(r"[^a-z0-9]+", spaced.lower()) if t]


# what a column measures, from its name first and its values second. "number" is a plain number that
# could still be an amount; "id" is a code that looks numeric (postal code, policy number).
def column_unit(name: str, values=()) -> str:
    kind = column_type(values)
    if kind == "date":
        return "date"
    toks = _tokens(name)
    if any(t in IDENTIFIER_HINTS for t in toks):
        return "id"
    if any(t in RATE_HINTS for t in toks):
        return "rate"
    if any(t in AMOUNT_HINTS for t in toks):
        return "amount"
    if any(t in COUNT_HINTS for t in toks):
        return "count"
    if kind == "text":
        return "text"
    nums = [float(str(v)) for v in values if _is_number(v)]
    if (
        nums
        and all(0.0 <= x <= 1.0 for x in nums)
        and any("." in str(v) for v in values)
    ):
        return "rate"
    return "number"


# the unit an output field should receive; only name and description are read, never a rule
def output_unit(out) -> str:
    if out.type == "date":
        return "date"
    toks = _tokens(f"{out.name} {out.description}")
    if any(t in RATE_HINTS for t in toks):
        return "rate"
    if any(t in AMOUNT_HINTS for t in toks):
        return "amount"
    if any(t in COUNT_HINTS for t in toks):
        return "count"
    return "number" if out.type == "number" else "text"


# a role may only take a column of a compatible unit. A plain number can still be an amount or a count.
def compatible(role_unit: str | None, col_unit: str) -> bool:
    if role_unit is None:
        return True
    return {
        "amount": col_unit in ("amount", "number"),
        "rate": col_unit == "rate",
        "date": col_unit == "date",
        "count": col_unit in ("count", "number"),
        "text": col_unit in ("text", "id"),
    }.get(role_unit, True)


# the unit an expression produces, so a rate cannot fill an amount and an amount cannot fill a rate
def result_unit(node: dict, units: dict) -> str:
    if "src" in node:
        return units[node["src"]]
    if "lit" in node:
        return "multiplier" if node["lit"] == 1 else "number"
    op, args = node.get("op"), node.get("args", ())
    if op in ("year", "days_between"):
        return "count"
    if op in ("concat", "lookup"):
        return "text"
    us = [result_unit(a, units) for a in args]
    if op in ("add", "sub"):
        if all(u == "amount" for u in us):
            return "amount"
        if "multiplier" in us and any(u == "rate" for u in us):
            return "multiplier"
        if all(u in ("count", "number") for u in us):
            return "count"
        return "number"
    if op == "mul":
        if "amount" in us and any(u in ("rate", "multiplier") for u in us):
            return "amount"
        return "number"
    if op == "div":
        return "rate" if all(u == "amount" for u in us) else "number"
    return "number"


def size(node: dict) -> int:
    return 1 + sum(size(a) for a in node.get("args", ()))


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


def candidates(
    cols, limit: int = LIMIT, out_type: str | None = None, tables=()
) -> list[dict]:
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
    lookups = [
        {"op": "lookup", "map": t, "args": [srcs[i]]}
        for i, (_, values) in enumerate(cols)
        for t in covering(values, tables)
    ]  # a code column that a reference table covers
    out = single + years + lookups + out
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
    unique.sort(key=lambda kc: (size(kc[1]), kc[0]))
    return [c for _, c in unique[:limit]]


# the reference tables whose codes include every non-empty value of a column
def covering(values, tables) -> list[dict]:
    vals = {str(v) for v in values if str(v).strip() != ""}
    return [t for t in tables if vals and vals <= set(t)]
