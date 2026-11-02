# Tiny transformation AST: evaluate and render. No eval(): only the ops below exist

from __future__ import annotations

import json
from datetime import date

_ARITH = {"add": "+", "sub": "-", "mul": "*", "div": "/"}
_PREC = {"add": 1, "sub": 1, "mul": 2, "div": 2}


def _day(x):
    return date.fromisoformat(str(x))


def evaluate(node: dict, row: dict):
    if "src" in node:
        return row[node["src"]]
    if "lit" in node:
        return node["lit"]
    op, args = node.get("op"), [evaluate(a, row) for a in node.get("args", ())]
    if op == "concat":
        return node["sep"].join(str(a) for a in args)
    if op == "add":
        return args[0] + args[1]
    if op == "sub":
        return args[0] - args[1]
    if op == "mul":
        return args[0] * args[1]
    if op == "div":
        return args[0] / args[1]
    if op == "year":
        return _day(args[0]).year
    if op == "days_between":
        return (_day(args[1]) - _day(args[0])).days
    if op == "lookup":
        return node["map"][str(args[0])]
    raise ValueError(f"unknown op {op!r}")


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
            if a.get("op") in _PREC and (_PREC[a["op"]] < _PREC[op] or (right and _PREC[a["op"]] == _PREC[op])):
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
