# Read a field description as the shape of the value it wants: the operator and the named operands.
# Only the description is read, never the rule, the kind, or a source id. A description this module
# cannot read returns None, and the caller keeps the old column-set question as a fallback.

from __future__ import annotations

import re
from dataclasses import dataclass

from .expressions import (
    AMOUNT_HINTS,
    COUNT_HINTS,
    DATE_HINTS,
    RATE_HINTS,
    _tokens,
    output_unit,
)
from .knowledge import Output


@dataclass(frozen=True)
class Role:
    phrase: str
    unit: str | None = None


@dataclass(frozen=True)
class Shape:
    op: str
    roles: tuple[Role, ...]
    sep: str | None = None

    def build(self, cols: list[dict]) -> dict:
        if self.op == "src":
            return cols[0]
        if self.op == "year":
            return {"op": "year", "args": [cols[0]]}
        if self.op == "lookup":
            return {"op": "lookup", "args": [cols[0]]}
        if self.op == "mul_plus":
            return {
                "op": "mul",
                "args": [cols[0], {"op": "add", "args": [{"lit": 1}, cols[1]]}],
            }
        if self.op == "mul_minus":
            return {
                "op": "mul",
                "args": [cols[0], {"op": "sub", "args": [{"lit": 1}, cols[1]]}],
            }
        if self.op in ("add", "sub", "div", "days_between"):
            return {"op": self.op, "args": [cols[0], cols[1]]}
        if self.op == "concat":
            return {"op": "concat", "sep": self.sep or " ", "args": list(cols)}
        if self.op == "address":
            inner = {"op": "concat", "sep": ", ", "args": [cols[0], cols[1]]}
            return {"op": "concat", "sep": " ", "args": [inner, cols[2]]}
        raise ValueError(f"unknown shape {self.op!r}")


def _role(phrase: str, out_unit: str) -> Role:
    phrase = phrase.strip(" .,:;")
    toks = _tokens(phrase)
    if any(t in RATE_HINTS for t in toks):
        unit = "rate"
    elif any(t in AMOUNT_HINTS for t in toks):
        unit = "amount"
    elif any(t in DATE_HINTS for t in toks):
        unit = "date"
    elif any(t in COUNT_HINTS for t in toks):
        unit = "count"
    else:
        unit = {
            "amount": "amount",
            "rate": "amount",
            "date": "date",
            "count": "count",
            "text": "text",
        }.get(out_unit)
    return Role(phrase, unit)


def _roles(phrases: list[str], out_unit: str) -> tuple[Role, ...]:
    return tuple(_role(p, out_unit) for p in phrases if p.strip())


def _match(pattern: str, text: str):
    m = re.fullmatch(pattern, text.strip(), flags=re.IGNORECASE)
    return tuple(g.strip() for g in m.groups()) if m else None


# the shape of a field, from its description; None means "ask the old column-set question"
def shape(out: Output, kind: str) -> Shape | None:
    text = " ".join(out.description.replace("\n", " ").split())
    lower = text.lower()
    unit = output_unit(out)

    if kind == "copy":
        return Shape("src", (Role(text or out.name.replace("_", " "), unit),))
    if kind == "year":
        base = lower.removeprefix("year of ").strip()
        return Shape("year", (Role(f"date of {base}", "date"),))
    if kind == "lookup":
        return Shape("lookup", (Role(f"code for {lower}", "text"),))

    # the formula after a colon, when the description first gives a summary
    tail = lower.split(":", 1)[1].strip() if ":" in lower else lower
    tail = tail.removesuffix(".").strip()

    if kind == "join":
        if (
            "given name then family name" in lower
            or "first name then last name" in lower
        ):
            return Shape(
                "concat",
                _roles(["given name", "family name"], unit),
                sep=" ",
            )
        if "address" in lower and all(
            w in lower for w in ("city", "province", "postal")
        ):
            return Shape("address", _roles(["city", "province", "postal code"], unit))
        return None

    if kind == "date_difference":
        got = _match(
            r"(?:number of )?days (?:between|from) (.+?) (?:and|to) (.+?)", tail
        )
        return Shape("days_between", _roles(list(got), unit)) if got else None

    if kind == "arithmetic":
        got = _match(r"(.+?) times \(1 \+ (.+?)\)", tail)
        if got:
            return Shape("mul_plus", _roles(list(got), unit))
        got = _match(r"(.+?) times \(1 - (.+?)\)", tail)
        if got:
            return Shape("mul_minus", _roles(list(got), unit))
        got = _match(r"(.+?) divided by (.+?)", tail)
        if got:
            return Shape("div", _roles(list(got), unit))
        got = _match(r"(.+?) minus (.+?)", tail)
        if got:
            return Shape("sub", _roles(list(got), unit))
        got = _match(r"(.+?) plus (.+?)", tail)
        if got:
            return Shape("add", _roles(list(got), unit))
        return None

    return None
