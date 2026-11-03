# Validate data/template.json. Run: uv run python -m columnmap.core.template

from __future__ import annotations

import json
import sys
from pathlib import Path

from .paths import DATA

TEMPLATE = DATA / "template.json"
OPS = {"concat", "add", "sub", "mul", "div", "year", "days_between", "lookup"}
GENS = {"id", "pick", "date", "money", "rate"}
KINDS = {"pass", "concat", "arith", "format", "lookup"}
# how a template output kind is named when the model is asked about it, and what each name means
KIND_LABEL = {
    "pass": "passthrough",
    "concat": "concat",
    "arith": "arithmetic",
    "format": "format",
    "lookup": "lookup",
}
KIND_DESC = {
    "passthrough": "the value is copied unchanged from one input column",
    "concat": "text from several input columns is joined together",
    "arithmetic": "the value is calculated from input columns with + - * / or a difference between dates",
    "format": "the value is a reformatted part of one input column, such as the year of a date",
    "lookup": "a code in an input column is translated into a label",
}


def load_template(path=TEMPLATE) -> dict:
    return json.loads(Path(path).read_text())


# Distinct source ids a rule reads, in first-use order
def rule_sources(rule) -> list[str]:
    if "src" in rule:
        return [rule["src"]]
    seen: list[str] = []
    for a in rule.get("args", ()):
        for s in rule_sources(a):
            if s not in seen:
                seen.append(s)
    return seen


def _rule_problems(rule, where, ids):
    if "src" in rule:
        return (
            [] if rule["src"] in ids else [f"{where}: unknown source {rule['src']!r}"]
        )
    if "lit" in rule:
        return []
    if rule.get("op") not in OPS:
        return [f"{where}: op {rule.get('op')!r} not allowed"]
    return [p for a in rule.get("args", ()) for p in _rule_problems(a, where, ids)]


def check(t: dict) -> list[str]:
    probs: list[str] = []
    ids = [s["id"] for s in t["sources"]]
    if len(set(ids)) != len(ids):
        probs.append("duplicate source ids")
    all_ids = ids + [d["id"] for d in t.get("decoys", ())]
    if len(set(all_ids)) != len(all_ids):
        probs.append("source/decoy id collision")
    for s in t["sources"]:
        if set(s.get("names", {})) != {"2", "3", "4", "5"}:
            probs.append(f"source {s['id']}: names must have keys 2,3,4,5")
    for s in t["sources"] + t.get("decoys", []):
        if s["gen"][0] not in GENS:
            probs.append(f"{s['id']}: unknown generator {s['gen'][0]!r}")
    names = [o["name"] for o in t["outputs"]]
    if len(set(names)) != len(names):
        probs.append("duplicate output names")
    used: set[str] = set()
    for o in t["outputs"]:
        probs += _rule_problems(o["rule"], f"output {o['name']}", set(ids))
        used |= set(rule_sources(o["rule"])) & set(ids)
        if o.get("kind") not in KINDS:
            probs.append(f"output {o['name']}: bad kind {o.get('kind')!r}")
        if (o.get("kind") == "pass") != ("src" in o["rule"]):
            probs.append(
                f"output {o['name']}: kind pass must be exactly a bare src rule"
            )
    probs += [f"source {i} feeds no output" for i in ids if i not in used]
    return probs


if __name__ == "__main__":
    problems = check(load_template())
    print("\n".join(problems) or "template ok")
    sys.exit(1 if problems else 0)
