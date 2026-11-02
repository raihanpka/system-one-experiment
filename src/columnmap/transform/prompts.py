"""Prompt text for the transformation ranking question."""
from __future__ import annotations

from ..core.dsl import render
from ..core.prompts import kind_state, samples
from ..core.template import rule_sources
from ..decisions.types import Choice, Question


def prepare(case: dict, cond: str, t: dict, o: dict):
    """State text, label and candidate texts for one derived output."""
    desc = {s["id"]: s["desc"] for s in t["sources"]}
    odesc = {x["name"]: x["description"] for x in t["outputs"]}
    name_of = {c["src"]: c["name"] for c in case["columns"] if c["src"]}
    cands = o["candidates"]
    texts = [render(c, name_of.get) for c in cands]
    texts = [x if texts.count(x) == 1 else f"{x}  (#{i})" for i, x in enumerate(texts)]
    ids = list(dict.fromkeys(s for c in cands for s in rule_sources(c)))
    state = kind_state([(name_of[s], desc[s], samples(case["rows"], name_of[s])) for s in ids], cond)
    label = f"'{o['name']}'" + (f" (meaning: {odesc[o['name']]})" if cond in ("desc", "both") else "")
    return state, label, texts


def pick_question(label: str, texts) -> Question:
    return Question("choice", f"Which expression produces the output field {label} from the input columns?", name="pick",
                    choices=tuple(Choice(x) for x in texts))
