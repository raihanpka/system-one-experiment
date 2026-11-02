"""Transformation ranking: Jev picks the right expression among candidates (1 truth + distractors).

Per derived, available output: one `choice` question over the rendered candidates and one `predicate`
per candidate. The winner is executed on the sample rows and compared with the expected values.
"""
from __future__ import annotations

import math

from .core.template import rule_sources
from .core.prompts import kind_state, samples
from .decisions.types import Choice, Question
from .core.dsl import render, safe_evaluate


def _same(a, b):
    return len(a) == len(b) and all(
        math.isclose(x, y, rel_tol=1e-9) if isinstance(x, float) and isinstance(y, float) else x == y for x, y in zip(a, b))


def _ok(rule, by_id, expected):
    return _same([safe_evaluate(rule, r) for r in by_id], expected)


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


def run_case(decider, case: dict, cond: str, t: dict) -> dict:
    name_of = {c["src"]: c["name"] for c in case["columns"] if c["src"]}
    by_id = [{s: row[n] for s, n in name_of.items()} for row in case["rows"]]
    res = {}
    for o in case["outputs"]:
        if not o["available"] or o["kind"] == "pass":
            continue
        cands = o["candidates"]
        state, label, texts = prepare(case, cond, t, o)
        qs = [pick_question(label, texts)]
        qs += [Question("predicate", f"The output field {label} is computed as: {x}", name=f"c{i}") for i, x in enumerate(texts)]
        a = decider.decide(state, qs).answers
        ci = texts.index(a[0].choice)
        probs = [x.probability for x in a[1:]]
        pi = max(range(len(probs)), key=probs.__getitem__)
        ti = cands.index(o["rule"])
        res[o["name"]] = {
            "kind": o["kind"], "cands": texts, "truth_idx": ti,
            "choice_idx": ci, "choice_p": [a[0].probabilities[x] for x in texts],
            "pred_idx": pi, "pred_p": probs,
            "choice_value_ok": _ok(cands[ci], by_id, o["expected"]), "pred_value_ok": _ok(cands[pi], by_id, o["expected"]),
        }
    return {"case_id": case["case_id"], "level": case["level"], "cond": cond, "transform": res}


def run(decider, cases, cond: str, store, t: dict) -> int:
    n = 0
    for c in cases:
        if not store.has("transform", cond, c["case_id"]):
            store.add("transform", cond, run_case(decider, c, cond, t))
            n += 1
    return n
