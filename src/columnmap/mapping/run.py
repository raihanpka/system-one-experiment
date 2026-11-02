"""Mapping runner (backend-agnostic); `python src` drives it.

Per case: for every input column (decoys included) a score for every output field -> matrix[input][output]
= {"p": probability, "d": s_true - s_false} (d lets any temperature be re-applied offline); per available
output one choice question for the transformation kind given its true source columns.

Modes: "options" (default, about 7x faster) asks ONE choice question per input column with all output fields as
options and reads each option's raw yes/no score (independent, so one column can feed several fields; the softmax
over options is ignored). "pairs" asks one yes/no question per (input column, output field): slower, calibrated.
"""
from __future__ import annotations

from ..decisions.base import Decider
from ..core.prompts import input_state, kind_question, kind_state, mapping_question, samples
from ..decisions.types import Choice, Question
from ..core.scoring import T0, pair_p

def run_case(decider, case: dict, cond: str, t: dict, mode: str = "options") -> dict:
    desc = {s["id"]: s["desc"] for s in t["sources"] + t["decoys"]}
    odesc = {o["name"]: o["description"] for o in t["outputs"]}
    sample = {c["name"]: samples(case["rows"], c["name"]) for c in case["columns"]}
    matrix = {}
    if mode == "pairs":
        qs = [mapping_question(o["name"], odesc[o["name"]], cond) for o in case["outputs"]]
    else:
        qs = [Question("choice", "Which output fields does this input column provide data for?", name="field",
                       choices=tuple(Choice(o["name"]) for o in case["outputs"]))]
    for c in case["columns"]:
        res = decider.decide(input_state(c, cond, desc[c["src"] or c["decoy"]], sample[c["name"]]), qs)
        if mode == "pairs":
            matrix[c["name"]] = {a.name: {"p": a.probability, "d": a.raw_scores["true"] - a.raw_scores["false"]} for a in res.answers}
        else:
            raw = res.answers[0].raw_scores
            matrix[c["name"]] = {k: {"p": pair_p({"d": d}, T0), "d": d} for k, d in raw.items()}
    col = {c["name"]: c for c in case["columns"]}
    kind = {}
    for o in case["outputs"]:
        if not o["available"]:
            continue
        cols = [(n, desc[col[n]["src"]], sample[n]) for n in o["sources"]]
        a = decider.decide(kind_state(cols, cond), [kind_question(o["name"], odesc[o["name"]], cond)]).answers[0]
        kind[o["name"]] = {"choice": a.choice, "p": a.probabilities}
    return {"case_id": case["case_id"], "level": case["level"], "cond": cond, "mode": mode, "matrix": matrix, "kind": kind}


def run(decider: Decider, cases, cond: str, store, t: dict, mode: str = "options") -> int:
    n = 0
    for c in cases:
        if not store.has("mapping", cond, c["case_id"]):
            store.add("mapping", cond, run_case(decider, c, cond, t, mode))
            n += 1
    return n
