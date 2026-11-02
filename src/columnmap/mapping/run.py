# mapping step: a score for every (input column, output field) pair

from __future__ import annotations

import time

from ..core.prompts import input_state, mapping_question, samples
from ..core.scoring import T0, pair_p
from ..core.usage import mark, since
from ..decisions.base import Decider
from ..decisions.types import Choice, Question

SECTIONS = {"options": "mapping", "pairs": "mapping_pairs"}


# options: one choice question per column, the raw score of each output field is read; pairs: one yes/no per field
def run_case(decider, case: dict, cond: str, mode: str = "options") -> dict:
    outs = [o["name"] for o in case["outputs"]]
    sample = {c["name"]: samples(case["rows"], c["name"]) for c in case["columns"]}
    if mode == "pairs":
        qs = [mapping_question(o) for o in outs]
    else:
        qs = [Question("choice", "Which output fields does this input column provide data for?", name="field",
                       choices=tuple(Choice(o) for o in outs))]
    matrix = {}
    for c in case["columns"]:
        res = decider.decide(input_state(c, cond, sample[c["name"]]), qs)
        if mode == "pairs":
            matrix[c["name"]] = {a.name: {"p": a.probability, "d": a.raw_scores["true"] - a.raw_scores["false"]} for a in res.answers}
        else:
            matrix[c["name"]] = {k: {"p": pair_p({"d": d}, T0), "d": d} for k, d in res.answers[0].raw_scores.items()}
    return {"case_id": case["case_id"], "level": case["level"], "cond": cond, "mode": mode, "matrix": matrix}


def run(decider: Decider, cases, cond: str, store, mode: str = "options") -> int:
    section, n = SECTIONS[mode], 0
    for c in cases:
        if not store.has(section, cond, c["case_id"]):
            t0, before = time.time(), mark(decider)
            rec = run_case(decider, c, cond, mode)
            rec["seconds"] = round(time.time() - t0, 1)
            rec.update(since(decider, before))
            store.add(section, cond, rec)
            n += 1
    return n
