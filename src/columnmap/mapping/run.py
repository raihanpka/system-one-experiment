# mapping step: a score for every (input column, output field) pair

from __future__ import annotations

import random
import time

from ..core.prompts import input_state, mapping_question, samples, table_options_question, table_pair_question, table_state
from ..core.scoring import T0, pair_p
from ..core.usage import mark, since
from ..decisions.base import Decider
from ..decisions.types import Choice, Question

SECTIONS = {"options": "mapping", "pairs": "mapping_pairs"}


# the option order of the k-th question about a column: the given order first, then shuffles fixed by column and k
def shuffled(names: list[str], col: str, k: int) -> list[str]:
    if k == 0:
        return list(names)
    out = list(names)
    random.Random(f"{col}:{k}").shuffle(out)
    return out


# options: one choice question per column, the raw score of each output field is read; pairs: one yes/no per field
def run_case(decider, case: dict, cond: str, t: dict, mode: str = "options", orders: int = 1) -> dict:
    desc = {s["id"]: s["desc"] for s in t["sources"] + t["decoys"]}
    odesc = {o["name"]: o["description"] for o in t["outputs"]}
    outs = [o["name"] for o in case["outputs"]]
    sample = {c["name"]: samples(case["rows"], c["name"]) for c in case["columns"]}
    pair_qs = [mapping_question(o, odesc[o], cond) for o in outs] if (mode == "pairs" and cond != "table") else None
    matrix = {}
    for c in case["columns"]:
        if cond == "table":
            state = table_state(case)
        else:
            state = input_state(c, cond, desc[c["src"] or c["decoy"]], sample[c["name"]])
        if mode == "pairs":
            qs = pair_qs or [table_pair_question(c["name"], o) for o in outs]
            res = decider.decide(state, qs)
            matrix[c["name"]] = {a.name: {"p": a.probability, "d": a.raw_scores["true"] - a.raw_scores["false"]} for a in res.answers}
            continue
        scores = {o: [] for o in outs}
        for k in range(orders):                                       # the same question with the options in another order
            order = shuffled(outs, c["name"], k)
            q = (table_options_question(c["name"], order) if cond == "table" else
                 Question("choice", "Which output fields does this input column provide data for?", name="field", choices=tuple(Choice(o) for o in order)))
            for f, d in decider.decide(state, [q]).answers[0].raw_scores.items():
                scores[f].append(d)
        matrix[c["name"]] = {f: {"p": pair_p({"d": sum(v) / len(v)}, T0), "d": sum(v) / len(v)} for f, v in scores.items()}
    return {"case_id": case["case_id"], "level": case["level"], "cond": cond, "mode": mode, "matrix": matrix}


def run(decider: Decider, cases, cond: str, store, t: dict, mode: str = "options", orders: int = 1) -> int:
    section, n = SECTIONS[mode], 0
    for c in cases:
        if not store.has(section, cond, c["case_id"]):
            t0, before = time.time(), mark(decider)
            rec = run_case(decider, c, cond, t, mode, orders)
            rec["seconds"] = round(time.time() - t0, 1)
            rec.update(since(decider, before))
            store.add(section, cond, rec)
            n += 1
    return n
