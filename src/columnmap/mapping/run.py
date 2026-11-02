# mapping step: a score for every (input column, output field) pair

from __future__ import annotations

import time

from ..core.prompts import input_state, mapping_question, samples, table_options_question, table_pair_question, table_state
from ..core.scoring import MASKED_D, T0, pair_p
from ..core.usage import mark, since
from ..decisions.base import Decider
from ..decisions.types import Choice, Question

SECTIONS = {"options": "mapping", "pairs": "mapping_pairs", "twostage": "mapping_twostage"}
SHORTLIST_FIELDS = 3                  # two-stage: fields per column that the yes/no stage checks


# the two columns before and after this one in the file (names only)
def _neighbors(case: dict, name: str) -> list[str]:
    names = [k["name"] for k in case["columns"]]
    i = names.index(name)
    return [names[j] for j in (i - 1, i + 1) if 0 <= j < len(names)]


def _context(case: dict, t: dict):
    desc = {s["id"]: s["desc"] for s in t["sources"] + t["decoys"]}
    odesc = {o["name"]: o["description"] for o in t["outputs"]}
    sample = {c["name"]: samples(case["rows"], c["name"]) for c in case["columns"]}
    return desc, odesc, sample


def _state(case, c, cond, desc, sample) -> str:
    if cond == "table":
        return table_state(case)
    others = _neighbors(case, c["name"]) if cond == "neighbors" else ()
    return input_state(c, cond, desc[c["src"] or c["decoy"]], sample[c["name"]], others)


# options: one choice question per column, the raw score of each output field is read; pairs: one yes/no per field
def run_case(decider, case: dict, cond: str, t: dict, mode: str = "options") -> dict:
    desc = {s["id"]: s["desc"] for s in t["sources"] + t["decoys"]}
    odesc = {o["name"]: o["description"] for o in t["outputs"]}
    outs = [o["name"] for o in case["outputs"]]
    sample = {c["name"]: samples(case["rows"], c["name"]) for c in case["columns"]}
    if cond != "table":
        qs = ([mapping_question(o, odesc[o], cond) for o in outs] if mode == "pairs" else
              [Question("choice", "Which output fields does this input column provide data for?", name="field",
                        choices=tuple(Choice(o) for o in outs))])
    matrix = {}
    for c in case["columns"]:
        if cond == "table":
            state = table_state(case)
            qs = [table_pair_question(c["name"], o) for o in outs] if mode == "pairs" else [table_options_question(c["name"], outs)]
        else:
            state = _state(case, c, cond, desc, sample)
        res = decider.decide(state, qs)
        if mode == "pairs":
            matrix[c["name"]] = {a.name: {"p": a.probability, "d": a.raw_scores["true"] - a.raw_scores["false"]} for a in res.answers}
        else:
            matrix[c["name"]] = {k: {"p": pair_p({"d": d}, T0), "d": d} for k, d in res.answers[0].raw_scores.items()}
    return {"case_id": case["case_id"], "level": case["level"], "cond": cond, "mode": mode, "matrix": matrix}


# two-stage: the options question shortlists the fields of each column, then yes/no questions check only those fields;
# fields that were not shortlisted get a masked score (MASKED_D), so calibration never sees them as answers
def run_twostage(decider, case: dict, cond: str, t: dict) -> dict:
    base = run_case(decider, case, cond, t, "options")
    desc, odesc, sample = _context(case, t)
    matrix = {}
    for c in case["columns"]:
        name = c["name"]
        scores = {f: v["d"] for f, v in base["matrix"][name].items()}
        top = sorted(scores, key=lambda f: (-scores[f], f))[:SHORTLIST_FIELDS]
        qs = ([table_pair_question(name, f) for f in top] if cond == "table"
              else [mapping_question(f, odesc[f], cond) for f in top])
        res = decider.decide(_state(case, c, cond, desc, sample), qs)
        checked = {a.name: a.raw_scores["true"] - a.raw_scores["false"] for a in res.answers}
        matrix[name] = {f: ({"p": pair_p({"d": checked[f]}, T0), "d": checked[f]} if f in checked
                            else {"p": pair_p({"d": MASKED_D}, T0), "d": MASKED_D}) for f in scores}
    return {"case_id": case["case_id"], "level": case["level"], "cond": cond, "mode": "twostage", "matrix": matrix}


def run(decider: Decider, cases, cond: str, store, t: dict, mode: str = "options") -> int:
    section, n = SECTIONS[mode], 0
    for c in cases:
        if not store.has(section, cond, c["case_id"]):
            t0, before = time.time(), mark(decider)
            rec = run_twostage(decider, c, cond, t) if mode == "twostage" else run_case(decider, c, cond, t, mode)
            rec["seconds"] = round(time.time() - t0, 1)
            rec.update(since(decider, before))
            store.add(section, cond, rec)
            n += 1
    return n
