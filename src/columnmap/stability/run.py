"""Option-order stability: re-ask each choice question with shuffled option order.

Per derived output: `k` orders of the candidate expressions (first = as generated); per available output
the transformation-kind question with `k` orders of its labels. Records picked text, picked slot and
whether the pick was the true one, so order sensitivity and position bias can be measured.
"""
from __future__ import annotations

import random

from ..decisions.base import Decider
from ..core.prompts import kind_question, kind_state, samples
from ..core.template import KIND_DESC, KIND_LABEL
from ..decisions.types import Choice, Question
from ..transform.prompts import pick_question, prepare


def _orders(n, k, rng):
    orders = [list(range(n))]
    for _ in range(k - 1):
        o = list(range(n))
        rng.shuffle(o)
        orders.append(o)
    return orders


def run_case(decider, case: dict, cond: str, t: dict, k: int = 3, seed: int = 0) -> dict:
    desc = {s["id"]: s["desc"] for s in t["sources"]}
    odesc = {x["name"]: x["description"] for x in t["outputs"]}
    col = {c["name"]: c for c in case["columns"]}
    transform, kind = {}, {}
    for o in case["outputs"]:
        if not o["available"]:
            continue
        rng = random.Random(f"{seed}:{case['case_id']}:{o['name']}")
        if o["kind"] != "pass":
            state, label, texts = prepare(case, cond, t, o)
            truth = texts[o["candidates"].index(o["rule"])]
            orders = _orders(len(texts), k, rng)
            picks, pos = [], []
            for od in orders:
                ordered = [texts[i] for i in od]
                a = decider.decide(state, [pick_question(label, ordered)]).answers[0]
                picks.append(a.choice)
                pos.append(ordered.index(a.choice))
            transform[o["name"]] = {"orders": orders, "picks": picks, "pos": pos, "correct": [p == truth for p in picks]}
        labels = list(KIND_DESC)
        want = KIND_LABEL[o["kind"]]
        cols = [(n, desc[col[n]["src"]], samples(case["rows"], n)) for n in o["sources"]]
        state = kind_state(cols, cond)
        orders = _orders(len(labels), k, rng)
        picks, pos = [], []
        for od in orders:
            q = kind_question(o["name"], odesc[o["name"]], cond)
            q = Question(q.type, q.instructions, q.name, tuple(Choice(labels[i], KIND_DESC[labels[i]]) for i in od))
            a = decider.decide(state, [q]).answers[0]
            picks.append(a.choice)
            pos.append([labels[i] for i in od].index(a.choice))
        kind[o["name"]] = {"orders": orders, "picks": picks, "pos": pos, "correct": [p == want for p in picks]}
    return {"case_id": case["case_id"], "level": case["level"], "cond": cond, "k": k, "transform": transform, "kind": kind}


def run(decider: Decider, cases, cond: str, store, t: dict, k: int = 3) -> int:
    n = 0
    for c in cases:
        if not store.has("permute", cond, c["case_id"]):
            store.add("permute", cond, run_case(decider, c, cond, t, k))
            n += 1
    return n
