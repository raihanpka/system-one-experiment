"""Option-order stability: re-ask each choice question with shuffled option order.
Run: uv run python -m src.permute --cond both --n 3 --k 3

Per derived output: `k` orders of the candidate expressions (first = as generated); per available output
the transformation-kind question with `k` orders of its labels. Records picked text, picked slot and
whether the pick was the true one, so order sensitivity and position bias can be measured.
"""
from __future__ import annotations

import argparse
import json
import random
import tomllib
from pathlib import Path

from .check_template import load_template
from .context import CONDS, KIND_DESC, KIND_LABEL, kind_question, kind_state, samples
from .decisions.types import Choice, Question
from .run_mapping import ROOT, _done, load_cases
from .run_transform import pick_question, prepare


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


def run(decider, cases, cond: str, out, t: dict, k: int = 3) -> int:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done, n = _done(out), 0
    with out.open("a") as f:
        for c in cases:
            if c["case_id"] in done:
                continue
            f.write(json.dumps(run_case(decider, c, cond, t, k), ensure_ascii=False) + "\n")
            f.flush()
            n += 1
    return n


if __name__ == "__main__":
    cfg = tomllib.loads((ROOT / "config.toml").read_text())["run"]
    ap = argparse.ArgumentParser()
    ap.add_argument("--cond", default="both")
    ap.add_argument("--levels", default=",".join(map(str, cfg["levels"])))
    ap.add_argument("--n", type=int, default=cfg["n_per_level"])
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--seed", type=int, default=cfg["seed"])
    ap.add_argument("--out-dir", default=cfg["out_dir"])
    a = ap.parse_args()
    from .decisions.jev import JevDecider
    t = load_template()
    cases = load_cases({int(x) for x in a.levels.split(",")}, a.n, a.seed)
    with JevDecider(cfg["quant"]) as d:
        for cond in a.cond.split(","):
            assert cond in CONDS, cond
            print(f"{cond}: {run(d, cases, cond, ROOT / a.out_dir / 'jev' / f'permute_{cond}.jsonl', t, a.k)} new cases")
