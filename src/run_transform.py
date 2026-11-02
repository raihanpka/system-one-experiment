"""Transformation ranking: Jev picks the right expression among candidates (1 truth + distractors).
Run: uv run python -m src.run_transform --cond both --n 5

Per derived, available output: one `choice` question over the rendered candidates and one `predicate`
per candidate. The winner is executed on the sample rows and compared with the expected values.
"""
from __future__ import annotations

import argparse
import json
import math
import tomllib
from pathlib import Path

from .check_template import load_template, rule_sources
from .context import CONDS, kind_state
from .decisions.types import Choice, Question
from .dsl import evaluate, render
from .run_mapping import ROOT, _done, load_cases


def _same(a, b):
    return len(a) == len(b) and all(math.isclose(x, y, rel_tol=1e-9) if isinstance(x, float) or isinstance(y, float) else x == y
                                    for x, y in zip(a, b))


def _ok(rule, by_id, expected):
    try:
        return _same([evaluate(rule, r) for r in by_id], expected)
    except Exception:
        return False


def prepare(case: dict, cond: str, t: dict, o: dict):
    """State text, label and candidate texts for one derived output."""
    desc = {s["id"]: s["desc"] for s in t["sources"]}
    odesc = {x["name"]: x["description"] for x in t["outputs"]}
    name_of = {c["src"]: c["name"] for c in case["columns"] if c["src"]}
    cands = o["candidates"]
    texts = [render(c, name_of.get) for c in cands]
    texts = [x if texts.count(x) == 1 else f"{x}  (#{i})" for i, x in enumerate(texts)]
    ids = list(dict.fromkeys(s for c in cands for s in rule_sources(c)))
    state = kind_state([(name_of[s], desc[s], [r[name_of[s]] for r in case["rows"][:3]]) for s in ids], cond)
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
            "choice_idx": ci, "choice_p": [a[0].probabilities[x] for x in texts], "choice_raw": a[0].raw_scores,
            "pred_idx": pi, "pred_p": probs, "pred_raw": [x.raw_scores for x in a[1:]],
            "choice_value_ok": _ok(cands[ci], by_id, o["expected"]), "pred_value_ok": _ok(cands[pi], by_id, o["expected"]),
        }
    return {"case_id": case["case_id"], "level": case["level"], "cond": cond, "transform": res}


def run(decider, cases, cond: str, out, t: dict) -> int:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done, n = _done(out), 0
    with out.open("a") as f:
        for c in cases:
            if c["case_id"] in done:
                continue
            f.write(json.dumps(run_case(decider, c, cond, t), ensure_ascii=False) + "\n")
            f.flush()
            n += 1
    return n


if __name__ == "__main__":
    cfg = tomllib.loads((ROOT / "config.toml").read_text())["run"]
    ap = argparse.ArgumentParser()
    ap.add_argument("--cond", default=",".join(cfg["conds"]))
    ap.add_argument("--levels", default=",".join(map(str, cfg["levels"])))
    ap.add_argument("--n", type=int, default=cfg["n_per_level"])
    ap.add_argument("--seed", type=int, default=cfg["seed"])
    ap.add_argument("--out-dir", default=cfg["out_dir"])
    a = ap.parse_args()
    from .decisions.jev import JevDecider
    t = load_template()
    cases = load_cases({int(x) for x in a.levels.split(",")}, a.n, a.seed)
    with JevDecider(cfg["quant"]) as d:
        for cond in a.cond.split(","):
            assert cond in CONDS, cond
            k = run(d, cases, cond, ROOT / a.out_dir / "jev" / f"transform_{cond}.jsonl", t)
            print(f"{cond}: {k} new cases")
