"""Mapping runner (backend-agnostic). Run: uv run python -m src.run_mapping --cond name,both --n 10

Per case: for every input column (decoys included) one predicate per output field -> matrix[input][output];
per available output one choice question for the transformation kind given its true source columns.
"""
from __future__ import annotations

import argparse
import json
import tomllib
from pathlib import Path

from .check_template import load_template
from .context import CONDS, KIND_LABEL, input_state, kind_question, kind_state, mapping_question

ROOT = Path(__file__).resolve().parents[1]


def _done(path: Path) -> set[str]:
    return {json.loads(l)["case_id"] for l in path.read_text().splitlines()} if path.exists() else set()


def run_case(decider, case: dict, cond: str, t: dict) -> dict:
    desc = {s["id"]: s["desc"] for s in t["sources"] + t["decoys"]}
    odesc = {o["name"]: o["description"] for o in t["outputs"]}
    sample = {c["name"]: [r[c["name"]] for r in case["rows"][:3]] for c in case["columns"]}
    qs = [mapping_question(o["name"], odesc[o["name"]], cond) for o in case["outputs"]]
    matrix = {}
    for c in case["columns"]:
        res = decider.decide(input_state(c, cond, desc[c["src"] or c["decoy"]], sample[c["name"]]), qs)
        matrix[c["name"]] = {a.name: {"p": a.probability, "raw": a.raw_scores} for a in res.answers}
    col = {c["name"]: c for c in case["columns"]}
    kind = {}
    for o in case["outputs"]:
        if not o["available"]:
            continue
        cols = [(n, desc[col[n]["src"]], sample[n]) for n in o["sources"]]
        a = decider.decide(kind_state(cols, cond), [kind_question(o["name"], odesc[o["name"]], cond)]).answers[0]
        kind[o["name"]] = {"choice": a.choice, "p": a.probabilities, "raw": a.raw_scores}
    return {"case_id": case["case_id"], "level": case["level"], "cond": cond, "matrix": matrix, "kind": kind}


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


def load_cases(gt, levels=None, n=None):
    cs = [json.loads(l) for l in Path(gt).read_text().splitlines()]
    if levels is not None:
        cs = [c for c in cs if c["level"] in levels]
    if n is not None:
        seen: dict[int, int] = {}
        keep = []
        for c in cs:
            seen[c["level"]] = seen.get(c["level"], 0) + 1
            if seen[c["level"]] <= n:
                keep.append(c)
        cs = keep
    return cs


if __name__ == "__main__":
    cfg = tomllib.loads((ROOT / "config.toml").read_text())["run"]
    ap = argparse.ArgumentParser()
    ap.add_argument("--cond", default=",".join(cfg["conds"]))
    ap.add_argument("--levels", default=",".join(map(str, cfg["levels"])))
    ap.add_argument("--n", type=int, default=cfg["n_per_level"])
    ap.add_argument("--gt", default=cfg["gt"])
    ap.add_argument("--out-dir", default=cfg["out_dir"])
    a = ap.parse_args()
    from .decisions.jev import JevDecider
    t = load_template()
    cases = load_cases(ROOT / a.gt, {int(x) for x in a.levels.split(",")}, a.n)
    with JevDecider(cfg["quant"]) as d:
        for cond in a.cond.split(","):
            assert cond in CONDS, cond
            k = run(d, cases, cond, ROOT / a.out_dir / "jev" / f"{cond}.jsonl", t)
            print(f"{cond}: {k} new cases")
