# synthetic samples: input files with named drift, many rows, anomalies and ground truth, rebuilt from a seed

from __future__ import annotations

import random
from datetime import date, timedelta

from ..core.dsl import safe_evaluate
from ..core.template import rule_sources
from .samples import LEVEL_WORDS

ROWS = (15, 30)                                            # rows per input file
STYLES = ("snake", "camel", "pascal", "upper", "kebab", "space", "title")
ARITH = ("add", "sub", "mul", "div")


def style(name: str, how: str) -> str:
    w = name.lower().split("_")
    return {"snake": "_".join(w), "camel": w[0] + "".join(x.capitalize() for x in w[1:]),
            "pascal": "".join(x.capitalize() for x in w), "upper": "_".join(w).upper(),
            "kebab": "-".join(w), "space": " ".join(w), "title": " ".join(x.capitalize() for x in w)}[how]


def sample(rng: random.Random, gen: list):
    k = gen[0]
    if k == "id":
        return gen[1] + str(rng.randrange(10 ** gen[2])).zfill(gen[2])
    if k == "pick":
        return rng.choice(gen[1])
    if k == "date":
        lo, hi = date.fromisoformat(gen[1]), date.fromisoformat(gen[2])
        return (lo + timedelta(days=rng.randrange((hi - lo).days + 1))).isoformat()
    if k == "money":
        step = gen[3] if len(gen) > 3 else 1000
        return gen[1] + rng.randrange((gen[2] - gen[1]) // step + 1) * step
    return round(rng.uniform(gen[1], gen[2]), gen[3])          # rate


def _paths(node, path=()):
    yield path, node
    for i, a in enumerate(node.get("args", ())):
        yield from _paths(a, path + (i,))


def _replace(node, path, new):
    if not path:
        return new
    args = list(node["args"])
    args[path[0]] = _replace(args[path[0]], path[1:], new)
    return {**node, "args": args}


# Single-edit variants of a rule: op swap, arg reversal, other same-type source, other literal, rotated map
def mutations(rule: dict, types: dict, present: list[str]):
    for path, n in _paths(rule):
        if n.get("op") in ARITH:
            for o in ARITH:
                if o != n["op"]:
                    yield _replace(rule, path, {**n, "op": o})
        if len(n.get("args", ())) == 2 and n.get("op") in ("sub", "div", "days_between", "concat"):
            yield _replace(rule, path, {**n, "args": n["args"][::-1]})
        if "src" in n:
            for s in present:
                if s != n["src"] and types[s] == types[n["src"]]:
                    yield _replace(rule, path, {"src": s})
        if "lit" in n:
            yield _replace(rule, path, {"lit": n["lit"] + 1})
        if n.get("op") in ("year", "lookup"):
            yield _replace(rule, path, n["args"][0])                      # drop the transform
        if n.get("op") == "lookup":
            v = list(n["map"].values())
            yield _replace(rule, path, {**n, "map": dict(zip(n["map"], v[1:] + v[:1]))})
            yield _replace(rule, path, {**n, "map": dict(zip(n["map"], v[::-1]))})


# 1-2 odd cells per file: a zero/negative/huge number, an empty or N/A text, a sentinel date
def _anomalies(rng, cols, rows):
    out, used = [], set()
    for _ in range(rng.randint(1, 2)):
        c, r = rng.choice(cols), rng.randrange(len(rows))
        if (c["name"], r) in used:
            continue
        used.add((c["name"], r))
        v = rows[r][c["name"]]
        kind, new = {
            "number": lambda: rng.choice([("zero", 0), ("negative", -abs(v) if v else -1), ("huge", v * 1000 + 1)]),
            "string": lambda: rng.choice([("empty", ""), ("na", "N/A")]),
            "date": lambda: ("sentinel", "1900-01-01"),
        }[c["type"]]()
        rows[r][c["name"]] = new
        out.append({"column": c["name"], "row": r, "kind": kind, "value": new})
    return out


def _case(t: dict, level: int, i: int, seed: int) -> dict:
    rng = random.Random(f"{seed}:{level}:{i}")
    srcs = {s["id"]: s for s in t["sources"]}
    types = {k: v["type"] for k, v in srcs.items()}
    present = sorted(rng.sample(list(srcs), rng.randint(14, 22)), key=list(srcs).index)
    case_style = "snake" if level == 0 else rng.choice(STYLES)

    cols = []
    for sid in present:
        lv = rng.randint(0, 5) if level == 6 else level
        base = sid if lv < 2 else srcs[sid]["names"][str(lv)]
        how = "snake" if lv == 0 else (rng.choice(STYLES[1:]) if lv == 1 else case_style)
        if lv == 5:
            how = "snake" if case_style == "snake" else "upper"   # unusual technical terms: OBJ_INFO_ADDR or obj_info_addr, never camelCase
        cols.append({"name": style(base, how), "src": sid, "level": lv, "decoy": None, "gen": srcs[sid]["gen"], "type": srcs[sid]["type"]})
    if level == 6:
        for d in rng.sample(t["decoys"], rng.randint(3, len(t["decoys"]))):
            cols.append({"name": style(d["id"], case_style), "src": None, "level": 6, "decoy": d["id"], "gen": d["gen"], "type": d["type"]})
    rng.shuffle(cols)
    assert len({c["name"] for c in cols}) == len(cols), "column name collision"

    rows = [{c["name"]: sample(rng, c["gen"]) for c in cols} for _ in range(rng.randint(*ROWS))]
    anomalies = _anomalies(rng, cols, rows)
    name_of = {c["src"]: c["name"] for c in cols if c["src"]}
    by_id = [{s: row[n] for s, n in name_of.items()} for row in rows]

    outputs = []
    for o in t["outputs"]:
        ids = rule_sources(o["rule"])
        avail = set(ids) <= set(name_of)
        out = {"name": o["name"], "kind": o["kind"], "rule": o["rule"], "source_ids": ids, "type": o["type"], "available": avail,
               "sources": [name_of[s] for s in ids] if avail else [], "expected": None, "candidates": []}
        if avail:
            out["expected"] = [safe_evaluate(o["rule"], r) for r in by_id]
            if o["kind"] != "pass":
                bad, seen = [], []
                for m in mutations(o["rule"], types, list(name_of)):
                    if m == o["rule"] or m in seen:
                        continue
                    seen.append(m)
                    vals = [safe_evaluate(m, r) for r in by_id]
                    if vals != out["expected"] and any(v is not None for v in vals):
                        bad.append(m)
                cands = [o["rule"]] + rng.sample(bad, min(3, len(bad)))
                rng.shuffle(cands)
                out["candidates"] = cands
        outputs.append(out)
    for c in cols:
        del c["gen"], c["type"]
    return {"case_id": f"{LEVEL_WORDS[level]}-{i:03d}", "level": level, "columns": cols, "rows": rows, "anomalies": anomalies,
            "outputs": outputs}


def generate(t: dict, n_per_level: int = 50, seed: int = 1234) -> list[dict]:
    return [_case(t, lv, i, seed) for lv in sorted(LEVEL_WORDS) for i in range(n_per_level)]
