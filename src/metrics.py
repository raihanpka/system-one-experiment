"""Metrics and report. Run: uv run python -m src.metrics   (reads results/jev/*.jsonl, writes results/summary.md)

Mapping: for each available output, input columns are ranked by P(true). top1 = best column is a true
source; exact = the best |sources| columns are exactly the sources; top3 = a true source is in the top 3.
Unavailable outputs should be rejected (best P below 0.5); decoy columns should never reach P >= 0.5.
Pairwise precision/recall/F1/ECE/Brier use all (input column, output field) pairs at threshold 0.5.
Temperature: Jev probabilities are sigmoid((s_true - s_false) / T); T is re-applied offline from raw scores.
"""
from __future__ import annotations

import json
import math
import re
import tomllib
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path

from .check_template import load_template
from .context import KIND_LABEL
from .gen_gt import generate

ROOT = Path(__file__).resolve().parents[1]
T0 = 0.8278650620942867                     # calibrated global temperature of Jev v3
TS = (0.5, 0.7, T0, 1.0, 1.5, 2.0, 3.0)


def pair_p(cell: dict, T: float) -> float:
    d = (cell["raw"]["true"] - cell["raw"]["false"]) / T
    return 1 / (1 + math.exp(-d)) if d > -700 else 0.0


def ece(ps, ys, bins: int = 10) -> float:
    if not ps:
        return 0.0
    tot = 0.0
    for b in range(bins):
        idx = [i for i, p in enumerate(ps) if (p >= b / bins and (p < (b + 1) / bins or b == bins - 1))]
        if idx:
            conf = sum(ps[i] for i in idx) / len(idx)
            acc = sum(bool(ys[i]) for i in idx) / len(idx)
            tot += len(idx) / len(ps) * abs(conf - acc)
    return tot


def _div(a, b):
    return a / b if b else 0.0


def mapping_metrics(rows, cases, T: float = T0) -> dict:
    top1 = exact = top3 = n = rej = nun = decoys = flag = tp = fp = fn = 0
    ps, ys = [], []
    for r in rows:
        c, mat = cases[r["case_id"]], r["matrix"]
        cols = [k["name"] for k in c["columns"]]
        for o in c["outputs"]:
            sc = {col: pair_p(mat[col][o["name"]], T) for col in cols}
            src = set(o["sources"])
            if o["available"]:
                rk = sorted(cols, key=lambda x: (-sc[x], x))
                n += 1
                top1 += rk[0] in src
                exact += set(rk[:len(src)]) == src
                top3 += bool(set(rk[:3]) & src)
            else:
                nun += 1
                rej += max(sc.values()) < 0.5
            for col in cols:
                y, p = col in src, sc[col]
                ps.append(p)
                ys.append(y)
                tp += (p >= 0.5) and y
                fp += (p >= 0.5) and not y
                fn += (p < 0.5) and y
        for k in c["columns"]:
            if k["decoy"]:
                decoys += 1
                flag += any(pair_p(mat[k["name"]][o["name"]], T) >= 0.5 for o in c["outputs"])
    prec, rec = _div(tp, tp + fp), _div(tp, tp + fn)
    return {"cases": len(rows), "n_outputs": n, "n_unavailable": nun, "top1": _div(top1, n), "exact": _div(exact, n),
            "top3": _div(top3, n), "reject_unavailable": _div(rej, nun), "decoy_flag": _div(flag, decoys),
            "precision": prec, "recall": rec, "f1": _div(2 * prec * rec, prec + rec), "ece": ece(ps, ys),
            "brier": _div(sum((p - y) ** 2 for p, y in zip(ps, ys)), len(ps))}


def sweep(rows, cases, temps=TS) -> list[dict]:
    out = []
    for T in temps:
        m = mapping_metrics(rows, cases, T)
        out.append({"T": T, "ece": m["ece"], "brier": m["brier"], "f1": m["f1"], "precision": m["precision"],
                    "recall": m["recall"], "top1": m["top1"]})
    return out


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def lexical_mapping(cases) -> dict:
    """Model-free baseline: string similarity between input column name and output field name."""
    top1 = exact = top3 = n = 0
    for c in cases:
        cols = [k["name"] for k in c["columns"]]
        for o in c["outputs"]:
            if not o["available"]:
                continue
            sc = {col: SequenceMatcher(None, _norm(col), _norm(o["name"])).ratio() for col in cols}
            rk = sorted(cols, key=lambda x: (-sc[x], x))
            src = set(o["sources"])
            n += 1
            top1 += rk[0] in src
            exact += set(rk[:len(src)]) == src
            top3 += bool(set(rk[:3]) & src)
    return {"n_outputs": n, "top1": _div(top1, n), "exact": _div(exact, n), "top3": _div(top3, n)}


def kind_metrics(rows, cases) -> dict:
    ok = n = 0
    for r in rows:
        for o in cases[r["case_id"]]["outputs"]:
            if o["available"]:
                n += 1
                ok += r["kind"][o["name"]]["choice"] == KIND_LABEL[o["kind"]]
    return {"n": n, "acc": _div(ok, n)}


def transform_metrics(rows) -> dict:
    v = [x for r in rows for x in r["transform"].values()]
    n = len(v)
    f = lambda k: _div(sum(bool(k(x)) for x in v), n)          # noqa: E731
    return {"n": n, "choice": f(lambda x: x["choice_idx"] == x["truth_idx"]), "predicate": f(lambda x: x["pred_idx"] == x["truth_idx"]),
            "choice_value_ok": f(lambda x: x["choice_value_ok"]), "pred_value_ok": f(lambda x: x["pred_value_ok"])}


def permute_metrics(rows, part: str) -> dict:
    v = [x for r in rows for x in r[part].values()]
    n = len(v)
    npos = sum(len(x["pos"]) for x in v)
    return {"n": n, "consistent": _div(sum(len(set(x["picks"])) == 1 for x in v), n),
            "acc_over_orders": _div(sum(sum(x["correct"]) / len(x["correct"]) for x in v), n),
            "slot0": _div(sum(p == 0 for x in v for p in x["pos"]), npos)}


# ---------------------------------------------------------------- report
def _load(path):
    p = Path(path)
    return [json.loads(l) for l in p.read_text().splitlines()] if p.exists() else []


def _by_level(rows):
    g = defaultdict(list)
    for r in rows:
        g[r["level"]].append(r)
    return dict(sorted(g.items()))


def _pct(x):
    return f"{100 * x:.1f}"


def _table(header, rows):
    return "\n".join(["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
                     + ["| " + " | ".join(map(str, r)) + " |" for r in rows])


def build_report(res_dir, cases_by_id, conds) -> str:
    out = ["# Results: column mapping with Jev-Style-2B-Decision-v3 (Q8_0, local)", "",
           "All numbers are percentages unless noted. Levels: 0 identical, 1 style only, 2 abbreviation, 3 synonym, "
           "4 Indonesian, 5 legacy codes (OBJ_INFO_ADDR), 6 mixed + decoys. Small samples: read as indications.", ""]
    jev = Path(res_dir) / "jev"
    lex = defaultdict(list)
    for c in cases_by_id.values():
        lex[c["level"]].append(c)
    allmap = {cond: _load(jev / f"{cond}.jsonl") for cond in conds}
    levels = sorted({r["level"] for rows in allmap.values() for r in rows}) or sorted(lex)

    out += ["## Mapping: which input column feeds each output field", ""]
    for cond in conds:
        rows = allmap[cond]
        if not rows:
            continue
        out += [f"### Context condition `{cond}`", ""]
        trs = []
        for lv, rs in _by_level(rows).items():
            m = mapping_metrics(rs, cases_by_id)
            trs.append([lv, m["cases"], _pct(m["top1"]), _pct(m["exact"]), _pct(m["top3"]), _pct(m["reject_unavailable"]),
                        _pct(m["decoy_flag"]) if lv == 6 else "-", _pct(m["f1"]), f"{m['ece']:.3f}"])
        m = mapping_metrics(rows, cases_by_id)
        trs.append(["all", m["cases"], _pct(m["top1"]), _pct(m["exact"]), _pct(m["top3"]), _pct(m["reject_unavailable"]),
                    "-", _pct(m["f1"]), f"{m['ece']:.3f}"])
        out += [_table(["level", "cases", "top1", "exact", "top3", "reject n/a", "decoy flagged", "pair F1", "ECE"], trs), ""]

    out += ["### Model-free baseline (string similarity of names, no description or values)", ""]
    out += [_table(["level", "top1", "exact", "top3"], [[lv, *[_pct(lexical_mapping(lex[lv])[k]) for k in ("top1", "exact", "top3")]] for lv in levels]), ""]

    out += ["## Temperature: effect on calibration (pooled over levels)", "",
            "Jev probability = sigmoid((s_true - s_false) / T). T = 0.828 is the calibrated value. top1 does not change with T.", ""]
    for cond in conds:
        if allmap[cond]:
            out += [f"`{cond}`", "", _table(["T", "ECE", "Brier", "pair F1", "precision", "recall"],
                                          [[f"{s['T']:.3f}", f"{s['ece']:.3f}", f"{s['brier']:.3f}", _pct(s["f1"]), _pct(s["precision"]), _pct(s["recall"])]
                                           for s in sweep(allmap[cond], cases_by_id)]), ""]

    out += ["## Transformation kind (choice, given the true source columns)", ""]
    trs = []
    for cond in conds:
        for lv, rs in _by_level(allmap[cond]).items():
            k = kind_metrics(rs, cases_by_id)
            trs.append([cond, lv, k["n"], _pct(k["acc"])])
    out += [_table(["cond", "level", "n outputs", "accuracy"], trs), ""]

    out += ["## Picking the right expression among candidates (1 true + 3 distractors, chance 25%)", ""]
    trs = []
    for cond in conds:
        for lv, rs in _by_level(_load(jev / f"transform_{cond}.jsonl")).items():
            t = transform_metrics(rs)
            trs.append([cond, lv, t["n"], _pct(t["choice"]), _pct(t["predicate"]), _pct(t["choice_value_ok"]), _pct(t["pred_value_ok"])])
    out += [_table(["cond", "level", "n", "choice", "predicate", "choice value ok", "predicate value ok"], trs), ""]

    out += ["## Option-order stability (same question, shuffled option order)", ""]
    trs = []
    for cond in conds:
        rows = _load(jev / f"permute_{cond}.jsonl")
        for part in ("transform", "kind"):
            if rows:
                p = permute_metrics(rows, part)
                trs.append([cond, part, p["n"], _pct(p["consistent"]), _pct(p["acc_over_orders"]), _pct(p["slot0"])])
    out += [_table(["cond", "question", "n", "same pick in every order", "mean accuracy over orders", "picked slot 0"], trs), ""]
    return "\n".join(out)


if __name__ == "__main__":
    cfg = tomllib.loads((ROOT / "config.toml").read_text())["run"]
    res = ROOT / cfg["out_dir"]
    cases = {c["case_id"]: c for c in generate(load_template(), cfg["n_per_level"], cfg["seed"])}
    present = [c for c in cfg["conds"] if (res / "jev" / f"{c}.jsonl").exists()]
    md = build_report(res, cases, present)
    (res / "summary.md").write_text(md)
    print(md)
