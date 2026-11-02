# Mapping metrics: for each available output, input columns are ranked by P(true)

from __future__ import annotations

import re
from difflib import SequenceMatcher

from ..core.scoring import T0, TS, ece, pair_p, rate, safe_div as _div
from ..core.template import KIND_LABEL

OUTCOMES = ["Correct column", "Wrong column", "Left empty (no source)", "False match (no source)"]


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
            "top3": _div(top3, n), "reject_unavailable": rate(rej, nun), "decoy_flag": rate(flag, decoys),
            "precision": prec, "recall": rec, "f1": _div(2 * prec * rec, prec + rec), "ece": ece(ps, ys),
            "brier": _div(sum((p - y) ** 2 for p, y in zip(ps, ys)), len(ps))}


def sweep(rows, cases, temps=TS) -> list[dict]:
    out = []
    for T in temps:
        m = mapping_metrics(rows, cases, T)
        out.append({"T": T, "ece": m["ece"], "brier": m["brier"], "f1": m["f1"], "precision": m["precision"],
                    "recall": m["recall"], "top1": m["top1"]})
    return out


# Probability and truth of every (input column, output field) pair
def pair_scores(rows, cases, T: float = T0):
    ps, ys = [], []
    for r in rows:
        c = cases[r["case_id"]]
        for col in r["matrix"]:
            for o in c["outputs"]:
                ps.append(pair_p(r["matrix"][col][o["name"]], T))
                ys.append(col in o["sources"])
    return ps, ys


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


# Model-free baseline: string similarity between input column name and output field name
def lexical_mapping(cases) -> dict:
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


def outcomes(rs, cases) -> dict:
    cnt = dict.fromkeys(OUTCOMES, 0)
    for r in rs:
        c = cases[r["case_id"]]
        cols = [k["name"] for k in c["columns"]]
        for o in c["outputs"]:
            sc = {col: pair_p(r["matrix"][col][o["name"]], T0) for col in cols}
            if o["available"]:
                best = max(cols, key=lambda x: (sc[x], x))
                cnt[OUTCOMES[0] if best in o["sources"] else OUTCOMES[1]] += 1
            else:
                cnt[OUTCOMES[2] if max(sc.values()) < 0.5 else OUTCOMES[3]] += 1
    return cnt
