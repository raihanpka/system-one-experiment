# deciding when a field is mapped: the best input column is kept only if its score clears a cutoff fitted on ground truth

from __future__ import annotations

import statistics

from ..core.scoring import rate, safe_div


# per output field: best score, how far it stands above the median column (lift), and whether the field has a source
def field_scores(rows, cases) -> list[dict]:
    out = []
    for r in rows:
        c = cases[r["case_id"]]
        decoys = {k["name"] for k in c["columns"] if k["decoy"]}
        for o in c["outputs"]:
            sc = sorted(((r["matrix"][col][o["name"]]["d"], col) for col in r["matrix"]), reverse=True)
            d1, c1 = sc[0]
            median = statistics.median(d for d, _ in sc)
            out.append({"case_id": r["case_id"], "output": o["name"], "has_source": o["available"], "d1": d1, "lift": d1 - median,
                        "top": c1, "top_ok": c1 in o["sources"], "top_is_decoy": c1 in decoys})
    return out


# the cutoff that best separates fields with a source (high value) from fields without one (low value)
def fit_cut(values, labels) -> float | None:
    if len({bool(y) for y in labels}) < 2:
        return None
    xs = sorted(set(values))
    cands = [xs[0] - 1.0] + [(a + b) / 2 for a, b in zip(xs, xs[1:])] + [xs[-1] + 1.0]
    pos = [v for v, y in zip(values, labels) if y]
    neg = [v for v, y in zip(values, labels) if not y]
    best, best_bal = cands[0], -1.0
    for c in cands:
        bal = (safe_div(sum(v >= c for v in pos), len(pos)) + safe_div(sum(v < c for v in neg), len(neg))) / 2
        if bal > best_bal:
            best, best_bal = c, bal
    return best


def fit_cuts(rows, cases) -> dict:
    feats = field_scores(rows, cases)
    ys = [f["has_source"] for f in feats]
    return {"d1": fit_cut([f["d1"] for f in feats], ys), "lift": fit_cut([f["lift"] for f in feats], ys)}


def reject_metrics(feats, accept) -> dict:
    kept = [accept(f["d1"], f["lift"]) for f in feats]
    with_src = [(f, k) for f, k in zip(feats, kept) if f["has_source"]]
    without = [(f, k) for f, k in zip(feats, kept) if not f["has_source"]]
    rec = rate(sum(k for _, k in with_src), len(with_src))
    rej = rate(sum(not k for _, k in without), len(without))
    kept_src = [f for f, k in with_src if k]
    known = [x for x in (rec, rej) if x is not None]
    return {"fields": len(feats), "kept": rec, "rejected": rej, "balanced": sum(known) / len(known) if known else None,
            "top1_when_kept": rate(sum(f["top_ok"] for f in kept_src), len(kept_src)),
            "decoys_picked": sum(k and f["top_is_decoy"] for f, k in zip(feats, kept))}
