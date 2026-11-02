from __future__ import annotations

from ..core.scoring import safe_div as _div


def permute_metrics(rows, part: str) -> dict:
    v = [x for r in rows for x in r[part].values()]
    n = len(v)
    npos = sum(len(x["pos"]) for x in v)
    return {"n": n, "consistent": _div(sum(len(set(x["picks"])) == 1 for x in v), n),
            "acc_over_orders": _div(sum(sum(x["correct"]) / len(x["correct"]) for x in v), n),
            "slot0": _div(sum(p == 0 for x in v for p in x["pos"]), npos)}
