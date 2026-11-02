from __future__ import annotations

from ..core.scoring import safe_div as _div


def transform_metrics(rows) -> dict:
    v = [x for r in rows for x in r["transform"].values()]
    n = len(v)
    f = lambda k: _div(sum(bool(k(x)) for x in v), n)          # noqa: E731
    return {"n": n, "choice": f(lambda x: x["choice_idx"] == x["truth_idx"]), "predicate": f(lambda x: x["pred_idx"] == x["truth_idx"]),
            "choice_value_ok": f(lambda x: x["choice_value_ok"]), "pred_value_ok": f(lambda x: x["pred_value_ok"])}
