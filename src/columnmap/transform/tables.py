from __future__ import annotations

from ..core.prompts import context_label as ctx
from ..core.tables import by_level, md_table, pct
from .metrics import transform_metrics


def transform_tables(store, conds, only=None) -> list[str]:
    trs = []
    for cond in conds:
        for lv, rs in by_level(store.rows("transform", cond, only)).items():
            t = transform_metrics(rs)
            trs.append([ctx(cond), lv, t["n"], pct(t["choice"]), pct(t["predicate"]), pct(t["choice_value_ok"]), pct(t["pred_value_ok"])])
    if not trs:
        return []
    return ["## Right expression among 4 candidates (chance 25%)", "",
            md_table(["Jev sees", "sample", "n", "choice", "yes/no per candidate", "choice value ok", "yes/no value ok"], trs), ""]
