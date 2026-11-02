from __future__ import annotations

from ..core.prompts import context_label as ctx
from ..core.tables import md_table, pct
from .metrics import permute_metrics


def stability_tables(store, conds, only=None) -> list[str]:
    trs = []
    for cond in conds:
        rows = store.rows("permute", cond, only)
        for part in ("transform", "kind"):
            if rows:
                p = permute_metrics(rows, part)
                trs.append([ctx(cond), part, p["n"], pct(p["consistent"]), pct(p["acc_over_orders"]), pct(p["slot0"])])
    if not trs:
        return []
    return ["## Option-order stability", "",
            md_table(["Jev sees", "question", "n", "same pick in every order", "mean accuracy over orders", "picked slot 0"], trs), ""]
