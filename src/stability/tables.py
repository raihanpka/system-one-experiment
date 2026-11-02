from __future__ import annotations

from ..core.tables import md_table, pct
from .metrics import permute_metrics


def stability_tables(store, conds, only=None) -> list[str]:
    trs = []
    for cond in conds:
        rows = store.rows("permute", cond, only)
        for part in ("transform", "kind"):
            if rows:
                p = permute_metrics(rows, part)
                trs.append([cond, part, p["n"], pct(p["consistent"]), pct(p["acc_over_orders"]), pct(p["slot0"])])
    if not trs:
        return []
    return ["## Option-order stability", "",
            md_table(["context", "question", "n", "same pick in every order", "mean accuracy over orders", "picked slot 0"], trs), ""]
