from __future__ import annotations

from ..core.scoring import T0, pair_p


def predictions(row, case):
    """Per output field: true source(s), predicted input column, probability, runner-up."""
    cols = [k["name"] for k in case["columns"]]
    out = []
    for o in case["outputs"]:
        sc = sorted(((pair_p(row["matrix"][c][o["name"]], T0), c) for c in cols), reverse=True)
        (p1, c1), (p2, c2) = sc[0], sc[1]
        ok = c1 in o["sources"] if o["available"] else p1 < 0.5
        out.append([o["name"], o["available"], "|".join(o["sources"]), c1, f"{p1:.4f}", c2, f"{p2:.4f}", ok])
    return out
