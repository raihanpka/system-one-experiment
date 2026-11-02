from __future__ import annotations

from collections import defaultdict

from ..core.tables import by_level, md_table, pct
from .metrics import kind_metrics, lexical_mapping, mapping_metrics, sweep


def mapping_tables(store, cases, conds, only=None) -> list[str]:
    ld = lambda cond: store.rows("mapping", cond, only)                           # noqa: E731
    allmap = {c: ld(c) for c in conds}
    lex = defaultdict(list)
    for c in cases.values():
        if only is None or c["case_id"] in only:
            lex[c["level"]].append(c)
    levels = sorted({r["level"] for rows in allmap.values() for r in rows}) or sorted(lex)

    out = ["## Mapping: which input column feeds each output field", ""]
    for cond in conds:
        rows = allmap[cond]
        if not rows:
            continue
        trs = []
        for lv, rs in by_level(rows).items():
            m = mapping_metrics(rs, cases)
            trs.append([lv, m["cases"], pct(m["top1"]), pct(m["exact"]), pct(m["top3"]), pct(m["reject_unavailable"]),
                        pct(m["decoy_flag"]) if lv == 6 else "-", pct(m["f1"]), f"{m['ece']:.3f}"])
        m = mapping_metrics(rows, cases)
        trs.append(["all", m["cases"], pct(m["top1"]), pct(m["exact"]), pct(m["top3"]), pct(m["reject_unavailable"]),
                    "-", pct(m["f1"]), f"{m['ece']:.3f}"])
        out += [f"### Context `{cond}`", "",
                md_table(["level", "cases", "top1", "exact", "top3", "reject n/a", "decoy flagged", "pair F1", "ECE"], trs), ""]

    out += ["### Baseline without a model (string similarity of names only)", "",
            md_table(["level", "top1", "exact", "top3"],
                     [[lv, *[pct(lexical_mapping(lex[lv])[k]) for k in ("top1", "exact", "top3")]] for lv in levels if lex[lv]]), ""]

    out += ["## Temperature and calibration (pooled)", "",
            "Probability = sigmoid((s_true - s_false) / T). T = 0.828 is the calibrated value. top1 does not depend on T.", ""]
    for cond in conds:
        if allmap[cond]:
            out += [f"`{cond}`", "", md_table(["T", "ECE", "Brier", "pair F1", "precision", "recall"],
                                              [[f"{s['T']:.3f}", f"{s['ece']:.3f}", f"{s['brier']:.3f}", pct(s["f1"]), pct(s["precision"]), pct(s["recall"])]
                                               for s in sweep(allmap[cond], cases)]), ""]

    trs = [[cond, lv, (k := kind_metrics(rs, cases))["n"], pct(k["acc"])]
           for cond in conds for lv, rs in by_level(allmap[cond]).items()]
    return out + ["## Transformation kind (given the true source columns)", "",
                  md_table(["context", "level", "n outputs", "accuracy"], trs), ""]
