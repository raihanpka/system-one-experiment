from __future__ import annotations

from collections import defaultdict

from ..core.prompts import context_label as ctx
from ..core.tables import by_level, md_table, pct
from .metrics import lexical_mapping, mapping_metrics, sweep


def mapping_tables(store, cases, conds, only=None, section="mapping") -> list[str]:
    ld = lambda cond: store.rows(section, cond, only)  # noqa: E731
    allmap = {c: ld(c) for c in conds}
    lex = defaultdict(list)
    for c in cases.values():
        if only is None or c["case_id"] in only:
            lex[c["case_id"].split("-")[0]].append(c)
    levels = [
        w
        for _, w in sorted(
            {
                (r["level"], r["case_id"].split("-")[0])
                for rows in allmap.values()
                for r in rows
            }
        )
    ] or sorted(lex)

    out = [
        f"## Mapping: Which Input Column Feeds Each Output Field ({'options' if section == 'mapping' else 'pairs'} mode)",
        "",
    ]
    for cond in conds:
        rows = allmap[cond]
        if not rows:
            continue
        trs = []
        for lv, rs in by_level(rows).items():
            m = mapping_metrics(rs, cases)
            trs.append(
                [
                    lv,
                    m["cases"],
                    pct(m["top1"]),
                    pct(m["exact"]),
                    pct(m["top3"]),
                    pct(m["reject_unavailable"]),
                    pct(m["decoy_flag"]),
                    pct(m["f1"]),
                    f"{m['ece']:.3f}",
                ]
            )
        m = mapping_metrics(rows, cases)
        trs.append(
            [
                "all",
                m["cases"],
                pct(m["top1"]),
                pct(m["exact"]),
                pct(m["top3"]),
                pct(m["reject_unavailable"]),
                "-",
                pct(m["f1"]),
                f"{m['ece']:.3f}",
            ]
        )
        out += [
            f"### Jev sees: {ctx(cond)}",
            "",
            md_table(
                [
                    "sample",
                    "cases",
                    "top1",
                    "exact",
                    "top3",
                    "reject n/a",
                    "decoy flagged",
                    "pair F1",
                    "ECE",
                ],
                trs,
            ),
            "",
        ]

    out += [
        "### Baseline without a model (string similarity of names only)",
        "",
        md_table(
            ["sample", "top1", "exact", "top3"],
            [
                [
                    lv,
                    *[
                        pct(lexical_mapping(lex[lv])[k])
                        for k in ("top1", "exact", "top3")
                    ],
                ]
                for lv in levels
                if lex[lv]
            ],
        ),
        "",
    ]

    out += [
        "## Temperature and calibration (pooled)",
        "",
        "Probability = sigmoid((s_true - s_false) / T). T = 0.828 is the calibrated value. top1 does not depend on T.",
        "",
    ]
    for cond in conds:
        if allmap[cond]:
            out += [
                f"Jev sees: {ctx(cond)}",
                "",
                md_table(
                    ["T", "ECE", "Brier", "pair F1", "precision", "recall"],
                    [
                        [
                            f"{s['T']:.3f}",
                            f"{s['ece']:.3f}",
                            f"{s['brier']:.3f}",
                            pct(s["f1"]),
                            pct(s["precision"]),
                            pct(s["recall"]),
                        ]
                        for s in sweep(allmap[cond], cases)
                    ],
                ),
                "",
            ]

    return out


# both scoring modes on the same files, side by side
def compare_tables(store, cases, conds, only=None) -> list[str]:
    trs = []
    for cond in conds:
        for section, name in (("mapping", "options"), ("mapping_pairs", "pairs")):
            rows = store.rows(section, cond, only)
            if not rows:
                continue
            m = mapping_metrics(rows, cases)
            secs = [r["seconds"] for r in rows if "seconds" in r]
            trs.append(
                [
                    ctx(cond),
                    name,
                    pct(m["top1"]),
                    pct(m["exact"]),
                    pct(m["reject_unavailable"]),
                    pct(m["f1"]),
                    f"{m['ece']:.3f}",
                    f"{sum(secs) / len(secs):.0f}" if secs else "-",
                ]
            )
    if len({r[0] for r in trs if r[1] == "pairs"}) == 0:
        return []
    return [
        "## Options Mode Versus Pairs Mode",
        "",
        md_table(
            [
                "Jev sees",
                "mode",
                "top1",
                "exact",
                "reject n/a (fixed 0.5)",
                "pair F1",
                "ECE",
                "seconds per file",
            ],
            trs,
        ),
        "",
    ]
