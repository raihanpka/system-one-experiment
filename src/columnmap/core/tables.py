# Markdown table helpers shared by the slices

from __future__ import annotations



def by_level(rows) -> dict:
    g: dict = {}
    for r in sorted(rows, key=lambda r: r["level"]):
        g.setdefault(r["case_id"].split("-")[0], []).append(r)
    return g


def pct(x) -> str:
    return "-" if x is None else f"{100 * x:.1f}"


def md_table(header, rows) -> str:
    return "\n".join(["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
                     + ["| " + " | ".join(map(str, r)) + " |" for r in rows])
