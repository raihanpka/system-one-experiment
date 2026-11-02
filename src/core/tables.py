"""Markdown table helpers shared by the slices."""
from __future__ import annotations

from collections import defaultdict


def by_level(rows) -> dict:
    g = defaultdict(list)
    for r in rows:
        g[r["level"]].append(r)
    return dict(sorted(g.items()))


def pct(x) -> str:
    return f"{100 * x:.1f}"


def md_table(header, rows) -> str:
    return "\n".join(["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
                     + ["| " + " | ".join(map(str, r)) + " |" for r in rows])
