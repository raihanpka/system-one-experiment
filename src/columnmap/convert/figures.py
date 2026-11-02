# precision against coverage: the more fields are filled automatically, the lower the precision tends to be

from __future__ import annotations

import matplotlib.pyplot as plt

from ..core.prompts import context_label as ctx
from ..core.viz import save


def fig_precision_coverage(curves: list[dict], target: float, path):
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    ax.axhline(target, color="gray", linestyle="--", label=f"target precision {target:.0%}")
    for c in curves:
        pts = c["points"]
        ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="o", markersize=3, label=ctx(c["cond"]))
        if c.get("chosen"):
            ax.plot([c["chosen"][0]], [c["chosen"][1]], marker="*", markersize=12, color="black", linestyle="none")
    ax.set(xlabel="coverage (share of fields with a source filled automatically)", ylabel="precision of the filled fields",
           xlim=(0, 1.02), ylim=(0, 1.02), title="Precision against coverage (star = cutoff in use)")
    ax.legend(fontsize=8, loc="lower left")
    return save(fig, path)
