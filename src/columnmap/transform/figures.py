# Transformation figure (SVG): picking the right expression among candidates

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from ..core.prompts import context_label as ctx
from ..core.viz import save
from .metrics import transform_metrics


# Left: choosing the right expression among 4 candidates. Right: naming the transformation kind
def fig_transform(transform_by_cond, labels, path):
    t_recs = []
    for cond, rows in transform_by_cond.items():
        for cid in labels:
            rs = [r for r in rows if r["case_id"] == cid]
            if rs:
                m = transform_metrics(rs)
                t_recs += [{"sample": labels[cid], "method": f"{ctx(cond)}: choose among 4", "acc": 100 * m["choice"]},
                           {"sample": labels[cid], "method": f"{ctx(cond)}: yes/no per candidate", "acc": 100 * m["predicate"]}]
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    sns.barplot(pd.DataFrame(t_recs), x="sample", y="acc", hue="method", ax=ax)
    ax.axhline(25, color="gray", linestyle="--", linewidth=1)
    ax.set(title="Right expression among 4 candidates (dashed: chance)", ylim=(0, 105), xlabel="", ylabel="accuracy (%)")
    ax.legend(title="", fontsize=7)
    return save(fig, path)
