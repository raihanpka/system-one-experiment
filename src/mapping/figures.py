"""Mapping figures (SVG): accuracy, outcome breakdown, drift regression, heatmap, calibration, kind accuracy."""
from __future__ import annotations

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from ..core.scoring import T0, TS, pair_p
from ..core.viz import save
from ..dataset.samples import LEVEL_NAMES
from .metrics import OUTCOMES, kind_metrics, lexical_mapping, mapping_metrics, outcomes, pair_scores


def fig_accuracy(rows_by_cond, cases, labels, path):
    """Top-1 mapping accuracy per sample: Jev conditions vs a name-similarity baseline."""
    recs = []
    ids = list(labels)
    for cid in ids:
        recs.append({"sample": labels[cid], "method": "Name similarity (no model)",
                     "top1": 100 * lexical_mapping([cases[cid]])["top1"]})
        for cond, rows in rows_by_cond.items():
            rs = [r for r in rows if r["case_id"] == cid]
            if rs:
                recs.append({"sample": labels[cid], "method": f"Jev, {cond}", "top1": 100 * mapping_metrics(rs, cases)["top1"]})
    fig, ax = plt.subplots(figsize=(7, 3.6))
    sns.barplot(pd.DataFrame(recs), x="sample", y="top1", hue="method", ax=ax)
    ax.set(ylim=(0, 105), xlabel="", ylabel="Top-1 accuracy (%)", title="Which input column feeds each output field")
    ax.legend(title="", loc="lower left", frameon=True)
    return save(fig, path)


def _order(case):
    """Input columns: true sources in output order first, decoys and unused last."""
    out, seen = [], set()
    for o in case["outputs"]:
        for n in o["sources"]:
            if n not in seen:
                seen.add(n)
                out.append(n)
    rest = [k["name"] for k in case["columns"] if k["name"] not in seen]
    return out + rest


def fig_heatmap(row, case, label, cond, path):
    """P(input column feeds output field) for one file. Black dots mark the true pairs."""
    cols = _order(case)
    outs = [o["name"] for o in case["outputs"]]
    data = pd.DataFrame([[pair_p(row["matrix"][c][o], T0) for o in outs] for c in cols], index=cols,
                        columns=[o + ("" if a else " (n/a)") for o, a in zip(outs, (o["available"] for o in case["outputs"]))])
    fig, ax = plt.subplots(figsize=(11, 0.22 * len(cols) + 2.6))
    sns.heatmap(data, vmin=0, vmax=1, cmap="viridis", ax=ax, cbar_kws={"label": "probability", "shrink": 0.6},
                linewidths=0.2, linecolor="white")
    for i, c in enumerate(cols):
        for j, o in enumerate(case["outputs"]):
            if c in o["sources"]:
                ax.plot(j + 0.5, i + 0.5, "o", color="black", markersize=3.5)
    ax.tick_params(axis="both", labelsize=6)
    ax.set_title(f"{label}, {cond}: probability per input column and output field (dot = true pair)")
    ax.set(xlabel="output field (template)", ylabel="input column (file)")
    plt.setp(ax.get_xticklabels(), rotation=90)
    return save(fig, path)


def fig_calibration(rows_by_cond, cases, path):
    """Left: ECE against temperature. Right: reliability at the calibrated temperature."""
    fig, (a, b) = plt.subplots(1, 2, figsize=(9, 3.6))
    recs, rel = [], []
    for cond, rows in rows_by_cond.items():
        if not rows:
            continue
        for T in TS:
            recs.append({"T": T, "ECE": mapping_metrics(rows, cases, T)["ece"], "condition": cond})
        ps, ys = pair_scores(rows, cases, T0)
        d = pd.DataFrame({"p": ps, "y": ys})
        d["bin"] = (d.p * 10).clip(upper=9.999).astype(int)
        g = d.groupby("bin").agg(conf=("p", "mean"), acc=("y", "mean"), n=("y", "size")).reset_index()
        g["condition"] = cond
        rel.append(g)
    if recs:
        sns.lineplot(pd.DataFrame(recs), x="T", y="ECE", hue="condition", marker="o", ax=a)
        a.axvline(T0, color="gray", linestyle=":")
        a.set(title="Calibration error vs temperature", xlabel="temperature T (dotted: calibrated 0.828)")
    if rel:
        sns.lineplot(pd.concat(rel), x="conf", y="acc", hue="condition", marker="o", ax=b)
        b.plot([0, 1], [0, 1], color="gray", linestyle="--")
        b.set(title="Reliability at T = 0.828", xlabel="mean predicted probability", ylabel="observed share true",
              xlim=(0, 1), ylim=(0, 1.02))
    return save(fig, path)


def fig_outcomes(rows_by_cond, cases, labels, path):
    """Stacked bars: what happened to every output field of a file, one panel per context."""
    conds = [c for c, r in rows_by_cond.items() if r]
    fig, axes = plt.subplots(1, len(conds), figsize=(4.4 * len(conds) + 2.2, 3.8), sharey=True, squeeze=False)
    colors = ["#2a9d8f", "#e76f51", "#a8c686", "#e9c46a"]
    for ax, cond in zip(axes[0], conds):
        recs = {}
        for cid, lab in labels.items():
            rs = [r for r in rows_by_cond[cond] if r["case_id"] == cid]
            if rs:
                recs[lab] = outcomes(rs, cases)
        df = pd.DataFrame(recs).T[OUTCOMES]
        df = df.div(df.sum(axis=1), axis=0) * 100
        df.plot(kind="bar", stacked=True, ax=ax, color=colors, width=0.7, legend=False)
        ax.set(title=f"context: {cond}", xlabel="", ylabel="share of output fields (%)")
        ax.tick_params(axis="x", rotation=0, labelsize=8)
    h, lab = axes[0][0].get_legend_handles_labels()
    fig.legend(h, lab, loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=False)
    return save(fig, path)


def fig_drift(rows_by_cond, cases, path):
    """Top-1 against name drift, one point per file, with a fitted line per method (slope = points lost per level)."""
    import numpy as np
    recs = []
    for cond, rows in rows_by_cond.items():
        for r in rows:
            recs.append({"level": r["level"], "top1": 100 * mapping_metrics([r], cases)["top1"], "method": f"Jev, {cond}"})
    ids = {r["case_id"] for rows in rows_by_cond.values() for r in rows}
    for cid in ids:
        recs.append({"level": cases[cid]["level"], "top1": 100 * lexical_mapping([cases[cid]])["top1"], "method": "Name similarity (no model)"})
    df = pd.DataFrame(recs)
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    for (method, g), color in zip(df.groupby("method"), sns.color_palette("colorblind")):
        slope = ""
        if g.level.nunique() >= 3:
            sns.regplot(g, x="level", y="top1", ci=None, scatter=False, color=color, ax=ax, line_kws={"linewidth": 1.4})
            slope = f" (slope {np.polyfit(g.level, g.top1, 1)[0]:+.1f}/level)"
        ax.scatter(g.level, g.top1, color=color, s=34, label=method + slope, zorder=3)
    lv = sorted(df.level.unique())
    ax.set(xticks=lv, xticklabels=[f"{i}\n{LEVEL_NAMES[i]}" for i in lv], ylim=(-3, 105), xlabel="name drift level",
           ylabel="top-1 accuracy (%)", title="Accuracy as input names drift away from the template")
    ax.tick_params(axis="x", labelsize=8)
    ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=1)
    return save(fig, path)


def fig_kind(kind_by_cond, cases, labels, path):
    """Naming the transformation kind, given the true source columns."""
    recs = []
    for cond, rows in kind_by_cond.items():
        for cid in labels:
            rs = [r for r in rows if r["case_id"] == cid]
            if rs:
                recs.append({"sample": labels[cid], "context": cond, "acc": 100 * kind_metrics(rs, cases)["acc"]})
    fig, ax = plt.subplots(figsize=(6, 3.4))
    sns.barplot(pd.DataFrame(recs), x="sample", y="acc", hue="context", ax=ax)
    ax.axhline(20, color="gray", linestyle="--", linewidth=1)
    ax.set(title="Transformation kind (dashed: chance)", ylim=(0, 105), xlabel="", ylabel="accuracy (%)")
    ax.legend(title="", fontsize=8)
    return save(fig, path)
