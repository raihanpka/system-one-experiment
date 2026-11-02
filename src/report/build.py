"""Assemble figures, per-field predictions and summary.md from a result store."""
from __future__ import annotations

import csv
from pathlib import Path

from ..core.viz import setup_style
from ..dataset.samples import SAMPLES
from ..mapping.figures import fig_accuracy, fig_calibration, fig_drift, fig_heatmap, fig_kind, fig_outcomes
from ..mapping.predictions import predictions
from ..transform.figures import fig_transform
from .tables import build_tables


def make_report(store, cases, letters, conds, out_dir) -> list[Path]:
    out = Path(out_dir)
    labels = {SAMPLES[x][0]: f"{x}: {SAMPLES[x][1]}" for x in letters}
    letter = {SAMPLES[x][0]: x for x in letters}
    rows_by_cond = {c: store.rows("mapping", c, set(labels)) for c in conds}
    rows_by_cond = {c: r for c, r in rows_by_cond.items() if r}
    transform = {c: store.rows("transform", c, set(labels)) for c in rows_by_cond}
    font_ok = setup_style()
    figs, paths = out / "figures", []
    if rows_by_cond:
        paths += [fig_accuracy(rows_by_cond, cases, labels, figs / "accuracy.svg"),
                  fig_outcomes(rows_by_cond, cases, labels, figs / "outcomes.svg"),
                  fig_drift(rows_by_cond, cases, figs / "drift.svg"),
                  fig_calibration(rows_by_cond, cases, figs / "calibration.svg")]
        paths.append(fig_kind(rows_by_cond, cases, labels, figs / "kind.svg"))
        if any(transform.values()):
            paths.append(fig_transform(transform, labels, figs / "transform.svg"))
        for cond, rows in rows_by_cond.items():
            for r in rows:
                x = letter[r["case_id"]]
                paths.append(fig_heatmap(r, cases[r["case_id"]], labels[r["case_id"]], cond, figs / f"heatmap_{x}_{cond}.svg"))
                p = out / f"predictions_{x}_{cond}.csv"
                with p.open("w", newline="", encoding="utf-8") as f:
                    w = csv.writer(f)
                    w.writerow(["output", "available", "true_input", "predicted_input", "p_predicted", "runner_up", "p_runner_up", "correct"])
                    w.writerows(predictions(r, cases[r["case_id"]]))
                paths.append(p)
    legend = "\n".join(f"| {x} | {SAMPLES[x][0]} | {SAMPLES[x][1]} |" for x in letters)
    names = [p.name for p in paths if p.suffix == ".svg" and not p.name.startswith("heatmap")]
    md = ["# Results", "", "Model: Jev-Style-2B-Decision-v3 (llama.cpp, local). Input names are the only thing the model sees "
          "besides 5 sample values per column; output fields are given by name only.", "",
          "| sample | case | input naming |", "|---|---|---|", legend, "",
          *(f"![{n[:-4]}](figures/{n})\n" for n in names), "Heatmaps: `figures/heatmap_<sample>_<context>.svg`. "
          "Per-field predictions: `predictions_<sample>_<context>.csv`.", "",
          build_tables(store, cases, list(rows_by_cond), set(labels))]
    if not font_ok:
        md.insert(2, "_Times New Roman was not found; figures use a fallback serif font._\n")
    (out / "summary.md").write_text("\n".join(md), encoding="utf-8")
    return paths + [out / "summary.md"]
