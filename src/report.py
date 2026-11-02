"""Samples, report files and figures from a result store."""
from __future__ import annotations

import csv
from pathlib import Path

from .metrics import T0, build_tables, pair_p
from . import plots

SAMPLES = {"a": ("L0-000", "identical names"), "b": ("L3-000", "synonyms"), "c": ("L5-000", "legacy codes"),
           "d": ("L6-000", "mixed, decoys"), "e": ("L1-000", "style only"), "f": ("L2-000", "abbreviations"),
           "g": ("L4-000", "Indonesian")}
DEFAULT = "abcd"


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


def make_report(store, cases, letters, conds, out_dir) -> list[Path]:
    out = Path(out_dir)
    labels = {SAMPLES[x][0]: f"{x}: {SAMPLES[x][1]}" for x in letters}
    letter = {SAMPLES[x][0]: x for x in letters}
    rows_by_cond = {c: store.rows("mapping", c, set(labels)) for c in conds}
    rows_by_cond = {c: r for c, r in rows_by_cond.items() if r}
    transform = {c: store.rows("transform", c, set(labels)) for c in rows_by_cond}
    font_ok = plots.setup_style()
    figs, paths = out / "figures", []
    if rows_by_cond:
        paths += [plots.fig_accuracy(rows_by_cond, cases, labels, figs / "accuracy.svg"),
                  plots.fig_outcomes(rows_by_cond, cases, labels, figs / "outcomes.svg"),
                  plots.fig_drift(rows_by_cond, cases, figs / "drift.svg"),
                  plots.fig_calibration(rows_by_cond, cases, figs / "calibration.svg")]
        if any(transform.values()):
            paths.append(plots.fig_transform(transform, rows_by_cond, cases, labels, figs / "transform.svg"))
        for cond, rows in rows_by_cond.items():
            for r in rows:
                x = letter[r["case_id"]]
                paths.append(plots.fig_heatmap(r, cases[r["case_id"]], labels[r["case_id"]], cond, figs / f"heatmap_{x}_{cond}.svg"))
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
