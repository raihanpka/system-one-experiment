"""Write everything a run produces: per-sample files, figures and summary.md.

run folder/
  result.json                       all raw Jev results (see core/store.py)
  template_header.csv               the empty target template
  <letter>_<word>/                  one folder per tested sample, e.g. c_legacy/
    input_<word>.csv                the messy input file
    expected_<word>.csv             what a perfect converter writes
    ground_truth_<word>.csv         true mapping and expression per output field
    converted_<word>_<context>.csv  the template filled from Jev's decisions
    decisions_<word>_<context>.csv  per output field: input column chosen, probability, kind, expression
    probabilities_<word>_<context>.csv  probability of every (input column, output field) pair
  report/summary.md, report/figures/*.svg
"""
from __future__ import annotations

from pathlib import Path

from ..convert.apply import convert_case
from ..core.csvio import write_csv
from ..core.scoring import T0, pair_p
from ..core.viz import setup_style
from ..dataset.export import write_case, write_template
from ..dataset.samples import SAMPLES, word
from ..mapping.figures import fig_accuracy, fig_calibration, fig_drift, fig_heatmap, fig_kind, fig_outcomes
from ..transform.figures import fig_transform
from .tables import build_tables

DECISION_COLUMNS = ["output", "available", "true_input", "predicted_input", "p", "runner_up", "p_runner_up",
                    "correct", "kind_predicted", "method", "expression"]


def sample_dir(run: Path, letter: str) -> Path:
    return run / f"{letter}_{word(SAMPLES[letter][0])}"


def make_report(store, cases, letters, conds, run_dir, template=None) -> list[Path]:
    run = Path(run_dir)
    out, figs, paths = run / "report", run / "report" / "figures", []
    labels = {SAMPLES[x][0]: f"{x}: {SAMPLES[x][1]}" for x in letters}
    letter = {SAMPLES[x][0]: x for x in letters}
    rows_by_cond = {c: store.rows("mapping", c, set(labels)) for c in conds}
    rows_by_cond = {c: r for c, r in rows_by_cond.items() if r}
    transform = {c: store.rows("transform", c, set(labels)) for c in rows_by_cond}
    font_ok = setup_style()

    if template:
        write_template(template, run)
    for x in letters:
        write_case(cases[SAMPLES[x][0]], sample_dir(run, x))
    if rows_by_cond:
        paths += [fig_accuracy(rows_by_cond, cases, labels, figs / "accuracy.svg"),
                  fig_outcomes(rows_by_cond, cases, labels, figs / "outcomes.svg"),
                  fig_drift(rows_by_cond, cases, figs / "drift.svg"),
                  fig_calibration(rows_by_cond, cases, figs / "calibration.svg"),
                  fig_kind(rows_by_cond, cases, labels, figs / "kind.svg")]
        if any(transform.values()):
            paths.append(fig_transform(transform, labels, figs / "transform.svg"))
        for cond, rows in rows_by_cond.items():
            tr = {r["case_id"]: r for r in transform[cond]}
            for r in rows:
                case, x = cases[r["case_id"]], letter[r["case_id"]]
                d, w = sample_dir(run, x), word(r["case_id"])
                header, conv, decisions = convert_case(r, tr.get(r["case_id"]), case)
                outs = [o["name"] for o in case["outputs"]]
                paths += [
                    fig_heatmap(r, case, labels[r["case_id"]], cond, figs / f"heatmap_{x}_{w}_{cond}.svg"),
                    write_csv(d / f"converted_{w}_{cond}.csv", header, conv),
                    write_csv(d / f"decisions_{w}_{cond}.csv", DECISION_COLUMNS, [[dc[k] for k in DECISION_COLUMNS] for dc in decisions]),
                    write_csv(d / f"probabilities_{w}_{cond}.csv", ["input_column", *outs],
                              [[c, *[f"{pair_p(r['matrix'][c][o], T0):.4f}" for o in outs]] for c in r["matrix"]])]
    legend = "\n".join(f"| {x} | {sample_dir(run, x).name}/ | {SAMPLES[x][1]} |" for x in letters)
    names = [p.name for p in paths if p.suffix == ".svg" and not p.name.startswith("heatmap")]
    md = ["# Results", "", "Model: Jev-Style-2B-Decision-v3 (llama.cpp, local). The model sees the input column name "
          "(and 5 sample values in context `samples`); output fields are given by name only.", "",
          "| sample | folder | input naming |", "|---|---|---|", legend, "",
          "Each sample folder holds the input, the expected file, the ground truth, and per context the converted file, "
          "the decisions and the probability matrix. Raw model output for everything: `result.json`.", "",
          *(f"![{n[:-4]}](figures/{n})\n" for n in names), "Heatmaps: `figures/heatmap_<sample>_<context>.svg`.", "",
          build_tables(store, cases, list(rows_by_cond), set(labels))]
    if not font_ok:
        md.insert(2, "_Times New Roman was not found; figures use a fallback serif font._\n")
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.md").write_text("\n".join(md), encoding="utf-8")
    return paths + [out / "summary.md"]
