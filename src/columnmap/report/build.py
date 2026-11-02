# Write everything a run produces: per-sample files, figures and summary.md

from __future__ import annotations

from pathlib import Path

from ..convert.honest import convert_fields, decisions
from ..convert.metrics import field_correct
from ..core.csvio import write_csv
from ..core.scoring import T0, pair_p
from ..core.viz import setup_style
from ..dataset.export import sample_folder
from ..dataset.samples import SAMPLES, word
from ..mapping.figures import fig_accuracy, fig_calibration, fig_drift, fig_heatmap, fig_outcomes
from .glance import at_a_glance
from .tables import build_tables

DECISION_COLUMNS = ["output", "chosen_input", "confidence", "method", "expression", "p_expr", "truth_available", "truth_input", "field_correct"]


def make_report(store, cases, letters, conds, run_dir, other=None) -> list[Path]:
    run = Path(run_dir)
    out, figs, paths = run / "report", run / "report" / "figures", []
    labels = {SAMPLES[x][0]: f"{x}: {SAMPLES[x][1]}" for x in letters}
    letter = {SAMPLES[x][0]: x for x in letters}
    ids = set(labels)
    section = "mapping" if any(store.rows("mapping", c, ids) for c in conds) else "mapping_pairs"
    rows_by_cond = {c: store.rows(section, c, ids) for c in conds}
    rows_by_cond = {c: r for c, r in rows_by_cond.items() if r}
    honest = {c: {r["case_id"]: r for r in store.rows("honest", c, ids)} for c in rows_by_cond}
    font_ok = setup_style()

    if rows_by_cond:
        paths += [fig_accuracy(rows_by_cond, cases, labels, figs / "accuracy.svg"),
                  fig_outcomes(rows_by_cond, cases, labels, figs / "outcomes.svg"),
                  fig_drift(rows_by_cond, cases, figs / "drift.svg"),
                  fig_calibration(rows_by_cond, cases, figs / "calibration.svg")]
        for cond, rows in rows_by_cond.items():
            for r in rows:
                case, x = cases[r["case_id"]], letter[r["case_id"]]
                d, w = run / sample_folder(x), word(r["case_id"])
                outs = [o["name"] for o in case["outputs"]]
                paths += [fig_heatmap(r, case, labels[r["case_id"]], cond, figs / f"heatmap_{x}_{w}_{cond}.svg"),
                          write_csv(d / f"probabilities_{w}_{cond}.csv", ["input_column", *outs],
                                    [[c, *[f"{pair_p(r['matrix'][c][o], T0):.4f}" for o in outs]] for c in r["matrix"]])]
                rec = honest[cond].get(r["case_id"])
                if rec:
                    header, conv = convert_fields(rec["fields"], case["rows"], outs)
                    dec = decisions(rec["fields"], case, field_correct(case, conv))
                    paths += [write_csv(d / f"converted_{w}_{cond}.csv", header, conv),
                              write_csv(d / f"decisions_{w}_{cond}.csv", DECISION_COLUMNS, [[x_[k] for k in DECISION_COLUMNS] for x_ in dec])]
    legend = "\n".join(f"| {x} | data/samples/{sample_folder(x)}/ | {sample_folder(x)}/ | {SAMPLES[x][1]} |" for x in letters)
    names = [p.name for p in paths if p.suffix == ".svg" and not p.name.startswith("heatmap")]
    md = ["# Results", "", "Model: Jev-Style-2B-Decision-v3 (llama.cpp, local). Output fields are always given by name only. "
          "For each input column the model sees: **name only** = the header text; **name + values** = the header and "
          "5 values taken from the rows (file keys: `name`, `samples`).", "",
          "| sample | input, expected, ground truth | Jev outputs (this run) | input naming |", "|---|---|---|---|", legend, "",
          "Per sample and context: the converted file (made without ground truth), the decisions and the probability matrix. "
          "Raw model output for everything: `result.json`.", "",
          *at_a_glance(store, cases, list(rows_by_cond), ids, other),
          *(f"![{n[:-4]}](figures/{n})\n" for n in names), "Heatmaps: `figures/heatmap_<sample>_<context>.svg`.", "",
          build_tables(store, cases, list(rows_by_cond), ids, other, section)]
    if not font_ok:
        md.insert(2, "_Times New Roman was not found; figures use a fallback serif font._\n")
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.md").write_text("\n".join(md), encoding="utf-8")
    return paths + [out / "summary.md"]
