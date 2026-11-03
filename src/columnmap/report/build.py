# Write everything a run produces: per-sample files, figures and summary.md

from __future__ import annotations

from pathlib import Path

from ..convert.honest import convert_fields, decisions
from ..convert.metrics import field_correct
from ..core.csvio import write_csv
from ..core.prompts import context_label as ctx
from ..core.scoring import T0, pair_p
from ..core.viz import setup_style
from ..dataset.export import sample_folder
from ..dataset.samples import SAMPLES, word
from ..mapping.figures import (
    fig_accuracy,
    fig_calibration,
    fig_drift,
    fig_heatmap,
    fig_outcomes,
)
from ..calibrate import evaluate, setup_key
from ..convert.auto import (
    REVIEW_COLUMNS,
    coverage_curve,
    field_decisions,
    review_rows,
    selective_metrics,
)
from ..convert.figures import fig_precision_coverage
from ..core.selective import choose_tau
from ..core.validate import field_valid
from ..core.calibration import Calibration
from ..core.tables import md_table, pct
from ..mapping.figures import fig_reliability
from .glance import at_a_glance
from .tables import build_tables

DECISION_COLUMNS = [
    "output",
    "chosen_input",
    "confidence",
    "calibrated_confidence",
    "auto_filled",
    "method",
    "expression",
    "p_expr",
    "truth_available",
    "truth_input",
    "field_correct",
]


def make_report(
    store,
    cases,
    letters,
    conds,
    run_dir,
    other=None,
    calibration: Calibration | None = None,
    target_precision: float = 0.99,
) -> list[Path]:
    run = Path(run_dir)
    out, figs, paths = run / "report", run / "report" / "figures", []
    labels = {SAMPLES[x][0]: f"{x}: {SAMPLES[x][1]}" for x in letters}
    letter = {SAMPLES[x][0]: x for x in letters}
    ids = set(labels)
    section = (
        "mapping"
        if any(store.rows("mapping", c, ids) for c in conds)
        else "mapping_pairs"
    )
    rows_by_cond = {c: store.rows(section, c, ids) for c in conds}
    rows_by_cond = {c: r for c, r in rows_by_cond.items() if r}
    honest = {
        c: {r["case_id"]: r for r in store.rows("honest", c, ids)} for c in rows_by_cond
    }
    font_ok = setup_style()

    auto_rows, curves, cutoffs = [], [], {}
    if rows_by_cond:
        paths += [
            fig_accuracy(rows_by_cond, cases, labels, figs / "accuracy.svg"),
            fig_outcomes(rows_by_cond, cases, labels, figs / "outcomes.svg"),
            fig_drift(rows_by_cond, cases, figs / "drift.svg"),
            fig_calibration(rows_by_cond, cases, figs / "calibration.svg"),
        ]
        meta = store.data["meta"]
        for cond, rows in rows_by_cond.items():
            entry, tau, items = None, None, []
            hrecs = honest[cond]
            if hrecs and calibration is not None and "quant" in meta:
                mode = next(iter(hrecs.values())).get("mode", "options")
                entry = calibration.get(
                    setup_key(meta, mode, cond, "field"), missing_ok=True
                )
                tau = (
                    choose_tau(entry["curve"], target_precision)
                    if entry and entry.get("curve")
                    else None
                )
            for r in rows:
                case, x = cases[r["case_id"]], letter[r["case_id"]]
                d, w = run / sample_folder(x), word(r["case_id"])
                outs = [o["name"] for o in case["outputs"]]
                paths += [
                    fig_heatmap(
                        r,
                        case,
                        labels[r["case_id"]],
                        cond,
                        figs / f"heatmap_{x}_{w}_{cond}.svg",
                    ),
                    write_csv(
                        d / f"probabilities_{w}_{cond}.csv",
                        ["input_column", *outs],
                        [
                            [c, *[f"{pair_p(r['matrix'][c][o], T0):.4f}" for o in outs]]
                            for c in r["matrix"]
                        ],
                    ),
                ]
                rec = hrecs.get(r["case_id"])
                if not rec:
                    continue
                _, full = convert_fields(rec["fields"], case["rows"], outs)
                valid = {
                    o["name"]: field_valid([row[j] for row in full], o["type"])
                    for j, o in enumerate(case["outputs"])
                }
                auto = field_decisions(rec["fields"], entry, tau, valid)
                ok = field_correct(case, full)
                header, conv = convert_fields(
                    rec["fields"],
                    case["rows"],
                    outs,
                    {f: v["auto"] for f, v in auto.items()},
                )
                dec = decisions(rec["fields"], case, ok, auto)
                m = selective_metrics(case, conv, auto, ok)
                has_src = {o["name"]: o["available"] for o in case["outputs"]}
                items += [
                    (auto[f]["conf_cal"], ok[f], has_src[f])
                    for f in outs
                    if rec["fields"][f]["best"] is not None
                ]
                auto_rows.append(
                    [
                        ctx(cond),
                        r["case_id"],
                        m["auto_total"],
                        m["auto_correct"],
                        pct(m["precision"]),
                        pct(m["coverage"]),
                        m["review"],
                    ]
                )
                paths += [
                    write_csv(d / f"converted_{w}_{cond}.csv", header, conv),
                    write_csv(
                        d / f"decisions_{w}_{cond}.csv",
                        DECISION_COLUMNS,
                        [[v[k] for k in DECISION_COLUMNS] for v in dec],
                    ),
                    write_csv(
                        d / f"review_{w}_{cond}.csv",
                        REVIEW_COLUMNS,
                        review_rows(rec["fields"], auto),
                    ),
                ]
            if items:
                pts = coverage_curve(items)
                chosen = next(
                    (
                        [p[0], p[1]]
                        for p in reversed(pts)
                        if tau is not None and p[2] >= tau
                    ),
                    None,
                )
                curves.append({"cond": cond, "points": pts, "chosen": chosen})
                cutoffs[cond] = (entry is not None, tau)
        if curves:
            paths.append(
                fig_precision_coverage(
                    curves, target_precision, figs / "precision_coverage.svg"
                )
            )
    auto_md = _auto_section(auto_rows, cutoffs, target_precision)
    calib_md = (
        _calibration_section(store, cases, rows_by_cond, ids, calibration, figs, paths)
        if rows_by_cond
        else []
    )
    legend = "\n".join(
        f"| {x} | data/samples/{sample_folder(x)}/ | {sample_folder(x)}/ | {SAMPLES[x][1]} |"
        for x in letters
    )
    names = [
        p.name
        for p in paths
        if p.suffix == ".svg" and not p.name.startswith(("heatmap", "reliability"))
    ]
    md = [
        "# Results",
        "",
        "Model: Jev-Style-2B-Decision-v3 (llama.cpp, local). Output fields are always given by name only. "
        "For each input column the model sees: **name only** = the header text; **name + values** = the header and "
        "5 values taken from the rows (file keys: `name`, `samples`).",
        "",
        "| sample | input, expected, ground truth | Jev outputs (this run) | input naming |",
        "|---|---|---|---|",
        legend,
        "",
        "Per sample and context: the converted file (made without ground truth), the decisions and the probability matrix. "
        "Raw model output for everything: `result.json`.",
        "",
        *at_a_glance(store, cases, list(rows_by_cond), ids, other),
        *(f"![{n[:-4]}](figures/{n})\n" for n in names),
        "Heatmaps: `figures/heatmap_<sample>_<context>.svg`.",
        "",
        *auto_md,
        *calib_md,
        build_tables(store, cases, list(rows_by_cond), ids, other, section),
    ]
    if not font_ok:
        md.insert(
            2, "_Times New Roman was not found; figures use a fallback serif font._\n"
        )
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.md").write_text("\n".join(md), encoding="utf-8")
    return paths + [out / "summary.md"]


# uncalibrated against calibrated reliability; says plainly when no calibration was fitted for this setup
def _calibration_section(
    store, cases, rows_by_cond, ids, calibration, figs, paths
) -> list[str]:
    meta = store.data["meta"]
    mode = (
        "options" if store.rows("mapping", next(iter(rows_by_cond)), ids) else "pairs"
    )
    evals = []
    if calibration is not None and "quant" in meta:
        for cond in rows_by_cond:
            evals += evaluate(store, meta, cases, calibration, mode, cond, ids)
    head = ["## Calibration", ""]
    if not evals:
        return head + [
            "No calibration fitted for this setup: probabilities are uncalibrated. Fit one with `--fit-calibration` on dev or "
            "calibration runs.",
            "",
        ]
    paths.append(fig_reliability(evals, figs / "reliability.svg"))
    trs = [
        [
            ctx(e["cond"]),
            e["level"],
            e["n"],
            f"{e['ece_raw']:.3f}",
            f"{e['ece_cal']:.3f}",
            "held-out" if e["held_out"] else "in-sample",
        ]
        for e in evals
    ]
    return head + [
        "![reliability](figures/reliability.svg)",
        "",
        "ECE is the average gap between predicted probability and observed frequency (0 is perfect). Held-out means this run was not used to fit.",
        "",
        md_table(
            [
                "Jev sees",
                "level",
                "points",
                "ECE uncalibrated",
                "ECE calibrated",
                "data",
            ],
            trs,
        ),
        "",
    ]


# what was filled automatically and how well; the cutoff comes from the calibration runs, never from this run
def _auto_section(auto_rows, cutoffs, target) -> list[str]:
    if not auto_rows:
        return []
    notes = []
    for cond, (calibrated, tau) in cutoffs.items():
        if not calibrated:
            notes.append(
                f"{ctx(cond)}: uncalibrated, so every answered field is filled and there is no precision guarantee."
            )
        elif tau is None:
            notes.append(
                f"{ctx(cond)}: no cutoff reaches {target:.0%} precision with enough fields, so nothing is filled automatically."
            )
        else:
            notes.append(
                f"{ctx(cond)}: fields are filled when the calibrated confidence is at least {tau:.4f} (target precision {target:.0%})."
            )
    return [
        "## Auto-Fill or Review",
        "",
        *[f"- {n}" for n in notes],
        "",
        "Fields below the cutoff stay empty in `converted_*.csv` and are listed with three candidates in `review_*.csv`. "
        "Precision counts a filled field as right only when every cell matches; a field without a source that was filled counts as wrong. "
        "Coverage is the share of fields with a source that were filled.",
        "",
        md_table(
            [
                "Jev sees",
                "sample",
                "fields filled",
                "filled correct",
                "precision %",
                "coverage %",
                "fields to review",
            ],
            auto_rows,
        ),
        "",
    ]
