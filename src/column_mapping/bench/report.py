# Write everything a run produces: per-sample files and summary.md with x / y counts

from __future__ import annotations

from pathlib import Path

from ..mapper import Fill, convert, is_filled
from .csvio import write_csv
from .export import sample_folder
from .metrics import field_correct
from .samples import SAMPLES, word
from .summary import error_lines, sample_counts, table

DECISION_COLUMNS = [
    "output",
    "chosen_input",
    "roles",
    "method",
    "expression",
    "score",
    "p_present",
    "p_none",
    "filled",
    "truth_available",
    "truth_input",
    "field_correct",
]


# one row per field for decisions_*.csv; the truth_ columns come from the synthetic ground truth and exist only for evaluation
def decisions(fields: dict, case: dict, correct: dict, fill: Fill) -> list[dict]:
    out = []
    for o in case["outputs"]:
        r = fields[o["name"]]
        out.append(
            {
                "output": o["name"],
                "chosen_input": "|".join(r["cols"]),
                "roles": "|".join(
                    f"{phrase}<-{col or 'none'}" for phrase, col in r.get("roles", [])
                ),
                "method": r["method"],
                "expression": r["best_text"] or "",
                "score": "" if r["score"] is None else round(r["score"], 4),
                "p_present": (
                    "" if r.get("p_present") is None else round(r["p_present"], 4)
                ),
                "p_none": "" if r.get("p_none") is None else round(r["p_none"], 4),
                "filled": is_filled(r, fill),
                "truth_available": o["available"],
                "truth_input": "|".join(o["sources"]),
                "field_correct": correct[o["name"]],
            }
        )
    return out


def make_report(store, cases, letters, run_dir, fill: Fill) -> list[Path]:
    run = Path(run_dir)
    letter = {SAMPLES[x][0]: x for x in letters}
    paths, per_sample = [], []
    honest = {r["case_id"]: r for r in store.rows("honest", set(letter))}
    for r in store.rows("mapping", set(letter)):
        rec = honest.get(r["case_id"])
        if rec is None:
            continue
        case, x = cases[r["case_id"]], letter[r["case_id"]]
        d, w = run / sample_folder(x), word(r["case_id"])
        outs = [o["name"] for o in case["outputs"]]
        header, conv = convert(rec["fields"], case["rows"], outs, fill)
        ok = field_correct(case, conv)
        dec = decisions(rec["fields"], case, ok, fill)
        paths += [
            write_csv(d / f"converted_{w}.csv", header, conv),
            write_csv(
                d / f"decisions_{w}.csv",
                DECISION_COLUMNS,
                [[v[k] for k in DECISION_COLUMNS] for v in dec],
            ),
            write_csv(
                d / f"scores_{w}.csv",
                ["input_column", *outs],
                [[c, *[f"{r['matrix'][c][o]:.3f}" for o in outs]] for c in r["matrix"]],
            ),
        ]
        per_sample.append((x, sample_counts(case, r["matrix"], rec, fill)))
    meta = store.data["meta"]
    used = [*store.rows("mapping", set(letter)), *store.rows("honest", set(letter))]
    questions = sum(r.get("questions", 0) for r in used)
    tokens = sum(r.get("tokens", 0) for r in used)
    seconds = sum(r.get("seconds", 0) for r in used)
    md = [
        "# Results",
        "",
        f"Model: {meta['model']} (hosted, OpenRouter). Seed {meta.get('seed')}, split {meta.get('split')}. "
        "The model sees each input column's name and sample values. "
        "Each field description is read as named roles; each role is bound to one column, and code builds the "
        f"expression. A field is filled when its source is present: p_present at least {fill.copy} for a copied "
        f"field or {fill.derived} for a derived one, and p_none at most {fill.none_ceiling}. "
        "Every count is x / y against the ground truth. Column top-1 only checks the column choice; "
        "fields right checks that the converted column equals the expected column.",
        "",
        f"Model calls: {questions} questions over {len(per_sample)} files, {tokens:,} tokens, {seconds:.0f} s total.",
        "",
    ]
    if per_sample:
        md += [
            table(per_sample),
            "",
            "Fields still wrong:",
            "",
            *(error_lines(per_sample) or ["- none"]),
        ]
    out = run / "report" / "summary.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(md) + "\n", encoding="utf-8")
    return [*paths, out]
