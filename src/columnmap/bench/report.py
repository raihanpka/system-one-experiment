# Write everything a run produces: per-sample files and summary.md with x / y counts

from __future__ import annotations

from pathlib import Path

from ..mapper import convert, is_filled
from .csvio import write_csv
from .export import sample_folder
from .metrics import field_correct
from .samples import SAMPLES, word
from .summary import error_lines, sample_counts, table

DECISION_COLUMNS = [
    "output",
    "chosen_input",
    "method",
    "expression",
    "score",
    "filled",
    "truth_available",
    "truth_input",
    "field_correct",
]


# one row per field for decisions_*.csv; the truth_ columns come from the synthetic ground truth and exist only for evaluation
def decisions(fields: dict, case: dict, correct: dict, fill_score: float) -> list[dict]:
    out = []
    for o in case["outputs"]:
        r = fields[o["name"]]
        out.append(
            {
                "output": o["name"],
                "chosen_input": "|".join(r["cols"]),
                "method": r["method"],
                "expression": r["best_text"] or "",
                "score": "" if r["score"] is None else round(r["score"], 4),
                "filled": is_filled(r, fill_score),
                "truth_available": o["available"],
                "truth_input": "|".join(o["sources"]),
                "field_correct": correct[o["name"]],
            }
        )
    return out


def make_report(store, cases, letters, run_dir, fill_score: float) -> list[Path]:
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
        header, conv = convert(rec["fields"], case["rows"], outs, fill_score)
        ok = field_correct(case, conv)
        dec = decisions(rec["fields"], case, ok, fill_score)
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
        per_sample.append((x, sample_counts(case, r["matrix"], rec, fill_score)))
    meta = store.data["meta"]
    md = [
        "# Results",
        "",
        f"Model: {meta['model']} (hosted, OpenRouter). Seed {meta.get('seed')}, split {meta.get('split')}. "
        "The model sees each input column's name and sample values. "
        f"A field is filled when its best expression scores at least {fill_score}. "
        "Every count is x / y against the ground truth. Column top-1 only checks the column choice; "
        "fields right checks that the converted column equals the expected column.",
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
