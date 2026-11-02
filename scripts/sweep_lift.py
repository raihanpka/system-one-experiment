# sweep the lift rule on the dev split: honest stage only, mapping scores reused; writes results/sweep/summary.md
# usage: uv run python scripts/sweep_lift.py <dev run folder>

import json
import shutil
import sys
from pathlib import Path

from columnmap.calibrate import load_run
from columnmap.convert.honest import convert_fields
from columnmap.convert.metrics import cell_match, field_causes, field_correct
from columnmap.core.paths import ROOT
from columnmap.main import main

TOPS = (0.5, 1.0, 2.0, 4.0)
SECONDS = (1.0, 2.0, 4.0)


def score(run: Path) -> dict:
    store, meta, cases = load_run(run)
    fields_ok = fields_src = wrong_fill = expr = cells_ok = cells_exp = 0
    for rec in store.rows("honest", "samples"):
        case = cases[rec["case_id"]]
        _, conv = convert_fields(rec["fields"], case["rows"], [o["name"] for o in case["outputs"]])
        ok = field_correct(case, conv)
        c = field_causes(case, rec["fields"], ok)
        fields_ok += c["correct"]
        fields_src += sum(c.values())
        expr += c["wrong_expression"]
        s = cell_match(case, conv)
        wrong_fill += s["wrong_fill"]
        cells_ok += s["correct"]
        cells_exp += s["expected_cells"]
    return {"fields_correct": fields_ok, "fields_with_source": fields_src, "wrong_expression": expr,
            "wrong_filled_cells": wrong_fill, "cells_correct_pct": round(100 * cells_ok / cells_exp, 1) if cells_exp else None}


def main_sweep(dev_run: Path) -> None:
    out_root = ROOT / "results" / "sweep"
    rows = []
    for top in TOPS:
        for second in SECONDS:
            tag = f"lift_{top}_{second}"
            folder = out_root / tag
            if folder.exists():
                shutil.rmtree(folder)
            folder.mkdir(parents=True)
            meta = json.loads((dev_run / "result.json").read_text())
            meta.pop("honest", None)
            meta["meta"].pop("keep_d", None)
            (folder / "result.json").write_text(json.dumps(meta))
            main(["--out", str(folder), "--samples", "abcd", "--conds", "samples", "--lift", f"{top},{second}"])
            rows.append((top, second, score(folder)))
    lines = ["# Lift sweep on dev (seed 11, four samples, name plus values)", "",
             "| lift_top | lift_second | fields correct | fields with source | wrong expression | wrong filled cells | cells correct % |",
             "|---|---|---|---|---|---|---|"]
    for top, second, s in rows:
        lines.append(f"| {top} | {second} | {s['fields_correct']} | {s['fields_with_source']} | {s['wrong_expression']} | "
                     f"{s['wrong_filled_cells']} | {s['cells_correct_pct']} |")
    (out_root / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main_sweep(Path(sys.argv[1]))
