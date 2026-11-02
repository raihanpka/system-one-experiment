# compare runs on the same files: top-1, fields correct without ground truth help, wrong fills, cells, tokens and seconds
# usage: uv run python scripts/compare_runs.py RUN [RUN ...] [--cond samples]

import sys

from columnmap.calibrate import load_run
from columnmap.convert.honest import convert_fields
from columnmap.convert.metrics import cell_match, field_causes, field_correct
from columnmap.core.tables import md_table, pct
from columnmap.mapping.metrics import mapping_metrics

args = sys.argv[1:]
cond = args[args.index("--cond") + 1] if "--cond" in args else "samples"
runs = [a for i, a in enumerate(args) if not a.startswith("--") and (i == 0 or args[i - 1] != "--cond")]
rows = []
for run in runs:
    store, meta, cases = load_run(run)
    section = next((sec for sec in ("mapping", "mapping_pairs", "mapping_twostage") if store.rows(sec, cond)), "mapping")
    m = mapping_metrics(store.rows(section, cond), cases)
    good = total = cells = expected = wrong = tokens = secs = 0
    for rec in store.rows("honest", cond):
        case = cases[rec["case_id"]]
        _, conv = convert_fields(rec["fields"], case["rows"], [o["name"] for o in case["outputs"]])
        c, s = field_causes(case, rec["fields"], field_correct(case, conv)), cell_match(case, conv)
        good, total, cells, expected, wrong = good + c["correct"], total + sum(c.values()), cells + s["correct"], expected + s["expected_cells"], wrong + s["wrong_fill"]
        tokens += rec.get("tokens", 0) + next((r.get("tokens", 0) for r in store.rows(section, cond) if r["case_id"] == rec["case_id"]), 0)
        secs += rec.get("seconds", 0) + next((r.get("seconds", 0) for r in store.rows(section, cond) if r["case_id"] == rec["case_id"]), 0)
    n = max(1, len(store.rows("honest", cond)))
    rows.append([run.rstrip("/").split("/")[-1][:34], cond, pct(m["top1"]), pct(m["exact"]), f"{good}/{total}", pct(cells / expected if expected else None), wrong,
                 f"{tokens / n / 1000:.0f}k", f"{secs / n:.0f}"])
print(md_table(["run", "context", "top-1", "exact", "fields correct", "cells correct", "wrong-fill cells", "tokens/file", "s/file"], rows))
