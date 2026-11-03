# test cases from a finished run, one per sample and context: the model's top column per output field against the true sources
# both sides are JSON strings so the DeepEval metrics read them back; no DeepEval import here

from __future__ import annotations

import json

from ..calibrate import SECTIONS, load_run
from ..core.scoring import T0, pair_p

FILL_P = 0.5  # a field is filled when its top column has p >= 0.5, the same line the mapping metrics use


def cases_from_run(run_dir, mode: str = "options") -> list[dict]:
    store, meta, cases = load_run(run_dir)
    out = []
    for cond in meta["contexts"]:
        for row in store.rows(SECTIONS[mode], cond):
            case = cases[row["case_id"]]
            cols = [k["name"] for k in case["columns"]]
            actual, expected = {}, {}
            for o in case["outputs"]:
                sc = {c: pair_p(row["matrix"][c][o["name"]], T0) for c in cols}
                top = min(cols, key=lambda c: (-sc[c], c))
                actual[o["name"]] = {"column": top, "p": round(sc[top], 4)}
                expected[o["name"]] = sorted(o["sources"]) if o["available"] else []
            out.append(
                {
                    "context": cond,
                    "name": f"{row['case_id']} / {cond}",
                    "input": f"{row['case_id']} context={cond} columns={', '.join(cols)}",
                    "actual_output": json.dumps(actual, sort_keys=True),
                    "expected_output": json.dumps(expected, sort_keys=True),
                }
            )
    return out
