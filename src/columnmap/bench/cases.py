# Read a finished run back into cases and records. No DeepEval import: calibration and the report can
# use this without the dev dependency.

from __future__ import annotations

import json
from pathlib import Path

from ..mapper import Fill, convert, is_filled
from .generate import generate
from .metrics import field_correct
from .store import Store
from .summary import top_column
from .template import load_template


def load_run(run_dir) -> tuple[Store, dict, dict]:
    store = Store(Path(run_dir) / "result.json")
    meta = store.data["meta"]
    cases = {c["case_id"]: c for c in generate(load_template(), 1, meta["seed"])}
    return store, meta, cases


def records_from_run(run_dir) -> tuple[list[dict], dict]:
    store, meta, cases = load_run(run_dir)
    honest = {r["case_id"]: r for r in store.rows("honest")}
    rows = []
    for row in store.rows("mapping"):
        rec = honest.get(row["case_id"])
        if rec is None:
            continue
        rows.append(
            {
                "case": cases[row["case_id"]],
                "fields": rec["fields"],
                "matrix": row["matrix"],
            }
        )
    return rows, meta


def cases_from_run(run_dir, fill: Fill) -> list[dict]:
    rows, _ = records_from_run(run_dir)
    out = []
    for item in rows:
        case = item["case"]
        names = [o["name"] for o in case["outputs"]]
        _, conv = convert(item["fields"], case["rows"], names, fill)
        right = field_correct(case, conv)
        actual, expected = {}, {}
        for o in case["outputs"]:
            f = o["name"]
            actual[f] = {
                "column": top_column(item["matrix"], f),
                "filled": is_filled(item["fields"][f], fill),
                "right": right[f],
            }
            expected[f] = sorted(o["sources"]) if o["available"] else []
        out.append(
            {
                "name": f"{case['case_id']}",
                "input": f"{case['case_id']} columns={len(case['columns'])}",
                "actual_output": json.dumps(actual, sort_keys=True),
                "expected_output": json.dumps(expected, sort_keys=True),
            }
        )
    return out
