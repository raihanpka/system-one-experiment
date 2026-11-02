"""CSV views of the template, the generated input files and the ground-truth map.
Run: uv run python -m src.export_csv   (writes under data/)"""
from __future__ import annotations

import csv
from pathlib import Path

from .check_template import load_template
from .dsl import render

ROOT = Path(__file__).resolve().parents[1]


def _write(path, header, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def write_template(t: dict, out) -> None:
    out = Path(out)
    _write(out / "template_header.csv", [o["name"] for o in t["outputs"]], [])
    _write(out / "template_columns.csv", ["name", "type", "kind", "description", "rule"],
           [[o["name"], o["type"], o["kind"], o["description"], render(o["rule"], lambda s: s)] for o in t["outputs"]])


def write_inputs(cases, out) -> None:
    for c in cases:
        names = [k["name"] for k in c["columns"]]
        _write(Path(out) / f"{c['case_id']}.csv", names, [[r[n] for n in names] for r in c["rows"]])


def write_truth(cases, path) -> None:
    rows = []
    for c in cases:
        name_of = {k["src"]: k["name"] for k in c["columns"] if k["src"]}
        for o in c["outputs"]:
            expr = render(o["rule"], lambda s: name_of.get(s, s)) if o["available"] else ""
            rows.append([c["case_id"], c["level"], o["name"], o["kind"], o["available"], "|".join(o["sources"]), expr])
    _write(path, ["case_id", "level", "output", "kind", "available", "input_columns", "expression"], rows)


if __name__ == "__main__":
    import json
    t = load_template()
    cases = [json.loads(l) for l in (ROOT / "data/gt.jsonl").read_text().splitlines()]
    write_template(t, ROOT / "data")
    write_inputs(cases, ROOT / "data/inputs")
    write_truth(cases, ROOT / "data/ground_truth_map.csv")
    print(f"template + {len(cases)} input CSVs + truth map written under data/")
