"""CSV views of the template, the generated input files and the ground-truth map.
Run: uv run python -m src.export_csv [--n 5]   (writes under data/)"""
from __future__ import annotations

import csv
from pathlib import Path

from .core.template import load_template
from .core.dsl import render

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


def write_cases(cases, out) -> None:
    """One folder per case: input.csv (the messy file), expected_output.csv (converted, upload-ready),
    mapping.csv (ground truth: which input columns feed each output, and how)."""
    for c in cases:
        d = Path(out) / c["case_id"]
        names = [k["name"] for k in c["columns"]]
        _write(d / "input.csv", names, [[r[n] for n in names] for r in c["rows"]])
        _write(d / "expected_output.csv", [o["name"] for o in c["outputs"]],
               [["" if not o["available"] or o["expected"][i] is None else o["expected"][i] for o in c["outputs"]]
                for i in range(len(c["rows"]))])
        name_of = {k["src"]: k["name"] for k in c["columns"] if k["src"]}
        rows = [[o["name"], o["kind"], o["available"], "|".join(o["sources"]),
                 render(o["rule"], name_of.get) if o["available"] else ""] for o in c["outputs"]]
        rows += [["", "decoy", False, k["name"], ""] for k in c["columns"] if k["decoy"]]
        rows += [["", "anomaly", False, f"{a['column']}@row{a['row'] + 1}", f"{a['kind']}: {a['value']!r}"] for a in c["anomalies"]]
        _write(d / "mapping.csv", ["output", "kind", "available", "input_columns", "expression"], rows)


if __name__ == "__main__":
    import argparse
    import tomllib

    from .gen_gt import generate
    cfg = tomllib.loads((ROOT / "config.toml").read_text())["run"]
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=cfg["n_per_level"], help="cases per drift level")
    a = ap.parse_args()
    t = load_template()
    cases = generate(t, a.n, cfg["seed"])
    write_template(t, ROOT / "data")
    write_cases(cases, ROOT / "data" / "cases")
    print(f"template + {len(cases)} cases (3 CSVs each) written under data/")
