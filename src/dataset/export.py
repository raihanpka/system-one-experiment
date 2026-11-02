"""CSV views of the template and of the generated cases. Run: uv run python -m src.dataset.export [--n 3]

A case folder holds three files named after the kind of input, e.g. for `legacy-000`:
input_legacy.csv (the messy file), expected_legacy.csv (the converted file, ready to upload) and
ground_truth_legacy.csv (which input columns feed each output field, and how).
"""
from __future__ import annotations

from pathlib import Path

from ..core.csvio import write_csv as _write
from ..core.dsl import render
from ..core.template import load_template
from .samples import word

ROOT = Path(__file__).resolve().parents[2]


def write_template(t: dict, out) -> None:
    out = Path(out)
    _write(out / "template_header.csv", [o["name"] for o in t["outputs"]], [])
    _write(out / "template_columns.csv", ["name", "type", "kind", "description", "rule"],
           [[o["name"], o["type"], o["kind"], o["description"], render(o["rule"], lambda s: s)] for o in t["outputs"]])


def write_case(case: dict, d) -> None:
    d, w = Path(d), word(case["case_id"])
    names = [k["name"] for k in case["columns"]]
    _write(d / f"input_{w}.csv", names, [[r[n] for n in names] for r in case["rows"]])
    _write(d / f"expected_{w}.csv", [o["name"] for o in case["outputs"]],
           [["" if not o["available"] or o["expected"][i] is None else o["expected"][i] for o in case["outputs"]]
            for i in range(len(case["rows"]))])
    name_of = {k["src"]: k["name"] for k in case["columns"] if k["src"]}
    rows = [[o["name"], o["kind"], o["available"], "|".join(o["sources"]),
             render(o["rule"], name_of.get) if o["available"] else ""] for o in case["outputs"]]
    rows += [["", "decoy", False, k["name"], ""] for k in case["columns"] if k["decoy"]]
    rows += [["", "anomaly", False, f"{a['column']}@row{a['row'] + 1}", f"{a['kind']}: {a['value']!r}"] for a in case["anomalies"]]
    _write(d / f"ground_truth_{w}.csv", ["output", "kind", "available", "input_columns", "expression"], rows)


def write_cases(cases, out) -> None:
    for c in cases:
        write_case(c, Path(out) / c["case_id"])


if __name__ == "__main__":
    import argparse
    import tomllib

    from .generate import generate
    cfg = tomllib.loads((ROOT / "config.toml").read_text())["run"]
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=cfg["n_per_level"], help="cases per drift level")
    a = ap.parse_args()
    t = load_template()
    cases = generate(t, a.n, cfg["seed"])
    write_template(t, ROOT / "data")
    write_cases(cases, ROOT / "data" / "cases")
    print(f"template + {len(cases)} cases (3 CSVs each) written under data/")
