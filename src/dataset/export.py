"""CSV files of the template and of the seven named samples: data/ is the single source of truth.
Run: uv run python -m src.dataset.export   (deterministic; `python src` refreshes the samples it uses)

A sample folder holds three files named after the kind of input, e.g. data/samples/c_coded/:
input_coded.csv (the messy file), expected_coded.csv (the converted file, ready to upload) and
ground_truth_coded.csv (which input columns feed each output field, and how).
"""
from __future__ import annotations

from pathlib import Path

from ..core.csvio import write_csv as _write
from ..core.dsl import render
from ..core.template import load_template
from .samples import SAMPLES, word

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


def sample_folder(letter: str) -> str:
    return f"{letter}_{word(SAMPLES[letter][0])}"


def write_samples(t: dict, cases: dict, letters: str, data_dir) -> None:
    """data/: the template and one folder per sample (input, expected output, ground truth). Deterministic."""
    data_dir = Path(data_dir)
    write_template(t, data_dir)
    for x in letters:
        write_case(cases[SAMPLES[x][0]], data_dir / "samples" / sample_folder(x))


if __name__ == "__main__":
    import tomllib

    from .generate import generate
    cfg = tomllib.loads((ROOT / "config.toml").read_text())["run"]
    t = load_template()
    cases = {c["case_id"]: c for c in generate(t, 1, cfg["seed"])}
    write_samples(t, cases, "".join(SAMPLES), ROOT / "data")
    print(f"template and {len(SAMPLES)} samples written under data/ (input, expected, ground truth per sample)")
