import csv

from src.check_template import load_template
from src.export_csv import write_inputs, write_template, write_truth
from src.gen_gt import generate

T = load_template()
CASES = generate(T, n_per_level=2, seed=5)


def rows(p):
    with open(p, newline="", encoding="utf-8") as f:
        return list(csv.reader(f))


def test_template_files(tmp_path):
    write_template(T, tmp_path)
    assert rows(tmp_path / "template_header.csv") == [[o["name"] for o in T["outputs"]]]
    cols = rows(tmp_path / "template_columns.csv")
    assert cols[0] == ["name", "type", "kind", "description", "rule"] and len(cols) == 1 + len(T["outputs"])
    gross = next(r for r in cols if r[0] == "gross_premium")
    assert gross[4] == "base_premium * (1 + premium_tax_rate)"


def test_input_files_and_truth(tmp_path):
    write_inputs(CASES, tmp_path / "inputs")
    c = next(c for c in CASES if c["level"] == 5)
    r = rows(tmp_path / "inputs" / f"{c['case_id']}.csv")
    assert r[0] == [k["name"] for k in c["columns"]] and len(r) == 1 + len(c["rows"])
    write_truth(CASES, tmp_path / "truth.csv")
    t = rows(tmp_path / "truth.csv")
    assert t[0] == ["case_id", "level", "output", "kind", "available", "input_columns", "expression"]
    line = next(x for x in t[1:] if x[0] == c["case_id"] and x[2] == "gross_premium")
    assert line[4] in ("True", "False")
    if line[4] == "True":
        assert "*" in line[6] and all(n in line[6] for n in line[5].split("|"))


def test_level5_has_legacy_style_codes():
    many = generate(T, n_per_level=12, seed=5)
    names = [k["name"] for c in many if c["level"] == 5 for k in c["columns"]]
    assert any(n.isupper() and "_" in n for n in names)          # e.g. OBJ_INFO_ADDR_C
