import csv

from src.core.template import load_template
from src.dataset.export import write_cases, write_template
from src.dataset.generate import generate

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
    assert next(r for r in cols if r[0] == "gross_premium")[4] == "base_premium * (1 + premium_tax_rate)"


def test_each_case_is_input_expected_output_and_mapping(tmp_path):
    write_cases(CASES, tmp_path)
    assert len(list(tmp_path.glob("*/input_*.csv"))) == len(CASES)
    c = next(c for c in CASES if c["level"] == 6)
    assert c["case_id"].startswith("mixed-")
    d = tmp_path / c["case_id"]

    inp = rows(d / "input_mixed.csv")
    assert inp[0] == [k["name"] for k in c["columns"]] and len(inp) == 1 + len(c["rows"])

    exp = rows(d / "expected_mixed.csv")                       # the converted file, ready to upload
    assert exp[0] == [o["name"] for o in T["outputs"]] and len(exp) == 1 + len(c["rows"])
    for j, o in enumerate(c["outputs"]):
        col = [r[j] for r in exp[1:]]
        assert col == (["" if v is None else str(v) for v in o["expected"]] if o["available"] else [""] * len(c["rows"]))

    m = rows(d / "ground_truth_mixed.csv")
    assert m[0] == ["output", "kind", "available", "input_columns", "expression"]
    gross = next(r for r in m[1:] if r[0] == "gross_premium")
    if gross[2] == "True":
        assert "*" in gross[4] and all(n in gross[4] for n in gross[3].split("|"))
    anom = [r for r in m[1:] if r[1] == "anomaly"]
    assert len(anom) == len(c["anomalies"]) and all("@row" in r[3] for r in anom)
    decoys = [r for r in m[1:] if r[1] == "decoy"]
    assert len(decoys) == sum(bool(k["decoy"]) for k in c["columns"]) > 0


def test_level5_has_legacy_style_codes():
    many = generate(T, n_per_level=12, seed=5)
    names = [k["name"] for c in many if c["level"] == 5 for k in c["columns"]]
    assert any(n.isupper() and "_" in n for n in names)          # e.g. OBJ_INFO_ADDR_C
