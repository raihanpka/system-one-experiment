import csv

from columnmap.core.template import load_template
from columnmap.dataset.export import write_samples, write_template
from columnmap.dataset.generate import generate

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


def test_samples_folder_is_the_single_source_of_truth(tmp_path):
    by_id = {c["case_id"]: c for c in generate(T, n_per_level=1, seed=5)}
    write_samples(T, by_id, "abcd", tmp_path)
    assert (tmp_path / "template_header.csv").exists()
    assert sorted(p.name for p in (tmp_path / "samples").iterdir()) == ["a_easy_identical", "b_medium_synonym", "c_hard_codes", "d_hardest_noisy"]
    c = by_id["hardest_noisy-000"]
    d = tmp_path / "samples" / "d_hardest_noisy"
    assert sorted(p.name for p in d.iterdir()) == ["expected_hardest_noisy.csv", "ground_truth_hardest_noisy.csv", "input_hardest_noisy.csv"]

    inp = rows(d / "input_hardest_noisy.csv")
    assert inp[0] == [k["name"] for k in c["columns"]] and len(inp) == 1 + len(c["rows"])

    exp = rows(d / "expected_hardest_noisy.csv")                        # the converted file, ready to upload
    assert exp[0] == [o["name"] for o in T["outputs"]] and len(exp) == 1 + len(c["rows"])
    for j, o in enumerate(c["outputs"]):
        col = [r[j] for r in exp[1:]]
        assert col == (["" if v is None else str(v) for v in o["expected"]] if o["available"] else [""] * len(c["rows"]))

    m = rows(d / "ground_truth_hardest_noisy.csv")
    assert m[0] == ["output", "kind", "available", "input_columns", "expression"]
    assert len([r for r in m[1:] if r[1] == "anomaly"]) == len(c["anomalies"]) and all("@row" in r[3] for r in m[1:] if r[1] == "anomaly")
    assert len([r for r in m[1:] if r[1] == "decoy"]) == sum(bool(k["decoy"]) for k in c["columns"]) > 0

    before = (d / "input_hardest_noisy.csv").read_text()
    write_samples(T, by_id, "abcd", tmp_path)                    # deterministic: rewriting changes nothing
    assert (d / "input_hardest_noisy.csv").read_text() == before


def test_level5_has_hard_codes_style_codes():
    many = generate(T, n_per_level=12, seed=5)
    names = [k["name"] for c in many if c["level"] == 5 for k in c["columns"]]
    assert any(n.isupper() and "_" in n for n in names)          # e.g. OBJ_INFO_ADDR_C
