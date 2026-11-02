import json

import pytest

from columnmap.core.template import load_template
from columnmap.core.dsl import safe_evaluate
from columnmap.dataset.generate import generate

T = load_template()
CASES = generate(T, n_per_level=6, seed=7)


def test_deterministic_and_covers_all_levels():
    assert json.dumps(generate(T, n_per_level=6, seed=7)) == json.dumps(CASES)
    assert json.dumps(generate(T, n_per_level=6, seed=8)) != json.dumps(CASES)
    assert sorted({c["level"] for c in CASES}) == [0, 3, 5, 6]
    assert len(CASES) == 4 * 6 and len({c["case_id"] for c in CASES}) == len(CASES)


def test_ground_truth_expressions_execute_to_expected_values():
    for c in CASES:
        col_of = {k["src"]: k["name"] for k in c["columns"] if k["src"]}
        for o in c["outputs"]:
            if not o["available"]:
                assert o["expected"] is None and o["sources"] == []
                continue
            assert o["sources"] == [col_of[s] for s in o["source_ids"]]
            got = [safe_evaluate(o["rule"], {s: row[col_of[s]] for s in col_of}) for row in c["rows"]]
            assert got == pytest.approx(o["expected"]) if o["kind"] in ("arith",) else got == o["expected"]


def test_availability_means_all_sources_present():
    for c in CASES:
        present = {k["src"] for k in c["columns"] if k["src"]}
        for o in c["outputs"]:
            assert o["available"] == set(o["source_ids"]).issubset(present)


def test_level0_pass_columns_are_identical_to_output_names():
    for c in (c for c in CASES if c["level"] == 0):
        for o in c["outputs"]:
            if o["available"] and o["kind"] == "pass":
                assert o["sources"] == [o["name"]]


def test_hard_cases_have_decoys_and_missing_outputs_and_fanout():
    hard = [c for c in CASES if c["level"] == 6]
    assert all(any(k["decoy"] for k in c["columns"]) for c in hard)
    assert not any(k["decoy"] for c in CASES if c["level"] < 6 for k in c["columns"])
    assert any(not o["available"] for c in CASES for o in c["outputs"])
    fan = [c for c in CASES for k in c["columns"] if k["src"]
           and sum(k["src"] in o["source_ids"] and o["available"] for o in c["outputs"]) >= 2]
    assert fan                                         # one input feeds 2+ outputs


def test_candidates_have_one_truth_and_distinct_distractors():
    seen = 0
    for c in CASES:
        col_of = {k["src"]: k["name"] for k in c["columns"] if k["src"]}
        for o in c["outputs"]:
            if not o["available"] or o["kind"] == "pass":
                assert o.get("candidates") in (None, [])
                continue
            cands = o["candidates"]
            assert len(cands) >= 3 and cands.count(o["rule"]) == 1
            truth = o["expected"]
            for cand in cands:
                if cand == o["rule"]:
                    continue
                vals = [safe_evaluate(cand, {s: row[col_of[s]] for s in col_of}) for row in c["rows"]]
                assert vals != truth and any(v is not None for v in vals)
            seen += 1
    assert seen > 0


def test_rows_are_many_and_diverse_with_one_or_two_anomalies():
    for c in CASES:
        assert 15 <= len(c["rows"]) <= 30
        assert 1 <= len(c["anomalies"]) <= 2
        for a in c["anomalies"]:
            assert c["rows"][a["row"]][a["column"]] == a["value"]
    big = [c for c in CASES if c["level"] == 0][0]
    cities = {r["insured_city"] for r in big["rows"] if "insured_city" in r}
    assert len(cities) >= 3 or "insured_city" not in big["rows"][0]
    kinds = {a["kind"] for c in generate(T, n_per_level=12, seed=7) for a in c["anomalies"]}
    assert {"zero", "empty", "na"} <= kinds
    zeros = [a for c in CASES for a in c["anomalies"] if a["kind"] == "zero"]
    assert all(a["value"] == 0 for a in zeros)
