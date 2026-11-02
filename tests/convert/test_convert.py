import math

from columnmap.convert.apply import convert_case
from columnmap.convert.metrics import cell_match
from columnmap.core.scoring import T0
from columnmap.core.template import KIND_LABEL, load_template
from columnmap.dataset.generate import generate
from columnmap.transform.run import run_case as transform_case
from tests.fakes import TransformOracle

T = load_template()
CASES = {c["case_id"]: c for c in generate(T, n_per_level=1, seed=4)}


# Mapping row where true pairs get p_true and all others 0.05; kinds are right; transform picks are right
def oracle_rows(case, wrong_kind=False, p_true=0.9):
    d = lambda p: {"p": p, "d": math.log(p / (1 - p)) * T0}                    # noqa: E731
    matrix = {k["name"]: {o["name"]: d(p_true if k["name"] in o["sources"] else 0.05) for o in case["outputs"]}
              for k in case["columns"]}
    kind = {o["name"]: {"choice": "passthrough" if wrong_kind else KIND_LABEL[o["kind"]], "p": {}}
            for o in case["outputs"] if o["available"]}
    m = {"case_id": case["case_id"], "level": case["level"], "cond": "name", "mode": "pairs", "matrix": matrix, "kind": kind}
    return m, transform_case(TransformOracle(case), case, "name", T)


def test_oracle_decisions_rebuild_the_expected_file():
    case = CASES["coded-000"]
    m, tr = oracle_rows(case)
    header, rows, decisions = convert_case(m, tr, case)
    assert header == [o["name"] for o in case["outputs"]] and len(rows) == len(case["rows"])
    s = cell_match(case, rows)
    assert s["correct"] == s["expected_cells"] > 0 and s["wrong_fill"] == 0
    by_out = {d["output"]: d for d in decisions}
    first_pass = next(o["name"] for o in case["outputs"] if o["available"] and o["kind"] == "pass")
    assert by_out[first_pass]["method"] == "copy" and by_out[first_pass]["correct"]
    calc = next(o["name"] for o in case["outputs"] if o["available"] and o["kind"] == "arith")
    assert by_out[calc]["method"] == "expression" and by_out[calc]["expression"] and by_out[calc]["kind_chosen"] == "arithmetic"


def test_low_probability_leaves_the_field_empty():
    case = CASES["coded-000"]
    m, tr = oracle_rows(case, p_true=0.2)                                     # nothing reaches 0.5
    header, rows, decisions = convert_case(m, tr, case)
    assert all(v is None for r in rows for v in r)
    assert {d["method"] for d in decisions} == {"unmapped"}
    s = cell_match(case, rows)
    assert s["correct"] == 0 and s["wrong_fill"] == 0 and s["expected_cells"] > 0


def test_wrong_kind_copies_instead_of_calculating():
    case = CASES["synonym-000"]
    m, tr = oracle_rows(case, wrong_kind=True)
    _, rows, decisions = convert_case(m, tr, case)
    calc = next(o["name"] for o in case["outputs"] if o["available"] and o["kind"] == "arith")
    assert [d for d in decisions if d["output"] == calc][0]["method"] == "copy"
    assert cell_match(case, rows)["correct"] < cell_match(case, rows)["expected_cells"]


def test_cell_match_counts_wrong_fill_and_float_tolerance():
    case = CASES["identical-000"]
    n = len(case["rows"])
    j = next(i for i, o in enumerate(case["outputs"]) if not o["available"])
    k = next(i for i, o in enumerate(case["outputs"]) if o["available"] and o["kind"] == "arith")
    rows = [[None] * len(case["outputs"]) for _ in range(n)]
    rows[0][j] = "oops"                                                        # filled but should be empty
    rows[1][k] = case["outputs"][k]["expected"][1] * (1 + 1e-12)               # float noise still counts as correct
    s = cell_match(case, rows)
    assert s["wrong_fill"] == 1 and s["correct"] == 1
