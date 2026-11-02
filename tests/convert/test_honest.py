from columnmap.convert.honest import convert_fields, decisions
from columnmap.convert.metrics import cell_match, field_correct
from columnmap.core.template import load_template
from columnmap.dataset.generate import generate
from columnmap.decisions.types import Answer, Response
from columnmap.honest.prompts import _fmt
from columnmap.honest.run import run_fields

T = load_template()
CASE = next(c for c in generate(T, 1, 1234) if c["case_id"] == "technical_terms-000")
OUTS = [o["name"] for o in CASE["outputs"]]


def matrix_from(case, true_d=2.0, other_d=-3.0, wrong=False):
    """Scores: true pairs high (or, with wrong=True, a different column high). The case is only read here, never by the stage."""
    cols = [k["name"] for k in case["columns"]]
    m = {c: {o["name"]: {"p": 0.5, "d": other_d} for o in case["outputs"]} for c in cols}
    for o in case["outputs"]:
        if not o["available"]:
            continue
        wrong_cols = [c for c in cols if c not in o["sources"]][:len(o["sources"])]
        for c in (wrong_cols if wrong else o["sources"]):
            m[c][o["name"]]["d"] = true_d
    return m


class Judge:
    """Picks the candidate whose three example results equal the expected ones (knows the expected file, like a perfect model)."""

    def __init__(self, case):
        n = len(case["rows"])
        idx = sorted({i * (n - 1) // 2 for i in range(3)})
        self.target = {o["name"]: "; ".join(_fmt(o["expected"][i]) for i in idx) for o in case["outputs"] if o["available"]}

    def decide(self, text, questions):
        field = next(f for f in OUTS if f"'{f}'" in questions[0].instructions)
        want = self.target.get(field, "<none>")
        out = []
        for q in questions:
            if q.type == "choice":
                names = [c.value for c in q.choices]
                out.append(Answer("choice", q.name, choice=names[0], confidence=1.0,
                                  probabilities={n: (0.9 if n.endswith("=> " + want) else 0.01) for n in names}))
            else:
                out.append(Answer("predicate", q.name, probability=0.9 if f"for example {want})" in q.instructions else 0.1,
                                  raw_scores={"false": 0.0, "true": 1.0}))
        return Response("judge", out, {})


def convert(matrix):
    recs = run_fields(Judge(CASE), OUTS, CASE["rows"], matrix)
    _, conv = convert_fields(recs, CASE["rows"], OUTS)
    return recs, conv


def test_right_scores_give_a_right_file_except_fields_outside_the_catalog():
    recs, conv = convert(matrix_from(CASE))
    ok = field_correct(CASE, conv)
    with_src = [o for o in CASE["outputs"] if o["available"]]
    assert all(ok[o["name"]] for o in with_src if o["kind"] != "lookup")
    assert all(ok[o["name"]] for o in CASE["outputs"] if not o["available"])           # fields without a source stay empty
    lookups = [o["name"] for o in CASE["outputs"] if o["kind"] == "lookup" and o["available"]]
    assert all(recs[f]["method"] in ("copy", "unmapped") for f in lookups)               # a code table is never invented


def test_wrong_scores_give_a_wrong_file_so_the_numbers_are_not_flattered():
    _, conv = convert(matrix_from(CASE, wrong=True))
    s = cell_match(CASE, conv)
    assert s["correct_rate"] < 0.3 and sum(field_correct(CASE, conv)[o["name"]] for o in CASE["outputs"] if o["available"]) < 5


def test_unmapped_fields_are_empty_and_decisions_carry_the_truth_columns_only_for_evaluation():
    recs, conv = convert(matrix_from(CASE))
    dec = decisions(recs, CASE, field_correct(CASE, conv))
    assert [d["output"] for d in dec] == OUTS
    empty = next(d for d, o in zip(dec, CASE["outputs"]) if not o["available"])
    assert empty["method"] == "unmapped" and empty["chosen_input"] == "" and empty["field_correct"] is True
    assert {"truth_available", "truth_input", "confidence", "expression"} <= set(dec[0])
    assert all(v is None for r in conv for v, o in zip(r, CASE["outputs"]) if not o["available"])


def test_field_causes_split_wrong_columns_from_wrong_expressions():
    from columnmap.convert.metrics import field_causes
    recs, conv = convert(matrix_from(CASE))
    causes = field_causes(CASE, recs, field_correct(CASE, conv))
    assert set(causes) == {"correct", "wrong_columns", "wrong_expression"}
    assert causes["correct"] + causes["wrong_columns"] + causes["wrong_expression"] == sum(o["available"] for o in CASE["outputs"])
    assert causes["wrong_columns"] == 0                                                   # the scores named the right columns
    recs2, conv2 = convert(matrix_from(CASE, wrong=True))
    c2 = field_causes(CASE, recs2, field_correct(CASE, conv2))
    assert c2["wrong_columns"] > c2["correct"]
