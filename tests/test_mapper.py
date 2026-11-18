# the library API on a tiny table: knowledge from a folder, a scripted decider, deterministic conversion
import json

from columnmap import Fill, Knowledge, Mapper, Output
from columnmap.decider import Answer, Response
from columnmap.mapper import (
    convert,
    infer_kinds,
    score_columns,
    set_candidates,
    top_columns,
)

ROWS = [
    {"born": "1981-01-09", "code": "A", "first": "Budi", "last": "Halim", "sex": "M"},
    {"born": "1953-05-25", "code": "R", "first": "Siti", "last": "Wijaya", "sex": "F"},
    {
        "born": "2005-11-23",
        "code": "P",
        "first": "Rina",
        "last": "Setiawan",
        "sex": "F",
    },
]
KNOWLEDGE = Knowledge(
    (
        Output("birth_year", "number", "Year of birth"),
        Output("status", "string", "Status in words"),
        Output("full_name", "string", "Given name then family name"),
        Output("gender", "string", "Gender of the insured"),
        Output("tax_id", "string", "Tax number of the insured"),
    ),
    ({"A": "Approved", "R": "Rejected", "P": "Pending"},),
)
# what the scripted model "knows": the kind of every field, the column set it needs, and the column scores of stage 1
KINDS = {
    "birth_year": "year",
    "status": "lookup",
    "full_name": "join",
    "gender": "lookup",
    "tax_id": "copy",
}
SETS = {
    "birth_year": "born",
    "status": "code",
    "full_name": "first & last",
    "gender": "sex",
}  # tax_id has no source
BIND = {
    "birth_year": ["born"],
    "status": ["code"],
    "full_name": ["first", "last"],
    "gender": ["sex"],
}  # tax_id has no role that a column can fill
SCORES = {c: {f: -3.0 for f in KINDS} for c in ROWS[0]}
for f, s in SETS.items():
    for c in s.split(" & "):
        SCORES[c][f] = 3.0


class Scripted:
    def __init__(self):
        self.questions = []
        self.calls = []

    def decide(self, text, questions):
        self.questions += questions
        self.calls.append([q.name for q in questions])
        out = []
        for q in questions:
            names = [c.value for c in q.choices]
            field = next(
                (
                    f
                    for f in KINDS
                    if f"Output field: {f}" in q.instructions
                    or f"'{f}'" in q.instructions
                    or f"field: {f}" in text
                ),
                None,
            )
            if q.type == "predicate":
                p = 0.9 if field in SETS or field in BIND else 0.05
                out.append(
                    Answer(
                        "predicate",
                        q.name,
                        probability=p,
                        raw_scores={"false": 0.0, "true": 0.0},
                    )
                )
            elif q.name.startswith("bind:"):
                _, f, i = q.name.split(":")
                i = int(i)
                target = BIND.get(f, [])
                if i < len(target) and target[i] in names:
                    p = {n: (0.9 if n == target[i] else 0.01) for n in names}
                else:
                    p = {n: (0.9 if n.startswith("none") else 0.01) for n in names}
                out.append(Answer("choice", q.name, probabilities=p))
            elif q.name == "field":
                col = text.split("\n")[0].removeprefix("Input column: ")
                out.append(
                    Answer(
                        "choice",
                        q.name,
                        probabilities={n: 0.2 for n in names},
                        raw_scores=SCORES[col],
                    )
                )
            elif q.name == "kind":
                f = text.split("\n")[0].removeprefix("Output field: ")
                p = {n: (0.9 if n == KINDS[f] else 0.02) for n in names}
                out.append(Answer("choice", q.name, probabilities=p))
            elif q.name == "set":
                p = {n: (0.9 if n == SETS.get(field) else 0.01) for n in names}
                if field not in SETS:
                    p = {n: (0.8 if n.startswith("none") else 0.01) for n in names}
                out.append(Answer("choice", q.name, probabilities=p))
            else:  # expr: prefer the lookup/year/concat over a plain copy
                p = {
                    n: (0.9 if ("lookup" in n or "year" in n or "concat" in n) else 0.1)
                    for n in names
                }
                out.append(Answer("choice", q.name, probabilities=p))
        return Response("scripted", out, {})


def test_kinds_are_read_from_the_descriptions_once_per_field():
    d = Scripted()
    assert infer_kinds(d, KNOWLEDGE) == KINDS
    assert [q.name for q in d.questions] == [
        "kind"
    ] * 15  # three shuffled orders per field


def test_scores_come_from_one_question_per_column_that_lists_the_field_descriptions():
    d = Scripted()
    matrix, raw = score_columns(d, ROWS, KNOWLEDGE)
    assert set(matrix) == set(ROWS[0]) and matrix["born"]["birth_year"] == 3.0
    q = d.questions[0]
    assert [c.value for c in q.choices][0] == "birth_year" and q.choices[
        0
    ].description == "Year of birth"
    assert "Sample values" in raw["born"]["input"]


def test_top_columns_rank_by_score_and_break_ties_by_name():
    m = {"b": {"f": 1.0}, "a": {"f": 1.0}, "c": {"f": 2.0}}
    assert top_columns(m, "f", 2) == ["c", "a"]


def test_the_mapper_chooses_the_set_then_the_expression_and_converts_deterministically():
    mapping = Mapper(Scripted(), KNOWLEDGE).map(ROWS)
    f = mapping.fields
    assert (
        f["birth_year"]["cols"] == ["born"]
        and f["birth_year"]["best_text"] == "year(born)"
    )
    assert (
        f["full_name"]["cols"] == ["first", "last"]
        and f["full_name"]["best_text"] == 'concat(first, " ", last)'
    )
    assert (
        f["status"]["best_text"].startswith("lookup(code")
        and f["status"]["score"] == 0.9
    )
    header, converted = mapping.convert(ROWS)
    assert header == list(KINDS)
    assert (
        converted[0][:3] == [1981, "Approved", "Budi Halim"]
        and converted[2][1] == "Pending"
    )


def test_a_kind_without_candidates_falls_back_to_a_plain_copy():
    # the model calls gender a lookup, but no code table covers it: the column is copied
    mapping = Mapper(Scripted(), KNOWLEDGE).map(ROWS)
    assert mapping.fields["gender"]["best_text"] == "sex"
    assert mapping.convert(ROWS)[1][1][3] == "F"


def test_a_field_whose_columns_are_not_there_says_none_and_stays_empty():
    mapping = Mapper(Scripted(), KNOWLEDGE).map(ROWS)
    rec = mapping.fields["tax_id"]
    assert rec["p_none"] >= 0.8 and rec["score"] < 0.1
    assert mapping.convert(ROWS)[1][0][4] is None  # below the fill score


def test_a_higher_fill_score_leaves_more_fields_empty():
    mapping = Mapper(Scripted(), KNOWLEDGE).map(ROWS)
    _, converted = mapping.convert(ROWS, Fill(0.95, 0.95))
    assert all(v is None for row in converted for v in row)


def test_set_candidates_use_exactly_the_chosen_columns():
    out = KNOWLEDGE.outputs[2]
    texts = {
        c["op"] if "op" in c else "src"
        for c in set_candidates(("first", "last"), ROWS, out, "join", ())
    }
    assert texts == {"concat"}
    assert set_candidates(("first",), ROWS, out, "join", ()) == [
        {"src": "first"}
    ]  # no join of one column: copy


def test_knowledge_round_trips_through_a_folder_and_is_all_a_project_needs_to_switch_data(
    tmp_path,
):
    KNOWLEDGE.save(tmp_path)
    assert json.loads((tmp_path / "outputs.json").read_text())[0] == {
        "name": "birth_year",
        "type": "number",
        "description": "Year of birth",
    }
    assert Knowledge.load(tmp_path) == KNOWLEDGE
    assert convert({"x": {"best": {"src": "born"}, "score": 1.0}}, ROWS, ["x"])[1][
        0
    ] == ["1981-01-09"]


def test_a_column_claimed_by_two_copied_fields_stays_with_the_higher_score():
    from columnmap.mapper import drop_double_claims

    def rec(score, col, kind="copy"):
        return {"kind": kind, "best": {"src": col}, "cols": [col], "score": score}

    fields = {
        "a": rec(0.9, "x"),
        "b": rec(0.5, "x"),
        "c": rec(0.6, "y"),
        "d": rec(0.8, "x", "year"),
    }
    out = drop_double_claims(fields)
    assert (
        out["a"]["score"] == 0.9
        and out["b"]["score"] == 0.0
        and out["b"]["claimed_by"] == "a"
    )
    assert (
        out["c"]["score"] == 0.6 and out["d"]["score"] == 0.8
    )  # only copied fields compete


def test_a_copied_field_needs_a_lower_score_than_a_derived_field_to_be_filled():
    from columnmap.mapper import Fill, is_filled

    copy = {"kind": "copy", "best": {"src": "x"}, "score": 0.5}
    derived = {"kind": "join", "best": {"src": "x"}, "score": 0.5}
    assert is_filled(copy, Fill(0.7, 0.4)) and not is_filled(derived, Fill(0.7, 0.4))
    assert not is_filled({**copy, "best": None}, Fill(0.1, 0.1))


def test_presence_and_none_decide_the_fill_not_the_joint_set_score():
    from columnmap.mapper import Fill, is_filled

    rec = {
        "kind": "copy",
        "best": {"src": "x"},
        "score": 0.95,
        "p_present": 0.3,
        "p_none": 0.1,
    }
    assert not is_filled(rec, Fill(0.7, 0.4))  # the source is not there
    assert not is_filled({**rec, "p_present": 0.9, "p_none": 0.8}, Fill(0.7, 0.4))
    assert is_filled({**rec, "p_present": 0.9, "p_none": 0.2}, Fill(0.7, 0.4))


def test_a_lookup_bind_question_shows_the_code_table():
    d = Scripted()
    Mapper(d, KNOWLEDGE).map(ROWS)
    q = next(q for q in d.questions if q.name == "bind:status:0")
    assert "Reference table: A -> Approved" in q.instructions


def test_role_questions_for_a_group_of_fields_share_one_request():
    d = Scripted()
    Mapper(d, KNOWLEDGE).map(ROWS)
    bind_calls = [c for c in d.calls if any(n.startswith("bind:") for n in c)]
    assert len(bind_calls) == 1  # the five shaped fields fit in one batch
    assert sum(n.startswith("bind:") for c in bind_calls for n in c) == 6


def test_a_presence_refusal_fails_closed_in_the_fallback():
    from columnmap.mapper import _field, is_filled

    class PresenceRefused:
        def decide(self, text, questions):
            out = []
            for q in questions:
                if q.name == "presence":
                    out.append(Answer("refusal", q.name, message="no"))
                else:
                    names = [c.value for c in q.choices]
                    out.append(
                        Answer(
                            "choice",
                            q.name,
                            choice=names[0],
                            probabilities={
                                n: (0.95 if i == 0 else 0.01)
                                for i, n in enumerate(names)
                            },
                        )
                    )
            return Response("refused", out, {})

    out = Output("mystery", "number", "Something computed elsewhere")
    rec = _field(
        PresenceRefused(),
        out,
        "arithmetic",
        [{"a": 1, "b": 2}],
        {"a": {"mystery": 3.0}, "b": {"mystery": 3.0}},
        ["a", "b"],
        (),
    )
    assert rec["p_present"] == 0.0 and not is_filled(rec, Fill(0.5, 0.5))


def test_an_amount_role_is_not_offered_a_rate_column():
    from columnmap.mapper import _columns_for
    from columnmap.roles import Role

    rows = [{"fee_pct": "0.05", "fee_amount": "10", "other": "x"}]
    allowed = _columns_for(
        Role("fee", "amount"),
        rows,
        {"fee_pct": "rate", "fee_amount": "amount", "other": "text"},
    )
    assert "fee_amount" in allowed and "fee_pct" not in allowed
    assert "other" not in allowed


def test_a_rate_field_does_not_take_an_amount_column():
    from columnmap.mapper import _columns_for
    from columnmap.roles import Role

    rows = [{"fee_pct": "0.05", "fee_amount": "10"}]
    allowed = _columns_for(
        Role("tax rate", "rate"), rows, {"fee_pct": "rate", "fee_amount": "amount"}
    )
    assert allowed == ["fee_pct"]


def test_a_refusal_fails_closed_instead_of_crashing_a_stage():
    class Refusal:
        def decide(self, text, questions):
            return Response(
                "refusal",
                [Answer("refusal", q.name, message="no answer") for q in questions],
                {},
            )

    assert infer_kinds(Refusal(), KNOWLEDGE) == {f: "copy" for f in KINDS}
    matrix, raw = score_columns(Refusal(), ROWS, KNOWLEDGE)
    assert all(v == 0.0 for row in matrix.values() for v in row.values())
    assert raw["born"]["error"] == "no answer"
