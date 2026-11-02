import inspect

from columnmap.core.dsl import safe_evaluate
from columnmap.decisions.types import Answer, Response
from columnmap.honest.run import chosen_columns, run_fields

ROWS = [{"first": "Budi", "last": "Santoso", "base": 1000.0, "tax": 0.1, "id": "A1"},
        {"first": "Siti", "last": "Wijaya", "base": 2000.0, "tax": 0.2, "id": "B2"},
        {"first": "Agus", "last": "Pratama", "base": 3000.0, "tax": 0.0, "id": "C3"}]
FIELDS = ["full_name", "gross", "ident", "nothing"]


def cell(d):
    return {"p": 0.5, "d": d}


# every input column scores every field; first+last feed full_name, base+tax feed gross, id feeds ident, nobody feeds nothing
MATRIX = {c: {f: cell(-3.0) for f in FIELDS} for c in ROWS[0]}
MATRIX["first"]["full_name"] = cell(2.0)
MATRIX["last"]["full_name"] = cell(1.5)
MATRIX["base"]["gross"] = cell(2.5)
MATRIX["tax"]["gross"] = cell(1.0)
MATRIX["id"]["ident"] = cell(3.0)


class Judge:
    """Prefers the candidate whose text starts with `want[field]`; records every prompt it receives."""

    def __init__(self, want):
        self.want, self.seen = want, []

    def decide(self, text, questions):
        self.seen.append((text, questions))
        field = next(f for f in FIELDS if f"'{f}'" in questions[0].instructions)
        out = []
        for q in questions:
            if q.type == "choice":
                names = [c.value for c in q.choices]
                out.append(Answer("choice", q.name, choice=names[0], confidence=1.0,
                                  probabilities={n: (0.9 if n.startswith(self.want[field]) else 0.1 / max(1, len(names) - 1)) for n in names}))
            else:
                p = 0.9 if q.instructions.split("computed as: ")[1].startswith(self.want[field]) else 0.1
                out.append(Answer("predicate", q.name, probability=p, raw_scores={"false": 0.0, "true": 1.0}))
        return Response("judge", out, {})


def test_chosen_columns_are_those_passing_the_rule_best_first_and_capped():
    assert chosen_columns(MATRIX, "full_name") == [("first", 2.0), ("last", 1.5)]
    assert chosen_columns(MATRIX, "nothing") == []
    many = {c: {"f": cell(1.0 + i)} for i, c in enumerate("abcde")}
    assert [c for c, _ in chosen_columns(many, "f")] == ["e", "d", "c"]
    assert chosen_columns(MATRIX, "full_name", keep=1.8) == [("first", 2.0)]


def test_only_field_names_rows_and_scores_go_in_never_ground_truth():
    assert list(inspect.signature(run_fields).parameters) == ["decider", "fields", "rows", "matrix", "keep", "max_cols"]


def test_single_text_column_is_copied_without_asking_the_model():
    j = Judge({f: "" for f in FIELDS})
    rec = run_fields(j, ["ident"], ROWS, MATRIX)["ident"]
    assert rec["method"] == "copy" and rec["cols"] == ["id"] and rec["best_text"] == "id" and j.seen == []
    assert rec["p_cols"] and 0 < rec["conf"] < 1


def test_two_columns_are_ranked_and_the_winner_runs_on_the_chosen_columns():
    j = Judge({"full_name": 'concat(first, " ", last)', "gross": "base * (1 + tax)"})
    recs = run_fields(j, FIELDS, ROWS, MATRIX)
    name = recs["full_name"]
    assert name["method"] == "expression" and name["best_text"] == 'concat(first, " ", last)'
    assert [safe_evaluate(name["best"], r) for r in ROWS] == ["Budi Santoso", "Siti Wijaya", "Agus Pratama"]
    gross = recs["gross"]
    assert [safe_evaluate(gross["best"], r) for r in ROWS] == [1100.0, 2400.0, 3000.0]
    assert recs["nothing"]["method"] == "unmapped" and recs["nothing"]["best"] is None


def test_prompts_show_computed_examples_and_only_chosen_columns():
    j = Judge({"full_name": 'concat(first, " ", last)', "gross": "base * (1 + tax)"})
    run_fields(j, ["full_name"], ROWS, MATRIX)
    texts = " ".join(t + " ".join(q.instructions + " ".join(c.value for c in q.choices) for q in qs) for t, qs in j.seen)
    assert "Budi Santoso" in texts and "Sample values" in texts            # the result of each candidate on real rows
    assert "base" not in texts and "tax" not in texts                      # columns that were not chosen never appear
    first_state = j.seen[0][0]
    assert "first" in first_state and "last" in first_state


def test_the_shortlist_gets_a_yes_no_check_and_confidence_combines_columns_and_expression():
    j = Judge({"full_name": 'concat(first, " ", last)'})
    rec = run_fields(j, ["full_name"], ROWS, MATRIX)["full_name"]
    assert len(rec["shortlist"]) <= 3 and rec["best_idx"] in rec["shortlist"] and 0 < rec["p_expr"] <= 1
    assert abs(rec["conf"] - rec["p_expr"] * rec["p_cols"][0] * rec["p_cols"][1]) < 1e-9
