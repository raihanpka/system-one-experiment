# the library API on a tiny table: knowledge from a folder, a scripted decider, deterministic conversion
import json

from columnmap import Knowledge, Mapper, Output
from columnmap.decider import Answer, Response
from columnmap.mapper import choose_expressions, convert, score_columns, top_columns

ROWS = [
    {"born": "1981-01-09", "code": "A", "first": "Budi", "last": "Halim"},
    {"born": "1953-05-25", "code": "R", "first": "Siti", "last": "Wijaya"},
    {"born": "2005-11-23", "code": "P", "first": "Rina", "last": "Setiawan"},
]
KNOWLEDGE = Knowledge(
    (
        Output("birth_year", "number", "Year of birth"),
        Output("status", "string", "Status in words"),
        Output("full_name", "string", "Given name then family name"),
    ),
    ({"A": "Approved", "R": "Rejected", "P": "Pending"},),
)


class Scripted:
    # column scores come from the table below; an expression is approved when its text says what it should
    SCORES = {
        "born": {"birth_year": 3.0, "status": -3.0, "full_name": -4.0},
        "code": {"birth_year": -4.0, "status": 3.0, "full_name": -3.0},
        "first": {"birth_year": -4.0, "status": -3.0, "full_name": 2.0},
        "last": {"birth_year": -4.0, "status": -3.0, "full_name": 1.0},
    }
    GOOD = {
        "birth_year": "year(born)",
        "status": "lookup(code",
        "full_name": 'concat(first, " ", last)',
    }

    def __init__(self):
        self.questions = []

    def decide(self, text, questions):
        self.questions += questions
        out = []
        for q in questions:
            if q.type == "choice" and q.name == "field":
                col = text.split("\n")[0].removeprefix("Input column: ")
                raw = self.SCORES[col]
                out.append(
                    Answer(
                        "choice",
                        q.name,
                        choice=max(raw, key=raw.get),
                        probabilities={k: 0.3 for k in raw},
                        raw_scores=raw,
                    )
                )
            elif (
                q.type == "choice"
            ):  # pick: the option with the good expression of the asked field first, then the option order
                names = [c.value for c in q.choices]
                want = next(
                    (
                        e
                        for f, e in self.GOOD.items()
                        if f"output field '{f}'" in q.instructions
                    ),
                    "",
                )
                p = {
                    n: (1.0 if want in n else 1 / (2 + i)) for i, n in enumerate(names)
                }
                out.append(
                    Answer("choice", q.name, choice=max(p, key=p.get), probabilities=p)
                )
            else:
                good = any(
                    f"output field '{f}'" in q.instructions and e in q.instructions
                    for f, e in self.GOOD.items()
                )
                out.append(
                    Answer("predicate", q.name, probability=0.97 if good else 0.1)
                )
        return Response("scripted", out, {})


def test_scores_come_from_one_question_per_column_that_lists_the_field_descriptions():
    d = Scripted()
    matrix, raw = score_columns(d, ROWS, KNOWLEDGE, "samples")
    assert (
        set(matrix) == {"born", "code", "first", "last"}
        and matrix["born"]["birth_year"] == 3.0
    )
    q = d.questions[0]
    assert [c.value for c in q.choices] == ["birth_year", "status", "full_name"]
    assert [c.description for c in q.choices][0] == "Year of birth"
    assert "Sample values" in raw["born"]["input"]


def test_top_columns_rank_by_score_and_break_ties_by_name():
    m = {"b": {"f": 1.0}, "a": {"f": 1.0}, "c": {"f": 2.0}}
    assert top_columns(m, "f", 2) == ["c", "a"]


def test_the_mapper_finds_a_year_a_lookup_and_a_concat_and_converts_deterministically():
    mapping = Mapper(Scripted(), KNOWLEDGE).map(ROWS)
    texts = {f: r["best_text"] for f, r in mapping.fields.items()}
    assert (
        texts["birth_year"] == "year(born)"
        and texts["full_name"] == 'concat(first, " ", last)'
    )
    assert texts["status"].startswith("lookup(code")
    header, converted = mapping.convert(ROWS)
    assert header == ["birth_year", "status", "full_name"]
    assert (
        converted[0] == [1981, "Approved", "Budi Halim"]
        and converted[2][1] == "Pending"
    )


def test_a_field_below_the_fill_score_stays_empty():
    mapping = Mapper(Scripted(), KNOWLEDGE).map(ROWS)
    _, converted = mapping.convert(ROWS, fill_score=0.99)
    assert all(v is None for row in converted for v in row)


def test_the_simplest_expression_wins_when_scores_are_close():
    class AllApprove(Scripted):
        def decide(self, text, questions):
            res = super().decide(text, questions)
            for a in res.answers:
                if a.type == "predicate":
                    a.probability = 0.95
            return res

    rec = choose_expressions(
        AllApprove(), ROWS, score_columns(AllApprove(), ROWS, KNOWLEDGE)[0], KNOWLEDGE
    )
    assert (
        rec["full_name"]["method"] == "copy"
    )  # a plain copy of a column, not a longer concat


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
