from columnmap.core.store import Store
from columnmap.core.template import load_template
from columnmap.dataset.generate import generate
from columnmap.decisions.types import Answer, Response
from columnmap.experiment import run_models

T = load_template()
CASE = next(c for c in generate(T, 1, 1234) if c["case_id"] == "easy_identical-000")


class Tokens:
    """Scores nothing useful; reports 7 prompt tokens per question."""

    def decide(self, text, questions):
        out = []
        for q in questions:
            if q.type == "choice":
                names = [c.value for c in q.choices]
                out.append(Answer("choice", q.name, choice=names[0], probabilities={n: 1 / len(names) for n in names},
                                  raw_scores={n: (1.0 if i == 0 else -1.0) for i, n in enumerate(names)}))
            else:
                out.append(Answer("predicate", q.name, probability=0.3, raw_scores={"false": 0.0, "true": -0.8}))
        return Response("t", out, {"input_tokens": 7 * len(questions)})


def test_every_record_carries_seconds_tokens_and_questions(tmp_path):
    store, log = Store(tmp_path / "r.json"), []
    run_models(Tokens(), [CASE], ["name"], ["options"], store, T, log=log.append)
    m = store.rows("mapping", "name")[0]
    h = store.rows("honest", "name")[0]
    assert m["questions"] == len(CASE["columns"]) and m["tokens"] == 7 * m["questions"] and "seconds" in m
    assert h["tokens"] >= 0 and "seconds" in h and "questions" in h
    assert not [l for l in log if l.startswith("warning")]


def test_a_long_run_is_announced_before_it_starts(tmp_path):
    log = []
    run_models(Tokens(), [CASE] * 60, ["name", "samples", "table"], ["options", "pairs"], Store(tmp_path / "r.json"), T, log=log.append)
    assert log[0].startswith("warning: this run is estimated at") and "minutes" in log[0]
