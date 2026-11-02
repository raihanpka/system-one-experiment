# every backend returns raw scores per option (no temperature applied), and the top raw score is the chosen option
import math

import pytest

from columnmap.decisions.jev import JevDecider
from columnmap.decisions.openrouter import OpenRouterDecider
from columnmap.decisions.types import Choice, Level, Question


class JevLike:
    # shaped like JevStyleDecisionGGUF.score_many: raw scores in "scores", probabilities already scaled
    def score_many(self, state, questions):
        out = []
        for q in questions:
            if q["t"] == "noul":
                names, scores = ["false", "true"], [-1.0, 0.5]
            elif q["t"] == "choice":
                names = [str(k) for k in q["crit"]]
                scores = [0.2 * i for i in range(len(names))]
            else:
                names = [str(i) for i in range(len(q["crit"]))]
                scores = [0.1, 1.5, -0.5][: len(names)]
            top = max(range(len(scores)), key=scores.__getitem__)
            probs = {n: 1 / (1 + math.exp(-s)) for n, s in zip(names, scores)}
            out.append({"answer": names[top], "probabilities": probs, "scores": dict(zip(names, scores)),
                        "entropy_concentration": 0.5, "input_tokens": 10, "state_tokens": 5, "model": "jev"})
        return out


def openrouter_with(answers):
    def transport(url, headers, body, timeout):
        return {"model": "m", "answers": answers}
    return OpenRouterDecider("k", "m", "u", transport=transport)


CHOICE = Question("choice", "which?", name="unit", choices=(Choice("EUR"), Choice("USD"), Choice("GBP")))
PRED = Question("predicate", "is it a date?", name="is_date")
SCORE = Question("score", "how urgent?", name="urgency", levels=(Level("low"), Level("mid"), Level("high")))


def deciders():
    jev = JevDecider(backend=JevLike())
    hosted = openrouter_with({
        "unit": {"choice": "USD", "probabilities": {"EUR": 0.2, "USD": 0.7, "GBP": 0.1}},
        "is_date": {"noul": 0.8},
        "urgency": {"score": 1.0, "legend": {"0": "low", "1": "mid", "2": "high"}, "probabilities": {"0": 0.2, "1": 0.6, "2": 0.2}},
    })
    return [("jev", jev, "unit"), ("openrouter", hosted, "unit")]


@pytest.mark.parametrize("name, decider, _", deciders())
def test_choice_raw_scores_name_every_option_and_the_top_one_is_the_answer(name, decider, _):
    a = decider.decide("column", [CHOICE]).answers[0]
    assert set(a.raw_scores) == {"EUR", "USD", "GBP"}
    assert max(a.raw_scores, key=a.raw_scores.get) == a.choice


@pytest.mark.parametrize("name, decider, _", deciders())
def test_predicate_gives_true_minus_false_as_the_decision_score(name, decider, _):
    a = decider.decide("column", [PRED]).answers[0]
    assert a.probability is not None
    d = a.raw_scores["true"] - a.raw_scores["false"]
    assert (d > 0) == (a.probability > 0.5)


@pytest.mark.parametrize("name, decider, _", deciders())
def test_score_probabilities_are_keyed_by_level_label(name, decider, _):
    a = decider.decide("column", [SCORE]).answers[0]
    assert set(a.probabilities) == {"low", "mid", "high"}
