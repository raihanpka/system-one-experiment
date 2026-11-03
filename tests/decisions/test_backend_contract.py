# the hosted backend returns raw scores per option (no temperature applied), and the top raw score is the chosen option

import pytest

from columnmap.backends.openrouter import OpenRouterDecider
from columnmap.decisions.types import Choice, Level, Question


def openrouter_with(answers):
    def transport(url, headers, body, timeout):
        return {"model": "m", "answers": answers}

    return OpenRouterDecider("k", "m", "u", transport=transport)


CHOICE = Question(
    "choice",
    "which?",
    name="unit",
    choices=(Choice("EUR"), Choice("USD"), Choice("GBP")),
)
PRED = Question("predicate", "is it a date?", name="is_date")
SCORE = Question(
    "score",
    "how urgent?",
    name="urgency",
    levels=(Level("low"), Level("mid"), Level("high")),
)


def deciders():
    hosted = openrouter_with(
        {
            "unit": {
                "choice": "USD",
                "probabilities": {"EUR": 0.2, "USD": 0.7, "GBP": 0.1},
            },
            "is_date": {"noul": 0.8},
            "urgency": {
                "score": 1.0,
                "legend": {"0": "low", "1": "mid", "2": "high"},
                "probabilities": {"0": 0.2, "1": 0.6, "2": 0.2},
            },
        }
    )
    return [("openrouter", hosted, "unit")]


@pytest.mark.parametrize("name, decider, _", deciders())
def test_choice_raw_scores_name_every_option_and_the_top_one_is_the_answer(
    name, decider, _
):
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
