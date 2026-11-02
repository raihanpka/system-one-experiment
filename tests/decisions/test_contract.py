import json
from pathlib import Path

import pytest

from src.decisions.types import (
    Choice, Level, Question, request_from_dict, request_to_dict, response_from_dict, response_to_dict,
)

FIX = Path(__file__).parents[1] / "fixtures"


def load(name):
    return json.loads((FIX / name).read_text())


def test_request_roundtrip():
    d = load("request.json")
    model, text, qs = request_from_dict(d)
    assert model == "jev-v3-2b" and len(qs) == 3
    assert qs[1].choices[2] == Choice("product")
    assert request_to_dict(model, text, qs) == d


def test_response_roundtrip():
    d = load("response.json")
    r = response_from_dict(d)
    assert r.answers[0].probability == 0.93 and r.answers[0].raw_scores["true"] == 0.12
    assert r.answers[1].choice == "shipping" and r.answers[1].probabilities["shipping"] == 0.75
    assert r.answers[2].score == 1.7 and list(r.answers[2].probabilities) == ["not urgent", "somewhat urgent", "urgent", "critical"]
    assert r.answers[3].type == "refusal"
    assert response_to_dict(r) == d


def test_question_validation():
    with pytest.raises(ValueError):
        Question("choice", "x", choices=(Choice("a"),))            # needs >= 2 choices
    with pytest.raises(ValueError):
        Question("score", "x", levels=(Level("a"),))               # needs >= 2 levels
    with pytest.raises(ValueError):
        Question("score", "x", levels=tuple(Level(str(i)) for i in range(11)))  # max 10
    with pytest.raises(ValueError):
        Question("predicate", "x", choices=(Choice("a"), Choice("b")))
    with pytest.raises(ValueError):
        Question("bogus", "x")
    with pytest.raises(ValueError):
        Question("predicate", "  ")
    with pytest.raises(ValueError):
        Question("choice", "x", choices=(Choice("a"), Choice("a")))  # duplicate value


def test_duplicate_question_names_rejected():
    q = Question("predicate", "x", name="n")
    with pytest.raises(ValueError):
        request_to_dict("m", "t", [q, q])
