import pytest

from column_mapping.decider import Choice, Level, Question


def test_question_validation():
    assert Question("choice", "x", choices=(Choice("a"),)).choices  # one field is valid
    with pytest.raises(ValueError):
        Question("choice", "x", choices=())  # needs at least one choice
    with pytest.raises(ValueError):
        Question("score", "x", levels=(Level("a"),))  # needs >= 2 levels
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
