from columnmap.bench.usage import Counting, estimate_minutes
from columnmap.decider import Answer, Question, Response


class Fake:
    def decide(self, text, questions):
        return Response(
            "f",
            [
                Answer(
                    "predicate",
                    q.name,
                    probability=0.5,
                    raw_scores={"false": 0, "true": 0},
                )
                for q in questions
            ],
            {
                "input_tokens": 10 * len(questions),
                "output_tokens": 0,
                "total_tokens": 10 * len(questions),
            },
        )


def test_counting_wrapper_adds_up_tokens_questions_and_calls():
    c = Counting(Fake())
    c.decide(
        "s",
        [Question("predicate", "q", name="a"), Question("predicate", "q", name="b")],
    )
    c.decide("s", [Question("predicate", "q", name="a")])
    assert (c.tokens, c.questions, c.calls) == (30, 3, 2)
    assert c.snapshot() == (30, 3, 2)


def test_estimate_grows_with_files():
    cases = [{"columns": [0] * 26, "outputs": [0] * 31}]
    one = estimate_minutes(cases)
    two = estimate_minutes(cases * 2)
    assert 0 < one < two == 2 * one
    assert 0.5 < one < 5  # about one to two minutes for one file with the hosted model
