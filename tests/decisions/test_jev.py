import json
from pathlib import Path

import pytest

from columnmap.core.paths import VENDOR

from columnmap.backends.jev_gguf import JevDecider
from columnmap.decisions.types import Choice, Level, Question, request_from_dict, response_to_dict

FIX = Path(__file__).parents[1] / "fixtures"
MODEL = "Jev-Style-2B-Decision-v3"


# Mimics JevStyleDecisionGGUF.score_many: dict questions -> result dicts
class FakeBackend:

    def __init__(self):
        self.calls = []

    def score_many(self, state, questions):
        self.calls.append((state, questions))
        out = []
        for q in questions:
            if q["t"] == "noul":
                names, scores, probs = ["false", "true"], [-1.2, 0.12], [0.2, 0.8]
            elif q["t"] == "choice":
                names = [str(k) for k in q["crit"]]
                scores = [0.0] * len(names); scores[-1] = 1.0
                probs = [0.1] * len(names); probs[-1] = 1 - 0.1 * (len(names) - 1)
            else:
                names = [str(i) for i in range(len(q["crit"]))]
                scores = [0.0] * len(names)
                probs = [0.0] * len(names); probs[1] = 0.5; probs[2] = 0.5
            i = max(range(len(probs)), key=probs.__getitem__)
            out.append({"answer": names[i], "probabilities": dict(zip(names, probs)),
                        "scores": dict(zip(names, scores)), "top_probability": probs[i],
                        "entropy_concentration": 0.4, "input_tokens": 50, "state_tokens": 30, "model": MODEL})
        return out


def test_maps_three_types_in_one_backend_call():
    _, text, qs = request_from_dict(json.loads((FIX / "request.json").read_text()))
    fb = FakeBackend()
    r = JevDecider(backend=fb).decide(text, qs)
    assert len(fb.calls) == 1 and fb.calls[0][0] == text          # state computed once
    qd = fb.calls[0][1]
    assert qd[0] == {"t": "noul", "ins": qs[0].instructions, "crit": None}
    assert qd[1]["t"] == "choice" and list(qd[1]["crit"]) == ["billing", "shipping", "product"]
    assert qd[1]["crit"]["billing"] == "Payments, invoices, and refunds" and qd[1]["crit"]["product"] is None
    assert qd[2]["t"] == "score" and qd[2]["crit"] == ["not urgent", "somewhat urgent", "urgent: needs attention today", "critical"]

    assert r.model == MODEL and [a.name for a in r.answers] == ["refund_requested", "team", "urgency"]
    p, c, s = r.answers
    assert p.type == "predicate" and p.probability == 0.8 and p.raw_scores == {"false": -1.2, "true": 0.12}
    assert c.choice == "product" and abs(sum(c.probabilities.values()) - 1) < 1e-9 and c.confidence == 0.4
    assert s.score == pytest.approx(1.5) and list(s.probabilities) == [lv.label for lv in qs[2].levels]
    # state tokens counted once: 30 + 3 * (50 - 30)
    assert r.usage["input_tokens"] == 90 and r.usage["output_tokens"] == 0 and r.usage["total_tokens"] == 90
    json.dumps(response_to_dict(r))                                # serialisable


def test_empty_questions_and_error_passthrough():
    d = JevDecider(backend=FakeBackend())
    assert d.decide("x", []).answers == []

    class Boom:
        def score_many(self, *_):
            raise ValueError("InputBudgetError: too long")

    with pytest.raises(ValueError, match="too long"):
        JevDecider(backend=Boom()).decide("x", [Question("predicate", "q")])


GGUF = VENDOR / "Jev-Style-2B-Decision-v3-Q8_0.gguf"


@pytest.mark.skipif(not GGUF.exists(), reason="run scripts/setup_jev.sh")
def test_real_model_card_example_and_budget():
    with JevDecider() as d:
        q = Question("choice", "What should the assistant do next?", name="act", choices=(
            Choice("send", "press Send now"), Choice("ask_user", "ask the user to review the draft"),
            Choice("discard", "delete the draft")))
        text = ("The user said: 'Write a reply to Bob with the Q3 report attached, but let me check it before "
                "it goes out.' The draft is written and the report is attached.")
        a = d.decide(text, [q]).answers[0]
        assert a.choice == "ask_user" and a.probabilities["ask_user"] > 0.8 and a.raw_scores
        with pytest.raises(ValueError):
            d.decide("word " * 40000, [Question("predicate", "Is this long?")])
        s = d.decide("urgent outage now", [Question("score", "Urgency?", levels=(Level("low"), Level("high")))]).answers[0]
        assert 0 <= s.score <= 1


def test_settings_built_options_reach_the_decider_for_each_question_type():
    from columnmap.core.settings import model_options, parse
    cfg = parse({"Backend": {"Type": "local"}, "Local": {"Quant": "Q8_0", "Ngl": 999, "FlashAttn": "default"},
                 "Device": {"Mode": "cpu"},
                 "Run": {"Mode": "options", "TargetPrecision": 1.0, "PromptVersion": "v1", "KeepD": 0.0, "Seed": 1234},
                 "Splits": {"Dev": [11], "Calibration": [12, 13], "Test": [21, 22]}}, env={})
    opts = model_options(cfg)
    assert opts["ngl"] == 0
    fb = FakeBackend()
    qs = [Question("predicate", "is the field a date?"),
          Question("choice", "which unit?", choices=(Choice("EUR"), Choice("USD"))),
          Question("score", "how urgent?", levels=(Level("low"), Level("mid"), Level("high")))]
    r = JevDecider(backend=fb, **opts).decide("column text", qs)
    assert [a.type for a in r.answers] == ["predicate", "choice", "score"]
    assert len(fb.calls) == 1
