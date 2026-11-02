import math

import pytest

from columnmap.decisions.openrouter import OpenRouterDecider
from columnmap.decisions.types import Choice, Level, Question


def fake(answers, model="~liquid/d1-ultra-latest"):
    calls = []

    def transport(url, headers, body, timeout):
        calls.append((url, headers, body))
        return {"model": model, "answers": answers}
    transport.calls = calls
    return transport


QS = [Question("predicate", "is it a date?", name="is_date"),
      Question("choice", "which unit?", name="unit", choices=(Choice("EUR"), Choice("USD"))),
      Question("score", "how urgent?", name="urgency", levels=(Level("low"), Level("high")))]


def test_request_shape_matches_the_decisions_api_and_the_key_goes_in_the_header():
    t = fake({"is_date": {"noul": 0.8}, "unit": {"choice": "EUR", "probabilities": {"EUR": 0.7, "USD": 0.3}},
              "urgency": {"score": 0.2, "probabilities": {"low": 0.8, "high": 0.2}}})
    d = OpenRouterDecider(api_key="sk-or-v1-test", model="~liquid/d1-ultra-latest", base_url="https://openrouter.ai/api/alpha/decisions",
                          transport=t)
    d.decide("column: amount_eur", QS)
    url, headers, body = t.calls[0]
    assert url == "https://openrouter.ai/api/alpha/decisions"
    assert headers["Authorization"] == "Bearer sk-or-v1-test"
    assert body["model"] == "~liquid/d1-ultra-latest" and body["state"] == ["column: amount_eur"]
    assert body["questions"]["is_date"]["type"] == "noul"
    assert body["questions"]["unit"]["criteria"] == {"EUR": "EUR", "USD": "USD"}
    assert body["questions"]["urgency"]["criteria"] == ["low", "high"]


def test_probabilities_become_scores_the_pipeline_can_use():
    t = fake({"is_date": {"noul": 0.8}, "unit": {"choice": "EUR", "probabilities": [{"value": "EUR", "probability": 0.7}, {"value": "USD", "probability": 0.3}]},
              "urgency": {"score": 0.2, "probabilities": {"low": 0.8, "high": 0.2}}})
    r = OpenRouterDecider("k", "m", "u", transport=t).decide("x", QS)
    pred, choice, score = r.answers
    assert pred.probability == pytest.approx(0.8)
    assert pred.raw_scores["true"] - pred.raw_scores["false"] == pytest.approx(math.log(0.8 / 0.2))
    assert choice.choice == "EUR" and choice.raw_scores["EUR"] > choice.raw_scores["USD"]
    assert score.score == pytest.approx(0.2) and score.probabilities == {"low": 0.8, "high": 0.2}


def test_a_missing_answer_is_a_refusal_not_a_crash():
    r = OpenRouterDecider("k", "m", "u", transport=fake({"is_date": {"noul": 0.5}})).decide("x", QS)
    assert [a.type for a in r.answers] == ["predicate", "refusal", "refusal"]


def test_duplicate_question_names_are_refused():
    dup = [Question("predicate", "a", name="n"), Question("predicate", "b", name="n")]
    with pytest.raises(ValueError, match="unique"):
        OpenRouterDecider("k", "m", "u", transport=fake({})).decide("x", dup)


def test_http_error_is_raised_with_status_and_no_key_in_the_message():
    def boom(url, headers, body, timeout):
        raise RuntimeError("OpenRouter returned 401: invalid key")
    with pytest.raises(RuntimeError, match="401") as e:
        OpenRouterDecider("sk-or-v1-secret", "m", "u", transport=boom).decide("x", QS)
    assert "sk-or-v1-secret" not in str(e.value)


def test_factory_picks_openrouter_when_the_setting_says_so(monkeypatch):
    from columnmap.core.settings import parse
    from columnmap.decisions.factory import make_decider
    cfg = parse({"Backend": {"Type": "openrouter"}, "OpenRouter": {"Model": "m", "BaseUrl": "u", "ApiKeyEnv": "OPENROUTER_API_KEY"},
                 "Local": {"Quant": "Q8_0", "Ngl": 999}, "Run": {"Mode": "options", "TargetPrecision": 1.0, "Seed": 1},
                 "Splits": {"Dev": [11], "Calibration": [12], "Test": [21]}}, env={})
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-x")
    d = make_decider(cfg)()
    assert isinstance(d, OpenRouterDecider)
