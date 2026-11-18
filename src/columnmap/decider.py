# The Decisions interface: question and answer types, the Decider protocol, and the OpenRouter Decisions API client
# The API returns calibrated probabilities, not logits. Scores are derived from them so the pipeline keeps its contract:
#   predicate: raw true = logit(p), raw false = 0, so true - false = logit(p)
#   choice, score: raw per option = logit(p), so 0 means p = 0.5 as in the earlier local runs; the ranking is unchanged

from __future__ import annotations

import json
import math
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Callable, Protocol, Sequence

TYPES = ("predicate", "choice", "score")


@dataclass(frozen=True)
class Choice:
    value: str
    description: str | None = None


@dataclass(frozen=True)
class Level:
    label: str
    description: str | None = None


@dataclass(frozen=True)
class Question:
    type: str
    instructions: str
    name: str | None = None
    choices: tuple[Choice, ...] = ()
    levels: tuple[Level, ...] = ()
    true: str | None = None  # predicate: what counts as yes
    false: str | None = None  # predicate: what counts as no

    def __post_init__(self):
        if self.type not in TYPES:
            raise ValueError(f"unknown question type {self.type!r}")
        if not self.instructions.strip():
            raise ValueError("instructions must not be blank")
        if self.type != "choice" and self.choices:
            raise ValueError("choices only allowed on choice questions")
        if self.type != "score" and self.levels:
            raise ValueError("levels only allowed on score questions")
        if self.type != "predicate" and (self.true or self.false):
            raise ValueError("true/false criteria only allowed on predicate questions")
        if self.type == "choice":
            vals = [c.value for c in self.choices]
            if len(vals) < 1 or len(set(vals)) != len(vals):
                raise ValueError("choice needs >= 1 unique choice")
        if self.type == "score" and not 2 <= len(self.levels) <= 10:
            raise ValueError("score needs 2..10 levels")


@dataclass
class Answer:
    type: str  # predicate | choice | score | refusal
    name: str | None = None
    probability: float | None = None  # predicate
    choice: str | None = None  # choice
    score: float | None = None  # score
    confidence: float | None = None
    probabilities: dict | None = (
        None  # choice: value -> p ; score: label -> p (level order)
    )
    message: str | None = None  # refusal
    raw_scores: dict | None = None  # extension: logit(yes)-logit(no) per option


@dataclass
class Response:
    model: str
    answers: list[Answer] = field(default_factory=list)
    usage: dict = field(default_factory=dict)


class Decider(Protocol):
    def decide(self, text: str, questions: Sequence[Question]) -> Response: ...


def _logit(p: float) -> float:
    p = min(max(p, P_CLIP), 1 - P_CLIP)
    return math.log(p / (1 - p))


P_CLIP = 1e-6


def _to_api(q: Question, name: str) -> dict:
    if q.type == "predicate":
        return {
            "type": "noul",
            "instructions": q.instructions,
            "criteria": {"true": q.true or "Yes", "false": q.false or "No"},
        }
    if q.type == "choice":
        return {
            "type": "choice",
            "instructions": q.instructions,
            "criteria": {c.value: (c.description or c.value) for c in q.choices},
        }
    return {
        "type": "score",
        "instructions": q.instructions,
        "criteria": [lv.label for lv in q.levels],
    }


def _distribution(raw) -> dict:
    # a dict {value: p} or a list of {value|label, probability} or {value|label: p} pairs
    if isinstance(raw, dict):
        return {str(k): float(v) for k, v in raw.items()}
    out = {}
    for row in raw:
        key = row.get("value", row.get("label"))
        out[str(key)] = float(row["probability"])
    return out


def _answer(q: Question, name: str, body: dict) -> Answer:
    if q.type == "predicate":
        p = float(body["noul"])
        p = min(max(p, P_CLIP), 1 - P_CLIP)
        return Answer(
            "predicate",
            name,
            probability=p,
            raw_scores={"true": _logit(p), "false": 0.0},
        )
    probs = _distribution(body["probabilities"])
    if q.type == "choice":
        raw = {k: _logit(v) for k, v in probs.items()}
        return Answer(
            "choice",
            name,
            choice=str(body["choice"]),
            probabilities=probs,
            raw_scores=raw,
        )
    # score: the API keys levels by index ("0", "1", ...) with a legend; relabel to the question's level labels
    labels = [lv.label for lv in q.levels]
    legend = body.get("legend") or {str(i): lab for i, lab in enumerate(labels)}
    probs = {legend.get(k, k): v for k, v in probs.items()}
    raw = {k: _logit(v) for k, v in probs.items()}
    return Answer(
        "score", name, score=float(body["score"]), probabilities=probs, raw_scores=raw
    )


def _post(url: str, headers: dict, body: dict, timeout: float) -> dict:
    req = urllib.request.Request(
        url, data=json.dumps(body).encode("utf-8"), headers=headers, method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        raise RuntimeError(
            f"OpenRouter returned {e.code}: {e.read()[:300].decode('utf-8', 'replace')}"
        ) from None


class OpenRouterDecider:
    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str,
        transport: Callable | None = None,
        timeout: float = 120.0,
    ):
        self._key, self._model, self._url = api_key, model, base_url
        self._transport = transport or _post
        self._timeout = timeout
        self.run_meta = {"model": model, "backend": "openrouter", "quant": "api"}

    def decide(self, text: str, questions: Sequence[Question]) -> Response:
        if not questions:
            return Response(self._model)
        names = [q.name or f"q{i}" for i, q in enumerate(questions)]
        if len(set(names)) != len(names):
            raise ValueError("question names must be unique")
        body = {
            "model": self._model,
            "state": [text],
            "questions": {n: _to_api(q, n) for n, q in zip(names, questions)},
        }
        headers = {
            "Authorization": f"Bearer {self._key}",
            "Content-Type": "application/json",
        }
        res = self._transport(self._url, headers, body, self._timeout)
        answers = res.get("answers", {})
        out = []
        for n, q in zip(names, questions):
            if n not in answers:
                out.append(Answer("refusal", n, message="no answer returned"))
                continue
            out.append(_answer(q, n, answers[n]))
        return Response(res.get("model", self._model), out, res.get("usage", {}))

    def close(self):
        return None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
