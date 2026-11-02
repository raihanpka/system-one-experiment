"""Decisions contract shaped like OpenAI's POST /v1/decisions.

Shape taken from LocalAI (OpenAI-compatible route), LangChain and TypeSafe docs; the official OpenAI page
was unreachable (see NOTES.md), so treat field names as unverified until T1's doc check is repeated.
Internal Answer keeps probabilities as {option: p}; the wire form is a list of objects.
"""
from __future__ import annotations

from dataclasses import dataclass, field

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

    def __post_init__(self):
        if self.type not in TYPES:
            raise ValueError(f"unknown question type {self.type!r}")
        if not self.instructions.strip():
            raise ValueError("instructions must not be blank")
        if self.type != "choice" and self.choices:
            raise ValueError("choices only allowed on choice questions")
        if self.type != "score" and self.levels:
            raise ValueError("levels only allowed on score questions")
        if self.type == "choice":
            vals = [c.value for c in self.choices]
            if len(vals) < 2 or len(set(vals)) != len(vals):
                raise ValueError("choice needs >= 2 unique choices")
        if self.type == "score" and not 2 <= len(self.levels) <= 10:
            raise ValueError("score needs 2..10 levels")


@dataclass
class Answer:
    type: str                      # predicate | choice | score | refusal
    name: str | None = None
    probability: float | None = None          # predicate
    choice: str | None = None                 # choice
    score: float | None = None                # score
    confidence: float | None = None
    probabilities: dict | None = None         # choice: value -> p ; score: label -> p (level order)
    message: str | None = None                # refusal
    raw_scores: dict | None = None            # extension (Jev): logit(yes)-logit(no) per option


@dataclass
class Response:
    model: str
    answers: list[Answer] = field(default_factory=list)
    usage: dict = field(default_factory=dict)


def _drop_none(d):
    return {k: v for k, v in d.items() if v is not None}


def question_to_dict(q: Question) -> dict:
    d = {"type": q.type, "name": q.name, "instructions": q.instructions}
    if q.choices:
        d["choices"] = [_drop_none({"value": c.value, "description": c.description}) for c in q.choices]
    if q.levels:
        d["levels"] = [_drop_none({"label": lv.label, "description": lv.description}) for lv in q.levels]
    return _drop_none(d)


def question_from_dict(d: dict) -> Question:
    return Question(
        d["type"], d["instructions"], d.get("name"),
        tuple(Choice(c["value"], c.get("description")) for c in d.get("choices", ())),
        tuple(Level(lv["label"], lv.get("description")) for lv in d.get("levels", ())),
    )


def request_to_dict(model: str, text: str, questions) -> dict:
    names = [q.name for q in questions if q.name is not None]
    if len(set(names)) != len(names):
        raise ValueError("question names must be unique")
    return {"model": model, "input": text, "questions": [question_to_dict(q) for q in questions]}


def request_from_dict(d: dict):
    return d["model"], d["input"], [question_from_dict(q) for q in d["questions"]]


def answer_to_dict(a: Answer) -> dict:
    d = {"name": a.name, "type": a.type}
    if a.type == "predicate":
        d["probability"] = a.probability
    elif a.type == "choice":
        d.update(choice=a.choice, confidence=a.confidence,
                 probabilities=[{"value": k, "probability": p} for k, p in a.probabilities.items()])
    elif a.type == "score":
        d.update(score=a.score, confidence=a.confidence,
                 probabilities=[{"value": i, "label": k, "probability": p}
                                for i, (k, p) in enumerate(a.probabilities.items())])
    elif a.type == "refusal":
        d["message"] = a.message
    d["raw_scores"] = a.raw_scores
    return _drop_none(d)


def answer_from_dict(d: dict) -> Answer:
    probs = d.get("probabilities")
    if probs is not None:
        probs = {(p["label"] if d["type"] == "score" else p["value"]): p["probability"] for p in probs}
    return Answer(d["type"], d.get("name"), d.get("probability"), d.get("choice"), d.get("score"),
                  d.get("confidence"), probs, d.get("message"), d.get("raw_scores"))


def response_to_dict(r: Response) -> dict:
    return {"model": r.model, "answers": [answer_to_dict(a) for a in r.answers], "usage": r.usage}


def response_from_dict(d: dict) -> Response:
    return Response(d["model"], [answer_from_dict(a) for a in d["answers"]], d.get("usage", {}))
