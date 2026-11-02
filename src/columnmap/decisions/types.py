# decision questions (predicate, choice, score) and their answers, as plain dataclasses

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
