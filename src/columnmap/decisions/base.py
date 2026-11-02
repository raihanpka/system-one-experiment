from __future__ import annotations

from typing import Protocol, Sequence

from .types import Question, Response


# backend-agnostic: a new model backend only has to implement this one method
class Decider(Protocol):

    def decide(self, text: str, questions: Sequence[Question]) -> Response: ...
