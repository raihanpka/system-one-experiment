from __future__ import annotations

from typing import Protocol, Sequence

from .types import Question, Response


class Decider(Protocol):
    """Backend-agnostic. Swap Jev local / OpenAI / Jev hosted by implementing this one method."""

    def decide(self, text: str, questions: Sequence[Question]) -> Response: ...
