# A refusal or a missing answer is a failed judgment, not a no. Retry the whole batch once; if it
# still refuses, return the refusal so the mapper fails closed and leaves the field empty.

from __future__ import annotations

from typing import Sequence

from ..decider import Decider, Question, Response


class Retrying:
    def __init__(self, decider: Decider, attempts: int = 2):
        self._decider = decider
        self._attempts = max(1, attempts)

    def decide(self, text: str, questions: Sequence[Question]) -> Response:
        last = None
        for _ in range(self._attempts):
            last = self._decider.decide(text, questions)
            if not any(a.type == "refusal" for a in last.answers):
                return last
        return last

    def __getattr__(self, name):
        return getattr(self._decider, name)
