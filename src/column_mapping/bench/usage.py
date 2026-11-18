# prompt tokens, questions and a time estimate, so a run's cost is visible before and after

from __future__ import annotations

OPTIONS_Q_SECONDS = (
    0.8  # one question with all output fields as options (hosted, measured)
)
HONEST_FILE_SECONDS = (
    45  # kind, column set and expression per field, per file (hosted, measured)
)


# wraps a decider and counts what it is asked
class Counting:
    def __init__(self, decider):
        self.decider, self.tokens, self.questions, self.calls = decider, 0, 0, 0

    def decide(self, text, questions):
        res = self.decider.decide(text, questions)
        self.tokens += res.usage.get("input_tokens", 0)
        self.questions += len(questions)
        self.calls += 1
        return res

    def snapshot(self) -> tuple[int, int, int]:
        return self.tokens, self.questions, self.calls


# counts for the work done since `before`; zeros when the decider is not wrapped
def since(decider, before=(0, 0, 0)) -> dict:
    now = decider.snapshot() if hasattr(decider, "snapshot") else (0, 0, 0)
    return {"tokens": now[0] - before[0], "questions": now[1] - before[1]}


def mark(decider) -> tuple[int, int, int]:
    return decider.snapshot() if hasattr(decider, "snapshot") else (0, 0, 0)


# rough minutes for a run: seconds per mapping question times the columns, plus the honest stage once per file
def estimate_minutes(cases) -> float:
    secs = sum(
        len(c["columns"]) * OPTIONS_Q_SECONDS + HONEST_FILE_SECONDS for c in cases
    )
    return secs / 60
