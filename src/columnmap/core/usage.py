# prompt tokens, questions and a time estimate, so a run's cost is visible before and after

from __future__ import annotations

OPTIONS_Q_SECONDS = (
    0.45  # one question with all output fields as options (measured, Q8_0 on an M3)
)
PAIR_Q_SECONDS = 0.1  # one yes/no question
HONEST_FILE_SECONDS = 12  # candidate ranking per file, measured


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


# rough minutes for a run: seconds per question times the questions each mode asks, plus the honest stage once per file
def estimate_minutes(cases, conds, modes) -> float:
    secs = 0.0
    for c in cases:
        cols, outs = len(c["columns"]), len(c["outputs"])
        for _ in conds:
            for m in modes:
                secs += (
                    cols * OPTIONS_Q_SECONDS
                    if m == "options"
                    else cols * outs * PAIR_Q_SECONDS
                )
            secs += HONEST_FILE_SECONDS
    return secs / 60
