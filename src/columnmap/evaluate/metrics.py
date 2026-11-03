# two DeepEval metrics that need no judge model: they compare the model's choices with the ground truth

from __future__ import annotations

import json

from deepeval.metrics import BaseMetric

from .cases import FILL_P


def _read(test_case) -> tuple[dict, dict]:
    return json.loads(test_case.actual_output), json.loads(test_case.expected_output)


class SourceAccuracy(BaseMetric):
    # share of the available output fields whose top column is a true source
    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold

    def measure(self, test_case, *args, **kwargs) -> float:
        actual, expected = _read(test_case)
        avail = [f for f, srcs in expected.items() if srcs]
        hit = [f for f in avail if actual[f]["column"] in expected[f]]
        self.score = len(hit) / len(avail) if avail else 1.0
        self.reason = (
            f"{len(hit)} of {len(avail)} available fields have the right top column"
        )
        self.success = self.score >= self.threshold
        return self.score

    async def a_measure(self, test_case, *args, **kwargs) -> float:
        return self.measure(test_case)

    def is_successful(self) -> bool:
        return self.success

    @property
    def __name__(self) -> str:
        return "Source accuracy"


class FillPrecision(BaseMetric):
    # share of the filled fields that are right; a field with no source that is filled counts as wrong
    def __init__(self, threshold: float = 0.9):
        self.threshold = threshold

    def measure(self, test_case, *args, **kwargs) -> float:
        actual, expected = _read(test_case)
        fills = [f for f in actual if actual[f]["p"] >= FILL_P]
        right = [f for f in fills if actual[f]["column"] in expected[f]]
        self.score = len(right) / len(fills) if fills else 1.0
        self.reason = f"{len(right)} of {len(fills)} filled fields are right"
        self.success = self.score >= self.threshold
        return self.score

    async def a_measure(self, test_case, *args, **kwargs) -> float:
        return self.measure(test_case)

    def is_successful(self) -> bool:
        return self.success

    @property
    def __name__(self) -> str:
        return "Fill precision"
