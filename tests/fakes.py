"""Fake deciders shared by the tests; they stand in for the model."""
from __future__ import annotations

from columnmap.core.dsl import render
from columnmap.core.template import KIND_DESC, KIND_LABEL
from columnmap.decisions.types import Answer, Response


def truth_texts(case):
    name_of = {k["src"]: k["name"] for k in case["columns"] if k["src"]}
    return {o["name"]: render(o["rule"], name_of.get) for o in case["outputs"] if o["available"] and o["kind"] != "pass"}


class TransformOracle:
    """Picks the true expression / true kind for choice, scores 0.9/0.1 for predicates."""

    def __init__(self, case, right=True):
        self.truth, self.right = truth_texts(case), right
        self.kinds = {o["name"]: KIND_LABEL[o["kind"]] for o in case["outputs"]}

    def decide(self, text, questions):
        first = questions[0].instructions
        target = next(n for n in self.kinds if f"'{n}'" in first)
        out = []
        for q in questions:
            if q.type == "choice":
                vals = [c.value for c in q.choices]
                pick = self.kinds[target] if set(vals) <= set(KIND_DESC) else self.truth[target]
                if not self.right:
                    pick = next(v for v in vals if v != pick)
                out.append(Answer("choice", q.name, choice=pick, confidence=1.0,
                                  probabilities={v: (0.7 if v == pick else 0.1) for v in vals},
                                  raw_scores={v: 0.0 for v in vals}))
            else:
                hit = q.instructions.endswith(self.truth[target])
                p = 0.9 if hit == self.right else 0.1
                out.append(Answer("predicate", q.name, probability=p, raw_scores={"false": 0.0, "true": 0.0}))
        return Response("oracle", out, {})
