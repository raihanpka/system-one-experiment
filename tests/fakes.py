# Fake deciders shared by the tests; they stand in for the model

from __future__ import annotations

from columnmap.core.dsl import render
from columnmap.core.template import KIND_DESC, KIND_LABEL
from columnmap.decisions.types import Answer, Response


def truth_texts(case):
    name_of = {k["src"]: k["name"] for k in case["columns"] if k["src"]}
    return {
        o["name"]: render(o["rule"], name_of.get)
        for o in case["outputs"]
        if o["available"] and o["kind"] != "pass"
    }


# Picks the true expression / true kind for choice, scores 0.9/0.1 for predicates
class TransformOracle:
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
                pick = (
                    self.kinds[target]
                    if set(vals) <= set(KIND_DESC)
                    else self.truth[target]
                )
                if not self.right:
                    pick = next(v for v in vals if v != pick)
                out.append(
                    Answer(
                        "choice",
                        q.name,
                        choice=pick,
                        confidence=1.0,
                        probabilities={v: (0.7 if v == pick else 0.1) for v in vals},
                        raw_scores={v: 0.0 for v in vals},
                    )
                )
            else:
                hit = q.instructions.endswith(self.truth[target])
                p = 0.9 if hit == self.right else 0.1
                out.append(
                    Answer(
                        "predicate",
                        q.name,
                        probability=p,
                        raw_scores={"false": 0.0, "true": 0.0},
                    )
                )
        return Response("oracle", out, {})


class Generic:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        pass

    def decide(self, text, questions):
        out = []
        for q in questions:
            if q.type == "predicate":
                out.append(
                    Answer(
                        "predicate",
                        q.name,
                        probability=0.3,
                        raw_scores={"false": 0.0, "true": -0.8},
                    )
                )
            else:
                names = [c.value for c in q.choices]
                raw = {n: (1.0 if i == 0 else -1.0) for i, n in enumerate(names)}
                out.append(
                    Answer(
                        "choice",
                        q.name,
                        choice=names[0],
                        confidence=0.5,
                        probabilities={n: 1 / len(names) for n in names},
                        raw_scores=raw,
                    )
                )
        return Response("fake", out, {})
