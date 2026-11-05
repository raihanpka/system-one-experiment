# Fake deciders shared by the tests; they stand in for the model

from __future__ import annotations

from columnmap.decider import Answer, Response


class Generic:
    p = 0.3  # probability every predicate gets
    run_meta = {
        "model": "~liquid/d1-ultra-latest",
        "backend": "openrouter",
        "quant": "api",
    }

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
                        probability=self.p,
                        raw_scores={"false": 0.0, "true": 0.0},
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


class Approves(Generic):
    p = 0.95  # every candidate expression is approved
