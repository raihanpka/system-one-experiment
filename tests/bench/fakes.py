# Fake deciders shared by the tests; they stand in for the model

from __future__ import annotations

from column_mapping.decider import Answer, Response


class Generic:
    p = 0.3  # probability every predicate gets
    top = None  # probability of the first option of every choice; None = equal shares
    run_meta = {
        "model": "~liquid/d1-ultra-latest",
        "backend": "openrouter",
        "quant": "api",
    }

    def __enter__(self):
        return self

    def __exit__(self, *a):
        pass

    def shares(self, names):
        if self.top is None:
            return {n: 1 / len(names) for n in names}
        return {
            n: self.top if i == 0 else (1 - self.top) / (len(names) - 1)
            for i, n in enumerate(names)
        }

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
                        probabilities=self.shares(names),
                        raw_scores=raw,
                    )
                )
        return Response("fake", out, {})


class Approves(Generic):
    top = 0.95  # the first option of every question gets 0.95, so every field is filled
