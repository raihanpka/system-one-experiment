# Fake deciders shared by the tests; they stand in for the model

from __future__ import annotations

from columnmap.decisions.types import Answer, Response


class Generic:
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
