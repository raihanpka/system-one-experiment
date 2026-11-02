# JevDecider: Decider backed by the local Jev-Style-2B-Decision-v3 GGUF runtime (vendor/)

from __future__ import annotations

import sys
from typing import Sequence

from ..core.paths import VENDOR
from .types import Answer, Question, Response


def _load_backend(quant: str, ngl: int = 999, flash_attn: str = "default", threads: int | None = None):
    sys.path.insert(0, str(VENDOR))
    from jev_style_decision_gguf import JevStyleDecisionGGUF  # noqa: PLC0415

    return JevStyleDecisionGGUF(model_dir=VENDOR, quant=quant, n_gpu_layers=ngl, flash_attn=flash_attn, threads=threads)


def _to_jev(q: Question) -> dict:
    if q.type == "predicate":
        return {"t": "noul", "ins": q.instructions, "crit": None}
    if q.type == "choice":
        return {"t": "choice", "ins": q.instructions, "crit": {c.value: c.description for c in q.choices}}
    levels = [f"{lv.label}: {lv.description}" if lv.description else lv.label for lv in q.levels]
    return {"t": "score", "ins": q.instructions, "crit": levels}


def _answer(q: Question, r: dict) -> Answer:
    raw = r["scores"]
    if q.type == "predicate":
        return Answer("predicate", q.name, probability=r["probabilities"]["true"], raw_scores=raw)
    conf = r["entropy_concentration"]
    if q.type == "choice":
        return Answer("choice", q.name, choice=r["answer"], confidence=conf, probabilities=r["probabilities"], raw_scores=raw)
    probs = {lv.label: r["probabilities"][str(i)] for i, lv in enumerate(q.levels)}
    score = sum(i * p for i, p in enumerate(probs.values()))
    return Answer("score", q.name, score=score, confidence=conf, probabilities=probs,
                  raw_scores={lv.label: raw[str(i)] for i, lv in enumerate(q.levels)})


class JevDecider:
    # ponytail: one subprocess, one model load (~20 s); inject `backend` in tests, close() or use `with` to free it
    def __init__(self, quant: str = "Q8_0", backend=None, **opts):
        self._b = backend if backend is not None else _load_backend(quant, **opts)

    def decide(self, text: str, questions: Sequence[Question]) -> Response:
        if not questions:
            return Response("Jev-Style-2B-Decision-v3")
        res = self._b.score_many(text, [_to_jev(q) for q in questions])    # state computed once
        state = res[0]["state_tokens"]
        tokens = state + sum(r["input_tokens"] - state for r in res)
        return Response(res[0]["model"], [_answer(q, r) for q, r in zip(questions, res)],
                        {"input_tokens": tokens, "output_tokens": 0, "total_tokens": tokens})

    def close(self):
        if hasattr(self._b, "close"):
            self._b.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
