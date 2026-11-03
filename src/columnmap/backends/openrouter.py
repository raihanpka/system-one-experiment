# OpenRouter Decisions API (hosted, default backend). Sends the state text to OpenRouter; use synthetic data until retention is checked.
# The API returns calibrated probabilities, not logits. Scores are derived from them so the pipeline keeps its contract:
#   predicate: raw true = logit(p), raw false = 0, so true - false = logit(p)
#   choice, score: raw per option = logit(p), so 0 means p = 0.5 as in the earlier local runs; the ranking is unchanged

from __future__ import annotations

import json
import math
import urllib.error
import urllib.request
from typing import Callable, Sequence

from ..decisions.types import Answer, Question, Response


def _logit(p: float) -> float:
    p = min(max(p, P_CLIP), 1 - P_CLIP)
    return math.log(p / (1 - p))


P_CLIP = 1e-6


def _to_api(q: Question, name: str) -> dict:
    if q.type == "predicate":
        return {
            "type": "noul",
            "instructions": q.instructions,
            "criteria": {"true": "Yes", "false": "No"},
        }
    if q.type == "choice":
        return {
            "type": "choice",
            "instructions": q.instructions,
            "criteria": {c.value: (c.description or c.value) for c in q.choices},
        }
    return {
        "type": "score",
        "instructions": q.instructions,
        "criteria": [lv.label for lv in q.levels],
    }


def _distribution(raw) -> dict:
    # a dict {value: p} or a list of {value|label, probability} or {value|label: p} pairs
    if isinstance(raw, dict):
        return {str(k): float(v) for k, v in raw.items()}
    out = {}
    for row in raw:
        key = row.get("value", row.get("label"))
        out[str(key)] = float(row["probability"])
    return out


def _answer(q: Question, name: str, body: dict) -> Answer:
    if q.type == "predicate":
        p = float(body["noul"])
        p = min(max(p, P_CLIP), 1 - P_CLIP)
        return Answer(
            "predicate",
            name,
            probability=p,
            raw_scores={"true": _logit(p), "false": 0.0},
        )
    probs = _distribution(body["probabilities"])
    if q.type == "choice":
        raw = {k: _logit(v) for k, v in probs.items()}
        return Answer(
            "choice",
            name,
            choice=str(body["choice"]),
            probabilities=probs,
            raw_scores=raw,
        )
    # score: the API keys levels by index ("0", "1", ...) with a legend; relabel to the question's level labels
    labels = [lv.label for lv in q.levels]
    legend = body.get("legend") or {str(i): lab for i, lab in enumerate(labels)}
    probs = {legend.get(k, k): v for k, v in probs.items()}
    raw = {k: _logit(v) for k, v in probs.items()}
    return Answer(
        "score", name, score=float(body["score"]), probabilities=probs, raw_scores=raw
    )


def _post(url: str, headers: dict, body: dict, timeout: float) -> dict:
    req = urllib.request.Request(
        url, data=json.dumps(body).encode("utf-8"), headers=headers, method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        raise RuntimeError(
            f"OpenRouter returned {e.code}: {e.read()[:300].decode('utf-8', 'replace')}"
        ) from None


class OpenRouterDecider:
    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str,
        transport: Callable | None = None,
        timeout: float = 120.0,
    ):
        self._key, self._model, self._url = api_key, model, base_url
        self._transport = transport or _post
        self._timeout = timeout
        self.run_meta = {"model": model, "backend": "openrouter", "quant": "api"}

    def decide(self, text: str, questions: Sequence[Question]) -> Response:
        if not questions:
            return Response(self._model)
        names = [q.name or f"q{i}" for i, q in enumerate(questions)]
        if len(set(names)) != len(names):
            raise ValueError("question names must be unique")
        body = {
            "model": self._model,
            "state": [text],
            "questions": {n: _to_api(q, n) for n, q in zip(names, questions)},
        }
        headers = {
            "Authorization": f"Bearer {self._key}",
            "Content-Type": "application/json",
        }
        res = self._transport(self._url, headers, body, self._timeout)
        answers = res.get("answers", {})
        out = []
        for n, q in zip(names, questions):
            if n not in answers:
                out.append(Answer("refusal", n, message="no answer returned"))
                continue
            out.append(_answer(q, n, answers[n]))
        return Response(res.get("model", self._model), out, res.get("usage", {}))

    def close(self):
        return None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
