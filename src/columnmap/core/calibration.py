# Platt scaling: probability = sigmoid(a * x + b), fitted on labeled data and kept in calibration.json
# the file is keyed by (model, quantization, mode, context, prompt version, level); a key that does not match is refused

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class CalibrationKeyError(KeyError):
    def __str__(self) -> str:
        return self.args[0]


def make_key(model: str, quant: str, mode: str, cond: str, prompt: str, level: str) -> str:
    return "|".join((model, quant, mode, cond, prompt, level))


def apply_platt(x: float, a: float, b: float) -> float:
    z = a * x + b
    return 1 / (1 + np.exp(-z)) if z > -700 else 0.0


# logistic regression on one feature by Newton steps; a small ridge keeps the slope near 1 when the data are thin
def fit_platt(xs, ys, iters: int = 60, ridge: float = 1e-3) -> tuple[float, float] | None:
    x, y = np.asarray(xs, float), np.asarray(ys, float)
    if len(set(y.tolist())) < 2:
        return None
    a, b = 1.0, 0.0
    for _ in range(iters):
        p = 1 / (1 + np.exp(-np.clip(a * x + b, -35, 35)))
        w = p * (1 - p) + 1e-9
        g = np.array([((p - y) * x).sum() + ridge * (a - 1), (p - y).sum()])
        h = np.array([[(w * x * x).sum() + ridge, (w * x).sum()], [(w * x).sum(), w.sum()]])
        step = np.linalg.solve(h, g)
        a, b = a - step[0], b - step[1]
        if np.abs(step).max() < 1e-9:
            break
    return float(a), float(b)


# equal-width bins of predicted probability: mean confidence, observed rate, count (empty bins are skipped)
def reliability_bins(ps, ys, bins: int = 10) -> list[dict]:
    out = []
    for k in range(bins):
        idx = [i for i, p in enumerate(ps) if p >= k / bins and (p < (k + 1) / bins or k == bins - 1)]
        if idx:
            out.append({"confidence": sum(ps[i] for i in idx) / len(idx), "observed": sum(bool(ys[i]) for i in idx) / len(idx), "n": len(idx)})
    return out


class Calibration:
    def __init__(self, path):
        self.path = Path(path)
        self.entries: dict = json.loads(self.path.read_text())["entries"] if self.path.exists() else {}

    def set(self, key: str, a: float, b: float, n: int, positives: int, seeds: list[int], curve: list | None = None) -> None:
        self.entries[key] = {"a": round(a, 4), "b": round(b, 4), "n": n, "positives": positives, "seeds": sorted(seeds)}
        if curve:
            self.entries[key]["curve"] = [[round(t, 4), round(p, 4), int(k)] for t, p, k in curve]      # precision against cutoff

    def get(self, key: str, missing_ok: bool = False):
        if key in self.entries:
            return self.entries[key]
        if missing_ok:
            return None
        parts = key.split("|")
        near = [k for k in self.entries if k.split("|")[-1] == parts[-1]]
        raise CalibrationKeyError(f"no calibration for {key}; refit it with --fit-calibration (the model, quantization, mode, context "
                                  f"or prompt version changed). Known keys: {', '.join(near) or 'none'}")

    def save(self) -> None:
        self.path.write_text(json.dumps({"version": 1, "entries": dict(sorted(self.entries.items()))}, indent=2) + "\n", encoding="utf-8")
