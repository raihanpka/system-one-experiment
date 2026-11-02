"""Pure scoring math shared by every slice: calibrated temperature, sigmoid probability, calibration error."""
from __future__ import annotations

import math

T0 = 0.8278650620942867                     # calibrated global temperature of Jev v3
TS = (0.5, 0.7, T0, 1.0, 1.5, 2.0, 3.0)     # temperatures swept offline


def pair_p(cell: dict, T: float) -> float:
    """P(yes) from a stored cell {"d": s_true - s_false}: sigmoid(d / T)."""
    d = cell["d"] / T
    return 1 / (1 + math.exp(-d)) if d > -700 else 0.0


def safe_div(a, b):
    return a / b if b else 0.0


def ece(ps, ys, bins: int = 10) -> float:
    """Expected calibration error over equal-width probability bins."""
    if not ps:
        return 0.0
    tot = 0.0
    for b in range(bins):
        idx = [i for i, p in enumerate(ps) if (p >= b / bins and (p < (b + 1) / bins or b == bins - 1))]
        if idx:
            conf = sum(ps[i] for i in idx) / len(idx)
            acc = sum(bool(ys[i]) for i in idx) / len(idx)
            tot += len(idx) / len(ps) * abs(conf - acc)
    return tot
