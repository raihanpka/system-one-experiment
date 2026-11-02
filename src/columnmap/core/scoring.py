# Pure scoring math shared by every slice: calibrated temperature, sigmoid probability, calibration error

from __future__ import annotations

import math

MASKED_D = -20.0                       # score of a field that was not shortlisted in two-stage mode
T0 = 0.8278650620942867                     # calibrated global temperature of Jev v3
TS = (0.5, 0.7, T0, 1.0, 1.5, 2.0, 3.0)     # temperatures swept offline


# P(yes) from a stored cell {"d": s_true - s_false}: sigmoid(d / T)
def pair_p(cell: dict, T: float) -> float:
    d = cell["d"] / T
    return 1 / (1 + math.exp(-d)) if d > -700 else 0.0


def safe_div(a, b):
    return a / b if b else 0.0


# like safe_div, but None when there is nothing to divide by (an undefined rate, not zero)
def rate(a, b):
    return a / b if b else None


# Expected calibration error over equal-width probability bins
def ece(ps, ys, bins: int = 10) -> float:
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


# a field is kept when the best input column passes a rule on (best score d1, lift above the median column)
def FIXED(d1: float, lift: float) -> bool:
    return d1 >= 0.0                                                              # probability 0.5


def accept_score(cut: float):
    return lambda d1, lift: d1 >= cut


def accept_lift(cut: float):
    return lambda d1, lift: lift >= cut
