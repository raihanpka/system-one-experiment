# selective prediction: fill a field only when its confidence is high enough, otherwise leave it for a person

from __future__ import annotations

import math

MIN_SUPPORT = 15  # a cutoff backed by fewer fields than this is not trusted
CONF = 0.95  # confidence of the precision lower bound used to choose a cutoff


# for each confidence cutoff: precision of the fields at or above it and how many there are; ascending cutoff
def precision_curve(confs, ok, points: int = 200) -> list[list[float]]:
    pairs = sorted(zip(confs, ok), reverse=True)
    curve, kept, good, i = [], 0, 0, 0
    while i < len(pairs):
        tau = pairs[i][0]
        while i < len(pairs) and pairs[i][0] == tau:
            kept += 1
            good += bool(pairs[i][1])
            i += 1
        curve.append([tau, good / kept, kept])
    curve.reverse()
    if len(curve) > points:
        step = len(curve) / points
        curve = [curve[int(k * step)] for k in range(points)]
    return curve


# P(X >= k) for X ~ Binomial(n, p), computed with log-gamma so large n does not overflow
def _binom_tail(k: int, n: int, p: float) -> float:
    if p <= 0.0:
        return 0.0
    if p >= 1.0:
        return 1.0
    lp, lq = math.log(p), math.log1p(-p)
    total = 0.0
    for i in range(k, n + 1):
        total += math.exp(
            math.lgamma(n + 1)
            - math.lgamma(i + 1)
            - math.lgamma(n - i + 1)
            + i * lp
            + (n - i) * lq
        )
    return min(total, 1.0)


# one-sided Clopper-Pearson lower bound: with confidence conf the true precision is at least this
def precision_lower_bound(correct: int, n: int, conf: float = CONF) -> float:
    if n <= 0 or correct <= 0:
        return 0.0
    if correct >= n:
        return (1 - conf) ** (1 / n)
    alpha = 1 - conf
    lo, hi = 0.0, 1.0
    for _ in range(100):
        mid = (lo + hi) / 2
        if _binom_tail(correct, n, mid) < alpha:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


# the loosest cutoff whose precision lower bound meets the target with enough fields behind it; None when no cutoff does.
# A target of 1.0 is never met: a lower bound is below 1 for any finite number of fields.
def choose_tau(
    curve, target: float, min_support: int = MIN_SUPPORT, conf: float = CONF
) -> float | None:
    ok = [
        c[0]
        for c in curve
        if c[2] >= min_support
        and precision_lower_bound(round(c[1] * c[2]), int(c[2]), conf) >= target
    ]
    return min(ok) if ok else None
