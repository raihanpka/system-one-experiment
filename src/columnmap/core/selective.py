# selective prediction: fill a field only when its confidence is high enough, otherwise leave it for a person

from __future__ import annotations

MIN_SUPPORT = 15            # a cutoff backed by fewer fields than this is not trusted


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


# the loosest cutoff whose precision meets the target with enough fields behind it; None when no cutoff does
def choose_tau(curve, target: float, min_support: int = MIN_SUPPORT) -> float | None:
    ok = [c[0] for c in curve if c[1] >= target and c[2] >= min_support]
    return min(ok) if ok else None
