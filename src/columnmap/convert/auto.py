# fill a field automatically only when its calibrated confidence reaches the cutoff; the rest goes to a review list

from __future__ import annotations


from ..core.calibration import apply_platt
from ..core.scoring import logit, rate



# per field: filled automatically or not, the calibrated confidence, and why not
# no calibration entry: every answered field is filled (uncalibrated); an entry without a trusted cutoff: nothing is filled
def field_decisions(fields: dict, entry: dict | None, tau: float | None, valid: dict | None = None) -> dict:
    # valid: field -> bool from the type check of its converted cells; an invalid value never fills automatically
    out = {}
    for f, r in fields.items():
        if r["best"] is None:
            out[f] = {"auto": False, "conf_cal": 0.0, "reason": "no column passes" if not r["cols"] else "no rule fits"}
        elif valid is not None and not valid.get(f, False):
            out[f] = {"auto": False, "conf_cal": r["conf"], "reason": "invalid value"}
        elif entry is None:
            out[f] = {"auto": True, "conf_cal": r["conf"], "reason": ""}
        else:
            c = apply_platt(logit(r["conf"]), entry["a"], entry["b"])
            ok = tau is not None and c >= tau
            out[f] = {"auto": ok, "conf_cal": c, "reason": "" if ok else "low confidence"}
    return out


# up to three candidate answers per field for a person to choose from, best first
def review_rows(fields: dict, decisions: dict) -> list[list]:
    rows = []
    for f, r in fields.items():
        if r["cands"] and r["shortlist"]:
            ps = r["shortlist_p"] or [r["conf"]]
            cand = sorted(((r["cands"][i], p) for i, p in zip(r["shortlist"], ps)), key=lambda t: -t[1])
        else:
            cand = [(c, p) for c, p in r["top_cols"]]
        cand = (cand + [("", "")] * 3)[:3]
        rows.append([f, decisions[f]["reason"] or "filled", round(decisions[f]["conf_cal"], 4),
                     *[x for t, p in cand for x in (t, round(p, 4) if p != "" else "")]])
    return rows


REVIEW_COLUMNS = ["output", "status", "confidence", "candidate_1", "p_1", "candidate_2", "p_2", "candidate_3", "p_3"]


# precision of what was filled automatically, and how much of the template that covers
def selective_metrics(case: dict, conv: list[list], decisions: dict, correct: dict) -> dict:
    with_src = [o["name"] for o in case["outputs"] if o["available"]]
    auto = [f for f, d in decisions.items() if d["auto"]]
    good = sum(correct[f] for f in auto)
    covered = [f for f in auto if f in set(with_src)]
    return {"with_source": len(with_src), "auto_total": len(auto), "auto_correct": good, "auto_wrong": len(auto) - good,
            "coverage": rate(len(covered), len(with_src)), "precision": rate(good, len(auto)),
            "review": len(decisions) - len(auto)}


# (coverage, precision, cutoff) for every cutoff, from items (calibrated confidence, field correct, field has a source)
def coverage_curve(items) -> list[tuple[float, float, float]]:
    total = sum(1 for _, _, src in items if src)
    ordered = sorted(items, key=lambda t: -t[0])
    out, kept, good, covered, i = [], 0, 0, 0, 0
    while i < len(ordered):
        tau = ordered[i][0]
        while i < len(ordered) and ordered[i][0] == tau:
            kept += 1
            good += bool(ordered[i][1])
            covered += bool(ordered[i][2])
            i += 1
        out.append((rate(covered, total) or 0.0, good / kept, tau))
    return out
