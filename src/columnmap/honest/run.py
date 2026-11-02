# the honest stage: from the mapping scores alone, choose columns, rank candidate expressions, and pick one
# no ground truth goes in: only field names, input rows and the score matrix

from __future__ import annotations

from ..core.candidates import candidates
import time

from ..core.scoring import T0, pair_p
from ..core.usage import mark, since
from ..decisions.base import Decider
from .prompts import check_question, expression_text, option_text, pick_question, pool_state

SHORTLIST = 3


# columns whose score for the field passes `keep` (0.0 is probability 0.5), best first, at most max_cols
def chosen_columns(matrix: dict, field: str, keep: float = 0.0, max_cols: int = 3) -> list[tuple[str, float]]:
    scored = sorted(((m[field]["d"], c) for c, m in matrix.items() if m[field]["d"] >= keep), reverse=True)[:max_cols]
    return [(c, d) for d, c in scored]


def _empty() -> dict:
    return {"cols": [], "p_cols": [], "cands": [], "shortlist": [], "shortlist_p": [], "top_cols": [], "best_idx": None, "best": None,
            "best_text": None, "p_expr": None, "method": "unmapped", "conf": 0.0}


# the three best columns for the field whatever the cutoff, with their probabilities, for the review list
def _top(matrix: dict, field: str) -> list[list]:
    best = sorted(((m[field]["d"], c) for c, m in matrix.items()), reverse=True)[:3]
    return [[c, pair_p({"d": d}, T0)] for d, c in best]


def _field(decider, field: str, rows: list[dict], matrix: dict, keep: float, max_cols: int) -> dict:
    chosen = chosen_columns(matrix, field, keep, max_cols)
    if not chosen:
        return {**_empty(), "top_cols": _top(matrix, field)}
    names = [c for c, _ in chosen]
    p_cols = [pair_p({"d": d}, T0) for _, d in chosen]
    cands = candidates([(n, [r[n] for r in rows]) for n in names])
    rec = {**_empty(), "cols": names, "p_cols": p_cols, "cands": [expression_text(c) for c in cands], "top_cols": _top(matrix, field)}
    if not cands:
        return rec                                                    # no rule in the catalog fits these columns: needs a person
    if len(cands) == 1:
        best, shortlist, ps, p_expr = 0, [0], [], None                # nothing to rank, so no model call
    else:
        state = pool_state(names, rows)
        if len(cands) > SHORTLIST:
            options = [option_text(c, rows) for c in cands]
            probs = decider.decide(state, [pick_question(field, options)]).answers[0].probabilities
            shortlist = sorted(range(len(cands)), key=lambda i: -probs[options[i]])[:SHORTLIST]
        else:
            shortlist = list(range(len(cands)))
        checks = [check_question(field, cands[i], rows, f"c{k}") for k, i in enumerate(shortlist)]
        ps = [a.probability for a in decider.decide(state, checks).answers]
        best = shortlist[max(range(len(ps)), key=ps.__getitem__)]
        p_expr = max(ps)
    ast = cands[best]
    conf = p_expr if p_expr is not None else 1.0
    for p in p_cols:
        conf *= p
    rec.update({"shortlist": shortlist, "shortlist_p": ps, "best_idx": best, "best": ast, "best_text": expression_text(ast), "p_expr": p_expr,
                "method": "copy" if "src" in ast else "expression", "conf": conf})
    return rec


def run_fields(decider: Decider, fields: list[str], rows: list[dict], matrix: dict, keep: float = 0.0, max_cols: int = 3) -> dict:
    return {f: _field(decider, f, rows, matrix, keep, max_cols) for f in fields}


# one record per case and context, built from the stored mapping scores
def run_case(decider: Decider, case: dict, cond: str, map_row: dict, mode: str = "options", keep: float = 0.0) -> dict:
    fields = [o["name"] for o in case["outputs"]]
    return {"case_id": case["case_id"], "level": case["level"], "cond": cond, "mode": mode,
            "fields": run_fields(decider, fields, case["rows"], map_row["matrix"], keep)}


def run(decider: Decider, cases, cond: str, store, mode: str = "options", keep: float = 0.0) -> int:
    section = {"options": "mapping", "pairs": "mapping_pairs", "twostage": "mapping_twostage"}[mode]
    maps = {r["case_id"]: r for r in store.rows(section, cond)}
    n = 0
    for c in cases:
        if c["case_id"] in maps and not store.has("honest", cond, c["case_id"]):
            t0, before = time.time(), mark(decider)
            rec = run_case(decider, c, cond, maps[c["case_id"]], mode, keep)
            rec["seconds"] = round(time.time() - t0, 1)
            rec.update(since(decider, before))
            store.add("honest", cond, rec)
            n += 1
    return n
