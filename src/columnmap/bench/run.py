# run the model over the samples and record every case in the store (resumable); no file layout, no argument parsing

from __future__ import annotations

import time

from ..decider import Decider
from ..knowledge import Knowledge
from ..mapper import choose_expressions, score_columns
from .usage import Counting, estimate_minutes, mark, since

WARN_MINUTES = 15


def _stage(decider, cases, cond, store, section, make) -> int:
    n = 0
    for c in cases:
        if store.has(section, cond, c["case_id"]):
            continue
        t0, before = time.time(), mark(decider)
        rec = {"case_id": c["case_id"], "level": c["level"], "cond": cond, **make(c)}
        rec["seconds"] = round(time.time() - t0, 1)
        rec.update(since(decider, before))
        store.add(section, cond, rec)
        n += 1
    return n


# for each context: the mapping scores of every input column, then the expression ranking on those scores
def run_models(
    decider: Decider, cases, conds, store, knowledge: Knowledge, log=print
) -> None:
    est = estimate_minutes(cases, conds)
    if est > WARN_MINUTES:
        log(
            f"warning: this run is estimated at {est:.0f} minutes; consider fewer samples or contexts"
        )
    decider, t0 = Counting(decider), time.time()
    for cond in conds:

        def mapping(c):
            matrix, raw = score_columns(decider, c["rows"], knowledge, cond)
            return {"matrix": matrix, "raw": raw}

        n = _stage(decider, cases, cond, store, "mapping", mapping)
        log(
            f"[{cond}] mapping: {n} new, {len(cases) - n} resumed ({time.time() - t0:.0f}s)"
        )
        stored = {r["case_id"]: r for r in store.rows("mapping", cond)}
        todo = [c for c in cases if c["case_id"] in stored]

        def honest(c):
            return {
                "fields": choose_expressions(
                    decider, c["rows"], stored[c["case_id"]]["matrix"], knowledge
                )
            }

        n = _stage(decider, todo, cond, store, "honest", honest)
        log(
            f"[{cond}] honest: {n} new, {len(cases) - n} resumed ({time.time() - t0:.0f}s)"
        )
