# run the model over the samples and record every case in the store (resumable); no file layout, no argument parsing

from __future__ import annotations

import time

from ..decider import Decider
from ..knowledge import Knowledge
from ..pipeline import Pipeline
from .usage import Counting, estimate_minutes, mark, since

WARN_MINUTES = 15


def _stage(decider, cases, store, section, make) -> int:
    n = 0
    for c in cases:
        if store.has(section, c["case_id"]):
            continue
        t0, before = time.time(), mark(decider)
        rec = {"case_id": c["case_id"], "level": c["level"], **make(c)}
        rec["seconds"] = round(time.time() - t0, 1)
        rec.update(since(decider, before))
        store.add(section, rec)
        n += 1
    return n


# the mapping scores of every input column, then the expression ranking on those scores
def run_models(decider: Decider, cases, store, knowledge: Knowledge, log=print) -> None:
    est = estimate_minutes(cases)
    if est > WARN_MINUTES:
        log(
            f"warning: this run is estimated at {est:.0f} minutes; consider fewer samples"
        )
    decider, t0 = Counting(decider), time.time()
    pipeline = Pipeline(decider, knowledge)

    def mapping(c):
        matrix, raw = pipeline.score(c["rows"])
        return {"matrix": matrix, "raw": raw}

    n = _stage(decider, cases, store, "mapping", mapping)
    log(f"mapping: {n} new, {len(cases) - n} resumed ({time.time() - t0:.0f}s)")
    stored = {r["case_id"]: r for r in store.rows("mapping")}
    todo = [c for c in cases if c["case_id"] in stored]

    kinds = store.data["meta"].get("kinds") or pipeline.kinds()
    pipeline.use_kinds(kinds)
    store.data["meta"]["kinds"] = (
        kinds  # read from the descriptions, the same for every file
    )
    store.save()

    def honest(c):
        matrix = stored[c["case_id"]]["matrix"]
        return {"fields": pipeline.decide(c["rows"], matrix, kinds)}

    n = _stage(decider, todo, store, "honest", honest)
    log(f"honest: {n} new, {len(todo) - n} resumed ({time.time() - t0:.0f}s)")
