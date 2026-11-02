# run the model over the samples and record everything in the store; no file layout, no argument parsing

from __future__ import annotations

import time

from .core.usage import Counting, estimate_minutes
from .decisions.base import Decider
from .honest import run as run_honest
from .mapping import run as run_mapping

WARN_MINUTES = 15


# for each context: mapping in every mode, then the honest stage on the scores of the first mode
def run_models(decider: Decider, cases, conds, modes, store, template, keep: float = 0.0, orders: int = 1, log=print) -> None:
    est = estimate_minutes(cases, conds, modes)
    if est > WARN_MINUTES:
        log(f"warning: this run is estimated at {est:.0f} minutes; consider fewer samples, contexts or modes")
    decider, t0 = Counting(decider), time.time()
    for cond in conds:
        for m in modes:
            n = run_mapping.run(decider, cases, cond, store, template, mode=m, orders=orders)
            log(f"[{cond}] mapping ({m}): {n} new, {len(cases) - n} resumed ({time.time() - t0:.0f}s)")
        n = run_honest.run(decider, cases, cond, store, mode=modes[0], keep=keep)
        log(f"[{cond}] honest: {n} new, {len(cases) - n} resumed ({time.time() - t0:.0f}s)")
