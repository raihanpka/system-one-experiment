# run the model over the samples and record everything in the store; no file layout, no argument parsing

from __future__ import annotations

import time

from .decisions.base import Decider
from .honest import run as run_honest
from .mapping import run as run_mapping


# for each context: mapping in every mode, then the honest stage on the scores of the first mode
def run_models(decider: Decider, cases, conds, modes, store, template, log=print) -> None:
    t0 = time.time()
    for cond in conds:
        for m in modes:
            n = run_mapping.run(decider, cases, cond, store, template, mode=m)
            log(f"[{cond}] mapping ({m}): {n} new, {len(cases) - n} resumed ({time.time() - t0:.0f}s)")
        n = run_honest.run(decider, cases, cond, store, mode=modes[0])
        log(f"[{cond}] honest: {n} new, {len(cases) - n} resumed ({time.time() - t0:.0f}s)")
