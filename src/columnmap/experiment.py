# run the model over the samples and record everything in the store; no file layout, no argument parsing

from __future__ import annotations

import time

from .decisions.base import Decider
from .mapping import run as run_mapping
from .stability import run as run_stability
from .transform import run as run_transform


# for each context: mapping (+ kind) in every mode, transformation ranking, optionally option-order stability
# the whole-table context only changes how columns are mapped, so transform and stability reuse the other contexts
def run_models(decider: Decider, cases, conds, modes, with_stability, store, template, log=print) -> None:
    t0 = time.time()
    for cond in conds:
        steps = [(f"mapping ({m})", lambda *a, m=m: run_mapping.run(*a, mode=m)) for m in modes]
        if cond != "table":
            steps.append(("transform", run_transform.run))
            if with_stability:
                steps.append(("permute", run_stability.run))
        for name, fn in steps:
            n = fn(decider, cases, cond, store, template)
            log(f"[{cond}] {name}: {n} new, {len(cases) - n} resumed ({time.time() - t0:.0f}s)")
