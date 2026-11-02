"""Run the model over the samples and record everything in the store. No file layout, no argument parsing."""
from __future__ import annotations

import time

from .mapping import run as run_mapping
from .stability import run as run_stability
from .transform import run as run_transform


def run_models(decider, cases, conds, mode, with_stability, store, template, log=print) -> None:
    """For each context: mapping (+ kind), transformation ranking, optionally option-order stability.
    Cases already in the store are skipped, so an interrupted run can be continued."""
    t0 = time.time()
    steps = [("mapping", lambda *a: run_mapping.run(*a, mode=mode)), ("transform", run_transform.run)]
    if with_stability:
        steps.append(("permute", run_stability.run))
    for cond in conds:
        for name, fn in steps:
            n = fn(decider, cases, cond, store, template)
            log(f"[{cond}] {name}: {n} new, {len(cases) - n} resumed ({time.time() - t0:.0f}s)")
