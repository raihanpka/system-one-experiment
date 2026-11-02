"""One command: samples -> Jev -> result.json -> converted files, figures and tables.

    uv run python src                      # samples a,b,c,d into results/<timestamp>_abcd/
"""
from __future__ import annotations

import argparse
import time
import tomllib
from datetime import datetime
from pathlib import Path

from .core.runs import latest_run, link_latest, new_run_dir
from .core.store import Store
from .core.template import load_template
from .dataset.generate import generate
from .dataset.samples import DEFAULT, SAMPLES
from .mapping import run as run_mapping
from .report.build import make_report
from .stability import run as permute
from .transform import run as run_transform

ROOT = Path(__file__).resolve().parents[1]
EPILOG = """examples:
  uv run python src                              samples a,b,c,d, fast mode, new run folder
  uv run python src --samples ac                 only a (identical) and c (legacy)
  uv run python src --all                        all seven samples
  uv run python src --mode pairs                 slow, calibrated probabilities
  uv run python src --permute                    also test option-order stability
  uv run python src --conds name                 only the column names, no sample values
  uv run python src --out results/20261102-120000_abcd      continue or extend an existing run
  uv run python src --report-only                rebuild files, figures and tables of the newest run (no model)
"""


def _abs(p: str) -> Path:
    return Path(p) if Path(p).is_absolute() else ROOT / p


def main(argv=None, decider_factory=None) -> int:
    cfg = tomllib.loads((ROOT / "config.toml").read_text())
    ap = argparse.ArgumentParser(prog="python src", description="Run Jev on sample files and write the results.",
                                 epilog=EPILOG, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--samples", help=f"letters from {''.join(SAMPLES)} (default {DEFAULT}): "
                    + ", ".join(f"{k} {v[1]}" for k, v in SAMPLES.items()))
    ap.add_argument("--all", action="store_true", help="all seven samples")
    ap.add_argument("--conds", default=",".join(cfg["run"]["conds"]),
                    help="what the model sees: name = column name; samples = name + 5 values (desc, both add column descriptions)")
    ap.add_argument("--mode", default=cfg["run"]["mode"], choices=("options", "pairs"),
                    help="options: one question per input column, fast; pairs: one yes/no per column and field, slow, calibrated")
    ap.add_argument("--permute", action="store_true", help="also test option-order stability (slower)")
    ap.add_argument("--report-only", action="store_true", help="rebuild files, figures and tables from result.json, no model")
    ap.add_argument("--root", default="results", help="folder that holds all runs (default results)")
    ap.add_argument("--out", help="use this run folder (continue it) instead of creating a new timestamped one")
    a = ap.parse_args(argv)

    root = _abs(a.root)
    if a.out:
        run = _abs(a.out)
        run.mkdir(parents=True, exist_ok=True)
    elif a.report_only:
        run = latest_run(root)
        if run is None:
            ap.error(f"no run found under {root}; run without --report-only first")
    else:
        letters_new = "".join(SAMPLES) if a.all else (a.samples or DEFAULT).replace(",", "")
        run = new_run_dir(root, letters_new)
    store = Store(run / "result.json")

    letters = "".join(SAMPLES) if a.all else (a.samples or (store.data["meta"].get("samples") if a.report_only else None) or DEFAULT).replace(",", "")
    if not letters or any(x not in SAMPLES for x in letters):
        ap.error(f"samples must be letters from {''.join(SAMPLES)}")
    conds = a.conds.split(",")
    t = load_template()
    cases = {c["case_id"]: c for c in generate(t, 1, cfg["run"]["seed"])}
    sel = [cases[SAMPLES[x][0]] for x in letters]
    print(f"samples {letters}, contexts {conds}, mode {a.mode}, run folder {run}")

    if not a.report_only:
        t0 = time.time()
        if decider_factory is None:
            from .decisions.jev import JevDecider
            decider_factory = lambda: JevDecider(**cfg["model"])              # noqa: E731
        with decider_factory() as d:
            for cond in conds:
                steps = (("mapping", lambda *x: run_mapping.run(*x, mode=a.mode)), ("transform", run_transform.run))
                for name, fn in steps + ((("permute", permute.run),) if a.permute else ()):
                    n = fn(d, sel, cond, store, t)
                    print(f"[{cond}] {name}: {n} new, {len(sel) - n} resumed ({time.time() - t0:.0f}s)")
        store.data["meta"].update({"model": "Jev-Style-2B-Decision-v3", "backend": "llama.cpp", **cfg["model"], "samples": letters,
                                   "seed": cfg["run"]["seed"], "mode": a.mode, "contexts": conds,
                                   "updated": datetime.now().isoformat(timespec="seconds")})
        store.save()

    paths = make_report(store, cases, letters, conds, run, template=t)
    if run.parent == root and not a.report_only:
        link_latest(root, run)
    print(f"{len(paths)} files written. Start with: {run / 'report' / 'summary.md'}")
    print(f"raw model output: {run / 'result.json'}")
    return 0
