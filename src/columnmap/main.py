# entry point: choose samples and a run folder, run the model, write the results

from __future__ import annotations

import argparse
import tomllib
from datetime import datetime
from pathlib import Path

from .core.paths import CONFIG, DATA, ROOT
from .core.runs import new_run_dir, newest_run
from .core.store import Store
from .core.template import load_template
from .dataset.export import write_samples
from .dataset.generate import generate
from .dataset.samples import DEFAULT, SAMPLES, word
from .experiment import run_models
from .report.build import make_report

EPILOG = """examples (run with: uv run columnmap ...):
  columnmap                       run samples a, b, c, d into a new folder results/<samples>_<timestamp>/
  columnmap --samples ac          run only samples a and c
  columnmap --all                 run all seven samples
  columnmap --conds name          show the model only the column names (no sample values)
  columnmap --mode pairs          slower, calibrated probabilities
  columnmap --mode both           run both modes and compare them
  columnmap --conds name,samples,table    also show the model the whole table
  columnmap --calibrate-from results/X    fit the rejection cutoffs on run X (a fair test on another case)
  columnmap --permute             also test how stable answers are when option order changes
  columnmap --out results/X       continue (or extend) the existing run folder X
  columnmap --report-only         rebuild files, figures and tables of the newest run, no model
"""


def _abs(p: str) -> Path:
    return Path(p) if Path(p).is_absolute() else ROOT / p


def main(argv=None, decider_factory=None) -> int:
    cfg = tomllib.loads(CONFIG.read_text())
    ap = argparse.ArgumentParser(prog="columnmap", description="Run Jev on sample files and write the results.",
                                 epilog=EPILOG, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--samples", help=f"letters from {''.join(SAMPLES)} (default {DEFAULT}): "
                    + ", ".join(f"{k} {v[1]}" for k, v in SAMPLES.items()))
    ap.add_argument("--all", action="store_true", help="all seven samples")
    ap.add_argument("--conds", default=None,
                    help="what the model sees: name = column name; samples = name + 5 values; table = name + the whole table "
                         "(desc, both add column descriptions). Default: the contexts already in the run folder, else name,samples")
    ap.add_argument("--mode", default=cfg["run"]["mode"], choices=("options", "pairs", "both"),
                    help="options: one question per input column, fast; pairs: one yes/no per column and field, slow, calibrated; "
                         "both: run and compare the two")
    ap.add_argument("--calibrate-from", metavar="RUN", help="run folder whose results fit the cutoffs that reject fields without a source")
    ap.add_argument("--permute", action="store_true", help="also test option-order stability (slower)")
    ap.add_argument("--report-only", action="store_true", help="rebuild files, figures and tables from result.json, no model")
    ap.add_argument("--root", default="results", help="folder that holds all runs (default results)")
    ap.add_argument("--data", default=str(DATA), help="folder with the template and the samples (default data/)")
    ap.add_argument("--out", help="use this run folder (continue it) instead of creating a new one named <samples>_<timestamp>")
    a = ap.parse_args(argv)

    root = _abs(a.root)
    if a.out:
        run = _abs(a.out)
        run.mkdir(parents=True, exist_ok=True)
    elif a.report_only:
        run = newest_run(root)
        if run is None:
            ap.error(f"no run found under {root}; run without --report-only first")
    else:
        letters_new = "".join(SAMPLES) if a.all else (a.samples or DEFAULT).replace(",", "")
        if any(x not in SAMPLES for x in letters_new):
            ap.error(f"samples must be letters from {''.join(SAMPLES)}")
        names = "all" if len(letters_new) == len(SAMPLES) else "-".join(word(SAMPLES[x][0]) for x in letters_new)
        run = new_run_dir(root, names)
    store = Store(run / "result.json")

    letters = "".join(SAMPLES) if a.all else (a.samples or (store.data["meta"].get("samples") if a.report_only else None) or DEFAULT).replace(",", "")
    if not letters or any(x not in SAMPLES for x in letters):
        ap.error(f"samples must be letters from {''.join(SAMPLES)}")
    conds = (a.conds or ",".join(store.data["meta"].get("contexts") or cfg["run"]["conds"])).split(",")
    modes = ["options", "pairs"] if a.mode == "both" else [a.mode]
    other = Store(_abs(a.calibrate_from) / "result.json") if a.calibrate_from else None
    t = load_template()
    cases = {c["case_id"]: c for c in generate(t, 1, cfg["run"]["seed"])}
    sel = [cases[SAMPLES[x][0]] for x in letters]
    write_samples(t, cases, letters, _abs(a.data))                      # data/ stays the single source of truth
    print(f"samples {letters}, contexts {conds}, mode {a.mode}, run folder {run}")

    if not a.report_only:
        if decider_factory is None:
            from .decisions.jev import JevDecider
            decider_factory = lambda: JevDecider(**cfg["model"])              # noqa: E731
        with decider_factory() as d:
            run_models(d, sel, conds, modes, a.permute, store, t)
        store.data["meta"].update({"model": "Jev-Style-2B-Decision-v3", "backend": "llama.cpp", **cfg["model"], "samples": letters,
                                   "seed": cfg["run"]["seed"], "mode": ",".join(modes), "contexts": list(dict.fromkeys([*store.data["meta"].get("contexts", []), *conds])),
                                   "updated": datetime.now().isoformat(timespec="seconds")})
        store.save()

    paths = make_report(store, cases, letters, conds, run, other)
    print(f"{len(paths)} files written. Start with: {run / 'report' / 'summary.md'}")
    print(f"raw model output: {run / 'result.json'}")
    return 0
