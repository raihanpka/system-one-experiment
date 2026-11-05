# entry point: choose samples and a run folder, run the model, write the results

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from ..config import DATA, ROOT, load_settings, openrouter_settings
from ..decider import OpenRouterDecider
from .export import write_samples
from .generate import generate
from .report import make_report
from .run import run_models
from .runs import new_run_dir, newest_run
from .samples import DEFAULT, SAMPLES, word
from .store import Store
from .template import knowledge_from_template, load_template

EPILOG = """examples (run with: uv run columnmap ...):
  columnmap                       run samples a, b, c, d into a new folder results/<samples>_<timestamp>/
  columnmap --samples ac          run only samples a and c
  columnmap --all                 run all four samples
  columnmap --seed 5              build the samples from another seed (not written to data/)
  columnmap --split holdout       one run per seed of a split from appsettings.json: dev, holdout, or test (test needs --final)
  columnmap --out results/X       continue (or extend) the existing run folder X
  columnmap --report-only         rebuild the files and summary of the newest run, no model
  columnmap --report-only --fill-score 0.8   same, with another score a field needs to be filled
  columnmap --deepeval results/X  DeepEval metrics on a finished run, no model
"""


def _abs(p: str) -> Path:
    return Path(p) if Path(p).is_absolute() else ROOT / p


def main(argv=None, decider_factory=None) -> int:
    cfg = load_settings()
    ap = argparse.ArgumentParser(
        prog="columnmap",
        description="Run the model on sample files and write the results.",
        epilog=EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--samples",
        help=f"letters from {''.join(SAMPLES)} (default {DEFAULT}): "
        + ", ".join(f"{k} {v[1]}" for k, v in SAMPLES.items()),
    )
    ap.add_argument("--all", action="store_true", help="all four samples")
    ap.add_argument(
        "--seed",
        type=int,
        help="seed that builds the samples (default: Run:Seed in appsettings.json)",
    )
    ap.add_argument(
        "--split",
        choices=("dev", "holdout", "test"),
        help="run every seed of this split from appsettings.json, one run each",
    )
    ap.add_argument(
        "--final",
        action="store_true",
        help="confirm that you really want to run the test split",
    )
    ap.add_argument(
        "--fill-score",
        type=float,
        help="score the best expression of a field needs to fill it (default: Run:FillScore in appsettings.json)",
    )
    ap.add_argument(
        "--deepeval",
        metavar="RUN",
        help="run the DeepEval metrics on this finished run folder (no model call) and stop",
    )
    ap.add_argument(
        "--report-only",
        action="store_true",
        help="rebuild files and summary from result.json, no model",
    )
    ap.add_argument(
        "--root", default="results", help="folder that holds all runs (default results)"
    )
    ap.add_argument(
        "--data",
        default=str(DATA),
        help="folder with the template and the samples (default data/)",
    )
    ap.add_argument(
        "--out",
        help="use this run folder (continue it) instead of creating a new one named <samples>_<timestamp>",
    )
    a = ap.parse_args(argv)

    if a.deepeval:
        from .deepeval import evaluate_run

        evaluate_run(_abs(a.deepeval), a.fill_score)
        return 0
    seeds = (
        cfg["splits"][a.split]
        if a.split
        else [a.seed if a.seed is not None else cfg["run"]["seed"]]
    )
    if a.split == "test" and not a.final:
        ap.error("the test split is looked at once, at the end: add --final to run it")
    if a.out and len(seeds) > 1:
        ap.error("--out continues one run, so it takes one seed")
    for seed in seeds:
        _run(ap, a, cfg, seed, decider_factory)
    return 0


def _run(ap, a, cfg, seed, decider_factory) -> None:
    root = _abs(a.root)
    baseline = seed == cfg["run"]["seed"]
    if a.out:
        run = _abs(a.out)
        run.mkdir(parents=True, exist_ok=True)
    elif a.report_only:
        run = newest_run(root)
        if run is None:
            ap.error(f"no run found under {root}; run without --report-only first")
    else:
        letters_new = (
            "".join(SAMPLES) if a.all else (a.samples or DEFAULT).replace(",", "")
        )
        if any(x not in SAMPLES for x in letters_new):
            ap.error(f"samples must be letters from {''.join(SAMPLES)}")
        names = (
            "all"
            if len(letters_new) == len(SAMPLES)
            else "-".join(word(SAMPLES[x][0]) for x in letters_new)
        )
        run = new_run_dir(root, names if baseline else f"{names}-seed{seed}")
    store = Store(run / "result.json")
    seed = store.data["meta"].get("seed", seed)  # an existing run keeps its own seed
    baseline = seed == cfg["run"]["seed"]

    letters = (
        "".join(SAMPLES)
        if a.all
        else (
            a.samples
            or (store.data["meta"].get("samples") if a.report_only else None)
            or DEFAULT
        ).replace(",", "")
    )
    if not letters or any(x not in SAMPLES for x in letters):
        ap.error(f"samples must be letters from {''.join(SAMPLES)}")
    fill_score = a.fill_score if a.fill_score is not None else cfg["run"]["fill_score"]
    t = load_template()
    cases = {c["case_id"]: c for c in generate(t, 1, seed)}
    sel = [cases[SAMPLES[x][0]] for x in letters]
    if baseline:
        write_samples(
            t, cases, letters, _abs(a.data)
        )  # data/ holds the baseline seed only
    print(f"samples {letters}, seed {seed}, run folder {run}")

    if not a.report_only:
        if decider_factory is None:
            try:
                opts = openrouter_settings(cfg)
            except ValueError as e:
                ap.error(str(e))
            decider_factory = lambda: OpenRouterDecider(**opts)
        with decider_factory() as d:
            run_models(d, sel, store, knowledge_from_template(t))
            identity = d.run_meta
        store.data["meta"].update(
            {
                **identity,
                "samples": letters,
                "seed": seed,
                "split": a.split or ("baseline" if baseline else "custom"),
                "updated": datetime.now().isoformat(timespec="seconds"),
            }
        )
        store.save()

    paths = make_report(store, cases, letters, run, fill_score)
    print(f"{len(paths)} files written. Start with: {run / 'report' / 'summary.md'}")
    print(f"raw model output: {run / 'result.json'}")
