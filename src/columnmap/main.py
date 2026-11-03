# entry point: choose samples and a run folder, run the model, write the results

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from .core.paths import DATA, ROOT
from .backends.openrouter import OpenRouterDecider
from .core.settings import load_settings, openrouter_settings
from .calibrate import fit as fit_calibration
from .core.calibration import Calibration
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
  columnmap --all                 run all four samples
  columnmap --conds name          show the model only the column names (no sample values)
  columnmap --mode pairs          slower, calibrated probabilities
  columnmap --mode both           run both modes and compare them
  columnmap --fit-calibration results/A results/B    fit calibration.json from dev or calibration runs A and B
  columnmap --calibrate-from results/X    fit the rejection cutoffs on run X (a fair test on another case)
  columnmap --seed 5              build the samples from another seed (not written to data/)
  columnmap --split calibration   one run per seed of a split from appsettings.json: dev, calibration, or test (test needs --final)
  columnmap --out results/X       continue (or extend) the existing run folder X
  columnmap --report-only         rebuild files, figures and tables of the newest run, no model
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
        "--conds",
        default=None,
        help="what the model sees: name = column name; samples = name + 5 values. "
        "Default: the contexts already in the run folder, else name,samples",
    )
    ap.add_argument(
        "--mode",
        default=cfg["run"]["mode"],
        choices=("options", "pairs", "both"),
        help="options: one question per input column, fast; pairs: one yes/no per column and field, slow, calibrated; "
        "both: run and compare the two",
    )
    ap.add_argument(
        "--seed",
        type=int,
        help="seed that builds the samples (default: Run:Seed in appsettings.json)",
    )
    ap.add_argument(
        "--split",
        choices=("dev", "calibration", "test"),
        help="run every seed of this split from appsettings.json, one run each",
    )
    ap.add_argument(
        "--final",
        action="store_true",
        help="confirm that you really want to run the test split",
    )
    ap.add_argument(
        "--calibrate-from",
        metavar="RUN",
        help="run folder whose results fit the cutoffs that reject fields without a source",
    )
    ap.add_argument(
        "--keep",
        type=float,
        help="score a column needs to be chosen for a field (default: Run:KeepD in appsettings.json; 0.0 is probability 0.5)",
    )
    ap.add_argument(
        "--fit-calibration",
        nargs="+",
        metavar="RUN",
        help="fit calibration parameters from these dev or calibration runs, write calibration.json, and stop",
    )
    ap.add_argument(
        "--calibration",
        default=str(ROOT / "calibration.json"),
        help="calibration file (default calibration.json in the project)",
    )
    ap.add_argument(
        "--report-only",
        action="store_true",
        help="rebuild files, figures and tables from result.json, no model",
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

    if a.fit_calibration:
        calibration = Calibration(_abs(a.calibration))
        try:
            lines = fit_calibration([_abs(p) for p in a.fit_calibration], calibration)
        except ValueError as e:
            ap.error(str(e))
        calibration.save()
        print("\n".join(lines) or "nothing to fit")
        print(f"written {calibration.path}")
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
    conds = (
        a.conds or ",".join(store.data["meta"].get("contexts") or cfg["run"]["conds"])
    ).split(",")
    modes = ["options", "pairs"] if a.mode == "both" else [a.mode]
    other = Store(_abs(a.calibrate_from) / "result.json") if a.calibrate_from else None
    if other is not None and other.data["meta"].get("seed", cfg["run"]["seed"]) != seed:
        ap.error(
            "--calibrate-from needs a run built from the same seed, otherwise its answers do not belong to these files"
        )
    stored = store.data["meta"].get("keep_d")
    keep = (
        stored
        if (a.report_only or a.out) and stored is not None
        else (a.keep if a.keep is not None else cfg["run"]["keep_d"])
    )
    t = load_template()
    cases = {c["case_id"]: c for c in generate(t, 1, seed)}
    sel = [cases[SAMPLES[x][0]] for x in letters]
    if baseline:
        write_samples(
            t, cases, letters, _abs(a.data)
        )  # data/ holds the baseline seed only
    print(
        f"samples {letters}, contexts {conds}, mode {a.mode}, seed {seed}, run folder {run}"
    )

    if not a.report_only:
        if decider_factory is None:
            try:
                opts = openrouter_settings(cfg)
            except ValueError as e:
                ap.error(str(e))
            decider_factory = lambda: OpenRouterDecider(**opts)
        with decider_factory() as d:
            run_models(d, sel, conds, modes, store, t, keep=keep)
            identity = d.run_meta
        store.data["meta"].update(
            {
                **identity,
                "samples": letters,
                "seed": seed,
                "split": a.split or ("baseline" if baseline else "custom"),
                "prompt_version": cfg["run"]["prompt_version"],
                "keep_d": keep,
                "mode": ",".join(modes),
                "contexts": list(
                    dict.fromkeys([*store.data["meta"].get("contexts", []), *conds])
                ),
                "updated": datetime.now().isoformat(timespec="seconds"),
            }
        )
        store.save()

    paths = make_report(
        store,
        cases,
        letters,
        conds,
        run,
        other,
        Calibration(_abs(a.calibration)),
        cfg["run"]["target_precision"],
    )
    print(f"{len(paths)} files written. Start with: {run / 'report' / 'summary.md'}")
    print(f"raw model output: {run / 'result.json'}")
