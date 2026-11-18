# entry point: choose samples and a run folder, run the model, write the results

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from ..config import DATA, ROOT, fill_from_settings, load_settings, openrouter_settings
from ..decider import OpenRouterDecider
from ..mapper import Fill
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
  columnmap --report-only --fill-score 0.8 --copy-fill-score 0.5   same, with other scores a field needs to be filled
  columnmap --deepeval results/X  DeepEval metrics on a finished run, no model

map a partner table (CSV, TSV, or XLSX):
  columnmap --table in.csv --knowledge my_knowledge/ --output converted.csv
  columnmap --table in.csv --schema --json        print the schema we extracted, no model
  columnmap --table in.csv --graph --review empty  run through the LangGraph harness, fail closed
"""


def _abs(p: str) -> Path:
    return Path(p) if Path(p).is_absolute() else ROOT / p


def _fill(cfg, a) -> Fill:
    base = fill_from_settings(cfg)
    return Fill(
        a.fill_score if a.fill_score is not None else base.derived,
        a.copy_fill_score if a.copy_fill_score is not None else base.copy,
        base.none_ceiling,
    )


def _print_schema(schema, as_json: bool) -> None:
    if as_json:
        print(
            json.dumps(
                {
                    "rows": schema.rows,
                    "columns": [asdict(column) for column in schema.columns],
                },
                indent=2,
            )
        )
        return
    for column in schema.columns:
        print(
            f"{column.name}\t{column.type}\t{column.unit}"
            f"\tnull={column.null_rate}\tdistinct={column.distinct}"
            f"\t{', '.join(column.samples)}"
        )


# the graph pauses only for fields close to the fill policy; the CLI never guesses
def _review_decision(run, mode: str) -> dict:
    fields = run.result["__interrupt__"][0].value.get("fields", {})
    if mode == "accept":
        return {name: "accept" for name in fields}
    if mode == "empty":
        return {name: "empty" for name in fields}
    decision = {}
    for name, info in fields.items():
        print(
            f"{name}: {info.get('reason')} · {info.get('expression')} "
            f"· p_present={info.get('p_present')} p_none={info.get('p_none')}"
        )
        answer = input(f"  keep {name}? [a]ccept/[e]mpty [e]: ").strip().lower()
        decision[name] = "accept" if answer.startswith("a") else "empty"
    return decision


def _map_table(ap, a, cfg, decider_factory=None) -> int:
    from ..knowledge import Knowledge
    from ..mapper import Mapper
    from ..schema import Column, Schema, extract
    from ..table import read_header, read_table
    from .csvio import write_csv

    rows = read_table(a.table)
    if not rows:
        header = read_header(a.table)
        if a.schema:
            _print_schema(
                Schema(
                    tuple(
                        Column(name, "unknown", "unknown", (), 0.0, 0)
                        for name in header
                    ),
                    0,
                ),
                a.json,
            )
            return 0
        knowledge_dir = Path(a.knowledge)
        if not knowledge_dir.exists():
            ap.error(f"knowledge folder not found: {knowledge_dir}")
        knowledge = Knowledge.load(knowledge_dir)
        targets = [o.name for o in knowledge.outputs]
        out = Path(a.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        write_csv(out, targets, [])
        summary = {
            "table": str(a.table),
            "knowledge": str(knowledge_dir),
            "output": str(out),
            "rows": 0,
            "fields": len(targets),
            "filled": 0,
            "empty": len(targets),
            "degraded": False,
            "errors": [],
            "reviewed": {},
        }
        print(
            json.dumps(summary, indent=2)
            if a.json
            else f"0 rows, {len(targets)} target fields written -> {out}"
        )
        return 0
    schema = extract(rows)
    if a.schema:
        _print_schema(schema, a.json)
        return 0
    knowledge_dir = Path(a.knowledge)
    if not knowledge_dir.exists():
        ap.error(f"knowledge folder not found: {knowledge_dir}")
    knowledge = Knowledge.load(knowledge_dir)
    fill = _fill(cfg, a)
    if decider_factory is None:
        try:
            opts = openrouter_settings(cfg)
        except ValueError as error:
            ap.error(str(error))
        decider_factory = lambda: OpenRouterDecider(**opts)
    reviewed, degraded, errors = {}, False, []
    with decider_factory() as decider:
        if a.graph:
            try:
                from ..harness import run_graph
            except ImportError as error:
                ap.error(str(error))
            run = run_graph(
                rows,
                decider,
                knowledge,
                fill=fill,
                thread_id=f"table:{Path(a.table).name}",
                review=a.review,
            )
            if run.paused:
                reviewed = _review_decision(run, a.review)
                run.resume(reviewed)
            header, converted = run.header, run.converted
            degraded, errors = run.degraded, run.errors
        else:
            mapping = Mapper(decider, knowledge).map(rows)
            header, converted = mapping.convert(rows, fill)
    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    write_csv(out, header, converted)
    filled = sum(
        1 for i in range(len(header)) if any(row[i] is not None for row in converted)
    )
    summary = {
        "table": str(a.table),
        "knowledge": str(knowledge_dir),
        "output": str(out),
        "rows": len(rows),
        "fields": len(header),
        "filled": filled,
        "empty": len(header) - filled,
        "degraded": degraded,
        "errors": errors,
        "reviewed": reviewed,
    }
    if a.json:
        print(json.dumps(summary, indent=2, default=str))
    else:
        print(f"{filled} / {len(header)} fields filled, {len(rows)} rows -> {out}")
        if degraded:
            print("degraded: " + "; ".join(errors))
    return 0


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
        help="probability of the column set that fills a derived field (default: Run:FillScore in appsettings.json)",
    )
    ap.add_argument(
        "--copy-fill-score",
        type=float,
        help="the same for a field copied from one column (default: Run:CopyFillScore in appsettings.json)",
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
    ap.add_argument(
        "--table",
        help="map a partner CSV/TSV/XLSX file instead of the synthetic samples",
    )
    ap.add_argument(
        "--knowledge",
        default=str(DATA / "knowledge"),
        help="folder with outputs.json, reference/*.csv, and prompts.json (default data/knowledge)",
    )
    ap.add_argument(
        "--output",
        default="converted.csv",
        help="where --table writes the converted CSV (default converted.csv)",
    )
    ap.add_argument(
        "--graph",
        action="store_true",
        help="map --table through the LangGraph harness (needs the graph extra)",
    )
    ap.add_argument(
        "--review",
        choices=("ask", "accept", "empty"),
        default="empty",
        help="what to do with graph review fields (default empty, fail closed)",
    )
    ap.add_argument(
        "--schema",
        action="store_true",
        help="print the extracted input schema and stop",
    )
    ap.add_argument(
        "--json", action="store_true", help="print a machine-readable summary"
    )
    a = ap.parse_args(argv)
    if a.schema and not a.table:
        ap.error("--schema needs --table")
    if a.graph and not a.table:
        ap.error("--graph needs --table")
    if a.review != "empty" and not a.table:
        ap.error("--review needs --table")
    if a.table:
        return _map_table(ap, a, cfg, decider_factory)

    if a.deepeval:
        from .deepeval import evaluate_run

        evaluate_run(_abs(a.deepeval), _fill(cfg, a))
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
    fill = _fill(cfg, a)
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

    paths = make_report(store, cases, letters, run, fill)
    print(f"{len(paths)} files written. Start with: {run / 'report' / 'summary.md'}")
    print(f"raw model output: {run / 'result.json'}")
