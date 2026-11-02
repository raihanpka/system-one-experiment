"""One command: samples -> Jev -> result.json -> figures and tables.   uv run python src --samples abcd"""
from __future__ import annotations

import argparse
import time
import tomllib
from datetime import datetime
from pathlib import Path

from .mapping import run as run_mapping
from .stability import run as permute
from .transform import run as run_transform
from .core.template import load_template
from .dataset.export import write_cases, write_template
from .dataset.generate import generate
from .dataset.samples import DEFAULT, SAMPLES
from .report.build import make_report
from .core.store import Store

ROOT = Path(__file__).resolve().parents[1]


def main(argv=None, decider_factory=None) -> int:
    cfg = tomllib.loads((ROOT / "config.toml").read_text())
    ap = argparse.ArgumentParser(prog="python src", description="Run Jev on sample files and write the report.")
    ap.add_argument("--samples", default=DEFAULT, help=f"letters from {''.join(SAMPLES)} (default {DEFAULT}); a identical, b synonyms, c legacy codes, d mixed+decoys")
    ap.add_argument("--all", action="store_true", help="all seven samples")
    ap.add_argument("--conds", default=",".join(cfg["run"]["conds"]), help="name, samples (desc, both give column descriptions)")
    ap.add_argument("--mode", default=cfg["run"]["mode"], choices=("options", "pairs"), help="options: fast, one question per column; pairs: slow, calibrated probabilities")
    ap.add_argument("--permute", action="store_true", help="also test option-order stability (slower)")
    ap.add_argument("--report-only", action="store_true", help="rebuild figures and tables from result.json, no model")
    ap.add_argument("--out", default="results/latest")
    a = ap.parse_args(argv)

    letters = "".join(SAMPLES) if a.all else a.samples.replace(",", "")
    if not letters or any(x not in SAMPLES for x in letters):
        ap.error(f"samples must be letters from {''.join(SAMPLES)}")
    conds = a.conds.split(",")
    out = Path(a.out) if Path(a.out).is_absolute() else ROOT / a.out
    t = load_template()
    cases = {c["case_id"]: c for c in generate(t, 1, cfg["run"]["seed"])}
    sel = [cases[SAMPLES[x][0]] for x in letters]
    store = Store(out / "result.json")

    write_template(t, out / "samples")
    write_cases(sel, out / "samples")
    print(f"samples {letters}, contexts {conds}, output {out}")

    if not a.report_only:
        t0 = time.time()
        with (decider_factory or __import__("src.decisions.jev", fromlist=["JevDecider"]).JevDecider.from_config)() as d:
            for cond in conds:
                steps = (("mapping", lambda *x: run_mapping.run(*x, mode=a.mode)), ("transform", run_transform.run))
                for name, fn in steps + ((("permute", permute.run),) if a.permute else ()):
                    n = fn(d, sel, cond, store, t)
                    print(f"[{cond}] {name}: {n} new, {len(sel) - n} resumed ({time.time() - t0:.0f}s)")
        store.data["meta"].update({"model": "Jev-Style-2B-Decision-v3", "backend": "llama.cpp", **cfg["model"],
                                   "seed": cfg["run"]["seed"], "mode": a.mode, "updated": datetime.now().isoformat(timespec="seconds")})
        store.save()

    paths = make_report(store, cases, letters, conds, out / "report")
    print(f"report: {out / 'report' / 'summary.md'}  ({len(paths)} files)")
    print(f"raw results: {out / 'result.json'}")
    return 0
