# fit Platt parameters from finished runs and check them on another run; one entry per (setup, level) in calibration.json

from __future__ import annotations

import math
from pathlib import Path

from .convert.honest import convert_fields
from .convert.metrics import field_correct
from .core.calibration import Calibration, apply_platt, fit_platt, make_key, reliability_bins
from .core.scoring import T0, ece, pair_p
from .core.selective import precision_curve
from .core.store import Store
from .core.template import load_template
from .dataset.generate import generate

SECTIONS = {"options": "mapping", "pairs": "mapping_pairs"}


def _logit(p: float) -> float:
    p = min(max(p, 1e-6), 1 - 1e-6)
    return math.log(p / (1 - p))


# the score of every (input column, output field) pair and whether the column is a true source
def pair_xy(rows, cases) -> tuple[list[float], list[bool]]:
    xs, ys = [], []
    for r in rows:
        for col, row in r["matrix"].items():
            for o in cases[r["case_id"]]["outputs"]:
                xs.append(row[o["name"]]["d"])
                ys.append(col in o["sources"])
    return xs, ys


# the confidence the honest stage gave each field it answered, and whether every cell of that field is right
def field_xy(hrows, cases) -> tuple[list[float], list[bool]]:
    xs, ys = [], []
    for rec in hrows:
        case = cases[rec["case_id"]]
        names = [o["name"] for o in case["outputs"]]
        _, conv = convert_fields(rec["fields"], case["rows"], names)
        ok = field_correct(case, conv)
        for f in names:
            if rec["fields"][f]["best"] is not None:
                xs.append(_logit(rec["fields"][f]["conf"]))
                ys.append(ok[f])
    return xs, ys


def load_run(run_dir) -> tuple[Store, dict, dict]:
    store = Store(Path(run_dir) / "result.json")
    meta = store.data["meta"]
    cases = {c["case_id"]: c for c in generate(load_template(), 1, meta["seed"])}
    return store, meta, cases


def setup_key(meta: dict, mode: str, cond: str, prompt: str, level: str) -> str:
    return make_key(meta["model"], meta["quant"], mode, cond, prompt, level)


# every run named here must be a dev or calibration run: a test run is never used for fitting
def fit(run_dirs, calibration: Calibration, prompt: str) -> list[str]:
    data: dict[str, tuple[list, list, set]] = {}
    for run in run_dirs:
        store, meta, cases = load_run(run)
        if meta.get("split") == "test":
            raise ValueError(f"{run} is a test run; the test split is never used for fitting")
        for mode, section in SECTIONS.items():
            for cond in store.data.get(section, {}):
                xs, ys = pair_xy(store.rows(section, cond), cases)
                key = setup_key(meta, mode, cond, prompt, "pair")
                bucket = data.setdefault(key, ([], [], set()))
                bucket[0].extend(xs), bucket[1].extend(ys), bucket[2].add(meta["seed"])
        for cond in store.data.get("honest", {}):
            rows = store.rows("honest", cond)
            xs, ys = field_xy(rows, cases)
            key = setup_key(meta, rows[0].get("mode", "options"), cond, prompt, "field")
            bucket = data.setdefault(key, ([], [], set()))
            bucket[0].extend(xs), bucket[1].extend(ys), bucket[2].add(meta["seed"])
    log = []
    for key, (xs, ys, seeds) in sorted(data.items()):
        fitted = fit_platt(xs, ys)
        if fitted is None:
            log.append(f"skipped {key}: only one class in {len(xs)} points")
            continue
        curve = precision_curve([apply_platt(x, *fitted) for x in xs], ys) if key.endswith("|field") else None
        calibration.set(key, *fitted, n=len(xs), positives=int(sum(ys)), seeds=sorted(seeds), curve=curve)
        log.append(f"{key}: a={fitted[0]:.2f} b={fitted[1]:.2f} on {len(xs)} points ({int(sum(ys))} positive), seeds {sorted(seeds)}")
    return log


# uncalibrated and calibrated reliability of one context, per level; None when no entry matches the setup
def evaluate(store, meta: dict, cases: dict, calibration: Calibration, mode: str, cond: str, prompt: str, only=None) -> list[dict]:
    out = []
    section = SECTIONS[mode]
    for level, xy in (("pair", lambda: pair_xy(store.rows(section, cond, only), cases)),
                      ("field", lambda: field_xy(store.rows("honest", cond, only), cases))):
        entry = calibration.get(setup_key(meta, mode, cond, prompt, level), missing_ok=True)
        xs, ys = xy()
        if entry is None or not xs:
            continue
        raw = [pair_p({"d": x}, T0) if level == "pair" else 1 / (1 + math.exp(-x)) for x in xs]
        cal = [apply_platt(x, entry["a"], entry["b"]) for x in xs]
        out.append({"level": level, "n": len(xs), "ece_raw": ece(raw, ys), "ece_cal": ece(cal, ys), "bins_raw": reliability_bins(raw, ys),
                    "bins_cal": reliability_bins(cal, ys), "held_out": meta["seed"] not in entry["seeds"], "cond": cond})
    return out
