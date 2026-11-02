import math

import pytest

from src.check_template import load_template
from src.decisions.types import Answer, Response
from src.gen_gt import generate
from src.metrics import ece, lexical_mapping, mapping_metrics, pair_p, sweep
from src.run_mapping import run_case

T = load_template()
CASES = {c["case_id"]: c for c in generate(T, n_per_level=1, seed=2)}
T0 = 0.8278650620942867


def logit_raw(p):
    return {"false": 0.0, "true": math.log(p / (1 - p)) * T0}      # so that sigmoid(raw/T0) == p


class Oracle:
    """p=0.9 for a true source, 0.05 otherwise; reads input name and output name from the prompt."""

    def __init__(self, case):
        self.src = {o["name"]: set(o["sources"]) for o in case["outputs"]}
        self.cols = [c["name"] for c in case["columns"]]

    def decide(self, text, questions):
        col = text.splitlines()[0].removeprefix("Input column: ")
        out = []
        for q in questions:
            if q.type == "predicate":
                p = 0.9 if col in self.src[q.name] else 0.05
                out.append(Answer("predicate", q.name, probability=p, raw_scores=logit_raw(p)))
            else:
                names = [c.value for c in q.choices]
                out.append(Answer("choice", q.name, choice=names[0], probabilities={n: 1 / len(names) for n in names}, raw_scores={}))
        return Response("o", out, {})


def rows_for(level, orc=True):
    case = next(c for c in CASES.values() if c["level"] == level)
    r = run_case(Oracle(case), case, "both", T, mode="pairs")
    return [r]


def test_pair_p_matches_stored_probability_and_temperature_flattens():
    cell = {"p": 0.9, "d": logit_raw(0.9)["true"]}
    assert pair_p(cell, T0) == pytest.approx(0.9)
    assert 0.5 < pair_p(cell, 2.0) < 0.9                      # higher T = less confident


def test_ece():
    assert ece([0.9] * 10, [True] * 9 + [False]) == pytest.approx(0.0)       # confident and right 90%
    assert ece([0.9] * 10, [True] * 5 + [False] * 5) == pytest.approx(0.4)   # confident, right 50%
    assert ece([], []) == 0.0


def test_oracle_scores_perfect_and_decoys_are_rejected():
    m = mapping_metrics(rows_for(6), CASES, T0)
    assert m["top1"] == 1.0 and m["exact"] == 1.0 and m["top3"] == 1.0
    assert m["reject_unavailable"] == 1.0 and m["decoy_flag"] == 0.0
    assert m["precision"] == 1.0 and m["recall"] == 1.0 and m["f1"] == 1.0
    assert m["n_outputs"] > 0 and m["n_unavailable"] > 0


def test_sweep_has_one_row_per_temperature_and_ece_changes():
    rows = rows_for(3)
    sw = sweep(rows, CASES, [0.5, T0, 2.0])
    assert [s["T"] for s in sw] == [0.5, T0, 2.0]
    assert sw[1]["ece"] == pytest.approx(0.1 - 0.0, abs=0.1)  # near-calibrated at stored T (oracle probabilities 0.9 / 0.05)
    assert sw[2]["ece"] > sw[1]["ece"]                        # too-flat probabilities are worse here


def test_lexical_baseline_is_perfect_at_level0_and_worse_when_names_drift():
    lv0 = lexical_mapping([c for c in CASES.values() if c["level"] == 0])
    lv5 = lexical_mapping([c for c in CASES.values() if c["level"] == 5])
    assert lv0["top1"] > 0.7                         # pass-through names are identical; derived fields are guesses
    assert lv5["top1"] < 0.5 and lv5["top1"] < lv0["top1"] - 0.3
