from src.core.template import load_template
from src.core.prompts import CONDS, input_state
from src.decisions.types import Answer, Response
from src.dataset.generate import generate
from src.mapping.run import run
from src.core.store import Store

T = load_template()
CASES = generate(T, n_per_level=2, seed=3)


class Fake:
    def __init__(self):
        self.calls = []

    def decide(self, text, questions):
        self.calls.append((text, questions))
        ans = []
        for q in questions:
            if q.type == "predicate":
                ans.append(Answer("predicate", q.name, probability=0.25, raw_scores={"false": 0.0, "true": -1.1}))
            else:
                names = [c.value for c in q.choices]
                ans.append(Answer("choice", q.name, choice=names[0], confidence=0.5,
                                  probabilities={n: 1 / len(names) for n in names}, raw_scores={n: 0.0 for n in names}))
        return Response("fake", ans, {})


def test_conditions_change_state_text():
    col = {"name": "PolicyNo", "src": "policy_number", "decoy": None}
    texts = {c: input_state(col, c, desc="Unique id of the policy", samples=["POL-1", "POL-2", "POL-3"]) for c in CONDS}
    assert "PolicyNo" in texts["name"] and "Description" not in texts["name"] and "POL-1" not in texts["name"]
    assert "Description: Unique id" in texts["desc"] and "POL-1" not in texts["desc"]
    assert "POL-1" in texts["samples"] and "Description" not in texts["samples"]
    assert "Description" in texts["both"] and "POL-1" in texts["both"]


def test_matrix_shape_kind_and_resume(tmp_path):
    store = Store(tmp_path / "r.json")
    d = Fake()
    run(d, CASES[:3], "name", store, T, mode="pairs")
    rows = store.rows("mapping", "name")
    assert len(rows) == 3
    c, r = CASES[0], rows[0]
    assert r["case_id"] == c["case_id"] and r["cond"] == "name"
    assert set(r["matrix"]) == {k["name"] for k in c["columns"]}              # every input column, decoys included
    assert all(set(v) == {o["name"] for o in c["outputs"]} for v in r["matrix"].values())
    cell = next(iter(next(iter(r["matrix"].values())).values()))
    assert cell == {"p": 0.25, "d": -1.1}
    avail = {o["name"] for o in c["outputs"] if o["available"]}
    assert set(r["kind"]) == avail and all("choice" in v and "p" in v for v in r["kind"].values())
    n = len(d.calls)
    run(d, CASES[:3], "name", Store(tmp_path / "r.json"), T, mode="pairs")                  # new process: resume, nothing new
    assert len(d.calls) == n and len(Store(tmp_path / "r.json").rows("mapping", "name")) == 3
    run(d, CASES[:4], "name", Store(tmp_path / "r.json"), T, mode="pairs")
    assert len(Store(tmp_path / "r.json").rows("mapping", "name")) == 4 and len(d.calls) > n


def test_options_mode_asks_one_question_per_column_and_reads_raw_scores(tmp_path):
    d = Fake()
    store = Store(tmp_path / "o.json")
    case = CASES[0]
    run(d, [case], "name", store, T)                                            # default mode = options
    r = store.rows("mapping", "name")[0]
    assert r["mode"] == "options"
    map_calls = [qs for _, qs in d.calls if qs[0].name == "field"]
    assert len(map_calls) == len(case["columns"]) and all(len(qs) == 1 for qs in map_calls)
    assert len(map_calls[0][0].choices) == len(case["outputs"])                 # all output fields are the options
    cell = next(iter(next(iter(r["matrix"].values())).values()))
    assert cell == {"p": 0.5, "d": 0.0}                                         # sigmoid(0) = 0.5
    assert set(r["matrix"]) == {k["name"] for k in case["columns"]}
    assert all(set(v) == {o["name"] for o in case["outputs"]} for v in r["matrix"].values())
