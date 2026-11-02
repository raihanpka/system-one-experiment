from src.core.store import Store
from src.core.template import load_template
from src.dataset.generate import generate
from src.transform.run import run
from tests.fakes import TransformOracle as Oracle

T = load_template()
CASES = generate(T, n_per_level=2, seed=11)


def test_oracle_gets_everything_right_and_resume(tmp_path):
    case = next(c for c in CASES if c["level"] == 3)
    store = Store(tmp_path / "t.json")
    assert run(Oracle(case), [case], "both", store, T) == 1
    r = store.rows("transform", "both")[0]
    derived = [o["name"] for o in case["outputs"] if o["available"] and o["kind"] != "pass"]
    assert set(r["transform"]) == set(derived) and derived
    for v in r["transform"].values():
        assert len(v["cands"]) >= 4 and len(set(v["cands"])) == len(v["cands"])
        assert v["choice_idx"] == v["truth_idx"] and v["pred_idx"] == v["truth_idx"]
        assert v["choice_value_ok"] and v["pred_value_ok"]
    assert run(Oracle(case), [case], "both", Store(tmp_path / "t.json"), T) == 0   # resume


def test_wrong_pick_is_detected_by_value_check(tmp_path):
    case = next(c for c in CASES if c["level"] == 2)
    store = Store(tmp_path / "t.json")
    run(Oracle(case, right=False), [case], "name", store, T)
    r = store.rows("transform", "name")[0]
    assert all(v["choice_idx"] != v["truth_idx"] and not v["choice_value_ok"] for v in r["transform"].values())
