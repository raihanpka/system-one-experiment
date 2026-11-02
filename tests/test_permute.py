from src.core.template import load_template
from src.decisions.types import Answer, Response
from src.dataset.generate import generate
from src.stability.run import run
from src.core.store import Store
from tests.test_run_transform import Oracle

T = load_template()
CASE = next(c for c in generate(T, n_per_level=2, seed=11) if c["level"] == 3)


class FirstOption:
    def decide(self, text, questions):
        q = questions[0]
        names = [c.value for c in q.choices]
        return Response("first", [Answer("choice", q.name, choice=names[0], confidence=1.0,
                                         probabilities={n: 1.0 if i == 0 else 0.0 for i, n in enumerate(names)},
                                         raw_scores={n: 0.0 for n in names}) for _ in questions[:1]], {})


def test_oracle_is_order_invariant(tmp_path):
    store = Store(tmp_path / "p.json")
    assert run(Oracle(CASE), [CASE], "both", store, T, k=3) == 1
    r = store.rows("permute", "both")[0]
    assert r["transform"] and r["kind"]
    for v in list(r["transform"].values()):
        assert len(v["orders"]) == 3 and v["orders"][0] == sorted(v["orders"][0])      # first order = as generated
        assert len(set(v["picks"])) == 1 and all(v["correct"])
    assert run(Oracle(CASE), [CASE], "both", Store(tmp_path / "p.json"), T, k=3) == 0   # resume


def test_position_bias_is_visible(tmp_path):
    store = Store(tmp_path / "p.json")
    run(FirstOption(), [CASE], "name", store, T, k=4)
    r = store.rows("permute", "name")[0]
    vs = list(r["transform"].values())
    assert all(v["pos"] == [0, 0, 0, 0] for v in vs)                                    # always picks slot 0
    assert any(len(set(v["picks"])) > 1 for v in vs)                                    # ...so the picked text changes with order
