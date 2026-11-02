from src.core.template import load_template
from src.core.template import KIND_DESC, KIND_LABEL
from src.decisions.types import Answer, Response
from src.core.dsl import render
from src.dataset.generate import generate
from src.run_transform import run
from src.core.store import Store

T = load_template()
CASES = generate(T, n_per_level=2, seed=11)


def truth_texts(case):
    name_of = {k["src"]: k["name"] for k in case["columns"] if k["src"]}
    return {o["name"]: render(o["rule"], name_of.get) for o in case["outputs"] if o["available"] and o["kind"] != "pass"}


class Oracle:
    """Picks the true expression / true kind for choice, scores 0.9/0.1 for predicates."""

    def __init__(self, case, right=True):
        self.truth, self.right = truth_texts(case), right
        self.kinds = {o["name"]: KIND_LABEL[o["kind"]] for o in case["outputs"]}

    def decide(self, text, questions):
        first = questions[0].instructions
        target = next(n for n in self.kinds if f"'{n}'" in first)
        out = []
        for q in questions:
            if q.type == "choice":
                vals = [c.value for c in q.choices]
                pick = self.kinds[target] if set(vals) <= set(KIND_DESC) else self.truth[target]
                if not self.right:
                    pick = next(v for v in vals if v != pick)
                out.append(Answer("choice", q.name, choice=pick, confidence=1.0,
                                  probabilities={v: (0.7 if v == pick else 0.1) for v in vals},
                                  raw_scores={v: 0.0 for v in vals}))
            else:
                hit = q.instructions.endswith(self.truth[target])
                p = 0.9 if hit == self.right else 0.1
                out.append(Answer("predicate", q.name, probability=p, raw_scores={"false": 0.0, "true": 0.0}))
        return Response("oracle", out, {})


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
