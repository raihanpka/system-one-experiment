from src.core.template import OPS, check, load_template, rule_sources


def test_template_is_valid():
    t = load_template()
    assert check(t) == []                       # no problems reported


def test_template_shape_requirements():
    t = load_template()
    outs = t["outputs"]
    derived = [o for o in outs if o["rule"].get("op") is not None]
    assert 28 <= len(outs) <= 34
    assert len(derived) >= 8
    assert sum(len(rule_sources(o["rule"])) >= 2 for o in outs) >= 2        # built from 2+ inputs
    shared = [s["id"] for s in t["sources"] if sum(s["id"] in rule_sources(o["rule"]) for o in outs) >= 2]
    assert shared                               # at least one input feeds 2+ outputs (1 -> many)
    assert {"pass", "concat", "arith", "format", "lookup"} <= {o["kind"] for o in outs}


def test_check_catches_problems():
    t = load_template()
    t["outputs"][0]["rule"] = {"src": "nope"}
    t["outputs"][1]["rule"] = {"op": "eval", "args": []}
    t["sources"][0]["names"].pop("4")
    probs = check(t)
    assert any("nope" in p for p in probs) and any("eval" in p for p in probs) and any("names" in p for p in probs)
    assert "concat" in OPS
