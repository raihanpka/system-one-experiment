import pytest

from columnmap.core.dsl import evaluate, render, safe_evaluate

S = lambda x: {"src": x}  # noqa: E731
L = lambda x: {"lit": x}  # noqa: E731


def test_pass_lit_and_arith():
    row = {"a": 1000.0, "b": 0.11, "c": 4}
    assert evaluate(S("a"), row) == 1000.0
    gross = {"op": "mul", "args": [S("a"), {"op": "add", "args": [L(1), S("b")]}]}
    assert evaluate(gross, row) == pytest.approx(1110.0)
    assert evaluate({"op": "sub", "args": [S("a"), S("c")]}, row) == 996.0
    assert evaluate({"op": "div", "args": [S("a"), S("c")]}, row) == 250.0


def test_concat_year_days_lookup():
    row = {"f": "Budi", "l": "Santoso", "d": "1990-05-17", "s": "2024-01-01", "e": "2024-01-31", "k": "A"}
    assert evaluate({"op": "concat", "sep": " ", "args": [S("f"), S("l")]}, row) == "Budi Santoso"
    assert evaluate({"op": "year", "args": [S("d")]}, row) == 1990
    assert evaluate({"op": "days_between", "args": [S("s"), S("e")]}, row) == 30
    assert evaluate({"op": "lookup", "map": {"A": "Approved"}, "args": [S("k")]}, row) == "Approved"


def test_errors_are_explicit_not_eval():
    row = {"a": 1, "z": 0, "k": "X"}
    with pytest.raises(ZeroDivisionError):
        evaluate({"op": "div", "args": [S("a"), S("z")]}, row)
    with pytest.raises(KeyError):
        evaluate({"op": "lookup", "map": {"A": "x"}, "args": [S("k")]}, row)
    with pytest.raises(ValueError):
        evaluate({"op": "__import__", "args": []}, row)
    with pytest.raises(KeyError):
        evaluate(S("missing"), row)
    with pytest.raises(ValueError):
        evaluate({"op": "year", "args": [{"lit": "not a date"}]}, row)


def test_render_uses_names():
    gross = {"op": "mul", "args": [S("a"), {"op": "add", "args": [L(1), S("b")]}]}
    assert render(gross, {"a": "Prem", "b": "Tax"}.get) == "Prem * (1 + Tax)"
    name = {"op": "concat", "sep": " ", "args": [S("f"), S("l")]}
    assert render(name, {"f": "fn", "l": "ln"}.get) == 'concat(fn, " ", ln)'
    assert render({"op": "days_between", "args": [S("s"), S("e")]}, {"s": "S", "e": "E"}.get) == "days_between(S, E)"
    assert render({"op": "lookup", "map": {"A": "x"}, "args": [S("k")]}, {"k": "K"}.get) == 'lookup(K, {"A": "x"})'
    assert render(S("a"), {"a": "Prem"}.get) == "Prem"


def test_safe_evaluate_returns_none_instead_of_raising():
    row = {"a": 1, "z": 0, "k": "X", "d": "0000-00-00"}
    assert safe_evaluate({"op": "div", "args": [S("a"), S("z")]}, row) is None
    assert safe_evaluate({"op": "lookup", "map": {"A": "x"}, "args": [S("k")]}, row) is None
    assert safe_evaluate({"op": "year", "args": [S("d")]}, row) is None
    assert safe_evaluate({"op": "add", "args": [S("a"), S("a")]}, row) == 2
