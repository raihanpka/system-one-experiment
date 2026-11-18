import pytest

from columnmap.expressions import (
    column_unit,
    compatible,
    evaluate,
    output_unit,
    render,
    result_unit,
    safe_evaluate,
)
from columnmap.knowledge import Output

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
    row = {
        "f": "Budi",
        "l": "Santoso",
        "d": "1990-05-17",
        "s": "2024-01-01",
        "e": "2024-01-31",
        "k": "A",
    }
    assert (
        evaluate({"op": "concat", "sep": " ", "args": [S("f"), S("l")]}, row)
        == "Budi Santoso"
    )
    assert evaluate({"op": "year", "args": [S("d")]}, row) == 1990
    assert evaluate({"op": "days_between", "args": [S("s"), S("e")]}, row) == 30
    assert (
        evaluate({"op": "lookup", "map": {"A": "Approved"}, "args": [S("k")]}, row)
        == "Approved"
    )


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
    assert (
        render(
            {"op": "days_between", "args": [S("s"), S("e")]}, {"s": "S", "e": "E"}.get
        )
        == "days_between(S, E)"
    )
    assert (
        render({"op": "lookup", "map": {"A": "x"}, "args": [S("k")]}, {"k": "K"}.get)
        == 'lookup(K, {"A": "x"})'
    )
    assert render(S("a"), {"a": "Prem"}.get) == "Prem"


def test_safe_evaluate_returns_none_instead_of_raising():
    row = {"a": 1, "z": 0, "k": "X", "d": "0000-00-00"}
    assert safe_evaluate({"op": "div", "args": [S("a"), S("z")]}, row) is None
    assert (
        safe_evaluate({"op": "lookup", "map": {"A": "x"}, "args": [S("k")]}, row)
        is None
    )
    assert safe_evaluate({"op": "year", "args": [S("d")]}, row) is None
    assert safe_evaluate({"op": "add", "args": [S("a"), S("a")]}, row) == 2


def test_arithmetic_refuses_text_instead_of_repeating_or_joining_it():
    row = {"id": "8532422441", "amount": 19281179, "a": "1", "b": "2"}
    mul = {"op": "mul", "args": [{"src": "id"}, {"src": "amount"}]}
    add = {"op": "add", "args": [{"src": "a"}, {"src": "b"}]}
    assert safe_evaluate(mul, row) is None  # was a 1.7 GB string
    assert safe_evaluate(add, row) is None  # was "12"
    assert safe_evaluate({"op": "add", "args": [{"lit": 1}, {"lit": 2}]}, row) == 3


def test_column_unit_reads_the_name_before_the_values_and_keeps_codes_out_of_math():
    money = ["1000.50", "2000.00"]
    assert column_unit("CHG_AMT_RAW", money) == "amount"
    assert column_unit("premium_tax_rate", ["0.11", "0.12"]) == "rate"
    assert column_unit("CHG_LVY_PCT", ["0.05", "0.06"]) == "rate"
    assert column_unit("policy_number", ["POL-1", "POL-2"]) == "id"
    assert column_unit("postal_code", ["54556", "04721"]) == "id"
    assert column_unit("excess", ["1000", "2000"]) == "number"
    assert column_unit("coverage_start_date", ["2024-01-01"]) == "date"
    assert column_unit("ratio", ["0.4", "0.5"]) == "rate"


def test_output_unit_comes_from_name_and_description():
    assert (
        output_unit(Output("total_charge", "number", "Transaction amount plus fee"))
        == "amount"
    )
    assert (
        output_unit(Output("claim_paid_ratio", "number", "Paid divided by premium"))
        == "rate"
    )
    assert (
        output_unit(Output("coverage_days", "number", "Days between start and end"))
        == "count"
    )
    assert (
        output_unit(Output("insured_city", "string", "City of the insured")) == "text"
    )


def test_result_unit_rejects_a_rate_where_an_amount_is_needed():
    units = {"fee": "amount", "pct": "rate", "premium": "amount", "days": "count"}
    assert result_unit(S("fee"), units) == "amount"
    assert (
        result_unit({"op": "add", "args": [S("fee"), S("premium")]}, units) == "amount"
    )
    assert result_unit({"op": "add", "args": [S("fee"), S("pct")]}, units) == "number"
    assert (
        result_unit(
            {
                "op": "mul",
                "args": [S("premium"), {"op": "add", "args": [L(1), S("pct")]}],
            },
            units,
        )
        == "amount"
    )
    assert result_unit({"op": "div", "args": [S("fee"), S("premium")]}, units) == "rate"
    assert (
        result_unit({"op": "days_between", "args": [S("days"), S("days")]}, units)
        == "count"
    )


def test_compatible_lets_a_plain_number_be_an_amount_but_not_a_rate():
    assert compatible("amount", "amount") and compatible("amount", "number")
    assert not compatible("amount", "rate") and not compatible("amount", "date")
    assert compatible("rate", "rate") and not compatible("rate", "amount")
    assert compatible("date", "date") and not compatible("date", "text")
