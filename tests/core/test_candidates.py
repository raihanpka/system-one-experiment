import math

import pytest

from columnmap.core.candidates import _size, candidates, column_type
from columnmap.core.dsl import render, safe_evaluate
from columnmap.core.template import load_template, rule_sources
from columnmap.dataset.generate import generate

T = load_template()


def texts(cs):
    return {render(c, lambda s: s) for c in cs}


def test_column_type_tolerates_a_few_odd_cells():
    assert column_type(["1", 2, 3.5, "N/A", ""]) == "number"
    assert column_type(["2025-01-02", "2024-12-31", "1900-01-01", ""]) == "date"
    assert column_type(["Budi", "Siti", "0"]) == "text"
    assert column_type([]) == "text"


def test_one_column_gives_copy_and_year_for_dates():
    assert texts(candidates([("a", ["x", "y"])])) == {"a"}
    d = texts(candidates([("born", ["1990-05-17", "1985-01-02"])]))
    assert d == {"born", "year(born)"}


def test_two_text_columns_give_concat_in_both_orders_with_several_separators():
    t = texts(candidates([("f", ["Budi"]), ("l", ["Santoso"])]))
    assert (
        'concat(f, " ", l)' in t
        and 'concat(l, " ", f)' in t
        and 'concat(f, ", ", l)' in t
        and 'concat(f, "-", l)' in t
    )


def test_two_number_columns_give_arithmetic_both_ways_and_the_tax_and_discount_forms():
    t = texts(candidates([("a", [1000, 2000]), ("b", [0.1, 0.2])]))
    assert {
        "a + b",
        "a - b",
        "b - a",
        "a * b",
        "a / b",
        "b / a",
        "a * (1 + b)",
        "a * (1 - b)",
    } <= t


def test_two_dates_give_the_difference_in_days_both_ways():
    t = texts(candidates([("s", ["2024-01-01"]), ("e", ["2024-01-31"])]))
    assert {"days_between(s, e)", "days_between(e, s)"} <= t


def test_three_text_columns_include_the_address_shape():
    t = texts(
        candidates(
            [("city", ["Bandung"]), ("prov", ["Jawa Barat"]), ("zip", ["40111"])]
        )
    )
    assert 'concat(concat(city, ", ", prov), " ", zip)' in t


def test_limit_order_and_determinism():
    cols = [("a", [1, 2]), ("b", [3, 4]), ("c", [5, 6])]
    one, two = candidates(cols), candidates(cols)
    assert one == two and 1 <= len(one) <= 24
    assert len(candidates(cols, limit=5)) == 5
    sizes = [_size(c) for c in candidates(cols, limit=8)]
    assert sizes == sorted(sizes)  # simplest first
    assert candidates([]) == []


def test_candidates_see_only_names_and_values_not_ground_truth():
    import inspect

    assert list(inspect.signature(candidates).parameters) == [
        "cols",
        "limit",
        "out_type",
    ]


def _same(a, b):
    return len(a) == len(b) and all(
        x == y
        or (
            isinstance(x, float)
            and isinstance(y, float)
            and math.isclose(x, y, rel_tol=1e-9)
        )
        for x, y in zip(a, b)
    )


@pytest.mark.parametrize("seed", [1234, 11])
def test_the_catalog_covers_every_derived_field_whose_sources_are_right_except_lookups(
    seed,
):
    covered = total = 0
    for case in generate(T, 1, seed):
        for o in case["outputs"]:
            if not o["available"] or o["kind"] == "pass":
                continue
            name_of = {k["src"]: k["name"] for k in case["columns"] if k["src"]}
            cols = [
                (name_of[s], [r[name_of[s]] for r in case["rows"]])
                for s in rule_sources(o["rule"])
            ]
            by_name = [
                {name_of[s]: r[name_of[s]] for s in name_of} for r in case["rows"]
            ]
            hit = any(
                _same([safe_evaluate(c, row) for row in by_name], o["expected"])
                for c in candidates(cols)
            )
            if o["kind"] == "lookup":
                assert not hit  # a code table cannot be guessed: it goes to review
                continue
            total += 1
            covered += hit
    assert total > 0 and covered == total, (
        f"catalog covers {covered} of {total} derived fields"
    )


def test_every_chosen_column_keeps_its_plain_copy_candidate():
    two = candidates([("a", ["x", "y"]), ("b", ["p", "q"])])
    three = candidates([("a", ["x"]), ("b", ["p"]), ("c", ["m"])])
    assert {"src": "a"} in two and {"src": "b"} in two
    assert {"src": "a"} in three and {"src": "b"} in three and {"src": "c"} in three


def test_date_column_keeps_year_next_to_copy_when_paired():
    c = candidates([("born", ["1990-05-17", "1985-01-02"]), ("name", ["Budi", "Siti"])])
    assert {"src": "born"} in c and {"op": "year", "args": [{"src": "born"}]} in c


def test_output_type_filters_candidates_whose_result_has_another_type():
    cols = [("born", ["1990-05-17", "1985-01-02"]), ("name", ["Budi", "Siti"])]
    date_out = [render(c, lambda s: s) for c in candidates(cols, out_type="date")]
    assert (
        "year(born)" not in date_out and "born" in date_out
    )  # year gives a number, not a date
    num_out = [
        render(c, lambda s: s)
        for c in candidates([("a", [1, 2]), ("b", [3, 4])], out_type="number")
    ]
    assert (
        'concat(a, " ", b)' not in num_out and "a + b" in num_out
    )  # concat gives text
    assert candidates(cols, out_type=None) == candidates(
        cols
    )  # no type given: no filtering
