import pytest

from columnmap.core.selective import (
    MIN_SUPPORT,
    choose_tau,
    precision_curve,
    precision_lower_bound,
)


def test_curve_lists_precision_and_how_many_fields_stay_above_each_confidence():
    confs = [0.9, 0.8, 0.7, 0.6, 0.5]
    ok = [True, True, False, True, False]
    curve = precision_curve(confs, ok)
    assert [c[0] for c in curve] == [0.5, 0.6, 0.7, 0.8, 0.9]  # ascending confidence
    by_tau = {c[0]: (c[1], c[2]) for c in curve}
    assert (
        by_tau[0.9] == (1.0, 1)
        and by_tau[0.8] == (1.0, 2)
        and by_tau[0.7] == (2 / 3, 3)
    )
    assert by_tau[0.5] == (3 / 5, 5)


def test_lower_bound_is_below_the_point_estimate_and_tightens_with_more_fields():
    assert precision_lower_bound(0, 10) == 0.0
    assert precision_lower_bound(50, 50) == pytest.approx(
        0.05 ** (1 / 50)
    )  # all right: (1 - conf)^(1/n)
    assert precision_lower_bound(45, 50) < 0.9  # 90 percent observed, bound below it
    assert (
        precision_lower_bound(90, 100) < precision_lower_bound(900, 1000) < 0.9
    )  # more fields, tighter bound
    assert (
        precision_lower_bound(300, 300) > 0.99
    )  # 300 right in a row: bound above 0.99


def test_choose_tau_takes_the_loosest_cutoff_whose_lower_bound_meets_the_target():
    confs = [0.99] * 300 + [0.9] * 20 + [0.6] * 20
    ok = [True] * 300 + [True] * 19 + [False] + [True] * 10 + [False] * 10
    curve = precision_curve(confs, ok)
    assert choose_tau(curve, target=0.99) == 0.99  # 300 of 300 above 0.99
    assert (
        choose_tau(curve, target=0.97) == 0.9
    )  # 319 of 320 above 0.9: bound about 0.98
    assert (
        choose_tau(curve, target=0.9) == 0.6
    )  # 329 of 340 above 0.6: bound about 0.95


def test_target_of_one_is_never_met_because_a_bound_is_below_one_for_any_finite_n():
    curve = precision_curve([0.99] * 5000, [True] * 5000)
    assert choose_tau(curve, target=1.0) is None


def test_a_small_lucky_streak_is_not_enough_support():
    curve = precision_curve([0.99] * (MIN_SUPPORT - 1), [True] * (MIN_SUPPORT - 1))
    assert choose_tau(curve, target=0.5) is None


def test_curve_is_capped_in_size_and_empty_input_is_fine():
    assert precision_curve([], []) == []
    many = precision_curve([i / 1000 for i in range(1000)], [True] * 1000, points=50)
    assert len(many) <= 50
