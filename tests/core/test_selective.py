from columnmap.core.selective import MIN_SUPPORT, choose_tau, precision_curve


def test_curve_lists_precision_and_how_many_fields_stay_above_each_confidence():
    confs = [0.9, 0.8, 0.7, 0.6, 0.5]
    ok = [True, True, False, True, False]
    curve = precision_curve(confs, ok)
    assert [c[0] for c in curve] == [0.5, 0.6, 0.7, 0.8, 0.9]                       # ascending confidence
    by_tau = {c[0]: (c[1], c[2]) for c in curve}
    assert by_tau[0.9] == (1.0, 1) and by_tau[0.8] == (1.0, 2) and by_tau[0.7] == (2 / 3, 3)
    assert by_tau[0.5] == (3 / 5, 5)


def test_choose_tau_takes_the_loosest_cutoff_that_still_meets_the_target_with_enough_support():
    confs = [0.99] * 20 + [0.9] * 20 + [0.6] * 20
    ok = [True] * 20 + [True] * 19 + [False] + [True] * 10 + [False] * 10
    curve = precision_curve(confs, ok)
    assert choose_tau(curve, target=1.0) == 0.99
    assert choose_tau(curve, target=0.97) == 0.9                                      # 39 of 40 above 0.9
    assert choose_tau(curve, target=0.7) == 0.6
    assert choose_tau(curve, target=1.0, min_support=25) is None                      # not enough fields to trust the cutoff


def test_a_small_lucky_streak_is_not_enough_support():
    curve = precision_curve([0.99] * (MIN_SUPPORT - 1), [True] * (MIN_SUPPORT - 1))
    assert choose_tau(curve, target=0.99) is None


def test_curve_is_capped_in_size_and_empty_input_is_fine():
    assert precision_curve([], []) == []
    many = precision_curve([i / 1000 for i in range(1000)], [True] * 1000, points=50)
    assert len(many) <= 50
