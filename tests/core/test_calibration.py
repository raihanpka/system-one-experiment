import json

import numpy as np
import pytest

from columnmap.core.calibration import (
    Calibration,
    CalibrationKeyError,
    apply_platt,
    fit_platt,
    make_key,
    reliability_bins,
)
from columnmap.core.scoring import ece


def synthetic(a, b, n=40000, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-4, 4, n)
    y = rng.random(n) < 1 / (1 + np.exp(-(a * x + b)))
    return x.tolist(), y.tolist()


def test_fit_recovers_known_slope_and_bias_within_five_percent():
    xs, ys = synthetic(1.3, -0.6)
    a, b = fit_platt(xs, ys)
    assert abs(a - 1.3) / 1.3 < 0.05 and abs(b + 0.6) / 0.6 < 0.05


def test_fit_needs_both_classes_and_is_stable_on_extreme_scores():
    assert fit_platt([1.0, 2.0], [True, True]) is None
    a, b = fit_platt(
        [-50, -40, 40, 50, -45, 45], [False, False, True, True, False, True]
    )
    assert np.isfinite(a) and np.isfinite(b) and a > 0


def test_calibrated_probabilities_sit_closer_to_the_diagonal_than_the_uncalibrated_ones():
    xs, ys = synthetic(1.3, -0.6, seed=1)
    a, b = fit_platt(xs, ys)
    raw = [1 / (1 + np.exp(-x / 0.8278)) for x in xs]
    cal = [apply_platt(x, a, b) for x in xs]
    assert ece(cal, ys) < ece(raw, ys) and ece(cal, ys) < 0.02


def test_reliability_bins_give_mean_confidence_observed_rate_and_count():
    bins = reliability_bins([0.1, 0.12, 0.9, 0.95], [False, False, True, True], bins=10)
    assert (
        [b["n"] for b in bins] == [2, 2]
        and bins[0]["confidence"] == pytest.approx(0.11)
        and bins[1]["observed"] == 1.0
    )


def test_calibration_file_round_trip_and_a_key_that_does_not_match_is_refused(tmp_path):
    key = make_key("Jev-v3", "Q8_0", "options", "samples", "v1", "field")
    c = Calibration(tmp_path / "calibration.json")
    c.set(key, a=1.2, b=-0.5, n=300, positives=40, seeds=[12, 13])
    c.save()
    again = Calibration(tmp_path / "calibration.json")
    assert again.get(key)["a"] == 1.2 and json.loads(
        (tmp_path / "calibration.json").read_text()
    )["entries"][key]["seeds"] == [12, 13]
    other = make_key("Jev-v3", "Q4_K_M", "options", "samples", "v1", "field")
    with pytest.raises(CalibrationKeyError) as e:
        again.get(other)
    assert "Q4_K_M" in str(e.value) and "refit" in str(e.value).lower()
    assert again.get(other, missing_ok=True) is None
