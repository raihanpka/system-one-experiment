from columnmap.core.template import load_template
from columnmap.dataset.generate import generate
from columnmap.core.scoring import FIXED, accept_lift, accept_score
from columnmap.mapping.reject import field_scores, fit_cut, fit_cuts, reject_metrics

T = load_template()
CASE = next(c for c in generate(T, n_per_level=1, seed=3) if c["case_id"] == "hardest_noisy-000")


def rows_with(score_true: float, score_false: float, score_decoy: float | None = None):
    """Matrix where every true pair scores score_true and every other pair score_false (decoys optionally other)."""
    matrix = {}
    for k in CASE["columns"]:
        base = score_decoy if (k["decoy"] and score_decoy is not None) else score_false
        matrix[k["name"]] = {o["name"]: {"p": 0.5, "d": score_true if k["name"] in o["sources"] else base} for o in CASE["outputs"]}
    return [{"case_id": CASE["case_id"], "level": CASE["level"], "matrix": matrix}]


def test_fit_cut_separates_two_groups_and_handles_one_class():
    cut = fit_cut([-3.0, -2.0, -2.5, 1.0, 2.0, 1.5], [False, False, False, True, True, True])
    assert -2.0 < cut < 1.0
    assert fit_cut([1.0, 2.0], [True, True]) is None                          # needs both classes
    noisy = fit_cut([-1, 0, 1, 2, 3, 0.5], [False, True, False, True, True, False])
    assert -1 <= noisy <= 3


def test_field_scores_and_metrics_with_a_clean_separation():
    feats = field_scores(rows_with(3.0, -4.0), {CASE["case_id"]: CASE})
    assert len(feats) == len(CASE["outputs"])
    for f in feats:
        if f["has_source"]:
            assert f["d1"] == 3.0 and f["top_ok"] and f["lift"] == 7.0
        else:
            assert f["d1"] == -4.0 and f["lift"] == 0.0
    m = reject_metrics(feats, FIXED)
    assert m["kept"] == 1.0 and m["rejected"] == 1.0 and m["balanced"] == 1.0 and m["top1_when_kept"] == 1.0


def test_fixed_rule_keeps_fields_without_a_source_when_all_scores_are_high():
    feats = field_scores(rows_with(2.0, 0.5), {CASE["case_id"]: CASE})           # a no-source field still scores 0.5 > 0
    assert reject_metrics(feats, FIXED)["rejected"] == 0.0
    cuts = fit_cuts(rows_with(2.0, 0.5), {CASE["case_id"]: CASE})
    assert 0.5 < cuts["d1"] <= 2.0                                                # the fitted cutoff sits between the groups
    fitted = reject_metrics(feats, accept_score(cuts["d1"]))
    assert fitted["rejected"] == 1.0 and fitted["kept"] == 1.0 and fitted["balanced"] == 1.0


def test_lift_rule_uses_the_distance_above_the_median_column():
    feats = field_scores(rows_with(2.0, 0.5), {CASE["case_id"]: CASE})
    cuts = fit_cuts(rows_with(2.0, 0.5), {CASE["case_id"]: CASE})
    assert 0.0 < cuts["lift"] <= 1.5
    m = reject_metrics(feats, accept_lift(cuts["lift"]))
    assert m["balanced"] == 1.0


def test_decoys_picked_counts_kept_fields_whose_best_column_is_a_decoy():
    rows = rows_with(1.0, -3.0, score_decoy=5.0)                                  # decoys outscore everything
    feats = field_scores(rows, {CASE["case_id"]: CASE})
    m = reject_metrics(feats, FIXED)
    assert m["decoys_picked"] == len(CASE["outputs"]) and m["top1_when_kept"] == 0.0
