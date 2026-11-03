import math

from columnmap.convert.auto import field_decisions, review_rows, selective_metrics
from columnmap.convert.honest import convert_fields
from columnmap.convert.metrics import field_correct
from columnmap.core.calibration import apply_platt
from tests.convert.test_honest import CASE, OUTS, convert, matrix_from


def records():
    recs, _ = convert(matrix_from(CASE))
    return recs


def test_top_columns_and_shortlist_probabilities_are_kept_for_the_review_list():
    recs = records()
    answered = next(
        r for r in recs.values() if r["best"] is not None and len(r["cands"]) > 1
    )
    assert len(answered["shortlist_p"]) == len(answered["shortlist"]) <= 3
    assert all(len(r["top_cols"]) <= 3 for r in recs.values()) and all(
        len(r["top_cols"]) == 3 for r in recs.values()
    )
    ps = [p for _, p in answered["top_cols"]]
    assert ps == sorted(ps, reverse=True)


def test_without_calibration_every_answered_field_is_filled_and_the_rest_wait_for_a_person():
    recs = records()
    dec = field_decisions(recs, entry=None, tau=None)
    for f, r in recs.items():
        assert dec[f]["auto"] == (r["best"] is not None)
    unmapped = next(f for f, r in recs.items() if r["method"] == "unmapped")
    assert dec[unmapped]["reason"] == "no column passes"


def test_with_calibration_only_fields_at_or_above_the_cutoff_are_filled():
    recs = records()
    entry = {"a": 1.0, "b": 0.0}
    answered = [f for f, r in recs.items() if r["best"] is not None]
    cal = {
        f: apply_platt(math.log(recs[f]["conf"] / (1 - recs[f]["conf"])), 1.0, 0.0)
        for f in answered
    }
    tau = sorted(cal.values())[len(cal) // 2]
    dec = field_decisions(recs, entry=entry, tau=tau)
    assert {f for f in answered if dec[f]["auto"]} == {
        f for f in answered if cal[f] >= tau
    }
    low = next(f for f in answered if not dec[f]["auto"])
    assert (
        dec[low]["reason"] == "low confidence"
        and abs(dec[low]["conf_cal"] - cal[low]) < 1e-9
    )
    assert not any(
        field_decisions(recs, entry=entry, tau=None)[f]["auto"] for f in answered
    )  # no trusted cutoff: all to review


def test_converted_file_holds_only_auto_filled_fields_and_review_rows_list_three_candidates():
    recs = records()
    dec = field_decisions(recs, entry={"a": 1.0, "b": 0.0}, tau=0.99)
    _, conv = convert_fields(
        recs, CASE["rows"], OUTS, auto={f: d["auto"] for f, d in dec.items()}
    )
    assert all(v is None for r in conv for v in r)  # nothing reached 0.99
    rows = review_rows(recs, dec)
    assert len(rows) == len(OUTS) and all(len(r) == 9 for r in rows)
    first = next(r for r in rows if r[3])
    assert first[3] and 0 <= first[4] <= 1


def test_selective_metrics_count_precision_coverage_and_wrong_fills():
    recs = records()
    dec = field_decisions(recs, entry=None, tau=None)
    _, conv = convert_fields(
        recs, CASE["rows"], OUTS, auto={f: d["auto"] for f, d in dec.items()}
    )
    m = selective_metrics(CASE, conv, dec, field_correct(CASE, conv))
    with_src = sum(o["available"] for o in CASE["outputs"])
    assert (
        m["with_source"] == with_src
        and 0 < m["coverage"] <= 1
        and 0 <= m["precision"] <= 1
    )
    assert (
        m["auto_total"] == m["auto_correct"] + m["auto_wrong"]
        and m["review"] == len(OUTS) - m["auto_total"]
    )


def test_an_invalid_value_never_fills_even_without_calibration():
    recs = {
        "a": {"best": "x", "cols": ["c"], "conf": 0.9},
        "b": {"best": "y", "cols": ["d"], "conf": 0.9},
    }
    dec = field_decisions(recs, entry=None, tau=None, valid={"a": True, "b": False})
    assert (
        dec["a"]["auto"] is True
        and dec["b"]["auto"] is False
        and dec["b"]["reason"] == "invalid value"
    )
