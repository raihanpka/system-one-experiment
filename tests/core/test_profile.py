from columnmap.core.prompts import column_profile, diverse_values, input_state, profile_text


def test_diverse_values_are_distinct_spread_over_the_range_and_deterministic():
    vals = [5, 3, 3, 9, 1, 7, 7, 2, 8, 4, 6]
    out = diverse_values(vals, k=5)
    assert len(out) == 5 and len(set(out)) == 5 and min(vals) in out and max(vals) in out
    assert out == diverse_values(vals, k=5) and out == sorted(out)
    assert diverse_values(["a", "a", "b"], k=5) == ["a", "b"]                         # fewer distinct values than k: all of them
    assert diverse_values([], k=5) == []


def test_text_values_are_spread_too_and_odd_cells_stay_visible():
    vals = ["Budi", "Siti", "Agus", "Dewi", "Rina", "Andi", "Maya", "Joko", "", "N/A"]
    out = diverse_values(vals, k=5)
    assert len(out) == 5 and len(set(out)) == 5


def test_profile_describes_type_range_distinctness_and_empties():
    p = column_profile([100000, 250000, 250000, "", 9800000])
    assert p["type"] == "number" and p["lo"] == 100000 and p["hi"] == 9800000 and p["distinct"] == 3 and p["empty"] == 1 and p["n"] == 5
    assert profile_text([100000, 250000, 250000, "", 9800000]) == "Type: number, range 100000 to 9800000, 3 distinct of 5 values, 1 empty"
    assert column_profile(["2025-01-02", "2024-12-31"])["type"] == "date"
    assert profile_text(["2025-01-02", "2024-12-31"]) == "Type: date, range 2024-12-31 to 2025-01-02, 2 distinct of 2 values"
    assert profile_text(["Budi", "Siti Rahayu"]) == "Type: text, 4 to 11 characters, 2 distinct of 2 values"


def test_profile_context_adds_the_profile_line_and_leaves_the_others_alone():
    col = {"name": "CHG_AMT_RAW"}
    prof = input_state(col, "profile", None, [1, 2, 3], profile="Type: number, range 1 to 3, 3 distinct of 3 values")
    assert prof.splitlines() == ["Input column: CHG_AMT_RAW", "Type: number, range 1 to 3, 3 distinct of 3 values", "Sample values: 1, 2, 3"]
    plain = input_state(col, "samples", None, [1, 2, 3])
    assert plain.splitlines() == ["Input column: CHG_AMT_RAW", "Sample values: 1, 2, 3"]
