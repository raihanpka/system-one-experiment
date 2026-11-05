from columnmap.prompts import input_state, samples


def rows(n):
    return [{"c": f"v{i}"} for i in range(n)]


def test_small_files_send_every_row_and_empty_files_send_none_without_error():
    assert samples(rows(0), "c") == []
    assert samples(rows(1), "c") == ["v0"]
    assert samples(rows(2), "c") == ["v0", "v1"]
    assert "Sample values" not in input_state("c", [])
    assert "Sample values: v0, v1" in input_state("c", ["v0", "v1"])


def test_larger_files_send_five_evenly_spaced_rows():
    assert samples(rows(9), "c") == ["v0", "v2", "v4", "v6", "v8"]
    assert len(samples(rows(1000), "c")) == 5


def test_profile_describes_every_row_of_a_column_by_type_range_and_format():
    from columnmap.prompts import profile

    assert (
        profile(["2024-01-05", "2025-03-01", ""])
        == "2 of 3 filled, 2 distinct; dates 2024-01-05 to 2025-03-01; values: 2024-01-05, 2025-03-01"
    )
    assert "decimals 0.08 to 0.29, all between 0 and 1" in profile([0.29, 0.08, 0.15])
    assert "whole numbers 5 to 90000" in profile([5, 90000, 123, 77, 8, 9, 10])
    assert "format AAA-9999" in profile(
        [
            "POL-1234",
            "POL-9999",
            "ABC-0000",
            "XYZ-1111",
            "QQQ-2222",
            "RRR-3333",
            "SSS-4444",
        ]
    )
    assert profile(["", " "]) == "all empty"
    assert "values: A, P, R" in profile(["A", "P", "R", "A"])
