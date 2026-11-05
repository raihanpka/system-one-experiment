from columnmap.prompts import input_state, samples


def rows(n):
    return [{"c": f"v{i}"} for i in range(n)]


def test_small_files_send_every_row_and_empty_files_send_none_without_error():
    assert samples(rows(0), "c") == []
    assert samples(rows(1), "c") == ["v0"]
    assert samples(rows(2), "c") == ["v0", "v1"]
    assert "Sample values" not in input_state("c", "samples", [])
    assert "Sample values: v0, v1" in input_state("c", "samples", ["v0", "v1"])
    assert "Sample values" not in input_state("c", "name", ["v0"])


def test_larger_files_send_five_evenly_spaced_rows():
    assert samples(rows(9), "c") == ["v0", "v2", "v4", "v6", "v8"]
    assert len(samples(rows(1000), "c")) == 5
