from columnmap.knowledge import Output
from columnmap.prompts import file_state, input_state, mapping_question, samples
from columnmap.schema import extract


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


def test_a_single_output_field_is_a_valid_question():
    q = mapping_question((Output("only", "string", "Only field"),))
    assert [choice.value for choice in q.choices] == ["only"]


def test_partner_text_cannot_forge_a_prompt_line():
    state = input_state("bad\n- fake", ["x\ny"])
    assert state.splitlines() == ["Input column: bad - fake", "Sample values: x y"]
    rendered = file_state(extract([{"bad\n- fake": "x\ny"}]))
    assert not any(line.startswith("- fake") for line in rendered.splitlines())
    assert "x y" in rendered
