import pytest

from columnmap.core.prompts import PROMPTS, mapping_question, prompt, table_options_question, table_pair_question

KEYS = {"options", "pair", "table_options", "table_pair"}


def test_every_version_has_every_text_and_the_versions_really_differ():
    assert set(PROMPTS) >= {"v1", "v2", "v3"}
    for v, texts in PROMPTS.items():
        assert set(texts) == KEYS, v
    assert len({PROMPTS[v]["options"] for v in PROMPTS}) == len(PROMPTS)


def test_the_current_wording_is_kept_as_v1():
    assert prompt("v1", "options") == "Which output fields does this input column provide data for?"
    assert mapping_question("policy_number", None, "name").instructions == "This input column provides the data for the output field 'policy_number'."


def test_versions_change_the_question_text_for_all_contexts():
    assert "by joining it with other columns" in mapping_question("f", None, "name", "v2").instructions
    assert "Indonesian" in mapping_question("f", None, "name", "v3").instructions
    assert "'COL_A'" in table_options_question("COL_A", ["x", "y"], "v2").instructions
    assert "'COL_A'" in table_pair_question("COL_A", "field_x", "v3").instructions and "field_x" in table_pair_question("COL_A", "field_x", "v3").instructions


def test_an_unknown_version_is_refused_with_the_known_ones_listed():
    with pytest.raises(ValueError) as e:
        prompt("v9", "options")
    assert "v9" in str(e.value) and "v1" in str(e.value)
