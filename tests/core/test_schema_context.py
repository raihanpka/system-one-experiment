from columnmap.core.prompts import context_label, input_state


def test_schema_context_lists_the_other_headers_after_the_values():
    col = {"name": "CHG_AMT_RAW"}
    s = input_state(col, "schema", None, [1, 2], others=["CHG_LVY_PCT", "PRTY_NM_1"])
    assert s.splitlines() == ["Input column: CHG_AMT_RAW", "Sample values: 1, 2", "Other columns in this file: CHG_LVY_PCT, PRTY_NM_1"]
    assert context_label("schema") == "name + values + other headers"


def test_other_contexts_never_show_the_other_headers():
    col = {"name": "A"}
    for cond in ("name", "samples", "table"):
        assert "Other columns" not in input_state(col, cond, None, [1], others=["B"])
